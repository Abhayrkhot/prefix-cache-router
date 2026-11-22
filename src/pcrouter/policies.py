"""Routing policies.

Every policy sees the same per-worker snapshot (:class:`WorkerView`) that the :class:`Router`
builds from its own bookkeeping, so the simulator and the live proxy run identical decisions.

Baselines
    ``random``, ``round-robin``, ``least-request`` (fewest in-flight), ``p2c`` (power of two
    choices on in-flight), ``chwbl`` (consistent hashing with bounded loads on a prefix key),
    ``legacy`` (this repo's original rule: a prefix key sticks to the worker that first got it,
    new keys go to the worker with the fewest keys).
Cache-aware
    ``threshold`` — the earlier sgl-router cache-aware rule: shortest queue when the cluster is
    imbalanced, else the longest-match worker if the match rate clears a threshold, else the
    worker whose cache is emptiest.
    ``affinity`` — the current sgl-router rule (``policies_reorg``): a worker that holds at least
    ``min_match_tokens`` of the prompt is preferred (fewest uncached tokens first); in balanced
    mode the owner yields to a power-of-two alternative when its prefill load exceeds the
    alternative's by both ``load_gap`` tokens and ``load_factor``x.
    ``ttft`` — ours: place the request where its *estimated time to first token* is lowest:
    queued prefill work plus this request's uncached prefill, priced by the calibrated
    step-time model (attention over the cached prefix included) and slowed down by the decodes
    sharing the worker's iterations.
"""

from __future__ import annotations

import bisect
import math
import random
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, ClassVar

from pcrouter.costmodel import BatchStats, StepModel

__all__ = [
    "CHWBL",
    "POLICIES",
    "Affinity",
    "LeastRequest",
    "Legacy",
    "Policy",
    "PowerOfTwo",
    "RandomPolicy",
    "RoundRobin",
    "RouteRequest",
    "TTFTCost",
    "Threshold",
    "WorkerView",
    "make_policy",
]


@dataclass(slots=True)
class RouteRequest:
    input_len: int
    max_tokens: int
    chunks: Sequence[int]
    chunk_tokens: int
    now: float = 0.0  # seconds, monotonic (simulated time in the simulator)


@dataclass(slots=True)
class WorkerView:
    """What the router knows about one healthy worker when routing one request."""

    id: int
    inflight: int  # requests sent and not finished
    prefilling: int  # sent, first token not seen yet
    decoding: int  # first token seen, not finished
    pending_prefill_tokens: int  # estimated uncached tokens queued for prefill
    pending_prefill_s: float  # the same work in estimated engine seconds
    match_chunks: int  # cached prefix of *this* request on this worker (router's estimate)
    cached_tokens: int  # match in tokens, capped at input_len - 1 like vLLM
    index_chunks: int  # chunks the index believes this worker caches
    capacity_chunks: int
    inflight_ctx: int = 0  # context tokens of in-flight requests (prompt + half the output)

    def uncached(self, input_len: int) -> int:
        return input_len - self.cached_tokens


def _mix64(x: int) -> int:
    """SplitMix64 finalizer: spreads keys and ring points uniformly."""
    x = (x + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
    return x ^ (x >> 31)


class Policy(ABC):
    name: ClassVar[str]
    uses_index: ClassVar[bool] = False

    def __init__(self, num_workers: int, seed: int = 0) -> None:
        self.num_workers = num_workers
        self.rng = random.Random(seed)

    @abstractmethod
    def choose(self, req: RouteRequest, views: Sequence[WorkerView]) -> int:
        """Return the id of one of ``views`` (all healthy, never empty)."""

    def params(self) -> dict[str, Any]:
        return {}

    def _argmin(self, views: Sequence[WorkerView], key: Callable[[WorkerView], Any]) -> int:
        """Lowest ``key``; ties broken uniformly at random (no bias toward worker 0)."""
        best: list[WorkerView] = []
        best_k: Any = None
        for v in views:
            k = key(v)
            if best_k is None or k < best_k:
                best, best_k = [v], k
            elif k == best_k:
                best.append(v)
        return best[0].id if len(best) == 1 else self.rng.choice(best).id


class RandomPolicy(Policy):
    name = "random"

    def choose(self, req: RouteRequest, views: Sequence[WorkerView]) -> int:
        return self.rng.choice(views).id


class RoundRobin(Policy):
    name = "round-robin"

    def __init__(self, num_workers: int, seed: int = 0) -> None:
        super().__init__(num_workers, seed)
        self._next = 0

    def choose(self, req: RouteRequest, views: Sequence[WorkerView]) -> int:
        ids = sorted(v.id for v in views)
        i = bisect.bisect_left(ids, self._next)
        w = ids[i % len(ids)]
        self._next = w + 1
        return w


class LeastRequest(Policy):
    name = "least-request"

    def choose(self, req: RouteRequest, views: Sequence[WorkerView]) -> int:
        return self._argmin(views, lambda v: v.inflight)


class PowerOfTwo(Policy):
    name = "p2c"

    def choose(self, req: RouteRequest, views: Sequence[WorkerView]) -> int:
        if len(views) == 1:
            return views[0].id
        a, b = self.rng.sample(list(views), 2)
        return a.id if a.inflight <= b.inflight else b.id


def _prefix_key(req: RouteRequest, depth: int) -> int | None:
    if not req.chunks:
        return None
    return req.chunks[min(depth, len(req.chunks)) - 1]


class CHWBL(Policy):
    """Consistent hashing with bounded loads (Mirrokni et al.) on the key of the first
    ``key_depth`` chunks. A worker may hold at most ``ceil(load_factor * (total+1) / n)``
    in-flight requests; a full worker passes the request clockwise along the ring."""

    name = "chwbl"

    def __init__(
        self,
        num_workers: int,
        seed: int = 0,
        key_depth: int = 2,
        load_factor: float = 1.25,
        vnodes: int = 160,
    ) -> None:
        super().__init__(num_workers, seed)
        if key_depth < 1 or load_factor < 1.0:
            raise ValueError("key_depth >= 1 and load_factor >= 1 required")
        self.key_depth = key_depth
        self.load_factor = load_factor
        ring = sorted((_mix64((w << 32) | v), w) for w in range(num_workers) for v in range(vnodes))
        self._points = [p for p, _ in ring]
        self._owners = [w for _, w in ring]

    def params(self) -> dict[str, Any]:
        return {"key_depth": self.key_depth, "load_factor": self.load_factor}

    def choose(self, req: RouteRequest, views: Sequence[WorkerView]) -> int:
        key = _prefix_key(req, self.key_depth)
        if key is None:
            return self._argmin(views, lambda v: v.inflight)
        by_id = {v.id: v for v in views}
        cap = math.ceil(self.load_factor * (sum(v.inflight for v in views) + 1) / len(views))
        start = bisect.bisect_left(self._points, _mix64(key))
        n = len(self._points)
        seen: set[int] = set()
        for i in range(n):
            w = self._owners[(start + i) % n]
            if w in seen or w not in by_id:
                continue
            if by_id[w].inflight < cap:
                return w
            seen.add(w)
            if len(seen) == len(by_id):
                break
        return self._argmin(views, lambda v: v.inflight)


class Legacy(Policy):
    """The rule this repository started with, generalised from exact strings to prefix keys:
    a key is pinned to the first worker it was given; new keys go to the worker holding the
    fewest keys. No notion of load or cache capacity."""

    name = "legacy"

    def __init__(self, num_workers: int, seed: int = 0, key_depth: int = 2) -> None:
        super().__init__(num_workers, seed)
        self.key_depth = key_depth
        self._owner: dict[int, int] = {}
        self._keys = [0] * num_workers

    def params(self) -> dict[str, Any]:
        return {"key_depth": self.key_depth}

    def choose(self, req: RouteRequest, views: Sequence[WorkerView]) -> int:
        key = _prefix_key(req, self.key_depth)
        alive = {v.id for v in views}
        if key is not None and self._owner.get(key, -1) in alive:
            return self._owner[key]
        w = self._argmin(views, lambda v: self._keys[v.id])
        if key is not None:
            self._owner[key] = w
            self._keys[w] += 1
        return w


class Threshold(Policy):
    """Earlier sgl-router cache-aware rule (router-side load = in-flight requests)."""

    name = "threshold"
    uses_index = True

    def __init__(
        self,
        num_workers: int,
        seed: int = 0,
        cache_threshold: float = 0.5,
        balance_abs: int = 32,
        balance_rel: float = 1.5,
    ) -> None:
        super().__init__(num_workers, seed)
        self.cache_threshold = cache_threshold
        self.balance_abs = balance_abs
        self.balance_rel = balance_rel

    def params(self) -> dict[str, Any]:
        return {
            "cache_threshold": self.cache_threshold,
            "balance_abs": self.balance_abs,
            "balance_rel": self.balance_rel,
        }

    def choose(self, req: RouteRequest, views: Sequence[WorkerView]) -> int:
        loads = [v.inflight for v in views]
        hi, lo = max(loads), min(loads)
        if hi - lo > self.balance_abs and hi > lo * self.balance_rel:
            return self._argmin(views, lambda v: v.inflight)
        best = max(v.match_chunks for v in views)
        total = max(1, len(req.chunks))
        if best > 0 and best / total > self.cache_threshold:
            return self._argmin(views, lambda v: (-v.match_chunks, v.inflight))
        return self._argmin(views, lambda v: (v.index_chunks, v.inflight))


class Affinity(Policy):
    """Current sgl-router cache-aware rule (prefix affinity, ``policies_reorg``).

    A *hit* is a worker caching >= ``min_match_tokens`` of the prompt; the hit with the fewest
    uncached tokens (then least prefill load) wins. ``mode="balanced"`` lets a power-of-two
    alternative take the request when the owner's prefill load (queued uncached tokens plus
    this request's) exceeds the alternative's by more than ``load_gap`` tokens *and*
    ``load_factor``x. No hit: power of two choices on prefill load.
    """

    name = "affinity"
    uses_index = True

    def __init__(
        self,
        num_workers: int,
        seed: int = 0,
        min_match_tokens: int = 1024,
        mode: str = "balanced",
        load_gap: int = 1024,
        load_factor: float = 2.0,
    ) -> None:
        super().__init__(num_workers, seed)
        if mode not in ("prefer", "balanced"):
            raise ValueError("mode must be 'prefer' or 'balanced'")
        self.min_match_tokens = min_match_tokens
        self.mode = mode
        self.load_gap = load_gap
        self.load_factor = load_factor

    def params(self) -> dict[str, Any]:
        return {
            "min_match_tokens": self.min_match_tokens,
            "mode": self.mode,
            "load_gap": self.load_gap,
            "load_factor": self.load_factor,
        }

    def _load(self, v: WorkerView, req: RouteRequest) -> int:
        return v.pending_prefill_tokens + v.uncached(req.input_len)

    def _p2c(self, req: RouteRequest, views: Sequence[WorkerView]) -> WorkerView:
        if len(views) == 1:
            return views[0]
        a, b = self.rng.sample(list(views), 2)
        return a if self._load(a, req) <= self._load(b, req) else b

    def choose(self, req: RouteRequest, views: Sequence[WorkerView]) -> int:
        hits = [v for v in views if v.cached_tokens >= self.min_match_tokens]
        if not hits:
            return self._p2c(req, views).id
        owner = min(
            hits,
            key=lambda v: (v.uncached(req.input_len), v.pending_prefill_tokens, v.id),
        )
        if self.mode == "prefer" or len(views) == 1:
            return owner.id
        alt = self._p2c(req, [v for v in views if v.id != owner.id])
        a, b = self._load(owner, req), self._load(alt, req)
        if a - b > self.load_gap and a > b * self.load_factor:
            return alt.id
        return owner.id


class TTFTCost(Policy):
    """Ours: place the request where its prefill is cheapest *for the cluster*.

    ``cost(w) = (queued_prefill_s(w) + (1 + beta) * prefill_s(uncached_w, cached_w))
    * slowdown(decoding_w) + lambda * max_tokens * decode_step_s(w)``

    ``prefill_s`` comes from the calibrated step model, so a cache hit on a long prompt is
    worth more than the same hit on a short one (the attention term grows with context), and
    ``slowdown`` charges the decode tokens and KV reads that share every iteration. With
    ``beta = 0`` the cost is the request's own estimated TTFT. Recomputing a prefix that is
    cached elsewhere also has costs the request does not see: the GPU time delays requests that
    arrive behind it, and the duplicate copy evicts other prefixes from the new worker's cache.
    ``beta`` charges those; ``beta = 8`` was the best worst case over both Mooncake traces.
    Sessions that stick to their worker also pile up decode work there (every iteration reads
    the KV of every running request), which the prefill terms do not see; the last term,
    weighted by ``lambda = decode_weight``, charges the request's own decode time on that
    worker. ``lambda`` in [0.125, 0.5] is flat on the traces and fixes long chat sessions;
    0.25 is the default (see ``scripts/experiments.py``: beta, decode).
    """

    name = "ttft"
    uses_index = True

    def __init__(
        self,
        num_workers: int,
        seed: int = 0,
        model: StepModel | None = None,
        budget: int = 8192,
        avg_context: int = 8192,
        beta: float = 8.0,
        decode_weight: float = 0.25,
    ) -> None:
        super().__init__(num_workers, seed)
        if model is None:
            raise ValueError("the ttft policy needs a step model")
        if beta < 0 or decode_weight < 0:
            raise ValueError("beta and decode_weight must be >= 0")
        self.model = model
        self.budget = budget
        self.avg_context = avg_context
        self.beta = float(beta)
        self.decode_weight = float(decode_weight)
        self._slow: dict[int, float] = {}

    def params(self) -> dict[str, Any]:
        return {
            "budget": self.budget,
            "avg_context": self.avg_context,
            "beta": self.beta,
            "decode_weight": self.decode_weight,
        }

    def decode_step_s(self, req: RouteRequest, v: WorkerView) -> float:
        """Estimated decode iteration time on ``v`` once ``req`` joins its batch."""
        s = BatchStats()
        s.num_tokens = s.num_seqs = v.inflight + 1
        s.decode_ctx = float(v.inflight_ctx + req.input_len)
        return self.model.step_time(s)

    def slowdown(self, decoding: int) -> float:
        s = self._slow.get(decoding)
        if s is None:
            s = self.model.decode_slowdown(decoding, self.avg_context, self.budget)
            self._slow[decoding] = s
        return s

    def prefill_s(self, req: RouteRequest, v: WorkerView) -> float:
        return self.model.prefill_time(v.uncached(req.input_len), v.cached_tokens, self.budget)

    def estimate(self, req: RouteRequest, v: WorkerView) -> float:
        """Estimated TTFT of ``req`` on ``v`` (seconds)."""
        return (v.pending_prefill_s + self.prefill_s(req, v)) * self.slowdown(v.decoding)

    def cost(self, req: RouteRequest, v: WorkerView) -> float:
        own = self.prefill_s(req, v)
        return (v.pending_prefill_s + (1.0 + self.beta) * own) * self.slowdown(v.decoding)

    def choose(self, req: RouteRequest, views: Sequence[WorkerView]) -> int:
        own: dict[int, float] = {}  # most workers share a match length (often 0): price once
        costs: dict[int, float] = {}
        for v in views:
            o = own.get(v.cached_tokens)
            if o is None:
                o = own[v.cached_tokens] = self.prefill_s(req, v)
            costs[v.id] = (v.pending_prefill_s + (1.0 + self.beta) * o) * self.slowdown(v.decoding)
            if self.decode_weight:
                costs[v.id] += self.decode_weight * req.max_tokens * self.decode_step_s(req, v)
        return self._argmin(views, lambda v: (round(costs[v.id], 6), v.inflight))


POLICIES: dict[str, type[Policy]] = {
    p.name: p
    for p in (
        RandomPolicy,
        RoundRobin,
        LeastRequest,
        PowerOfTwo,
        CHWBL,
        Legacy,
        Threshold,
        Affinity,
        TTFTCost,
    )
}


def make_policy(
    name: str,
    num_workers: int,
    *,
    seed: int = 0,
    model: StepModel | None = None,
    **params: Any,
) -> Policy:
    """Build a policy by name; ``params`` override its defaults."""
    try:
        cls = POLICIES[name]
    except KeyError:
        raise ValueError(f"unknown policy {name!r}; choose from {sorted(POLICIES)}") from None
    if cls is TTFTCost:
        return TTFTCost(num_workers, seed, model=model, **params)
    return cls(num_workers, seed, **params)
