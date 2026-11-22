"""Router state shared by the simulator and the live proxy.

The router keeps three things per worker, all from its own traffic (no engine polling is
required): requests in flight, the prefill work it has queued there (requests whose first
token has not come back yet), and a :class:`~pcrouter.index.PrefixIndex` of the chunks it
expects the worker to have cached. A routing decision returns a :class:`Ticket`; the caller
reports the first token and the end of the request against that ticket.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from pcrouter.costmodel import StepModel
from pcrouter.index import PrefixIndex
from pcrouter.policies import Policy, RouteRequest, WorkerView

__all__ = ["Router", "RouterConfig", "Ticket", "WorkerLoad"]


@dataclass
class RouterConfig:
    num_workers: int
    #: Tokens per router chunk (a multiple of the engine block size).
    chunk_tokens: int = 512
    #: KV-cache capacity of each worker in tokens (vLLM logs it as "GPU KV cache size").
    capacity_tokens: int | Sequence[int] = 1_000_000
    #: Share of that capacity the index assumes is holding reusable prefixes; the rest is
    #: taken by running requests' new blocks and generated tokens.
    capacity_factor: float = 1.0
    #: Prefill chunk budget of the engines (vLLM ``max_num_batched_tokens``).
    budget: int = 8192
    #: Use per-request cached-token feedback to forget chunks the index wrongly kept.
    feedback: bool = False

    def capacity_chunks(self) -> list[int]:
        caps = (
            [int(self.capacity_tokens)] * self.num_workers
            if isinstance(self.capacity_tokens, int)
            else [int(c) for c in self.capacity_tokens]
        )
        if len(caps) != self.num_workers:
            raise ValueError("capacity_tokens needs one value per worker")
        return [max(1, int(c * self.capacity_factor) // self.chunk_tokens) for c in caps]


@dataclass(slots=True)
class WorkerLoad:
    inflight: int = 0
    prefilling: int = 0
    pending_tokens: int = 0
    pending_s: float = 0.0
    routed: int = 0
    predicted_cached: int = 0
    actual_cached: int = 0
    feedback_n: int = 0
    ctx_tokens: int = 0

    @property
    def decoding(self) -> int:
        return self.inflight - self.prefilling


@dataclass(slots=True)
class Ticket:
    worker: int
    input_len: int
    chunks: Sequence[int]
    match_chunks: int
    cached_est: int
    uncached_est: int
    prefill_s_est: float
    ctx: int = 0
    first_seen: bool = False
    done: bool = False


class Router:
    def __init__(self, policy: Policy, cfg: RouterConfig, model: StepModel | None = None) -> None:
        if cfg.chunk_tokens < 1:
            raise ValueError("chunk_tokens must be positive")
        self.policy = policy
        self.cfg = cfg
        self.model = model
        self.index = PrefixIndex(cfg.num_workers, cfg.capacity_chunks())
        self.loads = [WorkerLoad() for _ in range(cfg.num_workers)]

    # -- decisions -----------------------------------------------------------------------
    def views(
        self,
        req: RouteRequest,
        healthy: Sequence[bool] | None = None,
        match: Sequence[int] | None = None,
    ) -> list[WorkerView]:
        n = self.cfg.num_workers
        if match is None:
            match = self.index.match(req.chunks) if req.chunks else [0] * n
        cap_tokens = max(0, req.input_len - 1)
        out = []
        for w in range(n):
            if healthy is not None and not healthy[w]:
                continue
            ld = self.loads[w]
            out.append(
                WorkerView(
                    id=w,
                    inflight=ld.inflight,
                    prefilling=ld.prefilling,
                    decoding=ld.decoding,
                    pending_prefill_tokens=ld.pending_tokens,
                    pending_prefill_s=ld.pending_s,
                    match_chunks=match[w],
                    cached_tokens=min(match[w] * req.chunk_tokens, cap_tokens),
                    index_chunks=self.index.size(w),
                    capacity_chunks=self.index.capacity[w],
                    inflight_ctx=ld.ctx_tokens,
                )
            )
        return out

    def route(
        self,
        input_len: int,
        max_tokens: int,
        chunks: Sequence[int],
        healthy: Sequence[bool] | None = None,
        match: Sequence[int] | None = None,
        now: float = 0.0,
    ) -> Ticket:
        """Pick a worker and charge the request to it. ``match`` overrides the index's
        estimate (the simulator's oracle mode passes the engines' true cache contents)."""
        req = RouteRequest(input_len, max_tokens, chunks, self.cfg.chunk_tokens, now)
        views = self.views(req, healthy, match)
        if not views:
            raise RuntimeError("no healthy workers")
        w = self.policy.choose(req, views)
        v = next(x for x in views if x.id == w)
        uncached = v.uncached(input_len)
        prefill_s = (
            self.model.prefill_time(uncached, v.cached_tokens, self.cfg.budget)
            if self.model is not None
            else 0.0
        )
        ld = self.loads[w]
        ld.inflight += 1
        ld.prefilling += 1
        ld.pending_tokens += uncached
        ld.pending_s += prefill_s
        ld.routed += 1
        ctx = input_len + max_tokens // 2
        ld.ctx_tokens += ctx
        if chunks:
            self.index.insert(w, chunks)
        return Ticket(
            w, input_len, chunks, v.match_chunks, v.cached_tokens, uncached, prefill_s, ctx
        )

    # -- feedback ------------------------------------------------------------------------
    def first_token(self, t: Ticket, cached_tokens: int | None = None) -> None:
        """The worker produced the first token; ``cached_tokens`` is what it really reused
        (vLLM: ``usage.prompt_tokens_details.cached_tokens``) when known."""
        if t.first_seen or t.done:
            return
        t.first_seen = True
        ld = self.loads[t.worker]
        ld.prefilling -= 1
        ld.pending_tokens -= t.uncached_est
        ld.pending_s -= t.prefill_s_est
        if ld.prefilling == 0:  # clear float drift
            ld.pending_tokens, ld.pending_s = 0, 0.0
        if cached_tokens is not None:
            self.cached_feedback(t, cached_tokens)

    def cached_feedback(self, t: Ticket, cached_tokens: int) -> None:
        """Report how many prompt tokens the worker actually reused for ``t`` (may arrive
        after the first token, e.g. in the final usage chunk of a stream)."""
        ld = self.loads[t.worker]
        ld.predicted_cached += t.cached_est
        ld.actual_cached += cached_tokens
        ld.feedback_n += 1
        if self.cfg.feedback and cached_tokens < t.cached_est:
            self._correct(t, cached_tokens)

    def _correct(self, t: Ticket, cached_tokens: int) -> None:
        # The index thought chunks [actual, predicted) were on the worker; they were not, so
        # its LRU is running ahead of the engine's. The request has just recomputed them, so
        # they *are* cached now; what was wrong is the capacity estimate. Shrink it by the
        # shortfall (evicting the index's oldest chunks), floored at a quarter of the start.
        short = (t.cached_est - cached_tokens) // self.cfg.chunk_tokens
        if short <= 0:
            return
        w = t.worker
        floor = max(1, self.cfg.capacity_chunks()[w] // 4)
        self.index.set_capacity(w, max(floor, self.index.capacity[w] - short))

    def finish(self, t: Ticket) -> None:
        if t.done:
            return
        if not t.first_seen:
            self.first_token(t)
        t.done = True
        self.loads[t.worker].inflight -= 1
        self.loads[t.worker].ctx_tokens -= t.ctx

    def fail(self, t: Ticket) -> None:
        """The request never reached the worker: undo its charge and its index insert."""
        if t.done:
            return
        if not t.first_seen:
            self.first_token(t)
        t.done = True
        self.loads[t.worker].inflight -= 1
        self.loads[t.worker].ctx_tokens -= t.ctx
        if t.chunks:
            self.index.forget(t.worker, t.chunks[t.match_chunks :])

    def reset_worker(self, w: int) -> None:
        """The worker lost its cache (restart, ``/reset_prefix_cache``)."""
        self.index.clear(w)
        self.index.set_capacity(w, self.cfg.capacity_chunks()[w])
