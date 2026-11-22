"""A vLLM-V1-like serving engine with automatic prefix caching, for the cluster simulator.

What is modelled, following vLLM 0.34's ``Scheduler`` and ``KVCacheManager``:

* **Block pool.** ``num_blocks`` blocks of ``block_size`` tokens. Free blocks sit in one LRU
  queue; never-used blocks come first, freed blocks join the tail, allocation takes the head,
  and taking a block that still holds a cached prefix evicts it.
* **Prefix cache.** A full prompt block becomes cached (hashable) once it is computed. On
  admission a request reuses the longest run of cached leading blocks, capped so that at least
  one token is computed (vLLM caps hits at ``num_tokens - 1``). Taking a cached block out of
  the free queue does not consume free capacity twice. A finished request's blocks are freed
  in reverse order, so its last blocks are evicted first.
* **Scheduling.** Each iteration has ``max_num_batched_tokens`` of budget. Running requests go
  first, in order (decodes take one token, unfinished prefills take what is left). If a running
  request cannot get a block, the most recently admitted running request is preempted
  (recompute: its blocks are freed and it goes back to the front of the queue). With no
  preemption in the iteration, waiting requests are admitted FCFS while budget, blocks and
  ``max_num_seqs`` allow; the last admitted prefill may be chunked.
* **Time.** Each iteration is priced by a :class:`~pcrouter.costmodel.StepModel`.

Generated tokens fill blocks that are never shared (their content is unique per request), so
they occupy and age the cache exactly like vLLM's, without ever producing hits.
"""

from __future__ import annotations

from collections import OrderedDict, deque
from dataclasses import dataclass, field

from pcrouter.costmodel import BatchStats, StepModel
from pcrouter.prompt import Prompt

__all__ = ["BlockPool", "Engine", "EngineConfig", "Seq"]


@dataclass
class EngineConfig:
    num_blocks: int
    block_size: int = 16
    max_num_batched_tokens: int = 8192
    max_num_seqs: int = 1024
    prefix_caching: bool = True

    def __post_init__(self) -> None:
        if self.num_blocks < 1 or self.block_size < 1:
            raise ValueError("num_blocks and block_size must be positive")
        if self.max_num_batched_tokens < 1 or self.max_num_seqs < 1:
            raise ValueError("max_num_batched_tokens and max_num_seqs must be positive")

    @property
    def capacity_tokens(self) -> int:
        return self.num_blocks * self.block_size


class BlockPool:
    """Block accounting. Entries are cache keys (>= 0) or unique negative ids (blocks whose
    content is never reused: partial blocks, generated tokens, duplicate computations)."""

    def __init__(self, num_blocks: int) -> None:
        self.num_blocks = num_blocks
        self.fresh = num_blocks
        self.free_q: OrderedDict[int, None] = OrderedDict()
        self.ref: dict[int, int] = {}
        self._uid = 0
        self.evictions = 0

    def num_free(self) -> int:
        return self.fresh + len(self.free_q)

    def usage(self) -> float:
        return 1.0 - self.num_free() / self.num_blocks

    def is_cached(self, key: int) -> bool:
        return key in self.ref or key in self.free_q

    def cached_prefix(self, keys: list[int], limit: int) -> int:
        n = 0
        while n < limit and (keys[n] in self.ref or keys[n] in self.free_q):
            n += 1
        return n

    def touch(self, key: int) -> None:
        r = self.ref.get(key)
        if r is None:
            del self.free_q[key]
            self.ref[key] = 1
        else:
            self.ref[key] = r + 1

    def alloc(self, n: int) -> list[int]:
        if n > self.num_free():
            raise RuntimeError("out of KV blocks")
        out = []
        for _ in range(n):
            if self.fresh:
                self.fresh -= 1
            else:
                key, _ = self.free_q.popitem(last=False)
                if key >= 0:
                    self.evictions += 1
            self._uid -= 1
            out.append(self._uid)
        return out

    def commit(self, blocks: list[int], i: int, key: int) -> None:
        """Block ``i`` of a request is now full and computed: make it cacheable as ``key``."""
        if blocks[i] >= 0 or key in self.ref or key in self.free_q:
            return  # already the cached copy, or a duplicate of one
        blocks[i] = key
        self.ref[key] = 1

    def free(self, blocks: list[int]) -> None:
        for e in reversed(blocks):
            if e >= 0:
                r = self.ref[e] - 1
                if r:
                    self.ref[e] = r
                    continue
                del self.ref[e]
            self.free_q[e] = None
        blocks.clear()


@dataclass(eq=False)
class Seq:
    """Engine-side state of one request."""

    rid: int
    prompt: Prompt
    max_out: int
    t_arrival: float
    keys: list[int] | None = None
    blocks: list[int] = field(default_factory=list)
    computed: int = 0
    out: int = 0
    cached: int = -1
    t_first: float = -1.0
    t_finish: float = -1.0
    preempted: int = 0
    rejected: bool = False

    @property
    def num_prompt(self) -> int:
        return self.prompt.num_tokens

    @property
    def total(self) -> int:
        return self.prompt.num_tokens + self.out


@dataclass
class StepEvents:
    sampled: list[Seq] = field(default_factory=list)  # produced one token this iteration
    first: list[Seq] = field(default_factory=list)
    finished: list[Seq] = field(default_factory=list)


class Engine:
    def __init__(self, cfg: EngineConfig, model: StepModel) -> None:
        self.cfg = cfg
        self.model = model
        self.pool = BlockPool(cfg.num_blocks)
        self.waiting: deque[Seq] = deque()
        self.running: list[Seq] = []
        self._sched: list[tuple[Seq, int]] = []
        # counters
        self.steps = 0
        self.busy_s = 0.0
        self.prefill_tokens = 0
        self.decode_tokens = 0
        self.prompt_tokens = 0
        self.hit_tokens = 0
        self.preemptions = 0

    # -- queries -------------------------------------------------------------------------
    def has_work(self) -> bool:
        return bool(self.waiting or self.running)

    def num_requests(self) -> int:
        return len(self.waiting) + len(self.running)

    def cached_prefix_tokens(self, prompt: Prompt) -> int:
        """Tokens of ``prompt`` this engine would reuse if admitted now (oracle routing)."""
        if not self.cfg.prefix_caching:
            return 0
        bs = self.cfg.block_size
        keys = prompt.block_keys(bs)
        limit = min(len(keys), (prompt.num_tokens - 1) // bs)
        return self.pool.cached_prefix(keys, limit) * bs

    def reset_prefix_cache(self) -> bool:
        """Drop every cached block not held by a running request (vLLM's endpoint refuses
        while requests run; we do the same)."""
        if self.running or self.waiting:
            return False
        self.pool = BlockPool(self.cfg.num_blocks)
        return True

    # -- iteration -----------------------------------------------------------------------
    def add(self, seq: Seq) -> bool:
        """Queue a request; False (and ``seq.rejected``) if it can never fit in the KV cache,
        which vLLM refuses up front (``max_model_len`` larger than the cache)."""
        if self._nblocks(seq.num_prompt + seq.max_out) > self.cfg.num_blocks:
            seq.rejected = True
            return False
        self.waiting.append(seq)
        return True

    def _nblocks(self, tokens: int) -> int:
        return -(-tokens // self.cfg.block_size)

    def _preempt(self, s: Seq) -> None:
        self.pool.free(s.blocks)
        s.computed = 0
        s.preempted += 1
        self.preemptions += 1
        self.waiting.appendleft(s)

    def begin_step(self) -> float:
        """Schedule one iteration; return its duration (0 if nothing could run)."""
        cfg, pool = self.cfg, self.pool
        bs = cfg.block_size
        budget = cfg.max_num_batched_tokens
        sched: list[tuple[Seq, int]] = []
        running = self.running
        preempted = False
        i = 0
        while i < len(running) and budget > 0:
            s = running[i]
            n = min(s.total - s.computed, budget)
            need = self._nblocks(s.computed + n) - len(s.blocks)
            scheduled = True
            while need > pool.num_free():
                victim = running.pop()
                self._preempt(victim)
                preempted = True
                if victim is s:
                    scheduled = False
                    break
            if not scheduled:
                break
            if need > 0:
                s.blocks.extend(pool.alloc(need))
            sched.append((s, n))
            budget -= n
            i += 1

        if not preempted:
            waiting = self.waiting
            while waiting and budget > 0 and len(running) < cfg.max_num_seqs:
                s = waiting[0]
                if s.keys is None:
                    s.keys = s.prompt.block_keys(bs)
                keys = s.keys
                hit = 0
                if cfg.prefix_caching:
                    hit = pool.cached_prefix(keys, min(len(keys), (s.total - 1) // bs))
                computed = hit * bs
                n = min(s.total - computed, budget)
                need = self._nblocks(computed + n) - hit
                in_free = sum(1 for k in keys[:hit] if k not in pool.ref)
                if need + in_free > pool.num_free():
                    break
                for k in keys[:hit]:
                    pool.touch(k)
                s.blocks = keys[:hit] + pool.alloc(need)
                s.computed = computed
                if s.cached < 0:
                    s.cached = computed
                    self.prompt_tokens += s.num_prompt
                    self.hit_tokens += computed
                waiting.popleft()
                running.append(s)
                sched.append((s, n))
                budget -= n

        self._sched = sched
        if not sched:
            return 0.0
        stats = BatchStats()
        for s, n in sched:
            stats.add(n, s.computed)
        dt = self.model.step_time(stats)
        self.steps += 1
        self.busy_s += dt
        return dt

    def end_step(self, now: float) -> StepEvents:
        """Apply the scheduled iteration's results at time ``now``."""
        ev = StepEvents()
        bs = self.cfg.block_size
        caching = self.cfg.prefix_caching
        pool = self.pool
        done = False
        for s, n in self._sched:
            before = s.computed
            s.computed = before + n
            if n > 1 or before < s.num_prompt:
                self.prefill_tokens += n
            else:
                self.decode_tokens += n
            if caching and s.keys is not None:
                hi = min(s.computed // bs, len(s.keys))
                for b in range(before // bs, hi):
                    pool.commit(s.blocks, b, s.keys[b])
            if s.computed == s.total:
                s.out += 1
                ev.sampled.append(s)
                if s.t_first < 0:
                    s.t_first = now
                    ev.first.append(s)
                if s.out >= s.max_out:
                    s.t_finish = now
                    pool.free(s.blocks)
                    s.keys = None
                    ev.finished.append(s)
                    done = True
        if done:
            self.running = [s for s in self.running if s.t_finish < 0]
        self._sched = []
        return ev
