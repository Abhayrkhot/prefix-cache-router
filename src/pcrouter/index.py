"""The router's model of which prefix chunks each worker holds in its KV cache.

The router never sees the engines' caches; it predicts them. Every request it sends to worker
``w`` leaves the request's prefix chunks in ``w``'s cache, so the index inserts them there. Each
worker's entry is an LRU bounded by that worker's estimated cache capacity, which mirrors how
vLLM evicts: freed blocks join the tail of one LRU free queue and are evicted from its head.
vLLM frees a request's blocks *in reverse*, so a prompt's last blocks are evicted before its
first ones; :meth:`PrefixIndex.insert` touches chunks in reverse for the same effect, which also
keeps parents resident at least as long as their children.

Matching walks the request's chunks once against a ``chunk -> worker bitmask`` map, so the cost
is O(matched chunks) regardless of the number of workers.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Sequence

__all__ = ["PrefixIndex"]


class PrefixIndex:
    """Per-worker LRU sets of chunk keys plus a reverse map for O(L) longest-prefix match."""

    def __init__(self, num_workers: int, capacity_chunks: int | Sequence[int]) -> None:
        if num_workers < 1:
            raise ValueError("need at least one worker")
        caps = (
            [int(capacity_chunks)] * num_workers
            if isinstance(capacity_chunks, int)
            else [int(c) for c in capacity_chunks]
        )
        if len(caps) != num_workers or min(caps) < 1:
            raise ValueError("need one positive capacity per worker")
        self.num_workers = num_workers
        self.capacity = caps
        self._lru: list[OrderedDict[int, None]] = [OrderedDict() for _ in range(num_workers)]
        self._owners: dict[int, int] = {}
        self.evictions = [0] * num_workers

    def __len__(self) -> int:
        return len(self._owners)

    def size(self, worker: int) -> int:
        """Number of chunks the index believes ``worker`` holds."""
        return len(self._lru[worker])

    def match(self, chunks: Sequence[int]) -> list[int]:
        """Length (in chunks) of the longest cached prefix of ``chunks`` on every worker."""
        n = self.num_workers
        out = [0] * n
        alive = (1 << n) - 1
        owners = self._owners
        for depth, key in enumerate(chunks):
            still = owners.get(key, 0) & alive
            dropped = alive & ~still
            while dropped:
                low = dropped & -dropped
                out[low.bit_length() - 1] = depth
                dropped ^= low
            alive = still
            if not alive:
                return out
        full = len(chunks)  # workers still alive matched every chunk
        while alive:
            low = alive & -alive
            out[low.bit_length() - 1] = full
            alive ^= low
        return out

    def best(self, chunks: Sequence[int]) -> tuple[int, int]:
        """(worker, matched chunks) with the longest match; worker -1 if nothing matches."""
        m = self.match(chunks)
        top = max(m)
        return (m.index(top), top) if top > 0 else (-1, 0)

    def insert(self, worker: int, chunks: Sequence[int]) -> None:
        """Record that ``worker`` now caches all of ``chunks`` (most recently used)."""
        lru = self._lru[worker]
        owners = self._owners
        bit = 1 << worker
        for key in reversed(chunks):
            if key in lru:
                lru.move_to_end(key)
            else:
                lru[key] = None
                owners[key] = owners.get(key, 0) | bit
        cap = self.capacity[worker]
        while len(lru) > cap:
            key, _ = lru.popitem(last=False)
            self._drop_owner(key, bit)
            self.evictions[worker] += 1

    def forget(self, worker: int, chunks: Sequence[int]) -> None:
        """Remove ``chunks`` from ``worker`` (e.g. feedback says they were not cached)."""
        lru = self._lru[worker]
        bit = 1 << worker
        for key in chunks:
            if key in lru:
                del lru[key]
                self._drop_owner(key, bit)

    def clear(self, worker: int | None = None) -> None:
        """Forget one worker's cache (it restarted or was reset), or every worker's."""
        workers = range(self.num_workers) if worker is None else (worker,)
        for w in workers:
            bit = 1 << w
            for key in self._lru[w]:
                self._drop_owner(key, bit)
            self._lru[w].clear()

    def set_capacity(self, worker: int, capacity_chunks: int) -> None:
        if capacity_chunks < 1:
            raise ValueError("capacity must be positive")
        self.capacity[worker] = capacity_chunks
        lru = self._lru[worker]
        bit = 1 << worker
        while len(lru) > capacity_chunks:
            key, _ = lru.popitem(last=False)
            self._drop_owner(key, bit)
            self.evictions[worker] += 1

    def _drop_owner(self, key: int, bit: int) -> None:
        rest = self._owners.get(key, 0) & ~bit
        if rest:
            self._owners[key] = rest
        else:
            self._owners.pop(key, None)
