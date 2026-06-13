"""Prefix cache aware router simulator.

Prefix caching is an optimisation that allows multiple inference
requests with the **same prefix** to reuse the precomputed KV cache
from a previous request【239421193071328†L60-L67】.  Rather than recomputing the
attention keys and values for the identical prompt, the server can
resume decoding from the cached state【239421193071328†L75-L83】.  This can
significantly reduce latency and cost for workloads with repeated
prompt structures such as chat systems or retrieval‑augmented
generation pipelines【239421193071328†L60-L67】【239421193071328†L139-L151】.

This module implements a simple **prefix cache router**.  The router
maintains a mapping from prompt prefixes to worker IDs.  When a new
request arrives, the router checks whether the prefix is already
assigned to a worker.  If so, it routes the request to that worker to
reuse the cached KV entries (a **cache hit**).  Otherwise, it assigns
the prefix to a worker with the least number of unique prefixes (a
simple load balancing heuristic) and records a **cache miss**.  The
router tracks hit/miss statistics and per‑worker assignments.

You can simulate a stream of requests with random prefixes using the
command‑line interface.  The simulator reports the overall cache hit
rate and per‑worker load distribution.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class RouterStats:
    """Statistics collected by the prefix cache router."""

    total_requests: int = 0
    hits: int = 0
    misses: int = 0
    assignments: Dict[str, int] = field(default_factory=dict)

    def hit_rate(self) -> float:
        return self.hits / self.total_requests if self.total_requests > 0 else 0.0


class PrefixCacheRouter:
    """A simple router that assigns request prefixes to workers to maximise cache hits."""

    def __init__(self, num_workers: int) -> None:
        if num_workers <= 0:
            raise ValueError("num_workers must be positive")
        self.num_workers = num_workers
        # Map from prefix to worker id
        self.prefix_to_worker: Dict[str, int] = {}
        # Map from worker id to number of unique prefixes assigned
        self.worker_load: Dict[int, int] = {i: 0 for i in range(num_workers)}
        self.stats = RouterStats()

    def route(self, prefix: str) -> int:
        """Route a request with the given prefix to a worker.

        Parameters
        ----------
        prefix : str
            The prompt prefix.  Exact match is required for a cache hit【239421193071328†L86-L87】.

        Returns
        -------
        int
            The ID of the worker to which the request is routed.
        """
        self.stats.total_requests += 1
        if prefix in self.prefix_to_worker:
            # Cache hit: reuse worker
            worker = self.prefix_to_worker[prefix]
            self.stats.hits += 1
        else:
            # Cache miss: assign to worker with fewest unique prefixes
            worker = min(self.worker_load, key=self.worker_load.get)
            self.prefix_to_worker[prefix] = worker
            self.worker_load[worker] += 1
            self.stats.misses += 1
        return worker

    def summary(self) -> Dict[str, object]:
        """Return a summary of routing statistics."""
        return {
            "total_requests": self.stats.total_requests,
            "hits": self.stats.hits,
            "misses": self.stats.misses,
            "hit_rate": self.stats.hit_rate(),
            "assignments": dict(self.prefix_to_worker),
            "worker_load": dict(self.worker_load),
        }


def simulate_router(num_workers: int, prefixes: List[str], repeat_prob: float,
                    num_requests: int, seed: Optional[int] = None) -> Dict[str, object]:
    """Simulate a stream of requests and return router statistics.

    Parameters
    ----------
    num_workers : int
        Number of workers available.
    prefixes : List[str]
        Pool of possible prefixes.  Requests will be drawn from this pool.
    repeat_prob : float
        Probability that a request repeats the previous request's prefix.
        A higher value increases the likelihood of cache hits.
    num_requests : int
        Total number of requests to simulate.
    seed : Optional[int]
        Random seed for reproducibility.

    Returns
    -------
    Dict[str, object]
        Summary statistics from the router.
    """
    rng = random.Random(seed)
    router = PrefixCacheRouter(num_workers)
    last_prefix: Optional[str] = None
    for _ in range(num_requests):
        if last_prefix is not None and rng.random() < repeat_prob:
            prefix = last_prefix
        else:
            prefix = rng.choice(prefixes)
        router.route(prefix)
        last_prefix = prefix
    return router.summary()


def cli(argv: Optional[List[str]] = None) -> None:
    """Command‑line interface for simulating the prefix cache router."""
    parser = argparse.ArgumentParser(description="Simulate a prefix cache aware router.")
    parser.add_argument("--workers", type=int, default=4, help="Number of workers")
    parser.add_argument("--prefixes", type=str, default="A,B,C", help="Comma‑separated prefix pool")
    parser.add_argument("--repeat-prob", type=float, default=0.5, help="Probability of repeating previous prefix")
    parser.add_argument("--requests", type=int, default=100, help="Number of requests to simulate")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    args = parser.parse_args(argv)

    prefixes = [p.strip() for p in args.prefixes.split(",") if p.strip()]
    result = simulate_router(
        num_workers=args.workers,
        prefixes=prefixes,
        repeat_prob=args.repeat_prob,
        num_requests=args.requests,
        seed=args.seed,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    cli()