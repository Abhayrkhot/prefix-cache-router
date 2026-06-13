"""Tests for the prefix cache router simulator."""

import importlib
import os
import sys

import pytest

# Adjust sys.path so that the module can be imported when running tests
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))  # noqa

from router import PrefixCacheRouter, simulate_router  # type: ignore  # noqa: E402


def test_basic_routing_assignments():
    """Requests with the same prefix should be routed to the same worker and assignments should balance."""
    router = PrefixCacheRouter(num_workers=3)
    # First request for prefix A -> assigns to worker 0
    w0 = router.route("A")
    assert w0 in {0, 1, 2}
    # Second request for the same prefix should hit the cache and route to the same worker
    w0_again = router.route("A")
    assert w0_again == w0
    # New prefix B should be assigned to a different worker (with the smallest load)
    w1 = router.route("B")
    assert w1 != w0
    # Another request for A should still go to w0
    assert router.route("A") == w0
    # Prefix C should be assigned to the remaining worker
    w2 = router.route("C")
    assert w2 not in (w0, w1)
    # Check statistics
    summary = router.summary()
    assert summary["total_requests"] == 5
    # There should be 3 misses (A, B, C first occurrences) and 2 hits
    assert summary["hits"] == 2
    assert summary["misses"] == 3
    # There should be exactly three unique assignments
    assert set(summary["assignments"].keys()) == {"A", "B", "C"}


def test_simulate_router_repeat_prob_one():
    """With repeat_prob=1.0, every request after the first should repeat the same prefix and be a cache hit."""
    prefixes = ["X", "Y"]
    result = simulate_router(num_workers=2, prefixes=prefixes, repeat_prob=1.0, num_requests=10, seed=123)
    # Exactly one miss (the first request), rest hits
    assert result["misses"] == 1
    assert result["hits"] == 9
    assert abs(result["hit_rate"] - 0.9) < 1e-6
    # The prefix assignments should contain exactly one prefix (the one chosen first)
    assert len(result["assignments"]) == 1
