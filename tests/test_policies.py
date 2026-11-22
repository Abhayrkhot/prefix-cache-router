from __future__ import annotations

import pytest

from pcrouter.costmodel import StepModel
from pcrouter.policies import (
    CHWBL,
    POLICIES,
    Affinity,
    Legacy,
    PowerOfTwo,
    RouteRequest,
    Threshold,
    TTFTCost,
    WorkerView,
    make_policy,
)

MODEL = StepModel.simple(0.005, 2e-5, attn_per_m=1e-3)


def view(
    i, inflight=0, match=0, pending=0, pending_s=0.0, decoding=0, index=0, chunk=512, input_len=4096
):
    return WorkerView(
        id=i,
        inflight=inflight,
        prefilling=max(0, inflight - decoding),
        decoding=decoding,
        pending_prefill_tokens=pending,
        pending_prefill_s=pending_s,
        match_chunks=match,
        cached_tokens=min(match * chunk, input_len - 1),
        index_chunks=index,
        capacity_chunks=1000,
    )


def req(chunks=(1, 2, 3, 4, 5, 6, 7, 8), input_len=4096):
    return RouteRequest(input_len, 64, list(chunks), 512)


def test_registry_and_factory():
    assert set(POLICIES) == {
        "random",
        "round-robin",
        "least-request",
        "p2c",
        "chwbl",
        "legacy",
        "threshold",
        "affinity",
        "ttft",
    }
    with pytest.raises(ValueError):
        make_policy("nope", 2)
    with pytest.raises(ValueError):
        make_policy("ttft", 2)  # needs a step model
    with pytest.raises(ValueError):
        make_policy("ttft", 2, model=MODEL, beta=-1)
    with pytest.raises(ValueError):
        make_policy("affinity", 2, mode="x")
    with pytest.raises(ValueError):
        make_policy("chwbl", 2, key_depth=0)
    for name in POLICIES:
        p = make_policy(name, 3, model=MODEL)
        assert isinstance(p.params(), dict)
        assert p.choose(req(), [view(0), view(1), view(2)]) in (0, 1, 2)
        assert p.choose(req(), [view(2)]) == 2


def test_round_robin_skips_unhealthy():
    p = make_policy("round-robin", 3)
    assert [p.choose(req(), [view(0), view(1), view(2)]) for _ in range(4)] == [0, 1, 2, 0]
    assert p.choose(req(), [view(0), view(2)]) == 2
    assert p.choose(req(), [view(0), view(2)]) == 0


def test_least_request_and_p2c():
    lr = make_policy("least-request", 3)
    assert lr.choose(req(), [view(0, 5), view(1, 1), view(2, 3)]) == 1
    p2 = PowerOfTwo(2)
    assert p2.choose(req(), [view(0, 9), view(1, 0)]) == 1


def test_random_tiebreak_is_unbiased():
    lr = make_policy("least-request", 4, seed=1)
    picks = {lr.choose(req(), [view(i) for i in range(4)]) for _ in range(200)}
    assert picks == {0, 1, 2, 3}


def test_chwbl_sticky_and_bounded():
    p = CHWBL(4, key_depth=2, load_factor=1.25)
    views = [view(i) for i in range(4)]
    w = p.choose(req(), views)
    assert all(p.choose(req(), views) == w for _ in range(5))
    # Same key, owner at the load bound: the request spills to another worker.
    loaded = [view(i, inflight=(10 if i == w else 0)) for i in range(4)]
    assert p.choose(req(), loaded) != w
    # No chunks: least-loaded fallback.
    assert p.choose(req(chunks=()), [view(0, 3), view(1, 0)]) == 1
    # Owner unhealthy: next on the ring.
    assert p.choose(req(), [v for v in views if v.id != w]) != w


def test_legacy_pins_keys():
    p = Legacy(2, key_depth=1)
    a = p.choose(req(chunks=(1,)), [view(0), view(1)])
    b = p.choose(req(chunks=(2,)), [view(0), view(1)])
    assert a != b
    assert p.choose(req(chunks=(1,)), [view(0, 50), view(1, 0)]) == a  # ignores load
    assert p.choose(req(chunks=(1,)), [view(1 - a)]) == 1 - a  # owner down: re-pin
    assert p.choose(req(chunks=()), [view(0), view(1)]) in (0, 1)


def test_threshold_rule():
    p = Threshold(3, cache_threshold=0.5, balance_abs=4, balance_rel=1.5)
    # Balanced, good match: longest match wins.
    assert p.choose(req(), [view(0, match=6), view(1, match=2), view(2)]) == 0
    # Match below threshold: emptiest cache.
    assert p.choose(req(), [view(0, match=2, index=50), view(1, index=5), view(2, index=9)]) == 1
    # Imbalanced: shortest queue regardless of cache.
    assert p.choose(req(), [view(0, 20, match=8), view(1, 2), view(2, 9)]) == 1


def test_affinity_rule():
    p = Affinity(3, min_match_tokens=1024, mode="balanced", load_gap=1024, load_factor=2.0)
    # Hit with acceptable load: owner (fewest uncached tokens).
    assert p.choose(req(), [view(0, match=4), view(1, match=6), view(2)]) == 1
    # Owner overloaded relative to the alternatives: yields.
    hot = [view(0, match=6, pending=50_000), view(1, pending=0), view(2, pending=0)]
    assert p.choose(req(), hot) in (1, 2)
    # Prefer mode never yields.
    pref = Affinity(3, mode="prefer")
    assert pref.choose(req(), hot) == 0
    # Below min_match_tokens: no hit, power of two on prefill load.
    assert p.choose(req(), [view(0, match=1, pending=9000), view(1, pending=0)]) == 1


def test_ttft_cost_prefers_cache_until_queue_dominates():
    p = TTFTCost(2, model=MODEL, beta=0.0)
    r = req(input_len=8192, chunks=tuple(range(16)))
    cached, empty = view(0, match=15, input_len=8192), view(1, input_len=8192)
    assert p.choose(r, [cached, empty]) == 0
    busy = view(0, match=15, pending_s=5.0, input_len=8192)
    assert p.choose(r, [busy, empty]) == 1
    assert p.estimate(r, busy) > p.estimate(r, empty)
    # A larger beta keeps the request on the cached worker for longer.
    mild = view(0, match=15, pending_s=0.3, input_len=8192)
    assert TTFTCost(2, model=MODEL, beta=0.0).choose(r, [mild, empty]) == 1
    assert TTFTCost(2, model=MODEL, beta=8.0).choose(r, [mild, empty]) == 0
    # Decodes sharing the iterations make a worker look slower.
    assert p.slowdown(64) > 1.0
    assert p.params()["beta"] == 0.0
