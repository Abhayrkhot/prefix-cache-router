"""Prompt keys, the prefix index, the step model and the router's bookkeeping."""

from __future__ import annotations

import random

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from pcrouter.costmodel import BatchStats, StepModel, load_step_model
from pcrouter.index import PrefixIndex
from pcrouter.policies import make_policy
from pcrouter.prompt import BLOCK, STREAM_SHIFT, Prompt, text_block_keys, token_block_keys
from pcrouter.router import Router, RouterConfig

# -- prompts -------------------------------------------------------------------------------


def test_segment_block_keys_and_truncation():
    p = Prompt(5 * BLOCK + 3, segments=[(7, 0, 2), (9, 2, 10)])
    keys = p.block_keys()
    assert len(keys) == 5  # partial last block excluded
    assert keys[:2] == [(7 << STREAM_SHIFT) | 0, (7 << STREAM_SHIFT) | 1]
    assert keys[2:] == [(9 << STREAM_SHIFT) | i for i in (2, 3, 4)]
    assert p.chunk_keys(2) == [keys[1], keys[3]]
    with pytest.raises(ValueError):
        p.block_keys(32)


def test_prompt_validation():
    with pytest.raises(ValueError):
        Prompt(0, keys=[])
    with pytest.raises(ValueError):
        Prompt(10)
    with pytest.raises(ValueError):
        Prompt(10, segments=[], keys=[])
    with pytest.raises(ValueError):
        Prompt(32, keys=[1, 2]).chunk_keys(0)
    assert "keys" in repr(Prompt(32, keys=[1, 2]))


@given(
    st.lists(st.integers(0, 2**31), min_size=0, max_size=200),
    st.lists(st.integers(0, 2**31), min_size=0, max_size=200),
    st.lists(st.integers(0, 2**31), min_size=0, max_size=200),
)
@settings(max_examples=60, deadline=None)
def test_token_keys_are_prefix_chained(common, a, b):
    ka = token_block_keys(common + a, 16)
    kb = token_block_keys(common + b, 16)
    shared = len(common) // 16
    assert ka[:shared] == kb[:shared]
    # Beyond the common prefix keys differ whenever the blocks differ.
    for i in range(shared, min(len(ka), len(kb))):
        if (common + a)[: (i + 1) * 16] != (common + b)[: (i + 1) * 16]:
            assert ka[i] != kb[i]


def test_same_block_different_parent_gets_different_key():
    blk = list(range(16))
    k1 = token_block_keys([1] * 16 + blk, 16)
    k2 = token_block_keys([2] * 16 + blk, 16)
    assert k1[1] != k2[1]
    assert token_block_keys(blk[:15], 16) == []


def test_text_keys():
    a = text_block_keys("x" * 100 + "tail-a", 32)
    b = text_block_keys("x" * 100 + "tail-b-longer", 32)
    assert a[:3] == b[:3]
    assert len(text_block_keys("é" * 40, 32)) == 2  # UTF-8 bytes, not characters


# -- prefix index ----------------------------------------------------------------------------


def test_index_match_insert_evict():
    idx = PrefixIndex(3, capacity_chunks=4)
    assert idx.match([1, 2, 3]) == [0, 0, 0]
    assert idx.best([1, 2]) == (-1, 0)
    idx.insert(0, [1, 2, 3])
    idx.insert(1, [1, 2])
    assert idx.match([1, 2, 3, 4]) == [3, 2, 0]
    assert idx.best([1, 2, 3]) == (0, 3)
    assert idx.match([]) == [0, 0, 0]
    # Capacity 4: inserting two new chunks evicts the LRU ones. Reverse insertion means the
    # last chunk of the oldest prompt goes first, so the prefix [1, 2] survives.
    idx.insert(0, [7, 8])
    assert idx.size(0) == 4
    assert idx.match([1, 2, 3]) == [2, 2, 0]
    assert idx.evictions[0] == 1
    idx.forget(0, [1])
    assert idx.match([1, 2]) == [0, 2, 0]
    idx.clear(1)
    assert idx.match([1, 2]) == [0, 0, 0]
    idx.set_capacity(0, 1)
    assert idx.size(0) == 1
    idx.clear()
    assert len(idx) == 0


def test_index_validation():
    with pytest.raises(ValueError):
        PrefixIndex(0, 1)
    with pytest.raises(ValueError):
        PrefixIndex(2, [1])
    with pytest.raises(ValueError):
        PrefixIndex(1, 1).set_capacity(0, 0)


@given(st.data())
@settings(max_examples=80, deadline=None)
def test_index_matches_reference_model(data):
    """Bitmask index == per-worker LRU sets with a naive prefix walk."""
    n = data.draw(st.integers(1, 5))
    cap = data.draw(st.integers(1, 12))
    idx = PrefixIndex(n, cap)
    ref: list[list[int]] = [[] for _ in range(n)]  # LRU order, last = most recent
    alphabet = st.integers(0, 15)
    for _ in range(data.draw(st.integers(1, 30))):
        w = data.draw(st.integers(0, n - 1))
        chunks = data.draw(st.lists(alphabet, min_size=0, max_size=8, unique=True))
        idx.insert(w, chunks)
        for c in reversed(chunks):
            if c in ref[w]:
                ref[w].remove(c)
            ref[w].append(c)
        while len(ref[w]) > cap:
            ref[w].pop(0)
        probe = data.draw(st.lists(alphabet, min_size=0, max_size=8, unique=True))
        expect = []
        for k in range(n):
            m = 0
            while m < len(probe) and probe[m] in ref[k]:
                m += 1
            expect.append(m)
        assert idx.match(probe) == expect
        assert [idx.size(k) for k in range(n)] == [len(r) for r in ref]


# -- step model ------------------------------------------------------------------------------


def test_packaged_step_model():
    m = load_step_model("h200-qwen2.5-7b")
    assert m.meta["engine"].startswith("vLLM")
    s = BatchStats()
    assert m.step_time(s) == 0.0
    small, big = BatchStats(), BatchStats()
    small.add(512, 0)
    big.add(8192, 0)
    assert 0 < m.step_time(small) < m.step_time(big)
    # Attention over a cached prefix makes the same new tokens costlier.
    assert m.prefill_time(4096, 16384) > m.prefill_time(4096, 0) > 0
    assert m.prefill_time(0, 100) == 0.0
    assert m.decode_slowdown(0) == 1.0
    assert m.decode_slowdown(256) > m.decode_slowdown(16) > 1.0
    assert m.decode_slowdown(100_000) >= m.decode_slowdown(256)
    rt = StepModel.from_dict(m.to_dict())
    assert rt.step_time(big) == pytest.approx(m.step_time(big))


def test_step_model_validation(tmp_path):
    with pytest.raises(ValueError):
        StepModel(coef={"const": 1.0})
    with pytest.raises(ValueError):
        StepModel.from_dict({"kind": "roofline", "coef": {}})
    bad = StepModel.simple(0.001, 1e-5).to_dict()
    bad["coef"]["const"] = -1.0
    path = tmp_path / "m.json"
    import json

    path.write_text(json.dumps(bad))
    with pytest.raises(ValueError):
        load_step_model(path)
    simple = StepModel.simple(0.01, 1e-5, attn_per_m=1e-3)
    s = BatchStats()
    s.add(1, 100)  # a decode
    assert simple.step_time(s) == pytest.approx(0.01 + 1e-5)


# -- router bookkeeping ----------------------------------------------------------------------


def _router(policy="ttft", n=2, **kw):
    model = StepModel.simple(0.005, 2e-5, attn_per_m=1e-3)
    cfg = RouterConfig(num_workers=n, chunk_tokens=32, capacity_tokens=32 * 100, **kw)
    return Router(make_policy(policy, n, model=model), cfg, model)


def test_router_accounting_and_affinity():
    r = _router()
    chunks = [11, 12, 13, 14]
    t1 = r.route(32 * 4, 8, chunks)
    assert t1.cached_est == 0 and t1.uncached_est == 128
    ld = r.loads[t1.worker]
    assert (ld.inflight, ld.prefilling, ld.pending_tokens) == (1, 1, 128)
    r.first_token(t1, cached_tokens=0)
    assert ld.prefilling == 0 and ld.pending_tokens == 0 and ld.decoding == 1
    r.finish(t1)
    r.finish(t1)  # idempotent
    assert ld.inflight == 0
    # The same prefix now prefers the worker that has it cached.
    t2 = r.route(32 * 4 + 5, 8, chunks)
    assert t2.worker == t1.worker and t2.cached_est == 128
    r.finish(t2)
    assert ld.prefilling == 0


def test_router_fail_undoes_insert_and_reset():
    r = _router("round-robin")
    t = r.route(64, 4, [1, 2])
    r.fail(t)
    assert r.loads[t.worker].inflight == 0
    assert r.index.match([1, 2])[t.worker] == 0
    t = r.route(64, 4, [1, 2])
    r.finish(t)
    r.reset_worker(t.worker)
    assert r.index.match([1, 2])[t.worker] == 0
    with pytest.raises(RuntimeError):
        r.route(64, 4, [1], healthy=[False, False])


def test_router_feedback_shrinks_capacity():
    r = _router(feedback=True)
    start = r.index.capacity[0]
    t = r.route(32 * 8, 4, list(range(8)), healthy=[True, False])
    r.finish(t)
    t = r.route(32 * 8, 4, list(range(8)), healthy=[True, False])
    assert t.cached_est == 32 * 8 - 1
    r.first_token(t, cached_tokens=0)  # nothing was actually cached
    assert r.index.capacity[0] < start
    assert r.loads[0].predicted_cached > r.loads[0].actual_cached
    r.finish(t)


def test_router_config_validation():
    with pytest.raises(ValueError):
        RouterConfig(num_workers=2, capacity_tokens=[1]).capacity_chunks()
    with pytest.raises(ValueError):
        Router(make_policy("random", 1), RouterConfig(num_workers=1, chunk_tokens=0))


def test_route_healthy_mask_is_respected():
    r = _router("random", n=3)
    rng = random.Random(0)
    for _ in range(50):
        mask = [rng.random() < 0.5 for _ in range(3)]
        if not any(mask):
            continue
        t = r.route(100, 4, [rng.randrange(5)], healthy=mask)
        assert mask[t.worker]
        r.finish(t)
