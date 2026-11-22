"""The engine model, the cluster simulator, workloads and metrics."""

from __future__ import annotations

import itertools
import json
import math

import numpy as np
import pytest

from pcrouter.costmodel import StepModel, load_step_model
from pcrouter.metrics import Record, quantile_ci, read_records, summarize, write_records
from pcrouter.policies import POLICIES, make_policy
from pcrouter.prompt import BLOCK, Prompt
from pcrouter.sim.cluster import simulate
from pcrouter.sim.engine import BlockPool, Engine, EngineConfig, Seq
from pcrouter.workloads import load_mooncake, multiturn, shared_prefix, summarize_workload

MODEL = StepModel.simple(0.002, 1e-5, attn_per_m=1e-4)


def run_to_end(eng: Engine, t: float = 0.0, limit: int = 100_000) -> float:
    for _ in range(limit):
        if not eng.has_work():
            return t
        dt = eng.begin_step()
        if dt == 0:
            continue
        t += dt
        eng.end_step(t)
    raise AssertionError("engine did not drain")


def seq(rid, segments, n_tokens, out=4, t=0.0):
    return Seq(rid, Prompt(n_tokens, segments=segments), out, t)


# -- engine ----------------------------------------------------------------------------------


def test_engine_prefix_hit_and_cap():
    eng = Engine(EngineConfig(num_blocks=1000), MODEL)
    a = seq(0, [(1, 0, 8)], 8 * BLOCK)
    eng.add(a)
    run_to_end(eng)
    assert a.cached == 0 and a.out == 4 and a.t_finish > a.t_first > 0
    # Identical prompt: everything but the last block is reused (vLLM recomputes >= 1 token).
    b = seq(1, [(1, 0, 8)], 8 * BLOCK)
    eng.add(b)
    run_to_end(eng)
    assert b.cached == 7 * BLOCK
    # Longer prompt sharing the prefix: all 8 blocks reused.
    c = seq(2, [(1, 0, 8), (2, 8, 4)], 12 * BLOCK + 5)
    eng.add(c)
    run_to_end(eng)
    assert c.cached == 8 * BLOCK
    assert eng.hit_tokens == 15 * BLOCK
    assert eng.cached_prefix_tokens(c.prompt) == 12 * BLOCK


def test_concurrent_requests_do_not_hit_uncomputed_blocks():
    eng = Engine(EngineConfig(num_blocks=1000, max_num_batched_tokens=64), MODEL)
    a = seq(0, [(1, 0, 16)], 16 * BLOCK)
    b = seq(1, [(1, 0, 16)], 16 * BLOCK)
    eng.add(a)
    eng.add(b)
    run_to_end(eng)
    assert a.cached == 0
    # b was admitted after a's first 64-token chunk only: it reuses what was computed by then.
    assert 0 <= b.cached < 16 * BLOCK


def test_chunked_prefill_respects_budget():
    eng = Engine(EngineConfig(num_blocks=1000, max_num_batched_tokens=100), MODEL)
    s = seq(0, [(1, 0, 20)], 20 * BLOCK, out=2)
    eng.add(s)
    steps = 0
    while eng.has_work():
        eng.begin_step()
        assert sum(n for _, n in eng._sched) <= 100
        eng.end_step(float(steps))
        steps += 1
    assert steps == math.ceil(320 / 100) + 1  # prefill chunks, then one decode


def test_lru_eviction_keeps_recent_prefixes():
    eng = Engine(EngineConfig(num_blocks=40), MODEL)
    for rid, stream in enumerate((1, 2, 3)):  # 16 blocks each + output blocks
        eng.add(seq(rid, [(stream, 0, 16)], 16 * BLOCK, out=1))
        run_to_end(eng)
    assert eng.pool.evictions > 0
    # The oldest prefix lost its tail first (reverse-order free): its head may survive.
    first = Prompt(16 * BLOCK, segments=[(1, 0, 16)])
    last = Prompt(16 * BLOCK, segments=[(3, 0, 16)])
    assert eng.cached_prefix_tokens(last) == 15 * BLOCK
    assert eng.cached_prefix_tokens(first) < 15 * BLOCK


def test_preemption_and_recovery():
    eng = Engine(EngineConfig(num_blocks=12, max_num_batched_tokens=4096), MODEL)
    a = seq(0, [(1, 0, 4)], 4 * BLOCK, out=60)
    b = seq(1, [(2, 0, 4)], 4 * BLOCK, out=60)
    eng.add(a)
    eng.add(b)
    run_to_end(eng)
    assert a.out == 60 and b.out == 60
    assert eng.preemptions >= 1
    assert a.preempted + b.preempted == eng.preemptions


def test_reject_and_reset():
    eng = Engine(EngineConfig(num_blocks=4), MODEL)
    big = seq(0, [(1, 0, 8)], 8 * BLOCK)
    assert not eng.add(big) and big.rejected
    ok = seq(1, [(1, 0, 2)], 2 * BLOCK, out=1)
    assert eng.add(ok)
    assert not eng.reset_prefix_cache()  # busy
    run_to_end(eng)
    assert eng.reset_prefix_cache()
    assert eng.cached_prefix_tokens(ok.prompt) == 0
    with pytest.raises(ValueError):
        EngineConfig(num_blocks=0)
    with pytest.raises(ValueError):
        EngineConfig(num_blocks=1, max_num_seqs=0)


def test_block_pool_invariants():
    pool = BlockPool(10)
    blocks = pool.alloc(3)
    pool.commit(blocks, 0, 100)
    pool.commit(blocks, 0, 100)  # idempotent
    other = pool.alloc(1)
    pool.commit(other, 0, 100)  # duplicate stays anonymous
    assert other[0] < 0
    pool.free(blocks)
    assert pool.is_cached(100) and pool.num_free() == 9
    pool.touch(100)
    pool.touch(100)
    assert pool.ref[100] == 2
    with pytest.raises(RuntimeError):
        pool.alloc(100)


# -- cluster ---------------------------------------------------------------------------------


def small_cfg(blocks=4000):
    return EngineConfig(num_blocks=blocks, max_num_batched_tokens=2048, max_num_seqs=64)


def test_simulate_every_policy_and_determinism():
    # 24 prefixes x 64 blocks do not fit in one worker's 900 blocks, but a third of them do:
    # partitioning prefixes across workers is the only way to reuse them.
    reqs = shared_prefix(300, rate=30.0, num_prefixes=24, zipf_s=0.5, prefix_len=1024, seed=3)
    out = {}
    for name in POLICIES:
        r1 = simulate(
            reqs, make_policy(name, 3, seed=1, model=MODEL), small_cfg(900), MODEL, chunk_tokens=256
        )
        r2 = simulate(
            reqs, make_policy(name, 3, seed=1, model=MODEL), small_cfg(900), MODEL, chunk_tokens=256
        )
        assert r1.summary["measured"] == 300
        assert r1.summary["ttft_p99"] == r2.summary["ttft_p99"]
        out[name] = r1.summary
    # Cache-aware placement reuses far more than round robin on a shared-prefix workload.
    assert out["ttft"]["hit_ratio"] > out["round-robin"]["hit_ratio"] + 0.1
    assert out["affinity"]["hit_ratio"] > out["round-robin"]["hit_ratio"] + 0.1
    assert 0.0 <= out["ttft"]["slo_attainment"] <= 1.0


def test_oracle_and_rejection():
    reqs = multiturn(60, session_rate=5.0, system_len=1024, seed=2)
    r = simulate(
        reqs, make_policy("ttft", 2, model=MODEL), small_cfg(), MODEL, chunk_tokens=256, oracle=True
    )
    assert r.summary["measured"] == len(reqs)
    # Requests that cannot fit in a tiny cache fail cleanly.
    tiny = simulate(
        reqs[:5], make_policy("random", 1), EngineConfig(num_blocks=8), MODEL, chunk_tokens=256
    )
    assert tiny.summary["failed"] > 0
    with pytest.raises(ValueError):
        simulate(reqs, make_policy("random", 3), [small_cfg()] * 2, MODEL)
    with pytest.raises(ValueError):
        simulate(reqs, make_policy("random", 2), small_cfg(), MODEL, chunk_tokens=100)


def test_simulator_against_packaged_h200_model():
    m = load_step_model("h200-qwen2.5-7b")
    reqs = shared_prefix(200, rate=5.0, seed=0)
    r = simulate(reqs, make_policy("ttft", 2, model=m), EngineConfig(num_blocks=100_000), m)
    # A cold ~4.3k-token prefill on an idle H200 for Qwen2.5-7B takes ~0.1 s; later requests
    # reuse the 4k-token prefix and only prefill their suffix.
    assert 0.05 < r.records[0].ttft < 0.5
    assert r.summary["ttft_p50"] < r.records[0].ttft
    assert r.summary["busy_imbalance"] >= 1.0


# -- workloads -------------------------------------------------------------------------------


def test_mooncake_loader(tmp_path):
    rows = [
        {"timestamp": 0, "input_length": 600, "output_length": 10, "hash_ids": [0, 1]},
        {"timestamp": 1000, "input_length": 1100, "output_length": 5, "hash_ids": [0, 1, 2]},
        {"timestamp": 2000, "input_length": 90000, "output_length": 5, "hash_ids": [0]},
        {"timestamp": 3000, "input_length": 300, "output_length": 0, "hash_ids": [0]},
    ]
    path = tmp_path / "t.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n\n")
    reqs = load_mooncake(path, time_scale=2.0, max_input_len=32768, max_output_len=8)
    assert [q.arrival for q in reqs] == [0.0, 0.5, 1.5]
    assert reqs[0].output_len == 8 and reqs[2].output_len == 1
    k0, k1 = reqs[0].prompt.block_keys(), reqs[1].prompt.block_keys()
    assert k1[: len(k0)] == k0  # shared hashes -> shared blocks
    assert len(load_mooncake(path, start_s=1.0, duration_s=1.5)) == 2
    assert len(load_mooncake(path, start_s=1.0, duration_s=1.5, max_input_len=32768)) == 1
    assert len(load_mooncake(path, limit=1)) == 1
    with pytest.raises(ValueError):
        load_mooncake(path, time_scale=0)
    s = summarize_workload(reqs)
    assert s["requests"] == 3 and 0 < s["ideal_hit_ratio"] < 1
    assert summarize_workload([]) == {"requests": 0}


def test_multiturn_conversations_grow():
    reqs = multiturn(20, seed=1, system_len=512, turns_mean=5)
    by = {}
    for q in reqs:
        by.setdefault(q.tag, []).append(q)
    longest = max(by.values(), key=len)
    longest.sort(key=lambda q: q.arrival)
    for a, b in itertools.pairwise(longest):
        assert b.input_len > a.input_len
        ka, kb = a.prompt.block_keys(), b.prompt.block_keys()
        assert kb[: len(ka)] == ka
    assert summarize_workload(reqs)["ideal_hit_ratio"] > 0.3


def test_shared_prefix_hot_key():
    reqs = shared_prefix(500, zipf_s=4.0, seed=0)
    tags = np.bincount([q.tag for q in reqs])
    assert tags.max() / len(reqs) > 0.85


# -- metrics ---------------------------------------------------------------------------------


def test_quantile_ci_brackets_estimate():
    x = np.random.default_rng(0).exponential(1.0, 2000)
    est, lo, hi = quantile_ci(x, 0.99)
    assert lo <= est <= hi
    assert math.isnan(quantile_ci([], 0.5)[0])


def test_summarize_and_roundtrip(tmp_path):
    recs = [
        Record(i, i % 2, float(i), 0.1 * (i + 1), 1.0 + i, 1000, 10, 500, 600) for i in range(10)
    ]
    recs.append(Record(10, 0, 10.0, -1, -1, 1000, 10, -1, ok=False))
    s = summarize(recs, slo_ttft=0.5, num_workers=2, warmup_s=1.0)
    assert s["failed"] == 1 and s["measured"] == 9
    assert s["hit_ratio"] == pytest.approx(0.5)
    assert s["est_overshoot"] == pytest.approx(0.2)
    assert 0 < s["slo_attainment"] < 1
    assert s["prefill_imbalance"] >= 1.0
    assert summarize([])["requests"] == 0
    p = tmp_path / "r.jsonl"
    write_records(p, recs)
    back = read_records(p)
    assert back == recs
    assert math.isnan(recs[-1].tpot)
