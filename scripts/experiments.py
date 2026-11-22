"""Simulation experiments behind the README. Each writes ``results/sim/<name>.json``.

    python scripts/experiments.py all          # everything (~15 min on 18 cores)
    python scripts/experiments.py load beta    # selected experiments

Cluster: 4 workers, each one H200 serving Qwen2.5-7B with vLLM 0.34 (2,022,400 KV tokens,
8192-token prefill budget), priced by the packaged step model. Traces are filtered exactly as
in the H200 runs (input <= 30720, output <= 2048). Latency figures exclude the first 60 s of
trace time (cold caches).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from pcrouter.costmodel import load_step_model  # noqa: E402
from pcrouter.policies import POLICIES, make_policy  # noqa: E402
from pcrouter.sim.cluster import simulate  # noqa: E402
from pcrouter.sim.engine import EngineConfig  # noqa: E402
from pcrouter.workloads import load_mooncake, multiturn, shared_prefix  # noqa: E402

OUT = ROOT / "results" / "sim"
KV = 2_022_400  # vLLM 0.34, Qwen2.5-7B, H200, --max-model-len 32768 (measured)
TRACES = {
    "conversation": ROOT / "data" / "mooncake_conversation_trace.jsonl",
    "toolagent": ROOT / "data" / "mooncake_toolagent_trace.jsonl",
}
FILTER = {"max_input_len": 30720, "max_output_len": 2048}
MAIN = ["round-robin", "least-request", "p2c", "chwbl", "legacy", "threshold", "affinity", "ttft"]
SLO = 2.0
_cache: dict[tuple[Any, ...], list[Any]] = {}


def workload(spec: dict[str, Any]) -> list[Any]:
    key = tuple(sorted((k, str(v)) for k, v in spec.items()))
    if key not in _cache:
        kind = spec["kind"]
        if kind in TRACES:
            _cache[key] = load_mooncake(
                TRACES[kind],
                time_scale=spec["scale"],
                start_s=spec.get("start_s", 0.0),
                duration_s=spec.get("duration_s"),
                **FILTER,
            )
        elif kind == "multiturn":
            _cache[key] = multiturn(
                spec["n"],
                session_rate=spec["rate"],
                zipf_s=spec["zipf"],
                num_system=spec.get("num_system", 8),
                system_len=spec.get("system_len", 2048),
                turns_mean=spec.get("turns", 4.0),
                user_len=spec.get("user_len", 400.0),
                reply_len=spec.get("reply_len", 300.0),
                seed=spec["seed"],
                max_input_len=FILTER["max_input_len"],
            )
        elif kind == "shared-prefix":
            _cache[key] = shared_prefix(
                spec["n"],
                rate=spec["rate"],
                zipf_s=spec["zipf"],
                num_prefixes=spec.get("num_prefixes", 64),
                prefix_len=spec.get("prefix_len", 4096),
                suffix_len=spec.get("suffix_len", 256.0),
                seed=spec["seed"],
            )
        else:
            raise ValueError(kind)
    return _cache[key]


def run(cell: dict[str, Any]) -> dict[str, Any]:
    """One simulation. ``cell`` = {workload: {...}, policy, params, workers, kv, ...}."""
    model = load_step_model("h200-qwen2.5-7b")
    reqs = workload(cell["workload"])
    w = cell.get("workers", 4)
    pol = make_policy(
        cell["policy"], w, seed=cell.get("seed", 0), model=model, **cell.get("params", {})
    )
    cfg = EngineConfig(num_blocks=cell.get("kv", KV) // 16)
    scale = cell["workload"].get("scale", 1.0)
    warm = 60.0 / scale if cell["workload"]["kind"] in TRACES else 30.0
    res = simulate(
        reqs,
        pol,
        cfg,
        model,
        chunk_tokens=cell.get("chunk_tokens", 512),
        capacity_factor=cell.get("capacity_factor", 1.0),
        feedback=cell.get("feedback", False),
        oracle=cell.get("oracle", False),
        slo_ttft=SLO,
        warmup_s=warm,
    )
    s = {k: v for k, v in res.summary.items() if not k.startswith("worker_")}
    s["wall_s"] = res.wall_s
    s["index_evictions"] = sum(e["index_evictions"] for e in res.engines)
    s["engine_evictions"] = sum(e["evictions"] for e in res.engines)
    return {**cell, "summary": s}


def grid(cells: list[dict[str, Any]], jobs: int) -> list[dict[str, Any]]:
    with ProcessPoolExecutor(jobs) as ex:
        return list(ex.map(run, cells, chunksize=1))


def save(name: str, rows: list[dict[str, Any]], meta: dict[str, Any]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{name}.json").write_text(
        json.dumps({"meta": meta, "rows": rows}, indent=1, default=float)
    )
    print(f"wrote {OUT / name}.json ({len(rows)} rows)", flush=True)


# -- experiments -----------------------------------------------------------------------------


def exp_load(jobs: int) -> None:
    """Every policy across load levels on both traces."""
    scales = {"conversation": [2, 3, 4, 4.5, 5, 5.5, 6, 7], "toolagent": [2, 3, 4, 4.5, 5, 5.5, 6]}
    cells = [
        {"workload": {"kind": t, "scale": s}, "policy": p}
        for t, ss in scales.items()
        for s in ss
        for p in [*MAIN, "random"]
    ]
    save("load", grid(cells, jobs), {"workers": 4, "kv": KV, "slo_ttft": SLO, **FILTER})


def exp_beta(jobs: int) -> None:
    """Sensitivity of the ttft policy to beta (how beta=8 was chosen)."""
    betas = [0, 1, 2, 4, 8, 16, 32, 64]
    cells = [
        {"workload": {"kind": t, "scale": s}, "policy": "ttft", "params": {"beta": b}}
        for t in TRACES
        for s in (3, 4, 5, 5.5, 6)
        for b in betas
    ]
    cells += [
        {"workload": {"kind": t, "scale": s}, "policy": "affinity"}
        for t in TRACES
        for s in (3, 4, 5, 5.5, 6)
    ]
    save("beta", grid(cells, jobs), {"betas": betas})


def exp_decode(jobs: int) -> None:
    """Sensitivity of the ttft policy to its decode-load weight (lambda)."""
    lams = [0.0, 0.125, 0.25, 0.5, 1.0]
    work = [{"kind": t, "scale": s} for t in TRACES for s in (4, 5, 5.5, 6)]
    work += [{**SYNTHETIC["chat"], "rate": r, "seed": sd} for r in (7.0, 10.0) for sd in (0, 1)]
    work += [{**SYNTHETIC["rag"], "rate": 60, "seed": sd} for sd in (0, 1)]
    cells = [
        {"workload": w, "policy": "ttft", "params": {"decode_weight": lam}}
        for w in work
        for lam in lams
    ]
    cells += [{"workload": w, "policy": p} for w in work for p in ("affinity", "threshold")]
    save("decode", grid(cells, jobs), {"lambdas": lams})


def _capacity(args: tuple[str, str, dict[str, Any]]) -> dict[str, Any]:
    """Largest time-scale at which TTFT p99 <= SLO (bisection, 0.05 resolution)."""
    trace, policy, params = args
    lo, hi = 1.0, 9.0
    trail = []
    while hi - lo > 0.05:
        mid = round((lo + hi) / 2, 3)
        r = run({"workload": {"kind": trace, "scale": mid}, "policy": policy, "params": params})
        ok = r["summary"]["ttft_p99"] <= SLO
        trail.append((mid, r["summary"]["ttft_p99"], ok))
        lo, hi = (mid, hi) if ok else (lo, mid)
    return {
        "trace": trace,
        "policy": policy,
        "params": params,
        "capacity_scale": lo,
        "trail": trail,
    }


def exp_capacity(jobs: int) -> None:
    """Load each policy sustains with TTFT p99 <= 2 s."""
    args = [(t, p, {}) for t in TRACES for p in MAIN]
    with ProcessPoolExecutor(jobs) as ex:
        rows = list(ex.map(_capacity, args, chunksize=1))
    save("capacity", rows, {"slo_ttft_p99": SLO})


def exp_kv(jobs: int) -> None:
    """KV-cache size per worker (smaller GPUs / bigger models leave less room)."""
    fracs = [0.125, 0.25, 0.5, 1.0, 2.0]
    pols = ["round-robin", "least-request", "chwbl", "affinity", "ttft"]
    cells = [
        {"workload": {"kind": t, "scale": 4}, "policy": p, "kv": int(KV * f), "kv_frac": f}
        for t in TRACES
        for f in fracs
        for p in pols
    ]
    save("kv", grid(cells, jobs), {"fracs": fracs, "scale": 4})


def exp_scale(jobs: int) -> None:
    """Cluster size, per-worker load fixed (trace replayed W times faster)."""
    pols = ["round-robin", "least-request", "chwbl", "affinity", "ttft"]
    cells = [
        {"workload": {"kind": t, "scale": base * w / 4}, "policy": p, "workers": w}
        for t, base in (("conversation", 4.5), ("toolagent", 4.5))
        for w in (2, 4, 8, 16, 32)
        for p in pols
    ]
    save("scale", grid(cells, jobs), {"per_4_workers_scale": 4.5})


SYNTHETIC = {
    # RAG / few-shot: 1024 documents of 8k tokens (8.4M tokens: fits in the cluster's
    # combined cache, not in one worker's), Zipf popularity, short unique questions.
    "rag": {
        "kind": "shared-prefix",
        "n": 12000,
        "zipf": 1.1,
        "num_prefixes": 1024,
        "prefix_len": 8192,
        "suffix_len": 256.0,
    },
    # One hot prefix (Zipf 2: the top prefix takes ~60% of requests) with long unique parts,
    # so the hot prefix's traffic alone can saturate one worker.
    "hot": {
        "kind": "shared-prefix",
        "n": 12000,
        "zipf": 2.0,
        "num_prefixes": 64,
        "prefix_len": 4096,
        "suffix_len": 2048.0,
    },
    # Chat: 16 system prompts of 4k tokens, sessions of ~6 turns that grow ~900 tokens each.
    "chat": {
        "kind": "multiturn",
        "n": 4000,
        "zipf": 1.0,
        "num_system": 16,
        "system_len": 4096,
        "turns": 6.0,
        "user_len": 600.0,
        "reply_len": 300.0,
    },
}
SYNTHETIC_RATES = {"rag": (60, 90, 120), "hot": (40, 50, 60), "chat": (7.0, 10.0, 13.0)}


def exp_synthetic(jobs: int) -> None:
    """Workloads with known structure, three load levels each, three seeds."""
    cells = []
    for name, spec in SYNTHETIC.items():
        for rate in SYNTHETIC_RATES[name]:
            for seed in (0, 1, 2):
                w = {**spec, "rate": rate, "seed": seed}
                cells += [{"workload": w, "policy": p, "seed": seed, "name": name} for p in MAIN]
    save("synthetic", grid(cells, jobs), {"seeds": 3, "specs": SYNTHETIC})


def exp_index(jobs: int) -> None:
    """How good must the router's cache model be? Oracle vs approximate index, chunk size,
    capacity estimate and cached-token feedback."""
    base = {"workload": {"kind": "conversation", "scale": 5}}
    cells = []
    for p in ("affinity", "ttft"):
        cells.append({**base, "policy": p, "variant": "approx"})
        cells.append({**base, "policy": p, "oracle": True, "variant": "oracle"})
        for ct in (128, 256, 1024, 2048):
            cells.append({**base, "policy": p, "chunk_tokens": ct, "variant": f"chunk{ct}"})
        for cf in (0.25, 0.5, 2.0, 4.0):
            cells.append({**base, "policy": p, "capacity_factor": cf, "variant": f"cap{cf}"})
            cells.append(
                {
                    **base,
                    "policy": p,
                    "capacity_factor": cf,
                    "feedback": True,
                    "variant": f"cap{cf}+fb",
                }
            )
    save("index", grid(cells, jobs), {})


def exp_fidelity(jobs: int) -> None:
    """The exact windows replayed on the H200s (first 1800 s of trace time), with the proxy's
    default cached-token feedback on."""
    cells = [
        {"workload": {"kind": t, "scale": s, "duration_s": 1800.0}, "policy": p, "feedback": True}
        for t in TRACES
        for s in (4, 5)
        for p in [
            "round-robin",
            "least-request",
            "chwbl",
            "legacy",
            "threshold",
            "affinity",
            "ttft",
        ]
    ]
    # H200 run 1 used the ttft policy before its decode term existed (lambda = 0).
    cells += [
        {
            "workload": {"kind": t, "scale": s, "duration_s": 1800.0},
            "policy": "ttft",
            "params": {"decode_weight": 0.0},
            "feedback": True,
            "variant": "lambda0",
        }
        for t in TRACES
        for s in (4, 5)
    ]
    save("fidelity", grid(cells, jobs), {"duration_s": 1800})


def exp_workloads(jobs: int) -> None:
    """Size and reuse ceiling of the filtered traces (infinite single cache)."""
    from pcrouter.workloads import summarize_workload

    rows = [
        {"trace": t, **summarize_workload(load_mooncake(p, **FILTER))} for t, p in TRACES.items()
    ]
    save("workloads", rows, FILTER)


EXPERIMENTS: dict[str, Callable[[int], None]] = {
    "workloads": exp_workloads,
    "load": exp_load,
    "beta": exp_beta,
    "decode": exp_decode,
    "capacity": exp_capacity,
    "kv": exp_kv,
    "scale": exp_scale,
    "synthetic": exp_synthetic,
    "index": exp_index,
    "fidelity": exp_fidelity,
}


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("names", nargs="+", choices=[*EXPERIMENTS, "all"])
    p.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    a = p.parse_args()
    names = list(EXPERIMENTS) if "all" in a.names else a.names
    assert set(POLICIES) >= set(MAIN)
    for n in names:
        t = time.time()
        EXPERIMENTS[n](a.jobs)
        print(f"{n}: {time.time() - t:.0f}s", flush=True)


if __name__ == "__main__":
    main()
