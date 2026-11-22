"""Run a policy x workload matrix through a live pcrouter in front of real (or fake) engines.

For every (workload, policy) cell: switch the router's policy (``/admin/policy``), reset the
router and every engine's prefix cache (``/admin/reset``), snapshot the engines' prefix-cache
counters, replay the workload open-loop, snapshot again, and write

    <out>/<workload>/<policy>/records.jsonl   per-request records
    <out>/<workload>/<policy>/summary.json    latency/reuse summary + engine-side counters

Usage (see bench/explorer-driver.sbatch):
    python bench/run_matrix.py --router http://127.0.0.1:8080 \
        --engines http://127.0.0.1:8100,http://127.0.0.1:8101 \
        --trace data/mooncake_conversation_trace.jsonl --scales 4,5 --duration-s 1800 \
        --policies round-robin,affinity,ttft --out results/h200
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import time
from pathlib import Path
from typing import Any

import aiohttp

from pcrouter.metrics import write_records
from pcrouter.server.replay import replay
from pcrouter.workloads import load_mooncake

_METRIC = re.compile(
    r"^(vllm:prefix_cache_(?:hits|queries)_total)(?:\{[^}]*\})?\s+([0-9.eE+]+)", re.MULTILINE
)


async def engine_counters(session: aiohttp.ClientSession, url: str) -> dict[str, float]:
    async with session.get(url + "/metrics") as r:
        text = await r.text()
    out: dict[str, float] = {}
    for name, val in _METRIC.findall(text):
        out[name] = out.get(name, 0.0) + float(val)
    return out


async def admin(
    session: aiohttp.ClientSession, url: str, path: str, body: dict[str, Any] | None = None
) -> dict[str, Any]:
    for _ in range(120):  # wait for in-flight stragglers to drain
        async with session.post(url + path, json=body or {}) as r:
            if r.status == 200:
                return dict(await r.json())
            if r.status != 409:
                raise RuntimeError(f"{path}: HTTP {r.status} {await r.text()}")
        await asyncio.sleep(1)
    raise RuntimeError(f"{path}: router never drained")


async def run_cell(
    a: argparse.Namespace, trace: str, scale: float, policy: str, params: dict[str, Any], out: Path
) -> dict[str, Any]:
    reqs = load_mooncake(
        trace,
        time_scale=scale,
        start_s=a.start_s,
        duration_s=a.duration_s,
        max_input_len=a.max_input_len,
        max_output_len=a.max_output_len,
    )
    engines = [u for u in a.engines.split(",") if u]
    async with aiohttp.ClientSession() as s:
        await admin(s, a.router, "/admin/policy", {"policy": policy, "params": params})
        reset = await admin(s, a.router, "/admin/reset")
        await asyncio.sleep(2)
        before = [await engine_counters(s, e) for e in engines]
        t0 = time.time()
        res = await replay(
            a.router,
            reqs,
            model=a.model,
            slo_ttft=a.slo_ttft,
            warmup_s=60.0 / scale,
            num_workers=len(engines),
        )
        wall = time.time() - t0
        after = [await engine_counters(s, e) for e in engines]
        async with s.get(a.router + "/debug/state") as r:
            state = await r.json()
    hits = sum(
        x.get("vllm:prefix_cache_hits_total", 0) - y.get("vllm:prefix_cache_hits_total", 0)
        for x, y in zip(after, before)
    )
    queries = sum(
        x.get("vllm:prefix_cache_queries_total", 0) - y.get("vllm:prefix_cache_queries_total", 0)
        for x, y in zip(after, before)
    )
    s_ = res.summary
    s_.update(
        {
            "policy": policy,
            "policy_params": params,
            "trace": Path(trace).name,
            "time_scale": scale,
            "start_s": a.start_s,
            "duration_s": a.duration_s,
            "max_input_len": a.max_input_len,
            "max_output_len": a.max_output_len,
            "engine_hit_ratio": hits / queries if queries else None,
            "engine_prefix_queries": queries,
            "wall_s": wall,
            "reset": reset,
            "router_state": state,
            "workers_seen": res.workers,
        }
    )
    _write(out, res.records, s_)
    return s_


def _write(out: Path, records: list[Any], summary: dict[str, Any]) -> None:
    out.mkdir(parents=True, exist_ok=True)
    write_records(out / "records.jsonl", records)
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=float))


async def main_async(a: argparse.Namespace) -> None:
    pols = [p for p in a.policies.split(",") if p]
    params = json.loads(a.params) if a.params else {}
    for trace in a.trace:
        for scale in [float(x) for x in a.scales.split(",")]:
            tag = f"{Path(trace).stem.replace('mooncake_', '').replace('_trace', '')}_x{scale:g}"
            for pol in pols:
                out = Path(a.out) / tag / pol
                if (out / "summary.json").exists() and not a.force:
                    print(f"skip {tag}/{pol} (done)", flush=True)
                    continue
                t = time.time()
                s = await run_cell(a, trace, scale, pol, params.get(pol, {}), out)
                print(
                    f"{tag:22s} {pol:14s} ttft p50 {s.get('ttft_p50', -1):7.3f} "
                    f"p99 {s.get('ttft_p99', -1):7.2f} | tpot p50 "
                    f"{s.get('tpot_p50', 0) * 1e3:6.1f} ms | hit {s.get('hit_ratio', 0):.3f} "
                    f"engine {s.get('engine_hit_ratio') or 0:.3f} | slo "
                    f"{s.get('slo_attainment', 0):.3f} | failed {s['failed']} | lag p99 "
                    f"{s['send_lag_p99_s'] * 1e3:.1f} ms | {time.time() - t:.0f}s",
                    flush=True,
                )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--router", required=True)
    p.add_argument("--engines", required=True, help="comma-separated engine base URLs")
    p.add_argument("--trace", action="append", required=True)
    p.add_argument("--scales", default="4")
    p.add_argument("--start-s", type=float, default=0.0)
    p.add_argument("--duration-s", type=float, default=1800.0)
    p.add_argument("--max-input-len", type=int, default=30720)
    p.add_argument("--max-output-len", type=int, default=2048)
    p.add_argument("--policies", default="round-robin,least-request,chwbl,threshold,affinity,ttft")
    p.add_argument("--params", default="", help='JSON {"policy": {params}}')
    p.add_argument("--model", default="m")
    p.add_argument("--slo-ttft", type=float, default=2.0)
    p.add_argument("--out", required=True)
    p.add_argument("--force", action="store_true", help="rerun cells that already have results")
    a = p.parse_args()
    try:
        import uvloop  # type: ignore[import-not-found]

        uvloop.install()
    except ImportError:
        pass
    asyncio.run(main_async(a))


if __name__ == "__main__":
    main()
