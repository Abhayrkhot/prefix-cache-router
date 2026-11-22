"""``pcrouter`` command line.

pcrouter stats     --trace data/mooncake_conversation_trace.jsonl
pcrouter simulate  --trace ... --policy ttft --workers 4 --time-scale 4
pcrouter compare   --trace ... --workers 4 --time-scale 4
pcrouter serve     --worker http://gpu0:8000 --worker http://gpu1:8000 --policy ttft
pcrouter fake-worker --port 8001 --kv-tokens 200000
pcrouter replay    --url http://127.0.0.1:8080 --trace ... --time-scale 4 --out run.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from typing import Any

from pcrouter.policies import POLICIES

H200_QWEN7B_KV_TOKENS = 2_022_400  # vLLM 0.34, Qwen2.5-7B BF16, one H200, max-model-len 32k


def _param(kv: str) -> tuple[str, Any]:
    if "=" not in kv:
        raise argparse.ArgumentTypeError(f"expected key=value, got {kv!r}")
    k, v = kv.split("=", 1)
    try:
        return k, json.loads(v)
    except json.JSONDecodeError:
        return k, v


def _workload_args(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("workload")
    src = g.add_mutually_exclusive_group(required=True)
    src.add_argument("--trace", help="Mooncake *_trace.jsonl")
    src.add_argument("--synthetic", choices=["multiturn", "shared-prefix"])
    g.add_argument("--time-scale", type=float, default=1.0, help="replay N times faster")
    g.add_argument("--start-s", type=float, default=0.0)
    g.add_argument("--duration-s", type=float, default=None)
    g.add_argument("--max-input-len", type=int, default=32768)
    g.add_argument("--n", type=int, default=2000, help="synthetic: sessions or requests")
    g.add_argument("--rate", type=float, default=2.0, help="synthetic: arrivals per second")
    g.add_argument("--zipf", type=float, default=1.1, help="synthetic: prefix popularity skew")
    g.add_argument("--seed", type=int, default=0)


def _cluster_args(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("cluster")
    g.add_argument("--workers", type=int, default=4)
    g.add_argument(
        "--kv-tokens",
        type=int,
        default=H200_QWEN7B_KV_TOKENS,
        help="KV-cache capacity per worker in tokens",
    )
    g.add_argument("--max-num-batched-tokens", type=int, default=8192)
    g.add_argument("--max-num-seqs", type=int, default=1024)
    g.add_argument(
        "--step-model",
        default="h200-qwen2.5-7b",
        help="packaged step model name or batchsim-format JSON",
    )
    g.add_argument("--chunk-tokens", type=int, default=512)
    g.add_argument("--capacity-factor", type=float, default=1.0)
    g.add_argument("--oracle", action="store_true", help="route on true engine cache state")
    g.add_argument("--slo-ttft", type=float, default=2.0)
    g.add_argument(
        "--warmup-s",
        type=float,
        default=None,
        help="exclude early arrivals (default: 60 s of trace time)",
    )


def _load_workload(a: argparse.Namespace) -> list[Any]:
    from pcrouter.workloads import load_mooncake, multiturn, shared_prefix

    if a.trace:
        return load_mooncake(
            a.trace,
            time_scale=a.time_scale,
            start_s=a.start_s,
            duration_s=a.duration_s,
            max_input_len=a.max_input_len,
        )
    if a.synthetic == "multiturn":
        return multiturn(
            a.n,
            session_rate=a.rate * a.time_scale,
            zipf_s=a.zipf,
            seed=a.seed,
            max_input_len=a.max_input_len,
        )
    return shared_prefix(a.n, rate=a.rate * a.time_scale, zipf_s=a.zipf, seed=a.seed)


def _run_one(
    a: argparse.Namespace, policy: str, params: dict[str, Any], reqs: list[Any]
) -> dict[str, Any]:
    from pcrouter.costmodel import load_step_model
    from pcrouter.policies import make_policy
    from pcrouter.sim.cluster import simulate
    from pcrouter.sim.engine import EngineConfig

    model = load_step_model(a.step_model)
    pol = make_policy(policy, a.workers, seed=a.seed, model=model, **params)
    cfg = EngineConfig(
        num_blocks=a.kv_tokens // 16,
        max_num_batched_tokens=a.max_num_batched_tokens,
        max_num_seqs=a.max_num_seqs,
    )
    warm = a.warmup_s if a.warmup_s is not None else (60.0 / a.time_scale if a.trace else 0.0)
    res = simulate(
        reqs,
        pol,
        cfg,
        model,
        chunk_tokens=a.chunk_tokens,
        capacity_factor=a.capacity_factor,
        oracle=a.oracle,
        slo_ttft=a.slo_ttft,
        warmup_s=warm,
    )
    s = res.summary
    s["wall_s"] = res.wall_s
    return s


def _fmt_row(name: str, s: dict[str, Any]) -> str:
    return (
        f"{name:16s} {s['ttft_p50']:8.3f} {s['ttft_p99']:8.2f} {s['tpot_p50'] * 1e3:8.1f} "
        f"{s['tpot_p99'] * 1e3:8.1f} {s['e2e_p99']:8.1f} {s.get('hit_ratio', 0):7.3f} "
        f"{s['slo_attainment']:7.3f}"
    )


HEADER = (
    f"{'policy':16s} {'TTFT p50':>8s} {'TTFT p99':>8s} {'TPOT p50':>8s} {'TPOT p99':>8s} "
    f"{'E2E p99':>8s} {'hit':>7s} {'SLO':>7s}\n"
    f"{'':16s} {'(s)':>8s} {'(s)':>8s} {'(ms)':>8s} {'(ms)':>8s} {'(s)':>8s}"
)


def cmd_stats(a: argparse.Namespace) -> int:
    from pcrouter.workloads import summarize_workload

    print(json.dumps(summarize_workload(_load_workload(a)), indent=2))
    return 0


def cmd_simulate(a: argparse.Namespace) -> int:
    s = _run_one(a, a.policy, dict(a.param), _load_workload(a))
    if a.json:
        print(json.dumps(s, indent=2, default=float))
    else:
        print(HEADER)
        print(_fmt_row(a.policy, s))
    return 0


def cmd_compare(a: argparse.Namespace) -> int:
    reqs = _load_workload(a)
    names = a.policies.split(",") if a.policies else list(POLICIES)
    print(HEADER)
    out = {}
    for name in names:
        s = _run_one(a, name, {}, reqs)
        out[name] = s
        print(_fmt_row(name, s), flush=True)
    if a.out:
        with open(a.out, "w") as f:
            json.dump(out, f, indent=1, default=float)
    return 0


def cmd_serve(a: argparse.Namespace) -> int:
    from pcrouter.server.app import serve_main

    return serve_main(a)


def cmd_fake_worker(a: argparse.Namespace) -> int:
    from pcrouter.server.fakeworker import fake_worker_main

    return fake_worker_main(a)


def cmd_replay(a: argparse.Namespace) -> int:
    from pcrouter.server.replay import replay_main

    return replay_main(a)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="pcrouter", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("stats", help="workload size, rate and ideal hit ratio")
    _workload_args(s)
    s.set_defaults(func=cmd_stats)

    s = sub.add_parser("simulate", help="simulate one policy")
    _workload_args(s)
    _cluster_args(s)
    s.add_argument("--policy", default="ttft", choices=sorted(POLICIES))
    s.add_argument(
        "--param",
        type=_param,
        action="append",
        default=[],
        help="policy parameter key=value (repeatable)",
    )
    s.add_argument("--json", action="store_true")
    s.set_defaults(func=cmd_simulate)

    s = sub.add_parser("compare", help="simulate several policies on one workload")
    _workload_args(s)
    _cluster_args(s)
    s.add_argument("--policies", default="", help="comma-separated (default: all)")
    s.add_argument("--out", default=None, help="write summaries as JSON")
    s.set_defaults(func=cmd_compare)

    s = sub.add_parser("serve", help="run the routing proxy")
    from pcrouter.server.config import add_serve_args

    add_serve_args(s)
    s.set_defaults(func=cmd_serve)

    s = sub.add_parser("fake-worker", help="a vLLM stand-in with a simulated prefix cache")
    from pcrouter.server.fakeworker import add_fake_worker_args

    add_fake_worker_args(s)
    s.set_defaults(func=cmd_fake_worker)

    s = sub.add_parser("replay", help="replay a trace against an OpenAI-compatible endpoint")
    from pcrouter.server.replay import add_replay_args

    add_replay_args(s)
    s.set_defaults(func=cmd_replay)
    return p


def main(argv: Sequence[str] | None = None) -> int:
    a = build_parser().parse_args(argv)
    return int(a.func(a))


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
