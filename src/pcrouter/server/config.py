"""Configuration of the routing proxy (CLI flags; every flag also reads ``PCROUTER_*`` env)."""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass, field

from pcrouter.policies import POLICIES

__all__ = ["ServeConfig", "add_serve_args", "config_from_args"]


@dataclass
class ServeConfig:
    workers: list[str]
    host: str = "0.0.0.0"
    port: int = 8080
    policy: str = "ttft"
    policy_params: dict[str, object] = field(default_factory=dict)
    chunk_tokens: int = 512
    kv_tokens: int | None = None  # None: discover from each worker's vLLM /metrics
    kv_tokens_fallback: int = 1_000_000
    capacity_factor: float = 1.0
    feedback: bool = True
    step_model: str = "h200-qwen2.5-7b"
    budget: int = 8192
    bytes_per_token: float = 4.0
    tokenizer: str | None = None
    health_interval_s: float = 5.0
    health_timeout_s: float = 2.0
    connect_timeout_s: float = 5.0
    keepalive_s: float = 4.0  # below uvicorn's 5 s default, see ProxyApp.on_startup
    reset_after_s: float = 30.0  # outage after which a worker's cache is assumed lost
    max_retries: int = 1
    max_inflight: int = 0  # 0 = unlimited
    admin_token: str | None = None  # None: admin endpoints accept loopback clients only
    seed: int = 0


def _env(name: str, default: str | None) -> str | None:
    return os.environ.get(f"PCROUTER_{name}", default)


def add_serve_args(p: argparse.ArgumentParser) -> None:
    w = _env("WORKERS", "")
    p.add_argument(
        "--worker",
        action="append",
        default=[u for u in (w or "").split(",") if u],
        help="worker base URL, e.g. http://10.0.0.5:8000 (repeatable)",
    )
    p.add_argument("--host", default=_env("HOST", "0.0.0.0"))
    p.add_argument("--port", type=int, default=int(_env("PORT", "8080") or 8080))
    p.add_argument("--policy", default=_env("POLICY", "ttft"), choices=sorted(POLICIES))
    p.add_argument(
        "--policy-param",
        action="append",
        default=[],
        help="policy parameter key=value (repeatable)",
    )
    p.add_argument("--chunk-tokens", type=int, default=int(_env("CHUNK_TOKENS", "512") or 512))
    p.add_argument(
        "--kv-tokens",
        type=int,
        default=None,
        help="KV capacity per worker in tokens (default: read vLLM /metrics)",
    )
    p.add_argument("--kv-tokens-fallback", type=int, default=1_000_000)
    p.add_argument("--capacity-factor", type=float, default=1.0)
    p.add_argument(
        "--no-feedback",
        action="store_true",
        help="ignore usage.prompt_tokens_details.cached_tokens",
    )
    p.add_argument("--step-model", default=_env("STEP_MODEL", "h200-qwen2.5-7b"))
    p.add_argument("--budget", type=int, default=8192, help="engines' max_num_batched_tokens")
    p.add_argument("--bytes-per-token", type=float, default=4.0)
    p.add_argument(
        "--tokenizer",
        default=_env("TOKENIZER", None),
        help="HF tokenizer name for exact text hashing (needs `transformers`)",
    )
    p.add_argument("--health-interval", type=float, default=5.0)
    p.add_argument("--max-retries", type=int, default=1)
    p.add_argument("--max-inflight", type=int, default=0, help="429 above this (0 = off)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--admin-token",
        default=_env("ADMIN_TOKEN", None),
        help="bearer token for /admin/* (default: loopback clients only)",
    )


def config_from_args(a: argparse.Namespace) -> ServeConfig:
    import json

    params: dict[str, object] = {}
    for kv in a.policy_param:
        k, _, v = kv.partition("=")
        try:
            params[k] = json.loads(v)
        except json.JSONDecodeError:
            params[k] = v
    if not a.worker:
        raise SystemExit("at least one --worker (or PCROUTER_WORKERS) is required")
    return ServeConfig(
        workers=[u.rstrip("/") for u in a.worker],
        host=a.host,
        port=a.port,
        policy=a.policy,
        policy_params=params,
        chunk_tokens=a.chunk_tokens,
        kv_tokens=a.kv_tokens,
        kv_tokens_fallback=a.kv_tokens_fallback,
        capacity_factor=a.capacity_factor,
        feedback=not a.no_feedback,
        step_model=a.step_model,
        budget=a.budget,
        bytes_per_token=a.bytes_per_token,
        tokenizer=a.tokenizer,
        health_interval_s=a.health_interval,
        max_retries=a.max_retries,
        max_inflight=a.max_inflight,
        seed=a.seed,
        admin_token=a.admin_token,
    )
