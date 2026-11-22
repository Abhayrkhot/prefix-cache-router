"""Open-loop replay of a workload against an OpenAI-compatible endpoint (router or engine).

Each request is sent at its arrival time as a token-id prompt to ``/v1/completions`` with
``max_tokens = min_tokens = output_len`` and ``ignore_eos`` (so output lengths match the trace),
streaming, with ``stream_options.include_usage`` to read the engine's cached-token count.
Prompt tokens are generated deterministically per prompt segment, so requests that share a
segment in the workload share exactly those tokens and the engine's prefix cache sees the
workload's real sharing structure.

Records are the same :class:`~pcrouter.metrics.Record` the simulator emits, so simulated and
measured runs are summarized by the same code.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import aiohttp
import numpy as np
import numpy.typing as npt

from pcrouter.metrics import Record, summarize, write_records
from pcrouter.prompt import BLOCK, Prompt
from pcrouter.workloads import Request, load_mooncake, multiturn, shared_prefix

__all__ = ["ReplayResult", "add_replay_args", "prompt_tokens", "replay", "replay_main"]

VOCAB = (1000, 150_000)  # inside Qwen2.5's vocabulary, below its special tokens
_TAIL = 1 << 62  # seed tag for per-request tails (stream ids are smaller)


def prompt_tokens(prompt: Prompt, rid: int, seed: int = 0) -> list[int]:
    """Token ids for ``prompt``: segment ``(stream, start, n)`` always yields the same
    ``n*16`` tokens (and a longer segment of the same stream extends a shorter one)."""
    if prompt.segments is None:
        raise ValueError("replay needs segment prompts")
    out: list[npt.NDArray[np.int64]] = []
    have = 0
    for stream, start, n in prompt.segments:
        if have >= prompt.num_tokens:
            break
        rng = np.random.default_rng((seed, stream, start))
        out.append(rng.integers(VOCAB[0], VOCAB[1], size=n * BLOCK, dtype=np.int64))
        have += n * BLOCK
    toks = np.concatenate(out) if out else np.empty(0, dtype=np.int64)
    if len(toks) < prompt.num_tokens:  # partial last block: unique to this request
        tail = np.random.default_rng((seed, _TAIL, rid)).integers(
            VOCAB[0], VOCAB[1], size=prompt.num_tokens - len(toks), dtype=np.int64
        )
        toks = np.concatenate([toks, tail])
    return [int(t) for t in toks[: prompt.num_tokens]]


@dataclass
class ReplayResult:
    records: list[Record]
    summary: dict[str, Any]
    send_lag_p99_s: float
    workers: dict[int, int] = field(default_factory=dict)


async def _one(
    session: aiohttp.ClientSession,
    url: str,
    model: str,
    q: Request,
    seed: int,
    t0: float,
    lags: list[float],
    timeout_s: float,
) -> Record:
    body = {
        "model": model,
        "prompt": prompt_tokens(q.prompt, q.id, seed),
        "max_tokens": q.output_len,
        "min_tokens": q.output_len,
        "ignore_eos": True,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    data = json.dumps(body).encode()
    send = time.perf_counter()
    lags.append(send - (t0 + q.arrival))
    first = -1.0
    cached = -1
    worker = -1
    ok = False
    try:
        async with session.post(
            url + "/v1/completions",
            data=data,
            headers={"Content-Type": "application/json"},
            timeout=aiohttp.ClientTimeout(total=timeout_s),
        ) as r:
            worker = int(r.headers.get("X-PCRouter-Worker", "-1"))
            if r.status == 200:
                buf = b""
                async for chunk in r.content.iter_any():
                    buf += chunk
                    while b"\n\n" in buf:
                        ev, buf = buf.split(b"\n\n", 1)
                        if not ev.startswith(b"data:"):
                            continue
                        payload = ev[5:].strip()
                        if payload == b"[DONE]":
                            ok = True
                            continue
                        if first < 0:
                            first = time.perf_counter()
                        if b'"usage"' in payload:
                            usage = json.loads(payload).get("usage") or {}
                            c = (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
                            if isinstance(c, int):
                                cached = c
            else:
                await r.read()
    except (aiohttp.ClientError, asyncio.TimeoutError):
        ok = False
    end = time.perf_counter()
    if not ok or first < 0:
        return Record(
            q.id, worker, q.arrival, -1.0, -1.0, q.input_len, q.output_len, cached, ok=False
        )
    # Latencies from the actual send time (send lag is reported separately).
    return Record(
        q.id, worker, q.arrival, first - send, end - send, q.input_len, q.output_len, cached
    )


async def replay(
    url: str,
    requests: Sequence[Request],
    *,
    model: str = "m",
    seed: int = 0,
    timeout_s: float = 600.0,
    slo_ttft: float = 2.0,
    warmup_s: float = 0.0,
    num_workers: int | None = None,
) -> ReplayResult:
    conn = aiohttp.TCPConnector(limit=0, force_close=False)
    lags: list[float] = []
    async with aiohttp.ClientSession(connector=conn) as session:
        t0 = time.perf_counter() + 0.5
        tasks = []
        for q in sorted(requests, key=lambda r: r.arrival):
            delay = t0 + q.arrival - time.perf_counter()
            if delay > 0:
                await asyncio.sleep(delay)
            tasks.append(
                asyncio.create_task(_one(session, url, model, q, seed, t0, lags, timeout_s))
            )
        records = list(await asyncio.gather(*tasks))
    records.sort(key=lambda r: r.rid)
    s = summarize(records, slo_ttft=slo_ttft, num_workers=num_workers, warmup_s=warmup_s)
    lag = float(np.percentile(lags, 99)) if lags else 0.0
    s["send_lag_p99_s"] = lag
    counts: dict[int, int] = {}
    for r in records:
        counts[r.worker] = counts.get(r.worker, 0) + 1
    return ReplayResult(records, s, lag, counts)


def add_replay_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--url", required=True, help="router or engine base URL")
    p.add_argument("--model", default="m", help="served model name")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--trace", help="Mooncake *_trace.jsonl")
    src.add_argument("--synthetic", choices=["multiturn", "shared-prefix"])
    p.add_argument("--time-scale", type=float, default=1.0)
    p.add_argument("--start-s", type=float, default=0.0)
    p.add_argument("--duration-s", type=float, default=None)
    p.add_argument("--max-input-len", type=int, default=32768)
    p.add_argument("--max-output-len", type=int, default=None)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--n", type=int, default=500)
    p.add_argument("--rate", type=float, default=2.0)
    p.add_argument(
        "--think-s",
        type=float,
        default=20.0,
        help="synthetic multiturn: mean gap between a session's turns",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--slo-ttft", type=float, default=2.0)
    p.add_argument("--warmup-s", type=float, default=0.0)
    p.add_argument("--workers", type=int, default=None, help="for per-worker balance stats")
    p.add_argument("--out", default=None, help="records JSONL (summary goes to <out>.json)")
    p.add_argument("--label", default="", help="free-form tag stored in the summary")


def replay_main(a: argparse.Namespace) -> int:  # pragma: no cover - process entry point
    if a.trace:
        reqs = load_mooncake(
            a.trace,
            time_scale=a.time_scale,
            start_s=a.start_s,
            duration_s=a.duration_s,
            max_input_len=a.max_input_len,
            max_output_len=a.max_output_len,
            limit=a.limit,
        )
    elif a.synthetic == "multiturn":
        reqs = multiturn(a.n, session_rate=a.rate * a.time_scale, think_s=a.think_s, seed=a.seed)
    else:
        reqs = shared_prefix(a.n, rate=a.rate * a.time_scale, seed=a.seed)
    try:
        import uvloop  # type: ignore[import-not-found]

        uvloop.install()
    except ImportError:
        pass
    res = asyncio.run(
        replay(
            a.url,
            reqs,
            model=a.model,
            seed=a.seed,
            slo_ttft=a.slo_ttft,
            warmup_s=a.warmup_s,
            num_workers=a.workers,
        )
    )
    res.summary["label"] = a.label
    res.summary["workers_seen"] = res.workers
    print(
        json.dumps(
            {k: v for k, v in res.summary.items() if not k.endswith("_ci")}, indent=1, default=float
        )
    )
    if a.out:
        write_records(a.out, res.records)
        with open(a.out + ".json", "w") as f:
            json.dump(res.summary, f, indent=1, default=float)
    return 0
