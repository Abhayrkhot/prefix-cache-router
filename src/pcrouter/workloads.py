"""Workloads: real Mooncake traces and parametric synthetic generators.

Mooncake / Kimi traces (FAST'25) give, per request, the arrival time, input and output
lengths, and a hash for every 512-token block of the prompt. The hashes are prefix-chained
(each hash occurs at exactly one position after exactly one parent), so hash ``h`` becomes
stream ``h`` and its 32 engine blocks get keys ``h<<20 | 0..31``: two requests share a block
key iff they share the whole prefix up to it, as in the trace.

Synthetic workloads build prompts from streams the same way:

* :func:`multiturn` — chat sessions: one of ``num_system`` system prompts (Zipf popularity)
  followed by a conversation that grows every turn.
* :func:`shared_prefix` — RAG / few-shot style: a popular long document or template (Zipf)
  followed by a unique question. A large ``zipf_s`` gives a single hot prefix.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt

from pcrouter.prompt import BLOCK, Prompt

__all__ = ["Request", "load_mooncake", "multiturn", "shared_prefix", "summarize_workload"]

MOONCAKE_BLOCK = 512


@dataclass(slots=True)
class Request:
    id: int
    arrival: float
    prompt: Prompt
    output_len: int
    tag: int = 0  # session / prefix group, for analysis

    @property
    def input_len(self) -> int:
        return self.prompt.num_tokens


def _rows(path: str | Path) -> Iterator[dict[str, object]]:
    with open(path) as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def load_mooncake(
    path: str | Path,
    *,
    time_scale: float = 1.0,
    start_s: float = 0.0,
    duration_s: float | None = None,
    max_input_len: int | None = None,
    max_output_len: int | None = None,
    limit: int | None = None,
) -> list[Request]:
    """Requests arriving in ``[start_s, start_s + duration_s)`` of trace time, replayed
    ``time_scale`` times faster. Requests longer than ``max_input_len`` are skipped, never
    truncated (truncating would change the sharing structure)."""
    if time_scale <= 0:
        raise ValueError("time_scale must be positive")
    per = MOONCAKE_BLOCK // BLOCK
    out: list[Request] = []
    for r in _rows(path):
        ts = float(r["timestamp"]) / 1000.0  # type: ignore[arg-type]
        if ts < start_s:
            continue
        if duration_s is not None and ts >= start_s + duration_s:
            break
        inp = int(r["input_length"])  # type: ignore[call-overload]
        if inp < 1 or (max_input_len is not None and inp > max_input_len):
            continue
        outl = max(1, int(r["output_length"]))  # type: ignore[call-overload]
        if max_output_len is not None:
            outl = min(outl, max_output_len)
        hashes = [int(h) for h in r["hash_ids"]]  # type: ignore[attr-defined]
        segs = [(h, 0, per) for h in hashes]
        out.append(
            Request(
                id=len(out),
                arrival=(ts - start_s) / time_scale,
                prompt=Prompt(inp, segments=segs),
                output_len=outl,
                tag=hashes[0] if hashes else -1,
            )
        )
        if limit is not None and len(out) >= limit:
            break
    out.sort(key=lambda q: (q.arrival, q.id))
    for i, q in enumerate(out):
        q.id = i
    return out


def _lognormal(rng: np.random.Generator, mean: float, cv: float, lo: int, hi: int) -> int:
    sigma2 = np.log1p(cv * cv)
    mu = np.log(mean) - sigma2 / 2
    return int(np.clip(float(rng.lognormal(mu, np.sqrt(sigma2))), lo, hi))


def _zipf_weights(n: int, s: float) -> npt.NDArray[np.float64]:
    w = 1.0 / np.arange(1, n + 1, dtype=float) ** s
    return np.asarray(w / w.sum())


def _segs(parts: list[tuple[int, int]]) -> list[tuple[int, int, int]]:
    # parts: [(stream, full_blocks_from_that_stream)] laid end to end; block indices in a
    # stream continue from where the previous part left off (absolute position in prompt).
    segs = []
    pos = 0
    for sid, n in parts:
        if n > 0:
            segs.append((sid, pos, n))
        pos += n
    return segs


def multiturn(
    num_sessions: int,
    *,
    session_rate: float = 1.0,
    turns_mean: float = 4.0,
    num_system: int = 8,
    zipf_s: float = 1.0,
    system_len: int = 2048,
    user_len: float = 400.0,
    reply_len: float = 300.0,
    think_s: float = 20.0,
    max_input_len: int = 32768,
    seed: int = 0,
) -> list[Request]:
    """Open-loop chat sessions (Poisson session starts, exponential gaps between turns)."""
    rng = np.random.default_rng(seed)
    weights = _zipf_weights(num_system, zipf_s)
    sys_blocks = system_len // BLOCK
    reqs: list[Request] = []
    t = 0.0
    next_stream = num_system  # 0..num_system-1 are the system prompts
    for sess in range(num_sessions):
        t += rng.exponential(1.0 / session_rate)
        sys_id = int(rng.choice(num_system, p=weights))
        stream = next_stream
        next_stream += 1
        context = system_len
        turns = max(1, int(rng.geometric(1.0 / turns_mean)))
        at = t
        for _ in range(turns):
            context += _lognormal(rng, user_len, 1.0, 8, 8192)
            if context > max_input_len:
                break
            out = _lognormal(rng, reply_len, 0.8, 4, 2048)
            parts = [(sys_id, sys_blocks), (stream, context // BLOCK - sys_blocks)]
            reqs.append(Request(0, at, Prompt(context, segments=_segs(parts)), out, tag=sess))
            context += out  # the reply becomes part of the next turn's prompt
            at += think_s * rng.exponential(1.0) + 1.0
    reqs.sort(key=lambda q: q.arrival)
    for i, q in enumerate(reqs):
        q.id = i
    return reqs


def shared_prefix(
    num_requests: int,
    *,
    rate: float = 4.0,
    num_prefixes: int = 64,
    zipf_s: float = 1.1,
    prefix_len: int = 4096,
    suffix_len: float = 256.0,
    output_len: float = 200.0,
    seed: int = 0,
) -> list[Request]:
    """Poisson arrivals; each request = Zipf-popular prefix + unique suffix."""
    rng = np.random.default_rng(seed)
    weights = _zipf_weights(num_prefixes, zipf_s)
    pre_blocks = prefix_len // BLOCK
    reqs: list[Request] = []
    t = 0.0
    for i in range(num_requests):
        t += rng.exponential(1.0 / rate)
        p = int(rng.choice(num_prefixes, p=weights))
        total = prefix_len + _lognormal(rng, suffix_len, 0.8, 1, 8192)
        parts = [(p, pre_blocks), (num_prefixes + i, total // BLOCK - pre_blocks)]
        out = _lognormal(rng, output_len, 0.8, 1, 2048)
        reqs.append(Request(i, t, Prompt(total, segments=_segs(parts)), out, tag=p))
    return reqs


def summarize_workload(reqs: list[Request], chunk_tokens: int = 512) -> dict[str, float]:
    """Size, rate and the hit ratio an infinite single cache would get (an upper bound)."""
    if not reqs:
        return {"requests": 0}
    seen: set[int] = set()
    hit = tot = 0
    for q in reqs:
        keys = q.prompt.block_keys(BLOCK)
        m = 0
        lim = min(len(keys), (q.input_len - 1) // BLOCK)
        while m < lim and keys[m] in seen:
            m += 1
        hit += m * BLOCK
        tot += q.input_len
        seen.update(keys)
    dur = max(q.arrival for q in reqs) - min(q.arrival for q in reqs)
    inp = np.array([q.input_len for q in reqs])
    outl = np.array([q.output_len for q in reqs])
    return {
        "requests": len(reqs),
        "duration_s": float(dur),
        "rate": len(reqs) / dur if dur > 0 else float("inf"),
        "input_mean": float(inp.mean()),
        "input_p99": float(np.percentile(inp, 99)),
        "output_mean": float(outl.mean()),
        "prompt_tok_per_s": float(inp.sum() / dur) if dur > 0 else float("inf"),
        "unique_tokens": len(seen) * BLOCK,
        "ideal_hit_ratio": hit / tot,
    }
