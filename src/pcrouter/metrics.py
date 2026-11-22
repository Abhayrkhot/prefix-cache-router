"""Per-request records and their summary, shared by the simulator and live replays.

Percentile confidence intervals are distribution-free order-statistic intervals: for the
``q``-quantile of ``n`` samples the ranks ``n*q -/+ z*sqrt(n*q*(1-q))`` bracket the true quantile
with ~95% probability. Means get a normal-approximation interval.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

__all__ = ["Record", "quantile_ci", "read_records", "summarize", "write_records"]

Z95 = 1.959963984540054


@dataclass(slots=True)
class Record:
    rid: int
    worker: int
    arrival: float
    ttft: float  # seconds, -1 if the request failed
    e2e: float
    input_len: int
    output_len: int
    cached: int  # prompt tokens the engine reused (-1 unknown)
    cached_est: int = -1  # router's estimate at routing time
    preempted: int = 0
    ok: bool = True

    @property
    def tpot(self) -> float:
        if self.output_len <= 1 or not self.ok:
            return float("nan")
        return (self.e2e - self.ttft) / (self.output_len - 1)


def quantile_ci(
    x: Sequence[float] | npt.NDArray[np.float64], q: float
) -> tuple[float, float, float]:
    """(estimate, lower, upper) for the ``q``-quantile."""
    a = np.sort(np.asarray(x, dtype=float))
    n = len(a)
    if n == 0:
        return (math.nan, math.nan, math.nan)
    est = float(np.quantile(a, q))
    half = Z95 * math.sqrt(n * q * (1 - q))
    lo = max(0, math.floor(n * q - half) - 1)
    hi = min(n - 1, math.ceil(n * q + half))
    return (est, float(a[lo]), float(a[hi]))


def _mean_ci(x: npt.NDArray[np.float64]) -> tuple[float, float, float]:
    if len(x) == 0:
        return (math.nan, math.nan, math.nan)
    m = float(x.mean())
    h = Z95 * float(x.std(ddof=1)) / math.sqrt(len(x)) if len(x) > 1 else 0.0
    return (m, m - h, m + h)


def summarize(
    records: Sequence[Record],
    *,
    slo_ttft: float = 2.0,
    num_workers: int | None = None,
    warmup_s: float = 0.0,
) -> dict[str, Any]:
    """Latency percentiles (with CIs), cache reuse, SLO attainment and load balance.

    Requests arriving before ``warmup_s`` are excluded from latency and reuse figures (the
    caches start empty) but still count toward per-worker load.
    """
    ok = [r for r in records if r.ok and r.arrival >= warmup_s]
    out: dict[str, Any] = {
        "requests": len(records),
        "measured": len(ok),
        "failed": sum(not r.ok for r in records),
    }
    if not ok:
        return out
    ttft = np.array([r.ttft for r in ok])
    e2e = np.array([r.e2e for r in ok])
    tpot = np.array([r.tpot for r in ok])
    tpot = tpot[np.isfinite(tpot)]
    for name, arr in (("ttft", ttft), ("e2e", e2e), ("tpot", tpot)):
        m = _mean_ci(arr)
        out[f"{name}_mean"] = m[0]
        out[f"{name}_mean_ci"] = [m[1], m[2]]
        for q in (0.5, 0.9, 0.99):
            est, lo, hi = quantile_ci(arr, q)
            key = f"{name}_p{round(q * 100)}"
            out[key] = est
            out[f"{key}_ci"] = [lo, hi]
    inp = sum(r.input_len for r in ok)
    known = [r for r in ok if r.cached >= 0]
    if known:
        out["hit_ratio"] = sum(r.cached for r in known) / max(1, sum(r.input_len for r in known))
    with_est = [r for r in known if r.cached_est >= 0]
    if with_est:
        pred = sum(r.cached_est for r in with_est)
        act = sum(r.cached for r in with_est)
        out["est_hit_ratio"] = pred / max(1, sum(r.input_len for r in with_est))
        out["est_overshoot"] = (pred - act) / max(1, act)
    out["slo_ttft_s"] = slo_ttft
    out["slo_attainment"] = float(np.mean(ttft <= slo_ttft))
    t0 = min(r.arrival for r in ok)
    t1 = max(r.arrival + r.e2e for r in ok)
    span = max(t1 - t0, 1e-9)
    out["throughput_rps"] = len(ok) / span
    out["output_tok_s"] = sum(r.output_len for r in ok) / span
    out["prompt_tokens"] = inp
    out["preempted_frac"] = float(np.mean([r.preempted > 0 for r in ok]))
    if num_workers:
        per = np.zeros(num_workers)
        cnt = np.zeros(num_workers)
        for r in records:
            if r.ok and r.cached >= 0:
                per[r.worker] += r.input_len - r.cached
            cnt[r.worker] += 1
        out["worker_requests"] = cnt.astype(int).tolist()
        out["worker_prefill_tokens"] = per.astype(int).tolist()
        mean = float(np.mean(per))
        out["prefill_imbalance"] = float(np.max(per)) / mean if mean > 0 else 1.0
    return out


def write_records(path: str | Path, records: Iterable[Record]) -> None:
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(asdict(r)) + "\n")


def read_records(path: str | Path) -> list[Record]:
    with open(path) as f:
        return [Record(**json.loads(line)) for line in f if line.strip()]
