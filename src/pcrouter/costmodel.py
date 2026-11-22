"""Per-iteration step-time model of a vLLM engine.

The packaged ``h200-qwen2.5-7b`` model was calibrated by `batchsim
<https://github.com/Abhayrkhot/Continuous-Batching-Simulator>`_ from ~19.5k CUDA-event-timed
vLLM 0.34 iterations on one H200 (held-out median step error 1.1%). Its features include the
query-key pairs of every prefill chunk, so a chunk computed *after a cached prefix* is priced
correctly: ``n`` new tokens after ``c`` cached ones cost ``n*c + n(n+1)/2`` attention pairs.

The simulator charges every engine iteration with :meth:`StepModel.step_time`; the TTFT-cost
routing policy uses :meth:`StepModel.prefill_time` and :meth:`StepModel.decode_slowdown` to
price a candidate placement.
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from typing import Any

__all__ = ["BatchStats", "StepModel", "load_step_model"]

_BASE_FEATURES = ("const", "tokens", "seqs", "prefill_seqs", "prefill_attn_m", "decode_ctx_m")


@dataclass(slots=True)
class BatchStats:
    """Shape of one engine iteration."""

    num_tokens: int = 0
    num_seqs: int = 0
    num_prefill: int = 0
    prefill_attn: float = 0.0
    decode_ctx: float = 0.0

    def add(self, new_tokens: int, context: int) -> None:
        """One sequence computing ``new_tokens`` after ``context`` already-computed tokens."""
        self.num_tokens += new_tokens
        self.num_seqs += 1
        if new_tokens == 1:
            self.decode_ctx += context + 1
        else:
            self.num_prefill += 1
            self.prefill_attn += new_tokens * context + new_tokens * (new_tokens + 1) / 2


def _pad(n: int, sizes: Sequence[int]) -> int:
    """Token count after padding to the next CUDA-graph capture size (if any fits)."""
    if not sizes or n > sizes[-1]:
        return n
    lo, hi = 0, len(sizes) - 1
    while lo < hi:
        mid = (lo + hi) // 2
        if sizes[mid] >= n:
            hi = mid
        else:
            lo = mid + 1
    return sizes[lo]


@dataclass
class StepModel:
    """``t = c0 + c_tok*T' + sum_k h_k*max(0,T'-k) + c_seq*S + c_pseq*P + c_pa*A + c_dc*D``.

    ``T'`` tokens padded to the CUDA-graph size, ``S`` sequences, ``P`` prefill sequences,
    ``A`` prefill attention pairs (millions), ``D`` KV tokens read by decodes (millions).
    Same JSON format as batchsim's ``LinearStepModel``.
    """

    coef: dict[str, float]
    knots: tuple[int, ...] = ()
    capture_sizes: tuple[int, ...] = ()
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.knots = tuple(int(k) for k in self.knots)
        self.capture_sizes = tuple(int(c) for c in self.capture_sizes)
        names = (*_BASE_FEATURES, *(f"hinge_{k}" for k in self.knots))
        missing = [n for n in names if n not in self.coef]
        if missing:
            raise ValueError(f"missing coefficients: {missing}")
        c = self.coef
        self._c = [float(c[n]) for n in _BASE_FEATURES]
        self._h = [(k, float(c[f"hinge_{k}"])) for k in self.knots]

    def step_time(self, s: BatchStats) -> float:
        if s.num_seqs == 0:
            return 0.0
        t = float(_pad(s.num_tokens, self.capture_sizes))
        c0, ct, cs, cp, ca, cd = self._c
        out = (
            c0
            + ct * t
            + cs * s.num_seqs
            + cp * s.num_prefill
            + ca * s.prefill_attn * 1e-6
            + cd * s.decode_ctx * 1e-6
        )
        for k, h in self._h:
            if t > k:
                out += h * (t - k)
        return out

    def prefill_time(self, new_tokens: int, context: int, budget: int = 8192) -> float:
        """Seconds of engine time to prefill ``new_tokens`` after ``context`` cached tokens,
        computed alone in chunks of ``budget`` tokens."""
        if new_tokens <= 0:
            return 0.0
        total = 0.0
        done = 0
        while done < new_tokens:
            n = min(budget, new_tokens - done)
            s = BatchStats()
            s.add(n, context + done)
            total += self.step_time(s)
            done += n
        return total

    def decode_slowdown(self, decoding: int, avg_context: int = 8192, budget: int = 8192) -> float:
        """How much longer prefill takes while ``decoding`` sequences share every iteration:
        they take ``decoding`` tokens of the budget and add their KV reads to each step."""
        d = max(0, min(decoding, budget - 1))
        if d == 0:
            return 1.0
        base = BatchStats()
        base.add(budget, avg_context)
        mixed = BatchStats()
        mixed.add(budget - d, avg_context)
        for _ in range(min(d, 64)):
            mixed.add(1, avg_context)
        # The remaining decodes scale linearly; avoid a d-long loop for large batches.
        if d > 64:
            mixed.num_tokens += d - 64
            mixed.num_seqs += d - 64
            mixed.decode_ctx += (d - 64) * (avg_context + 1)
        per_tok_base = self.step_time(base) / budget
        per_tok_mixed = self.step_time(mixed) / (budget - d)
        return max(1.0, per_tok_mixed / per_tok_base)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "linear",
            "coef": self.coef,
            "knots": list(self.knots),
            "capture_sizes": list(self.capture_sizes),
            "meta": self.meta,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> StepModel:
        if d.get("kind", "linear") != "linear":
            raise ValueError(f"unsupported step model kind {d.get('kind')!r}")
        return cls(
            coef={k: float(v) for k, v in d["coef"].items()},
            knots=tuple(d.get("knots", ())),
            capture_sizes=tuple(d.get("capture_sizes", ())),
            meta=dict(d.get("meta", {})),
        )

    @classmethod
    def simple(cls, const: float, per_token: float, attn_per_m: float = 0.0) -> StepModel:
        """Hand-written model for tests/examples: ``const + per_token*T + attn*A``."""
        coef = dict.fromkeys(_BASE_FEATURES, 0.0)
        coef.update(const=const, tokens=per_token, prefill_attn_m=attn_per_m)
        return cls(coef=coef)


PACKAGED = ("h200-qwen2.5-7b",)


def load_step_model(name_or_path: str | Path) -> StepModel:
    """Load a packaged model by name (``h200-qwen2.5-7b``) or a batchsim-format JSON file."""
    name = str(name_or_path)
    if name in PACKAGED:
        text = resources.files("pcrouter.data").joinpath(f"{name}.json").read_text()
    else:
        text = Path(name).read_text()
    model = StepModel.from_dict(json.loads(text))
    if not all(math.isfinite(v) and v >= 0 for v in model.coef.values()):
        raise ValueError("step model coefficients must be finite and non-negative")
    return model
