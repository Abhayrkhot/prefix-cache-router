"""Discrete-event simulation of a router in front of N engines.

Arrivals are routed by a :class:`~pcrouter.router.Router` (the same object the live proxy
uses) using only router-side knowledge, then queued on that worker's
:class:`~pcrouter.sim.engine.Engine`. Engines run iterations back to back while they have
work; the router hears about first tokens (with the engine's true cached-token count, as vLLM
reports it per request) and completions as they happen.

``oracle=True`` replaces the router's predicted prefix matches with the engines' actual cache
contents: an upper bound for what exact cache-state feedback (e.g. vLLM KV events) could give.
"""

from __future__ import annotations

import heapq
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from pcrouter.costmodel import StepModel
from pcrouter.metrics import Record, summarize
from pcrouter.policies import Policy
from pcrouter.prompt import BLOCK
from pcrouter.router import Router, RouterConfig, Ticket
from pcrouter.sim.engine import Engine, EngineConfig, Seq
from pcrouter.workloads import Request

__all__ = ["SimResult", "simulate"]


@dataclass
class SimResult:
    records: list[Record]
    summary: dict[str, Any]
    engines: list[dict[str, Any]] = field(default_factory=list)
    wall_s: float = 0.0


def simulate(
    requests: Sequence[Request],
    policy: Policy,
    engine_cfg: EngineConfig | Sequence[EngineConfig],
    model: StepModel,
    *,
    num_workers: int | None = None,
    chunk_tokens: int = 512,
    capacity_factor: float = 1.0,
    feedback: bool = False,
    oracle: bool = False,
    slo_ttft: float = 2.0,
    warmup_s: float = 0.0,
) -> SimResult:
    cfgs = (
        list(engine_cfg)
        if not isinstance(engine_cfg, EngineConfig)
        else [engine_cfg] * (num_workers or policy.num_workers)
    )
    n = len(cfgs)
    if n != policy.num_workers:
        raise ValueError(f"policy built for {policy.num_workers} workers, cluster has {n}")
    if chunk_tokens % BLOCK:
        raise ValueError(f"chunk_tokens must be a multiple of {BLOCK}")
    bpc = chunk_tokens // BLOCK
    engines = [Engine(c, model) for c in cfgs]
    router = Router(
        policy,
        RouterConfig(
            num_workers=n,
            chunk_tokens=chunk_tokens,
            capacity_tokens=[c.capacity_tokens for c in cfgs],
            capacity_factor=capacity_factor,
            budget=cfgs[0].max_num_batched_tokens,
            feedback=feedback,
        ),
        model,
    )
    tickets: dict[int, Ticket] = {}
    workers: dict[int, int] = {}
    records: list[Record] = []
    heap: list[tuple[float, int, int]] = []  # (time, seq, worker) of step completions
    busy = [False] * n
    order = 0
    wall0 = time.perf_counter()

    def start(w: int, now: float) -> None:
        nonlocal order
        eng = engines[w]
        # An iteration that preempted every running request schedules nothing (vLLM admits
        # no waiting request in a step that preempted); the next attempt admits them.
        for _ in range(3):
            if not eng.has_work():
                break
            dt = eng.begin_step()
            if dt > 0:
                busy[w] = True
                order += 1
                heapq.heappush(heap, (now + dt, order, w))
                return
        busy[w] = False

    reqs = sorted(requests, key=lambda r: r.arrival)
    i = 0
    inf = float("inf")
    while i < len(reqs) or heap:
        t_arr = reqs[i].arrival if i < len(reqs) else inf
        t_step = heap[0][0] if heap else inf
        if t_arr <= t_step:
            q = reqs[i]
            i += 1
            chunks = q.prompt.chunk_keys(bpc)
            match = None
            if oracle:
                match = [e.cached_prefix_tokens(q.prompt) // chunk_tokens for e in engines]
            tk = router.route(q.input_len, q.output_len, chunks, match=match, now=q.arrival)
            w = tk.worker
            seq = Seq(q.id, q.prompt, q.output_len, q.arrival)
            if not engines[w].add(seq):
                router.fail(tk)
                records.append(
                    Record(
                        q.id,
                        w,
                        q.arrival,
                        -1.0,
                        -1.0,
                        q.input_len,
                        q.output_len,
                        -1,
                        tk.cached_est,
                        0,
                        ok=False,
                    )
                )
                continue
            tickets[q.id] = tk
            workers[q.id] = w
            if not busy[w]:
                start(w, q.arrival)
        else:
            now, _, w = heapq.heappop(heap)
            ev = engines[w].end_step(now)
            for s in ev.first:
                router.first_token(tickets[s.rid], s.cached)
            for s in ev.finished:
                tk = tickets.pop(s.rid)
                router.finish(tk)
                records.append(
                    Record(
                        s.rid,
                        workers.pop(s.rid),
                        s.t_arrival,
                        s.t_first - s.t_arrival,
                        s.t_finish - s.t_arrival,
                        s.num_prompt,
                        s.max_out,
                        s.cached,
                        tk.cached_est,
                        s.preempted,
                    )
                )
            busy[w] = False
            start(w, now)

    stuck = [s for e in engines for s in (*e.waiting, *e.running)]
    for s in stuck:  # pragma: no cover - only if a request can never be scheduled
        records.append(
            Record(
                s.rid,
                workers.get(s.rid, -1),
                s.t_arrival,
                -1.0,
                -1.0,
                s.num_prompt,
                s.max_out,
                -1,
                -1,
                s.preempted,
                ok=False,
            )
        )
    records.sort(key=lambda r: r.rid)
    summary = summarize(records, slo_ttft=slo_ttft, num_workers=n, warmup_s=warmup_s)
    summary["policy"] = policy.name
    summary["policy_params"] = policy.params()
    est = [
        {
            "steps": e.steps,
            "busy_s": e.busy_s,
            "prefill_tokens": e.prefill_tokens,
            "decode_tokens": e.decode_tokens,
            "hit_tokens": e.hit_tokens,
            "prompt_tokens": e.prompt_tokens,
            "preemptions": e.preemptions,
            "evictions": e.pool.evictions,
            "index_evictions": router.index.evictions[k],
        }
        for k, e in enumerate(engines)
    ]
    busy_s = [e["busy_s"] for e in est]
    mean_busy = sum(busy_s) / n
    summary["busy_imbalance"] = max(busy_s) / mean_busy if mean_busy > 0 else 1.0
    return SimResult(records, summary, est, time.perf_counter() - wall0)
