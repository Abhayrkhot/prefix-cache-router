"""A vLLM stand-in for tests and demos: the simulator's engine, run in real time.

It serves ``/v1/completions`` and ``/v1/chat/completions`` (streaming and not), ``/health``,
``/v1/models``, ``/reset_prefix_cache`` and ``/metrics`` with vLLM's metric names, and reports
``usage.prompt_tokens_details.cached_tokens`` like vLLM with ``--enable-prompt-tokens-details``.
Behind the API is :class:`~pcrouter.sim.engine.Engine`: continuous batching, chunked prefill
and a prefix cache with LRU eviction, each iteration sleeping for the step model's duration
divided by ``--speedup``. Token-id prompts are hashed exactly in 16-token blocks; text is
treated as ``--bytes-per-token`` bytes per token.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import time
import uuid
from typing import Any

from aiohttp import web

from pcrouter.costmodel import load_step_model
from pcrouter.prompt import Prompt, text_block_keys, token_block_keys
from pcrouter.server.keys import render_chat
from pcrouter.sim.engine import Engine, EngineConfig, Seq

__all__ = ["FakeWorker", "add_fake_worker_args", "fake_worker_main"]


class FakeWorker:
    def __init__(
        self,
        kv_tokens: int = 200_000,
        speedup: float = 1.0,
        step_model: str = "h200-qwen2.5-7b",
        model_name: str = "fake",
        bytes_per_token: float = 4.0,
        max_num_batched_tokens: int = 8192,
    ) -> None:
        self.cfg = EngineConfig(
            num_blocks=max(1, kv_tokens // 16), max_num_batched_tokens=max_num_batched_tokens
        )
        self.engine = Engine(self.cfg, load_step_model(step_model))
        self.speedup = speedup
        self.model_name = model_name
        self.bytes_per_token = bytes_per_token
        self._queues: dict[int, asyncio.Queue[tuple[str, int]]] = {}
        self._next = 0
        self._wake = asyncio.Event()
        self._task: asyncio.Task[None] | None = None
        self.queries = 0
        self.hits = 0

    # -- engine loop ---------------------------------------------------------------------
    async def _loop(self) -> None:
        while True:
            if not self.engine.has_work():
                self._wake.clear()
                await self._wake.wait()
                continue
            dt = self.engine.begin_step()
            if dt <= 0:
                await asyncio.sleep(0.001)
                continue
            await asyncio.sleep(dt / self.speedup)
            ev = self.engine.end_step(time.monotonic())
            for s in ev.first:
                self.queries += s.num_prompt
                self.hits += max(0, s.cached)
            finished = {id(s) for s in ev.finished}
            for s in ev.sampled:
                q = self._queues.get(s.rid)
                if q is not None:
                    q.put_nowait(("done" if id(s) in finished else "tok", s.cached))

    async def on_startup(self, app: web.Application) -> None:
        self._task = asyncio.create_task(self._loop())

    async def on_cleanup(self, app: web.Application) -> None:
        if self._task:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task

    def make_app(self) -> web.Application:
        app = web.Application(client_max_size=64 * 1024 * 1024)
        app.router.add_post("/v1/completions", self.handle)
        app.router.add_post("/v1/chat/completions", self.handle)
        app.router.add_get("/v1/models", self.models)
        app.router.add_get("/health", self.health)
        app.router.add_get("/metrics", self.metrics)
        app.router.add_post("/reset_prefix_cache", self.reset)
        app.on_startup.append(self.on_startup)
        app.on_cleanup.append(self.on_cleanup)
        return app

    # -- request handling ----------------------------------------------------------------
    def _prompt(self, path: str, body: dict[str, Any]) -> Prompt:
        if path.endswith("/chat/completions"):
            text = render_chat(body)
        else:
            p = body.get("prompt", "")
            if isinstance(p, list) and p and isinstance(p[0], int):
                toks = [int(t) for t in p]
                return Prompt(max(1, len(toks)), keys=token_block_keys(toks, 16))
            if isinstance(p, list) and p and isinstance(p[0], list):
                toks = [int(t) for t in p[0]]
                return Prompt(max(1, len(toks)), keys=token_block_keys(toks, 16))
            text = str(p[0]) if isinstance(p, list) and p else str(p)
        # Bytes stand in for tokens: one 16-token block = 16 * bytes_per_token bytes.
        n = max(1, round(len(text.encode("utf-8")) / self.bytes_per_token))
        keys = text_block_keys(text, max(1, round(16 * self.bytes_per_token)))
        return Prompt(n, keys=keys[: n // 16])

    async def handle(self, request: web.Request) -> web.StreamResponse:
        body = await request.json()
        chat = request.path.endswith("/chat/completions")
        max_tokens = int(body.get("max_tokens") or body.get("max_completion_tokens") or 16)
        prompt = self._prompt(request.path, body)
        rid = self._next
        self._next += 1
        seq = Seq(rid, prompt, max_tokens, time.monotonic())
        q: asyncio.Queue[tuple[str, int]] = asyncio.Queue()
        self._queues[rid] = q
        if not self.engine.add(seq):
            del self._queues[rid]
            return web.json_response(
                {"error": {"message": "prompt does not fit in the KV cache", "code": 400}},
                status=400,
            )
        self._wake.set()
        oid = f"{'chatcmpl' if chat else 'cmpl'}-{uuid.uuid4().hex[:16]}"
        usage: dict[str, Any] = {
            "prompt_tokens": prompt.num_tokens,
            "completion_tokens": max_tokens,
            "total_tokens": prompt.num_tokens + max_tokens,
        }
        try:
            if not body.get("stream"):
                cached = 0
                while True:
                    kind, cached = await q.get()
                    if kind == "done":
                        break
                usage["prompt_tokens_details"] = {"cached_tokens": cached}
                text = " tok" * max_tokens
                choice: dict[str, Any] = (
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": text},
                        "finish_reason": "length",
                    }
                    if chat
                    else {"index": 0, "text": text, "finish_reason": "length"}
                )
                return web.json_response(
                    {
                        "id": oid,
                        "object": "chat.completion" if chat else "text_completion",
                        "model": self.model_name,
                        "choices": [choice],
                        "usage": usage,
                    }
                )
            resp = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
            await resp.prepare(request)
            include_usage = bool((body.get("stream_options") or {}).get("include_usage"))
            cached = 0
            while True:
                kind, cached = await q.get()
                done = kind == "done"
                delta: dict[str, Any] = (
                    {
                        "index": 0,
                        "delta": {"content": " tok"},
                        "finish_reason": "length" if done else None,
                    }
                    if chat
                    else {"index": 0, "text": " tok", "finish_reason": "length" if done else None}
                )
                chunk = {
                    "id": oid,
                    "object": "chat.completion.chunk" if chat else "text_completion",
                    "model": self.model_name,
                    "choices": [delta],
                }
                await resp.write(b"data: " + json.dumps(chunk).encode() + b"\n\n")
                if done:
                    break
            if include_usage:
                usage["prompt_tokens_details"] = {"cached_tokens": cached}
                tail = {"id": oid, "model": self.model_name, "choices": [], "usage": usage}
                await resp.write(b"data: " + json.dumps(tail).encode() + b"\n\n")
            await resp.write(b"data: [DONE]\n\n")
            await resp.write_eof()
            return resp
        finally:
            self._queues.pop(rid, None)

    async def models(self, request: web.Request) -> web.StreamResponse:
        return web.json_response(
            {
                "object": "list",
                "data": [{"id": self.model_name, "object": "model", "owned_by": "pcrouter"}],
            }
        )

    async def health(self, request: web.Request) -> web.StreamResponse:
        return web.Response(text="")

    async def reset(self, request: web.Request) -> web.StreamResponse:
        ok = self.engine.reset_prefix_cache()
        return web.Response(status=200 if ok else 409)

    async def metrics(self, request: web.Request) -> web.StreamResponse:
        e = self.engine
        lines = [
            f"vllm:num_requests_running {len(e.running)}",
            f"vllm:num_requests_waiting {len(e.waiting)}",
            f"vllm:kv_cache_usage_perc {e.pool.usage():.6f}",
            f"vllm:prefix_cache_queries_total {self.queries}",
            f"vllm:prefix_cache_hits_total {self.hits}",
            f'vllm:cache_config_info{{block_size="16",num_gpu_blocks="{self.cfg.num_blocks}"}} 1.0',
        ]
        return web.Response(text="\n".join(lines) + "\n", content_type="text/plain")


def add_fake_worker_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8001)
    p.add_argument("--kv-tokens", type=int, default=200_000)
    p.add_argument(
        "--speedup",
        type=float,
        default=1.0,
        help="run iterations this many times faster than the step model says",
    )
    p.add_argument("--step-model", default="h200-qwen2.5-7b")
    p.add_argument("--model-name", default="fake")


def fake_worker_main(a: argparse.Namespace) -> int:  # pragma: no cover - process entry point
    w = FakeWorker(a.kv_tokens, a.speedup, a.step_model, a.model_name)
    web.run_app(w.make_app(), host=a.host, port=a.port, access_log=None)
    return 0
