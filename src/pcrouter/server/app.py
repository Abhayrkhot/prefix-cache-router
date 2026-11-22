"""The routing proxy: an OpenAI-compatible HTTP front end for a pool of vLLM workers.

Per request it extracts prefix chunk keys (:mod:`pcrouter.server.keys`), asks the shared
:class:`~pcrouter.router.Router` for a worker, and streams the worker's response back
unchanged. Along the way it feeds the router the signals the policies use:

* the first streamed event marks the end of prefill (the request leaves the worker's
  prefill queue);
* the final usage chunk carries ``prompt_tokens_details.cached_tokens`` (vLLM with
  ``--enable-prompt-tokens-details``): the tokens the worker really reused. The proxy asks for
  it with ``stream_options.include_usage`` and strips the extra chunk if the client did not
  ask for it.

Operations: active health checks with ejection and re-admission, one retry on connection
failure, optional admission limit (429), KV capacity discovery from vLLM's
``vllm:cache_config_info`` metric, Prometheus metrics, and admin endpoints to switch policy or
reset caches between experiments.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import hmac
import json
import logging
import re
import time
from dataclasses import dataclass
from typing import Any

import aiohttp
from aiohttp import web

from pcrouter.costmodel import load_step_model
from pcrouter.policies import make_policy
from pcrouter.router import Router, RouterConfig, Ticket
from pcrouter.server.config import ServeConfig, config_from_args
from pcrouter.server.keys import KeyExtractor
from pcrouter.server.metrics import Metrics

__all__ = ["ProxyApp", "parse_cache_config", "serve_main"]

log = logging.getLogger("pcrouter")

_CACHE_INFO = re.compile(r"^vllm:cache_config_info\{([^}]*)\}", re.MULTILINE)
_LABEL = re.compile(r'(\w+)="([^"]*)"')
_HOP = {"host", "content-length", "transfer-encoding", "connection", "keep-alive", "upgrade"}


def parse_cache_config(metrics_text: str) -> int | None:
    """KV capacity in tokens from vLLM's ``vllm:cache_config_info`` labels."""
    m = _CACHE_INFO.search(metrics_text)
    if not m:
        return None
    labels = dict(_LABEL.findall(m.group(1)))
    try:
        blocks = int(float(labels["num_gpu_blocks"]))
        size = int(float(labels.get("block_size", "16")))
    except (KeyError, ValueError):
        return None
    return blocks * size if blocks > 0 else None


@dataclass
class _Ctx:
    router: Router
    ticket: Ticket
    worker: Worker
    strip_usage: bool
    t_start: float


@dataclass
class Worker:
    idx: int
    url: str
    healthy: bool = True
    kv_tokens: int = 0
    failures: int = 0
    down_since: float = 0.0


class ProxyApp:
    def __init__(self, cfg: ServeConfig, metrics: Metrics | None = None) -> None:
        self.cfg = cfg
        self.workers = [Worker(i, u) for i, u in enumerate(cfg.workers)]
        self.model = load_step_model(cfg.step_model)
        self.metrics = metrics or Metrics([w.url for w in self.workers])
        tok = None
        if cfg.tokenizer:
            from transformers import AutoTokenizer  # type: ignore[import-not-found]

            hf = AutoTokenizer.from_pretrained(cfg.tokenizer)

            def tok(text: str) -> list[int]:
                return list(hf.encode(text, add_special_tokens=False))

        self.keys = KeyExtractor(cfg.chunk_tokens, cfg.bytes_per_token, tok)
        self.session: aiohttp.ClientSession | None = None
        self.router: Router = self._build_router(cfg.policy, cfg.policy_params)
        self._health_task: asyncio.Task[None] | None = None

    # -- setup ---------------------------------------------------------------------------
    def _build_router(self, policy: str, params: dict[str, Any]) -> Router:
        n = len(self.workers)
        caps = [
            w.kv_tokens or self.cfg.kv_tokens or self.cfg.kv_tokens_fallback for w in self.workers
        ]
        pol = make_policy(policy, n, seed=self.cfg.seed, model=self.model, **params)
        rc = RouterConfig(
            num_workers=n,
            chunk_tokens=self.cfg.chunk_tokens,
            capacity_tokens=caps,
            capacity_factor=self.cfg.capacity_factor,
            budget=self.cfg.budget,
            feedback=self.cfg.feedback,
        )
        return Router(pol, rc, self.model)

    async def _discover(self) -> None:
        assert self.session is not None
        for w in self.workers:
            if self.cfg.kv_tokens:
                w.kv_tokens = self.cfg.kv_tokens
                continue
            try:
                async with self.session.get(
                    w.url + "/metrics", timeout=aiohttp.ClientTimeout(total=5)
                ) as r:
                    cap = parse_cache_config(await r.text()) if r.status == 200 else None
            except (aiohttp.ClientError, asyncio.TimeoutError):
                cap = None
            w.kv_tokens = cap or self.cfg.kv_tokens_fallback
            log.info(
                "worker %s: KV capacity %d tokens (%s)",
                w.url,
                w.kv_tokens,
                "discovered" if cap else "fallback",
            )
        self.router = self._build_router(self.router.policy.name, self.router.policy.params())

    async def on_startup(self, app: web.Application) -> None:
        self.session = aiohttp.ClientSession(
            # Idle pooled connections are dropped before the worker's own keep-alive timeout
            # (uvicorn, which serves vLLM, closes them after 5 s): reusing a connection the
            # server has just closed fails as "server disconnected" without the worker being
            # down.
            connector=aiohttp.TCPConnector(
                limit=0, ttl_dns_cache=300, keepalive_timeout=self.cfg.keepalive_s
            ),
            timeout=aiohttp.ClientTimeout(total=None, sock_connect=self.cfg.connect_timeout_s),
            auto_decompress=False,
        )
        await self._discover()
        self._health_task = asyncio.create_task(self._health_loop())

    async def on_cleanup(self, app: web.Application) -> None:
        if self._health_task:
            self._health_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._health_task
        if self.session:
            await self.session.close()

    def make_app(self) -> web.Application:
        app = web.Application(client_max_size=64 * 1024 * 1024)
        app.router.add_post("/v1/completions", self.handle_generate)
        app.router.add_post("/v1/chat/completions", self.handle_generate)
        app.router.add_get("/v1/models", self.handle_models)
        app.router.add_get("/health", self.handle_health)
        app.router.add_get("/metrics", self.handle_metrics)
        app.router.add_get("/debug/state", self.handle_state)
        app.router.add_post("/admin/policy", self.handle_set_policy)
        app.router.add_post("/admin/reset", self.handle_reset)
        app.on_startup.append(self.on_startup)
        app.on_cleanup.append(self.on_cleanup)
        return app

    # -- health --------------------------------------------------------------------------
    async def _probe(self, w: Worker) -> None:
        assert self.session is not None
        try:
            async with self.session.get(
                w.url + "/health", timeout=aiohttp.ClientTimeout(total=self.cfg.health_timeout_s)
            ) as r:
                ok = r.status == 200
        except (aiohttp.ClientError, asyncio.TimeoutError):
            ok = False
        if ok and not w.healthy:
            down = time.monotonic() - w.down_since
            w.healthy = True
            w.failures = 0
            if down >= self.cfg.reset_after_s:  # long enough to have restarted
                log.warning("worker %s back after %.0fs; its cache is assumed lost", w.url, down)
                self.router.reset_worker(w.idx)
            else:
                log.warning("worker %s back after %.1fs; keeping its cache model", w.url, down)
        elif not ok and w.healthy:
            log.warning("worker %s failed its health check; ejected", w.url)
            w.healthy = False
            w.down_since = time.monotonic()
        self.metrics.healthy.labels(w.url).set(1 if w.healthy else 0)

    async def _health_loop(self) -> None:
        while True:
            await asyncio.gather(*(self._probe(w) for w in self.workers))
            await asyncio.sleep(self.cfg.health_interval_s)

    def _mark_down(self, w: Worker) -> None:
        w.failures += 1
        if w.healthy:
            log.warning("worker %s unreachable; ejected until it passes a health check", w.url)
            w.down_since = time.monotonic()
        w.healthy = False
        self.metrics.healthy.labels(w.url).set(0)

    async def _post(
        self, w: Worker, path: str, raw: bytes, headers: dict[str, str]
    ) -> aiohttp.ClientResponse:
        assert self.session is not None
        try:
            return await self.session.post(w.url + path, data=raw, headers=headers)
        except (aiohttp.ServerDisconnectedError, aiohttp.ClientOSError):
            # Most likely a pooled connection the worker closed: one retry on a fresh one.
            self.metrics.retries.inc()
            return await self.session.post(w.url + path, data=raw, headers=headers)

    # -- generation ----------------------------------------------------------------------
    async def handle_generate(self, request: web.Request) -> web.StreamResponse:
        t_start = time.perf_counter()
        raw = await request.read()
        try:
            body = json.loads(raw)
            if not isinstance(body, dict):
                raise ValueError
        except ValueError:
            return _error(400, "request body must be a JSON object")
        if self.cfg.max_inflight and self._inflight() >= self.cfg.max_inflight:
            self.metrics.rejected.inc()
            return _error(429, "router at max in-flight requests")
        key = self.keys.extract(request.path, body)
        stream = bool(body.get("stream"))
        strip_usage = False
        if stream:
            so = body.get("stream_options") or {}
            if not so.get("include_usage"):
                body["stream_options"] = {**so, "include_usage": True}
                strip_usage = True
                raw = json.dumps(body).encode()
        headers = {k: v for k, v in request.headers.items() if k.lower() not in _HOP}
        excluded: set[int] = set()
        assert self.session is not None
        router = self.router  # the ticket belongs to this router even if the policy changes
        for attempt in range(self.cfg.max_retries + 1):
            healthy = [w.healthy and w.idx not in excluded for w in self.workers]
            if not any(healthy):
                return _error(503, "no healthy workers")
            t0 = time.perf_counter()
            ticket = router.route(
                key.input_len, key.max_tokens, key.chunks, healthy=healthy, now=time.monotonic()
            )
            self.metrics.decision.observe(time.perf_counter() - t0)
            w = self.workers[ticket.worker]
            try:
                upstream = await self._post(w, request.path, raw, headers)
            except (aiohttp.ClientConnectorError, asyncio.TimeoutError):
                router.fail(ticket)  # cannot connect at all: the worker is down
                self._mark_down(w)
                excluded.add(w.idx)
                if attempt < self.cfg.max_retries:
                    self.metrics.retries.inc()
                continue
            except aiohttp.ClientError:
                router.fail(ticket)  # transient failure: fail over, but keep the worker
                excluded.add(w.idx)
                if attempt < self.cfg.max_retries:
                    self.metrics.retries.inc()
                continue
            break
        else:
            return _error(502, "all attempts to reach a worker failed")
        self.metrics.routed.labels(w.url).inc()
        self.metrics.prompt_tokens.labels(w.url).inc(key.input_len)
        self.metrics.predicted_cached.labels(w.url).inc(ticket.cached_est)
        ctx = _Ctx(router, ticket, w, strip_usage, t_start)
        try:
            if stream and upstream.status == 200:
                return await self._relay_stream(request, upstream, ctx)
            data = await upstream.read()
            router.first_token(ticket)
            if upstream.status == 200:
                self._feedback(ctx, _usage_cached(data))
            self.metrics.requests.labels(w.url, str(upstream.status)).inc()
            return web.Response(
                body=data,
                status=upstream.status,
                headers={
                    "Content-Type": upstream.headers.get("Content-Type", "application/json"),
                    "X-PCRouter-Worker": str(w.idx),
                },
            )
        finally:
            upstream.release()
            router.finish(ticket)
            self.metrics.duration.observe(time.perf_counter() - t_start)

    def _feedback(self, ctx: _Ctx, cached: int | None) -> None:
        if cached is None:
            return
        ctx.router.cached_feedback(ctx.ticket, cached)
        self.metrics.actual_cached.labels(ctx.worker.url).inc(cached)

    async def _relay_stream(
        self, request: web.Request, upstream: aiohttp.ClientResponse, ctx: _Ctx
    ) -> web.StreamResponse:
        w = ctx.worker
        resp = web.StreamResponse(
            status=200,
            headers={
                "Content-Type": upstream.headers.get("Content-Type", "text/event-stream"),
                "Cache-Control": "no-cache",
                "X-PCRouter-Worker": str(w.idx),
            },
        )
        await resp.prepare(request)
        buf = b""
        status = "200"
        try:
            async for chunk in upstream.content.iter_any():
                buf += chunk
                while b"\n\n" in buf:
                    event, buf = buf.split(b"\n\n", 1)
                    if self._on_event(event, ctx):
                        await resp.write(event + b"\n\n")
            if buf:
                await resp.write(buf)
            await resp.write_eof()
        except ConnectionResetError:
            status = "client_closed"  # the client went away; dropping upstream cancels it
        except aiohttp.ClientError:
            status = "upstream_error"  # the health checker decides whether the worker is down
        finally:
            self.metrics.requests.labels(w.url, status).inc()
        return resp

    def _on_event(self, event: bytes, ctx: _Ctx) -> bool:
        """Observe one SSE event; return whether to forward it to the client."""
        if not event.startswith(b"data:"):
            return True
        payload = event[5:].strip()
        if payload == b"[DONE]":
            return True
        if not ctx.ticket.first_seen:
            ctx.router.first_token(ctx.ticket)
            self.metrics.ttft.observe(time.perf_counter() - ctx.t_start)
        if b'"usage"' not in payload:
            return True
        try:
            obj = json.loads(payload)
        except ValueError:
            return True
        usage = obj.get("usage") if isinstance(obj, dict) else None
        if usage:
            details = usage.get("prompt_tokens_details") or {}
            cached = details.get("cached_tokens")
            if isinstance(cached, int):
                self._feedback(ctx, cached)
        return not (ctx.strip_usage and usage and not obj.get("choices"))

    # -- admin & introspection -------------------------------------------------------------
    def _inflight(self) -> int:
        return sum(ld.inflight for ld in self.router.loads)

    async def handle_models(self, request: web.Request) -> web.StreamResponse:
        assert self.session is not None
        for w in self.workers:
            if not w.healthy:
                continue
            try:
                async with self.session.get(w.url + "/v1/models") as r:
                    return web.Response(
                        body=await r.read(), status=r.status, content_type="application/json"
                    )
            except aiohttp.ClientError:
                self._mark_down(w)
        return _error(503, "no healthy workers")

    async def handle_health(self, request: web.Request) -> web.StreamResponse:
        up = sum(w.healthy for w in self.workers)
        return web.json_response(
            {"healthy_workers": up, "workers": len(self.workers)}, status=200 if up else 503
        )

    async def handle_metrics(self, request: web.Request) -> web.StreamResponse:
        for w, ld in zip(self.workers, self.router.loads):
            self.metrics.inflight.labels(w.url).set(ld.inflight)
            self.metrics.pending.labels(w.url).set(ld.pending_tokens)
            self.metrics.index_chunks.labels(w.url).set(self.router.index.size(w.idx))
        body, ctype = self.metrics.render()
        return web.Response(body=body, headers={"Content-Type": ctype})

    async def handle_state(self, request: web.Request) -> web.StreamResponse:
        r = self.router
        return web.json_response(
            {
                "policy": r.policy.name,
                "params": r.policy.params(),
                "chunk_tokens": r.cfg.chunk_tokens,
                "workers": [
                    {
                        "url": w.url,
                        "healthy": w.healthy,
                        "kv_tokens": w.kv_tokens,
                        "inflight": ld.inflight,
                        "prefilling": ld.prefilling,
                        "pending_prefill_tokens": ld.pending_tokens,
                        "routed": ld.routed,
                        "index_chunks": r.index.size(w.idx),
                        "index_capacity": r.index.capacity[w.idx],
                        "predicted_cached": ld.predicted_cached,
                        "actual_cached": ld.actual_cached,
                    }
                    for w, ld in zip(self.workers, r.loads)
                ],
            }
        )

    def _admin_denied(self, request: web.Request) -> web.Response | None:
        """Admin calls need the bearer token if one is configured, else a loopback client."""
        token = self.cfg.admin_token
        if token:
            given = request.headers.get("Authorization", "")
            if not hmac.compare_digest(given.encode(), f"Bearer {token}".encode()):
                return _error(401, "admin token required")
            return None
        if request.remote not in ("127.0.0.1", "::1"):
            return _error(403, "admin endpoints are loopback-only without --admin-token")
        return None

    async def handle_set_policy(self, request: web.Request) -> web.StreamResponse:
        if denied := self._admin_denied(request):
            return denied
        if self._inflight():
            return _error(409, "requests in flight; drain before switching policy")
        try:
            body = await request.json()
            router = self._build_router(str(body["policy"]), dict(body.get("params") or {}))
        except (ValueError, KeyError, TypeError) as e:
            return _error(400, f"bad policy request: {e}")
        self.router = router
        log.info("policy set to %s %s", router.policy.name, router.policy.params())
        return web.json_response({"policy": router.policy.name, "params": router.policy.params()})

    async def handle_reset(self, request: web.Request) -> web.StreamResponse:
        """Reset the router's state and every worker's prefix cache (``/reset_prefix_cache``,
        vLLM needs ``VLLM_SERVER_DEV_MODE=1``)."""
        if denied := self._admin_denied(request):
            return denied
        if self._inflight():
            return _error(409, "requests in flight; drain before resetting")
        assert self.session is not None
        results = {}
        for w in self.workers:
            try:
                async with self.session.post(w.url + "/reset_prefix_cache") as r:
                    results[w.url] = r.status
            except aiohttp.ClientError as e:
                results[w.url] = str(e)  # type: ignore[assignment]
        self.router = self._build_router(self.router.policy.name, self.router.policy.params())
        return web.json_response({"reset": results})


def _usage_cached(data: bytes) -> int | None:
    try:
        obj = json.loads(data)
    except ValueError:
        return None
    usage = obj.get("usage") if isinstance(obj, dict) else None
    if not isinstance(usage, dict):
        return None
    cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
    return cached if isinstance(cached, int) else None


def _error(status: int, msg: str) -> web.Response:
    return web.json_response({"error": {"message": msg, "code": status}}, status=status)


def serve_main(a: argparse.Namespace) -> int:  # pragma: no cover - process entry point
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = config_from_args(a)
    proxy = ProxyApp(cfg)
    log.info(
        "pcrouter on %s:%d, policy %s, %d workers", cfg.host, cfg.port, cfg.policy, len(cfg.workers)
    )
    web.run_app(
        proxy.make_app(), host=cfg.host, port=cfg.port, access_log=None, shutdown_timeout=30
    )
    return 0
