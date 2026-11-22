"""End to end: fake vLLM workers behind the proxy, all in one event loop."""

from __future__ import annotations

import asyncio
import contextlib
import json
import socket
from collections.abc import AsyncIterator

import aiohttp
import pytest
from aiohttp import web

from pcrouter.server.app import ProxyApp, parse_cache_config
from pcrouter.server.config import ServeConfig
from pcrouter.server.fakeworker import FakeWorker
from pcrouter.server.keys import KeyExtractor, render_chat
from pcrouter.server.replay import prompt_tokens, replay
from pcrouter.workloads import multiturn


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


@contextlib.asynccontextmanager
async def serving(app: web.Application) -> AsyncIterator[str]:
    runner = web.AppRunner(app)
    await runner.setup()
    port = free_port()
    site = web.TCPSite(runner, "127.0.0.1", port)
    await site.start()
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        await runner.cleanup()


@contextlib.asynccontextmanager
async def cluster(n: int = 2, policy: str = "ttft", **cfg_kw) -> AsyncIterator[tuple]:
    fakes = [FakeWorker(kv_tokens=50_000, speedup=50.0) for _ in range(n)]
    async with contextlib.AsyncExitStack() as stack:
        urls = [await stack.enter_async_context(serving(f.make_app())) for f in fakes]
        cfg = ServeConfig(
            workers=urls, policy=policy, chunk_tokens=64, health_interval_s=0.2, **cfg_kw
        )
        proxy = ProxyApp(cfg)
        purl = await stack.enter_async_context(serving(proxy.make_app()))
        yield proxy, purl, fakes, urls


def run(coro):
    return asyncio.run(coro)


async def post(session, url, body):
    async with session.post(url, json=body) as r:
        return r.status, r.headers.get("X-PCRouter-Worker"), await r.read()


def test_streaming_affinity_and_feedback():
    async def go():
        async with cluster(2) as (proxy, purl, _fakes, _), aiohttp.ClientSession() as s:
            prompt = list(range(2000, 2000 + 1024))
            body = {"model": "fake", "prompt": prompt, "max_tokens": 5, "stream": True}
            st, w1, raw = await post(s, purl + "/v1/completions", body)
            assert st == 200
            events = [e for e in raw.split(b"\n\n") if e.startswith(b"data:")]
            assert events[-1] == b"data: [DONE]"
            # The proxy asked for usage but the client did not: no usage chunk leaks through.
            assert not any(b'"usage"' in e for e in events)
            assert len(events) == 6
            st, w2, _ = await post(s, purl + "/v1/completions", {**body, "prompt": [*prompt, 7]})
            assert w2 == w1  # same prefix, same worker
            ld = proxy.router.loads[int(w1)]
            assert ld.actual_cached >= 1024 - 64  # cached-token feedback arrived
            assert ld.inflight == 0 and ld.prefilling == 0
            # A client that asks for usage gets it.
            body2 = {**body, "stream_options": {"include_usage": True}}
            _, _, raw = await post(s, purl + "/v1/completions", body2)
            assert b'"cached_tokens"' in raw

    run(go())


def test_non_streaming_chat_and_text():
    async def go():
        async with cluster(2) as (_proxy, purl, *_), aiohttp.ClientSession() as s:
            chat = {
                "model": "fake",
                "max_tokens": 3,
                "messages": [
                    {"role": "system", "content": "s" * 3000},
                    {"role": "user", "content": "hello"},
                ],
            }
            st, w, raw = await post(s, purl + "/v1/chat/completions", chat)
            assert st == 200
            out = json.loads(raw)
            assert out["choices"][0]["message"]["content"]
            assert "cached_tokens" in out["usage"]["prompt_tokens_details"]
            st, w2, _ = await post(s, purl + "/v1/chat/completions", chat)
            assert w2 == w
            st, _, raw = await post(
                s,
                purl + "/v1/completions",
                {"model": "fake", "prompt": "abc " * 200, "max_tokens": 2},
            )
            assert st == 200 and json.loads(raw)["choices"][0]["text"]
            st, _, _ = await post(s, purl + "/v1/completions", [1, 2])
            assert st == 400
            async with s.post(purl + "/v1/completions", data=b"{not json") as r:
                assert r.status == 400

    run(go())


def test_worker_failure_retry_and_recovery():
    async def go():
        async with (
            cluster(2, policy="round-robin") as (proxy, purl, _fakes, urls),
            aiohttp.ClientSession() as s,
        ):
            proxy.workers[1].url = f"http://127.0.0.1:{free_port()}"  # nothing listens
            body = {"model": "fake", "prompt": list(range(100)), "max_tokens": 2}
            statuses = [(await post(s, purl + "/v1/completions", body))[0] for _ in range(4)]
            assert statuses == [200] * 4  # the failure was retried on the healthy worker
            assert not proxy.workers[1].healthy
            async with s.get(purl + "/health") as r:
                assert (await r.json())["healthy_workers"] == 1
            proxy.workers[1].url = urls[1]
            for _ in range(50):
                if proxy.workers[1].healthy:
                    break
                await asyncio.sleep(0.05)
            assert proxy.workers[1].healthy  # re-admitted by the health checker
            proxy.workers[0].url = proxy.workers[1].url = f"http://127.0.0.1:{free_port()}"
            st, _, _ = await post(s, purl + "/v1/completions", body)
            assert st in (502, 503)

    run(go())


def test_admin_metrics_state_and_admission():
    async def go():
        async with (
            cluster(2, max_inflight=1) as (proxy, purl, _fakes, _),
            aiohttp.ClientSession() as s,
        ):
            async with s.get(purl + "/v1/models") as r:
                assert (await r.json())["data"][0]["id"] == "fake"
            async with s.post(
                purl + "/admin/policy", json={"policy": "affinity", "params": {"mode": "prefer"}}
            ) as r:
                assert r.status == 200
            assert proxy.router.policy.name == "affinity"
            async with s.post(purl + "/admin/policy", json={"policy": "nope"}) as r:
                assert r.status == 400
            body = {"model": "fake", "prompt": list(range(300)), "max_tokens": 4000, "stream": True}
            slow = asyncio.create_task(post(s, purl + "/v1/completions", body))
            await asyncio.sleep(0.05)
            st, _, _ = await post(s, purl + "/v1/completions", {**body, "max_tokens": 1})
            assert st == 429
            async with s.post(purl + "/admin/reset") as r:
                assert r.status == 409  # busy
            await slow
            async with s.post(purl + "/admin/reset") as r:
                assert r.status == 200
            async with s.get(purl + "/metrics") as r:
                text = await r.text()
            assert "pcrouter_routed_total" in text and "pcrouter_decision_seconds" in text
            async with s.get(purl + "/debug/state") as r:
                state = await r.json()
            assert state["policy"] == "affinity" and len(state["workers"]) == 2
            assert state["workers"][0]["kv_tokens"] == 50_000  # discovered from /metrics

    run(go())


def test_replay_client_against_proxy():
    async def go():
        async with cluster(2) as (_proxy, purl, *_):
            reqs = multiturn(
                6,
                session_rate=20.0,
                system_len=512,
                user_len=100,
                reply_len=8,
                think_s=0.05,
                seed=4,
            )
            res = await replay(purl, reqs, model="fake", num_workers=2)
            assert res.summary["failed"] == 0
            assert res.summary["measured"] == len(reqs)
            assert res.summary["hit_ratio"] > 0.2  # later turns reuse earlier ones
            assert set(res.workers) <= {0, 1}

    run(go())


def test_keys_and_helpers():
    ke = KeyExtractor(chunk_tokens=16)
    k = ke.extract("/v1/completions", {"prompt": list(range(40)), "max_tokens": 9})
    assert (k.input_len, k.max_tokens, len(k.chunks)) == (40, 9, 2)
    k = ke.extract("/v1/completions", {"prompt": [list(range(32)), [1]]})
    assert k.input_len == 32 and k.max_tokens == 256
    k = ke.extract("/v1/completions", {"prompt": ["a" * 128, "b"]})
    assert k.input_len == 32 and len(k.chunks) == 2
    k = ke.extract(
        "/v1/chat/completions",
        {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "x" * 64},
                        {"type": "image_url", "image_url": {"url": "u"}},
                    ],
                },
                {"role": "assistant", "content": None, "tool_calls": [{"id": "1"}]},
                "junk",
            ],
            "tools": [{"type": "function"}],
        },
    )
    assert k.input_len > 16
    assert render_chat({"messages": [{"role": "user", "content": {"a": 1}}]})
    ke_tok = KeyExtractor(16, tokenizer=lambda t: [ord(c) for c in t])
    assert ke_tok.extract("/v1/completions", {"prompt": "z" * 33}).tokens == [ord("z")] * 33
    with pytest.raises(ValueError):
        KeyExtractor(0)
    assert (
        parse_cache_config('vllm:cache_config_info{block_size="16",num_gpu_blocks="100"} 1') == 1600
    )
    assert parse_cache_config("nothing") is None
    assert parse_cache_config('vllm:cache_config_info{block_size="16"} 1') is None
    from pcrouter.prompt import Prompt

    toks = prompt_tokens(Prompt(40, segments=[(5, 0, 2)]), rid=1)
    assert len(toks) == 40
    assert toks[:32] == prompt_tokens(Prompt(32, segments=[(5, 0, 2)]), rid=2)
    with pytest.raises(ValueError):
        prompt_tokens(Prompt(16, keys=[1]), rid=0)


def test_cache_model_survives_short_outage_only():
    async def blip(reset_after_s: float) -> int:
        async with (
            cluster(2, policy="ttft", reset_after_s=reset_after_s) as (proxy, purl, _f, urls),
            aiohttp.ClientSession() as s,
        ):
            body = {"model": "fake", "prompt": list(range(5000, 6024)), "max_tokens": 2}
            _, w, _ = await post(s, purl + "/v1/completions", body)
            w = int(w)
            assert proxy.router.index.size(w) > 0
            proxy.workers[w].url = f"http://127.0.0.1:{free_port()}"  # health checks fail
            for _ in range(50):
                if not proxy.workers[w].healthy:
                    break
                await asyncio.sleep(0.05)
            assert not proxy.workers[w].healthy
            proxy.workers[w].url = urls[w]
            for _ in range(50):
                if proxy.workers[w].healthy:
                    break
                await asyncio.sleep(0.05)
            assert proxy.workers[w].healthy
            return proxy.router.index.size(w)

    assert run(blip(30.0)) > 0  # a blip: keep what the index knows
    assert run(blip(0.0)) == 0  # a real outage: the worker may have restarted


def test_admin_token():
    async def go():
        async with (
            cluster(1, admin_token="s3cret") as (_proxy, purl, *_),
            aiohttp.ClientSession() as s,
        ):
            body = {"policy": "round-robin"}
            async with s.post(purl + "/admin/policy", json=body) as r:
                assert r.status == 401
            hdr = {"Authorization": "Bearer s3cret"}
            async with s.post(purl + "/admin/policy", json=body, headers=hdr) as r:
                assert r.status == 200
            async with s.post(purl + "/admin/reset", headers={"Authorization": "Bearer x"}) as r:
                assert r.status == 401

    run(go())
