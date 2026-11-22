"""Turn an OpenAI-style request body into what the router needs: an input-length estimate,
``max_tokens`` and prefix chunk keys.

* Token-id prompts (``/v1/completions`` with ``prompt: [int, ...]``) are hashed exactly, in
  ``chunk_tokens`` blocks.
* Text prompts are hashed in ``chunk_tokens * bytes_per_token`` byte chunks unless a tokenizer
  is configured, in which case they are tokenized first and hashed exactly.
* Chat requests are rendered to one canonical string (tools, then every message's role and
  content). Chat templates are prefix-stable, so two conversations share a rendered prefix
  exactly when their templated prompts share one, up to the template's last turn.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from pcrouter.prompt import text_block_keys, token_block_keys

__all__ = ["KeyExtractor", "RoutingKey", "render_chat"]


@dataclass(slots=True)
class RoutingKey:
    input_len: int
    max_tokens: int
    chunks: list[int]
    tokens: list[int] | None = None


def _content_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):  # content parts
        parts = []
        for p in content:
            if isinstance(p, dict):
                if p.get("type") == "text":
                    parts.append(str(p.get("text", "")))
                else:  # images/audio: their identity, not their bytes
                    parts.append(json.dumps(p, sort_keys=True)[:256])
            else:
                parts.append(str(p))
        return "".join(parts)
    return json.dumps(content, sort_keys=True)


def render_chat(body: dict[str, Any]) -> str:
    out = []
    tools = body.get("tools")
    if tools:
        out.append("<tools>" + json.dumps(tools, sort_keys=True) + "</tools>")
    for m in body.get("messages", []) or []:
        if not isinstance(m, dict):
            continue
        out.append(f"<{m.get('role', '')}>")
        out.append(_content_text(m.get("content")))
        if m.get("tool_calls"):
            out.append(json.dumps(m["tool_calls"], sort_keys=True))
        out.append("</>")
    return "".join(out)


class KeyExtractor:
    def __init__(
        self,
        chunk_tokens: int,
        bytes_per_token: float = 4.0,
        tokenizer: Callable[[str], Sequence[int]] | None = None,
        default_max_tokens: int = 256,
    ) -> None:
        if chunk_tokens < 1 or bytes_per_token <= 0:
            raise ValueError("chunk_tokens and bytes_per_token must be positive")
        self.chunk_tokens = chunk_tokens
        self.bytes_per_token = bytes_per_token
        self.tokenizer = tokenizer
        self.default_max_tokens = default_max_tokens

    def _from_tokens(self, tokens: list[int]) -> tuple[int, list[int]]:
        return len(tokens), token_block_keys(tokens, self.chunk_tokens)

    def _from_text(self, text: str) -> tuple[int, list[int], list[int] | None]:
        if self.tokenizer is not None:
            toks = list(self.tokenizer(text))
            n, keys = self._from_tokens(toks)
            return n, keys, toks
        data_len = len(text.encode("utf-8"))
        n = max(1, round(data_len / self.bytes_per_token))
        chunk_bytes = max(1, round(self.chunk_tokens * self.bytes_per_token))
        return n, text_block_keys(text, chunk_bytes), None

    def extract(self, path: str, body: dict[str, Any]) -> RoutingKey:
        max_tokens = body.get("max_tokens") or body.get("max_completion_tokens")
        if not isinstance(max_tokens, int) or max_tokens < 1:
            max_tokens = self.default_max_tokens
        if path.endswith("/chat/completions"):
            n, keys, toks = self._from_text(render_chat(body))
            return RoutingKey(n, max_tokens, keys, toks)
        prompt = body.get("prompt", "")
        if isinstance(prompt, list) and prompt:
            first = prompt[0]
            if isinstance(first, int):
                toks = [int(t) for t in prompt]
                n, keys = self._from_tokens(toks)
                return RoutingKey(n, max_tokens, keys, toks)
            if isinstance(first, list):  # batch of token lists: route by the first
                toks = [int(t) for t in first]
                n, keys = self._from_tokens(toks)
                return RoutingKey(n, max_tokens, keys, toks)
            prompt = str(first)  # batch of strings: route by the first
        n, keys, toks = self._from_text(str(prompt))
        return RoutingKey(n, max_tokens, keys, toks)
