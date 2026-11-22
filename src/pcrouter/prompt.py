"""Prompts as sequences of prefix-chained block keys.

A KV-cache block can be reused only when *every* token before it is identical too, so the
key of block ``i`` must identify the whole prefix ``[0, (i+1)*block)``. vLLM gets this by
hashing each block together with its parent's hash; we do the same.

Two representations share one interface:

* :class:`Prompt` with *segments* ``(stream, start, n)`` — used by the simulator. A stream is
  an append-only token stream (a system prompt, a conversation, one Mooncake 512-token
  block...). Block ``start + j`` of a stream gets key ``stream << 20 | (start + j)``. Because a
  stream is append-only and its id is unique to its prefix, equal keys imply equal prefixes,
  which is exactly the chained-hash property, without storing per-block keys.
* :class:`Prompt` with explicit *keys* — produced by :func:`token_block_keys` /
  :func:`text_block_keys` from real token ids or text in the live router and fake worker.

``chunk_keys(k)`` gives the router's coarser view: the key of every ``k``-th block. Since
block keys are chained, the last block key of a chunk names the chunk *and* its whole prefix.
"""

from __future__ import annotations

import hashlib
import sys
from array import array
from collections.abc import Sequence
from itertools import islice

__all__ = [
    "BLOCK",
    "STREAM_SHIFT",
    "Prompt",
    "text_block_keys",
    "token_block_keys",
]

#: Engine block size in tokens (vLLM's default on CUDA). Segment prompts are in these units.
BLOCK = 16
#: Bits reserved for the block index inside a stream (16M blocks = 268M tokens per stream).
STREAM_SHIFT = 20
_MASK63 = (1 << 63) - 1


class Prompt:
    """A prompt of ``num_tokens`` tokens, described by block keys (see module docstring)."""

    __slots__ = ("_keys", "num_tokens", "segments")

    def __init__(
        self,
        num_tokens: int,
        segments: Sequence[tuple[int, int, int]] | None = None,
        keys: Sequence[int] | None = None,
    ) -> None:
        if num_tokens < 1:
            raise ValueError("a prompt needs at least one token")
        if (segments is None) == (keys is None):
            raise ValueError("give exactly one of segments= or keys=")
        self.num_tokens = num_tokens
        self.segments: tuple[tuple[int, int, int], ...] | None = (
            tuple(segments) if segments is not None else None
        )
        self._keys: list[int] | None = list(keys) if keys is not None else None

    def num_full_blocks(self, block_size: int = BLOCK) -> int:
        return self.num_tokens // block_size

    def block_keys(self, block_size: int = BLOCK) -> list[int]:
        """Keys of the full blocks, in order. Segment prompts are always in ``BLOCK`` units."""
        n = self.num_tokens // block_size
        if self._keys is not None:
            return self._keys[:n]
        if block_size != BLOCK:
            raise ValueError(f"segment prompts use {BLOCK}-token blocks, not {block_size}")
        out: list[int] = []
        assert self.segments is not None
        for stream, start, count in self.segments:
            if len(out) >= n:
                break
            base = stream << STREAM_SHIFT
            out.extend(range(base + start, base + start + count))
        del out[n:]
        return out

    def chunk_keys(self, blocks_per_chunk: int, block_size: int = BLOCK) -> list[int]:
        """Router view: one key per full chunk of ``blocks_per_chunk`` blocks."""
        if blocks_per_chunk < 1:
            raise ValueError("blocks_per_chunk must be >= 1")
        keys = self.block_keys(block_size)
        return list(islice(keys, blocks_per_chunk - 1, None, blocks_per_chunk))

    def __repr__(self) -> str:
        kind = "segments" if self.segments is not None else "keys"
        return f"Prompt(num_tokens={self.num_tokens}, {kind})"


def _digest(parent: int, payload: bytes) -> int:
    h = hashlib.blake2b(payload, digest_size=8, key=parent.to_bytes(8, "little"))
    return int.from_bytes(h.digest(), "little") & _MASK63


def token_block_keys(tokens: Sequence[int], block_size: int) -> list[int]:
    """Chained 63-bit keys of every full ``block_size``-token block of ``tokens``."""
    n = len(tokens) // block_size
    if n == 0:
        return []
    try:
        arr = array("I", tokens[: n * block_size])
    except OverflowError:  # negative or >= 2**32 ids: hash their low 32 bits
        arr = array("I", (t & 0xFFFFFFFF for t in tokens[: n * block_size]))
    if sys.byteorder == "big":  # pragma: no cover - keys are defined little-endian
        arr.byteswap()
    packed = arr.tobytes()
    step = arr.itemsize * block_size
    keys: list[int] = []
    parent = 0
    for i in range(n):
        parent = _digest(parent, packed[i * step : (i + 1) * step])
        keys.append(parent)
    return keys


def text_block_keys(text: str, chunk_bytes: int) -> list[int]:
    """Chained keys of every full ``chunk_bytes``-byte chunk of the UTF-8 text.

    Tokenizer-free approximation for routing text prompts: two prompts share their first
    ``k`` chunk keys iff their first ``k*chunk_bytes`` bytes are identical.
    """
    data = text.encode("utf-8")
    keys: list[int] = []
    parent = 0
    for i in range(len(data) // chunk_bytes):
        parent = _digest(parent, data[i * chunk_bytes : (i + 1) * chunk_bytes])
        keys.append(parent)
    return keys
