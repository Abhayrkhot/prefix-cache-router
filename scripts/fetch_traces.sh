#!/usr/bin/env bash
# Download the Mooncake (FAST'25) traces into data/ and verify their checksums.
set -euo pipefail
cd "$(dirname "$0")/../data"
BASE=https://raw.githubusercontent.com/kvcache-ai/Mooncake/main/FAST25-release/traces
for t in conversation toolagent; do
  f=mooncake_${t}_trace.jsonl
  [ -s "$f" ] || curl -fsSL "$BASE/${t}_trace.jsonl" -o "$f"
done
if command -v sha256sum >/dev/null; then sha256sum -c SHA256SUMS; else shasum -a 256 -c SHA256SUMS; fi
