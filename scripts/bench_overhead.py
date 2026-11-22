"""Router CPU cost per request: chunk hashing + routing decision, by prompt size, cluster
size and policy. Writes results/overhead.json.

    python scripts/bench_overhead.py
"""

from __future__ import annotations

import json
import platform
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from pcrouter.costmodel import load_step_model  # noqa: E402
from pcrouter.policies import make_policy  # noqa: E402
from pcrouter.prompt import token_block_keys  # noqa: E402
from pcrouter.router import Router, RouterConfig  # noqa: E402


def cpu_name() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def main() -> None:
    model = load_step_model("h200-qwen2.5-7b")
    rng = random.Random(0)
    out: dict[str, object] = {"hash_us": {}, "route_us": {}}
    for n_tok in (1024, 8192, 32768):
        toks = [rng.randrange(150_000) for _ in range(n_tok)]
        reps = 200
        t = time.perf_counter()
        for _ in range(reps):
            token_block_keys(toks, 512)
        out["hash_us"][str(n_tok)] = (time.perf_counter() - t) / reps * 1e6  # type: ignore[index]
    # Routing: a warm index (every worker full of other prefixes), prompts sharing a prefix.
    for workers in (4, 16, 64):
        for policy in ("round-robin", "affinity", "ttft"):
            r = Router(
                make_policy(policy, workers, model=model),
                RouterConfig(workers, chunk_tokens=512, capacity_tokens=2_022_400),
                model,
            )
            for _ in range(workers * 400):
                r.finish(r.route(16384, 100, [rng.randrange(1 << 60) for _ in range(32)]))
            shared = [rng.randrange(1 << 60) for _ in range(40)]
            reqs = [
                shared[: rng.randrange(1, 40)] + [rng.randrange(1 << 60) for _ in range(20)]
                for _ in range(2000)
            ]
            t = time.perf_counter()
            for chunks in reqs:
                tk = r.route(len(chunks) * 512, 100, chunks)
                r.first_token(tk, 0)
                r.finish(tk)
            us = (time.perf_counter() - t) / len(reqs) * 1e6
            out["route_us"][f"{policy}/{workers}"] = us  # type: ignore[index]
    out["python"] = sys.version.split()[0]
    out["cpu"] = cpu_name()
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "overhead.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
