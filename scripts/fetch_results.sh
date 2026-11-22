#!/usr/bin/env bash
# Copy H200 run summaries (not per-request records) from the cluster into results/.
#   EXPLORER="user@login.explorer.northeastern.edu" SSH_KEY=~/.ssh/key bash scripts/fetch_results.sh
set -euo pipefail
cd "$(dirname "$0")/.."
: "${EXPLORER:?set EXPLORER=user@host}"
REMOTE=${REMOTE:-/scratch/${EXPLORER%%@*}/pcrouter/results}
SSH="ssh ${SSH_KEY:+-i $SSH_KEY}"
for run in h200 h200-v2; do
  rsync -az -e "$SSH" --include '*/' --include 'summary.json' --include 'engine-config.txt' \
    --include 'matrix.log' --exclude '*' "$EXPLORER:$REMOTE/$run/" "results/$run/" || true
done
# Drop cluster host names and ports from the copied summaries.
python3 - <<'PY'
import json, pathlib, re
for p in pathlib.Path("results").glob("h200*/*/*/summary.json"):
    d = json.loads(p.read_text())
    d.pop("router_state", None)
    d.pop("reset", None)
    p.write_text(re.sub(r"http://[\w.-]+:\d+", "http://engine", json.dumps(d, indent=1)))
PY
