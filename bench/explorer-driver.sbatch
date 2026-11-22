#!/bin/bash
# CPU job: waits for the engine jobs' URLs, runs pcrouter in front of them and replays the
# policy x workload matrix (bench/run_matrix.py). Writes $ROOT/run/$RUN/done at the end so the
# engine jobs exit. Router and client run inside the vLLM image (aiohttp, numpy,
# prometheus_client, uvloop are already there), from the repo's src/.
#
#SBATCH --job-name=pcr-driver
#SBATCH --partition=short
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=09:00:00
#SBATCH --output=%x-%j.log

set -uo pipefail
ROOT=${PCR_ROOT:-/scratch/$USER/pcrouter}
REPO=$ROOT/repo
RUN=${RUN:?set RUN}
N=${N:-4}
SIF=${VLLM_SIF:-/scratch/$USER/icp/vllm.sif}
TRACES=${TRACES:-conversation,toolagent}
SCALES=${SCALES:-4,5}
DURATION=${DURATION:-1800}
POLICIES=${POLICIES:-round-robin,least-request,chwbl,legacy,threshold,affinity,ttft}
DIR=$ROOT/run/$RUN
OUT=${OUT:-$ROOT/results/$RUN}
mkdir -p "$DIR" "$OUT"
finish() { rm -rf "$DIR/done"; touch "$DIR/done"; }  # tells the engine jobs to exit
trap finish EXIT
export no_proxy='*' NO_PROXY='*' PYTHONNOUSERSITE=1
module load apptainer 2>/dev/null || true

in_image() {
  apptainer exec --bind "/scratch/$USER" "$SIF" "$@"
}

echo "waiting for $N engines in $DIR"
t0=$(date +%s)
while (( $(ls "$DIR"/engine*.url 2>/dev/null | wc -l) < N )); do
  if (( $(date +%s) - t0 > 4 * 3600 )); then echo "engines never all came up"; exit 1; fi
  sleep 15
done
ENGINES=$(cat "$DIR"/engine*.url | sort | paste -sd, -)
echo "engines: $ENGINES (waited $(( $(date +%s) - t0 ))s)"

RPORT=$(python3 -c 'import socket; s=socket.socket(); s.bind(("",0)); print(s.getsockname()[1])')
ROUTER=http://127.0.0.1:$RPORT
WARGS=$(echo "$ENGINES" | tr , '\n' | sed 's/^/--worker /' | paste -sd' ' -)
# shellcheck disable=SC2086  # WARGS is intentionally word-split
setsid apptainer exec --bind "/scratch/$USER" --env PYTHONPATH="$REPO/src" "$SIF" \
  python3 -m pcrouter.cli serve --host 127.0.0.1 --port "$RPORT" $WARGS --policy round-robin \
  --chunk-tokens 512 > "$OUT/router-${SLURM_JOB_ID:-local}.log" 2>&1 &
RPG=$!
trap 'kill -TERM -- "-$RPG" 2>/dev/null; finish' EXIT
until curl -sf "$ROUTER/health" >/dev/null; do sleep 2; done
curl -s "$ROUTER/debug/state" | tee "$OUT/router-state.json"; echo
for f in "$DIR"/engine*.gpu; do echo "$f: $(cat "$f")"; done
grep -hoE "GPU KV cache size: [0-9,]+ tokens|max_num_batched_tokens=[0-9]+|max_num_seqs=[0-9]+" \
  "$DIR"/engine*.log | sort | uniq -c | tee "$OUT/engine-config.txt"
in_image python3 -c 'import vllm; print("vllm", vllm.__version__)' 2>/dev/null | tee -a "$OUT/engine-config.txt"

TRACE_ARGS=()
for t in ${TRACES//,/ }; do TRACE_ARGS+=(--trace "$REPO/data/mooncake_${t}_trace.jsonl"); done
cd "$REPO" && apptainer exec --bind "/scratch/$USER" --env PYTHONPATH="$REPO/src" "$SIF" \
  python3 bench/run_matrix.py --router "$ROUTER" --engines "$ENGINES" "${TRACE_ARGS[@]}" \
  --scales "$SCALES" --duration-s "$DURATION" --policies "$POLICIES" --out "$OUT" \
  2>&1 | tee -a "$OUT/matrix.log"
echo "done: $(date)"
