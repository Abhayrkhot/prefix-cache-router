#!/bin/bash
# One vLLM 0.34 replica of Qwen2.5-7B-Instruct on one H200. Publishes its URL to
# $ROOT/run/$RUN/engine$IDX.url once healthy and serves until the driver job writes
# $ROOT/run/$RUN/done. Submitted 4x by bench/submit_explorer.sh (the gpu QOS allows one GPU
# per job), so replicas may land on different nodes and the router reaches them over the
# cluster network, as in a real deployment.
#
#SBATCH --job-name=pcr-engine
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=16G
#SBATCH --time=07:45:00
#SBATCH --output=%x-%j.log

set -uo pipefail
ROOT=${PCR_ROOT:-/scratch/$USER/pcrouter}
RUN=${RUN:?set RUN}
IDX=${IDX:?set IDX}
SIF=${VLLM_SIF:-/scratch/$USER/icp/vllm.sif}
MODEL=${MODEL:-Qwen/Qwen2.5-7B-Instruct}
DIR=$ROOT/run/$RUN
mkdir -p "$DIR" "$ROOT/cache"
LOG=$DIR/engine$IDX.log

export HF_HOME=${HF_HOME:-/scratch/$USER/hf} HF_HUB_OFFLINE=1 PYTHONNOUSERSITE=1
export XDG_CACHE_HOME=$ROOT/cache TRITON_CACHE_DIR=$ROOT/cache/triton VLLM_CACHE_ROOT=$ROOT/cache/vllm
export no_proxy='*' NO_PROXY='*'
export VLLM_SERVER_DEV_MODE=1  # enables /reset_prefix_cache
module load apptainer 2>/dev/null || true

PORT=$(python3 -c 'import socket; s=socket.socket(); s.bind(("",0)); print(s.getsockname()[1])')
HOST=$(hostname -f 2>/dev/null || hostname)
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader > "$DIR/engine$IDX.gpu"

# setsid: the engine and its workers form one process group, killed exactly on exit.
# Never a node-wide pkill: H200 nodes are shared with other jobs.
setsid apptainer exec --nv --bind "/scratch/$USER" "$SIF" bash -c '
  export LD_LIBRARY_PATH=/usr/local/cuda/compat${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
  exec vllm serve "$0" --host 0.0.0.0 --port "$1" --served-model-name m --max-model-len 32768 \
    --gpu-memory-utilization 0.9 --enable-prompt-tokens-details' "$MODEL" "$PORT" >"$LOG" 2>&1 &
PG=$!
trap 'kill -TERM -- "-$PG" 2>/dev/null; sleep 10; kill -KILL -- "-$PG" 2>/dev/null' EXIT

t0=$(date +%s)
until curl -sf "http://127.0.0.1:$PORT/health" >/dev/null; do
  if ! kill -0 "$PG" 2>/dev/null; then echo "engine died"; tail -40 "$LOG"; exit 1; fi
  if (( $(date +%s) - t0 > 1500 )); then echo "engine not ready"; exit 1; fi
  sleep 5
done
echo "ready after $(( $(date +%s) - t0 ))s: $(grep -oE 'GPU KV cache size: [0-9,]+ tokens' "$LOG" | tail -1)"
echo "http://$HOST:$PORT" > "$DIR/engine$IDX.url.tmp" && mv "$DIR/engine$IDX.url.tmp" "$DIR/engine$IDX.url"

while [ ! -f "$DIR/done" ]; do
  if ! kill -0 "$PG" 2>/dev/null; then echo "engine died while serving"; tail -40 "$LOG"; exit 1; fi
  sleep 20
done
echo "driver finished; shutting down"
