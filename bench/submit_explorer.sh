#!/bin/bash
# Submit N one-GPU engine jobs plus the CPU driver job for one experiment run.
#   RUN=h200 bash bench/submit_explorer.sh
#   RUN=h200-tool TRACES=toolagent SCALES=4 POLICIES=affinity,ttft bash bench/submit_explorer.sh
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT=${PCR_ROOT:-/scratch/$USER/pcrouter}
RUN=${RUN:?set RUN}
N=${N:-4}
DIR=$ROOT/run/$RUN
mkdir -p "$DIR"
rm -f "$DIR"/engine*.url "$DIR"/done
ids=()
for i in $(seq 0 $((N - 1))); do
  ids+=("$(RUN=$RUN IDX=$i sbatch --parsable --export=ALL bench/explorer-engine.sbatch)")
done
drv=$(RUN=$RUN N=$N sbatch --parsable --export=ALL bench/explorer-driver.sbatch)
echo "run $RUN: engines ${ids[*]}, driver $drv"
