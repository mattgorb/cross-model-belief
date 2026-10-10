#!/usr/bin/env bash
# Leave-one-dataset-out at 0.5 and 0.75, both probes.
#
# The depth grid otherwise has every phase except transfer, and transfer is half
# of what the detection result claims -- a signature that is strong in-domain but
# does not travel is a different finding from one that does. Without this, an
# earlier layer could only be compared to the final layer on the in-domain half.
#
# Last and separate because it is the most expensive thing queued: cross-validated
# fits with permutation controls over a four-dataset pool, roughly two hours per
# run, four runs.
set -uo pipefail
cd "$(dirname "$0")/.."
log(){ echo "[$(date '+%H:%M:%S')] $*"; }

run(){ local probe="$1" depth="$2" tag="$3"
  if [ -f "results/sweep_loo${tag}.csv" ]; then
    log "SKIP loo/$probe d=$depth (already present)"; return
  fi
  log "START loo/$probe d=$depth"
  OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \
    python3 scripts/sweep.py --phase loo --probe "$probe" \
      --layer "$depth" --no-map-features --tag "$tag" \
      > "/tmp/layer_loo${tag}.log" 2>&1
  log "DONE  loo/$probe d=$depth -> sweep_loo${tag}.csv"
}

log "waiting for the depth grid"
while pgrep -f queue_layers2.sh >/dev/null || pgrep -f queue_layers.sh >/dev/null; do
  sleep 60
done
log "depth grid clear"

run mass-mean 0.50 _d50_mm
run mass-mean 0.75 _d75_mm
run lr        0.50 _d50_lr
run lr        0.75 _d75_lr
log "LOO DEPTH SWEEP COMPLETE"
