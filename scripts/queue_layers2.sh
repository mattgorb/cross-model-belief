#!/usr/bin/env bash
# Fill the depth grid: every phase and both probes at 0.5 and 0.75.
#
# The first layer chain covered only in-domain/mass-mean and base/logistic.
# This adds the rest, so each depth has the same measurements as the final
# layer and nothing has to be compared across different setups.
#
# Cheapest first, so the most is salvaged if it is interrupted: the label-free
# scores need no fitting, the base phase is a closed-form probe fit, and the
# in-domain runs do cross-validated fits with permutation controls.
#
# Only the ten locally extracted models hold more than a final layer, so each
# run covers 49 pairs; sweep.py drops any pair it cannot read at the requested
# depth rather than mixing depths across a pair.
set -uo pipefail
cd "$(dirname "$0")/.."
log(){ echo "[$(date '+%H:%M:%S')] $*"; }

run(){ local phase="$1" probe="$2" depth="$3" tag="$4"
  if [ -f "results/sweep_${5}${tag}.csv" ]; then
    log "SKIP $phase/$probe d=$depth (already present)"; return
  fi
  log "START $phase/$probe d=$depth"
  OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \
    python3 scripts/sweep.py --phase "$phase" --probe "$probe" \
      --layer "$depth" --no-map-features --tag "$tag" \
      > "/tmp/layer${tag}.log" 2>&1
  log "DONE  $phase/$probe d=$depth -> sweep_${5}${tag}.csv"
}

log "waiting for the first layer chain"
while pgrep -f queue_layers.sh >/dev/null; do sleep 60; done
log "first chain clear"

# label-free scores: minutes each
run unsup    mass-mean 0.50 _d50_mm unsup
run unsup    mass-mean 0.75 _d75_mm unsup
run unsup    lr        0.50 _d50_lr unsup
run unsup    lr        0.75 _d75_lr unsup
# false-agreement decomposition for the second probe
run base     mass-mean 0.50 _d50_mm base
run base     mass-mean 0.75 _d75_mm base
# the expensive pair, last
run indomain lr        0.50 _d50_lr indomain
run indomain lr        0.75 _d75_lr indomain
log "DEPTH GRID COMPLETE"
