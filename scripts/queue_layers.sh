#!/usr/bin/env bash
# Probe depth sweep: does reading the probe from an earlier layer help?
#
# Truth directions are often reported as strongest in middle layers, and every
# result in the paper so far reads the final layer. This repeats the two
# measurements that matter at 0.5 and 0.75 of model depth.
#
# Only the ten locally extracted models hold more than a final layer -- the
# remote extraction saved one layer to keep the transfers small -- so these runs
# cover 45 pairs, not 128, and sweep.py drops any pair it cannot read at the
# requested depth rather than mixing depths across a pair.
#
# One shard, deliberately: the CCS sweep is running alongside, and four
# concurrent sweeps exhausted memory on this machine once already.
set -uo pipefail
cd "$(dirname "$0")/.."
log(){ echo "[$(date '+%H:%M:%S')] $*"; }

run(){ # phase probe depth tag
  local phase="$1" probe="$2" depth="$3" tag="$4"
  log "START $phase/$probe at depth $depth"
  OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \
    python3 scripts/sweep.py --phase "$phase" --probe "$probe" \
      --layer "$depth" --no-map-features --tag "$tag" \
      > "/tmp/layer${tag}.log" 2>&1
  log "DONE $phase/$probe at depth $depth"
}

# detection first: this is where an earlier layer is most likely to matter, and
# mass-mean is the better detector of the two
run indomain mass-mean 0.50 _d50_mm
run indomain mass-mean 0.75 _d75_mm
# then the headline null, to check it is not an artifact of the final layer
run base lr 0.50 _d50_lr
run base lr 0.75 _d75_lr
log "LAYER SWEEP DONE"
