#!/usr/bin/env bash
# C: label-free detection. Supervised probes, unsupervised detector.
#
# The supervised detector needs ground-truth false agreements on the
# distribution being overseen, which a deployment does not have. These scores
# need none -- they come from the probes' own outputs or from the geometry of
# the unlabelled both-agree pool -- and labels are used only to score them.
#
# Cheap: no gradient descent and no cross-validation, just scoring an existing
# pool, so this is a different order of cost from the CCS sweep it replaces.
set -uo pipefail
cd "$(dirname "$0")/.."
log(){ echo "[$(date '+%H:%M:%S')] $*"; }

run(){ local probe="$1" tag="$2"
  log "START unsup / $probe"
  local i=0
  for rng in 0:64 64:128; do
    [ -f "results/sweep_unsup${tag}.csv" ] && \
      cp "results/sweep_unsup${tag}.csv" "results/sweep_unsup${tag}_s$i.csv"
    OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \
      python3 scripts/sweep.py --phase unsup --probe "$probe" \
        --tag "${tag}_s$i" --pairs "$rng" > "/tmp/unsup${tag}_s$i.log" 2>&1 &
    i=$((i+1))
  done
  wait
  python3 - "results/sweep_unsup${tag}.csv" <<'PY'
import csv, glob, sys, os
main=sys.argv[1]; cols=None; rows={}
for f in [main]+sorted(glob.glob(main.replace(".csv","_s*.csv"))):
    if not os.path.exists(f) or os.path.getsize(f)==0: continue
    with open(f) as fh:
        r=csv.DictReader(fh)
        if r.fieldnames: cols=cols or r.fieldnames
        for row in r: rows[(row["overseer"],row["target"],row["held_out"])]=row
if cols:
    with open(main,"w",newline="") as fh:
        w=csv.DictWriter(fh,fieldnames=cols); w.writeheader()
        for k in sorted(rows): w.writerow(rows[k])
    print(f"merged -> {main}: {len(rows)} rows")
PY
  rm -f "results/sweep_unsup${tag}"_s*.csv
  log "DONE unsup / $probe"
}

run mass-mean _mm
run lr        _lr
log "C DONE"
