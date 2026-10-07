#!/usr/bin/env bash
# In-domain detection: fit and score the false-agreement direction inside one
# dataset, by 5-fold CV over its held-out rows. Separates "is the blind spot
# linearly detectable" from "does the direction transfer across domains" -- the
# LOO phase conflates the two. No map, both probes, two shards.
set -uo pipefail
cd "$(dirname "$0")/.."
log(){ echo "[$(date '+%H:%M:%S')] $*"; }

merge(){ python3 - "$1" <<'PY'
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
rm -f ${1%.csv}_s*.csv
}

run(){ local probe="$1" out="$2" tag="$3"
  log "START $(basename "$out") probe=$probe"
  local i=0
  for rng in 0:64 64:128; do
    [ -f "$out" ] && cp "$out" "${out%.csv}_s$i.csv"
    OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \
      python3 scripts/sweep.py --phase indomain --probe "$probe" \
        --tag "${tag}_s$i" --pairs "$rng" > "/tmp/indomain${tag}_s$i.log" 2>&1 &
    i=$((i+1))
  done
  wait
  merge "$out"
  log "DONE $(basename "$out")"
}

run mass-mean results/sweep_indomain_mm.csv _mm
run lr        results/sweep_indomain_lr.csv _lr
log "ALL DONE"
