#!/usr/bin/env bash
# Remaining analysis chain, two shards wide.
#
# Width is the point: four concurrent shards exhausted memory on this 16GB
# machine and had to be killed. Two is what fits alongside normal desktop use.
# Every stage resumes from its merged output, so a kill costs at most the cells
# in flight -- rerunning this script after any interruption is safe.
set -uo pipefail
cd "$(dirname "$0")/.."
log(){ echo "[$(date '+%H:%M:%S')] $*"; }

merge(){ python3 - "$1" "$2" <<'PY'
import csv, glob, sys, os
main, key = sys.argv[1], tuple(sys.argv[2].split(","))
cols=None; rows={}
for f in [main]+sorted(glob.glob(main.replace(".csv","_s*.csv"))):
    if not os.path.exists(f) or os.path.getsize(f)==0: continue
    with open(f) as fh:
        r=csv.DictReader(fh)
        if r.fieldnames: cols=cols or r.fieldnames
        for row in r: rows[tuple(row[k] for k in key)]=row
if cols:
    with open(main,"w",newline="") as fh:
        w=csv.DictWriter(fh,fieldnames=cols); w.writeheader()
        for k in sorted(rows): w.writerow(rows[k])
    print(f"merged -> {main}: {len(rows)} rows")
PY
for f in $(ls ${1%.csv}_s*.csv 2>/dev/null); do rm -f "$f"; done
}

run(){ # phase probe out key ranges... (two ranges)
  local phase="$1" probe="$2" out="$3" key="$4" r0="$5" r1="$6"; shift 6
  local base; base=$(basename "${out%.csv}" | sed 's/^sweep_loo//;s/^sweep_base//')
  log "START $(basename "$out")  ranges $r0 $r1"
  local i=0
  for rng in "$r0" "$r1"; do
    [ -f "$out" ] && cp "$out" "${out%.csv}_s$i.csv"
    OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \
      python3 scripts/sweep.py --phase "$phase" --probe "$probe" "$@" \
        --tag "${base}_s$i" --pairs "$rng" > "/tmp/${base}_s$i.log" 2>&1 &
    i=$((i+1))
  done
  wait
  merge "$out" "$key"
  log "DONE $(basename "$out")"
}

LOOK=overseer,target,held_out
BASEK=overseer,target,dataset

run loo lr   results/sweep_loo_lr16.csv    $LOOK  47:111 111:128
run loo lr   results/sweep_loo_nomap16.csv $LOOK  0:64   64:128  --no-map-features
run base mass-mean results/sweep_base_rev.csv $BASEK 0:120 120:240 --all-directions
log "ALL DONE (CCS not included -- run scripts/pair_table.py --probe ccs separately if wanted)"
