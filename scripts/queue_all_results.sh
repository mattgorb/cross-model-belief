#!/usr/bin/env bash
# Every remaining Paper 1 run, both probes, both directions.
#
# Four jobs. The two LOO runs give contribution 3; the two base runs give
# contributions 1 and 2. Each probe appears in both, so every claim can be stated
# as holding across probes rather than resting on one estimator.
#
# The base runs pass --all-directions, which is a superset of the default
# overseer<=target filter: one pass per probe covers the scalable-oversight
# direction AND the reverse (large model checking small), so the size-filter
# ablation needs no separate job. 1142 cells per probe, of which the mass-mean
# run already has 611 seeded from sweep_base_all.csv.
#
# No linear map anywhere: measured on 228 matched cells, the map cost 0.012
# detector AUROC (paired t=5.90) and lost on all five datasets. Map results live
# in results/paper2/.
#
# Two shards wide -- four exhausted memory on this 16GB machine. Every stage
# resumes from its merged output, so rerunning after an interruption is safe.
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
rm -f ${1%.csv}_s*.csv
}

run(){ # phase probe out key tag extra...
  local phase="$1" probe="$2" out="$3" key="$4" tag="$5"; shift 5
  log "START $(basename "$out")  phase=$phase probe=$probe $*"
  local i=0
  for rng in 0:120 120:240; do
    [ -f "$out" ] && cp "$out" "${out%.csv}_s$i.csv"
    OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \
      python3 scripts/sweep.py --phase "$phase" --probe "$probe" "$@" \
        --tag "${tag}_s$i" --pairs "$rng" > "/tmp/$(basename ${out%.csv})_s$i.log" 2>&1 &
    i=$((i+1))
  done
  wait
  merge "$out" "$key"
  log "DONE $(basename "$out")"
}

LOOK=overseer,target,held_out
BASEK=overseer,target,dataset

log "waiting for the running no-map LR LOO shards"
while pgrep -f "sweep.py --phase loo .*--tag _nomap16_lr" >/dev/null; do sleep 45; done
merge results/sweep_loo_nomap16_lr.csv $LOOK
log "LOO no-map LR complete"

# Base runs first: they carry contributions 1 and 2, which have no fallback
# table. Contribution 3 already has a publishable version in
# results/sweep_loo_nomap.csv (228 cells, 10 models, no map, mass-mean), so the
# LOO runs widen a result that already exists.
run base lr        results/sweep_base_all_lr.csv $BASEK _all_lr --all-directions
run base mass-mean results/sweep_base_all_mm.csv $BASEK _all_mm --all-directions

# Last, and expected to be incomplete: the 16-model mass-mean LOO. Its 10-model
# counterpart (sweep_loo_nomap.csv) already covers probe robustness for
# contribution 3, so whatever this finishes is a bonus. It writes cells as it
# goes and resumes from its merged output, so a partial table is still usable --
# just report the cell count it actually reached.
run loo mass-mean results/sweep_loo_nomap16_mm.csv $LOOK _nomap16_mm --no-map-features
log "ALL DONE"
