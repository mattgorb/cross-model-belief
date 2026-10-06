#!/usr/bin/env bash
# Stage 3: lift the scalable-oversight size filter and retest the family null
# with large models overseeing small ones.
#
# The standing filter keeps overseer <= target, so the same/cross-family
# contrast is measured on a slice of the pair space. This adds the other 112
# ordered pairs (531 new cells) so the null can be checked in both directions.
# Seeded from the existing base results, so nothing already computed is redone.
set -uo pipefail
cd "$(dirname "$0")/.."
log(){ echo "[$(date '+%H:%M:%S')] $*"; }

log "waiting for stage 2"
while pgrep -f "queue_overnight.sh" >/dev/null; do sleep 60; done
log "stage 2 clear; starting reverse-direction base sweep"

for i in 0 1; do
  lo=$((i*120)); hi=$(((i+1)*120))
  cp results/sweep_base_all.csv "results/sweep_base_rev_s$i.csv"
  OMP_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 MKL_NUM_THREADS=3 \
    python3 scripts/sweep.py --phase base --probe mass-mean --all-directions \
      --tag "_rev_s$i" --pairs "$lo:$hi" > "/tmp/rev_s$i.log" 2>&1 &
done
wait

log "merging"
python3 - <<'PY'
import csv, glob, os
cols=None; rows={}
for f in ["results/sweep_base_all.csv"]+sorted(glob.glob("results/sweep_base_rev_s*.csv")):
    if not os.path.exists(f) or os.path.getsize(f)==0: continue
    with open(f) as fh:
        r=csv.DictReader(fh)
        if r.fieldnames: cols=cols or r.fieldnames
        for row in r: rows[(row["overseer"],row["target"],row["dataset"])]=row
with open("results/sweep_base_both_directions.csv","w",newline="") as fh:
    w=csv.DictWriter(fh,fieldnames=cols); w.writeheader()
    for k in sorted(rows): w.writerow(rows[k])
print(f"-> results/sweep_base_both_directions.csv: {len(rows)} cells")
PY
log "STAGE 3 DONE"
