#!/usr/bin/env bash
# Re-run the mass-mean label-free scores so they carry every review budget.
#
# The first pass stored only the 10% budget. The file has to be removed rather
# than resumed: sweep.py appends without rewriting the header, so new rows with
# more columns would be written against the old 19-field header.
set -uo pipefail
cd "$(dirname "$0")/.."
log(){ echo "[$(date '+%H:%M:%S')] $*"; }

log "waiting for the logistic unsup run"
while pgrep -f "sweep.py --phase unsup .*--tag _lr" >/dev/null; do sleep 30; done

mv -f results/sweep_unsup_mm.csv results/sweep_unsup_mm_10only.csv
log "old 10%-only table kept as sweep_unsup_mm_10only.csv"

i=0
for rng in 0:64 64:128; do
  OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \
    python3 scripts/sweep.py --phase unsup --probe mass-mean \
      --tag "_mm_s$i" --pairs "$rng" > "/tmp/unsup_mm_redo_s$i.log" 2>&1 &
  i=$((i+1))
done
wait

python3 - <<'PY'
import csv, glob, os
cols=None; rows={}
for f in sorted(glob.glob("results/sweep_unsup_mm_s*.csv")):
    if not os.path.exists(f) or os.path.getsize(f)==0: continue
    with open(f) as fh:
        r=csv.DictReader(fh)
        if r.fieldnames: cols=cols or r.fieldnames
        for row in r: rows[(row["overseer"],row["target"],row["held_out"])]=row
if cols:
    with open("results/sweep_unsup_mm.csv","w",newline="") as fh:
        w=csv.DictWriter(fh,fieldnames=cols); w.writeheader()
        for k in sorted(rows): w.writerow(rows[k])
    print(f"-> results/sweep_unsup_mm.csv: {len(rows)} rows, all budgets")
PY
rm -f results/sweep_unsup_mm_s*.csv
log "DONE"
