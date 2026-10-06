#!/usr/bin/env bash
# Paper 1's detection experiments: 16 models, NO linear map, three probes.
#
# The map is out of Paper 1 -- measured, not assumed: on the same 228 cells it
# cost 0.012 detector AUROC (paired t=5.9) and lost on every held-out dataset.
# So every run here passes --no-map-features and the two halves of the detector's
# features stay in their own coordinates.
#
# Two shards wide: four exhausted memory on this 16GB machine. Each stage resumes
# from its merged output, so rerunning after any interruption is safe.
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

run(){ # probe, out
  local probe="$1" out="$2"
  local base; base=$(basename "${out%.csv}" | sed 's/^sweep_loo//')
  log "START $(basename "$out")  probe=$probe"
  local i=0
  for rng in 0:64 64:128; do
    [ -f "$out" ] && cp "$out" "${out%.csv}_s$i.csv"
    OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \
      python3 scripts/sweep.py --phase loo --probe "$probe" --no-map-features \
        --tag "${base}_s$i" --pairs "$rng" > "/tmp/sweep_loo${base}_s$i.log" 2>&1 &
    i=$((i+1))
  done
  wait
  merge "$out"
  log "DONE $(basename "$out")"
}

run lr        results/sweep_loo_nomap16_lr.csv
run mass-mean results/sweep_loo_nomap16_mm.csv
# CCS dropped: measured at chance on every base model (0.527-0.548) and on
# gemma4-12b (0.478-0.499), and it is the one job with no bounded runtime --
# 1000 epochs x 10 restarts per fit against LR's closed form. One sentence in the
# paper covers it; run `scripts/sweep.py --phase loo --probe ccs
# --no-map-features` separately if it is ever wanted.
# Contributions 1 and 2 currently rest on sweep_base_all.csv, which is mass-mean
# only, while contribution 3 is LR. The decomposition terms rho / coverage /
# p1 / p2 exist nowhere else -- pair_table.csv is LR but carries none of them --
# so without this the paper mixes probes across its own contributions. Running it
# also gives probe robustness on the headline null rather than a single probe.
log "START base sweep, LR (contributions 1 and 2 on the headline probe)"
i=0
for rng in 0:64 64:128; do
  [ -f results/sweep_base_lr.csv ] && cp results/sweep_base_lr.csv "results/sweep_base_lr_s$i.csv"
  OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \
    python3 scripts/sweep.py --phase base --probe lr \
      --tag "_lr_s$i" --pairs "$rng" > "/tmp/sweep_base_lr_s$i.log" 2>&1 &
  i=$((i+1))
done
wait
python3 - <<'PYMERGE'
import csv, glob, os
cols=None; rows={}
for f in ["results/sweep_base_lr.csv"]+sorted(glob.glob("results/sweep_base_lr_s*.csv")):
    if not os.path.exists(f) or os.path.getsize(f)==0: continue
    with open(f) as fh:
        r=csv.DictReader(fh)
        if r.fieldnames: cols=cols or r.fieldnames
        for row in r: rows[(row["overseer"],row["target"],row["dataset"])]=row
if cols:
    with open("results/sweep_base_lr.csv","w",newline="") as fh:
        w=csv.DictWriter(fh,fieldnames=cols); w.writeheader()
        for k in sorted(rows): w.writerow(rows[k])
    print(f"merged -> results/sweep_base_lr.csv: {len(rows)} rows")
PYMERGE
rm -f results/sweep_base_lr_s*.csv
log "base sweep LR done"

log "ALL DONE"
