#!/usr/bin/env bash
# Run the remaining analyses back to back, so nothing needs babysitting and no
# two memory-hungry stages overlap on a 16GB box.
#
#   1. wait for the running base sweep shards, merge them
#   2. LOO phase, all 16 models, sharded      (gives caught@10 etc.)
#   3. CCS pair table, all 16 models          (unsupervised probe)
set -uo pipefail
cd "$(dirname "$0")/.."
log(){ echo "[$(date '+%H:%M:%S')] $*"; }

merge_sweep(){ python3 - "$1" <<'PY'
import csv, glob, sys, os
main=sys.argv[1]; cols=None; rows={}
key=("overseer","target","dataset") if "base" in main else ("overseer","target","held_out")
for f in [main]+sorted(glob.glob(main.replace(".csv","_s*.csv"))):
    if not os.path.exists(f): continue
    with open(f) as fh:
        r=csv.DictReader(fh)
        if r.fieldnames: cols=cols or r.fieldnames
        for row in r: rows[tuple(row[k] for k in key)]=row
with open(main,"w",newline="") as fh:
    w=csv.DictWriter(fh,fieldnames=cols); w.writeheader()
    for k in sorted(rows): w.writerow(rows[k])
print(f"merged -> {main}: {len(rows)} rows")
PY
}

log "waiting for base sweep shards"
while pgrep -f "sweep.py --phase base" >/dev/null; do sleep 30; done
log "base done; merging"
merge_sweep results/sweep_base_all.csv

log "LOO phase, 2 shards"
for i in 0 1; do
  lo=$((i*64)); hi=$(((i+1)*64))
  cp results/sweep_loo_all.csv "results/sweep_loo_all_s$i.csv"
  OMP_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 MKL_NUM_THREADS=3 \
    python3 scripts/sweep.py --phase loo --probe mass-mean \
      --tag "_all_s$i" --pairs "$lo:$hi" > "/tmp/loo_s$i.log" 2>&1 &
done
wait
log "LOO done; merging"
merge_sweep results/sweep_loo_all.csv

log "CCS pair table (unsupervised), 2 shards"
for i in 0 1; do
  lo=$((i*60)); hi=$(((i+1)*60))
  : > "results/pair_table_ccs_s$i.csv"
  OMP_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 MKL_NUM_THREADS=3 \
    python3 scripts/pair_table.py --probe ccs --tag "_ccs_s$i" \
      --pairs "$lo:$hi" > "/tmp/ccs_s$i.log" 2>&1 &
done
wait
log "CCS done; merging"
python3 - <<'PY'
import csv, glob, os
cols=None; rows={}
for f in sorted(glob.glob("results/pair_table_ccs_s*.csv")):
    if os.path.getsize(f)==0: continue
    with open(f) as fh:
        r=csv.DictReader(fh)
        if r.fieldnames: cols=cols or r.fieldnames
        for row in r: rows[(row["model_a"],row["model_b"],row["dataset"])]=row
if cols:
    with open("results/pair_table_ccs.csv","w",newline="") as fh:
        w=csv.DictWriter(fh,fieldnames=cols); w.writeheader()
        for k in sorted(rows): w.writerow(rows[k])
    print(f"merged -> results/pair_table_ccs.csv: {len(rows)} cells")
PY
log "ALL DONE"
