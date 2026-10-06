#!/usr/bin/env bash
# Stage 4: CCS (unsupervised) pair table, after the sweeps are clear.
#
# The first attempt failed every cell: pair_table.py fits probes through one
# interface, fit(Xp, Xn, labels), but CCSProbe.fit took `epochs` third, so the
# label array became an epoch count. Fixed in cmb/probes.py by accepting and
# ignoring labels. Shard files are NOT pre-created here -- pair_table.py writes
# a header only when the file does not yet exist.
set -uo pipefail
cd "$(dirname "$0")/.."
log(){ echo "[$(date '+%H:%M:%S')] $*"; }

log "waiting for stage 3"
while pgrep -f "queue_reverse.sh" >/dev/null; do sleep 60; done
log "stage 3 clear; CCS pair table, 2 shards"

for i in 0 1; do
  lo=$((i*60)); hi=$(((i+1)*60))
  OMP_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 MKL_NUM_THREADS=3 \
    python3 scripts/pair_table.py --probe ccs --tag "_ccs_s$i" \
      --pairs "$lo:$hi" > "/tmp/ccs_s$i.log" 2>&1 &
done
wait

log "merging"
python3 - <<'PY'
import csv, glob, os
cols=None; rows={}
for f in sorted(glob.glob("results/pair_table_ccs_s*.csv")):
    if not os.path.exists(f) or os.path.getsize(f)==0: continue
    with open(f) as fh:
        r=csv.DictReader(fh)
        if r.fieldnames: cols=cols or r.fieldnames
        for row in r: rows[(row["model_a"],row["model_b"],row["dataset"])]=row
if cols:
    with open("results/pair_table_ccs.csv","w",newline="") as fh:
        w=csv.DictWriter(fh,fieldnames=cols); w.writeheader()
        for k in sorted(rows): w.writerow(rows[k])
    print(f"-> results/pair_table_ccs.csv: {len(rows)} cells")
else:
    print("NO CCS ROWS -- check /tmp/ccs_s*.log")
PY
log "STAGE 4 DONE"
