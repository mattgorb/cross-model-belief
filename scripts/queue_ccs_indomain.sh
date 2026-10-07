#!/usr/bin/env bash
# In-domain detection with CCS probes, after the logistic sweep finishes.
#
# The paper excludes CCS from its main results because it fails Gate A on
# TruthfulQA -- it reads curated factual statements at 0.99 AUROC and scores
# below the confidence baseline on adversarial misconceptions. That is a claim
# about CCS as a *belief* probe.
#
# This asks a different question: when both readouts are unsupervised, is the
# false-agreement direction still findable? If so, the oversight claim becomes
# label-free rather than "fit once, deploy without labels". Expect a high
# failure rate -- CCS was at chance on every base model tested (0.527-0.548) and
# on gemma4-12b (0.478-0.499) -- so cells returning NaN or near chance are the
# result, not a bug.
#
# Cheaper than it looks: the in-domain phase caches the probe per (model,
# dataset), so this is at most 80 CCS fits per shard rather than one per cell.
set -uo pipefail
cd "$(dirname "$0")/.."
log(){ echo "[$(date '+%H:%M:%S')] $*"; }

log "waiting for the logistic in-domain sweep"
while pgrep -f "sweep.py --phase indomain .*--tag _lr" >/dev/null; do sleep 60; done
log "logistic clear; starting CCS"

i=0
for rng in 0:64 64:128; do
  if [ -f results/sweep_indomain_ccs.csv ]; then
    cp results/sweep_indomain_ccs.csv "results/sweep_indomain_ccs_s$i.csv"
  fi
  OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \
    python3 scripts/sweep.py --phase indomain --probe ccs \
      --tag "_ccs_s$i" --pairs "$rng" > "/tmp/indomain_ccs_s$i.log" 2>&1 &
  i=$((i+1))
done
wait

python3 - <<'PY'
import csv, glob, os
cols=None; rows={}
for f in ["results/sweep_indomain_ccs.csv"]+sorted(glob.glob("results/sweep_indomain_ccs_s*.csv")):
    if not os.path.exists(f) or os.path.getsize(f) == 0:
        continue
    with open(f) as fh:
        r = csv.DictReader(fh)
        if r.fieldnames:
            cols = cols or r.fieldnames
        for row in r:
            rows[(row["overseer"], row["target"], row["held_out"])] = row
if cols:
    with open("results/sweep_indomain_ccs.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols); w.writeheader()
        for k in sorted(rows): w.writerow(rows[k])
    print(f"merged -> results/sweep_indomain_ccs.csv: {len(rows)} rows")
else:
    print("NO CCS ROWS -- check /tmp/indomain_ccs_s*.log")
PY
rm -f results/sweep_indomain_ccs_s*.csv
log "CCS IN-DOMAIN DONE"
