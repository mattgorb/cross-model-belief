#!/usr/bin/env bash
# Stage 2, run after queue_all.sh: the two remaining expensive jobs.
#
#   1. LOO with LR          -> sweep_loo_lr16.csv      (headline detection probe)
#   2. LOO with LR, no map  -> sweep_loo_nomap16.csv   (does the map earn its place?)
#
# Both use LR so the ablation is a clean A/B: same probe, map vs no map. The
# existing *_nomap.csv is 10 models and an unrecorded probe, so it is left alone
# rather than appended to.
set -uo pipefail
cd "$(dirname "$0")/.."
log(){ echo "[$(date '+%H:%M:%S')] $*"; }

merge_loo(){ python3 - "$1" <<'PY'
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
}

run_stage(){                      # name, out, extra flags
  local name="$1" out="$2"; shift 2
  log "$name"
  for i in 0 1; do
    local lo=$((i*64)) hi=$(((i+1)*64))
    [ -f "$out" ] && cp "$out" "${out%.csv}_s$i.csv" || : > "${out%.csv}_s$i.csv"
    OMP_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 MKL_NUM_THREADS=3 \
      python3 scripts/sweep.py --phase loo --probe lr "$@" \
        --tag "$(basename "${out%.csv}" | sed 's/^sweep_loo//')_s$i" \
        --pairs "$lo:$hi" > "/tmp/$(basename "${out%.csv}")_s$i.log" 2>&1 &
  done
  wait
  merge_loo "$out"
  log "$name finished"
}

log "waiting for stage 1 (queue_all.sh)"
while pgrep -f "queue_all.sh" >/dev/null; do sleep 60; done
log "stage 1 clear"

# seed the LR run with the 94 LR rows that already exist, so they are not redone
[ -f results/sweep_loo_lr.csv ] && cp results/sweep_loo_lr.csv results/sweep_loo_lr16.csv

run_stage "LOO + LR (with map)"  results/sweep_loo_lr16.csv
run_stage "LOO + LR (no map)"    results/sweep_loo_nomap16.csv --no-map-features

log "ALL OVERNIGHT WORK DONE"
for f in results/sweep_base_all.csv results/sweep_loo_all.csv \
         results/pair_table.csv results/pair_table_ccs.csv \
         results/sweep_loo_lr16.csv results/sweep_loo_nomap16.csv; do
  [ -f "$f" ] && printf "  %-38s %s rows\n" "$f" "$(( $(wc -l < "$f") - 1 ))"
done
