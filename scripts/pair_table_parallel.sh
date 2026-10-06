#!/usr/bin/env bash
# Shard pair_table.py across N processes on disjoint pair slices.
#
# Each shard gets its own output file seeded with the existing results, so its
# resume logic skips every cell already computed -- sharding adds work, it never
# repeats it. Writes stay in separate files because appending to one CSV from
# several processes interleaves rows. Merge with --merge when they finish.
set -uo pipefail
cd "$(dirname "$0")/.."
MAIN=results/pair_table.csv
N=${N:-3}

if [ "${1:-}" = "--merge" ]; then
  python3 - "$MAIN" <<'PY'
import csv, glob, sys, os
main = sys.argv[1]
cols, rows = None, {}
for f in [main] + sorted(glob.glob("results/pair_table_s*.csv")):
    if not os.path.exists(f): continue
    with open(f) as fh:
        r = csv.DictReader(fh)
        if r.fieldnames: cols = cols or r.fieldnames
        for row in r:
            rows[(row["model_a"], row["model_b"], row["dataset"])] = row
with open(main, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=cols); w.writeheader()
    for k in sorted(rows): w.writerow(rows[k])
print(f"merged -> {main}: {len(rows)} unique cells")
PY
  exit 0
fi

TOTAL=$(python3 -c "
import sys,os,itertools; sys.path.insert(0,'.')
from cmb.config import CACHE_SEARCH_DIRS
d={}
for root in CACHE_SEARCH_DIRS:
    if not root.exists(): continue
    for x in sorted(os.listdir(root)):
        if (root/x).is_dir() and x!='manifests':
            ds=sorted(f.replace('_nall.npz','') for f in os.listdir(root/x) if f.endswith('_nall.npz'))
            if ds: d.setdefault(x,ds)
print(sum(1 for a,b in itertools.combinations(sorted(d),2) if len(set(d[a])&set(d[b]))>=3))")
echo "$TOTAL pairs across $N shards"
STEP=$(( (TOTAL + N - 1) / N ))
for i in $(seq 0 $((N-1))); do
  LO=$((i*STEP)); HI=$(((i+1)*STEP))
  [ "$LO" -ge "$TOTAL" ] && break
  cp "$MAIN" "results/pair_table_s${i}.csv"
  # one BLAS thread each: these are big matmuls and N processes x 10 threads
  # thrashes a 10-core box harder than it parallelises
  OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \
    nohup python3 scripts/pair_table.py --probe lr --tag "_s${i}" \
      --pairs "${LO}:${HI}" > "/tmp/pt_s${i}.log" 2>&1 &
  echo "  shard $i: pairs ${LO}:${HI}  pid $!"
done
