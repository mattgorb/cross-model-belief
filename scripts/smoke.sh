#!/usr/bin/env bash
# Fast end-to-end check on the synthetic backend: every script, small N.
set -euo pipefail
cd "$(dirname "$0")/.."
export CMB_CACHE=${CMB_CACHE:-/tmp/cmb-smoke/cache}
export CMB_RESULTS=${CMB_RESULTS:-/tmp/cmb-smoke/results}
python3 -m pytest tests -q
CMB_N=300 scripts/run_all.sh --synthetic --keep-going > /tmp/cmb-smoke/log.txt 2>&1 || {
  tail -40 /tmp/cmb-smoke/log.txt; exit 1; }
grep -E "VERDICT|Row-2 rate|OFF-DIAGONAL" /tmp/cmb-smoke/log.txt
echo "smoke OK — full log at /tmp/cmb-smoke/log.txt"
