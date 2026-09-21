#!/usr/bin/env bash
# The DESIGN.md §10 milestone order, as one command.
#
#   scripts/run_all.sh --synthetic          # full pipeline, no GPU, ~1 min
#   scripts/run_all.sh                      # the real thing (needs GPUs)
#
# The gates are gates: Exp 0 and Gate A exit non-zero on failure and stop the
# run, because nothing downstream of a failed gate means anything (DESIGN.md
# §7). Pass --keep-going to override and see every number anyway.
set -uo pipefail
cd "$(dirname "$0")/.."

EXTRA=()
KEEP_GOING=0
for arg in "$@"; do
  if [[ "$arg" == "--keep-going" ]]; then KEEP_GOING=1; else EXTRA+=("$arg"); fi
done
ARGS=("${EXTRA[@]+"${EXTRA[@]}"}")

N=${CMB_N:-800}
PAIR=${CMB_PAIR:-cross-family}
DATASET=${CMB_DATASET:-truthfulqa}

step() {
  local gate=$1; shift
  echo; echo "### $*"
  python3 "$@"
  local rc=$?
  if [[ $rc -ne 0 && $gate == "gate" && $KEEP_GOING -eq 0 ]]; then
    echo
    echo "GATE FAILED (exit $rc). Stopping — see the verdict above and the"
    echo "decision tree in DESIGN.md §7. Re-run with --keep-going to continue."
    exit $rc
  fi
}

step gate experiments/exp0_baseline_gate.py    --pair "$PAIR" --dataset "$DATASET" --n "$N" "${ARGS[@]+"${ARGS[@]}"}"
step gate experiments/gate_a_probe_truth.py    --pair "$PAIR" --dataset "$DATASET" --n "$N" "${ARGS[@]+"${ARGS[@]}"}"
step gate experiments/gate_b_transport.py      --pair "$PAIR" --dataset "$DATASET" --n "$N" "${ARGS[@]+"${ARGS[@]}"}"
step run  experiments/exp1_eight_cell.py       --pair all     --dataset "$DATASET" --n "$N" "${ARGS[@]+"${ARGS[@]}"}"
step run  experiments/exp2_bidirectional.py    --pair "$PAIR" --dataset "$DATASET" --n "$N" "${ARGS[@]+"${ARGS[@]}"}"
step run  experiments/exp3_separability.py     --pair "$PAIR" --dataset "$DATASET" --n "$N" "${ARGS[@]+"${ARGS[@]}"}"
step run  experiments/exp4_generalization_matrix.py --pair "$PAIR" --n "$N" --held-out-pair "${ARGS[@]+"${ARGS[@]}"}"
step run  experiments/exp5_layer_sweep.py      --pair "$PAIR" --dataset "$DATASET" --n "$N" "${ARGS[@]+"${ARGS[@]}"}"

echo
echo "All results in results/*.json. Write up the outcome in results/RESULTS.md,"
echo "classified against the DESIGN.md §7 decision tree."
