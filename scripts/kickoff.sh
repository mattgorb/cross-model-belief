#!/usr/bin/env bash
# Kick off a real run: warm the cache, then run every experiment.
#
#   scripts/kickoff.sh --dry-run                        # print the plan, do nothing
#   scripts/kickoff.sh --pair cross-family --n 2000     # the usual first run
#   scripts/kickoff.sh --pair all --datasets truthfulqa,geometry_of_truth --n 2000
#   scripts/kickoff.sh --synthetic --n 300              # no GPU, proves the wiring
#
# Two phases with very different costs, which is the whole reason this script
# exists: extraction is GPU-bound, unbatched and unrepeatable in practice, while
# the experiments are seconds of CPU off the cache. So extraction happens first,
# deliberately, and is verified before anything else runs — if the box goes away
# mid-analysis you lose nothing.
set -uo pipefail
cd "$(dirname "$0")/.."

PAIR=cross-family
DATASETS=truthfulqa
N=2000
DRY=0
EXTRA=()        # -> the experiment scripts
EX_ARGS=()      # -> scripts/extract.py, which has a different flag set

# The two phases take different flags, so unknown arguments cannot simply be
# forwarded to both: `--keep-going` means something to run_all.sh and is an error
# to extract.py. Only the flags both understand are duplicated.
while [[ $# -gt 0 ]]; do
  case "$1" in
    --pair)      PAIR="$2"; shift 2 ;;
    --dataset|--datasets) DATASETS="$2"; shift 2 ;;
    --n)         N="$2"; shift 2 ;;
    --dry-run)   DRY=1; shift ;;
    --synthetic) EXTRA+=("$1"); EX_ARGS+=("$1"); shift ;;
    --refresh)   EXTRA+=("$1"); EX_ARGS+=("$1"); shift ;;
    --layers)    EX_ARGS+=("$1" "$2"); shift 2 ;;      # extraction only
    -h|--help)   sed -n '2,16p' "$0"; exit 0 ;;
    *)           EXTRA+=("$1"); shift ;;   # --keep-going, --probe X, --tag X, ...
  esac
done
ARGS=("${EXTRA[@]+"${EXTRA[@]}"}")
EXARGS=("${EX_ARGS[@]+"${EX_ARGS[@]}"}")
PRIMARY="${DATASETS%%,*}"                    # first dataset drives the single-pair steps

# Which models the requested pair needs, straight from the registry so this can
# never drift from cmb/config.py.
MODELS=$(python3 - "$PAIR" <<'PY'
import sys
sys.path.insert(0, ".")
from cmb.config import PAIRS, RESEARCH_PAIRS, POST_TRAINING_PAIR
want = sys.argv[1]
keys = list(RESEARCH_PAIRS) if want == "all" else [want]
models = []
for k in keys:
    models += list(PAIRS[k]) if k in PAIRS else [m.strip() for m in k.split(",")]
models += list(POST_TRAINING_PAIR)           # Exp 6 runs in every pass
print(",".join(dict.fromkeys(models)))
PY
)
if [[ -z "$MODELS" ]]; then
  echo "could not resolve models for --pair $PAIR (see cmb/config.py PAIRS)" >&2
  exit 2
fi

echo "=== plan ==="
echo "  pair(s)   : $PAIR"
echo "  datasets  : $DATASETS   (single-pair steps use: $PRIMARY)"
echo "  N         : $N"
echo "  models    : $MODELS"
echo "  extract   : ${EXARGS[*]-none}"
echo "  experiments: ${ARGS[*]-none}"
echo "  cache     : ${CMB_CACHE:-<unset — vectors will land in the repo>}"
echo "  results   : ${CMB_RESULTS:-results/}"
echo
echo "  phase 1  extract (GPU, the expensive part, once)"
echo "  phase 2  gates: exp0 -> gate A -> gate B   (a failure here stops the run)"
echo "  phase 3  results: exp1..exp6"

if [[ " ${EXARGS[*]-} " != *"--synthetic"* ]]; then
  [[ -z "${CMB_CACHE:-}" ]] && echo && echo "WARNING: CMB_CACHE is unset. On a spot box, point it at /mnt/data/activations
         or an interruption costs you the extraction (docs/RUNBOOK.md §1)."
  [[ -z "${HF_TOKEN:-}" ]] && echo "WARNING: HF_TOKEN is unset — Llama-3.1 is gated and will 401."
  command -v nvidia-smi >/dev/null 2>&1 || echo "WARNING: no nvidia-smi on PATH — extraction will be CPU-slow."
fi

if [[ $DRY -eq 1 ]]; then echo; echo "(dry run — nothing executed)"; exit 0; fi

echo
echo "### phase 1 — extract"
python3 scripts/extract.py --models "$MODELS" --datasets "$DATASETS" --n "$N" "${EXARGS[@]+"${EXARGS[@]}"}"
rc=$?
if [[ $rc -ne 0 ]]; then
  echo "extraction failed (exit $rc). Nothing downstream can run; fix this first." >&2
  exit $rc
fi
python3 scripts/extract.py --list

echo
echo "### phases 2 and 3 — experiments"
CMB_N="$N" CMB_PAIR="$PAIR" CMB_DATASET="$PRIMARY" \
  scripts/run_all.sh "${ARGS[@]+"${ARGS[@]}"}"
rc=$?

echo
if [[ $rc -eq 0 ]]; then
  echo "Done. Verdicts above, machine-readable copies in ${CMB_RESULTS:-results/}."
  echo "Next: fill in results/RESULTS.md against the DESIGN.md §7 decision tree,"
  echo "then carry the numbers into paper/ (abstract.tex still has the results"
  echo "sentence blank on purpose)."
else
  echo "A gate failed (exit $rc). That is a result, not a bug — read the verdict"
  echo "and the decision tree in DESIGN.md §7 before re-running with --keep-going."
fi
echo "Before you terminate the box: scripts/extract.py --list, and confirm the"
echo ".npz files are on persistent storage."
exit $rc
