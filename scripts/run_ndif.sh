#!/usr/bin/env bash
# Remote extraction via NDIF. Needs NDIF_API_KEY exported in this shell; the
# cache is kept separate from the AWS-generated activations.
set -uo pipefail
cd "$(dirname "$0")/.."
export CMB_CACHE="$PWD/activations_ndif"

: "${NDIF_API_KEY:?export NDIF_API_KEY first}"

STEP="${1:-smoke}"
ALL_DS=rte,truthfulqa,boolq,geometry_of_truth,imdb
WORKING=llama31-70b,llama31-70b-base,llama31-405b

case "$STEP" in
  # Settle the two unverified fixes on 8 items before spending real time.
  smoke)
    python3 scripts/extract.py --backend ndif --batch-size 8 --n 8 --refresh \
      --models gemma2-9b,gptj-6b --datasets rte,truthfulqa
    ;;
  # Everything, from scratch. --refresh is required, not optional: the files
  # extracted before the position_ids fix encoded each claim at an offset that
  # depended on its batch, so they have to be overwritten rather than skipped.
  all)
    python3 scripts/extract.py --backend ndif --batch-size 8 --refresh \
      --models meta-llama/Llama-3.1-8B,$WORKING,gptj-6b,gemma2-9b \
      --datasets $ALL_DS
    ;;
  # Small datasets first, as a cheaper checkpoint before the big three.
  small)
    python3 scripts/extract.py --backend ndif --batch-size 8 --refresh \
      --models meta-llama/Llama-3.1-8B,$WORKING,gptj-6b,gemma2-9b \
      --datasets rte,truthfulqa
    ;;
  # Only the cells with no file yet. Leaves the 9 pre-fix cells in place, so
  # the cache ends up mixing two position conventions -- cheaper, but the 70B
  # and 405B truthfulqa/rte numbers then come from the older code path.
  resume)
    python3 scripts/extract.py --backend ndif --batch-size 8 \
      --models meta-llama/Llama-3.1-8B,$WORKING,gptj-6b,gemma2-9b \
      --datasets $ALL_DS
    ;;
  *) echo "usage: $0 {smoke|small|resume|all}"; exit 2 ;;
esac
