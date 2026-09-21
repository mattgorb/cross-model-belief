#!/usr/bin/env python3
"""Experiment 2 — bidirectional hard-core false agreement.

Computes the Row-2 set twice, once with A->B transport and once with B->A, and
reports the intersection: the false agreements that survive *both* transport
directions. Those are the hardest, most trustworthy-looking wrong items, and
they are the right target for Experiment 3 — a Row-2 item that appears only
under one transport direction may be an artifact of the map rather than a real
shared belief error.

    python experiments/exp2_bidirectional.py --pair cross-family --dataset truthfulqa
"""

from __future__ import annotations

import json

import numpy as np

from common import PairRun, base_parser, header, resolve_pair, write_result

from cmb import metrics
from cmb.config import RESULTS_DIR


def main() -> int:
    ap = base_parser(__doc__.splitlines()[0])
    args = ap.parse_args()
    a, b = resolve_pair(args.pair)

    run = PairRun(a, b, args.dataset, args.layer_frac, args.n,
                  args.synthetic, args.seed).build(refresh=args.refresh)
    ids = np.array(run.acts_a.item_ids)[run.te]

    sets = {}
    for mode in ("native", "a_to_b", "b_to_a"):
        v1, v2, gt = run.verdicts(mode)
        sets[mode] = metrics.row2_mask(v1, v2, gt)

    core = sets["a_to_b"] & sets["b_to_a"]
    union = sets["a_to_b"] | sets["b_to_a"]
    all_three = core & sets["native"]
    n_false = int((run.labels_test == 0).sum())

    header(f"EXP 2 — bidirectional Row 2   [{a} | {b}]  {args.dataset}")
    for mode in ("native", "a_to_b", "b_to_a"):
        k = int(sets[mode].sum())
        print(f"  Row 2 via {mode:<8} {k:>5}   rate {k / max(n_false,1):.3f}")
    print(f"\n  intersection (transport-robust core)  {int(core.sum()):>5}"
          f"   rate {core.sum() / max(n_false,1):.3f}")
    print(f"  union                                 {int(union.sum()):>5}")
    print(f"  jaccard(a_to_b, b_to_a)               "
          f"{core.sum() / max(union.sum(), 1):.3f}"
          f"   <-- low means the map, not shared belief, is picking the set")
    print(f"  core ∩ native-probe Row 2             {int(all_three.sum()):>5}")

    core_ids = ids[core].tolist()
    path = RESULTS_DIR / f"exp2_row2_core{('_' + args.tag) if args.tag else ''}.json"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(
        {"pair": [a, b], "dataset": args.dataset, "item_ids": core_ids}, indent=2))
    print(f"\n  [wrote the hard-core item ids to {path}]")

    write_result("exp2_bidirectional",
                 {"pair": [a, b], "kind": run.kind, "dataset": args.dataset,
                  "n_false": n_false,
                  "counts": {k: int(v.sum()) for k, v in sets.items()},
                  "core": int(core.sum()), "union": int(union.sum()),
                  "jaccard": float(core.sum() / max(union.sum(), 1)),
                  "core_and_native": int(all_three.sum()),
                  "core_item_ids": core_ids}, args.tag)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
