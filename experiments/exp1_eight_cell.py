#!/usr/bin/env python3
"""Experiment 1 — the 8-cell table and the Row-2 rate (the headline).

Fills all eight cells of the joint table on a labeled test split for a model
pair, and reports:

    Row-2 rate = P(m1 says True AND m2 says True | GT = False)

plus the probes' error correlation, which drives how much coverage a label
buys. Run it for both the same-family and the cross-family pair: the question
in DESIGN.md §2.3 is whether Row 2 *drops* with model independence (a floor,
boundable) or stays put (a ceiling, irreducible).

    python experiments/exp1_eight_cell.py --pair same-family --dataset truthfulqa
    python experiments/exp1_eight_cell.py --pair all --dataset truthfulqa
"""

from __future__ import annotations

from common import (PairRun, base_parser, header, layer_specs,
                    resolve_pair, write_result)

from cmb import metrics
from cmb.config import PAIRS


def run_pair(a, b, args):
    la, lb = layer_specs(args)
    run = PairRun(a, b, args.dataset, la, lb, args.n,
                  args.synthetic, args.seed).build(refresh=args.refresh)
    v1, v2, gt = run.verdicts(args.mode)
    table = metrics.eight_cell(v1, v2, gt)
    lo, hi = metrics.wilson_interval(table.row2_count, table.n_false)

    header(f"EXP 1 — 8-cell table   [{a} | {b}]  {run.kind}  {args.dataset}"
           f"  mode={args.mode}")
    print(table.render())
    print(f"  Row-2 95% CI (Wilson)               = [{lo:.3f}, {hi:.3f}]")
    if table.row2_count < 20:
        print(f"  NOTE: only {table.row2_count} Row-2 items — too few to "
              f"characterize. See the data-scarcity note in DESIGN.md §4.")
    return run, {"pair": [a, b], "kind": run.kind, "table": table.to_dict(),
                 "row2_ci": [lo, hi], "cka": run.cka()}


def main() -> int:
    ap = base_parser(__doc__.splitlines()[0])
    ap.add_argument("--mode", default="native",
                    choices=["native", "a_to_b", "b_to_a"],
                    help="native probes, or one probe transported to both models")
    args = ap.parse_args()

    pairs = (list(PAIRS.values()) if args.pair == "all"
             else [resolve_pair(args.pair)])
    out = []
    for a, b in pairs:
        _, res = run_pair(a, b, args)
        out.append(res)

    if len(out) > 1:
        header("Row-2 rate by pair independence  (floor vs ceiling, §2.3)")
        for r in sorted(out, key=lambda r: r["kind"]):
            t = r["table"]
            print(f"  {r['kind']:<14} {r['pair'][0]:>10} | {r['pair'][1]:<10}"
                  f"  Row-2 {t['row2_rate']:.3f}"
                  f"  err-corr {t['error_correlation']:+.3f}"
                  f"  CKA {r['cka']:.3f}")
        same = [r for r in out if r["kind"] == "same-family"]
        cross = [r for r in out if r["kind"] == "cross-family"]
        if same and cross:
            ds = sum(r["table"]["row2_rate"] for r in same) / len(same)
            dc = sum(r["table"]["row2_rate"] for r in cross) / len(cross)
            print(f"\n  same-family {ds:.3f}  vs  cross-family {dc:.3f}"
                  f"   (delta {dc - ds:+.3f})")
            print("  " + ("FLOOR: Row 2 drops with independence — agreement is "
                          "trustworthy and improves with better model choices."
                          if dc < ds else
                          "CEILING: independence does not help — shared training "
                          "errors keep both probes wrong together."))

    write_result("exp1_eight_cell",
                 {"dataset": args.dataset, "mode": args.mode, "results": out},
                 args.tag)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
