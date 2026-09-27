#!/usr/bin/env python3
"""Experiment 1 — the 8-cell table and the Row-2 rate (the headline).

Fills all eight cells of the joint table on a labeled test split for a model
pair, and reports:

    Row-2 rate = P(m1 says True AND m2 says True | GT = False)

plus the decomposition of that rate (DESIGN.md §2.4, paper/theory.tex): the two
false-positive rates, their error correlation, the Frechet bounds the rate has to
sit inside, and the coverage a disagreement router therefore buys. The rate alone
does not say whether a pair is near independence or near maximal overlap, and
that is the finding.

Run it for both the same-family and the cross-family pair: the question in
DESIGN.md §2.3 is whether Row 2 *drops* with model independence (a floor,
boundable) or stays put (a ceiling, irreducible).

Every run also prints the cost of transport — the rate with native probes next to
the rate with a transported one — because the map inflates both the
false-positive rate (by epsilon) and, separately, the error correlation.

    python experiments/exp1_eight_cell.py --pair same-family --dataset truthfulqa
    python experiments/exp1_eight_cell.py --pair all --dataset truthfulqa
"""

from __future__ import annotations

from common import (base_parser, header, make_run, resolve_pair, write_result)

from cmb import metrics
from cmb.config import PAIRS, RESEARCH_PAIRS


def run_pair(a, b, args):
    run = make_run(a, b, args).build(refresh=args.refresh)
    v1, v2, gt = run.verdicts(args.mode)
    table = metrics.eight_cell(v1, v2, gt)
    fa = metrics.false_agreement(v1, v2, gt)
    lo, hi = metrics.wilson_interval(table.row2_count, table.n_false)

    header(f"EXP 1 — 8-cell table   [{a} | {b}]  {run.kind}  {args.dataset}"
           f"  probe={run.probe_kind}  mode={args.mode}")
    print(table.render())
    print(f"  Row-2 95% CI (Wilson)               = [{lo:.3f}, {hi:.3f}]")
    if table.row2_count < 20:
        print(f"  NOTE: only {table.row2_count} Row-2 items — too few to "
              f"characterize. See the data-scarcity note in DESIGN.md §4.")

    print("\n  false-agreement decomposition (§2.4)")
    print(fa.render())

    transport = transport_cost(run, fa)
    return run, {"pair": [a, b], "kind": run.kind, "table": table.to_dict(),
                 "row2_ci": [lo, hi], "cka": run.cka(),
                 "false_agreement": fa.to_dict(), "transport": transport}


def transport_cost(run, native_fa):
    """Report FA natively and under each transport direction.

    Two things move when a probe is carried across models instead of fitted in
    place: its false-positive rate degrades by epsilon, and the error correlation
    is inflated, because the map is fitted from the source model's geometry and
    so imports that model's structure into the target's readout. The second is
    the one a caveat would miss, which is why rho is recomputed on the
    transported readouts rather than reused from the native fit.
    """
    out = {"native": native_fa.to_dict()}
    print("\n  cost of transport (native vs transported probes)")
    print(f"    {'mode':<10} {'p1':>6} {'p2':>6} {'eps':>7} {'rho':>7}"
          f" {'rho/max':>8} {'FA':>7} {'dFA':>7}")
    print(f"    {'native':<10} {native_fa.p1:>6.3f} {native_fa.p2:>6.3f}"
          f" {'—':>7} {native_fa.rho:>+7.3f} {native_fa.rho_normalized:>+8.3f}"
          f" {native_fa.fa:>7.4f} {'—':>7}")
    for mode in ("a_to_b", "b_to_a"):
        v1, v2, gt = run.verdicts(mode)
        t = metrics.false_agreement(v1, v2, gt)
        # The transported reader is m2 under a_to_b and m1 under b_to_a.
        eps = (t.p2 - native_fa.p2) if mode == "a_to_b" else (t.p1 - native_fa.p1)
        print(f"    {mode:<10} {t.p1:>6.3f} {t.p2:>6.3f} {eps:>+7.3f}"
              f" {t.rho:>+7.3f} {t.rho_normalized:>+8.3f} {t.fa:>7.4f}"
              f" {t.fa - native_fa.fa:>+7.4f}")
        out[mode] = {**t.to_dict(), "epsilon": float(eps),
                     "delta_fa": float(t.fa - native_fa.fa),
                     "delta_rho": float(t.rho - native_fa.rho)}
    print("    (eps = transport-induced inflation of the transported reader's "
          "false-positive rate;\n     dFA = what that plus the inflated rho "
          "costs in false agreement — the price of the map)")
    return out


def main() -> int:
    ap = base_parser(__doc__.splitlines()[0])
    ap.add_argument("--mode", default="native",
                    choices=["native", "a_to_b", "b_to_a"],
                    help="native probes, or one probe transported to both models")
    args = ap.parse_args()

    pairs = ([PAIRS[k] for k in RESEARCH_PAIRS] if args.pair == "all"
             else [resolve_pair(args.pair)])
    out = []
    for a, b in pairs:
        _, res = run_pair(a, b, args)
        out.append(res)

    if len(out) > 1:
        header("Row-2 rate by pair independence  (floor vs ceiling, §2.3)")
        for r in sorted(out, key=lambda r: r["kind"]):
            t = r["table"]
            fa = r["false_agreement"]
            print(f"  {r['kind']:<14} {r['pair'][0]:>10} | {r['pair'][1]:<10}"
                  f"  Row-2 {t['row2_rate']:.3f}"
                  f"  bounds [{fa['bounds'][0]:.3f},{fa['bounds'][1]:.3f}]"
                  f"  rho/max {fa['rho_normalized']:+.2f}"
                  f"  coverage {fa['coverage']:.2f}"
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
            # The prediction to check against paper/PLAN.md §3: same-family near
            # the upper bound, cross-family partly decorrelated but not
            # independent. rho/max is the scale-free way to read that, since raw
            # rho is not comparable across pairs with different marginals.
            ns = sum(r["false_agreement"]["rho_normalized"] for r in same) / len(same)
            nc = sum(r["false_agreement"]["rho_normalized"] for r in cross) / len(cross)
            print(f"  rho/feasible-max  same-family {ns:+.3f}  vs  "
                  f"cross-family {nc:+.3f}"
                  f"   (prediction: both positive, cross < same)")

    write_result("exp1_eight_cell",
                 {"dataset": args.dataset, "mode": args.mode, "results": out},
                 args.tag)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
