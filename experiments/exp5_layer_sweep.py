#!/usr/bin/env python3
"""Experiment 5 — layer sweep.

Repeats Gate A, the Row-2 rate (Exp 1) and separability (Exp 3) at several
depths. Generalizing truth structure and prominence artifacts often live at
different depths, and this is nearly free once activations are cached — the
whole sweep comes out of the same forward pass.

    python experiments/exp5_layer_sweep.py --pair cross-family --dataset truthfulqa
"""

from __future__ import annotations

import numpy as np

from common import PairRun, base_parser, header, resolve_pair, write_result

from cmb import metrics
from cmb.config import LAYER_SWEEP
from exp3_separability import fit_and_score, subset_for
from gate_a_probe_truth import gate_a_for


def main() -> int:
    ap = base_parser(__doc__.splitlines()[0])
    ap.add_argument("--sweep", default=",".join(str(f) for f in LAYER_SWEEP),
                    help="comma-separated layer specs to sweep; applied to both "
                         "models (fractions keep the site comparable across "
                         "models of different depth)")
    ap.add_argument("--C", type=float, default=0.1)
    ap.add_argument("--mode", default="native",
                    choices=["native", "a_to_b", "b_to_a"])
    args = ap.parse_args()
    a, b = resolve_pair(args.pair)
    specs = [s.strip() for s in args.sweep.split(",") if s.strip()]

    header(f"EXP 5 — layer sweep   [{a} | {b}]  {args.dataset}")
    print(f"  {'layer':>8} {'idx':>10} {'gateA edge':>12} {'transferAUROC':>14}"
          f" {'row2':>8} {'sep AUROC':>10}")
    rows = []
    for spec in specs:
        run = PairRun(a, b, args.dataset, spec, spec, args.n, args.synthetic,
                      args.seed).build(refresh=args.refresh)
        gt = run.labels_test
        ga = gate_a_for(a, run.native_belief("a"), run.acts_a.p_yes[run.te], gt, 0.5)
        gb = gate_a_for(b, run.native_belief("b"), run.acts_b.p_yes[run.te], gt, 0.5)
        transfer = metrics.auroc(run.transported_belief("a"), gt)

        v1, v2, g = run.verdicts(args.mode)
        table = metrics.eight_cell(v1, v2, g)
        X_tr, y_tr, _ = subset_for(run, "train", args.mode)
        X_te, y_te, _ = subset_for(run, "test", args.mode)
        sep, _ = fit_and_score(X_tr, y_tr, X_te, y_te, args.C, args.seed)

        edge = min(ga["edge_slice"], gb["edge_slice"])
        print(f"  {spec:>8} {run.layer_a:>4}/{run.layer_b:<5}"
              f" {edge:>+12.3f} {transfer:>14.3f}"
              f" {table.row2_rate:>8.3f} {sep:>10.3f}")
        rows.append({"layer": spec, "layers": [run.layer_a, run.layer_b],
                     "gate_a_edge_min": edge, "transfer_auroc": transfer,
                     "row2_rate": table.row2_rate, "row2_count": table.row2_count,
                     "separability_auroc": sep})

    best_probe = max(rows, key=lambda r: r["gate_a_edge_min"])
    valid = [r for r in rows if not np.isnan(r["separability_auroc"])]
    best_sep = max(valid, key=lambda r: r["separability_auroc"]) if valid else None
    print(f"\n  best Gate A layer       {best_probe['layer']}"
          f"   (edge {best_probe['gate_a_edge_min']:+.3f})")
    if best_sep:
        print(f"  best separability layer {best_sep['layer']}"
              f"   (AUROC {best_sep['separability_auroc']:.3f})")
        if best_sep["layers"] != best_probe["layers"]:
            print("  NOTE: belief and false-agreement structure peak at different "
                  "depths — report both, and do not tune one at the other's layer.")

    write_result("exp5_layer_sweep",
                 {"pair": [a, b], "dataset": args.dataset, "mode": args.mode,
                  "rows": rows, "best_gate_a": best_probe,
                  "best_separability": best_sep}, args.tag)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
