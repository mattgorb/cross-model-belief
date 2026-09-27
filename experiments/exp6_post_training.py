#!/usr/bin/env python3
"""Experiment 6 — does post-training degrade transport? (DESIGN.md §4)

One model, two checkpoints: base and instruct. Same family, same size, same
tokenizer, same data — the only variable is post-training. Fit the same probe on
both, transport between them, and ask whether the map still carries the probe.

The question is layer-specific and that is the point. The literature probes
middle layers (Marks & Tegmark use layer 13 of 40; the sleeper-agent probes are
also middle-layer), while HELIX and the platonic-representation argument point at
the *last* layer for the alignment map. If post-training rewrites the last layer
more than the middle, the last-layer default is buying map quality at the cost of
transport stability, and this experiment is where that shows up.

Read the output as a control, not a result: base <-> instruct is as favourable as
transport ever gets, so its epsilon is a lower bound on what cross-family pairs
will pay.

    python experiments/exp6_post_training.py --sweep final,0.6 --dataset truthfulqa
"""

from __future__ import annotations

from common import base_parser, header, make_run, resolve_pair, write_result

from cmb import metrics
from cmb.config import POST_TRAINING_PAIR

TOLERANCE = 0.05


def main() -> int:
    ap = base_parser(__doc__.splitlines()[0])
    ap.add_argument("--sweep", default="final,0.6",
                    help="comma-separated layer specs to compare (the last-layer "
                         "default first, then the mid-depth site the probe "
                         "literature uses)")
    ap.add_argument("--tolerance", type=float, default=TOLERANCE)
    ap.set_defaults(pair="post-training")
    args = ap.parse_args()
    a, b = resolve_pair(args.pair) if args.pair != "post-training" else POST_TRAINING_PAIR

    header(f"EXP 6 — post-training and transport   [{a} (base) | {b} (instruct)]"
           f"  {args.dataset}  probe={args.probe}")

    rows = []
    for spec in [s.strip() for s in args.sweep.split(",")]:
        run = make_run(a, b, args, layers=(spec, spec)).build(refresh=args.refresh)
        gt = run.labels_test
        native_base = metrics.auroc(run.native_belief("a"), gt)
        native_inst = metrics.auroc(run.native_belief("b"), gt)
        # base probe read on the instruct checkpoint, and the reverse
        base_to_inst = metrics.auroc(run.transported_belief("a"), gt)
        inst_to_base = metrics.auroc(run.transported_belief("b"), gt)
        v1, v2, gt_ = run.verdicts("native")
        fa = metrics.false_agreement(v1, v2, gt_)
        rows.append({
            "layer_spec": spec, "layer": [run.layer_a, run.layer_b],
            "native": {"base": native_base, "instruct": native_inst},
            "transfer": {"base_to_instruct": base_to_inst,
                         "instruct_to_base": inst_to_base},
            "gap": {"base_to_instruct": native_inst - base_to_inst,
                    "instruct_to_base": native_base - inst_to_base},
            "cka": run.cka(), "false_agreement": fa.to_dict()})

    print(f"  {'layer':<8} {'AUROC base':>11} {'AUROC inst':>11}"
          f" {'b->i':>7} {'gap':>7} {'i->b':>7} {'gap':>7} {'CKA':>6} {'FA':>7}")
    for r in rows:
        print(f"  {r['layer_spec']:<8} {r['native']['base']:>11.3f}"
              f" {r['native']['instruct']:>11.3f}"
              f" {r['transfer']['base_to_instruct']:>7.3f}"
              f" {r['gap']['base_to_instruct']:>+7.3f}"
              f" {r['transfer']['instruct_to_base']:>7.3f}"
              f" {r['gap']['instruct_to_base']:>+7.3f}"
              f" {r['cka']:>6.3f} {r['false_agreement']['fa']:>7.4f}")

    worst = {r["layer_spec"]: max(r["gap"].values()) for r in rows}
    final = next((k for k in worst if k in ("final", "last", "-1")), None)
    others = [k for k in worst if k != final]
    print()
    if final and others:
        best_other = min(others, key=lambda k: worst[k])
        if worst[final] > worst[best_other] + args.tolerance:
            print(f"  VERDICT: post-training degrades transport at the LAST layer"
                  f" (gap {worst[final]:+.3f}) more than at {best_other}"
                  f" (gap {worst[best_other]:+.3f}).")
            print(f"  Action: fit the map at the probe's own layer rather than "
                  f"defaulting to `final`, and report the layer in every table.")
        else:
            print(f"  VERDICT: no last-layer-specific degradation "
                  f"(worst gap {worst[final]:+.3f} vs {worst[best_other]:+.3f} "
                  f"at {best_other}). The `final` default stands.")
    print("  Note: an epsilon this pair cannot avoid is one every other pair "
          "inherits — see Gate B for the per-pair number.")

    write_result("exp6_post_training",
                 {"pair": [a, b], "dataset": args.dataset, "probe": args.probe,
                  "tolerance": args.tolerance, "rows": rows}, args.tag)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
