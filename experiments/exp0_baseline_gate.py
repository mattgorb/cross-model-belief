#!/usr/bin/env python3
"""Experiment 0 — is the activation signal even worth it? (DESIGN.md §4)

Compares activation-space belief against the output-logprob signal on the same
items, for single models and for the cross-model disagreement signal. If
activations do not beat logprobs, the internal-space framing is not justified
and the right move is to say so and stop escalating.

    python experiments/exp0_baseline_gate.py --pair cross-family --dataset truthfulqa
"""

from __future__ import annotations

import numpy as np

from common import PairRun, base_parser, header, resolve_pair, write_result

from cmb import metrics


def main() -> int:
    ap = base_parser(__doc__.splitlines()[0])
    args = ap.parse_args()
    a, b = resolve_pair(args.pair)

    run = PairRun(a, b, args.dataset, args.layer_frac, args.n,
                  args.synthetic, args.seed).build(refresh=args.refresh)
    gt = run.labels_test
    p_yes_a = run.acts_a.p_yes[run.te]
    p_yes_b = run.acts_b.p_yes[run.te]
    bel_a = run.native_belief("a")
    bel_b = run.native_belief("b")

    header(f"EXP 0 — activations vs logprobs   [{a} | {b}]  {args.dataset}")
    rows = {
        "logprob_a": metrics.auroc(p_yes_a, gt),
        "probe_a": metrics.auroc(bel_a, gt),
        "logprob_b": metrics.auroc(p_yes_b, gt),
        "probe_b": metrics.auroc(bel_b, gt),
    }
    print(f"  {'signal':<28}{'AUROC vs ground truth'}")
    for k, v in rows.items():
        print(f"  {k:<28}{v:.3f}")

    # Routing comparison. Comparing the two disagreement signals by AUROC
    # against "someone is wrong" is not neutral — each space defines its own
    # verdicts, so whichever space supplies the target wins. The comparable
    # quantity is what routing actually buys: among items where the two models
    # AGREE (the router stays silent), how often is the agreed verdict wrong?
    # That is the Row-2 rate generalized over both labels, and it is measured
    # against ground truth in both spaces.
    route = {}
    for space, (s1, s2) in {"logprob": (p_yes_a, p_yes_b),
                            "activation": (bel_a, bel_b)}.items():
        v1, v2 = (s1 >= 0.5).astype(int), (s2 >= 0.5).astype(int)
        agree = v1 == v2
        route[space] = {
            "coverage": float(agree.mean()),
            "silent_error_rate": float((v1[agree] != gt[agree]).mean())
            if agree.any() else float("nan"),
            "flagged_error_rate": float(
                ((v1[~agree] != gt[~agree]) | (v2[~agree] != gt[~agree])).mean())
            if (~agree).any() else float("nan"),
            "own_space_auroc": metrics.auroc(
                np.abs(s1 - s2), ((v1 != gt) | (v2 != gt)).astype(int)),
        }

    print("\n  routing: what the router misses when it stays silent")
    print(f"  {'space':<14}{'coverage':>10}{'silent err':>12}{'flagged err':>13}")
    for k, r in route.items():
        print(f"  {k:<14}{r['coverage']:>10.3f}{r['silent_error_rate']:>12.3f}"
              f"{r['flagged_error_rate']:>13.3f}")

    edge_single = max(rows["probe_a"] - rows["logprob_a"],
                      rows["probe_b"] - rows["logprob_b"])
    edge_route = (route["logprob"]["silent_error_rate"]
                  - route["activation"]["silent_error_rate"])
    passed = bool(edge_single > 0 and edge_route > 0)

    print(f"\n  best single-model probe edge over logprob: {edge_single:+.3f}")
    print(f"  silent-error reduction (logprob - activation): {edge_route:+.3f}")
    if abs(route["logprob"]["coverage"] - route["activation"]["coverage"]) > 0.1:
        print("  CAUTION: the two spaces agree at very different rates, so the "
              "silent-error rates are not measured at matched coverage.")
    print(f"\n  VERDICT: {'PASS' if passed else 'FAIL'} — "
          + ("activation space earns the framing."
             if passed else
             "activations do not beat logprobs; the internal-space framing is "
             "not justified on this data. Report it and stop escalating "
             "(DESIGN.md §7 rule 1)."))

    write_result("exp0_baseline_gate",
                 {"pair": [a, b], "kind": run.kind, "dataset": args.dataset,
                  "layer": [run.layer_a, run.layer_b], "n_test": int(len(gt)),
                  "auroc": rows, "routing": route,
                  "edge_single": edge_single, "edge_routing": edge_route,
                  "passed": passed},
                 args.tag)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
