#!/usr/bin/env python3
"""Gate A — does the probe read truth, not confidence? (DESIGN.md §4)

For each model: native CCS probe, AUROC against ground truth on the *confident
slice* (top 50% of items by the model's own output confidence), against the
raw-confidence baseline on the same slice.

Pass  = probe AUROC > confidence-baseline AUROC on the slice.
Fail  = the probe is reading confidence, not truth. Halt: nothing downstream
        means anything (DESIGN.md §7 rule 2).

    python experiments/gate_a_probe_truth.py --pair cross-family --dataset truthfulqa
"""

from __future__ import annotations

import numpy as np

from common import PairRun, base_parser, header, resolve_pair, write_result

from cmb import metrics


def gate_a_for(name, belief, p_yes, gt, slice_q):
    conf = metrics.confidence(p_yes)
    mask = metrics.confident_slice(p_yes, slice_q)
    out = {
        "model": name,
        "n": int(len(gt)),
        "n_slice": int(mask.sum()),
        "auroc_full": metrics.auroc(belief, gt),
        "baseline_full": metrics.auroc(p_yes, gt),
        "auroc_slice": metrics.auroc(belief[mask], gt[mask]),
        "baseline_slice": metrics.auroc(conf[mask] * np.sign(p_yes[mask] - 0.5),
                                        gt[mask]),
        "error_rate_slice": float((( p_yes[mask] >= 0.5).astype(int)
                                   != gt[mask]).mean()),
    }
    out["edge_slice"] = out["auroc_slice"] - out["baseline_slice"]
    out["passed"] = bool(out["edge_slice"] > 0)
    return out


def report(r) -> None:
    print(f"\n  --- {r['model']} ---")
    print(f"  probe AUROC (all items)       {r['auroc_full']:.3f}"
          f"   baseline {r['baseline_full']:.3f}")
    print(f"  confident slice  n={r['n_slice']}"
          f"   (confidently wrong: {r['error_rate_slice']:.2f})")
    print(f"  baseline AUROC on slice       {r['baseline_slice']:.3f}")
    print(f"  PROBE AUROC on slice          {r['auroc_slice']:.3f}"
          f"   <-- the number that matters")
    print(f"  edge                          {r['edge_slice']:+.3f}"
          f"   {'PASS' if r['passed'] else 'FAIL'}")


def main() -> int:
    ap = base_parser(__doc__.splitlines()[0])
    ap.add_argument("--slice-q", type=float, default=0.5,
                    help="confident slice = top this fraction by output confidence")
    args = ap.parse_args()
    a, b = resolve_pair(args.pair)

    run = PairRun(a, b, args.dataset, args.layer_frac, args.n,
                  args.synthetic, args.seed).build(refresh=args.refresh)
    gt = run.labels_test

    header(f"GATE A — truth or confidence?   [{a} | {b}]  {args.dataset}  "
           f"layers {run.layer_a}/{run.layer_b}")
    ra = gate_a_for(a, run.native_belief("a"), run.acts_a.p_yes[run.te], gt, args.slice_q)
    rb = gate_a_for(b, run.native_belief("b"), run.acts_b.p_yes[run.te], gt, args.slice_q)
    report(ra)
    report(rb)

    passed = ra["passed"] and rb["passed"]
    print(f"\n  VERDICT: {'PASS' if passed else 'FAIL'} — "
          + ("probes carry truth signal beyond output confidence."
             if passed else
             "at least one probe does not beat confidence on its own confident "
             "slice. It is reading confidence, not truth: stop here, no "
             "downstream result is meaningful (DESIGN.md §7 rule 2)."))

    write_result("gate_a_probe_truth",
                 {"pair": [a, b], "kind": run.kind, "dataset": args.dataset,
                  "layer": [run.layer_a, run.layer_b], "slice_q": args.slice_q,
                  "models": [ra, rb], "passed": passed}, args.tag)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
