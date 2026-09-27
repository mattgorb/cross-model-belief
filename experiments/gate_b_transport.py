#!/usr/bin/env python3
"""Gate B — does the map carry the probe? (DESIGN.md §4)

Fit the probe on A, transport it to B through the linear map, test on B.
Pass = transfer AUROC within ~0.05 of native-B AUROC. Fail = the cross-model
framing is dead weight for this pair; fall back to native probes and say so
(DESIGN.md §7 rule 3).

The gate also reports **epsilon**, the degradation expressed as an inflation of
the transported probe's false-positive rate at a *matched* positive rate. AUROC
answers whether the probe still ranks; epsilon is what the false-agreement
identity needs, because it enters as `p2' = p2 + eps` in the same units as p2
(paper/theory.tex §2.5). Experiment 1 then reports false agreement twice, native
and transported, and the gap is the price of the map.

CKA and map R^2 are reported as context, not as results: per the HELIX finding,
tokenizer compatibility and size gap drive alignment, so a weak cross-family
transfer at low CKA is a map problem, not necessarily a probe problem.

    python experiments/gate_b_transport.py --pair cross-family --dataset truthfulqa
"""

from __future__ import annotations

import numpy as np

from common import (base_parser, header, make_run, resolve_pair, write_result)

from cmb import metrics

TOLERANCE = 0.05


def main() -> int:
    ap = base_parser(__doc__.splitlines()[0])
    ap.add_argument("--tolerance", type=float, default=TOLERANCE)
    args = ap.parse_args()
    a, b = resolve_pair(args.pair)

    run = make_run(a, b, args).build(refresh=args.refresh)
    gt = run.labels_test

    native_a = metrics.auroc(run.native_belief("a"), gt)
    native_b = metrics.auroc(run.native_belief("b"), gt)
    transfer_to_b = metrics.auroc(run.transported_belief("a"), gt)   # A's probe on B
    transfer_to_a = metrics.auroc(run.transported_belief("b"), gt)   # B's probe on A

    # epsilon at matched positive rate: hold the fraction of items the probe
    # calls True fixed, then read off how much the false-positive rate grows.
    # Matching the rate rather than the threshold keeps the two probes'
    # arbitrary score scales out of the comparison.
    eps = {}
    for src, native_score in (("a_to_b", run.native_belief("b")),
                              ("b_to_a", run.native_belief("a"))):
        trans = (run.transported_belief("a") if src == "a_to_b"
                 else run.transported_belief("b"))
        v_native = (native_score >= 0.5).astype(int)
        rate = float(v_native.mean())
        thr = metrics.threshold_at_positive_rate(trans, rate) if 0 < rate < 1 else 0.5
        fp_native = metrics.false_positive_rate(v_native, gt)
        fp_trans = metrics.false_positive_rate((trans >= thr).astype(int), gt)
        eps[src] = {"positive_rate": rate, "threshold": thr,
                    "fp_native": fp_native, "fp_transported": fp_trans,
                    "epsilon": float(fp_trans - fp_native)}

    Xa = run.acts_a.pos[run.layer_a][run.te]
    Xb = run.acts_b.pos[run.layer_b][run.te]
    ctx = {"cka": run.cka(),
           "map_r2_a_to_b": run.map_ab.r2(Xa, Xb),
           "map_r2_b_to_a": run.map_ba.r2(Xb, Xa)}

    header(f"GATE B — does the map carry the probe?   [{a} | {b}]  {args.dataset}")
    print(f"  pair kind: {run.kind}")
    print(f"  native AUROC   A={native_a:.3f}   B={native_b:.3f}")
    print(f"  transfer AUROC A->B={transfer_to_b:.3f}  (gap {transfer_to_b - native_b:+.3f})")
    print(f"  transfer AUROC B->A={transfer_to_a:.3f}  (gap {transfer_to_a - native_a:+.3f})")
    print(f"\n  epsilon — false-positive inflation at matched positive rate:")
    for src, e in eps.items():
        print(f"    {src:<8} p_false native {e['fp_native']:.3f} -> "
              f"transported {e['fp_transported']:.3f}   eps {e['epsilon']:+.3f}"
              f"   (positive rate held at {e['positive_rate']:.2f})")
    print(f"    this is the term Exp 1 adds to p2 when it reports transported "
          f"false agreement")
    print(f"\n  map context (not a result on its own):")
    print(f"    linear CKA(A,B)   {ctx['cka']:.3f}")
    print(f"    map R^2  A->B     {ctx['map_r2_a_to_b']:.3f}")
    print(f"    map R^2  B->A     {ctx['map_r2_b_to_a']:.3f}")

    gap_ab = native_b - transfer_to_b
    gap_ba = native_a - transfer_to_a
    passed = bool(gap_ab <= args.tolerance and gap_ba <= args.tolerance)
    print(f"\n  VERDICT: {'PASS' if passed else 'FAIL'} — "
          + (f"transfer stays within {args.tolerance} of native both ways."
             if passed else
             "transport collapses. Single-model story only; drop the "
             "cross-model framing for this pair (DESIGN.md §7 rule 3). Check "
             "CKA first — a low value points at the map, not the probe."))

    write_result("gate_b_transport",
                 {"pair": [a, b], "kind": run.kind, "dataset": args.dataset,
                  "layer": [run.layer_a, run.layer_b],
                  "native": {"a": native_a, "b": native_b},
                  "transfer": {"a_to_b": transfer_to_b, "b_to_a": transfer_to_a},
                  "gap": {"a_to_b": gap_ab, "b_to_a": gap_ba},
                  "epsilon": eps,
                  "context": ctx, "tolerance": args.tolerance,
                  "passed": passed}, args.tag)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
