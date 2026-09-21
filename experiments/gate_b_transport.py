#!/usr/bin/env python3
"""Gate B — does the map carry the probe? (DESIGN.md §4)

Fit the probe on A, transport it to B through the linear map, test on B.
Pass = transfer AUROC within ~0.05 of native-B AUROC. Fail = the cross-model
framing is dead weight for this pair; fall back to native probes and say so
(DESIGN.md §7 rule 3).

CKA and map R^2 are reported as context, not as results: per the HELIX finding,
tokenizer compatibility and size gap drive alignment, so a weak cross-family
transfer at low CKA is a map problem, not necessarily a probe problem.

    python experiments/gate_b_transport.py --pair cross-family --dataset truthfulqa
"""

from __future__ import annotations

import numpy as np

from common import PairRun, base_parser, header, resolve_pair, write_result

from cmb import metrics

TOLERANCE = 0.05


def main() -> int:
    ap = base_parser(__doc__.splitlines()[0])
    ap.add_argument("--tolerance", type=float, default=TOLERANCE)
    args = ap.parse_args()
    a, b = resolve_pair(args.pair)

    run = PairRun(a, b, args.dataset, args.layer_frac, args.n,
                  args.synthetic, args.seed).build(refresh=args.refresh)
    gt = run.labels_test

    native_a = metrics.auroc(run.native_belief("a"), gt)
    native_b = metrics.auroc(run.native_belief("b"), gt)
    transfer_to_b = metrics.auroc(run.transported_belief("a"), gt)   # A's probe on B
    transfer_to_a = metrics.auroc(run.transported_belief("b"), gt)   # B's probe on A

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
                  "context": ctx, "tolerance": args.tolerance,
                  "passed": passed}, args.tag)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
