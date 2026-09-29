#!/usr/bin/env python3
"""Operating points — what the detector buys an operator, in items routed.

AUROC says the direction ranks untrustworthy agreement above trustworthy
agreement. It does not say what happens if you deploy it, because deployment is a
budget: you can afford to route some fraction of agreeing items to an expensive
check, and you want to know what share of the blind spot that buys.

The chain, per DESIGN.md §2.1:

  * the two probes **disagree**  -> the router already fires. Free coverage.
  * the two probes **agree true** -> the router is silent. Row 2 hides here, and
    this is the only place the direction can help.

So the table reports, for each budget b:

  `caught`   share of ALL false claims flagged = router coverage + what the
             direction adds inside the silent region
  `precision` of the direction's own flags, i.e. how many routed items are
             actually false agreements rather than wasted checks
  `wasted`   items routed per true positive found — the operator's real cost

The budget is a fraction of the *agree-true* set, not of everything, because that
is the set the operator is choosing to spend extra checks on.

    python experiments/exp5_operating_point.py --pair same-family --probe mass-mean
"""

from __future__ import annotations

import numpy as np

from common import base_parser, header, resolve_pair, write_result
from exp4_loo import features, load_pair

from cmb import align, metrics, probes
from cmb.config import MATRIX_DATASETS
from cmb.probes import LinearDirection

BUDGETS = (0.05, 0.10, 0.20, 0.30, 0.50)


def main() -> int:
    ap = base_parser(__doc__.splitlines()[0])
    ap.add_argument("--datasets", default=",".join(MATRIX_DATASETS))
    ap.add_argument("--held-out", default="truthfulqa",
                    help="dataset to deploy on; everything else is used to fit")
    args = ap.parse_args()
    a_key, b_key = resolve_pair(args.pair)
    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]

    data = {}
    for ds in datasets:
        try:
            data[ds] = load_pair(a_key, b_key, ds, args.layer, args.pooling)
        except Exception as e:
            print(f"  [skip] {ds}: {type(e).__name__}")
    pool = [d for d in data if d != args.held_out]

    # fit probes + map on the pool's train split (nothing sees the held-out set)
    Xa, Xb = [], []
    for ds in pool:
        a, b, la, lb, tr = data[ds]
        Xa.append(np.concatenate([a.pos[la][tr], a.neg[la][tr]]))
        Xb.append(np.concatenate([b.pos[lb][tr], b.neg[lb][tr]]))
    map_ab, map_ba = align.fit_map_both_ways(np.concatenate(Xa), np.concatenate(Xb))

    fitted = []
    for which in (0, 1):
        P, N, Y = [], [], []
        for ds in pool:
            a, b, la, lb, tr = data[ds]
            src, l = (a, la) if which == 0 else (b, lb)
            P.append(src.pos[l][tr]); N.append(src.neg[l][tr]); Y.append(src.labels[tr])
        P, N, Y = np.concatenate(P), np.concatenate(N), np.concatenate(Y)
        p = probes.make_probe(args.probe, P.shape[1])
        p.fit(P, N, seed=args.seed) if args.probe == "ccs" else p.fit(P, N, Y, seed=args.seed)
        p.resolve_sign(P, N, Y)
        fitted.append(p)
    probe_a, probe_b = fitted

    # direction, fitted on the pool's test split
    Xd, yd = [], []
    for ds in pool:
        a, b, la, lb, tr = data[ds]
        te = ~tr
        v1 = probe_a.vote(a.pos[la][te], a.neg[la][te])
        v2 = probe_b.vote(b.pos[lb][te], b.neg[lb][te])
        both = metrics.both_true_mask(v1, v2)
        sel = np.zeros(len(a.labels), bool); sel[np.where(te)[0][both]] = True
        Xd.append(features(a, b, la, lb, sel, map_ba))
        yd.append((a.labels[te][both] == 0).astype(int))
    direction = LinearDirection.fit(np.concatenate(Xd), np.concatenate(yd))

    # -- deploy on the held-out dataset -----------------------------------------
    a, b, la, lb, _ = data[args.held_out]
    v1 = probe_a.vote(a.pos[la], a.neg[la])
    v2 = probe_b.vote(b.pos[lb], b.neg[lb])
    gt = a.labels
    agree_true = metrics.both_true_mask(v1, v2)
    disagree = v1 != v2
    n_false = int((gt == 0).sum())

    # what the router alone gets: every false claim on which the probes disagree
    router_catch = int(((gt == 0) & disagree).sum())
    row2 = int(((gt == 0) & agree_true).sum())

    score = direction.score(features(a, b, la, lb, agree_true, map_ba))
    is_row2 = (gt[agree_true] == 0).astype(int)

    header(f"OPERATING POINTS — deploy on {args.held_out}   [{a_key} | {b_key}]"
           f"  probe={args.probe}")
    print(f"  {args.held_out}: {len(gt)} items, {n_false} false")
    print(f"  probes disagree on {int(disagree.sum())} items -> router fires, "
          f"catching {router_catch}/{n_false} false claims "
          f"({router_catch / n_false:.1%})")
    print(f"  probes agree 'true' on {int(agree_true.sum())} items -> router "
          f"silent; {row2} of those are false (Row 2)")
    print(f"  direction AUROC inside that silent set: "
          f"{metrics.auroc(score, is_row2):.3f}\n")

    print(f"  {'budget':>8}{'routed':>9}{'Row-2 found':>13}{'recall|Row2':>13}"
          f"{'precision':>11}{'checks/hit':>12}{'total caught':>14}")
    rows = []
    order = np.argsort(-score)
    for bfrac in BUDGETS:
        k = max(1, int(round(bfrac * len(score))))
        picked = order[:k]
        hits = int(is_row2[picked].sum())
        prec = hits / k
        total = (router_catch + hits) / n_false
        print(f"  {bfrac:>7.0%}{k:>9}{hits:>13}{hits / max(row2,1):>13.1%}"
              f"{prec:>11.1%}{(k / max(hits,1)):>12.1f}{total:>14.1%}")
        rows.append({"budget": bfrac, "routed": k, "row2_found": hits,
                     "recall_row2": hits / max(row2, 1), "precision": prec,
                     "checks_per_hit": k / max(hits, 1), "total_caught": total})

    base = row2 / max(int(agree_true.sum()), 1)
    print(f"\n  base rate inside the silent set: {base:.1%} "
          f"(a random check would hit this often)")
    print(f"  router alone catches {router_catch / n_false:.1%} of all false "
          f"claims; the rest is what the direction can bid for.")

    write_result("exp5_operating_point",
                 {"pair": [a_key, b_key], "held_out": args.held_out,
                  "probe": args.probe, "n_false": n_false, "row2": row2,
                  "router_catch": router_catch, "base_rate": base,
                  "auroc": metrics.auroc(score, is_row2), "budgets": rows},
                 args.tag)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
