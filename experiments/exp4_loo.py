#!/usr/bin/env python3
"""Experiment 4 (LOO) — does the false-agreement signature exist at all?

The pairwise matrix in `exp4_generalization_matrix.py` fits the direction on one
dataset at a time, which leaves 5-100 false agreements to learn from and, on the
easy datasets, as few as 7 to test on. Cells that small cannot carry a verdict.

This is the leave-one-dataset-out form: pool every other dataset to fit, test on
the held-out one. Fitting gets a few hundred positives instead of a few dozen, so
a failure here is about the signature rather than about sample size.

**Nothing crosses the boundary.** For each held-out dataset, the probes, the
alignment map, and the direction are all fitted on the remaining datasets only.
The held-out dataset's activations are never seen by any of the three, so the
transfer is clean in a way the pairwise matrix could not be.

Within the fitting pool the probes are fitted on the train split and the direction
on the test split, so the direction never learns from items whose verdicts the
probe could have memorized.

    python experiments/exp4_loo.py --pair same-family --probe mass-mean
"""

from __future__ import annotations

import numpy as np

from common import (base_parser, header, map_datasets, parse_n, resolve_pair,
                    write_result)

from cmb import align, metrics, probes
from cmb.cache import cache_path, load
from cmb.config import MATRIX_DATASETS, TEST_FRAC, resolve_layer
from cmb.data import load_items, split_items
from cmb.probes import LinearDirection


def load_pair(model_a, model_b, ds, layer_spec, pooling):
    """Both models' activations for one dataset, aligned item-for-item."""
    a = load(cache_path(model_a, ds, None))
    b = load(cache_path(model_b, ds, None)).reorder_to(a.item_ids)
    if pooling == "last":
        a, b = a.with_pooling("last"), b.with_pooling("last")
    la = resolve_layer(layer_spec, a.n_layers)
    lb = resolve_layer(layer_spec, b.n_layers)
    items = {it.item_id: it for it in load_items(ds, None)}
    ordered = [items[i] for i in a.item_ids]
    train_items, _ = split_items(ordered, TEST_FRAC)
    tr = np.array([i in {t.item_id for t in train_items} for i in a.item_ids])
    return a, b, la, lb, tr


def features(a, b, la, lb, mask, map_ba):
    """Aligned feature space: contrast difference and contrast mean, both models.

    Each probe votes on its *own* model's activations (native); the map appears
    only here, to put both models' features in one space so a single direction can
    be fitted across them.

    Same construction as `PairRun.aligned_features` — the mean is kept because a
    shared prominence confound lives in the part of the pair the difference
    cancels, which is exactly what Row 2 is made of.
    """
    pa, na = a.pos[la][mask], a.neg[la][mask]
    pb, nb = map_ba(b.pos[lb][mask]), map_ba(b.neg[lb][mask])
    return np.concatenate([pa - na, (pa + na) / 2, pb - nb, (pb + nb) / 2], axis=1)


def main() -> int:
    ap = base_parser(__doc__.splitlines()[0])
    ap.add_argument("--datasets", default=",".join(MATRIX_DATASETS))
    ap.add_argument("--min-row2", type=int, default=20,
                    help="refuse to score a held-out set with fewer Row-2 items")
    args = ap.parse_args()
    a_key, b_key = resolve_pair(args.pair)
    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    layer = args.layer

    # Load once; every fold reuses these.
    data = {}
    for ds in datasets:
        try:
            data[ds] = load_pair(a_key, b_key, ds, layer, args.pooling)
        except Exception as e:
            print(f"  [skip] {ds}: {type(e).__name__}: {e}")
    datasets = [d for d in datasets if d in data]

    header(f"EXP 4 (LOO) — fit on all but one, test on the held-out dataset"
           f"   [{a_key} | {b_key}]  probe={args.probe}")

    rows = []
    for held in datasets:
        pool = [d for d in datasets if d != held]

        # -- probes and map, fitted on the pool's TRAIN split only -------------
        Xa_tr, Xb_tr, ya_tr = [], [], []
        for ds in pool:
            a, b, la, lb, tr = data[ds]
            Xa_tr.append(np.concatenate([a.pos[la][tr], a.neg[la][tr]]))
            Xb_tr.append(np.concatenate([b.pos[lb][tr], b.neg[lb][tr]]))
        Xa_tr, Xb_tr = np.concatenate(Xa_tr), np.concatenate(Xb_tr)
        map_ab, map_ba = align.fit_map_both_ways(Xa_tr, Xb_tr)

        probe_a = probes.make_probe(args.probe, Xa_tr.shape[1])
        probe_b = probes.make_probe(args.probe, Xb_tr.shape[1])
        for probe, acts_key in ((probe_a, 0), (probe_b, 1)):
            P, N, Y = [], [], []
            for ds in pool:
                a, b, la, lb, tr = data[ds]
                src, l = (a, la) if acts_key == 0 else (b, lb)
                P.append(src.pos[l][tr]); N.append(src.neg[l][tr]); Y.append(src.labels[tr])
            P, N, Y = np.concatenate(P), np.concatenate(N), np.concatenate(Y)
            if args.probe == "ccs":
                probe.fit(P, N, seed=args.seed)
            else:
                probe.fit(P, N, Y, seed=args.seed)
            probe.resolve_sign(P, N, Y)

        # -- the direction, fitted on the pool's TEST split --------------------
        Xd, yd = [], []
        for ds in pool:
            a, b, la, lb, tr = data[ds]
            te = ~tr
            v1 = probe_a.vote(a.pos[la][te], a.neg[la][te])
            v2 = probe_b.vote(b.pos[lb][te], b.neg[lb][te])
            gt = a.labels[te]
            both = metrics.both_true_mask(v1, v2)
            idx = np.where(te)[0][both]
            sel = np.zeros(len(a.labels), bool); sel[idx] = True
            Xd.append(features(a, b, la, lb, sel, map_ba))
            yd.append((gt[both] == 0).astype(int))
        Xd, yd = np.concatenate(Xd), np.concatenate(yd)

        # -- score on the held-out dataset ------------------------------------
        a, b, la, lb, tr = data[held]
        v1 = probe_a.vote(a.pos[la], a.neg[la])
        v2 = probe_b.vote(b.pos[lb], b.neg[lb])
        both = metrics.both_true_mask(v1, v2)
        Xh = features(a, b, la, lb, both, map_ba)
        yh = (a.labels[both] == 0).astype(int)

        n_fit, n_test = int(yd.sum()), int(yh.sum())
        if len(np.unique(yd)) < 2 or len(np.unique(yh)) < 2:
            print(f"  {held:<20} unscoreable (one class absent)")
            continue
        d = LinearDirection.fit(Xd, yd)
        auroc = metrics.auroc(d.score(Xh), yh)
        # chance control: the same fit on permuted labels
        rng = np.random.default_rng(args.seed)
        shuf = np.mean([metrics.auroc(
            LinearDirection.fit(Xd, rng.permutation(yd)).score(Xh), yh)
            for _ in range(5)])
        flag = "" if n_test >= args.min_row2 else f"  <-- only {n_test} Row-2, not scoreable"
        print(f"  held out {held:<20} AUROC {auroc:.3f}  shuffled {shuf:.3f}"
              f"   (fit on {n_fit} Row-2 from {len(pool)} datasets, "
              f"test {n_test}/{len(yh)}){flag}")
        rows.append({"held_out": held, "auroc": auroc, "shuffled": shuf,
                     "n_row2_fit": n_fit, "n_row2_test": n_test,
                     "n_both_true_test": int(len(yh)), "pool": pool,
                     "scoreable": n_test >= args.min_row2})

    ok = [r for r in rows if r["scoreable"]]
    if ok:
        m = float(np.mean([r["auroc"] for r in ok]))
        ms = float(np.mean([r["shuffled"] for r in ok]))
        print(f"\n  mean over scoreable folds ({len(ok)}/{len(rows)}): "
              f"AUROC {m:.3f}  vs shuffled {ms:.3f}")
        print("  " + ("GENERALIZES — the signature survives a dataset it never saw."
                      if m > ms + 0.10 else
                      "DOES NOT GENERALIZE — pooling the other datasets does not "
                      "produce a signature that transfers. The blind spot is "
                      "idiosyncratic (DESIGN.md §7 rule 5)."))
    write_result("exp4_loo", {"pair": [a_key, b_key], "probe": args.probe,
                              "layer": layer, "pooling": args.pooling,
                              "folds": rows}, args.tag)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
