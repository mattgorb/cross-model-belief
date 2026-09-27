#!/usr/bin/env python3
"""Experiment 3 — is Row 2 separable? (the bet)

Inside the {both probes say True} subset, fit a linear classifier to separate
GT-true from GT-false in the aligned activation space. The question: does
"true-because-true" look different internally from
"true-because-shared-confound"?

Two controls are run alongside, because a positive result here is easy to fake:

  * **shuffled-label control** — the same fit with permuted labels, which
    bounds how much of the separability is small-sample overfitting.
  * **output-confidence baseline** (sign-resolved) — if P(Yes) alone separates
    the subset just as well, nothing about the activation space is doing the
    work. Note the orientation is not obvious a priori: when the confound that
    creates Row 2 also inflates P(Yes), it is *high* confidence that flags it.

A `--prominence` mode implements the third DESIGN.md §4 mitigation: instead of
fitting the rare class from scratch, get a prominence direction from familiar-
vs-defamiliarized pairs and test whether Row-2 items lie along it.

    python experiments/exp3_separability.py --pair cross-family --dataset truthfulqa
"""

from __future__ import annotations

import numpy as np

from common import (base_parser, header, make_run, resolve_pair, write_result)

from cmb import metrics
from cmb.probes import LinearDirection
from cmb.config import SEED


def fit_and_score(X_tr, y_tr, X_te, y_te, C=1.0, seed=SEED):
    if len(np.unique(y_tr)) < 2 or len(np.unique(y_te)) < 2:
        return float("nan"), None
    d = LinearDirection.fit(X_tr, y_tr, C=C, seed=seed)
    # y = 1 means GT-false, i.e. the score points at "untrustworthy agreement".
    return metrics.auroc(d.score(X_te), y_te), d


def subset_for(run, split, mode):
    """Aligned features, GT-false flag, and P(Yes) for the {both True} subset."""
    v1, v2, gt = run.verdicts(mode, split)
    mask = metrics.both_true_mask(v1, v2)
    X = run.aligned_features(split)[mask]
    y = (gt[mask] == 0).astype(int)            # 1 = Row 2
    sel = run.te if split == "test" else run.tr
    conf = ((run.acts_a.p_yes[sel] + run.acts_b.p_yes[sel]) / 2)[mask]
    return X, y, conf


def main() -> int:
    ap = base_parser(__doc__.splitlines()[0])
    ap.add_argument("--mode", default="native",
                    choices=["native", "a_to_b", "b_to_a"])
    ap.add_argument("--C", type=float, default=0.1,
                    help="inverse regularization; Row 2 is small, keep it strong")
    args = ap.parse_args()
    a, b = resolve_pair(args.pair)

    run = make_run(a, b, args).build(refresh=args.refresh)

    X_tr, y_tr, _ = subset_for(run, "train", args.mode)
    X_te, y_te, conf_te = subset_for(run, "test", args.mode)

    header(f"EXP 3 — is Row 2 separable?   [{a} | {b}]  {args.dataset}")
    print(f"  both-say-True subset: train n={len(y_tr)} (Row 2: {int(y_tr.sum())})"
          f"   test n={len(y_te)} (Row 2: {int(y_te.sum())})")
    if int(y_tr.sum()) < 20 or int(y_te.sum()) < 10:
        print("  WARNING: Row 2 is too small here to support a claim either way.")
        print("  DESIGN.md §4 mitigations: confound-rich data (TruthfulQA,")
        print("  popular falsehoods), deliberate Row-2 enrichment, or borrow the")
        print("  prominence direction instead of fitting the rare class.")

    auc, direction = fit_and_score(X_tr, y_tr, X_te, y_te, args.C, args.seed)

    rng = np.random.default_rng(args.seed)
    shuffled = [fit_and_score(X_tr, rng.permutation(y_tr), X_te, y_te,
                              args.C, args.seed)[0] for _ in range(10)]
    shuffled_mean = float(np.nanmean(shuffled))
    # Sign-resolved: the control gets its best shot. In a world where the
    # confound driving Row 2 also drives P(Yes), *high* confidence flags Row 2,
    # and a fixed orientation would flatter the probe by accident.
    conf_auc = metrics.auroc_signed(conf_te, y_te)

    print(f"\n  separability AUROC (held-out)      {auc:.3f}")
    print(f"  shuffled-label control (mean of 10) {shuffled_mean:.3f}")
    print(f"  output-confidence baseline          {conf_auc:.3f}")
    margin = auc - max(shuffled_mean, conf_auc)
    print(f"  margin over the better control      {margin:+.3f}")

    separable = bool(margin > 0.05 and auc > 0.6)
    print(f"\n  VERDICT: {'SEPARABLE' if separable else 'NOT SEPARABLE'} "
          f"in-distribution.")
    print("  " + ("Necessary, not sufficient — the claim is generalization. "
                  "Run exp4_generalization_matrix.py next."
                  if separable else
                  "Within this dataset, false agreement has no linear signature "
                  "beyond the controls. If this holds across datasets, report "
                  "the honest cap (DESIGN.md §7 rule 5)."))

    payload = {"pair": [a, b], "kind": run.kind, "dataset": args.dataset,
               "mode": args.mode, "C": args.C,
               "n_train": int(len(y_tr)), "n_test": int(len(y_te)),
               "row2_train": int(y_tr.sum()), "row2_test": int(y_te.sum()),
               "auroc": auc, "shuffled_control": shuffled_mean,
               "confidence_baseline": conf_auc, "margin": margin,
               "separable": separable}
    if direction is not None:
        np.save(run_direction_path(args), direction.w)
        payload["direction_norm"] = float(np.linalg.norm(direction.w))
    write_result("exp3_separability", payload, args.tag)
    return 0


def run_direction_path(args):
    from cmb.config import RESULTS_DIR
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    return RESULTS_DIR / f"exp3_direction_{args.dataset}{('_' + args.tag) if args.tag else ''}.npy"


if __name__ == "__main__":
    raise SystemExit(main())
