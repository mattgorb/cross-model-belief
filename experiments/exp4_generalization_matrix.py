#!/usr/bin/env python3
"""Experiment 4 — the generalization matrix (the honesty test).

Leave-one-dataset-out: fit the false-agreement direction on dataset i, test it
on dataset j, all pairs, as a full AUROC matrix. The diagonal is in-distribution
and nearly useless; **the off-diagonal is the result**, and the *pattern* is the
finding — graceful degradation with domain distance, a cliff at domain
boundaries, or flat noise.

IMDB -> TruthfulQA is the designated hardest cell and the headline honesty
check: a sentiment confound asked to predict factual misconception.

**Attribution control (DESIGN.md §4).** Probe, map, and false-agreement
direction are three separately fitted objects. Probes and maps are refit *per
cell* on that cell's train split, so the only object that crosses a dataset
boundary is the direction itself. Without this, a failed cell is unattributable.

    python experiments/exp4_generalization_matrix.py --pair cross-family
    python experiments/exp4_generalization_matrix.py --pair cross-family --held-out-pair
"""

from __future__ import annotations

import numpy as np

from common import (PairRun, base_parser, header, layer_specs,
                    resolve_pair, write_result)

from cmb import align, metrics
from cmb.config import MATRIX_DATASETS, PAIRS
from cmb.probes import LinearDirection
from exp3_separability import subset_for


def build_cells(pair, args, datasets):
    """One PairRun per dataset — refit from scratch, per the attribution control."""
    a, b = pair
    cells = {}
    for ds in datasets:
        la, lb = layer_specs(args)
        run = PairRun(a, b, ds, la, lb, args.n, args.synthetic,
                      args.seed).build(refresh=args.refresh)
        X_tr, y_tr, _ = subset_for(run, "train", args.mode)
        X_te, y_te, _ = subset_for(run, "test", args.mode)
        cells[ds] = {"run": run, "train": (X_tr, y_tr), "test": (X_te, y_te)}
        print(f"  [{ds}] both-True train n={len(y_tr)} (Row 2 {int(y_tr.sum())})"
              f"  test n={len(y_te)} (Row 2 {int(y_te.sum())})")
    return cells


def matrix(cells, datasets, C, seed):
    """AUROC[i][j]: direction fitted on i, evaluated on j's held-out split."""
    M = np.full((len(datasets), len(datasets)), np.nan)
    for i, src in enumerate(datasets):
        X_tr, y_tr = cells[src]["train"]
        if len(np.unique(y_tr)) < 2 or y_tr.sum() < 5:
            continue
        d = LinearDirection.fit(X_tr, y_tr, C=C, seed=seed)
        for j, tgt in enumerate(datasets):
            X_te, y_te = cells[tgt]["test"]
            if len(np.unique(y_te)) < 2:
                continue
            M[i, j] = metrics.auroc(d.score(X_te), y_te)
    return M


def render(M, datasets) -> str:
    w = max(len(d) for d in datasets) + 2
    lines = ["  " + "fit \\ test".ljust(w)
             + "".join(f"{d[:11]:>12}" for d in datasets),
             "  " + "-" * (w + 12 * len(datasets))]
    for i, d in enumerate(datasets):
        cells = "".join("         nan" if np.isnan(M[i, j]) else f"{M[i, j]:>12.3f}"
                        for j in range(len(datasets)))
        lines.append(f"  {d:<{w}}{cells}")
    return "\n".join(lines)


def held_out_pair_transfer(cells, datasets, args, source_pair):
    """Transfer the direction to a model pair it was never fitted on.

    The two pairs' aligned spaces are different vector spaces, so an extra
    anchor map is needed. It is fitted on the *train* split of a shared dataset
    and reported with its R^2 — this map is itself a confound, and a weak
    transfer at low R^2 says nothing about the direction.
    """
    ho_a, ho_b = PAIRS["held-out"]
    if {ho_a, ho_b} & set(source_pair):
        print(f"\n  NOTE: held-out pair ({ho_a},{ho_b}) shares a model with the "
              f"source pair — treat this as a weak independence test.")
    out = {}
    for ds in datasets:
        la, lb = layer_specs(args)
        run = PairRun(ho_a, ho_b, ds, la, lb, args.n, args.synthetic,
                      args.seed).build(refresh=args.refresh)
        src = cells[ds]
        X_src_tr, y_src_tr = src["train"]
        X_ho_tr, _, _ = subset_for(run, "train", args.mode)
        X_ho_te, y_ho_te = subset_for(run, "test", args.mode)[:2]
        if len(np.unique(y_src_tr)) < 2 or y_src_tr.sum() < 5 or len(y_ho_te) < 10:
            out[ds] = {"auroc": float("nan"), "anchor_r2": float("nan")}
            continue
        n = min(len(X_ho_tr), len(X_src_tr))
        anchor = align.fit_map(X_ho_tr[:n], X_src_tr[:n])
        d = LinearDirection.fit(X_src_tr, y_src_tr, C=args.C, seed=args.seed)
        out[ds] = {"auroc": metrics.auroc(d.score(anchor(X_ho_te)), y_ho_te),
                   "anchor_r2": anchor.r2(X_ho_tr[:n], X_src_tr[:n]),
                   "n_test": int(len(y_ho_te)), "row2_test": int(y_ho_te.sum())}
    return out


def main() -> int:
    ap = base_parser(__doc__.splitlines()[0])
    ap.add_argument("--mode", default="native",
                    choices=["native", "a_to_b", "b_to_a"])
    ap.add_argument("--C", type=float, default=0.1)
    ap.add_argument("--datasets", default=",".join(MATRIX_DATASETS))
    ap.add_argument("--held-out-pair", action="store_true",
                    help="also transfer the direction to the held-out model pair")
    args = ap.parse_args()
    pair = resolve_pair(args.pair)
    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]

    header(f"EXP 4 — generalization matrix   [{pair[0]} | {pair[1]}]")
    print("  refitting probe + map per cell (attribution control)\n")
    cells = build_cells(pair, args, datasets)

    M = matrix(cells, datasets, args.C, args.seed)
    print("\n  AUROC of the false-agreement direction, fit on row, tested on column:\n")
    print(render(M, datasets))

    diag = np.nanmean(np.diag(M))
    off = M.copy()
    np.fill_diagonal(off, np.nan)
    off_mean = float(np.nanmean(off))
    print(f"\n  diagonal mean (in-distribution)  {diag:.3f}")
    print(f"  OFF-DIAGONAL mean (the result)   {off_mean:.3f}")

    hardest = None
    if "imdb" in datasets and "truthfulqa" in datasets:
        i, j = datasets.index("imdb"), datasets.index("truthfulqa")
        hardest = float(M[i, j])
        print(f"  IMDB -> TruthfulQA (headline honesty cell)  {hardest:.3f}")

    generalizes = bool(off_mean > 0.6)
    print(f"\n  VERDICT: {'GENERALIZES' if generalizes else 'DOES NOT GENERALIZE'}")
    print("  " + ("Untrustworthy agreement has a signature that survives a "
                  "dataset change — this is the result (DESIGN.md §7 rule 4)."
                  if generalizes else
                  "Each false agreement is idiosyncratic. Cross-model belief "
                  "agreement has an irreducible blind spot; report the cap "
                  "honestly (DESIGN.md §7 rule 5)."))

    payload = {"pair": list(pair), "datasets": datasets, "mode": args.mode,
               "matrix": M.tolist(), "diagonal_mean": float(diag),
               "off_diagonal_mean": off_mean, "imdb_to_truthfulqa": hardest,
               "generalizes": generalizes}

    if args.held_out_pair:
        header("Held-out model pair transfer")
        ho = held_out_pair_transfer(cells, datasets, args, pair)
        for ds, r in ho.items():
            print(f"  fit on {ds:<18} -> held-out pair  AUROC {r['auroc']:.3f}"
                  f"   (anchor map R^2 {r['anchor_r2']:.3f})")
        payload["held_out_pair"] = {"pair": list(PAIRS["held-out"]), "cells": ho}

    write_result("exp4_generalization_matrix", payload, args.tag)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
