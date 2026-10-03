#!/usr/bin/env python3
"""One row per (model pair, dataset): probe, transport, error overlap, detection.

The pipeline per cell, since every column is one step of it. Split by item group
into train/test.

  1. Fit a linear probe on model A's train activations against ground truth, and
     the same on model B. `--probe lr` is logistic regression, `mass-mean` is the
     difference of class means; both are linear, they differ in the fitting rule.
  2. Fit the ridge alignment map both ways on the same train items.
  3. Score each probe on its own model's test activations -> auroc_a, auroc_b.
  4. Push B's test activations through map B->A and apply **A's probe, unchanged**
     -> auroc_a_on_b, and the mirror -> auroc_b_on_a. `transfer_loss` is the AUROC
     given up by reading the probe on the other model. That is transferability.
     CKA and map R^2 sit beside it and are different questions: CKA compares the
     two models' whole similarity structure, R^2 asks how much of one model's
     activation variance the map reconstructs. A probe needs one direction to
     survive and those grade all of them, so they disagree -- measured on
     qwen3-8b|llama-8b at CKA 0.49 and R^2 ~0.5, the transported probe lost 0.004.
  5. Threshold at 0.5 for verdicts and count the error overlap: how many items
     each got wrong, how many both got wrong, and how that splits into both
     saying true when the claim was false (the silent blind spot) and both saying
     false when it was true.
  6. Cross-detection: can A's readout tell you when B is wrong? The score is how
     much A's belief **contradicts B's verdict** -- `1 - belief_A` where B said
     true, `belief_A` where B said false -- scored against B actually being wrong.
     Scoring A's raw belief against B's correctness instead is confounded and was
     the first thing tried here: B is usually right on true claims and wrong on
     false ones, so its correctness is not monotone in A's belief and the measure
     returns chance regardless of the signal. 0.5 means A cannot see B's errors.

    scripts/pair_table.py --probe lr
    scripts/pair_table.py --probe mass-mean --tag _mm
"""

from __future__ import annotations

import argparse
import csv
import itertools
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cmb import align, metrics, probes                          # noqa: E402
from cmb.cache import cache_path, load                          # noqa: E402
from cmb.config import RESULTS_DIR, SEED, TEST_FRAC, resolve_layer  # noqa: E402
from cmb.data import load_items, split_items                    # noqa: E402

CACHE = Path("activations_cache")
ALPHA = 0.1
SIZE = {"qwen3-1.7b": 1.7, "olmo3-7b": 7, "qwen3-8b": 8, "qwen3-8b-base": 8,
        "llama-8b": 8, "gemma4-12b": 12, "gemma4-12b-base": 12,
        "qwen38-27b": 27, "gemma4-31b": 31, "qwen3-32b": 32,
        "llama31-70b": 70, "llama31-70b-base": 70, "llama31-405b": 405,
        "gemma2-9b": 9, "gptj-6b": 6}
FAMILY = {"qwen3-1.7b": "qwen", "qwen3-8b": "qwen", "qwen3-8b-base": "qwen",
          "qwen3-32b": "qwen", "qwen38-27b": "qwen", "gemma4-12b": "gemma",
          "gemma4-12b-base": "gemma", "gemma4-31b": "gemma",
          "llama-8b": "llama", "olmo3-7b": "olmo",
          "llama31-70b": "llama", "llama31-70b-base": "llama",
          "llama31-405b": "llama", "gemma2-9b": "gemma", "gptj-6b": "gptj"}

COLS = ["model_a", "model_b", "dataset", "kind", "size_a", "size_b",
        "n_train", "n_test", "n_false_test",
        "auroc_a", "auroc_b",
        "auroc_a_on_b", "transfer_loss_a_to_b",
        "auroc_b_on_a", "transfer_loss_b_to_a",
        "cka", "map_r2_a_to_b", "map_r2_b_to_a",
        "acc_a", "acc_b", "n_wrong_a", "n_wrong_b", "n_both_wrong",
        "n_both_wrong_say_true", "n_both_wrong_say_false",
        "both_wrong_rate", "jaccard_errors",
        "a_detects_b_error", "b_detects_a_error"]


def slim_load(path, pooling="mean"):
    """Read only the final layer's two halves, not the whole file.

    `cmb.cache.load` materialises every array in the npz -- three layers, two
    contrast halves, two poolings -- and a sweep that keeps one layer then throws
    eleven arrays away pays ~1GB per model-dataset for 165MB of signal. On a
    machine with little free RAM that is the difference between computing and
    swapping. npz access is lazy per key, so naming the keys avoids the rest.
    """
    z = np.load(path, allow_pickle=True)
    pre = "pos" if pooling == "mean" else "posalt"
    layers = sorted(int(k[len(pre) + 2:]) for k in z.files if k.startswith(pre + "_l"))
    L = max(layers)
    out = {"pos": z[f"{pre}_l{L}"],
           "neg": z[f"{'neg' if pooling == 'mean' else 'negalt'}_l{L}"],
           "labels": z["labels"], "p_yes": z["p_yes"],
           "ids": [str(s) for s in z["item_ids"]]}
    z.close()
    return out


def get(model, ds):
    a = slim_load(cache_path(model, ds, None))
    items = {it.item_id: it for it in load_items(ds, None)}
    train, _ = split_items([items[i] for i in a["ids"]], TEST_FRAC)
    tr_ids = {t.item_id for t in train}
    a["tr"] = np.array([i in tr_ids for i in a["ids"]])
    return a


def row(ma, mb, ds, probe_kind):
    A, B = get(ma, ds), get(mb, ds)
    if A["ids"] != B["ids"]:
        o = {k: i for i, k in enumerate(B["ids"])}
        idx = np.array([o[k] for k in A["ids"]])
        B = {"pos": B["pos"][idx], "neg": B["neg"][idx],
             "labels": B["labels"][idx], "ids": A["ids"], "tr": A["tr"]}
    tr, te = A["tr"], ~A["tr"]
    y_te = A["labels"][te]

    pa = probes.make_probe(probe_kind, A["pos"].shape[1])
    pb = probes.make_probe(probe_kind, B["pos"].shape[1])
    for p, D in ((pa, A), (pb, B)):
        p.fit(D["pos"][tr], D["neg"][tr], D["labels"][tr], seed=SEED)
        p.resolve_sign(D["pos"][tr], D["neg"][tr], D["labels"][tr])

    Xa = np.concatenate([A["pos"][tr], A["neg"][tr]])
    Xb = np.concatenate([B["pos"][tr], B["neg"][tr]])
    map_ab, map_ba = align.fit_map_both_ways(Xa, Xb, alpha=ALPHA)

    ba = pa.belief(A["pos"][te], A["neg"][te])
    bb = pb.belief(B["pos"][te], B["neg"][te])
    au_a, au_b = metrics.auroc(ba, y_te), metrics.auroc(bb, y_te)
    a_on_b = pa.belief(map_ba(B["pos"][te]), map_ba(B["neg"][te]))
    b_on_a = pb.belief(map_ab(A["pos"][te]), map_ab(A["neg"][te]))
    au_ab, au_ba = metrics.auroc(a_on_b, y_te), metrics.auroc(b_on_a, y_te)

    va, vb = (ba >= 0.5).astype(int), (bb >= 0.5).astype(int)
    wa, wb = va != y_te, vb != y_te
    both = wa & wb
    union = int((wa | wb).sum())
    # signed contradiction: high when A disagrees with the verdict B gave
    contra_a = np.where(vb == 1, 1.0 - ba, ba)   # A's objection to B's call
    contra_b = np.where(va == 1, 1.0 - bb, bb)   # B's objection to A's call

    return {"model_a": ma, "model_b": mb, "dataset": ds,
            "kind": "same-family" if FAMILY[ma] == FAMILY[mb] else "cross-family",
            "size_a": SIZE[ma], "size_b": SIZE[mb],
            "n_train": int(tr.sum()), "n_test": int(te.sum()),
            "n_false_test": int((y_te == 0).sum()),
            "auroc_a": au_a, "auroc_b": au_b,
            "auroc_a_on_b": au_ab, "transfer_loss_a_to_b": au_b - au_ab,
            "auroc_b_on_a": au_ba, "transfer_loss_b_to_a": au_a - au_ba,
            "cka": align.linear_cka(A["pos"][te], B["pos"][te]),
            "map_r2_a_to_b": map_ab.r2(A["pos"][te], B["pos"][te]),
            "map_r2_b_to_a": map_ba.r2(B["pos"][te], A["pos"][te]),
            "acc_a": float((~wa).mean()), "acc_b": float((~wb).mean()),
            "n_wrong_a": int(wa.sum()), "n_wrong_b": int(wb.sum()),
            "n_both_wrong": int(both.sum()),
            "n_both_wrong_say_true": int((both & (va == 1)).sum()),
            "n_both_wrong_say_false": int((both & (va == 0)).sum()),
            "both_wrong_rate": float(both.mean()),
            "jaccard_errors": (both.sum() / union) if union else float("nan"),
            "a_detects_b_error": metrics.auroc(contra_a, wb.astype(int))
            if len(np.unique(wb)) > 1 else float("nan"),
            "b_detects_a_error": metrics.auroc(contra_b, wa.astype(int))
            if len(np.unique(wa)) > 1 else float("nan")}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--probe", default="lr", choices=["lr", "mass-mean", "ccs"])
    ap.add_argument("--tag", default="")
    ap.add_argument("--pairs", default="")
    args = ap.parse_args()

    have = {d for d in os.listdir(CACHE) if (CACHE / d).is_dir() and d != "manifests"}
    dsets = {m: sorted(f.replace("_nall.npz", "") for f in os.listdir(CACHE / m))
             for m in have}
    pairs = [(a, b, sorted(set(dsets[a]) & set(dsets[b])))
             for a, b in itertools.combinations(sorted(have), 2)]
    pairs = [p for p in pairs if len(p[2]) >= 3]
    if args.pairs:
        lo, hi = (int(x) if x else None for x in args.pairs.split(":"))
        pairs = pairs[lo:hi]

    path = RESULTS_DIR / f"pair_table{args.tag}.csv"
    done = set()
    if path.exists():
        with path.open() as f:
            done = {(r["model_a"], r["model_b"], r["dataset"])
                    for r in csv.DictReader(f)}
    print(f"{len(pairs)} pairs, probe={args.probe}, {len(done)} cells done",
          flush=True)

    for i, (ma, mb, dss) in enumerate(pairs):
        t0 = time.time()
        for ds in dss:
            if (ma, mb, ds) in done:
                continue
            try:
                r = row(ma, mb, ds, args.probe)
            except Exception as e:
                print(f"  FAIL {ma}|{mb}/{ds}: {type(e).__name__}: {e}", flush=True)
                continue
            new = not path.exists()
            with path.open("a", newline="") as f:
                w = csv.DictWriter(f, fieldnames=COLS)
                if new:
                    w.writeheader()
                w.writerow(r)
        print(f"[{i + 1}/{len(pairs)}] {ma} | {mb}  {time.time() - t0:.0f}s",
              flush=True)
    print(f"-> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
