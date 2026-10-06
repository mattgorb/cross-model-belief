#!/usr/bin/env python3
"""Run the whole pipeline over every sensible (overseer, target) pair -> CSV.

Two phases per pair, both written incrementally so the sweep is resumable: rerun
the same command and finished rows are skipped.

  **base**  (results/sweep_base.csv)  one row per (pair, dataset). Probes and map
            fitted on that dataset's train split, everything scored on its test
            split: Gate A edges, native and transported AUROC both directions,
            and the false-agreement decomposition of DESIGN.md §2.4.

  **loo**   (results/sweep_loo.csv)   one row per (pair, held-out dataset). The
            probes, the map and the false-agreement direction are fitted on the
            *other* datasets only, then deployed on the held-out one: direction
            AUROC against a shuffled control, plus operating points — what share
            of false claims a router catches free, and what a budget of extra
            checks inside the silent region buys.

Pairs are ordered (overseer, target) and only run when the overseer is no larger
than the target: a big model watching a small one is not the oversight question.

    scripts/sweep.py --workers 2
    scripts/sweep.py --phase loo --pairs 0:10
"""

from __future__ import annotations

import argparse
import csv
import itertools
import os
import sys
import time
from multiprocessing import Process
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cmb import align, metrics, probes                        # noqa: E402
from cmb.cache import cache_path, load                        # noqa: E402
from cmb.config import RESULTS_DIR, SEED, TEST_FRAC, resolve_layer  # noqa: E402
from cmb.data import load_items, split_items                   # noqa: E402
from cmb.probes import LinearDirection                         # noqa: E402

CACHE = Path("activations_cache")
BUDGETS = (0.05, 0.10, 0.20)
# The permutation control is noisy: a direction fitted on permuted labels is
# effectively a random projection of a 16k-dimensional space, and its AUROC on the
# held-out set has a wide distribution when the classes differ along any dominant
# nuisance axis. Two reps gave controls of 0.62-0.70 where ~0.5 was expected, which
# is variance, not bias -- so take more reps and carry the spread, and treat any
# margin smaller than the control's own sd as unmeasured.
SHUFFLES = 4
# Fixed ridge strength instead of a held-out search: across the runs that chose it
# the selection landed on 0.01-1.0 and the transfer verdicts did not move, so the
# 5x cost of searching per cell is not worth it at sweep scale.
ALPHA = 0.1
# Cap on the rows used to fit the false-agreement direction. The feature space is
# 4 x d columns (20k for the largest models), so an uncapped pool of four datasets
# builds a matrix that sklearn copies several times over -- enough to push a
# laptop into swap, where the sweep makes no progress at all. Subsampled
# stratified, so the rare class is not thinned.
MAX_DIRECTION_ROWS = 4000
# None -> concatenate the two models' features instead of mapping one into the
# other's space. See `aligned`.
USE_MAP_FOR_FEATURES = True
# False -> only overseer <= target (the default). True -> every ordered pair,
# so a large model overseeing a small one is included too.
ALL_DIRECTIONS = False

SIZE = {"qwen3-1.7b": 1.7, "olmo3-7b": 7, "qwen3-8b": 8, "qwen3-8b-base": 8,
        "llama-8b": 8, "gemma4-12b": 12, "gemma4-12b-base": 12,
        "qwen38-27b": 27, "gemma4-31b": 31, "qwen3-32b": 32,
        "llama31-8b-base": 8, "llama31-70b": 70, "llama31-70b-base": 70, "llama31-405b": 405,
        "gemma2-9b": 9, "gptj-6b": 6}

BASE_COLS = ["overseer", "target", "dataset", "n", "n_false",
             "auroc_a", "auroc_b", "gate_a_edge_a", "gate_a_edge_b",
             "conf_wrong_a", "conf_wrong_b",
             "transfer_a_to_b", "gap_a_to_b", "transfer_b_to_a", "gap_b_to_a",
             "cka", "map_r2_a_to_b", "map_r2_b_to_a",
             "fa_rate", "row2_n", "p_overseer", "p_target", "rho",
             "rho_over_max", "fa_independent", "frechet_hi",
             "coverage_strong", "coverage_target"]

LOO_COLS = ["overseer", "target", "held_out", "n_pool_datasets",
            "pos_rate_overseer", "pos_rate_target",
            "row2_fit_n", "both_true_test_n", "row2_test_n", "base_rate",
            "direction_auroc", "shuffled_auroc", "shuffled_sd", "margin",
            "router_coverage", "n_false",
            "prec_05", "recall_05", "caught_05",
            "prec_10", "recall_10", "caught_10",
            "prec_20", "recall_20", "caught_20"]


# ---------------------------------------------------------------- data loading

_cache: dict = {}


def slim_load(path, pooling="mean"):
    """Read only the final layer's two halves, not the whole file.

    `cmb.cache.load` materialises every array in the npz -- three layers, two
    contrast halves, two poolings -- so a sweep that keeps one layer then discards
    eleven arrays pays ~1GB per model-dataset for 165MB of signal. On a machine
    with little free RAM that is the difference between computing and swapping
    (measured: 56s of CPU across 38 minutes of wall clock). npz access is lazy per
    key, so naming the keys avoids reading the rest.
    """
    z = np.load(path, allow_pickle=True)
    pre, npre = ("pos", "neg") if pooling == "mean" else ("posalt", "negalt")
    L = max(int(k[len(pre) + 2:]) for k in z.files if k.startswith(pre + "_l"))
    out = {"pos": z[f"{pre}_l{L}"], "neg": z[f"{npre}_l{L}"],
           "labels": z["labels"], "p_yes": z["p_yes"],
           "ids": [str(x) for x in z["item_ids"]]}
    z.close()
    return out


def get(model: str, ds: str):
    """Final-layer mean-pooled activations for one (model, dataset), trimmed.

    Only the final layer and only the mean pooling are kept: the sweep holds two
    models x five datasets in memory at once, and the full file is ~10x larger.
    """
    key = (model, ds)
    if key in _cache:
        return _cache[key]
    a = slim_load(cache_path(model, ds, None))
    items = {it.item_id: it for it in load_items(ds, None)}
    train_items, _ = split_items([items[i] for i in a["ids"]], TEST_FRAC)
    tr_ids = {t.item_id for t in train_items}
    a["tr"] = np.array([i in tr_ids for i in a["ids"]])
    _cache[key] = a
    return a


def aligned(A, B, mask, map_ba):
    """Difference and mean of each model's contrast pair, side by side.

    The mean is kept because a shared prominence confound sits in the part of the
    pair that the difference cancels, and that is what Row 2 is made of.

    `map_ba=None` concatenates the two models' features in their own coordinates
    rather than pushing B into A's space. A classifier does not need a shared
    space -- it fits one weight vector over the concatenation, and the halves may
    even have different widths -- and the map is lossy, so on a pair it transports
    badly the mapped half is partly reconstruction error. Measured: detection
    margin tracks map quality on TruthfulQA (0.039 with worse maps, 0.093 with
    better), which is a confound rather than a property of the blind spot.
    """
    pa, na = A["pos"][mask], A["neg"][mask]
    if map_ba is None:
        pb, nb = B["pos"][mask], B["neg"][mask]
    else:
        pb, nb = map_ba(B["pos"][mask]), map_ba(B["neg"][mask])
    # float32: this is 4 x d columns (20k for the largest models) and the
    # classifier copies it several times. In float64 that drove the machine into
    # swap -- 56s of CPU across 38 minutes of wall clock. lbfgs keeps float32.
    return np.concatenate([pa - na, (pa + na) / 2, pb - nb, (pb + nb) / 2],
                          1).astype(np.float32)


def calibrated_vote(probe, pos, neg, rate):
    """Vote on a dataset the probe was not fitted on, at a matched positive rate.

    A probe keeps its direction across datasets but not its threshold: activation
    scale shifts, the fitted decision boundary lands in the wrong place, and the
    probe then answers "true" almost always or almost never. AUROC hides this
    because it is threshold-free, but the both-agree subset is defined by the
    threshold, so it silently destroys the operating points -- measured on one pair
    it produced zero agreements on Geometry of Truth and 99.7% disagreement on
    BoolQ.

    The fix is label-free: hold the probe's own positive rate fixed at what it was
    where it was fitted, and read the threshold off the deployment scores. It
    assumes the probe's verdict rate is stable across datasets, not that the labels
    are balanced.
    """
    b = probe.belief(pos, neg)
    thr = metrics.threshold_at_positive_rate(b, rate) if 0 < rate < 1 else 0.5
    return (b >= thr).astype(int)


def positive_rate(probe, pos, neg):
    return float((probe.belief(pos, neg) >= 0.5).mean())


PROBE = "mass-mean"      # set from --probe in main(); module-level so the
                         # worker processes inherit it after the fork


def fit_probe(P, N, Y, seed=SEED):
    p = probes.make_probe(PROBE, P.shape[1])
    p.fit(P, N, Y, seed=seed) if PROBE != "ccs" else p.fit(P, N, seed=seed)
    p.resolve_sign(P, N, Y)
    return p


def gate_a_edge(belief, p_yes, gt):
    """Probe AUROC minus signed-confidence AUROC, on the confident half."""
    conf = metrics.confidence(p_yes)
    m = metrics.confident_slice(p_yes)
    base = metrics.auroc(conf[m] * np.sign(p_yes[m] - 0.5), gt[m])
    err = float(((p_yes[m] >= 0.5).astype(int) != gt[m]).mean())
    return metrics.auroc(belief[m], gt[m]) - base, err


# --------------------------------------------------------------------- phase A

def base_row(ov, tg, ds):
    A, B = get(ov, ds), get(tg, ds)
    if A["ids"] != B["ids"]:
        order = {k: i for i, k in enumerate(B["ids"])}
        idx = np.array([order[k] for k in A["ids"]])
        B = {**{k: (v[idx] if isinstance(v, np.ndarray) else v) for k, v in B.items()},
             "ids": A["ids"], "tr": A["tr"]}
    tr, te = A["tr"], ~A["tr"]
    gt = A["labels"][te]

    Xa = np.concatenate([A["pos"][tr], A["neg"][tr]])
    Xb = np.concatenate([B["pos"][tr], B["neg"][tr]])
    map_ab, map_ba = align.fit_map_both_ways(Xa, Xb, alpha=ALPHA)

    pa = fit_probe(A["pos"][tr], A["neg"][tr], A["labels"][tr])
    pb = fit_probe(B["pos"][tr], B["neg"][tr], B["labels"][tr])

    ba = pa.belief(A["pos"][te], A["neg"][te])
    bb = pb.belief(B["pos"][te], B["neg"][te])
    # transported: read each probe on the other model through the map
    t_ab = pa.belief(map_ba(B["pos"][te]), map_ba(B["neg"][te]))   # A's probe on B
    t_ba = pb.belief(map_ab(A["pos"][te]), map_ab(A["neg"][te]))   # B's probe on A

    v1, v2 = (ba >= 0.5).astype(int), (bb >= 0.5).astype(int)
    fa = metrics.false_agreement(v1, v2, gt)
    ea, wa = gate_a_edge(ba, A["p_yes"][te], gt)
    eb, wb = gate_a_edge(bb, B["p_yes"][te], gt)
    au_ab, au_ba = metrics.auroc(t_ab, gt), metrics.auroc(t_ba, gt)
    n_a, n_b = metrics.auroc(ba, gt), metrics.auroc(bb, gt)
    cov_t = 1 - fa.fa / fa.p2 if fa.p2 > 1e-9 else float("nan")

    return {"overseer": ov, "target": tg, "dataset": ds, "n": len(gt),
            "n_false": fa.n_false, "auroc_a": n_a, "auroc_b": n_b,
            "gate_a_edge_a": ea, "gate_a_edge_b": eb,
            "conf_wrong_a": wa, "conf_wrong_b": wb,
            "transfer_a_to_b": au_ab, "gap_a_to_b": n_b - au_ab,
            "transfer_b_to_a": au_ba, "gap_b_to_a": n_a - au_ba,
            "cka": align.linear_cka(A["pos"][te], B["pos"][te]),
            "map_r2_a_to_b": map_ab.r2(A["pos"][te], B["pos"][te]),
            "map_r2_b_to_a": map_ba.r2(B["pos"][te], A["pos"][te]),
            "fa_rate": fa.fa, "row2_n": fa.count,
            "p_overseer": fa.p1, "p_target": fa.p2, "rho": fa.rho,
            "rho_over_max": fa.rho_normalized,
            "fa_independent": fa.fa_independent, "frechet_hi": fa.bounds[1],
            "coverage_strong": fa.coverage, "coverage_target": cov_t}


# --------------------------------------------------------------------- phase B

_probe_cache: dict = {}


def _pooled_probe(model: str, pool):
    """Probe fitted on `model` over the pooled datasets, plus its positive rate.

    Cached because a probe depends only on (model, pool) -- the train mask comes
    from a seeded split over item ids, and the ids are identical across models,
    so the partner in the pair does not enter. Without the cache a 16-model sweep
    refits the same probe once per pair it appears in: 1222 fits where 80 are
    distinct.
    """
    key = (model, tuple(pool))
    if key in _probe_cache:
        return _probe_cache[key]
    P, N, Y = [], [], []
    for ds in pool:
        D = get(model, ds)
        tr = D["tr"]
        P.append(D["pos"][tr]); N.append(D["neg"][tr]); Y.append(D["labels"][tr])
    P, N, Y = np.concatenate(P), np.concatenate(N), np.concatenate(Y)
    probe = fit_probe(P, N, Y)
    out = (probe, positive_rate(probe, P, N))
    _probe_cache[key] = out
    return out


def loo_row(ov, tg, held, pool):
    P, N, Y, Xa, Xb = [], [], [], [], []
    for ds in pool:
        A, B = get(ov, ds), get(tg, ds)
        tr = A["tr"]
        Xa.append(np.concatenate([A["pos"][tr], A["neg"][tr]]))
        Xb.append(np.concatenate([B["pos"][tr], B["neg"][tr]]))
    # Only B -> A is ever read here, and only to build the detector's features.
    # `LOO_COLS` carries no map columns, so under --no-map-features the map is
    # pure waste -- and it is the most expensive thing in the cell, a ridge solve
    # on four datasets pooled. Fitting it conditionally is what makes the no-map
    # runs cheap rather than merely different.
    map_ba = None
    if USE_MAP_FOR_FEATURES:
        _, map_ba = align.fit_map_both_ways(np.concatenate(Xa),
                                            np.concatenate(Xb), alpha=ALPHA)

    (pa, rate_a), (pb, rate_b) = (_pooled_probe(ov, pool), _pooled_probe(tg, pool))
    rates = [rate_a, rate_b]

    Xd, yd = [], []
    for ds in pool:
        A, B = get(ov, ds), get(tg, ds)
        te = ~A["tr"]
        v1 = pa.vote(A["pos"][te], A["neg"][te])
        v2 = pb.vote(B["pos"][te], B["neg"][te])
        both = metrics.both_true_mask(v1, v2)
        sel = np.zeros(len(A["labels"]), bool)
        sel[np.where(te)[0][both]] = True
        Xd.append(aligned(A, B, sel, map_ba if USE_MAP_FOR_FEATURES else None))
        yd.append((A["labels"][te][both] == 0).astype(int))
    Xd, yd = np.concatenate(Xd), np.concatenate(yd)
    if len(yd) > MAX_DIRECTION_ROWS:
        rng = np.random.default_rng(SEED)
        pos = np.where(yd == 1)[0]
        neg = np.where(yd == 0)[0]
        # keep every positive we can, then fill with negatives
        k_pos = min(len(pos), MAX_DIRECTION_ROWS // 2)
        k_neg = MAX_DIRECTION_ROWS - k_pos
        keep = np.concatenate([rng.choice(pos, k_pos, replace=False),
                               rng.choice(neg, min(k_neg, len(neg)), replace=False)])
        Xd, yd = Xd[keep], yd[keep]

    A, B = get(ov, held), get(tg, held)
    gt = A["labels"]
    v1 = calibrated_vote(pa, A["pos"], A["neg"], rates[0])
    v2 = calibrated_vote(pb, B["pos"], B["neg"], rates[1])
    both = metrics.both_true_mask(v1, v2)
    n_false = int((gt == 0).sum())
    router = int(((gt == 0) & (v1 != v2)).sum())
    yh = (gt[both] == 0).astype(int)
    row = {"overseer": ov, "target": tg, "held_out": held,
           "n_pool_datasets": len(pool), "row2_fit_n": int(yd.sum()),
           "both_true_test_n": int(both.sum()), "row2_test_n": int(yh.sum()),
           "base_rate": float(yh.mean()) if len(yh) else float("nan"),
           "router_coverage": router / max(n_false, 1), "n_false": n_false,
           "pos_rate_overseer": rates[0], "pos_rate_target": rates[1]}
    if len(np.unique(yd)) < 2 or len(np.unique(yh)) < 2:
        row.update({c: float("nan") for c in LOO_COLS if c not in row})
        return row

    Xh = aligned(A, B, both, map_ba if USE_MAP_FOR_FEATURES else None)
    d = LinearDirection.fit(Xd, yd)
    score = d.score(Xh)
    rng = np.random.default_rng(SEED)
    row["direction_auroc"] = metrics.auroc(score, yh)
    shuf = [metrics.auroc(LinearDirection.fit(Xd, rng.permutation(yd)).score(Xh), yh)
            for _ in range(SHUFFLES)]
    row["shuffled_auroc"] = float(np.mean(shuf))
    row["shuffled_sd"] = float(np.std(shuf))
    row["margin"] = row["direction_auroc"] - row["shuffled_auroc"]
    order = np.argsort(-score)
    for b in BUDGETS:
        k = max(1, int(round(b * len(score))))
        hits = int(yh[order[:k]].sum())
        tag = f"{int(b * 100):02d}"
        row[f"prec_{tag}"] = hits / k
        row[f"recall_{tag}"] = hits / max(int(yh.sum()), 1)
        row[f"caught_{tag}"] = (router + hits) / max(n_false, 1)
    return row


# ------------------------------------------------------------------- the sweep

def all_pairs():
    # Models live across the cache search path -- locally extracted ones in
    # activations_cache, NDIF ones in activations_ndif -- and the sweep treats
    # them as one pool, so the listing unions every directory on the path.
    from cmb.config import CACHE_SEARCH_DIRS

    dsets: dict[str, list[str]] = {}
    for root in CACHE_SEARCH_DIRS:
        if not root.exists():
            continue
        for d in os.listdir(root):
            if not (root / d).is_dir() or d == "manifests":
                continue
            ds = sorted(f.replace("_nall.npz", "") for f in os.listdir(root / d)
                        if f.endswith("_nall.npz"))
            if ds:
                dsets.setdefault(d, ds)
    have = set(dsets)
    out = []
    for a, b in itertools.permutations(sorted(have), 2):
        # The standing filter keeps the overseer no larger than the target, which
        # is the scalable-oversight regime: a weaker model checking a stronger
        # one. That filter also means the same/cross-family contrast is drawn
        # from a slice rather than the whole space, so ALL_DIRECTIONS lifts it
        # and lets the null be retested with large models overseeing small ones.
        if not ALL_DIRECTIONS and SIZE.get(a, 99) > SIZE.get(b, 0):
            continue
        shared = sorted(set(dsets[a]) & set(dsets[b]))
        if len(shared) >= 3:
            out.append((a, b, shared))
    return out


def done_rows(path, keys):
    if not path.exists():
        return set()
    with path.open() as f:
        return {tuple(r[k] for k in keys) for r in csv.DictReader(f)}


def append(path, cols, row):
    new = not path.exists()
    with path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        if new:
            w.writeheader()
        w.writerow({c: row.get(c, "") for c in cols})


def run(pairs, phase, tag="", only_folds=()):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    bpath = RESULTS_DIR / f"sweep_base{tag}.csv"
    lpath = RESULTS_DIR / f"sweep_loo{tag}.csv"
    bdone = done_rows(bpath, ("overseer", "target", "dataset"))
    ldone = done_rows(lpath, ("overseer", "target", "held_out"))

    for i, (ov, tg, dss) in enumerate(pairs):
        t0 = time.time()
        if phase in ("base", "all"):
            for ds in dss:
                if (ov, tg, ds) in bdone:
                    continue
                try:
                    append(bpath, BASE_COLS, base_row(ov, tg, ds))
                except Exception as e:
                    print(f"  FAIL base {ov}->{tg}/{ds}: {type(e).__name__}: {e}",
                          flush=True)
        if phase in ("loo", "all"):
            for held in (only_folds or dss):
                if (ov, tg, held) in ldone:
                    continue
                if held not in dss:
                    continue
                pool = [d for d in dss if d != held]
                try:
                    append(lpath, LOO_COLS, loo_row(ov, tg, held, pool))
                except Exception as e:
                    print(f"  FAIL loo {ov}->{tg}/{held}: {type(e).__name__}: {e}",
                          flush=True)
        _cache.clear()
        print(f"[{i + 1}/{len(pairs)}] {ov} -> {tg}  "
              f"{time.time() - t0:.0f}s", flush=True)


def main() -> int:
    global ALL_DIRECTIONS
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--phase", default="all", choices=["base", "loo", "all"])
    ap.add_argument("--pairs", default="", help="slice, e.g. 0:10")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--tag", default="")
    ap.add_argument("--probe", default="mass-mean",
                    choices=["mass-mean", "lr", "ccs"])
    ap.add_argument("--all-directions", action="store_true",
                    help="also pair a large overseer with a small target, "
                         "lifting the scalable-oversight size filter")
    ap.add_argument("--no-map-features", action="store_true",
                    help="concatenate the two models' features instead of mapping "
                         "one into the other's space (see `aligned`)")
    ap.add_argument("--held-out", default="",
                    help="only run these held-out folds (comma-separated). One "
                         "fold costs a fifth of the full sweep and truthfulqa is "
                         "the fold that decides anything.")
    args = ap.parse_args()
    if args.all_directions:
        ALL_DIRECTIONS = True
    global PROBE, USE_MAP_FOR_FEATURES
    PROBE = args.probe
    USE_MAP_FOR_FEATURES = not args.no_map_features

    pairs = all_pairs()
    if args.pairs:
        lo, hi = (int(x) if x else None for x in args.pairs.split(":"))
        pairs = pairs[lo:hi]
    folds = tuple(d.strip() for d in args.held_out.split(",") if d.strip())
    print(f"{len(pairs)} ordered pairs, phase={args.phase}, "
          f"workers={args.workers}, folds={folds or 'all'}", flush=True)

    if args.workers <= 1:
        run(pairs, args.phase, args.tag, folds)
        return 0
    # Shard by stride so each worker sees a mix of cheap and expensive pairs.
    procs = [Process(target=run, args=(pairs[w::args.workers], args.phase,
                                       f"{args.tag}_w{w}", folds))
             for w in range(args.workers)]
    for p in procs:
        p.start()
    for p in procs:
        p.join()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
