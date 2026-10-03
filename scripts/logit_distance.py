#!/usr/bin/env python3
"""Does logit distance predict whether a probe survives the map?

\\citet{nielsen2026logit} identify a logit-based distributional distance as the
quantity that controls linear representational similarity: closeness in it yields
linear similarity guarantees, while closeness in KL does not. That makes it the
principled predictor of probe transportability, where we had been reporting CKA.

The test is cheap because the extraction already cached each model's out-loud
P(claim is true), which is a two-outcome distribution per item. For models A and
B on the same items we compute

    logit_i = log p_i - log (1 - p_i)                 (the model's own logit)
    d_logit = mean_i | logit_i^A - logit_i^B |        (and the RMS variant)

and the KL between the two Bernoulli distributions for contrast, then correlate
each against the transfer loss already measured in results/pair_table.csv.

Caveats, stated because they bound what this can show. The distance in the theory
is over the model's full conditional distribution; ours is over a single binary
question, so it is a projection of theirs onto the axis this task happens to
probe. And a shared prompt template means both models see identical inputs, which
is what makes the comparison well posed at all.

    scripts/logit_distance.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cmb.cache import cache_path                       # noqa: E402
from cmb.config import RESULTS_DIR                     # noqa: E402

EPS = 1e-6


def pyes(model: str, ds: str):
    z = np.load(cache_path(model, ds, None), allow_pickle=True)
    out = (np.clip(z["p_yes"].astype(np.float64), EPS, 1 - EPS),
           [str(x) for x in z["item_ids"]])
    z.close()
    return out


def distances(ma: str, mb: str, ds: str) -> dict:
    pa, ida = pyes(ma, ds)
    pb, idb = pyes(mb, ds)
    if ida != idb:                      # align on item id before comparing
        order = {k: i for i, k in enumerate(idb)}
        pb = pb[np.array([order[k] for k in ida])]
    la, lb = np.log(pa / (1 - pa)), np.log(pb / (1 - pb))
    kl = (pa * np.log(pa / pb) + (1 - pa) * np.log((1 - pa) / (1 - pb)))
    return {"logit_dist_mean": float(np.abs(la - lb).mean()),
            "logit_dist_rms": float(np.sqrt(((la - lb) ** 2).mean())),
            "kl_mean": float(kl.mean()),
            "pyes_corr": float(np.corrcoef(pa, pb)[0, 1])}


def main() -> int:
    d = pd.read_csv(RESULTS_DIR / "pair_table.csv")
    rows = []
    for _, r in d.iterrows():
        try:
            rows.append({**distances(r.model_a, r.model_b, r.dataset),
                         "model_a": r.model_a, "model_b": r.model_b,
                         "dataset": r.dataset})
        except Exception as e:
            print(f"  skip {r.model_a}|{r.model_b}/{r.dataset}: {type(e).__name__}")
    j = d.merge(pd.DataFrame(rows), on=["model_a", "model_b", "dataset"])
    j["transfer_loss_worst"] = j[["transfer_loss_a_to_b",
                                  "transfer_loss_b_to_a"]].max(axis=1)
    j.to_csv(RESULTS_DIR / "logit_distance.csv", index=False)

    targets = ["transfer_loss_a_to_b", "transfer_loss_b_to_a",
               "transfer_loss_worst"]
    preds = ["logit_dist_mean", "logit_dist_rms", "kl_mean", "pyes_corr",
             "cka", "map_r2_a_to_b"]
    print(f"n = {len(j)} cells, {j.groupby(['model_a','model_b']).ngroups} pairs\n")
    print("Pearson correlation with transfer loss (negative = predicts better "
          "transfer):\n")
    print(f"  {'predictor':<18}" + "".join(f"{t.replace('transfer_loss_',''):>14}"
                                           for t in targets))
    for p in preds:
        cells = "".join(f"{np.corrcoef(j[p], j[t])[0, 1]:>14.3f}" for t in targets)
        print(f"  {p:<18}{cells}")

    print("\nwithin each dataset (worst-direction transfer loss):")
    print(f"  {'dataset':<20}{'logit_dist':>12}{'kl':>10}{'cka':>10}{'n':>5}")
    for ds, g in j.groupby("dataset"):
        if len(g) < 5:
            continue
        f = lambda c: np.corrcoef(g[c], g.transfer_loss_worst)[0, 1]
        print(f"  {ds:<20}{f('logit_dist_mean'):>12.3f}{f('kl_mean'):>10.3f}"
              f"{f('cka'):>10.3f}{len(g):>5}")
    print(f"\n-> {RESULTS_DIR / 'logit_distance.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
