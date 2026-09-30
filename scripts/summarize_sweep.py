#!/usr/bin/env python3
"""Turn the sweep CSVs into the tables the paper needs.

Merges the per-worker shards, then prints (and writes) summaries that answer the
questions the sweep was run for:

  * does false agreement fall with model independence, across every pair
  * does the false-agreement signature survive a dataset it never saw
  * what does an overseer actually catch, free and for a budget
  * is the signature's detectability predicted by the error correlation

Cells are reported with their Row-2 counts, because most of the misleading
numbers in this project came from AUROCs computed on a handful of positives.

    scripts/summarize_sweep.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cmb.config import RESULTS_DIR                        # noqa: E402

MIN_ROW2 = 20          # below this a fold is reported but never averaged
FAMILY = {"qwen3-1.7b": "qwen", "qwen3-8b": "qwen", "qwen3-8b-base": "qwen",
          "qwen3-32b": "qwen", "qwen38-27b": "qwen", "gemma4-12b": "gemma",
          "gemma4-12b-base": "gemma", "gemma4-31b": "gemma",
          "llama-8b": "llama", "olmo3-7b": "olmo"}
SIZE = {"qwen3-1.7b": 1.7, "olmo3-7b": 7, "qwen3-8b": 8, "qwen3-8b-base": 8,
        "llama-8b": 8, "gemma4-12b": 12, "gemma4-12b-base": 12,
        "qwen38-27b": 27, "gemma4-31b": 31, "qwen3-32b": 32}


def read(stem: str) -> pd.DataFrame:
    parts = sorted(RESULTS_DIR.glob(f"{stem}*.csv"))
    if not parts:
        sys.exit(f"no {stem}*.csv in {RESULTS_DIR}")
    df = pd.concat([pd.read_csv(p) for p in parts], ignore_index=True)
    df["kind"] = np.where(df.overseer.map(FAMILY) == df.target.map(FAMILY),
                          "same-family", "cross-family")
    df["scale_gap"] = df.target.map(SIZE) / df.overseer.map(SIZE)
    return df


def section(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def main() -> int:
    base, loo = read("sweep_base"), read("sweep_loo")
    base.to_csv(RESULTS_DIR / "sweep_base_all.csv", index=False)
    loo.to_csv(RESULTS_DIR / "sweep_loo_all.csv", index=False)
    print(f"base: {len(base)} rows, {base.groupby(['overseer','target']).ngroups} pairs")
    print(f"loo : {len(loo)} rows, {loo.groupby(['overseer','target']).ngroups} pairs")

    # ---- 1. false agreement vs independence --------------------------------
    section("1. False agreement by pair kind (base, per dataset)")
    t = (base.groupby(["kind", "dataset"])
             .agg(fa=("fa_rate", "mean"), rho_max=("rho_over_max", "mean"),
                  cov=("coverage_target", "mean"), n=("fa_rate", "size"))
             .round(3))
    print(t.to_string())

    section("2. Does Row 2 fall with independence? (per dataset, same vs cross)")
    for ds, g in base.groupby("dataset"):
        s = g[g.kind == "same-family"].fa_rate.mean()
        c = g[g.kind == "cross-family"].fa_rate.mean()
        verdict = "floor" if c < s else "CEILING"
        print(f"  {ds:<20} same {s:.3f}  cross {c:.3f}  delta {c - s:+.3f}  {verdict}")

    # ---- 3. gates ----------------------------------------------------------
    section("3. Gate A (probe beats confidence on the confident slice)")
    ga = pd.concat([
        base[["dataset", "overseer", "gate_a_edge_a", "conf_wrong_a"]]
            .rename(columns={"overseer": "model", "gate_a_edge_a": "edge",
                             "conf_wrong_a": "conf_wrong"}),
        base[["dataset", "target", "gate_a_edge_b", "conf_wrong_b"]]
            .rename(columns={"target": "model", "gate_a_edge_b": "edge",
                             "conf_wrong_b": "conf_wrong"})]).drop_duplicates()
    print(ga.groupby("dataset")
            .agg(edge=("edge", "mean"), pass_rate=("edge", lambda s: (s > 0).mean()),
                 conf_wrong=("conf_wrong", "mean")).round(3).to_string())

    section("4. Gate B (does the map carry the probe) — gap, both directions")
    gb = base.assign(worst=base[["gap_a_to_b", "gap_b_to_a"]].max(axis=1))
    print(gb.groupby("kind").agg(
        gap_a_to_b=("gap_a_to_b", "mean"), gap_b_to_a=("gap_b_to_a", "mean"),
        worst=("worst", "mean"), pass_rate=("worst", lambda s: (s <= 0.05).mean()),
        cka=("cka", "mean")).round(3).to_string())

    # ---- 5. generalization -------------------------------------------------
    section(f"5. LOO generalization (folds with >= {MIN_ROW2} Row-2 test items)")
    ok = loo[loo.row2_test_n >= MIN_ROW2].copy()
    print(f"  {len(ok)} of {len(loo)} folds are scoreable")
    print(ok.groupby("held_out").agg(
        auroc=("direction_auroc", "mean"), shuffled=("shuffled_auroc", "mean"),
        margin=("margin", "mean"), sd=("shuffled_sd", "mean"),
        n=("direction_auroc", "size")).round(3).to_string())
    print()
    print(ok.groupby("kind").agg(
        auroc=("direction_auroc", "mean"), margin=("margin", "mean"),
        n=("margin", "size")).round(3).to_string())

    # ---- 6. what an overseer catches ---------------------------------------
    section("6. Oversight: free router coverage and what a 10% budget adds")
    o = ok.groupby(["overseer", "target"]).agg(
        router=("router_coverage", "mean"), base_rate=("base_rate", "mean"),
        auroc=("direction_auroc", "mean"), margin=("margin", "mean"),
        prec10=("prec_10", "mean"), caught10=("caught_10", "mean"),
        folds=("margin", "size")).round(3).sort_values("caught10", ascending=False)
    print(o.to_string())

    # ---- 7. is detectability predicted by error correlation? ---------------
    section("7. Does error correlation predict detectability?")
    rho = (base.groupby(["overseer", "target"])
               .agg(rho_max=("rho_over_max", "mean"), fa=("fa_rate", "mean"),
                    cka=("cka", "mean")))
    j = o.join(rho, how="inner").dropna(subset=["rho_max", "margin"])
    if len(j) >= 5:
        for x in ("rho_max", "fa", "cka", "base_rate"):
            r = np.corrcoef(j[x], j["margin"])[0, 1]
            print(f"  corr(margin, {x:<10}) = {r:+.3f}   over {len(j)} pairs")
        print("\n  A positive corr with rho_max is the prediction: a signature "
              "exists\n  only where the two models are wrong for the same reason, "
              "not where\n  their errors merely coincide.")
    else:
        print("  too few pairs to correlate")
    print(f"\nmerged CSVs -> {RESULTS_DIR}/sweep_base_all.csv, sweep_loo_all.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
