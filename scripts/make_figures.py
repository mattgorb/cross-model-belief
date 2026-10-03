#!/usr/bin/env python3
"""Build the paper's figures from the result CSVs.

Four figures, each answering one question the results section asks. Colors are
the validated three-slot categorical palette (blue / orange / aqua) plus a
de-emphasis gray; the aqua slot sits below 3:1 against the surface, so every
chart that uses it carries direct labels, which the relief rule requires.

Static PDF for LaTeX, so the hover layer the interactive guidance asks for does
not apply; the per-figure data is in the CSVs beside this script.

    scripts/make_figures.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cmb.config import RESULTS_DIR                      # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "paper" / "figures"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
GRAY, INK, INK2 = "#b9b8b4", "#0b0b0b", "#52514e"
SURFACE = "#fcfcfb"

NICE = {"geometry_of_truth": "Geometry of Truth", "truthfulqa": "TruthfulQA",
        "boolq": "BoolQ", "imdb": "IMDB", "rte": "RTE"}
# datasets ordered by how often both models are wrong together -- the quantity
# every figure is about, so the order is the same everywhere
ORDER = ["geometry_of_truth", "imdb", "boolq", "truthfulqa", "rte"]


def style(ax, xlabel="", ylabel=""):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRAY)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=INK2, labelsize=8, length=3, width=0.8)
    ax.grid(axis="x", color=GRAY, alpha=0.35, linewidth=0.6)
    ax.set_axisbelow(True)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=8.5, color=INK2)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=8.5, color=INK2)


def headroom(ax, n, extra=0.55):
    """Leave space above the top row so its value label cannot meet the legend."""
    ax.set_ylim(-0.6, n - 1 + extra)


def legend_above(ax, ncol=3):
    """Legend outside the data area. Inside, it collided with the right-hand
    annotations on every figure here."""
    ax.legend(frameon=False, fontsize=7.5, labelcolor=INK2, handletextpad=0.4,
              loc="lower left", bbox_to_anchor=(0, 1.02, 1, 0.12), mode="expand",
              ncol=ncol, borderaxespad=0)


def save(fig, name):
    fig.patch.set_facecolor(SURFACE)
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print(f"  {name}.pdf")


# --- 1. false agreement against the bound it must sit inside -----------------
def fig_bound(base):
    g = (base.groupby("dataset")
             .agg(fa=("fa_rate", "mean"), indep=("fa_independent", "mean"),
                  hi=("frechet_hi", "mean")).reindex(ORDER))
    y = np.arange(len(g))[::-1]
    col_x = g.hi.max() * 1.12            # fixed column for the right-hand note
    fig, ax = plt.subplots(figsize=(6.4, 2.7))
    # the feasible interval, as a track from independence to the ceiling
    ax.hlines(y, g.indep, g.hi, color=GRAY, linewidth=6, alpha=0.5,
              capstyle="round", zorder=1)
    ax.plot(g.hi, y, "|", color=INK2, markersize=10, markeredgewidth=1.4,
            zorder=2, label="ceiling (better probe's error rate)")
    ax.plot(g.indep, y, "|", color=INK2, markersize=10, markeredgewidth=1.4,
            zorder=2, label="if errors were independent")
    ax.plot(g.fa, y, "o", color=BLUE, markersize=9, zorder=3,
            markeredgecolor=SURFACE, markeredgewidth=2, label="measured")
    for yi, (ds, r) in zip(y, g.iterrows()):
        ax.annotate(f"{r.fa:.3f}", (r.fa, yi), textcoords="offset points",
                    xytext=(0, 9), ha="center", fontsize=7.5, color=INK)
        frac = (r.fa - r.indep) / max(r.hi - r.indep, 1e-9)
        ax.annotate(f"{frac:.0%} of the way to the ceiling", (col_x, yi),
                    textcoords="offset points", xytext=(0, -3), ha="left",
                    fontsize=7, color=INK2)
    ax.set_yticks(y, [NICE[d] for d in g.index], fontsize=8.5, color=INK)
    style(ax, "false agreement, as a share of false claims")
    ax.set_xlim(-0.01, col_x * 1.52)
    headroom(ax, len(g))
    legend_above(ax)
    save(fig, "fig1_bound")


# --- 2. what transport costs, by dataset -------------------------------------
def fig_transport(pt):
    g = (pt.assign(worst=pt[["transfer_loss_a_to_b", "transfer_loss_b_to_a"]]
                   .max(axis=1))
           .groupby("dataset")
           .agg(ab=("transfer_loss_a_to_b", "mean"),
                ba=("transfer_loss_b_to_a", "mean"),
                ok=("worst", lambda s: (s <= 0.05).mean())).reindex(ORDER))
    y = np.arange(len(g))[::-1]
    h = 0.34
    col_x = max(g.ab.max(), g.ba.max()) * 1.12
    fig, ax = plt.subplots(figsize=(6.4, 2.8))
    ax.barh(y + h / 2 + 0.02, g.ab, height=h, color=BLUE,
            label="overseer's probe, read on the target")
    ax.barh(y - h / 2 - 0.02, g.ba, height=h, color=ORANGE,
            label="target's probe, read on the overseer")
    ax.axvline(0.05, color=INK2, linewidth=1, linestyle=(0, (3, 2)))
    ax.annotate("0.05 tolerance", (0.05, max(y) + 0.45), fontsize=7,
                color=INK2, ha="center")
    for yi, (ds, r) in zip(y, g.iterrows()):
        ax.annotate(f"{r.ab:.3f}", (r.ab, yi + h / 2 + 0.02),
                    xytext=(4, -3), textcoords="offset points", fontsize=7.5,
                    color=INK)
        ax.annotate(f"{r.ba:.3f}", (r.ba, yi - h / 2 - 0.02),
                    xytext=(4, -3), textcoords="offset points", fontsize=7.5,
                    color=INK)
        ax.annotate(f"{r.ok:.0%} of pairs within it", (col_x, yi),
                    xytext=(0, -3), textcoords="offset points", fontsize=7,
                    color=INK2)
    ax.set_yticks(y, [NICE[d] for d in g.index], fontsize=8.5, color=INK)
    style(ax, "AUROC lost by reading the probe on the other model")
    ax.set_xlim(0, col_x * 1.42)
    headroom(ax, len(g))
    legend_above(ax, ncol=2)
    save(fig, "fig2_transport")


# --- 3. is the blind spot detectable on an unseen dataset? -------------------
def fig_detect(loo):
    ok = loo[loo.row2_test_n >= 20]
    g = (ok.groupby("held_out")
           .agg(auroc=("direction_auroc", "mean"),
                shuf=("shuffled_auroc", "mean"), sd=("shuffled_sd", "mean"),
                n=("margin", "size")).reindex(ORDER))
    y = np.arange(len(g))[::-1]
    fig, ax = plt.subplots(figsize=(6.4, 2.7))
    # the control's own noise band, so a margin inside it reads as not-measured
    for yi, (_, r) in zip(y, g.iterrows()):
        ax.add_patch(plt.Rectangle((r.shuf - 2 * r.sd, yi - 0.22),
                                   4 * r.sd, 0.44, color=GRAY, alpha=0.45,
                                   linewidth=0))
    ax.plot(g.shuf, y, "|", color=INK2, markersize=11, markeredgewidth=1.4,
            label="shuffled-label control ($\\pm$2 sd)")
    ax.hlines(y, g.shuf, g.auroc, color=BLUE, linewidth=2, zorder=2)
    ax.plot(g.auroc, y, "o", color=BLUE, markersize=9, zorder=3,
            markeredgecolor=SURFACE, markeredgewidth=2, label="detector")
    for yi, (ds, r) in zip(y, g.iterrows()):
        inside = abs(r.auroc - r.shuf) < 2 * r.sd
        ax.annotate(f"+{r.auroc - r.shuf:.3f}" + ("  (inside the band)" if inside else ""),
                    (r.auroc, yi), xytext=(10, -3), textcoords="offset points",
                    fontsize=7.5, color=INK2 if inside else INK)
    ax.set_yticks(y, [NICE[d] for d in g.index], fontsize=8.5, color=INK)
    style(ax, "AUROC separating false agreement from genuine agreement")
    ax.set_xlim(0.40, 1.04)
    headroom(ax, len(g))
    legend_above(ax, ncol=2)
    save(fig, "fig3_detect")


# --- 4. which pairs catch the most, and where it comes from ------------------
def fig_pairs(loo, k=12):
    ok = loo[(loo.held_out == "truthfulqa") & (loo.row2_test_n >= 20)].copy()
    ok["detector"] = ok.caught_10 - ok.router_coverage
    ok = ok.sort_values("caught_10", ascending=False)
    top = ok.head(k).iloc[::-1]
    y = np.arange(len(top))
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    ax.barh(y, top.router_coverage, color=BLUE, height=0.62,
            label="free: the two probes disagree")
    ax.barh(y, top.detector, left=top.router_coverage + 0.004, color=ORANGE,
            height=0.62, label="added by the detector, 10% check budget")
    for yi, (_, r) in zip(y, top.iterrows()):
        ax.annotate(f"{r.caught_10:.2f}", (r.caught_10, yi), xytext=(6, -3),
                    textcoords="offset points", fontsize=7.5, color=INK)
    ax.set_yticks(y, [f"{r.overseer} $\\rightarrow$ {r.target}"
                      for _, r in top.iterrows()], fontsize=7.5, color=INK)
    style(ax, "share of all false claims caught, TruthfulQA held out")
    ax.set_xlim(0, top.caught_10.max() * 1.16)
    headroom(ax, len(top), extra=0.75)
    legend_above(ax, ncol=2)
    save(fig, "fig4_pairs")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    base = pd.read_csv(RESULTS_DIR / "sweep_base_all.csv")
    loo = pd.read_csv(RESULTS_DIR / "sweep_loo_all.csv")
    pt = pd.read_csv(RESULTS_DIR / "pair_table.csv")
    print("writing figures:")
    fig_bound(base)
    fig_transport(pt)
    fig_detect(loo)
    fig_pairs(loo)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
