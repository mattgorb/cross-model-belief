#!/usr/bin/env python3
"""Build the paper's numbered figures from the result CSVs.

Four figures, each answering one question the results section asks. Every one
reports both probe families side by side: the two disagree often enough that a
single-probe figure would overstate the finding, and the logistic probe is the
better belief probe while the mass-mean probe is the better detector.

Figures 1 and 2 are the panels of the teaser (`make_teaser.py`), computed from
the same loaders in `figstyle.py` so the two cannot disagree.

Static PDF for LaTeX plus a PNG for eyeballing; the per-figure data is in the
CSVs beside this script.

    scripts/make_figures.py
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from figstyle import (GRAY, INK, INK2, NICE, ORDER, PROBES, SURFACE,
                      load_base, load_indomain, load_loo, save, style)
from make_teaser import panel_bound, panel_diversity


# --- 1. false agreement against the interval it must sit inside --------------
def fig_bound(frames):
    """Teaser panel A, standalone and with the rates labelled."""
    fig, ax = plt.subplots(figsize=(6.6, 3.2))
    panel_bound(ax, frames, labels=True)
    ax.set_title("Measured false agreement between independence and maximal "
                 "overlap", fontsize=10.5, color=INK, loc="left", pad=10)
    ax.annotate("left tick: independence    right tick: maximal overlap",
                (0, -0.30), xycoords="axes fraction", fontsize=7.5, color=INK2)
    save(fig, "fig1_bound")


# --- 2. what family diversity changes, and what it does not ------------------
def fig_diversity(frames):
    """Teaser panel B, standalone."""
    fig, ax = plt.subplots(figsize=(6.6, 2.9))
    panel_diversity(ax, frames)
    ax.set_title("Model-family diversity changes correlation, not the blind spot",
                 fontsize=10.5, color=INK, loc="left", pad=10)
    save(fig, "fig2_diversity")


# --- 3. does the blind spot survive an unseen dataset? -----------------------
def fig_detect(loos, min_rows=20):
    """Ranking quality and one operating point, leave-one-dataset-out.

    AUROC alone is not enough here: a detector can rank well and still surface
    few false agreements inside a budget a reviewer would actually spend, so the
    recall at a 10% budget sits beside it.
    """
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 3.0))
    y = np.arange(len(ORDER))[::-1]
    for i, (name, color, key) in enumerate(PROBES):
        df = loos[key]
        g = (df[df.row2_test_n >= min_rows].groupby("held_out")
             .agg(auroc=("direction_auroc", "mean"),
                  rec=("recall_10", "mean")).reindex(ORDER))
        off = 0.16 if i == 0 else -0.16
        axes[0].plot(g.auroc, y + off, "o", color=color, markersize=7,
                     markeredgecolor=SURFACE, markeredgewidth=1.6, label=name)
        axes[1].plot(100 * g.rec, y + off, "o", color=color, markersize=7,
                     markeredgecolor=SURFACE, markeredgewidth=1.6, label=name)
    axes[0].axvline(0.5, color=INK2, linewidth=1, linestyle=(0, (3, 2)))
    for ax, xlab, title in (
            (axes[0], "leave-one-dataset-out AUROC", "A  Ranking performance"),
            (axes[1], "false agreements recovered at a 10% budget",
             "B  Operating point")):
        ax.set_yticks(y, [NICE[d] for d in ORDER], fontsize=8.5, color=INK)
        ax.set_ylim(-0.7, len(ORDER) - 0.15)
        style(ax, xlab)
        ax.set_title(title, fontsize=10.5, color=INK, loc="left", pad=10)
    axes[1].xaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")
    # below the axes: at upper right it lands on the Geometry of Truth dot,
    # which is the rightmost point in the panel
    axes[1].legend(frameon=False, fontsize=8, labelcolor=INK2, ncol=2,
                   loc="upper left", bbox_to_anchor=(0, -0.22),
                   handletextpad=0.4)
    fig.subplots_adjust(wspace=0.42)
    save(fig, "fig3_detect")


# --- 4. where the caught errors actually come from ---------------------------
def fig_pairs(loos):
    """Coverage at a 10% budget, split into what disagreement already supplies
    and what the detector adds.

    Reporting `caught` alone credits the detector with errors the two probes
    flagged by disagreeing, which needs no detector at all. The split is the
    honest version, and on most datasets the hatched part is the smaller one.
    """
    fig, ax = plt.subplots(figsize=(7.4, 3.2))
    y = np.arange(len(ORDER))[::-1]
    h = 0.34
    for i, (name, color, key) in enumerate(PROBES):
        g = (loos[key].groupby("held_out")
             .agg(router=("router_coverage", "mean"),
                  caught=("caught_10", "mean")).reindex(ORDER))
        off = h / 2 + 0.02 if i == 0 else -h / 2 - 0.02
        ax.barh(y + off, 100 * g.router, height=h, color=color,
                label=f"{name}: disagreement")
        ax.barh(y + off, 100 * (g.caught - g.router), height=h,
                left=100 * g.router, color=color, alpha=0.30, hatch="///",
                edgecolor=color, linewidth=0, label=f"{name}: detector adds")
        for yi, v in zip(y + off, g.caught):
            ax.annotate(f"{100 * v:.1f}%", (100 * v, yi), xytext=(4, -3),
                        textcoords="offset points", fontsize=7.5, color=INK)
    ax.set_yticks(y, [NICE[d] for d in ORDER], fontsize=8.5, color=INK)
    ax.set_ylim(-0.7, len(ORDER) - 0.15)
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")
    style(ax, "share of all false claims caught")
    ax.set_xlim(0, None)
    ax.set_title("Disagreement supplies most coverage at a 10% review budget",
                 fontsize=10.5, color=INK, loc="left", pad=10)
    ax.legend(frameon=False, fontsize=7.5, labelcolor=INK2, ncol=2,
              loc="upper left", bbox_to_anchor=(0, -0.22), handletextpad=0.5)
    save(fig, "fig4_pairs")


# --- 5. detectable in-domain, transferable only sometimes --------------------
def fig_indomain(inds, loos, min_rows=20):
    """What the detector achieves inside a dataset, and what survives transfer.

    Drawn as an arrow from the in-domain score to the leave-one-dataset-out
    score, because the two numbers only mean something together: a dataset can
    be highly detectable and barely transferable (IMDB) or both (geometry of
    truth), and reporting either alone hides which. The arrow's length is the
    transfer cost.

    Cells with fewer than `min_rows` false agreements are dropped: in-domain
    evaluation of a rare event inside one dataset leaves many cells with too few
    positives to score, and an AUROC over a handful of them is noise.
    """
    probes = [(n, c, k) for n, c, k in PROBES if k in inds]
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 3.2))
    y = np.arange(len(ORDER))[::-1]
    for i, (name, color, key) in enumerate(probes):
        ind = inds[key]; loo = loos[key]
        gi = (ind[ind.row2_test_n >= min_rows].groupby("held_out")
              .agg(a=("direction_auroc", "mean"), r=("recall_10", "mean"))
              .reindex(ORDER))
        gl = (loo[loo.row2_test_n >= min_rows].groupby("held_out")
              .agg(a=("direction_auroc", "mean"), r=("recall_10", "mean"))
              .reindex(ORDER))
        off = 0.16 if i == 0 else -0.16
        for ax, col, scale in ((axes[0], "a", 1.0), (axes[1], "r", 100.0)):
            for yi, hi, lo in zip(y + off, scale * gi[col], scale * gl[col]):
                if not (np.isfinite(hi) and np.isfinite(lo)):
                    continue
                ax.annotate("", xy=(lo, yi), xytext=(hi, yi),
                            arrowprops=dict(arrowstyle="-|>", color=GRAY,
                                            linewidth=1.6, shrinkA=3, shrinkB=0))
            ax.plot(scale * gi[col], y + off, "o", color=color, markersize=7,
                    markeredgecolor=SURFACE, markeredgewidth=1.6,
                    label=f"{name}: in-domain", zorder=3)
            ax.plot(scale * gl[col], y + off, "o", color=color, markersize=6,
                    markerfacecolor=SURFACE, markeredgecolor=color,
                    markeredgewidth=1.8, label=f"{name}: transferred", zorder=3)
    axes[0].axvline(0.5, color=INK2, linewidth=1, linestyle=(0, (3, 2)))
    for ax, xlab, title in (
            (axes[0], "AUROC", "A  Ranking performance"),
            (axes[1], "false agreements recovered at a 10% budget",
             "B  Operating point")):
        ax.set_yticks(y, [NICE[d] for d in ORDER], fontsize=8.5, color=INK)
        ax.set_ylim(-0.7, len(ORDER) - 0.15)
        style(ax, xlab)
        ax.set_title(title, fontsize=10.5, color=INK, loc="left", pad=10)
    axes[1].xaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")
    axes[0].legend(frameon=False, fontsize=7.5, labelcolor=INK2, ncol=2,
                   loc="upper left", bbox_to_anchor=(0, -0.26),
                   handletextpad=0.4)
    fig.subplots_adjust(wspace=0.42)
    save(fig, "fig5_indomain")


def main() -> int:
    frames = {k: load_base(k) for _, _, k in PROBES}
    loos = {k: load_loo(k) for _, _, k in PROBES}
    # the logistic in-domain sweep may still be running; draw what exists
    inds = {}
    for _, _, k in PROBES:
        try:
            inds[k] = load_indomain(k)
        except FileNotFoundError:
            print(f"  (no in-domain table for {k} yet, skipping it)")
    print("writing figures:")
    fig_bound(frames)
    fig_diversity(frames)
    fig_detect(loos)
    fig_pairs(loos)
    if inds:
        fig_indomain(inds, loos)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
