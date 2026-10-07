#!/usr/bin/env python3
"""Figures for Paper 2 (transport, the linear map, representational similarity).

Separate from `make_figures.py` because Paper 1 uses no linear map: on 228
matched cells the map cost 0.012 detector AUROC (paired t = 5.90) and lost on
every held-out dataset, so it was cut from Paper 1 and everything about
transport moved here.

`fig_transport` is recovered unchanged from the Paper 1 figure script, where it
produced fig2 before fig2_diversity replaced it. Its inputs -- transfer_loss_*,
cka, map_r2_* -- are columns Paper 1 does not report.

Output goes to results/paper2/figures/, not paper/figures/, so a Paper 1 build
cannot pick it up by accident.

    scripts/make_figures_paper2.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from figstyle import BLUE, GRAY, INK, INK2, NICE, ORANGE, ORDER, SURFACE, style

OUT = Path(__file__).resolve().parent.parent / "results" / "paper2" / "figures"


def headroom(ax, n, extra=0.55):
    ax.set_ylim(-0.6, n - 1 + extra)


def legend_above(ax, ncol=3):
    ax.legend(frameon=False, fontsize=7.5, labelcolor=INK2, handletextpad=0.4,
              loc="lower left", bbox_to_anchor=(0, 1.02, 1, 0.12), mode="expand",
              ncol=ncol, borderaxespad=0)


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.patch.set_facecolor(SURFACE)
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight", facecolor=SURFACE)
    fig.savefig(OUT / f"{name}.png", bbox_inches="tight", facecolor=SURFACE,
                dpi=200)
    plt.close(fig)
    print(f"  {name}.pdf + .png")


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


def main() -> int:
    from cmb.config import RESULTS_DIR
    print("writing Paper 2 figures:")
    fig_transport(pd.read_csv(RESULTS_DIR / "pair_table.csv"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
