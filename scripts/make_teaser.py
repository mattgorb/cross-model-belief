#!/usr/bin/env python3
"""Build the teaser (Figure 1): the paper's two headline results side by side.

Panel A is the measured false-agreement rate inside the Frechet interval it must
occupy; panel B is what model-family diversity changes and what it does not.
Both panels are the same computation as Figures 2 and 3 -- the teaser is a
summary, not a separate analysis -- so they share `figstyle.load_base` and
cannot report different numbers.

    scripts/make_teaser.py
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from figstyle import (BLUE, GRAY, INK, INK2, NICE, ORANGE, ORDER, PROBES,
                      SURFACE, load_base, save, style)


def panel_bound(ax, frames, labels=False):
    """False agreement against the interval the identity confines it to.

    The grey track runs from the rate two independent probes would hit to the
    ceiling min(p1, p2); the ticks mark both ends. A dot near the left end means
    the probes fail independently, near the right end that they fail together.
    """
    y = np.arange(len(ORDER))[::-1]
    for i, (name, color, key) in enumerate(PROBES):
        g = (frames[key].groupby("dataset")
             .agg(fa=("fa_rate", "mean"), indep=("fa_independent", "mean"),
                  hi=("frechet_hi", "mean")).reindex(ORDER))
        off = 0.17 if i == 0 else -0.17
        ax.hlines(y + off, g.indep, g.hi, color=GRAY, linewidth=5, alpha=0.5,
                  capstyle="round", zorder=1)
        for end in (g.indep, g.hi):
            ax.plot(end, y + off, "|", color=INK2, markersize=7,
                    markeredgewidth=1.2, zorder=2)
        ax.plot(g.fa, y + off, "o", color=color, markersize=7, zorder=3,
                markeredgecolor=SURFACE, markeredgewidth=1.6, label=name)
        if labels:
            for yi, v in zip(y + off, g.fa):
                ax.annotate(f"{v:.3f}", (v, yi), textcoords="offset points",
                            xytext=(0, 7), ha="center", fontsize=7, color=INK)
    ax.set_yticks(y, [NICE[d] for d in ORDER], fontsize=8.5, color=INK)
    ax.set_ylim(-0.7, len(ORDER) - 0.15)
    style(ax, "false agreement, share of false claims")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK2, loc="upper right",
              handletextpad=0.3)


def panel_diversity(ax, frames):
    """Cross-family minus within-family, in percentage points.

    Three rows, because the story needs all three: the blind spot does not move,
    the error correlation does, and the correlation effect disappears once the
    two models are of similar size -- which is what makes it a size effect rather
    than a family effect.
    """
    rows = [("False agreement", "fa_rate", None),
            ("Error correlation", "rho", None),
            ("Correlation, size-matched", "rho", 1.5)]
    y = np.arange(len(rows))[::-1]
    for i, (name, color, key) in enumerate(PROBES):
        df = frames[key]
        off = 0.14 if i == 0 else -0.14
        for yi, (_, col, gap) in zip(y, rows):
            d = df if gap is None else df[df.gap < gap]
            delta = 100 * (d[d.family == "cross"][col].mean()
                           - d[d.family == "same"][col].mean())
            ax.plot(delta, yi + off, "o", color=color, markersize=7, zorder=3,
                    markeredgecolor=SURFACE, markeredgewidth=1.6,
                    label=name if yi == y[0] else None)
            ax.annotate(f"{delta:+.1f}", (delta, yi + off),
                        textcoords="offset points",
                        xytext=(7 if delta >= 0 else -7, -3),
                        ha="left" if delta >= 0 else "right",
                        fontsize=7.5, color=INK)
    ax.axvline(0, color=INK2, linewidth=1.1, zorder=2)
    ax.set_yticks(y, [r[0] for r in rows], fontsize=8.5, color=INK)
    # headroom above the top row, and room to the right of the widest label:
    # placed tightly, the legend lands on the "+0.7" annotation
    ax.set_ylim(-0.7, len(rows) + 0.15)
    lo, hi = ax.get_xlim()
    ax.set_xlim(lo, max(hi, 5.0))
    style(ax, "cross-family minus within-family (percentage points)")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK2, loc="upper left",
              handletextpad=0.3, ncol=2)


def main() -> int:
    frames = {k: load_base(k) for _, _, k in PROBES}
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 3.1))
    panel_bound(axes[0], frames)
    panel_diversity(axes[1], frames)
    for ax, letter, title in ((axes[0], "A", "False agreement is above independence"),
                              (axes[1], "B", "Family diversity does not reduce it")):
        ax.set_title(f"{letter}  {title}", fontsize=10.5, color=INK,
                     loc="left", pad=10)
    fig.subplots_adjust(wspace=0.46)
    print("writing teaser:")
    save(fig, "fig0_teaser")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
