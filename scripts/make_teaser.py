#!/usr/bin/env python3
"""Build the teaser (Figure 1): every measured cell against independence.

One point per model-pair--dataset cell. The horizontal axis is the false
agreement two probes with these accuracies would produce if their errors were
independent, $p_1 p_2$; the vertical axis is what was actually measured. The
diagonal is independence.

The figure carries both headline results without averaging anything away.
Points sit above the diagonal, so the two readers fail together more often than
chance -- the blind spot is larger than probe accuracy alone implies. And the
two colours are intermingled rather than separated, so choosing a judge from a
different model family does not move a pair off that line.

Both probe families are shown because they disagree about the sign of the
family effect, and a single-probe figure would hide that.

    scripts/make_teaser.py
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from figstyle import (BLUE, GRAY, INK, INK2, NICE, ORANGE, ORDER, PROBES,
                      SURFACE, load_base, save, style)

FAM_COLOR = {"same": BLUE, "cross": ORANGE}
FAM_LABEL = {"same": "same model family", "cross": "different family"}


def panel(ax, df, title, lo, hi):
    # Log axes: most cells sit near the origin on a linear scale -- false
    # agreement is a few percent on three of the five datasets -- and the dense
    # corner hides exactly the pairs an operator would most like to see. The
    # diagonal is still a straight line under log-log, so "above the line" reads
    # the same.
    ax.plot([lo, hi], [lo, hi], color=INK2, linewidth=1.2, zorder=2,
            linestyle=(0, (4, 3)))
    ax.set_xscale("log"); ax.set_yscale("log")
    # on the line itself, at 45 degrees: the axes are log with equal decades and
    # an equal aspect, so the diagonal really is at 45 degrees
    ax.annotate("errors independent", (lo * 7.0, lo * 6.2), fontsize=7.5,
                color=INK2, ha="left", va="top", rotation=45,
                rotation_mode="anchor")

    for fam in ("cross", "same"):            # same drawn last, it is the smaller set
        d = df[df.family == fam]
        ax.scatter(d.fa_independent, d.fa_rate, s=13, alpha=0.55,
                   color=FAM_COLOR[fam], linewidths=0, zorder=3,
                   label=f"{FAM_LABEL[fam]}  ($n={len(d)}$)")

    above = (df.fa_rate > df.fa_independent).mean()
    ax.annotate(f"{above:.0%} of cells above the line",
                (0.035, 0.955), xycoords="axes fraction", fontsize=8,
                color=INK, va="top")

    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    style(ax, "false agreement if errors were independent",
          "measured false agreement")
    ax.grid(axis="y", color=GRAY, alpha=0.35, linewidth=0.6)
    ax.set_title(title, fontsize=10.5, color=INK, loc="left", pad=8)
    ax.legend(frameon=False, fontsize=7.5, labelcolor=INK2, loc="lower right",
              handletextpad=0.3, scatterpoints=1)


def main() -> int:
    frames = {k: load_base(k) for _, _, k in PROBES}
    # one pair of limits for both panels, so the two probes are comparable and
    # nothing is clipped onto an axis -- clipping produced a false column of
    # points on the left edge, 64 of 571 cells for the logistic probe
    lo = min(min(d.fa_independent.min(), d.fa_rate.min())
             for d in frames.values()) * 0.7
    hi = max(max(d.fa_independent.max(), d.fa_rate.max())
             for d in frames.values()) * 1.4
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.3))
    for ax, (name, _, key) in zip(axes, PROBES):
        panel(ax, frames[key], f"{name} probes", lo, hi)
    fig.subplots_adjust(wspace=0.28)
    print("writing teaser:")
    save(fig, "fig0_teaser")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
