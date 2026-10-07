#!/usr/bin/env python3
"""Build the teaser (Figure 1): the four outcomes when two probes read a false claim.

A 2x2 of what the two readouts say. Every claim counted here is false, so
saying "true" is the error. Three of the four outcomes are survivable: the two
off-diagonal cells produce a disagreement an overseer can route for review, and
the bottom-right cell is both readers getting it right. The top-left cell is the
one that matters -- both call it true, there is no disagreement to route, and
nothing flags it.

Concept and result in one object: the grid is the oversight situation, and the
numbers in it are measured. Each cell carries both the within-family and the
cross-family rate, so the headline null is read by comparing two numbers in the
same box rather than two pictures.

    scripts/make_teaser.py
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch

from figstyle import BLUE, GRAY, INK, INK2, ORANGE, SURFACE, load_base, save

PROBE_KEY = "lr"


def rates(d):
    """(p1, p2, fa) using the per-cell max and min.

    Not (overseer, target): the base table holds each pair in both orders, since
    false agreement is symmetric in the two models, so after deduplication which
    model is called the overseer is decided by row order.
    """
    return (np.maximum(d.p_overseer, d.p_target).mean(),
            np.minimum(d.p_overseer, d.p_target).mean(),
            d.fa_rate.mean())


def outcomes(d):
    p1, p2, fa = rates(d)
    return {"tt": fa, "tf": p1 - fa, "ft": p2 - fa, "ff": 1 - p1 - p2 + fa}


def main() -> int:
    df = load_base(PROBE_KEY)
    same, cross = outcomes(df[df.family == "same"]), outcomes(df[df.family == "cross"])
    n_s, n_c = (df.family == "same").sum(), (df.family == "cross").sum()

    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    W = H = 1.0
    spec = {
        "tt": (0, H, "blind spot", "no disagreement:\nnothing flags it", INK2, 0.80),
        "tf": (W, H, "caught", "they disagree:\nroute for review", BLUE, 0.22),
        "ft": (0, 0, "caught", "they disagree:\nroute for review", ORANGE, 0.22),
        "ff": (W, 0, "both correct", "both reject the\nfalse claim", GRAY, 0.18),
    }
    for key, (x, y, tag, note, color, alpha) in spec.items():
        dark = key == "tt"
        ax.add_patch(FancyBboxPatch((x + 0.03, y + 0.03), W - 0.06, H - 0.06,
                                    boxstyle="round,pad=0.015,rounding_size=0.04",
                                    facecolor=color, alpha=alpha,
                                    edgecolor=color if not dark else INK2,
                                    linewidth=1.4 if dark else 0.0, zorder=2))
        ink = "white" if dark else INK
        cx, cy = x + W / 2, y + H / 2
        ax.annotate(tag.upper(), (cx, cy + 0.30), ha="center", va="center",
                    fontsize=8, color=ink if dark else INK2,
                    fontweight="bold" if dark else "normal", zorder=4)
        for dx, lab, val in ((-0.255, "same", same[key]),
                             (0.255, "cross", cross[key])):
            ax.annotate(f"{val:.0%}", (cx + dx, cy + 0.05), ha="center",
                        va="center", fontsize=17, color=ink, zorder=4,
                        fontweight="bold" if dark else "normal")
            ax.annotate(lab, (cx + dx, cy - 0.14), ha="center", va="center",
                        fontsize=7, color=ink if dark else INK2, zorder=4)
        ax.annotate(note, (cx, cy - 0.33), ha="center", va="center",
                    fontsize=7.5, color=ink if dark else INK2, zorder=4)

    # one group label per axis, then a short label per column and row, so the
    # long phrase is not repeated twice and cannot collide with itself
    ax.annotate("says TRUE", (W / 2, 2 * H + 0.10), ha="center", va="center",
                fontsize=9, color=INK)
    ax.annotate("says FALSE", (W + W / 2, 2 * H + 0.10), ha="center",
                va="center", fontsize=9, color=INK)
    ax.annotate("FIRST PROBE", (W, 2 * H + 0.30), ha="center", va="center",
                fontsize=8, color=INK2, fontweight="bold")
    ax.annotate("says TRUE", (-0.13, H + H / 2), rotation=90, ha="center",
                va="center", fontsize=9, color=INK)
    ax.annotate("says FALSE", (-0.13, H / 2), rotation=90, ha="center",
                va="center", fontsize=9, color=INK)
    ax.annotate("SECOND PROBE", (-0.34, H), rotation=90, ha="center",
                va="center", fontsize=8, color=INK2, fontweight="bold")

    ax.annotate("every claim here is false, so saying TRUE is the error",
                (W, -0.20), ha="center", va="center", fontsize=8, color=INK2)
    ax.annotate(f"share of all false claims; same-family $n={n_s}$, "
                f"cross-family $n={n_c}$", (W, -0.35), ha="center", va="center",
                fontsize=7.5, color=GRAY)

    ax.set_xlim(-0.44, 2 * W + 0.02); ax.set_ylim(-0.44, 2 * H + 0.42)
    ax.set_aspect("equal"); ax.axis("off"); ax.set_facecolor(SURFACE)
    print("writing teaser:")
    save(fig, "fig0_teaser")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
