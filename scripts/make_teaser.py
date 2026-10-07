#!/usr/bin/env python3
"""Build the teaser (Figure 1): what disagreement-based oversight cannot see.

Two circles, one per probe, each covering the false claims that probe calls
true. Their union is everything at least one reader gets wrong; the lens where
they overlap is \\emph{false agreement}, the case both get wrong together. Only
the crescents produce a disagreement, so only the crescents are visible to an
oversight scheme that routes on disagreement. The lens is the blind spot.

The circles are area-proportional: circle areas are the measured error rates
$p_1$ and $p_2$, and the lens area is the measured false-agreement rate, with the
centre distance solved numerically so the lens comes out right. The picture is
therefore a drawing of the data, not an illustration beside it.

Panel B repeats the construction for cross-family pairs. If choosing a judge
from another model family shrank the blind spot, the lens would be visibly
smaller. It is not.

    scripts/make_teaser.py
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle
from scipy.optimize import brentq

from figstyle import BLUE, GRAY, INK, INK2, ORANGE, PROBES, SURFACE, load_base, save

PROBE_KEY = "lr"          # headline probe; mass-mean gives the same picture


def lens_area(d, r1, r2):
    """Area of the intersection of two circles, centres `d` apart."""
    if d >= r1 + r2:
        return 0.0
    if d <= abs(r1 - r2):
        return np.pi * min(r1, r2) ** 2
    a1 = r1 ** 2 * np.arccos((d ** 2 + r1 ** 2 - r2 ** 2) / (2 * d * r1))
    a2 = r2 ** 2 * np.arccos((d ** 2 + r2 ** 2 - r1 ** 2) / (2 * d * r2))
    a3 = 0.5 * np.sqrt(max(0.0, (-d + r1 + r2) * (d + r1 - r2)
                           * (d - r1 + r2) * (d + r1 + r2)))
    return a1 + a2 - a3


def centre_distance(r1, r2, target):
    """Centre distance that makes the lens area equal `target`."""
    lo, hi = abs(r1 - r2) + 1e-9, r1 + r2 - 1e-9
    if lens_area(hi, r1, r2) >= target:
        return hi
    if lens_area(lo, r1, r2) <= target:
        return lo
    return brentq(lambda d: lens_area(d, r1, r2) - target, lo, hi)


def rates(d):
    """(weaker probe's error rate, stronger probe's, false agreement).

    Not (overseer, target): the base table holds each pair in both orders,
    because false agreement and the error correlation are symmetric in the two
    models, and deduplicating it leaves which model is called the overseer
    decided by row order. Taking the per-cell max and min instead is invariant
    to that, and names something real --- the smaller circle is exactly the
    Frechet ceiling min(p1, p2), so the figure also shows that the overlap can
    never be larger than the stronger probe's own error rate.
    """
    import numpy as _np
    return (_np.maximum(d.p_overseer, d.p_target).mean(),
            _np.minimum(d.p_overseer, d.p_target).mean(),
            d.fa_rate.mean())


def venn(ax, p1, p2, fa, title, subtitle, lim):
    r1, r2 = np.sqrt(p1 / np.pi), np.sqrt(p2 / np.pi)
    d = centre_distance(r1, r2, fa)
    c1, c2 = (-d / 2, 0.0), (d / 2, 0.0)

    for (cx, cy), r, col in ((c1, r1, BLUE), (c2, r2, ORANGE)):
        ax.add_patch(Circle((cx, cy), r, facecolor=col, alpha=0.30,
                            edgecolor=col, linewidth=1.6, zorder=2))
    # the lens, drawn by clipping one circle against the other
    lens = Circle(c1, r1, facecolor=INK2, alpha=0.55, edgecolor="none", zorder=3)
    ax.add_patch(lens)
    lens.set_clip_path(Circle(c2, r2, transform=ax.transData))

    ax.annotate(f"both wrong\n{fa:.1%}", (0, 0), ha="center", va="center",
                fontsize=8.5, color=SURFACE, zorder=4, fontweight="bold")
    # placed on the far side of each crescent, above the midline, so neither
    # label can land on the lens however the circles are sized
    ax.annotate("weaker probe\nalone wrong", (c1[0] - r1 * 0.52, r1 * 0.42),
                ha="center", va="center", fontsize=7.5, color=INK2, zorder=4)
    ax.annotate("stronger probe\nalone wrong", (c2[0] + r2 * 0.55, r2 * 0.46),
                ha="center", va="center", fontsize=7.5, color=INK2, zorder=4)

    # one scale for both panels: with independent limits the circles would be
    # drawn the same size whatever the error rates, and the comparison the
    # figure exists to make would be invisible
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim * 0.74, lim * 0.74)
    ax.set_aspect("equal"); ax.axis("off")
    ax.set_facecolor(SURFACE)
    ax.set_title(title, fontsize=10.5, color=INK, loc="center", pad=4)
    ax.annotate(subtitle, (0.5, -0.02), xycoords="axes fraction", ha="center",
                fontsize=8, color=INK2)


def main() -> int:
    df = load_base(PROBE_KEY)
    name = dict((k, n) for n, _, k in PROBES)[PROBE_KEY]
    groups = {f: df[df.family == f] for f in ("same", "cross")}
    lim = 0.0
    for d in groups.values():
        p1, p2, fa = rates(d)
        r1, r2 = np.sqrt(p1 / np.pi), np.sqrt(p2 / np.pi)
        lim = max(lim, max(r1, r2) + centre_distance(r1, r2, fa) / 2 + 0.05)

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.5))
    for ax, fam, title in ((axes[0], "same", "Judge from the same model family"),
                           (axes[1], "cross", "Judge from a different family")):
        d = groups[fam]
        venn(ax, *rates(d), title, f"{len(d)} pair--dataset cells", lim)
    fig.suptitle("Only disagreement is visible. The overlap is not.",
                 fontsize=12, color=INK, y=1.02)
    fig.text(0.5, -0.07,
             "Circle areas are the share of false claims each probe calls true; "
             "the overlap is false agreement, which no disagreement flags. "
             f"Areas to scale across both panels, {name} probes.",
             ha="center", fontsize=8, color=INK2)
    fig.subplots_adjust(wspace=0.08)
    print("writing teaser:")
    save(fig, "fig0_teaser")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
