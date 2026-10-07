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
from matplotlib.patches import Circle, Rectangle
from scipy.optimize import brentq

from figstyle import BLUE, GRAY, INK, INK2, ORANGE, PROBES, SURFACE, load_base, save

PROBE_KEY = "lr"          # headline probe; mass-mean gives the same picture
# Box height; width is 1/H so the area stays 1. Tall enough for the widest
# circle, no taller -- the margin above and below the circles is not data.
BOX_H = 0.70


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
    """One panel: the four outcomes on false claims, drawn to scale.

    The box is every false claim. Each circle is the set one probe calls true,
    so its area is that probe's error rate; the lens is false agreement. What is
    left -- the box outside both circles -- is the case both probes get right,
    which is most of the area and is what makes the lens readable as a fraction
    of the whole rather than of the errors alone.

    Areas are exact: the box is a unit square, circle radii are sqrt(p/pi), and
    the centre distance is solved so the lens area equals the measured rate.
    """
    r1, r2 = np.sqrt(p1 / np.pi), np.sqrt(p2 / np.pi)
    d = centre_distance(r1, r2, fa)
    # The box has area 1 but is drawn wide rather than square. A square box is
    # mostly empty, because the circles only reach across its middle, and the
    # empty margin is not data -- the "both right" area is the same either way.
    # A wide box of the same area keeps every proportion exact and removes the
    # whitespace above and below the circles.
    H = BOX_H
    W = 1.0 / H
    span = d + r1 + r2
    cx = W / 2 - span / 2 + r1
    c1, c2 = (cx, H / 2), (cx + d, H / 2)

    ax.add_patch(Rectangle((0, 0), W, H, facecolor="white",
                           edgecolor=GRAY, linewidth=1.0, zorder=1))
    for (x, y), r, col in ((c1, r1, BLUE), (c2, r2, ORANGE)):
        ax.add_patch(Circle((x, y), r, facecolor=col, alpha=0.30,
                            edgecolor=col, linewidth=1.5, zorder=2))
    lens = Circle(c1, r1, facecolor=INK2, alpha=0.60, edgecolor="none", zorder=3)
    ax.add_patch(lens)
    lens.set_clip_path(Circle(c2, r2, transform=ax.transData))

    both_right = 1.0 - p1 - p2 + fa
    # each label at the middle of the region it names, not at the circle centre:
    # the lens is wide enough that a label placed at a circle's centre lands on
    # top of the lens label
    left, right = c2[0] - r2, c1[0] + r1          # the lens spans these
    ax.annotate(f"both wrong\n{fa:.0%}", ((left + right) / 2, H / 2),
                ha="center", va="center", fontsize=8.5, color="white",
                fontweight="bold", zorder=5)
    ax.annotate(f"only the weaker\nprobe wrong\n{p1 - fa:.0%}",
                (((c1[0] - r1) + left) / 2, H / 2), ha="center", va="center",
                fontsize=7.5, color=INK, zorder=5)
    ax.annotate(f"only the stronger\nprobe wrong\n{p2 - fa:.0%}",
                (c2[0] + r2 + 0.14, H / 2), ha="left", va="center",
                fontsize=7.5, color=INK, zorder=5)
    ax.annotate(f"both right  {both_right:.0%}", (W - 0.02, 0.025),
                ha="right", va="bottom", fontsize=8, color=INK2, zorder=5)
    ax.annotate("every false claim", (0.015, H - 0.02), ha="left", va="top",
                fontsize=7, color=GRAY, zorder=5)

    ax.set_xlim(-0.02, W + 0.02); ax.set_ylim(-0.02, H + 0.02)
    ax.set_aspect("equal"); ax.axis("off"); ax.set_facecolor(SURFACE)
    ax.set_title(title, fontsize=10.5, color=INK, pad=6)
    ax.annotate(subtitle, (0.5, -0.10), xycoords="axes fraction", ha="center",
                fontsize=7.5, color=INK2)


def main() -> int:
    df = load_base(PROBE_KEY)
    name = dict((k, n) for n, _, k in PROBES)[PROBE_KEY]
    groups = {f: df[df.family == f] for f in ("same", "cross")}
    # the unit box is the common scale, so no shared limit has to be computed
    lim = 1.0
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 2.9))
    for ax, fam, title in ((axes[0], "same", "Judge from the same model family"),
                           (axes[1], "cross", "Judge from a different family")):
        d = groups[fam]
        venn(ax, *rates(d), title, f"{len(d)} pair--dataset cells", lim)
    fig.suptitle("Only disagreement is visible. The overlap is not.",
                 fontsize=12, color=INK, y=1.04)
    fig.text(0.5, -0.06,
             "The box is every false claim; circle areas are the share each probe "
             "calls true, and the overlap is false agreement, which no "
             f"disagreement flags. All areas to scale, {name} probes.",
             ha="center", fontsize=8, color=INK2)
    fig.subplots_adjust(wspace=0.10)
    print("writing teaser:")
    save(fig, "fig0_teaser")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
