#!/usr/bin/env python3
"""Shared style and data loading for the paper's figures.

Split out so `make_figures.py` and `make_teaser.py` cannot drift: the teaser is
built from the same panels as Figures 1 and 2, and a reviewer comparing them
should not find different numbers.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402
import pandas as pd               # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cmb.config import RESULTS_DIR                      # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "paper" / "figures"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
GRAY, INK, INK2 = "#b9b8b4", "#0b0b0b", "#52514e"
SURFACE = "#fcfcfb"

PROBES = (("Logistic", BLUE, "lr"), ("Mass-mean", ORANGE, "mm"))

NICE = {"geometry_of_truth": "Geometry of Truth", "truthfulqa": "TruthfulQA",
        "boolq": "BoolQ", "imdb": "IMDB", "rte": "RTE"}
# datasets ordered by how often both models are wrong together -- the quantity
# every figure is about, so the order is the same everywhere
ORDER = ["geometry_of_truth", "imdb", "boolq", "truthfulqa", "rte"]

FAMILY = {"qwen3-1.7b": "qwen", "qwen3-8b": "qwen", "qwen3-8b-base": "qwen",
          "qwen3-32b": "qwen", "qwen38-27b": "qwen", "gemma4-12b": "gemma",
          "gemma4-12b-base": "gemma", "gemma4-31b": "gemma", "gemma2-9b": "gemma",
          "llama-8b": "llama", "llama31-8b-base": "llama", "llama31-70b": "llama",
          "llama31-70b-base": "llama", "llama31-405b": "llama",
          "olmo3-7b": "olmo", "gptj-6b": "gptj"}
SIZE = {"qwen3-1.7b": 1.7, "olmo3-7b": 7, "gptj-6b": 6, "qwen3-8b": 8,
        "qwen3-8b-base": 8, "llama-8b": 8, "llama31-8b-base": 8, "gemma2-9b": 9,
        "gemma4-12b": 12, "gemma4-12b-base": 12, "qwen38-27b": 27,
        "gemma4-31b": 31, "qwen3-32b": 32, "llama31-70b": 70,
        "llama31-70b-base": 70, "llama31-405b": 405}


def load_base(probe: str) -> pd.DataFrame:
    """Base sweep for one probe, deduplicated and annotated.

    The `--all-directions` tables hold every cell twice: false agreement, rho
    and coverage are all symmetric in the model pair, so (a,b) and (b,a) carry
    identical values. Averaging over the raw 1142 rows is harmless for a mean but
    wrong for any standard error, so the duplicate is dropped here once rather
    than in each figure.
    """
    df = pd.read_csv(RESULTS_DIR / f"sweep_base_all_{probe}.csv")
    key = df.apply(lambda r: tuple(sorted((r.overseer, r.target))) + (r.dataset,),
                   axis=1)
    df = df[~key.duplicated()].copy()
    df["family"] = [
        "same" if FAMILY.get(o) == FAMILY.get(t) else "cross"
        for o, t in zip(df.overseer, df.target)]
    df["gap"] = [max(SIZE[o], SIZE[t]) / min(SIZE[o], SIZE[t])
                 for o, t in zip(df.overseer, df.target)]
    return df


def load_loo(probe: str) -> pd.DataFrame:
    return pd.read_csv(RESULTS_DIR / f"sweep_loo_nomap16_{probe}.csv")


def load_indomain(probe: str) -> pd.DataFrame:
    return pd.read_csv(RESULTS_DIR / f"sweep_indomain_{probe}.csv")


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


def save(fig, name, png=True):
    """PDF for LaTeX, PNG beside it so the figure can be eyeballed without a
    build. Both are committed; the PDF is what `\\includegraphics` picks up."""
    OUT.mkdir(parents=True, exist_ok=True)
    fig.patch.set_facecolor(SURFACE)
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight", facecolor=SURFACE)
    if png:
        fig.savefig(OUT / f"{name}.png", bbox_inches="tight",
                    facecolor=SURFACE, dpi=200)
    plt.close(fig)
    print(f"  {name}.pdf" + (" + .png" if png else ""))
