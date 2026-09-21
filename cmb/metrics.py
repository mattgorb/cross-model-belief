"""Metrics and decision rules (DESIGN.md §7).

The 8-cell table and the Row-2 rate live here; so does the sign-resolved AUROC
convention and the confident-slice machinery Gate A turns on.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
from sklearn.metrics import roc_auc_score

from .config import CONF_SLICE_Q


def auroc_signed(score, y) -> float:
    """AUROC with CCS's arbitrary sign resolved by taking the better orientation.

    Only valid as a *diagnostic*: it spends a bit of label information. Where
    the sign has already been fixed on a calibration split (`CCSProbe.
    resolve_sign`), prefer `auroc` so the number is honest.
    """
    y = np.asarray(y)
    if len(np.unique(y)) < 2:
        return float("nan")
    a = roc_auc_score(y, score)
    return float(max(a, 1 - a))


def auroc(score, y) -> float:
    y = np.asarray(y)
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, score))


def confidence(p_yes: np.ndarray) -> np.ndarray:
    """Output-space confidence: distance of P(Yes) from the 0.5 fence."""
    return np.abs(np.asarray(p_yes) - 0.5) * 2


def confident_slice(p_yes: np.ndarray, q: float = CONF_SLICE_Q) -> np.ndarray:
    """Boolean mask for the top-q fraction of items by output confidence.

    Gate A lives or dies on this slice: the confident-*wrong* items are the
    natural high-confidence errors, and a probe that only works where output
    confidence already works is reading confidence, not truth.
    """
    c = confidence(p_yes)
    return c >= np.quantile(c, 1 - q)


@dataclass
class EightCell:
    """The DESIGN.md §2.1 joint table for one model pair on one dataset."""
    counts: dict[str, int]
    n: int
    row2_rate: float             # P(both say True | GT False) — the headline
    row2_count: int
    n_false: int
    agreement_rate: float
    router_fire_rate: float      # fraction of items where disagreement flags
    error_correlation: float     # phi between the two probes' errors
    accuracy: tuple[float, float]

    def to_dict(self):
        return asdict(self)

    def render(self) -> str:
        order = [("1", "T", "T", "T"), ("2", "T", "T", "F"), ("3", "T", "F", "T"),
                 ("4", "T", "F", "F"), ("5", "F", "F", "T"), ("6", "F", "F", "F"),
                 ("7", "F", "T", "T"), ("8", "F", "T", "F")]
        lines = ["  #  m1  m2  GT   count   note",
                 "  -- --- --- ---- ------- ----"]
        note = {"1": "correct agreement", "2": "<-- FALSE AGREEMENT",
                "3": "router fires", "4": "router fires", "5": "router fires",
                "6": "correct agreement", "7": "router fires", "8": "router fires"}
        for num, m1, m2, gt in order:
            k = f"{m1}{m2}{gt}"
            lines.append(f"  {num:>2} {m1:>3} {m2:>3} {gt:>4} {self.counts[k]:>7}"
                         f"   {note[num]}")
        lines += [
            f"\n  n = {self.n}   (GT-false items: {self.n_false})",
            f"  Row-2 rate  P(both True | GT False) = {self.row2_rate:.3f}"
            f"  [{self.row2_count}/{self.n_false}]",
            f"  agreement rate                      = {self.agreement_rate:.3f}",
            f"  router fire rate (disagreement)     = {self.router_fire_rate:.3f}",
            f"  probe error correlation (phi)       = {self.error_correlation:.3f}",
            f"  probe accuracy m1 / m2              = "
            f"{self.accuracy[0]:.3f} / {self.accuracy[1]:.3f}",
        ]
        return "\n".join(lines)


def eight_cell(v1: np.ndarray, v2: np.ndarray, gt: np.ndarray) -> EightCell:
    """Fill the 8-cell table from two probe verdicts and ground truth."""
    v1, v2, gt = map(lambda a: np.asarray(a).astype(int), (v1, v2, gt))
    counts = {}
    for a in (1, 0):
        for b in (1, 0):
            for g in (1, 0):
                k = f"{'T' if a else 'F'}{'T' if b else 'F'}{'T' if g else 'F'}"
                counts[k] = int(((v1 == a) & (v2 == b) & (gt == g)).sum())

    n_false = int((gt == 0).sum())
    row2 = counts["TTF"]
    e1, e2 = (v1 != gt), (v2 != gt)
    return EightCell(
        counts=counts, n=len(gt),
        row2_rate=float(row2 / n_false) if n_false else float("nan"),
        row2_count=row2, n_false=n_false,
        agreement_rate=float((v1 == v2).mean()),
        router_fire_rate=float((v1 != v2).mean()),
        error_correlation=phi(e1, e2),
        accuracy=(float((v1 == gt).mean()), float((v2 == gt).mean())))


def phi(a: np.ndarray, b: np.ndarray) -> float:
    """Phi coefficient between two boolean vectors (here: the two error sets).

    This is what drives the amplification factor — how much coverage a label
    buys. Independent errors mean disagreement finds most of them; correlated
    errors mean Row 2 swallows them.
    """
    a, b = np.asarray(a).astype(float), np.asarray(b).astype(float)
    if a.std() < 1e-12 or b.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def row2_mask(v1, v2, gt) -> np.ndarray:
    """Items in Row 2: both probes say true, ground truth is false."""
    v1, v2, gt = map(np.asarray, (v1, v2, gt))
    return (v1 == 1) & (v2 == 1) & (gt == 0)


def both_true_mask(v1, v2) -> np.ndarray:
    """The {both say True} subset — Experiment 3's universe."""
    return (np.asarray(v1) == 1) & (np.asarray(v2) == 1)


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson 95% CI. Row-2 counts are small by construction (DESIGN.md §4
    data-scarcity note), so a bare rate without an interval is misleading."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (float(max(0.0, c - h)), float(min(1.0, c + h)))
