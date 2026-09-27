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


# ---------------------------------------------------------------------------
# The algebra of false agreement (DESIGN.md §2.4; paper/theory.tex).
#
# Everything below is restricted to the GT-false items, which are the only ones
# on which false agreement is defined. Note the difference from
# `EightCell.error_correlation`, which is phi between the two *error* vectors
# over *all* items: that is a summary of the whole table, while `rho` here is
# phi between the two false-positive indicators on the false items only, and it
# is the rho that enters the identity below.
# ---------------------------------------------------------------------------


def false_positive_rate(v: np.ndarray, gt: np.ndarray) -> float:
    """P(probe says True | GT False) — the p_i of the identity."""
    v, gt = np.asarray(v).astype(int), np.asarray(gt).astype(int)
    n_false = int((gt == 0).sum())
    if n_false == 0:
        return float("nan")
    return float(((v == 1) & (gt == 0)).sum() / n_false)


def fa_from_correlation(p1: float, p2: float, rho: float) -> float:
    """FA = p1*p2 + rho*sqrt(p1(1-p1))*sqrt(p2(1-p2))   (exact, not a bound).

    Holds because Cov of two indicators is P(both) - p1*p2. The point of the
    decomposition is that the second term can dominate: two accurate probes with
    correlated errors beat two worse probes with independent errors, the wrong
    way round.
    """
    return float(p1 * p2 + rho * np.sqrt(p1 * (1 - p1)) * np.sqrt(p2 * (1 - p2)))


def frechet_bounds(p1: float, p2: float) -> tuple[float, float]:
    """(lower, upper) on FA with the marginals fixed: max(0, p1+p2-1), min(p1,p2).

    The upper bound is maximal error overlap — every error of the better probe is
    also an error of the worse one, i.e. the second reader adds nothing (the
    self-oversight degenerate case). The lower bound needs anti-correlated
    errors, which is not a realistic operating point for models sharing
    pretraining data.
    """
    return (float(max(0.0, p1 + p2 - 1.0)), float(min(p1, p2)))


def rho_feasible_range(p1: float, p2: float) -> tuple[float, float]:
    """(rho_min, rho_max) attainable given the marginals.

    rho spans [-1, 1] only when p1 == p2, so a raw phi is not comparable across
    pairs with different false-positive rates — hence `normalized_rho`.
    """
    s = np.sqrt(p1 * (1 - p1) * p2 * (1 - p2))
    if s < 1e-12:
        return (float("nan"), float("nan"))
    lo, hi = frechet_bounds(p1, p2)
    return (float((lo - p1 * p2) / s), float((hi - p1 * p2) / s))


def normalized_rho(rho: float, p1: float, p2: float) -> float:
    """rho as a fraction of its feasible extreme, in [-1, 1].

    +1 means maximal attainable error overlap for these marginals (not rho = 1),
    0 means independence.
    """
    lo, hi = rho_feasible_range(p1, p2)
    if not np.isfinite(lo) or not np.isfinite(hi):
        return float("nan")
    ref = hi if rho >= 0 else abs(lo)
    return float(rho / ref) if ref > 1e-12 else float("nan")


def detectable_coverage(fa: float, p1: float, p2: float) -> float:
    """Fraction of the *stronger* probe's false positives a router would flag.

        C = 1 - FA / min(p1, p2)

    90% under independence in the running example (p1=10%, p2=5%), 0% under
    maximal overlap. The complement of the headline number, from the same table:
    disagreement flags an item without saying which reader is wrong, so C is
    coverage bought per labeled check, not a correction rate.
    """
    p_strong = min(p1, p2)
    if p_strong < 1e-12:
        return float("nan")
    return float(1.0 - fa / p_strong)


def n_eff(n: int, rho_bar: float) -> float:
    """Effective number of independent overseers, N / (1 + (N-1) * rho_bar).

    Standard ensemble theory (bias-variance-covariance), cited not claimed. The
    correlation floor: as N grows this tends to 1/rho_bar, so once rho_bar is
    positive, adding overseers stops helping. rho_bar = 0.8 caps the pool at 1.25
    effective readers; it takes rho_bar ~ 0.3 for N = 20 to be worth about 3.
    """
    d = 1.0 + (n - 1) * rho_bar
    if d < 1e-12:
        return float("nan")
    return float(n / d)


def threshold_at_positive_rate(score: np.ndarray, rate: float) -> float:
    """Threshold that makes a probe fire on `rate` of items.

    Used to compare a native probe against a transported one at *matched*
    positive rate, so the Gate B degradation epsilon comes out as a realized
    false-positive-rate inflation in the same units as p2, rather than as an
    AUROC gap that has to be converted.
    """
    score = np.asarray(score, dtype=float)
    if score.size == 0 or not 0.0 < rate < 1.0:
        return 0.5
    return float(np.quantile(score, 1.0 - rate))


@dataclass
class FalseAgreement:
    """The identity, its bounds, and the coverage it implies, for one pair.

    Reported alongside the raw rate because the rate alone does not say whether
    a pair is near independence or near maximal overlap — and that, not the
    individual accuracies, is what bounds the oversight signal.
    """
    fa: float                       # P(both True | GT False) = the Row-2 rate
    count: int
    n_false: int
    p1: float                       # each probe's false-positive rate
    p2: float
    rho: float                      # phi between the two FP indicators
    rho_range: tuple[float, float]
    rho_normalized: float
    bounds: tuple[float, float]     # Frechet (lower, upper)
    fa_independent: float           # p1 * p2, the reference point
    coverage: float                 # 1 - FA / min(p1, p2)
    identity_residual: float        # sanity check: should be ~0

    def to_dict(self):
        return asdict(self)

    def render(self) -> str:
        lo, hi = self.bounds
        rlo, rhi = self.rho_range
        pos = ("near maximal overlap — the second reader adds little"
               if self.rho_normalized > 0.66 else
               "partial decorrelation — some independence, not full"
               if self.rho_normalized > 0.2 else
               "near independence — errors coincide about as often as by chance")
        return "\n".join([
            f"  false-positive rates   p1 = {self.p1:.3f}   p2 = {self.p2:.3f}",
            f"  FA (both True | False) = {self.fa:.4f}"
            f"   [{self.count}/{self.n_false}]",
            f"    independence ref     = {self.fa_independent:.4f}  (p1*p2)",
            f"    Frechet bounds       = [{lo:.4f}, {hi:.4f}]",
            f"  error correlation rho  = {self.rho:+.3f}"
            f"   feasible [{rlo:+.3f}, {rhi:+.3f}]",
            f"    rho / feasible max   = {self.rho_normalized:+.3f}   <-- {pos}",
            f"  detectable coverage    = {self.coverage:.3f}"
            f"   (share of the stronger probe's false positives a router flags)",
            f"  identity residual      = {self.identity_residual:+.2e}",
        ])


def false_agreement(v1: np.ndarray, v2: np.ndarray, gt: np.ndarray) -> FalseAgreement:
    """Decompose the false-agreement rate for one pair of verdicts.

    The FA number itself is `EightCell.row2_rate`; this adds the decomposition
    that says whether the number is small because the probes are accurate or
    because their errors are independent — two very different findings
    (DESIGN.md §2.3's floor vs ceiling).
    """
    v1, v2, gt = map(lambda a: np.asarray(a).astype(int), (v1, v2, gt))
    f = gt == 0
    n_false = int(f.sum())
    if n_false == 0:
        nan = float("nan")
        return FalseAgreement(nan, 0, 0, nan, nan, nan, (nan, nan), nan,
                              (nan, nan), nan, nan, nan)

    e1, e2 = (v1[f] == 1), (v2[f] == 1)
    p1, p2 = float(e1.mean()), float(e2.mean())
    count = int((e1 & e2).sum())
    fa = count / n_false
    rho = phi(e1, e2)
    # With a degenerate marginal phi is undefined; recover it from the identity
    # where possible so the residual check below stays meaningful.
    resid = (fa - fa_from_correlation(p1, p2, rho)) if np.isfinite(rho) else float("nan")
    return FalseAgreement(
        fa=fa, count=count, n_false=n_false, p1=p1, p2=p2, rho=rho,
        rho_range=rho_feasible_range(p1, p2),
        rho_normalized=normalized_rho(rho, p1, p2) if np.isfinite(rho) else float("nan"),
        bounds=frechet_bounds(p1, p2),
        fa_independent=float(p1 * p2),
        coverage=detectable_coverage(fa, p1, p2),
        identity_residual=float(resid))
