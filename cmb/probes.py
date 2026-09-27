"""Probes: the unsupervised CCS belief probe, and the supervised direction used
to ask whether Row 2 is separable.

Two different objects, deliberately kept apart:

  `CCSProbe`      — no labels, fit per model on contrast pairs. This is the
                    "belief" instrument whose validity Gate A interrogates.
  `LinearDirection` — a plain supervised logistic direction. Used only inside
                    the {both say True} subset for Experiments 3 and 4, where
                    labels are available by construction of the experiment.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from .config import SEED


def standardizer(X: np.ndarray):
    """Mean/scale normalizer fitted on train activations.

    CCS is scale-sensitive and activation norms differ wildly across models and
    layers, so every probe sees normalized input and the *same* normalizer is
    applied at test time.
    """
    mu = X.mean(0, keepdims=True)
    sd = X.std(0, keepdims=True) + 1e-6
    return lambda Z: (Z - mu) / sd


class CCSProbe(nn.Module):
    """Contrast-Consistent Search (Burns et al. 2022).

        loss = consistency [p+ - (1 - p-)]^2 + confidence min(p+, p-)^2

    Unsupervised. The sign of the learned direction is arbitrary — it is
    resolved against labels *outside* the fit, in `resolve_sign`, which is a
    validation step and not part of training.
    """

    def __init__(self, d: int):
        super().__init__()
        self.w = nn.Linear(d, 1)
        self.sign = 1.0
        self._norm = None

    # -- fitting ------------------------------------------------------------

    def fit(self, Xp: np.ndarray, Xn: np.ndarray, epochs: int = 1000,
            lr: float = 1e-3, ntries: int = 10, weight_decay: float = 0.0,
            seed: int = SEED) -> float:
        """Fit with restarts (CCS is notoriously seed-sensitive)."""
        self._norm = standardizer(np.concatenate([Xp, Xn], 0))
        tp = torch.tensor(self._norm(Xp), dtype=torch.float32)
        tn = torch.tensor(self._norm(Xn), dtype=torch.float32)
        g = torch.Generator().manual_seed(seed)
        best_state, best_loss = None, float("inf")
        for t in range(ntries):
            with torch.no_grad():
                bound = 1.0 / np.sqrt(self.w.in_features)
                self.w.weight.uniform_(-bound, bound, generator=g)
                self.w.bias.uniform_(-bound, bound, generator=g)
            opt = torch.optim.Adam(self.parameters(), lr=lr,
                                   weight_decay=weight_decay)
            loss = torch.tensor(float("nan"))
            for _ in range(epochs):
                opt.zero_grad()
                pp = torch.sigmoid(self.w(tp))
                pn = torch.sigmoid(self.w(tn))
                loss = ((pp - (1 - pn)) ** 2).mean() + torch.min(pp, pn).pow(2).mean()
                loss.backward()
                opt.step()
            if loss.item() < best_loss:
                best_loss = loss.item()
                best_state = {k: v.detach().clone() for k, v in self.state_dict().items()}
        self.load_state_dict(best_state)
        return best_loss

    # -- inference ----------------------------------------------------------

    @torch.no_grad()
    def _raw(self, X: np.ndarray) -> np.ndarray:
        Z = self._norm(X) if self._norm is not None else X
        t = torch.tensor(Z, dtype=torch.float32)
        return torch.sigmoid(self.w(t)).squeeze(-1).numpy()

    def belief(self, Xp: np.ndarray, Xn: np.ndarray | None = None) -> np.ndarray:
        """P(claim is true) in [0, 1], sign-resolved.

        With both halves supplied we use the CCS-consistent average
        (p+ + (1 - p-)) / 2, which is lower-variance than either half alone.
        """
        p = self._raw(Xp)
        if Xn is not None:
            p = 0.5 * (p + (1 - self._raw(Xn)))
        return p if self.sign > 0 else 1 - p

    def vote(self, Xp, Xn=None, threshold: float = 0.5) -> np.ndarray:
        """The model's internal true/false verdict — the input to the 8-cell table."""
        return (self.belief(Xp, Xn) >= threshold).astype(int)

    def resolve_sign(self, Xp, Xn, labels) -> float:
        """Pick the orientation that agrees with labels on a *calibration* split.

        CCS gives no orientation, so one bit of label information is unavoidable.
        Keeping it here, on data disjoint from the test split, makes it explicit
        and auditable rather than hidden inside `max(a, 1-a)` at scoring time.
        """
        self.sign = 1.0
        agree = ((self.belief(Xp, Xn) >= 0.5).astype(int) == labels).mean()
        self.sign = 1.0 if agree >= 0.5 else -1.0
        return max(agree, 1 - agree)

    def direction(self) -> np.ndarray:
        w = self.w.weight.detach().numpy().ravel()
        return self.sign * w


@dataclass
class LinearDirection:
    """Supervised logistic direction (Experiments 3 and 4).

    Deliberately a plain linear model: the claim under test is that Row 2 is a
    *direction/region*, so a nonlinear classifier would answer a different,
    weaker question.
    """
    w: np.ndarray
    b: float
    mu: np.ndarray
    sd: np.ndarray

    @classmethod
    def fit(cls, X: np.ndarray, y: np.ndarray, C: float = 1.0,
            seed: int = SEED) -> "LinearDirection":
        from sklearn.linear_model import LogisticRegression

        mu, sd = X.mean(0), X.std(0) + 1e-6
        clf = LogisticRegression(C=C, max_iter=2000, random_state=seed)
        clf.fit((X - mu) / sd, y)
        return cls(clf.coef_.ravel(), float(clf.intercept_[0]), mu, sd)

    def score(self, X: np.ndarray) -> np.ndarray:
        return ((X - self.mu) / self.sd) @ self.w + self.b


class SupervisedBeliefProbe:
    """Labeled baselines for the belief probe, with the `CCSProbe` interface.

    Same data, same activations, same layer as CCS — only the fitting rule
    changes, which is the comparison paper/PLAN.md §3 asks for:

      `mass-mean`  theta = mean(F | true) - mean(F | false), the strongest causal
                   baseline in Marks & Tegmark (2024);
      `lr`         logistic regression on the same features; cheap, and expected
                   to be largely redundant with mass-mean. Reporting it is how we
                   show that rather than assert it.

    Features are the *contrast difference* `pos - neg`, so the probe reads the
    same object CCS effectively reads and the two are comparable at the same
    site. These probes use labels, so they are baselines for the unsupervised
    case, never a substitute for it: only CCS answers the question of whether
    belief is recoverable without supervision.

    `sign` exists only for interface parity — a supervised fit already has an
    orientation, and `resolve_sign` is a no-op that reports train agreement.
    """

    KINDS = ("mass-mean", "lr")

    def __init__(self, d: int, kind: str = "mass-mean"):
        if kind not in self.KINDS:
            raise ValueError(f"kind must be one of {self.KINDS}, got {kind!r}")
        self.kind = kind
        self.d = d
        self.sign = 1.0
        self._norm = None
        self.w = None
        self.b = 0.0
        self._scale = 1.0

    # -- fitting ------------------------------------------------------------

    def fit(self, Xp: np.ndarray, Xn: np.ndarray, labels: np.ndarray,
            seed: int = SEED, **kw) -> float:
        """Fit on the train split. Returns train accuracy (not a loss)."""
        self._norm = standardizer(np.concatenate([Xp, Xn], 0))
        F = self._norm(Xp) - self._norm(Xn)
        y = np.asarray(labels).astype(int)

        if self.kind == "mass-mean":
            mu_t, mu_f = F[y == 1].mean(0), F[y == 0].mean(0)
            self.w = mu_t - mu_f
            # Threshold midway between the class means along theta — the plain
            # mass-mean classifier, no extra fitting.
            self.b = -0.5 * float((mu_t + mu_f) @ self.w)
        else:
            from sklearn.linear_model import LogisticRegression

            clf = LogisticRegression(max_iter=2000, random_state=seed).fit(F, y)
            self.w = clf.coef_.ravel()
            self.b = float(clf.intercept_[0])

        s = F @ self.w + self.b
        self._scale = float(np.std(s)) + 1e-9
        return float(((s >= 0).astype(int) == y).mean())

    # -- inference ----------------------------------------------------------

    def _score(self, Xp: np.ndarray, Xn: np.ndarray) -> np.ndarray:
        F = self._norm(Xp) - self._norm(Xn)
        return F @ self.w + self.b

    def belief(self, Xp: np.ndarray, Xn: np.ndarray | None = None) -> np.ndarray:
        """P(claim is true) in [0, 1].

        The squashing is monotone in the projection and scaled by the train
        spread, so AUROC is unaffected and the 0.5 fence is the fitted decision
        boundary — the verdict the 8-cell table wants.
        """
        if Xn is None:
            raise ValueError("supervised probes read the contrast pair; pass Xn")
        s = self._score(Xp, Xn) / self._scale
        p = 1.0 / (1.0 + np.exp(-s))
        return p if self.sign > 0 else 1 - p

    def vote(self, Xp, Xn=None, threshold: float = 0.5) -> np.ndarray:
        return (self.belief(Xp, Xn) >= threshold).astype(int)

    def resolve_sign(self, Xp, Xn, labels) -> float:
        """No-op: a supervised fit is already oriented. Reports train agreement."""
        self.sign = 1.0
        return float(((self.belief(Xp, Xn) >= 0.5).astype(int)
                      == np.asarray(labels).astype(int)).mean())

    def direction(self) -> np.ndarray:
        return self.sign * np.asarray(self.w).ravel()


PROBE_KINDS = ("ccs",) + SupervisedBeliefProbe.KINDS


def make_probe(kind: str, d: int):
    """Probe factory. `ccs` is the case that matters; the rest are baselines."""
    if kind == "ccs":
        return CCSProbe(d)
    return SupervisedBeliefProbe(d, kind)
