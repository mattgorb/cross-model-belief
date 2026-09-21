"""Cross-model transport: the linear alignment map, and CKA as map context.

The map is the project's transport mechanism (DESIGN.md §2.2). It is also the
project's main silent failure mode — it reaches only as far as the aligned
subspace — so `fit_map` is deliberately fit on a *train* split and scored on a
held-out one everywhere it is used.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class LinearMap:
    """Ridge map from A-space to B-space, fitted on paired activations."""
    W: np.ndarray            # [d_a, d_b]
    b: np.ndarray            # [d_b]
    mu_a: np.ndarray
    mu_b: np.ndarray
    alpha: float

    def __call__(self, Xa: np.ndarray) -> np.ndarray:
        return (Xa - self.mu_a) @ self.W + self.mu_b + self.b

    def r2(self, Xa: np.ndarray, Xb: np.ndarray) -> float:
        """Fraction of B's variance the map explains — how much got carried."""
        pred = self(Xa)
        ss_res = ((Xb - pred) ** 2).sum()
        ss_tot = ((Xb - Xb.mean(0)) ** 2).sum()
        return float(1 - ss_res / ss_tot)


ALPHA_GRID = (1e-3, 1e-2, 1e-1, 1.0, 10.0)


def _solve(A: np.ndarray, B: np.ndarray, alpha: float) -> np.ndarray:
    """Ridge solve with the penalty scaled to the data, so one alpha grid works
    across models and layers whose activation norms differ by orders."""
    d = A.shape[1]
    scale = np.trace(A.T @ A) / d
    return np.linalg.solve(A.T @ A + alpha * scale * np.eye(d), A.T @ B)


def fit_map(Xa: np.ndarray, Xb: np.ndarray,
            alpha: float | None = None, seed: int = 0) -> LinearMap:
    """Ridge least squares A -> B on paired activations (same inputs, both models).

    Activation dimension routinely exceeds the number of paired items, so the
    problem is underdetermined and the ridge strength matters more than the
    solver. With `alpha=None` it is chosen on an inner validation split by
    held-out R^2 — an alpha tuned on the fit itself would report a map that is
    memorizing the pairs rather than carrying structure.
    """
    mu_a, mu_b = Xa.mean(0), Xb.mean(0)
    A, B = Xa - mu_a, Xb - mu_b
    if alpha is None:
        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(A))
        cut = max(1, int(0.8 * len(A)))
        tr, va = idx[:cut], idx[cut:]
        best, best_r2 = ALPHA_GRID[len(ALPHA_GRID) // 2], -np.inf
        if len(va) >= 5:
            for cand in ALPHA_GRID:
                W = _solve(A[tr], B[tr], cand)
                pred = A[va] @ W
                r2 = 1 - ((B[va] - pred) ** 2).sum() / ((B[va] - B[va].mean(0)) ** 2).sum()
                if r2 > best_r2:
                    best, best_r2 = cand, r2
        alpha = best
    W = _solve(A, B, alpha)
    return LinearMap(W=W, b=np.zeros(Xb.shape[1]), mu_a=mu_a, mu_b=mu_b, alpha=alpha)


def fit_map_both_ways(Xa, Xb, alpha: float | None = None):
    """(A->B, B->A). Experiment 2 needs both directions."""
    return fit_map(Xa, Xb, alpha), fit_map(Xb, Xa, alpha)


def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """Linear CKA between two activation sets over the same items.

    Reported as *context* for a transfer number, never as a result on its own:
    the HELIX finding is that tokenizer compatibility and size gap drive
    alignment, so a weak cross-family transfer with low CKA is a map problem,
    not necessarily a probe problem.
    """
    X = X - X.mean(0)
    Y = Y - Y.mean(0)
    hsic = np.linalg.norm(X.T @ Y, "fro") ** 2
    nx = np.linalg.norm(X.T @ X, "fro")
    ny = np.linalg.norm(Y.T @ Y, "fro")
    return float(hsic / (nx * ny + 1e-12))
