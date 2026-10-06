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
    """Ridge map from A-space to B-space, fitted on paired activations.

    Both spaces are standardized *per dimension* before the fit and destandardized
    after. This is not cosmetic: LLM activations contain a handful of massive
    dimensions whose scale dwarfs the rest (measured on qwen3-8b, the largest
    per-dimension std is 542x the median at 0.5 depth, and the variance has an
    effective rank of ~1). Mean-centering alone leaves them, and then both the
    least-squares objective and the R^2 that scores it are answering "did you
    predict the one giant dimension", which is easy and uninformative. Scaling
    makes the map fit the directions that actually distinguish items.
    """
    W: np.ndarray            # [d_a, d_b]
    b: np.ndarray            # [d_b]
    mu_a: np.ndarray
    mu_b: np.ndarray
    alpha: float
    sd_a: np.ndarray | None = None   # None = a legacy mean-centred-only map
    sd_b: np.ndarray | None = None

    def __call__(self, Xa: np.ndarray) -> np.ndarray:
        if self.sd_a is None:
            return (Xa - self.mu_a) @ self.W + self.mu_b + self.b
        z = (Xa - self.mu_a) / self.sd_a
        return (z @ self.W + self.b) * self.sd_b + self.mu_b

    def r2(self, Xa: np.ndarray, Xb: np.ndarray) -> float:
        """Fraction of B's variance the map explains — how much got carried.

        Scored per dimension on the map's own scale, so a few massive dimensions
        cannot carry the number. On raw activations this reads 0.99 at mid-depth
        purely because one dimension holds nearly all the variance.
        """
        pred, Xb = self(Xa), np.asarray(Xb, dtype=float)
        sd = self.sd_b if self.sd_b is not None else 1.0
        res, tgt = (Xb - pred) / sd, (Xb - Xb.mean(0)) / sd
        return float(1 - (res ** 2).sum() / (tgt ** 2).sum())


ALPHA_GRID = (1e-3, 1e-2, 1e-1, 1.0, 10.0)


def _gram(A: np.ndarray) -> np.ndarray:
    """`A A^T`, the n-by-n Gram matrix the dual ridge solve needs."""
    return A @ A.T


def _dual_weights(G: np.ndarray, B: np.ndarray, alpha: float,
                  scale: float) -> np.ndarray:
    """`(A A^T + lambda I)^-1 B`, the dual coefficients. `W = A^T @ this`."""
    n = G.shape[0]
    return np.linalg.solve(G + alpha * scale * np.eye(n), B)


def _penalty_scale(A: np.ndarray, G: np.ndarray | None = None) -> float:
    """Penalty normalizer: mean squared singular value, as the primal form used.

    `trace(A^T A) == trace(A A^T)`, so this is identical whichever Gram matrix
    is to hand and the alpha grid keeps its original meaning.
    """
    t = np.trace(G) if G is not None else (A * A).sum()
    return float(t) / A.shape[1]


def _solve(A: np.ndarray, B: np.ndarray, alpha: float) -> np.ndarray:
    """Ridge solve with the penalty scaled to the data, so one alpha grid works
    across models and layers whose activation norms differ by orders.

    Solved in whichever of the two equivalent forms is smaller. The primal normal
    equations invert a d-by-d matrix, which at d=16384 (the 405B) is a 2.4e12-flop
    operation repeated once per alpha per direction -- around five minutes per
    cell, and the reason a full sweep was projected at six hours. The identity

        (A^T A + lambda I)^-1 A^T  ==  A^T (A A^T + lambda I)^-1

    moves that to an n-by-n inverse, and n here is the item count, so for any
    model wider than its dataset the dual form is the cheaper side. Same answer
    to floating point, so this is a speed choice and nothing else.
    """
    n, d = A.shape
    if n < d:
        G = _gram(A)
        return A.T @ _dual_weights(G, B, alpha, _penalty_scale(A, G))
    scale = _penalty_scale(A)
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
    sd_a, sd_b = Xa.std(0) + 1e-6, Xb.std(0) + 1e-6
    A, B = (Xa - mu_a) / sd_a, (Xb - mu_b) / sd_b
    if alpha is None:
        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(A))
        cut = max(1, int(0.8 * len(A)))
        tr, va = idx[:cut], idx[cut:]
        best, best_r2 = ALPHA_GRID[len(ALPHA_GRID) // 2], -np.inf
        if len(va) >= 5:
            Atr, Btr, Ava = A[tr], B[tr], A[va]
            denom = ((B[va] - B[va].mean(0)) ** 2).sum()
            dual = Atr.shape[0] < Atr.shape[1]
            if dual:
                # One Gram matrix and one cross-product serve every candidate:
                # only the ridge term changes between them, so the expensive
                # products are not repeated five times. Predictions come from
                # A_va A_tr^T @ dual_weights, which never forms W at all -- and W
                # for the 405B is a 16384x16384 array, a gigabyte in float32.
                G = _gram(Atr)
                scale = _penalty_scale(Atr, G)
                K = Ava @ Atr.T
                for cand in ALPHA_GRID:
                    pred = K @ _dual_weights(G, Btr, cand, scale)
                    r2 = 1 - ((B[va] - pred) ** 2).sum() / denom
                    if r2 > best_r2:
                        best, best_r2 = cand, r2
            else:
                for cand in ALPHA_GRID:
                    pred = Ava @ _solve(Atr, Btr, cand)
                    r2 = 1 - ((B[va] - pred) ** 2).sum() / denom
                    if r2 > best_r2:
                        best, best_r2 = cand, r2
        alpha = best
    W = _solve(A, B, alpha)
    return LinearMap(W=W, b=np.zeros(Xb.shape[1]), mu_a=mu_a, mu_b=mu_b,
                     alpha=alpha, sd_a=sd_a, sd_b=sd_b)


def fit_map_both_ways(Xa, Xb, alpha: float | None = None):
    """(A->B, B->A). Experiment 2 needs both directions."""
    return fit_map(Xa, Xb, alpha), fit_map(Xb, Xa, alpha)


def linear_cka(X: np.ndarray, Y: np.ndarray, standardize: bool = False) -> float:
    """Linear CKA between two activation sets over the same items.

    Reported as *context* for a transfer number, never as a result on its own:
    the HELIX finding is that tokenizer compatibility and size gap drive
    alignment, so a weak cross-family transfer with low CKA is a map problem,
    not necessarily a probe problem.

    **Read it knowing what dominates it.** CKA weights directions by variance, and
    LLM activations carry a few massive dimensions that hold nearly all of it (on
    qwen3-8b the largest per-dimension std is 542x the median at 0.5 depth, where
    the variance has an effective rank of ~1). So CKA largely reports what *those*
    dimensions do. Measured on qwen3-8b vs qwen3-32b it falls 0.998 -> 0.786 from
    mid-depth to the final layer, which reads as representational divergence but
    is mostly the massive dimensions shrinking with depth: per-dimension
    standardized, the same pair is flat (0.983 / 0.976 / 0.978), and the
    transported probe loses nothing at the final layer either.

    `standardize=True` gives that scale-free variant. It is a *different measure*,
    not a corrected one — standardizing axes destroys the invariance to rotation
    that makes CKA a representational-similarity measure at all — so it belongs
    beside the raw number as a diagnostic, never as a replacement. The map
    (`fit_map`) standardizes unconditionally, which is a separate matter: there it
    is regression preconditioning and no invariance is claimed.
    """
    X, Y = np.asarray(X, dtype=float), np.asarray(Y, dtype=float)
    if standardize:
        X = X / (X.std(0) + 1e-6)
        Y = Y / (Y.std(0) + 1e-6)
    X = X - X.mean(0)
    Y = Y - Y.mean(0)
    hsic = np.linalg.norm(X.T @ Y, "fro") ** 2
    nx = np.linalg.norm(X.T @ X, "fro")
    ny = np.linalg.norm(Y.T @ Y, "fro")
    return float(hsic / (nx * ny + 1e-12))
