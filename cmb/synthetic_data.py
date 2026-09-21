"""Synthetic item manifests, so the dataset loaders need no network.

Used only with `--synthetic`. Claim strings are placeholders; what matters is
that the ids, labels and group structure match the shape of the real loaders,
because the synthetic backend derives its latents from the id.
"""

from __future__ import annotations

import numpy as np

from .data import Item


def synthetic_items(dataset: str, n: int = 400, seed: int = 0) -> list[Item]:
    rng = np.random.default_rng(abs(hash(dataset)) % 2**32 + seed)
    items = []
    for i in range(n // 2):
        g = f"{dataset}/syn/{i}"
        items.append(Item(f"{g}/t", f"[{dataset}] claim {i} (true)", 1, dataset, g))
        items.append(Item(f"{g}/f", f"[{dataset}] claim {i} (false)", 0, dataset, g))
    idx = rng.permutation(len(items))
    return [items[i] for i in idx][:n]


def install() -> None:
    """Point every loader at the synthetic generator (idempotent)."""
    from . import data

    for name in data.LOADERS:
        data.LOADERS[name] = (lambda ds: lambda n: synthetic_items(ds, n or 400))(name)
