"""Run a (model, dataset) activation pass, cached.

This is the only place that touches a GPU. Every experiment script goes through
`get_activations`, so a second run of anything is free.
"""

from __future__ import annotations

import sys

import numpy as np

from .cache import ActivationSet, cache_path, load, save
from .config import DEFAULT_LAYER_FRAC, LAYER_SWEEP, layer_index
from .data import Item, load_items
from .models import load_backend


def _layers_for(backend, fracs) -> list[int]:
    n = backend.n_layers
    return sorted({layer_index(n, f) for f in fracs})


def extract(model: str, dataset: str, n: int | None = None,
            synthetic: bool = False, fracs=None, backend=None,
            progress: bool = True) -> ActivationSet:
    """Forward-pass a dataset through a model and collect contrast activations."""
    if synthetic:
        from .synthetic_data import install
        install()
    items = load_items(dataset, n)
    backend = backend or load_backend(model, synthetic=synthetic)
    fracs = tuple(fracs) if fracs else tuple(sorted(set(LAYER_SWEEP) |
                                                   {DEFAULT_LAYER_FRAC}))
    layers = _layers_for(backend, fracs)

    pos = {l: [] for l in layers}
    neg = {l: [] for l in layers}
    p_yes, labels = [], []
    for i, it in enumerate(items):
        if hasattr(backend, "features_for_item"):
            f = backend.features_for_item(it, layers)
        else:
            f = backend.features(it.claim, layers)
        for l in layers:
            pos[l].append(f.pos[l])
            neg[l].append(f.neg[l])
        p_yes.append(f.p_yes)
        labels.append(it.label)
        if progress and i % 50 == 0:
            print(f"  [{model}/{dataset}] {i}/{len(items)}", file=sys.stderr)

    return ActivationSet(
        model=model, dataset=dataset, item_ids=[it.item_id for it in items],
        labels=np.array(labels), p_yes=np.array(p_yes, dtype=float),
        pos={l: np.stack(v) for l, v in pos.items()},
        neg={l: np.stack(v) for l, v in neg.items()},
        n_layers=backend.n_layers)


def get_activations(model: str, dataset: str, n: int | None = None,
                    synthetic: bool = False, refresh: bool = False,
                    **kw) -> ActivationSet:
    """Cached `extract`. Synthetic runs get their own cache namespace."""
    key = f"synthetic-{model}" if synthetic else model
    path = cache_path(key, dataset, n)
    if synthetic:
        from .synthetic_data import install
        install()
    if path.exists() and not refresh:
        acts = load(path)
        wanted = [it.item_id for it in load_items(dataset, n)]
        return acts.reorder_to(wanted)
    acts = extract(model, dataset, n=n, synthetic=synthetic, **kw)
    acts.model = key
    save(acts, path)
    return acts


def paired(model_a: str, model_b: str, dataset: str, n: int | None = None,
           synthetic: bool = False, **kw):
    """Two models' activations over the *same* items, in the same order.

    Paired order is a precondition for fitting the alignment map at all.
    """
    a = get_activations(model_a, dataset, n, synthetic=synthetic, **kw)
    b = get_activations(model_b, dataset, n, synthetic=synthetic, **kw)
    ids = [i for i in a.item_ids if i in set(b.item_ids)]
    return a.reorder_to(ids), b.reorder_to(ids)
