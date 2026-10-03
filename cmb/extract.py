"""Run a (model, dataset) activation pass, cached.

This is the only place that touches a GPU. Every experiment script goes through
`get_activations`, so a second run of anything is free.
"""

from __future__ import annotations

import sys

import numpy as np

from .cache import ActivationSet, cache_path, load, save
from .config import DEFAULT_LAYER_FRAC, LAYER_SWEEP, layer_index, resolve_layer
from .data import Item, load_items
from .models import load_backend


def _layers_for(backend, fracs, extra=()) -> list[int]:
    """Concrete hidden-state indices to extract for one model.

    `fracs` are depth fractions (the standing sweep); `extra` are per-model
    layer specs from the CLI, resolved against this model's own depth. All of
    them come out of the same forward pass, so asking for more is nearly free.
    """
    n = backend.n_layers
    return sorted({layer_index(n, f) for f in fracs}
                  | {resolve_layer(e, n) for e in extra})


def extract(model: str, dataset: str, n: int | None = None,
            synthetic: bool = False, fracs=None, backend=None,
            progress: bool = True, extra_layers=()) -> ActivationSet:
    """Forward-pass a dataset through a model and collect contrast activations."""
    if synthetic:
        from .synthetic_data import install
        install()
    items = load_items(dataset, n)
    backend = backend or load_backend(model, synthetic=synthetic)
    fracs = tuple(fracs) if fracs else tuple(sorted(set(LAYER_SWEEP) |
                                                   {DEFAULT_LAYER_FRAC}))
    layers = _layers_for(backend, fracs, extra_layers)
    if hasattr(backend, "features_batch"):
        # Remote traces pay for every saved tensor in bandwidth, so the sweep
        # layers are not free there the way they are from one local forward pass.
        layers = layers[-1:]

    pos = {l: [] for l in layers}
    neg = {l: [] for l in layers}
    # The alternate pooling (last token), when the backend provides it. One
    # forward pass gives both, so there is never a reason not to keep it.
    pos_alt = {l: [] for l in layers}
    neg_alt = {l: [] for l in layers}
    p_yes, labels = [], []

    def feats():
        """Yield one ItemFeatures per item, batching where the backend can.

        A remote backend pays a network round trip per trace, so item-at-a-time
        would dominate everything else; `features_batch` amortizes it. Local
        backends keep the per-item path, which is simpler and already fast.
        """
        if hasattr(backend, "features_batch"):
            step = getattr(backend, "batch_size", 16)
            for k in range(0, len(items), step):
                chunk = items[k:k + step]
                if progress:
                    print(f"  [{model}/{dataset}] {k}/{len(items)}",
                          file=sys.stderr)
                yield from backend.features_batch([c.claim for c in chunk],
                                                  layers[-1])
            return
        for k, it in enumerate(items):
            if progress and k % 50 == 0:
                print(f"  [{model}/{dataset}] {k}/{len(items)}", file=sys.stderr)
            yield (backend.features_for_item(it, layers)
                   if hasattr(backend, "features_for_item")
                   else backend.features(it.claim, layers))

    for it, f in zip(items, feats()):
        for l in layers:
            pos[l].append(f.pos[l])
            neg[l].append(f.neg[l])
            if getattr(f, "pos_last", None):
                pos_alt[l].append(f.pos_last[l])
                neg_alt[l].append(f.neg_last[l])
        p_yes.append(f.p_yes)
        labels.append(it.label)

    return ActivationSet(
        model=model, dataset=dataset, item_ids=[it.item_id for it in items],
        labels=np.array(labels), p_yes=np.array(p_yes, dtype=float),
        pos={l: np.stack(v) for l, v in pos.items()},
        neg={l: np.stack(v) for l, v in neg.items()},
        n_layers=backend.n_layers,
        pos_alt={l: np.stack(v) for l, v in pos_alt.items() if v},
        neg_alt={l: np.stack(v) for l, v in neg_alt.items() if v},
        pooling="mean")


def get_activations(model: str, dataset: str, n: int | None = None,
                    synthetic: bool = False, refresh: bool = False,
                    extra_layers=(), **kw) -> ActivationSet:
    """Cached `extract`. Synthetic runs get their own cache namespace.

    A cache hit that lacks a requested layer triggers re-extraction: silently
    probing the nearest cached layer instead would make a `--layer-a` flag mean
    something other than what it says.
    """
    key = f"synthetic-{model}" if synthetic else model
    path = cache_path(key, dataset, n)
    if synthetic:
        from .synthetic_data import install
        install()
    if path.exists() and not refresh:
        acts = load(path)
        need = {resolve_layer(e, acts.n_layers) for e in extra_layers}
        if need <= set(acts.layers):
            wanted = [it.item_id for it in load_items(dataset, n)]
            return acts.reorder_to(wanted)
        print(f"  [{model}/{dataset}] cache lacks layer(s) "
              f"{sorted(need - set(acts.layers))}; re-extracting")
    acts = extract(model, dataset, n=n, synthetic=synthetic,
                   extra_layers=extra_layers, **kw)
    acts.model = key
    save(acts, path)
    return acts


def paired(model_a: str, model_b: str, dataset: str, n: int | None = None,
           synthetic: bool = False, layer_a=(), layer_b=(), **kw):
    """Two models' activations over the *same* items, in the same order.

    Paired order is a precondition for fitting the alignment map at all. The
    two models take separate layer specs, since the probe site is a per-model
    choice — models of different depth have no shared layer numbering.
    """
    a = get_activations(model_a, dataset, n, synthetic=synthetic,
                        extra_layers=layer_a, **kw)
    b = get_activations(model_b, dataset, n, synthetic=synthetic,
                        extra_layers=layer_b, **kw)
    ids = [i for i in a.item_ids if i in set(b.item_ids)]
    return a.reorder_to(ids), b.reorder_to(ids)
