"""Activation cache, keyed (model, dataset, item, layer, pos|neg).

DESIGN.md §9: the expensive 32B pass runs once and is never recomputed. One
`.npz` per (model, dataset, n) holds every swept layer, since all layers come
out of a single forward pass for free. Item ids are stored alongside and
verified on load — a cache file that does not match the requested manifest is
rejected rather than silently misaligned.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .config import CACHE_DIR


@dataclass
class ActivationSet:
    """Cached activations for one (model, dataset) pass."""
    model: str
    dataset: str
    item_ids: list[str]
    labels: np.ndarray                 # [n] ground truth
    p_yes: np.ndarray                  # [n] output-space belief
    pos: dict[int, np.ndarray]         # layer -> [n, d], the ACTIVE pooling
    neg: dict[int, np.ndarray]
    n_layers: int = 0                  # model depth, so a layer fraction is
                                       # resolvable from the cache alone
    # The other pooling, carried alongside. `pos`/`neg` are what every consumer
    # reads, so switching pooling is `with_pooling()` rather than a change at
    # each use site. Empty for caches written before both were stored.
    pos_alt: dict[int, np.ndarray] = field(default_factory=dict)
    neg_alt: dict[int, np.ndarray] = field(default_factory=dict)
    pooling: str = "mean"

    @property
    def layers(self) -> list[int]:
        return sorted(self.pos)

    def diff(self, layer: int) -> np.ndarray:
        """pos - neg. The contrast direction is what CCS effectively reads, and
        it cancels claim-content structure common to both halves."""
        return self.pos[layer] - self.neg[layer]

    def subset(self, mask) -> "ActivationSet":
        mask = np.asarray(mask)
        idx = np.where(mask)[0] if mask.dtype == bool else mask
        return ActivationSet(
            self.model, self.dataset, [self.item_ids[i] for i in idx],
            self.labels[idx], self.p_yes[idx],
            {l: v[idx] for l, v in self.pos.items()},
            {l: v[idx] for l, v in self.neg.items()}, self.n_layers,
            {l: v[idx] for l, v in self.pos_alt.items()},
            {l: v[idx] for l, v in self.neg_alt.items()}, self.pooling)

    def with_pooling(self, pooling: str) -> "ActivationSet":
        """Swap which pooling `pos`/`neg` expose. Same cache, no refit upstream.

        `mean` is what the alignment work assumes; `last` (the Yes/No token) is
        what the probing literature reads. Both were written in one pass.
        """
        if pooling == self.pooling:
            return self
        if not self.pos_alt:
            raise RuntimeError(
                f"{self.model}/{self.dataset} was cached with only "
                f"'{self.pooling}' pooling — re-extract with --refresh to get "
                f"both (one forward pass, no extra GPU time).")
        return ActivationSet(
            self.model, self.dataset, self.item_ids, self.labels, self.p_yes,
            self.pos_alt, self.neg_alt, self.n_layers,
            self.pos, self.neg, pooling)

    def reorder_to(self, item_ids: list[str]) -> "ActivationSet":
        """Align this set to a given item order (needed to pair two models)."""
        pos_of = {k: i for i, k in enumerate(self.item_ids)}
        missing = [k for k in item_ids if k not in pos_of]
        if missing:
            raise KeyError(f"{self.model}/{self.dataset} is missing "
                           f"{len(missing)} items, e.g. {missing[:3]}")
        return self.subset(np.array([pos_of[k] for k in item_ids]))


def cache_path(model: str, dataset: str, n: int | None) -> Path:
    """Cache file for one (model, dataset, n).

    A HuggingFace repo id contains a slash, which would silently nest the cache
    one directory deeper and break every listing; flatten it instead.
    """
    safe = model.replace("/", "--")
    return CACHE_DIR / safe / f"{dataset}_n{n if n is not None else 'all'}.npz"


def save(acts: ActivationSet, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"item_ids": np.array(acts.item_ids, dtype=object),
               "labels": acts.labels, "p_yes": acts.p_yes,
               "model": acts.model, "dataset": acts.dataset,
               "n_layers": acts.n_layers}
    for l, v in acts.pos.items():
        payload[f"pos_l{l}"] = v
    for l, v in acts.neg.items():
        payload[f"neg_l{l}"] = v
    payload["pooling"] = acts.pooling
    for l, v in acts.pos_alt.items():
        payload[f"posalt_l{l}"] = v
    for l, v in acts.neg_alt.items():
        payload[f"negalt_l{l}"] = v
    # np.savez appends ".npz" unless the name already ends with it, so the
    # temp file is named to survive that.
    tmp = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(tmp, **payload)
    tmp.replace(path)


def load(path: Path) -> ActivationSet:
    z = np.load(path, allow_pickle=True)
    pos = {int(k[5:]): z[k] for k in z.files if k.startswith("pos_l")}
    neg = {int(k[5:]): z[k] for k in z.files if k.startswith("neg_l")}
    pos_alt = {int(k[8:]): z[k] for k in z.files if k.startswith("posalt_l")}
    neg_alt = {int(k[8:]): z[k] for k in z.files if k.startswith("negalt_l")}
    return ActivationSet(model=str(z["model"]), dataset=str(z["dataset"]),
                         item_ids=[str(s) for s in z["item_ids"]],
                         labels=z["labels"], p_yes=z["p_yes"],
                         pos=pos, neg=neg,
                         n_layers=int(z["n_layers"]) if "n_layers" in z.files
                         else max(pos) if pos else 0,
                         pos_alt=pos_alt, neg_alt=neg_alt,
                         pooling=str(z["pooling"]) if "pooling" in z.files
                         else "mean")
