"""Synthetic activation backend — a testbed with a known answer.

No GPU, no downloads. It generates activations from a small latent model in
which the quantities the project measures actually exist, so the whole pipeline
can be run end to end and checked against a known ground truth:

  * a **truth** latent (the thing a probe should find),
  * a **prominence** latent shared across models (a popular-falsehood confound —
    this is what manufactures Row 2: both models read a familiar falsehood as
    true, and they do it *for the same reason*),
  * a **content** subspace shared across models (so a linear A->B map exists),
  * a **private** subspace per model (so transport is lossy, as in reality),
  * output P(Yes) driven by prominence more than truth, so the logprob baseline
    is real but beatable.

Because prominence is a direction in the shared subspace, Row 2 here *is*
separable and *does* generalize — the synthetic world is a "case 4" world under
the DESIGN.md §7 decision tree. That makes it a positive control: if an
experiment script cannot recover the planted structure, the bug is in the code,
not in the science. Off-diagonal generalization is deliberately imperfect
(each dataset gets its own nuisance direction) so Experiment 4 produces a
matrix with structure rather than a wall of 1.0.
"""

from __future__ import annotations

import hashlib
from typing import Sequence

import numpy as np

from .models import ItemFeatures

SHARED_DIM = 24          # content + truth + prominence live here
PRIVATE_DIM = 16         # model-specific structure the map cannot carry

_DIMS = {"qwen-1.5b": 96, "qwen-7b": 128, "qwen-32b": 160, "llama-8b": 112}
_FAMILY_SEED = {"qwen": 11, "llama": 29}


def _hash_unit(*parts: str) -> float:
    """Deterministic uniform(0,1) from strings — stands in for 'how familiar'."""
    h = hashlib.sha256("|".join(parts).encode()).digest()
    return int.from_bytes(h[:8], "big") / 2**64


def _rng(*parts: str) -> np.random.Generator:
    h = hashlib.sha256("|".join(parts).encode()).digest()
    return np.random.default_rng(int.from_bytes(h[:8], "big"))


class SyntheticModel:
    """Backend with the same interface as `HFModel`, driven by latents.

    `features()` needs the ground-truth label and the item id, which a claim
    string does not carry, so callers pass an `Item` through `features_for_item`.
    The plain `features(claim, layers)` path derives both from the claim text,
    which is enough for ad-hoc use but is not what the experiments call.
    """

    def __init__(self, key: str, n_layers: int = 32, noise: float = 0.25,
                 flip_noise: float = 0.08):
        self.key = key
        self.n_layers = n_layers
        self.noise = noise
        self.flip_noise = flip_noise
        d = _DIMS.get(key, 128)
        family = "qwen" if key.startswith("qwen") else "llama"
        r = np.random.default_rng(_FAMILY_SEED.get(family, 7) + len(key))
        # Shared subspace basis: same latents, model-specific embedding.
        self.W_shared = r.normal(size=(SHARED_DIM, d)) / np.sqrt(SHARED_DIM)
        self.W_private = r.normal(size=(PRIVATE_DIM, d)) / np.sqrt(PRIVATE_DIM)
        self.bias = r.normal(size=d) * 0.1
        self.dim = d

    # -- latents ------------------------------------------------------------

    def _latents(self, item_id: str, label: int, dataset: str):
        prominence = _hash_unit("prom", item_id)          # shared across models
        content = _rng("content", item_id).normal(size=SHARED_DIM - 3) * 0.5
        nuisance = _hash_unit("nuisance", dataset, item_id)
        return prominence, content, nuisance

    def internal_belief(self, item_id: str, label: int, dataset: str) -> int:
        """What this model privately thinks — label, corrupted by prominence.

        A false claim with high prominence reads as true to *both* models (the
        shared term), plus a small model-private flip. That asymmetry is the
        whole point: Row 2 is correlated error, not independent error.
        """
        prominence, _, _ = self._latents(item_id, label, dataset)
        shared_flip = (label == 0) and (prominence > 0.72)
        private = _hash_unit("flip", self.key, item_id) < self.flip_noise
        belief = label
        if shared_flip:
            belief = 1
        if private:
            belief = 1 - belief
        return int(belief)

    # -- backend interface --------------------------------------------------

    def features_for_item(self, item, layers: Sequence[int]) -> ItemFeatures:
        label, item_id, dataset = item.label, item.item_id, item.dataset
        prominence, content, nuisance = self._latents(item_id, label, dataset)
        belief = self.internal_belief(item_id, label, dataset)

        rng = _rng("noise", self.key, item_id)
        priv = rng.normal(size=PRIVATE_DIM)

        def embed(assert_true: bool) -> np.ndarray:
            # z = [belief-signed truth axis, prominence, dataset nuisance, content]
            sign = 1.0 if assert_true else -1.0
            z = np.empty(SHARED_DIM)
            z[0] = sign * (1.0 if belief else -1.0)     # what CCS should find
            z[1] = prominence - 0.5                      # the confound axis
            z[2] = nuisance - 0.5                        # dataset-specific
            z[3:] = content
            h = z @ self.W_shared + priv @ self.W_private + self.bias
            return h + rng.normal(size=self.dim) * self.noise

        # Deeper layers carry the truth axis more strongly; a layer sweep should
        # see a curve rather than a flat line.
        out_pos, out_neg = {}, {}
        for li in layers:
            scale = 0.5 + 1.5 * (li / max(self.n_layers, 1))
            g = _rng("layer", self.key, item_id, str(li))
            out_pos[li] = embed(True) * scale + g.normal(size=self.dim) * 0.05
            out_neg[li] = embed(False) * scale + g.normal(size=self.dim) * 0.05

        # Output-space belief: mostly prominence, only weakly truth, so the
        # logprob baseline is informative but should lose to the probe.
        logit = 1.0 * (prominence - 0.45) + 1.8 * (belief - 0.5)
        logit += _rng("pyes", self.key, item_id).normal() * 0.9
        p_yes = float(1 / (1 + np.exp(-logit)))
        return ItemFeatures(pos=out_pos, neg=out_neg, p_yes=p_yes)

    def features(self, claim: str, layers: Sequence[int]) -> ItemFeatures:
        from .data import Item
        label = int(_hash_unit("label", claim) > 0.5)
        return self.features_for_item(
            Item(item_id=claim[:64], claim=claim, label=label,
                 dataset="adhoc", group=claim[:64]), layers)

    def default_layer(self, frac: float = 0.6) -> int:
        return max(1, min(self.n_layers, int(round(self.n_layers * frac))))
