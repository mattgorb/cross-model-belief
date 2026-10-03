"""NDIF/NNsight backend: activations from models too large to host locally.

Why bother. Parameter count predicts error correlation at $-0.24$ across the
local models, which spans 1.7B--32B -- a range too narrow to say anything about
scale. NDIF hosts Llama-3.1 at 8B, 70B and 405B, and reaching 70B extends the
range by an order of magnitude, which is what the scale claim needs.

The backend implements the same contract as `HFModel`, so `cmb/extract.py` and
every experiment downstream are unchanged: given a claim, return the mean-pooled
and last-token hidden states of both halves of the contrast pair, and the model's
out-loud P(yes).

Two differences from the local path that matter. Remote execution is a network
round trip per trace, so items are batched rather than sent one at a time, and a
batch failure costs the batch. And only the final layer is requested: the sweep
layers are free locally because they come out of one forward pass, but over the
wire each saved tensor is paid for in bandwidth.

Requires `pip install nnsight` and an NDIF API key in `NDIF_API_KEY`; the key is
read from the environment and never written to disk.
"""

from __future__ import annotations

import os
from typing import Sequence

import numpy as np

from .models import ItemFeatures
from .prompts import SUFFIX, claim_text, contrast_pair

# Module path to the decoder layers, by architecture family. nnsight addresses
# submodules by attribute path, which differs between model families.
LAYER_PATHS = ("model.layers", "transformer.h", "gpt_neox.layers",
               "model.decoder.layers")


class NDIFModel:
    """A model executed remotely on NDIF, read through nnsight."""

    def __init__(self, key: str, hf_name: str | None = None,
                 remote: bool = True, batch_size: int = 16):
        from nnsight import CONFIG, LanguageModel

        api_key = os.environ.get("NDIF_API_KEY")
        if remote and not api_key:
            raise RuntimeError(
                "NDIF_API_KEY is not set. Export it in the shell that runs the "
                "extraction; it is never read from a file here.")
        if api_key:
            CONFIG.set_default_api_key(api_key)

        self.key = key
        self.remote = remote
        self.batch_size = batch_size
        # Parameters are not downloaded for a remote model, so this is cheap.
        self.lm = LanguageModel(hf_name or key)
        self.tok = self.lm.tokenizer
        self.layers = self._layers()
        self.n_layers = len(self.layers)

    def _layers(self):
        for path in LAYER_PATHS:
            obj = self.lm
            try:
                for part in path.split("."):
                    obj = getattr(obj, part)
                len(obj)
                return obj
            except (AttributeError, TypeError):
                continue
        raise RuntimeError(
            f"cannot find the decoder layers of {self.key}; tried {LAYER_PATHS}. "
            f"Add its path to LAYER_PATHS.")

    # -- the one remote call ------------------------------------------------

    def _trace_batch(self, texts: Sequence[str], layer: int):
        """Mean-pooled and last-token hidden states for a batch, one round trip."""
        import torch

        enc = self.tok(list(texts), return_tensors="pt", padding=True,
                       truncation=True, max_length=512)
        with self.lm.trace(enc, remote=self.remote):
            h = self.layers[layer].output[0].save()
            logits = self.lm.output.logits.save()
        h = h.detach().float().cpu().numpy()
        mask = enc["attention_mask"].numpy().astype(bool)
        mean = np.stack([h[i][mask[i]].mean(0) for i in range(len(texts))])
        last = np.stack([h[i][mask[i]][-1] for i in range(len(texts))])
        lg = logits.detach().float().cpu()
        idx = mask.sum(1) - 1                       # final real token per row
        final = torch.stack([lg[i, idx[i]] for i in range(len(texts))])
        return mean, last, final

    def _p_yes(self, final_logits, yes_id: int, no_id: int) -> np.ndarray:
        import torch

        pair = torch.stack([final_logits[:, yes_id], final_logits[:, no_id]], -1)
        return torch.softmax(pair.float(), -1)[:, 0].numpy()

    def features_batch(self, claims: Sequence[str], layer: int | None = None):
        """One `ItemFeatures` per claim. Batched because each trace is a round trip."""
        layer = self.n_layers - 1 if layer is None else layer
        yes_id = self.tok.encode(" Yes", add_special_tokens=False)[-1]
        no_id = self.tok.encode(" No", add_special_tokens=False)[-1]

        out = []
        for i in range(0, len(claims), self.batch_size):
            chunk = list(claims[i:i + self.batch_size])
            pos_t, neg_t = zip(*(contrast_pair(c) for c in chunk))
            pm, pl, _ = self._trace_batch(pos_t, layer)
            nm, nl, _ = self._trace_batch(neg_t, layer)
            # P(yes) is read at the question mark, before either verdict token,
            # so it is the model's own answer rather than a reading of the prompt
            prefix = [f"{claim_text(c)}{SUFFIX}" for c in chunk]
            _, _, flog = self._trace_batch(prefix, layer)
            py = self._p_yes(flog, yes_id, no_id)
            for j in range(len(chunk)):
                out.append(ItemFeatures(
                    pos={layer: pm[j]}, neg={layer: nm[j]}, p_yes=float(py[j]),
                    pos_last={layer: pl[j]}, neg_last={layer: nl[j]}))
        return out

    def features(self, claim: str, layers: Sequence[int]) -> ItemFeatures:
        return self.features_batch([claim], layers[-1] if layers else None)[0]
