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

from typing import Sequence

import numpy as np

from .models import ItemFeatures
from .prompts import SUFFIX, claim_text, contrast_pair

# NOTE: nothing else may be imported at module level. NDIF serializes the trace
# body together with its enclosing scope and refuses anything outside its
# whitelist -- an `import os` here is enough to get the whole request rejected
# with "Module os is not whitelisted", even though the trace never calls it.

# Module path to the decoder layers, by architecture family. nnsight addresses
# submodules by attribute path, which differs between model families.
LAYER_PATHS = ("model.layers", "transformer.h", "gpt_neox.layers",
               "model.decoder.layers")

# Rows per verdict trace, for families whose lm head will not take the full
# batch. Anything absent uses `batch_size` unchanged.
LOGIT_BATCH_CAP = {"gemma2": 2, "gemma3": 4, "gemma": 2}


class NDIFModel:
    """A model executed remotely on NDIF, read through nnsight."""

    def __init__(self, key: str, hf_name: str | None = None,
                 remote: bool = True, batch_size: int = 8,
                 logit_batch_size: int | None = None):
        import os

        from nnsight import CONFIG, LanguageModel

        api_key = os.environ.get("NDIF_API_KEY")
        if remote and not api_key:
            raise RuntimeError(
                "NDIF_API_KEY is not set. Export it in the shell that runs the "
                "extraction; it is never read from a file here.")
        if api_key:
            CONFIG.set_default_api_key(api_key)

        from .config import MODELS

        spec = MODELS.get(key)
        self.key = key
        self.hf_name = hf_name or (spec.hf_name if spec else key)
        self.remote = remote
        self.batch_size = batch_size
        # The lm head is the memory hot spot, not the decoder stack: NDIF caps a
        # process at ~20GiB, and a model with a large vocabulary materialises
        # [batch, seq, vocab] there. Gemma-2 is the worst case -- 256k vocab and
        # `final_logit_softcapping`, which divides the whole logits tensor and
        # leaves several copies live at once -- so the verdict pass gets its own,
        # smaller batch while the hidden-state pass keeps the full one.
        # Largest row count the service has accepted so far. A remote OOM is a
        # property of the model, the claim length and whatever else shares the
        # GPU, none of which are knowable up front, so it is learned at runtime
        # and carried across batches instead of guessed per dataset.
        self._max_rows: int | None = None
        cap = LOGIT_BATCH_CAP.get(self._family(), batch_size)
        self.logit_batch_size = max(1, min(batch_size, logit_batch_size or cap))
        # Parameters are not downloaded for a remote model, so this is cheap.
        self.lm = LanguageModel(self.hf_name)
        self.tok = self.lm.tokenizer
        self.layers = self._layers()
        self.n_layers = len(self.layers)

    def _family(self) -> str:
        """Lowercased architecture family, for the per-family batch caps."""
        name = (self.hf_name or self.key).lower()
        for fam in sorted(LOGIT_BATCH_CAP, key=len, reverse=True):
            if fam in name:
                return fam
        return name

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

    def _trace(self, texts: Sequence[str], layer: int, *,
               hidden: bool = True, logits: bool = True):
        """Pooled hidden states for a batch, in one round trip.

        Two things keep this fast, and both matter more than batch size.

        The pooling happens *remotely*. A naive version saves the block's output,
        which is [batch, seq, hidden], and then keeps a mean and one row of it --
        paying for the whole sequence in bandwidth to use two vectors of it. The
        mean is a reduction nnsight can trace, so only [batch, hidden] comes back
        and the response shrinks by roughly the sequence length: measured on RTE,
        5.7MB per trace becomes a few hundred kilobytes.

        And the rows each item needs are sent together rather than one trace
        each, because they are independent. The contrast pair (the Yes half and
        the No half) goes in one call with `logits=False`, and the prefix the
        out-loud answer is read from goes in another with `hidden=False`. They
        are split because only the second needs the lm head, which is the memory
        hot spot for a large vocabulary: NDIF GPUs are shared and allow a process
        roughly 20 GiB, and materialising [batch, seq, vocab] there is what runs
        out of memory, not the decoder stack. So the hidden-state pass keeps the
        full batch while the verdict pass is capped separately.
        """
        import torch

        side = self.tok.padding_side
        self.tok.padding_side = "left"
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        enc = self.tok(list(texts), return_tensors="pt", padding=True,
                       truncation=True, max_length=512)
        self.tok.padding_side = side
        # Left padding moves every real token to the right, and a model does not
        # derive positions from the attention mask -- it counts from zero over the
        # padded tensor. Without this the pad tokens consume the low positions and
        # the claim is encoded at the wrong offsets, so the same claim gives
        # different activations depending on what else shared its batch. Counting
        # positions over unmasked tokens only makes a row independent of its
        # padding, which is what makes the batched result match a solo pass.
        enc["position_ids"] = (enc["attention_mask"].cumsum(-1) - 1).clamp(min=0)
        # Only position -1 of the logits is ever read, but the lm head otherwise
        # runs over the whole sequence: [batch, seq, vocab]. With a 256k-vocab
        # model that is 2+ GiB for one boolq batch, which is what actually blew
        # the 20 GiB cap -- and it happened even on the pass that saves no logits,
        # because not saving a tensor does not stop the model computing it.
        # Asking for one position makes the head cost independent of length.
        enc["logits_to_keep"] = 1
        m = enc["attention_mask"].unsqueeze(-1).float()   # [b, s, 1]

        lm, block = self.lm, self.layers[layer]
        mean = last = final = None
        with lm.trace(enc, remote=self.remote):
            if hidden:
                h = block.output
                h = h[0] if isinstance(h, tuple) else h
                # the mask is built locally and the model runs on the remote GPU,
                # so it has to be moved before it can multiply the hidden states
                mm = m.to(h.device).to(h.dtype)
                mean = ((h * mm).sum(1) / mm.sum(1)).save()   # [b, hidden]
                last = h[:, -1, :].save()                     # left padding: -1
            if logits:
                final = lm.output.logits[:, -1, :].save()      # [b, vocab]
        return (None if mean is None else mean.detach().float().cpu().numpy(),
                None if last is None else last.detach().float().cpu().numpy(),
                None if final is None else final.detach().float().cpu())

    @staticmethod
    def _is_oom(e: Exception) -> bool:
        """Whether a remote failure was memory, as opposed to a real error.

        NDIF reports an OOM as a generic NNsightException wrapping the remote
        traceback, so there is no exception type to catch -- the text is the only
        signal available.
        """
        t = f"{type(e).__name__}: {e}"
        return ("OutOfMemoryError" in t or "out of memory" in t.lower()
                or "CUDA error" in t)

    def _trace_adaptive(self, texts, layer: int, *, hidden=True, logits=True):
        """`_trace`, but split and retried when the service runs out of memory.

        A batch that OOMs is halved and each half sent separately, recursively,
        down to a single row. The working size is remembered so the rest of the
        dataset goes straight there rather than failing once per batch. Without
        this an OOM anywhere loses the whole model/dataset cell: the failures
        were all on boolq and imdb, whose claims are several times longer than
        truthfulqa's, so the row count that fits on one dataset is too large on
        another even for the same model.
        """
        import torch

        texts = list(texts)
        limit = self._max_rows or len(texts)
        out = []
        i = 0
        while i < len(texts):
            take = min(limit, len(texts) - i)
            while True:
                try:
                    out.append(self._trace(texts[i:i + take], layer,
                                           hidden=hidden, logits=logits))
                    break
                except Exception as e:
                    if not self._is_oom(e) or take == 1:
                        raise
                    take = max(1, take // 2)
                    limit = take
                    self._max_rows = take
                    print(f"    remote OOM -> retrying at {take} rows/trace",
                          flush=True)
            i += take

        def cat(parts):
            parts = [x for x in parts if x is not None]
            if not parts:
                return None
            if isinstance(parts[0], np.ndarray):
                return np.concatenate(parts, 0)
            return torch.cat(parts, 0)

        return tuple(cat([o[j] for o in out]) for j in range(3))

    def _p_yes(self, final_logits, yes_id: int, no_id: int) -> np.ndarray:
        """P(yes) against P(no) at the position after the question, as locally."""
        import torch

        pair = torch.stack([final_logits[:, yes_id], final_logits[:, no_id]], -1)
        return torch.softmax(pair.float(), -1)[:, 0].numpy()

    def _block_index(self, layer: int | None) -> int:
        """Map the caller's layer number onto this model's decoder blocks.

        Two conventions meet here. The rest of the codebase indexes layers the way
        `hidden_states` does -- entry 0 is the embedding output, so the final layer
        of a 32-layer model is 32 -- while `self.layers` is the ModuleList of
        decoder blocks, indexed 0..31. Passing the first straight into the second
        is an off-by-one that raises IndexError on every call.
        """
        if layer is None:
            return self.n_layers - 1
        return max(0, min(self.n_layers - 1, layer - 1))

    def features_batch(self, claims: Sequence[str], layer: int | None = None):
        """One `ItemFeatures` per claim. Batched because each trace is a round trip."""
        out_layer = self.n_layers if layer is None else layer
        layer = self._block_index(layer)
        yes_id = self.tok.encode(" Yes", add_special_tokens=False)[-1]
        no_id = self.tok.encode(" No", add_special_tokens=False)[-1]

        # Batch claims of similar length together. In dataset order a 31-token
        # claim shares a batch with a 162-token one and every row is padded to
        # the longest -- on RTE that is a 5x wider remote tensor than the
        # content needs, paid for in both bandwidth and remote memory. Sorting
        # by length first makes each batch nearly uniform; the results are
        # unsorted back to the caller's order at the end, so nothing downstream
        # sees the reordering.
        order = sorted(range(len(claims)),
                       key=lambda j: len(self.tok.encode(str(claims[j]))))
        out: list = [None] * len(claims)
        for i in range(0, len(order), self.batch_size):
            idx = order[i:i + self.batch_size]
            chunk = [claims[j] for j in idx]
            pos_t, neg_t = zip(*(contrast_pair(c) for c in chunk))
            # P(yes) is read at the question mark, before either verdict token,
            # so it is the model's own answer rather than a reading of the prompt
            prefix = [f"{claim_text(c)}{SUFFIX}" for c in chunk]
            k = len(chunk)
            mean, last, _ = self._trace_adaptive(list(pos_t) + list(neg_t),
                                                 layer, logits=False)
            pm, nm = mean[:k], mean[k:2 * k]
            pl, nl = last[:k], last[k:2 * k]
            py = np.concatenate([
                self._p_yes(self._trace_adaptive(
                    prefix[j:j + self.logit_batch_size],
                    layer, hidden=False)[2], yes_id, no_id)
                for j in range(0, k, self.logit_batch_size)])
            for j, dest in enumerate(idx):
                # keyed by the caller's layer number so the cache matches local
                out[dest] = ItemFeatures(
                    pos={out_layer: pm[j]}, neg={out_layer: nm[j]},
                    p_yes=float(py[j]),
                    pos_last={out_layer: pl[j]}, neg_last={out_layer: nl[j]})
        assert all(o is not None for o in out)
        return out

    def features(self, claim: str, layers: Sequence[int]) -> ItemFeatures:
        return self.features_batch([claim], layers[-1] if layers else None)[0]
