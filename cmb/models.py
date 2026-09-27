"""Model backends.

A backend answers one question per item: given a claim, what are the mean-pooled
hidden states of its positive and negative contrast halves at each requested
layer, and what does the model say out loud (P(Yes)) about the claim?

The out-loud number is not decoration — it is the Experiment 0 baseline and the
Gate A confident-slice selector. Activations only earn the framing if they beat
it (DESIGN.md §4).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, Sequence

import numpy as np

from .config import DEFAULT_LAYER_FRAC, MAX_LENGTH, MODELS, layer_index
from .prompts import SUFFIX, claim_text, contrast_pair
from .tokalign import TokenStates


@dataclass
class ItemFeatures:
    """Per-item output of a backend.

    Both poolings come out of the *same* forward pass, so caching both costs
    disk and no GPU time. That matters because the choice is not recoverable
    later: mean pooling is what the linear-alignment / embedding-API setting
    assumes, while the probing literature (Marks & Tegmark) reads the final
    token. Extracting once and deciding afterwards is the only cheap order.
    """
    pos: dict[int, np.ndarray]        # layer -> mean-pooled hidden state
    neg: dict[int, np.ndarray]
    p_yes: float                      # output-space P(claim is true), the baseline
    pos_last: dict[int, np.ndarray] = field(default_factory=dict)   # final token
    neg_last: dict[int, np.ndarray] = field(default_factory=dict)


class Backend(Protocol):
    key: str
    n_layers: int

    def features(self, claim: str, layers: Sequence[int]) -> ItemFeatures: ...

    def token_states(self, text: str, layers: Sequence[int]) -> TokenStates: ...


class HFModel:
    """HuggingFace causal LM, fp16, hidden states out.

    fp16 is not a performance choice: DESIGN.md §4 forbids 4-bit for extraction
    because quantization perturbs exactly what the probe reads.
    """

    def __init__(self, key: str, device_map: str = "auto"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        spec = MODELS[key] if key in MODELS else None
        hf_name = spec.hf_name if spec else key
        self.key = key
        self.spec = spec
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(hf_name, use_fast=True)
        if not self.tok.is_fast:
            raise RuntimeError(
                f"{hf_name} has no fast tokenizer, so it returns no character "
                "offsets and cannot be span-aligned against another model. "
                "Fall back to --map-pairs item (pooled) for this model.")

        # The reasoning-era releases (Qwen 3.5+ and Gemma 4) are native VLMs and
        # expose ...ForConditionalGeneration, which AutoModelForCausalLM refuses.
        # Text-only input through the wrapper still runs the language stack and
        # still returns its hidden states, which is all the probe needs.
        loaders = [AutoModelForCausalLM]
        if spec is not None and spec.multimodal:
            loaders = []
        try:
            from transformers import AutoModelForImageTextToText
            loaders.append(AutoModelForImageTextToText)
        except ImportError:                      # older transformers
            pass
        from transformers import AutoModel
        loaders.append(AutoModel)

        errs = []
        self.lm = None
        for loader in loaders:
            try:
                # `output_hidden_states` belongs on the forward call, not here:
                # the ...ForConditionalGeneration wrappers route unknown kwargs
                # into the generation config and ignore it, which left
                # `out.hidden_states` as None. `dtype` replaced `torch_dtype`.
                try:
                    self.lm = loader.from_pretrained(
                        hf_name, dtype=torch.float16, device_map=device_map)
                except TypeError:                # transformers < 4.56
                    self.lm = loader.from_pretrained(
                        hf_name, torch_dtype=torch.float16, device_map=device_map)
                break
            except (ValueError, KeyError, OSError) as e:
                errs.append(f"{loader.__name__}: {type(e).__name__}: {e}")
        if self.lm is None:
            raise RuntimeError(
                f"could not load {hf_name} with any Auto class. A very new "
                f"architecture usually means transformers is too old — check "
                f"requirements.txt. Tried:\n  " + "\n  ".join(errs))
        self.lm.eval()
        self.n_layers = self._depth(self.lm.config)
        if spec is not None and spec.n_layers != self.n_layers:
            # The registry documents depth so the docs and the layer flags can be
            # read without downloading weights; the loaded config wins.
            print(f"[{key}] registry says {spec.n_layers} layers, config says "
                  f"{self.n_layers} — using the config")

    @staticmethod
    def _depth(config) -> int:
        """Hidden-layer count, reaching into `text_config` for VLM wrappers."""
        for cfg in (getattr(config, "text_config", None), config):
            n = getattr(cfg, "num_hidden_layers", None) if cfg is not None else None
            if n:
                return int(n)
        raise RuntimeError(f"cannot determine depth from config {type(config)}")

    def default_layer(self, frac: float = DEFAULT_LAYER_FRAC) -> int:
        return layer_index(self.n_layers, frac)

    def _hidden(self, text: str, layers: Sequence[int]):
        """(mean-pooled, last-token) hidden states per layer, one forward pass."""
        torch = self.torch
        with torch.no_grad():
            ids = self.tok(text, return_tensors="pt", truncation=True,
                           max_length=MAX_LENGTH).to(self.lm.device)
            out = self.lm(**ids, output_hidden_states=True)
            if out.hidden_states is None:
                raise RuntimeError(
                    f"{self.key}: the model returned no hidden states. The "
                    f"forward call asks for them explicitly, so this means the "
                    f"architecture ignores the flag — check whether the wrapper "
                    f"needs the language submodule called directly.")
            mask = ids["attention_mask"][0].bool()
            mean, last = {}, {}
            for li in layers:
                h = out.hidden_states[li][0][mask]     # [seq, d], real tokens only
                mean[li] = h.mean(0).float().cpu().numpy()
                last[li] = h[-1].float().cpu().numpy()  # the Yes/No token
        return mean, last

    def _p_yes(self, claim: str) -> float:
        """Output-space belief: softmax over the ' Yes' / ' No' continuations."""
        torch = self.torch
        # Same prefix the contrast pair uses, and the same clipping, so the
        # out-loud number and the probe are read off the same prompt.
        prefix = f"{claim_text(claim)}{SUFFIX}"
        if len(self.tok(prefix + " Yes")["input_ids"]) > MAX_LENGTH:
            prefix = f"{self._fit_claim(claim)}{SUFFIX}"
        lps = []
        with torch.no_grad():
            for cont in (" Yes", " No"):
                ids = self.tok(prefix + cont, return_tensors="pt",
                               truncation=True, max_length=MAX_LENGTH)
                n_prefix = len(self.tok(prefix)["input_ids"])
                ids = {k: v.to(self.lm.device) for k, v in ids.items()}
                logits = self.lm(**ids).logits[0, :-1]
                tgt = ids["input_ids"][0, 1:]
                lp = torch.log_softmax(logits.float(), -1)[
                    range(len(tgt)), tgt]
                # score only the continuation tokens
                lps.append(lp[n_prefix - 1:].sum().item())
        a, b = lps
        return float(np.exp(a) / (np.exp(a) + np.exp(b)))

    def token_states(self, text: str, layers: Sequence[int]) -> TokenStates:
        """Per-token hidden states with character offsets, specials dropped.

        Specials and padding are exactly the tokens whose offset span is empty
        (end <= start); they have no character extent to align on.
        """
        torch = self.torch
        with torch.no_grad():
            enc = self.tok(text, return_tensors="pt", truncation=True,
                           max_length=MAX_LENGTH, return_offsets_mapping=True)
            offsets = enc.pop("offset_mapping")[0].numpy()
            enc = {k: v.to(self.lm.device) for k, v in enc.items()}
            out = self.lm(**enc, output_hidden_states=True)
            keep = offsets[:, 1] > offsets[:, 0]
            if "attention_mask" in enc:
                keep &= enc["attention_mask"][0].bool().cpu().numpy()
            idx = np.where(keep)[0]
            states = {li: out.hidden_states[li][0][idx].float().cpu().numpy()
                      for li in layers}
        return TokenStates(states=states, starts=offsets[idx, 0],
                           ends=offsets[idx, 1])

    def _fit_claim(self, claim: str) -> str:
        """Clip the claim so the question and the verdict token always survive.

        `truncation=True` cuts from the right, so a claim long enough to fill the
        window took the "Is this claim true? Yes/No" suffix with it — leaving the
        two halves of the contrast pair *identical*. That is an empty pair, not an
        error, and it would have been invisible in the results. Clipping the claim
        instead keeps the suffix, and the assertion below keeps it honest.
        """
        room = MAX_LENGTH - len(self.tok(SUFFIX + " Yes")["input_ids"]) - 8
        ids = self.tok(claim_text(claim), truncation=True, max_length=room)["input_ids"]
        return self.tok.decode(ids, skip_special_tokens=True)

    def features(self, claim: str, layers: Sequence[int]) -> ItemFeatures:
        pos_text, neg_text = contrast_pair(claim)
        if len(self.tok(pos_text)["input_ids"]) > MAX_LENGTH:
            # Re-render from a clipped claim rather than let the window eat the
            # verdict token.
            pos_text, neg_text = contrast_pair(self._fit_claim(claim))
        assert (self.tok(pos_text)["input_ids"][-1]
                != self.tok(neg_text)["input_ids"][-1]), (
            f"contrast pair collapsed for a claim of {len(claim)} chars — the "
            f"two halves end on the same token, so there is nothing to contrast")
        pos_mean, pos_last = self._hidden(pos_text, layers)
        neg_mean, neg_last = self._hidden(neg_text, layers)
        return ItemFeatures(pos=pos_mean, neg=neg_mean, p_yes=self._p_yes(claim),
                            pos_last=pos_last, neg_last=neg_last)


def load_backend(key: str, synthetic: bool = False, **kw) -> Backend:
    """Backend for a model key; `synthetic=True` gives the no-GPU stand-in."""
    if synthetic:
        from .synthetic import SyntheticModel
        return SyntheticModel(key, **kw)
    return HFModel(key, **kw)
