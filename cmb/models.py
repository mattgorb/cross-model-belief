"""Model backends.

A backend answers one question per item: given a claim, what are the mean-pooled
hidden states of its positive and negative contrast halves at each requested
layer, and what does the model say out loud (P(Yes)) about the claim?

The out-loud number is not decoration — it is the Experiment 0 baseline and the
Gate A confident-slice selector. Activations only earn the framing if they beat
it (DESIGN.md §4).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

import numpy as np

from .config import DEFAULT_LAYER_FRAC, MAX_LENGTH, MODELS, layer_index
from .prompts import claim_text, contrast_pair


@dataclass
class ItemFeatures:
    """Per-item output of a backend."""
    pos: dict[int, np.ndarray]   # layer index -> mean-pooled hidden state
    neg: dict[int, np.ndarray]
    p_yes: float                 # output-space P(claim is true), the baseline


class Backend(Protocol):
    key: str
    n_layers: int

    def features(self, claim: str, layers: Sequence[int]) -> ItemFeatures: ...


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
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(hf_name)
        self.lm = AutoModelForCausalLM.from_pretrained(
            hf_name, torch_dtype=torch.float16, device_map=device_map,
            output_hidden_states=True)
        self.lm.eval()
        self.n_layers = int(self.lm.config.num_hidden_layers)

    def default_layer(self, frac: float = DEFAULT_LAYER_FRAC) -> int:
        return layer_index(self.n_layers, frac)

    def _hidden(self, text: str, layers: Sequence[int]) -> dict[int, np.ndarray]:
        torch = self.torch
        with torch.no_grad():
            ids = self.tok(text, return_tensors="pt", truncation=True,
                           max_length=MAX_LENGTH).to(self.lm.device)
            out = self.lm(**ids)
            mask = ids["attention_mask"][0].bool()
            res = {}
            for li in layers:
                h = out.hidden_states[li][0][mask]     # [seq, d], real tokens only
                res[li] = h.mean(0).float().cpu().numpy()
        return res

    def _p_yes(self, claim: str) -> float:
        """Output-space belief: softmax over the ' Yes' / ' No' continuations."""
        torch = self.torch
        prefix = f"{claim_text(claim)}\nIs this claim true?"
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

    def features(self, claim: str, layers: Sequence[int]) -> ItemFeatures:
        pos_text, neg_text = contrast_pair(claim)
        return ItemFeatures(pos=self._hidden(pos_text, layers),
                            neg=self._hidden(neg_text, layers),
                            p_yes=self._p_yes(claim))


def load_backend(key: str, synthetic: bool = False, **kw) -> Backend:
    """Backend for a model key; `synthetic=True` gives the no-GPU stand-in."""
    if synthetic:
        from .synthetic import SyntheticModel
        return SyntheticModel(key, **kw)
    return HFModel(key, **kw)
