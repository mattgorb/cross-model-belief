"""Tokenizer-difference handling for cross-model transport.

Two independently trained models tokenize the same string differently, so there
is no token-index correspondence to fit a map on. The fix, ported from the
HELIX / linear-alignment work (`lm_pertoken.py` there), is to align in
**character space** rather than token space, using the fast tokenizers' offset
mapping:

  * **span alignment** (default) — take the union of all character boundaries
    from both tokenizers, and mean-pool each model's tokens overlapping each
    common span. Every span exists in both models by construction, so a
    Qwen 1-token word and a Llama 3-token word become one comparable pair.
  * **token alignment** — match A's token ending at character `e` to B's token
    whose end offset is `e` or the next greater. Cheaper, slightly lossier,
    matches `build_alignment_pairs` in the source work.

Why it matters here and not only for tidiness: pooled item vectors give one
`(x, y)` pair per item, so fitting a `d x d` map from a few hundred items is
hopeless. Span alignment gives a pair per span — tens of thousands from the
same forward passes — which is what makes the ridge map well-posed.

`tokenizer_compatibility` reproduces the HELIX compatibility score. The finding
there was that tokenizer compatibility and size gap predict whether alignment
succeeds, so it belongs next to CKA whenever a transfer number is reported:
a weak cross-family result at low compatibility indicts the map, not the probe.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# The HELIX weighting: vocabulary overlap and length similarity dominate.
COMPAT_WEIGHTS = {"vocab_jaccard": 0.3, "length_correlation": 0.2,
                  "length_similarity": 0.3, "exact_match_rate": 0.2}
COMPAT_THRESHOLD = 0.6


@dataclass
class TokenStates:
    """Per-token hidden states for one text, with character offsets.

    Specials and padding are already dropped — they are exactly the tokens with
    `end <= start` in the offset mapping, and they have no character span to
    align on.
    """
    states: dict[int, np.ndarray]   # layer -> [n_tokens, d]
    starts: np.ndarray              # [n_tokens] character start offsets
    ends: np.ndarray                # [n_tokens] character end offsets

    def __len__(self) -> int:
        return len(self.ends)


def span_pairs(a: TokenStates, b: TokenStates, layer_a: int, layer_b: int):
    """Character-span-aligned (X_a, X_b) pairs for one text.

    Boundaries from both tokenizers are merged, and each resulting span is
    represented by the mean of the tokens overlapping it on each side.
    """
    if len(a) == 0 or len(b) == 0:
        return np.empty((0, a.states[layer_a].shape[1])), \
               np.empty((0, b.states[layer_b].shape[1]))

    bounds = np.unique(np.concatenate([a.starts, a.ends, b.starts, b.ends]))
    xs, ys = [], []
    for lo, hi in zip(bounds[:-1], bounds[1:]):
        ma = (a.starts < hi) & (a.ends > lo)
        mb = (b.starts < hi) & (b.ends > lo)
        if not ma.any() or not mb.any():
            continue
        xs.append(a.states[layer_a][ma].mean(0))
        ys.append(b.states[layer_b][mb].mean(0))
    if not xs:
        return np.empty((0, a.states[layer_a].shape[1])), \
               np.empty((0, b.states[layer_b].shape[1]))
    return np.stack(xs), np.stack(ys)


def token_pairs(a: TokenStates, b: TokenStates, layer_a: int, layer_b: int):
    """Nearest-end-offset token alignment (the cheaper variant)."""
    if len(a) == 0 or len(b) == 0:
        return np.empty((0, a.states[layer_a].shape[1])), \
               np.empty((0, b.states[layer_b].shape[1]))
    j = np.searchsorted(b.ends, a.ends, side="left")
    keep = j < len(b.ends)
    return a.states[layer_a][keep], b.states[layer_b][j[keep]]


def build_pairs(states_a: list[TokenStates], states_b: list[TokenStates],
                layer_a: int, layer_b: int, method: str = "span"):
    """Stack aligned pairs over a corpus, with the match rate as a diagnostic."""
    fn = {"span": span_pairs, "token": token_pairs}[method]
    xs, ys, n_a = [], [], 0
    for sa, sb in zip(states_a, states_b):
        x, y = fn(sa, sb, layer_a, layer_b)
        n_a += len(sa)
        if len(x):
            xs.append(x)
            ys.append(y)
    if not xs:
        raise RuntimeError(f"{method} alignment produced no pairs — check that "
                           "both tokenizers are 'fast' and return offsets")
    X, Y = np.concatenate(xs), np.concatenate(ys)
    return X, Y, {"method": method, "n_pairs": int(len(X)),
                  "n_source_tokens": int(n_a),
                  "pairs_per_token": float(len(X) / max(n_a, 1))}


# ---------------------------------------------------------------------------
# compatibility score
# ---------------------------------------------------------------------------

def _normalize_token(t: str) -> str:
    """Strip the family-specific space markers so 'Ġthe', '▁the' and 'the' match."""
    return t.lstrip("Ġ▁ ").lower()


def tokenizer_compatibility(tok_a, tok_b, texts: list[str]) -> dict:
    """HELIX tokenizer-compatibility score in [0, 1] for a model pair.

    Vocabulary Jaccard, tokenized-length correlation, length similarity and
    position-wise token match rate, combined with the weights used in that work.
    Compatible above ~0.6.
    """
    va, vb = set(tok_a.get_vocab()), set(tok_b.get_vocab())
    jaccard = len(va & vb) / len(va | vb) if (va | vb) else 0.0

    lens, diffs, matches = [], [], []
    for text in texts:
        ta, tb = tok_a.tokenize(text), tok_b.tokenize(text)
        la, lb = len(ta), len(tb)
        if max(la, lb) == 0:
            continue
        lens.append((la, lb))
        diffs.append(abs(la - lb) / max(la, lb))
        matches.append(sum(_normalize_token(x) == _normalize_token(y)
                           for x, y in zip(ta, tb)) / max(la, lb))

    arr = np.array(lens)
    corr = float(np.corrcoef(arr[:, 0], arr[:, 1])[0, 1]) if len(lens) > 1 else 0.0
    parts = {"vocab_jaccard": jaccard,
             "length_correlation": corr if np.isfinite(corr) else 0.0,
             "length_similarity": 1 - float(np.mean(diffs)) if diffs else 0.0,
             "exact_match_rate": float(np.mean(matches)) if matches else 0.0}
    score = sum(COMPAT_WEIGHTS[k] * v for k, v in parts.items())
    return {**parts, "vocab_size_a": len(va), "vocab_size_b": len(vb),
            "compatibility_score": float(score),
            "compatible": bool(score > COMPAT_THRESHOLD)}
