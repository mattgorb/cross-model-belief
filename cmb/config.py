"""Global configuration: paths, seeds, the model registry, layer selection.

Everything the experiments need to agree on lives here so that a cached
activation written by one script is readable by every other one.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = Path(os.environ.get("CMB_CACHE", ROOT / "activations_cache"))
RESULTS_DIR = Path(os.environ.get("CMB_RESULTS", ROOT / "results"))

SEED = 0

# Default probe site = the FINAL layer. Mean-pooling the last hidden state is
# exactly what an embedding API hands back, which is the setting the linear
# alignment result was established in — so the default keeps this project on
# the ground the prior work already covers. DESIGN.md §4's 0.6 x depth remains
# available per model, and Exp 5 sweeps depth.
DEFAULT_LAYER_SPEC = "final"
DEFAULT_LAYER_FRAC = 1.0
LAYER_SWEEP = (0.4, 0.5, 0.6, 0.7, 0.8, 1.0)

# fp16 always for extraction — DESIGN.md §4/§9 forbids 4-bit here, quantization
# perturbs exactly the thing the probe reads.
DTYPE = "float16"
MAX_LENGTH = 512


@dataclass(frozen=True)
class ModelSpec:
    key: str          # short name used in cache keys and CLI flags
    hf_name: str      # HuggingFace repo id
    family: str       # "qwen" | "llama" — drives same/cross-family pairing


MODELS = {
    m.key: m
    for m in [
        ModelSpec("qwen-1.5b", "Qwen/Qwen2.5-1.5B-Instruct", "qwen"),
        ModelSpec("qwen-7b", "Qwen/Qwen2.5-7B-Instruct", "qwen"),
        ModelSpec("qwen-32b", "Qwen/Qwen2.5-32B-Instruct", "qwen"),
        ModelSpec("llama-8b", "meta-llama/Llama-3.1-8B-Instruct", "llama"),
    ]
}

# DESIGN.md §6. The held-out pair is never used to fit anything in Exp 3/4.
SAME_FAMILY_PAIR = ("qwen-7b", "qwen-32b")
CROSS_FAMILY_PAIR = ("qwen-7b", "llama-8b")
HELD_OUT_PAIR = ("qwen-1.5b", "llama-8b")
PAIRS = {
    "same-family": SAME_FAMILY_PAIR,
    "cross-family": CROSS_FAMILY_PAIR,
    "held-out": HELD_OUT_PAIR,
}

# DESIGN.md §5. `truthfulqa` is the Row-2 enrichment set; `imdb` is the
# sentiment-confound extreme; IMDB -> TruthfulQA is the headline honesty cell.
DATASETS = ("geometry_of_truth", "truthfulqa", "boolq", "imdb", "rte", "mmlu")
MATRIX_DATASETS = ("geometry_of_truth", "truthfulqa", "boolq", "imdb", "rte")

# Test-split fraction used everywhere a train/test split is needed.
TEST_FRAC = 0.4
# "Confident slice" for Gate A = top this fraction of items by output confidence.
CONF_SLICE_Q = 0.5


def pair_kind(a: str, b: str) -> str:
    """'same-family' or 'cross-family' for a model pair (Exp 1 reports by this)."""
    return "same-family" if MODELS[a].family == MODELS[b].family else "cross-family"


def layer_index(n_layers: int, frac: float) -> int:
    """Hidden-state index for a depth fraction, clamped into range.

    Index 0 of `hidden_states` is the embedding output, so valid probe layers
    are 1..n_layers; `frac=1.0` is the final layer.
    """
    return max(1, min(n_layers, int(round(n_layers * frac))))


def parse_layer_spec(spec) -> float | int:
    """Parse a per-model layer argument.

    Accepts `"final"` (the default), a depth fraction like `0.6`, a negative
    index like `-1` or `-3` counting back from the final layer, or an absolute
    hidden-state index like `24`. Fractions are resolved against each model's
    own depth, so one flag means the same *relative* site in models of
    different sizes; absolute indices do not, which is usually what you want
    only when the two models have the same depth.
    """
    if isinstance(spec, (int, float)):
        return spec
    s = str(spec).strip().lower()
    if s in ("final", "last", "-1"):
        return 1.0
    if s.startswith("-"):
        return int(s)
    v = float(s)
    return v if 0 < v <= 1.0 and "." in s else int(v)


def resolve_layer(spec, n_layers: int) -> int:
    """Turn a layer spec into a concrete hidden-state index for one model."""
    spec = parse_layer_spec(spec)
    if isinstance(spec, int):
        return n_layers + 1 + spec if spec < 0 else max(1, min(n_layers, spec))
    return layer_index(n_layers, spec)
