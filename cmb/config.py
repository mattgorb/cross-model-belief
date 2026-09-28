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
# Three sites, not six: the final layer (the default, and where the alignment
# result was established), mid-depth (where the probing literature works), and
# one in between. All three come out of one forward pass, so this is a disk
# decision, not a GPU one — but a layer outside this set forces a full
# re-extraction, which costs another GPU rental.
LAYER_SWEEP = (0.5, 0.75, 1.0)

# fp16 always for extraction — DESIGN.md §4/§9 forbids 4-bit here, quantization
# perturbs exactly the thing the probe reads.
DTYPE = "float16"
MAX_LENGTH = 512


# Extraction uses raw completion-style prompts and NO chat template, which is
# what keeps reasoning models tractable here: with no template there is no
# `<think>` block, so nothing is ever generated and the probe reads the claim's
# representation *before* any serialized deliberation. See DESIGN.md §6.1 — this
# is a deliberate choice, not an oversight, and Exp 7 is where the other reading
# (belief after the reasoning trace) is specified.
USE_CHAT_TEMPLATE = False
THINKING = False


@dataclass(frozen=True)
class ModelSpec:
    key: str          # short name used in cache keys and CLI flags
    hf_name: str      # HuggingFace repo id
    family: str       # "qwen" | "llama" | "gemma" — drives same/cross-family pairing
    n_layers: int     # documented depth; the loaded config is authoritative
    reasoning: bool = False   # hybrid thinking model (thinking is left OFF, see §6.1)
    base: bool = False        # pre-post-training checkpoint
    gated: bool = False       # needs HF_TOKEN and an accepted license
    multimodal: bool = False  # ...ForConditionalGeneration wrapper, not ...ForCausalLM


MODELS = {
    m.key: m
    for m in [
        # -- Qwen 3, the size ladder. The only current family shipping small
        # dense checkpoints *with* matching bases, which is why the anchor and
        # the held-out pair live here.
        ModelSpec("qwen3-1.7b", "Qwen/Qwen3-1.7B", "qwen", 28, reasoning=True),
        ModelSpec("qwen3-1.7b-base", "Qwen/Qwen3-1.7B-Base", "qwen", 28, base=True),
        ModelSpec("qwen3-4b", "Qwen/Qwen3-4B", "qwen", 36, reasoning=True),
        ModelSpec("qwen3-8b", "Qwen/Qwen3-8B", "qwen", 36, reasoning=True),
        ModelSpec("qwen3-32b", "Qwen/Qwen3-32B", "qwen", 64, reasoning=True),
        ModelSpec("qwen3-8b-base", "Qwen/Qwen3-8B-Base", "qwen", 36, base=True),
        # -- Newest Qwen generation. Dense 27B is the smallest open one in this
        # line (no 4B/8B/14B this generation), and it is a native VLM, so it
        # loads through the conditional-generation wrapper.
        ModelSpec("qwen38-27b", "Qwen/Qwen3.8-27B", "qwen", 64,
                  reasoning=True, multimodal=True),
        # -- Gemma 4: the third family, and the cleanest cross-family contrast,
        # since it is reasoning-era like Qwen 3 and ships base + instruct.
        ModelSpec("gemma4-12b", "google/gemma-4-12B-it", "gemma", 48,
                  reasoning=True, multimodal=True),
        ModelSpec("gemma4-12b-base", "google/gemma-4-12B", "gemma", 48,
                  base=True, multimodal=True),
        # The large-end cross-family partner for qwen3-32b: the only same-size
        # cross-family pair on the board, which is what makes it possible to ask
        # whether error correlation changes with scale at matched size. Note this
        # is the regular Gemma 4 architecture, not the 12B's encoder-free
        # `gemma4_unified` variant, so gemma-12b vs gemma-31b is not a clean
        # within-family scale contrast.
        ModelSpec("gemma4-31b", "google/gemma-4-31B-it", "gemma", 60,
                  reasoning=True, multimodal=True),
        ModelSpec("gemma4-31b-base", "google/gemma-4-31B", "gemma", 60,
                  base=True, multimodal=True),
        # -- OLMo 3: a fourth family, and the only one whose pretraining corpus
        # is public. That makes it the one pair where "shared training data
        # drives the error correlation" is checkable rather than assumed.
        ModelSpec("olmo3-7b", "allenai/Olmo-3-7B-Instruct", "olmo", 32,
                  reasoning=True),
        # -- Llama: still the newest *dense* small Llama (Llama 4 is MoE and has
        # no 8B dense). Non-reasoning, which makes it useful rather than stale:
        # it is the only pre-reasoning-era point on the board.
        ModelSpec("llama-8b", "meta-llama/Llama-3.1-8B-Instruct", "llama", 32,
                  gated=True),
        ModelSpec("llama-8b-base", "meta-llama/Llama-3.1-8B", "llama", 32,
                  base=True, gated=True),
    ]
}

# DESIGN.md §6. One variable per pair wherever the model zoo allows it; where it
# does not, the confound is named in the key.
SAME_FAMILY_PAIR = ("qwen3-8b", "qwen3-32b")        # scale only, one generation
CROSS_FAMILY_PAIR = ("qwen3-8b", "gemma4-12b")      # family; both reasoning-era
CROSS_ERA_PAIR = ("qwen3-8b", "llama-8b")           # size-matched, but family AND era
CROSS_GENERATION_PAIR = ("qwen3-8b", "qwen38-27b")  # same family, newest generation
CROSS_FAMILY_LARGE_PAIR = ("qwen3-32b", "gemma4-31b")   # cross-family at matched size
HELD_OUT_PAIR = ("qwen3-1.7b", "olmo3-7b")          # never used to fit anything
WEAK_STRONG_PAIR = ("qwen3-1.7b", "qwen3-32b")      # fit where labels are cheap
# Base <-> instruct, same weights up to post-training: Experiment 6.
POST_TRAINING_PAIR = ("qwen3-8b-base", "qwen3-8b")
PAIRS = {
    "same-family": SAME_FAMILY_PAIR,
    "cross-family": CROSS_FAMILY_PAIR,
    "cross-era": CROSS_ERA_PAIR,
    "cross-generation": CROSS_GENERATION_PAIR,
    "cross-family-large": CROSS_FAMILY_LARGE_PAIR,
    "held-out": HELD_OUT_PAIR,
    "weak-strong": WEAK_STRONG_PAIR,
    "post-training": POST_TRAINING_PAIR,
    # A second Exp 6 control in another family, since "post-training" now also
    # means "where thinking was installed" (DESIGN.md §6.1).
    "post-training-gemma": ("gemma4-12b-base", "gemma4-12b"),
    # A second Exp 6 control in another family, since "post-training" now also
    # means "where thinking was installed" (DESIGN.md §6.1).
    "post-training-gemma": ("gemma4-12b-base", "gemma4-12b"),
}
# The pairs that answer the floor-vs-ceiling question (`--pair all`); the
# post-training pair is a control for Exp 6, not a point on that curve.
RESEARCH_PAIRS = ("same-family", "cross-family", "cross-family-large",
                  "cross-era", "held-out")

# DESIGN.md §5. `truthfulqa` is the Row-2 enrichment set; `imdb` is the
# sentiment-confound extreme; IMDB -> TruthfulQA is the headline honesty cell.
DATASETS = ("geometry_of_truth", "truthfulqa", "boolq", "imdb", "rte", "mmlu")
MATRIX_DATASETS = ("geometry_of_truth", "truthfulqa", "boolq", "imdb", "rte")

# Per-dataset ceiling, applied even under `--n all`. IMDB's test split is 25k
# items, an order of magnitude more than anything else here, and it exists only
# as the sentiment-confound extreme for one cell of the Exp 4 matrix — so it
# would dominate the extraction budget while contributing one number. The cap is
# label-balanced, not a head of the shuffle.
DATASET_CAPS = {"imdb": 5000}

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
