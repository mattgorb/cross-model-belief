"""Dataset loaders (DESIGN.md §5).

Every dataset is reduced to the same unit: a single declarative claim with a
ground-truth true/false label. That is what CCS contrast pairs are built from
and what the 8-cell table in Experiment 1 is scored against.

Item ids are stable strings — they are part of the activation cache key, so a
loader must be deterministic. The item list for a (dataset, n) request is
materialized once to `activations_cache/manifests/` and reused, which keeps the
cache valid even if an upstream dataset is re-shuffled or re-released.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .config import CACHE_DIR, SEED

MANIFEST_DIR = CACHE_DIR / "manifests"


@dataclass(frozen=True)
class Item:
    item_id: str
    claim: str
    label: int          # 1 = ground-truth true, 0 = ground-truth false
    dataset: str
    group: str = ""     # items sharing a group came from one source question


def _finalize(items: list[Item], n: int | None, seed: int = SEED) -> list[Item]:
    """Deterministic shuffle + truncate, balanced-ish on the label."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(items))
    items = [items[i] for i in idx]
    if n is not None:
        items = items[:n]
    return items


# ---------------------------------------------------------------------------
# individual loaders
# ---------------------------------------------------------------------------

def _load_geometry_of_truth(n):
    """Marks & Tegmark curated true/false statements — the primary set.

    Read straight from the paper's repo (`saprmarks/geometry-of-truth`), which is
    a set of per-topic CSVs of (statement, label). The HF community mirrors that
    used to hold these are 401 now, and the raw CSVs are the authoritative copy
    anyway. Set `CMB_GOT_DIR` to a local clone's `datasets/` folder to work
    offline.
    """
    import csv
    import io
    import os
    import urllib.request

    GOT_RAW = ("https://raw.githubusercontent.com/saprmarks/geometry-of-truth"
               "/main/datasets")
    # One topic per file; these four are the ones the paper trains probes on.
    topics = ["cities", "sp_en_trans", "larger_than", "companies_true_false"]
    local = os.environ.get("CMB_GOT_DIR")

    items, failures = [], []
    for topic in topics:
        try:
            if local:
                text = (Path(local) / f"{topic}.csv").read_text()
            else:
                with urllib.request.urlopen(f"{GOT_RAW}/{topic}.csv", timeout=30) as r:
                    text = r.read().decode("utf-8")
        except Exception as e:
            failures.append(f"{topic}: {type(e).__name__}: {e}")
            continue
        for i, row in enumerate(csv.DictReader(io.StringIO(text))):
            stmt = row.get("statement") or row.get("claim") or row.get("text")
            lab = row.get("label", row.get("truth"))
            if not stmt or lab is None or str(lab).strip() == "":
                continue
            items.append(Item(f"got/{topic}/{i}", str(stmt).strip(), int(lab),
                              "geometry_of_truth", group=f"got/{topic}/{i}"))
    if not items:
        raise RuntimeError(
            "Geometry of Truth not reachable. Clone "
            "github.com/saprmarks/geometry-of-truth and set CMB_GOT_DIR to its "
            "datasets/ folder, or run with --synthetic. Tried:\n  "
            + "\n  ".join(failures))
    return _finalize(items, n)


def _load_truthfulqa(n):
    """TruthfulQA — popular misconceptions, the Row-2 enrichment set.

    One true and one false claim per question, so false agreements caused by a
    *familiar* phrasing land in the data by construction.
    """
    from datasets import load_dataset

    ds = load_dataset("truthfulqa/truthful_qa", "generation", split="validation")
    items = []
    for i, r in enumerate(ds):
        q = r["question"]
        correct = r["correct_answers"][0] if r["correct_answers"] else None
        wrong = r["incorrect_answers"][0] if r["incorrect_answers"] else None
        if not correct or not wrong:
            continue
        g = f"tqa/{i}"
        items.append(Item(f"{g}/t", f"{q} {correct}", 1, "truthfulqa", group=g))
        items.append(Item(f"{g}/f", f"{q} {wrong}", 0, "truthfulqa", group=g))
    return _finalize(items, n)


def _load_boolq(n):
    from datasets import load_dataset

    ds = load_dataset("google/boolq", split="validation")
    items = []
    for i, r in enumerate(ds):
        g = f"boolq/{i}"
        claim = f"{r['passage'].strip()}\n{r['question'].strip()}? Yes."
        items.append(Item(g, claim, int(bool(r["answer"])), "boolq", group=g))
    return _finalize(items, n)


def _load_imdb(n):
    """IMDB as a claim about sentiment — the sentiment-confound extreme.

    DESIGN.md §4: IMDB -> TruthfulQA is the designated hardest cell of the
    generalization matrix, so this set exists to be maximally unlike the others.
    """
    from datasets import load_dataset

    ds = load_dataset("stanfordnlp/imdb", split="test")
    items = []
    for i, r in enumerate(ds):
        text = " ".join(r["text"].split())[:800]
        g = f"imdb/{i}"
        # Assert "positive" for every item; the label makes half of them false.
        items.append(Item(g, f'Review: "{text}"\nThe sentiment of this review '
                             f"is positive.", int(r["label"] == 1), "imdb",
                          group=g))
    return _finalize(items, n)


def _load_rte(n):
    from datasets import load_dataset

    ds = load_dataset("nyu-mll/glue", "rte", split="validation")
    items = []
    for i, r in enumerate(ds):
        g = f"rte/{i}"
        claim = (f"Premise: {r['sentence1'].strip()}\n"
                 f"It follows that: {r['sentence2'].strip()}")
        # GLUE RTE: label 0 = entailment, 1 = not entailment.
        items.append(Item(g, claim, int(r["label"] == 0), "rte", group=g))
    return _finalize(items, n)


MMLU_SUBJECTS = ["college_physics", "college_chemistry", "college_mathematics",
                 "professional_law", "college_biology"]


def _load_mmlu(n, chosen_wrong: dict[str, int] | None = None):
    """MMLU, converted per the DESIGN.md §5 note.

    Two claims per question: the true one and *one* wrong one — never all three
    distractors, which would skew the set 3:1 and muddy the Row-2 rate. The
    wrong claim should be the model's own chosen wrong answer (pass
    `chosen_wrong`: group -> option index). Without it we fall back to a
    deterministic distractor, which is coverage-only data; say so in results.
    """
    from datasets import load_dataset

    items = []
    for s in MMLU_SUBJECTS:
        ds = load_dataset("cais/mmlu", s, split="test")
        for i, r in enumerate(ds):
            g = f"mmlu/{s}/{i}"
            gold = int(r["answer"])
            wrong_idx = (chosen_wrong or {}).get(g)
            if wrong_idx is None or wrong_idx == gold:
                wrong_idx = (gold + 1) % len(r["choices"])
            q = r["question"]
            items.append(Item(f"{g}/t", f"{q} The answer is {r['choices'][gold]}.",
                              1, "mmlu", group=g))
            items.append(Item(f"{g}/f", f"{q} The answer is {r['choices'][wrong_idx]}.",
                              0, "mmlu", group=g))
    return _finalize(items, n)


LOADERS = {
    "geometry_of_truth": _load_geometry_of_truth,
    "truthfulqa": _load_truthfulqa,
    "boolq": _load_boolq,
    "imdb": _load_imdb,
    "rte": _load_rte,
    "mmlu": _load_mmlu,
}


# ---------------------------------------------------------------------------
# manifest-backed entry point
# ---------------------------------------------------------------------------

def load_items(dataset: str, n: int | None = None, refresh: bool = False) -> list[Item]:
    """Items for a dataset, materialized once and pinned to a manifest file."""
    if dataset not in LOADERS:
        raise KeyError(f"unknown dataset {dataset!r}; have {sorted(LOADERS)}")
    path = MANIFEST_DIR / f"{dataset}_n{n if n is not None else 'all'}.jsonl"
    if path.exists() and not refresh:
        return [Item(**json.loads(line)) for line in path.read_text().splitlines()]

    items = LOADERS[dataset](n)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(asdict(it)) + "\n" for it in items))
    return items


def split_items(items: list[Item], test_frac: float, seed: int = SEED):
    """Group-aware train/test split.

    Items from one source question (a TruthfulQA true/false pair, an MMLU
    true/chosen-wrong pair) go to the same side, so the split never leaks a
    claim's twin across it.
    """
    groups = sorted({it.group or it.item_id for it in items})
    rng = np.random.default_rng(seed)
    rng.shuffle(groups)
    n_test = int(round(len(groups) * test_frac))
    test_groups = set(groups[:n_test])
    train = [it for it in items if (it.group or it.item_id) not in test_groups]
    test = [it for it in items if (it.group or it.item_id) in test_groups]
    return train, test
