"""Shared plumbing for the experiment scripts.

Holds the argument parser every script uses, the result writer, and the one
pipeline object (`PairRun`) that turns "a model pair on a dataset" into fitted
probes, a fitted map, and test-split verdicts — fitted on train, scored on test,
with the split shared across experiments so their numbers are comparable.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cmb import align, metrics, probes                     # noqa: E402
from cmb.config import (DEFAULT_LAYER_SPEC, PAIRS, RESULTS_DIR,  # noqa: E402
                        SEED, TEST_FRAC, pair_kind, resolve_layer)
from cmb.data import load_items, split_items               # noqa: E402
from cmb.extract import paired                             # noqa: E402


def base_parser(description: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--pair", default="cross-family",
                   help="named pair (same-family|cross-family|held-out) or 'a,b'")
    p.add_argument("--dataset", default="truthfulqa")
    p.add_argument("--n", type=int, default=800, help="items per dataset")
    p.add_argument("--layer", default=DEFAULT_LAYER_SPEC,
                   help="probe layer for both models: 'final' (default), a "
                        "depth fraction like 0.6, a negative index like -3, or "
                        "an absolute index like 24")
    p.add_argument("--layer-a", default=None,
                   help="override the layer for model A (same formats)")
    p.add_argument("--layer-b", default=None,
                   help="override the layer for model B (same formats)")
    p.add_argument("--pooling", default="mean", choices=["mean", "last"],
                   help="which cached pooling to read: 'mean' over the whole "
                        "prompt (the alignment setting) or 'last', the Yes/No "
                        "token (what the probing literature reads). Both are in "
                        "the cache, so switching is free.")
    p.add_argument("--probe", default="ccs", choices=probes.PROBE_KINDS,
                   help="belief probe: 'ccs' (unsupervised, the case that "
                        "matters) or the labeled baselines 'mass-mean' / 'lr'")
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--synthetic", action="store_true",
                   help="run against the synthetic backend (no GPU, no download)")
    p.add_argument("--refresh", action="store_true", help="ignore cached activations")
    p.add_argument("--tag", default="", help="suffix for the results filename")
    return p


def layer_specs(args) -> tuple[str, str]:
    """(layer_a, layer_b) from the CLI, each falling back to --layer."""
    return (args.layer_a or args.layer, args.layer_b or args.layer)


def resolve_pair(spec: str) -> tuple[str, str]:
    if spec in PAIRS:
        return PAIRS[spec]
    a, b = spec.split(",")
    return a.strip(), b.strip()


def make_run(a: str, b: str, args, dataset: str | None = None,
             layers: tuple | None = None) -> "PairRun":
    """Build a `PairRun` for a model pair from parsed CLI args.

    One place where the CLI meets the pipeline, so a new flag (e.g. `--probe`)
    reaches every experiment at once instead of being threaded by hand.
    """
    la, lb = layers if layers is not None else layer_specs(args)
    return PairRun(a, b, dataset or args.dataset, la, lb, args.n,
                   args.synthetic, args.seed,
                   probe_kind=getattr(args, "probe", "ccs"),
                   pooling=getattr(args, "pooling", "mean"))


def write_result(name: str, payload: dict, tag: str = "") -> Path:
    """Persist a machine-readable result next to the human-readable stdout."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / f"{name}{('_' + tag) if tag else ''}.json"
    payload = dict(payload)
    payload["written_at"] = datetime.now(timezone.utc).isoformat()
    path.write_text(json.dumps(payload, indent=2, default=_jsonable))
    print(f"\n[wrote {path}]")
    return path


def _jsonable(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if hasattr(o, "to_dict"):
        return o.to_dict()
    raise TypeError(type(o))


def header(title: str) -> None:
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")


@dataclass
class PairRun:
    """One model pair on one dataset at one layer: probes, map, test verdicts.

    Everything fittable is fitted on the train split only. That is what makes
    Experiment 4 interpretable — per DESIGN.md §4's attribution control, the
    probe and the map are refit per cell, so the only object that ever crosses
    a dataset boundary is the false-agreement direction.
    """
    model_a: str
    model_b: str
    dataset: str
    layer_a_spec: str = DEFAULT_LAYER_SPEC
    layer_b_spec: str = DEFAULT_LAYER_SPEC
    n: int | None = 800
    synthetic: bool = False
    seed: int = SEED
    test_frac: float = TEST_FRAC
    probe_kind: str = "ccs"
    pooling: str = "mean"

    # filled by build()
    layer_a: int = field(init=False, default=0)
    layer_b: int = field(init=False, default=0)

    def build(self, refresh: bool = False) -> "PairRun":
        a, b = paired(self.model_a, self.model_b, self.dataset, self.n,
                      synthetic=self.synthetic, refresh=refresh,
                      layer_a=(self.layer_a_spec,), layer_b=(self.layer_b_spec,))
        # Pooling is a view on the same cache, applied once here so every
        # downstream reader sees one convention.
        self.acts_a = a.with_pooling(self.pooling)
        self.acts_b = b.with_pooling(self.pooling)
        self.layer_a = resolve_layer(self.layer_a_spec, a.n_layers)
        self.layer_b = resolve_layer(self.layer_b_spec, b.n_layers)

        items = load_items(self.dataset, self.n)
        by_id = {it.item_id: it for it in items}
        ordered = [by_id[i] for i in a.item_ids]
        train_items, test_items = split_items(ordered, self.test_frac, self.seed)
        train_ids = {it.item_id for it in train_items}
        self.tr = np.array([i in train_ids for i in a.item_ids])
        self.te = ~self.tr

        self._fit_probes()
        self._fit_maps()
        return self

    # -- fitting ------------------------------------------------------------

    def _fit_probes(self) -> None:
        self.probe_a = self._fit_one(self.acts_a, self.layer_a)
        self.probe_b = self._fit_one(self.acts_b, self.layer_b)

    def _fit_one(self, acts, layer):
        p = probes.make_probe(self.probe_kind, acts.pos[layer].shape[1])
        Xp, Xn = acts.pos[layer][self.tr], acts.neg[layer][self.tr]
        if self.probe_kind == "ccs":
            p.fit(Xp, Xn, seed=self.seed)
        else:
            # The labeled baselines fit on the train split's labels; CCS never
            # sees them (paper/PLAN.md §3).
            p.fit(Xp, Xn, acts.labels[self.tr], seed=self.seed)
        # One bit of label info for CCS's arbitrary orientation, spent on the
        # train split only; a no-op for the supervised baselines.
        p.resolve_sign(Xp, Xn, acts.labels[self.tr])
        return p

    def _fit_maps(self) -> None:
        """Maps are fitted on pos and neg halves jointly (the probe sees both)."""
        Xa = np.concatenate([self.acts_a.pos[self.layer_a][self.tr],
                             self.acts_a.neg[self.layer_a][self.tr]])
        Xb = np.concatenate([self.acts_b.pos[self.layer_b][self.tr],
                             self.acts_b.neg[self.layer_b][self.tr]])
        self.map_ab, self.map_ba = align.fit_map_both_ways(Xa, Xb)

    # -- accessors ----------------------------------------------------------

    @property
    def labels_test(self) -> np.ndarray:
        return self.acts_a.labels[self.te]

    def native_belief(self, which: str, split: str = "test") -> np.ndarray:
        acts, probe, layer = self._sel(which)
        m = self.te if split == "test" else self.tr
        return probe.belief(acts.pos[layer][m], acts.neg[layer][m])

    def transported_belief(self, source: str, split: str = "test") -> np.ndarray:
        """Read `source`'s probe on the *other* model's activations via the map.

        The other model's activations are pushed into the source's space, which
        is the direction that keeps the characterized instrument fixed — the
        point of transport is to reuse a probe you have already validated.
        """
        m = self.te if split == "test" else self.tr
        if source == "a":
            acts, probe, layer, mp = self.acts_b, self.probe_a, self.layer_b, self.map_ba
        else:
            acts, probe, layer, mp = self.acts_a, self.probe_b, self.layer_a, self.map_ab
        return probe.belief(mp(acts.pos[layer][m]), mp(acts.neg[layer][m]))

    def cka(self) -> float:
        return align.linear_cka(self.acts_a.pos[self.layer_a][self.te],
                                self.acts_b.pos[self.layer_b][self.te])

    def aligned_features(self, split: str = "test") -> np.ndarray:
        """Joint representation of an item in the aligned space.

        Two halves per model, and both are needed:

          * the **contrast difference** (pos - neg), which is what the belief
            probe effectively reads — it cancels whatever the two halves share;
          * the **contrast mean** ((pos + neg) / 2), which is what the
            difference threw away: the claim's content and its familiarity.

        A shared prominence confound sits in the *common* part of the pair, so a
        feature space built only from differences cancels the very thing that
        produces Row 2 — and Experiment 3 would then be asking whether false
        agreement is visible in a space engineered not to show it.

        B's activations are pushed into A's space first, so the result lives in
        one space for a given pair.
        """
        m = self.te if split == "test" else self.tr
        pa, na = self.acts_a.pos[self.layer_a][m], self.acts_a.neg[self.layer_a][m]
        pb = self.map_ba(self.acts_b.pos[self.layer_b][m])
        nb = self.map_ba(self.acts_b.neg[self.layer_b][m])
        return np.concatenate([pa - na, (pa + na) / 2,
                               pb - nb, (pb + nb) / 2], axis=1)

    def verdicts(self, mode: str = "native", split: str = "test"):
        """(v1, v2, ground truth) for the 8-cell table.

        mode='native'    each model's own probe
        mode='a_to_b'    A's probe everywhere (B read through the map)
        mode='b_to_a'    B's probe everywhere (A read through the map)
        """
        m = self.te if split == "test" else self.tr
        gt = self.acts_a.labels[m]
        if mode == "native":
            s1, s2 = self.native_belief("a", split), self.native_belief("b", split)
        elif mode == "a_to_b":
            s1, s2 = self.native_belief("a", split), self.transported_belief("a", split)
        elif mode == "b_to_a":
            s1, s2 = self.transported_belief("b", split), self.native_belief("b", split)
        else:
            raise ValueError(mode)
        return (s1 >= 0.5).astype(int), (s2 >= 0.5).astype(int), gt

    def _sel(self, which: str):
        if which == "a":
            return self.acts_a, self.probe_a, self.layer_a
        return self.acts_b, self.probe_b, self.layer_b

    @property
    def kind(self) -> str:
        try:
            return pair_kind(self.model_a, self.model_b)
        except KeyError:
            return "unknown"
