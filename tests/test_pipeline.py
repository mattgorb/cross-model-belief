"""Tests. Everything runs on the synthetic backend — no GPU, no downloads.

They check the plumbing (cache round-trips, split hygiene, metric definitions)
and use the synthetic world as a positive control: its latents contain a truth
direction and a shared prominence confound, so a correct pipeline must recover
both. A failure here is a bug in the code, not a result about models.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

_TMP = tempfile.mkdtemp(prefix="cmb-test-")
os.environ.setdefault("CMB_CACHE", str(Path(_TMP) / "cache"))
os.environ.setdefault("CMB_RESULTS", str(Path(_TMP) / "results"))

from cmb import align, metrics                      # noqa: E402
from cmb.cache import cache_path, load, save        # noqa: E402
from cmb.data import split_items                    # noqa: E402
from cmb.extract import get_activations             # noqa: E402
from cmb.probes import CCSProbe, LinearDirection    # noqa: E402
from cmb.synthetic_data import synthetic_items      # noqa: E402
from common import PairRun                          # noqa: E402

PAIR = ("qwen-7b", "llama-8b")
N = 400


@pytest.fixture(scope="module")
def run():
    return PairRun(*PAIR, "truthfulqa", "final", "final", N, synthetic=True).build()


# -- plumbing ---------------------------------------------------------------

def test_cache_roundtrip():
    acts = get_activations("qwen-7b", "boolq", 100, synthetic=True)
    path = cache_path("synthetic-qwen-7b", "boolq", 100)
    assert path.exists(), "extraction must write a cache file"
    again = load(path)
    assert again.item_ids == acts.item_ids
    assert np.allclose(again.pos[acts.layers[0]], acts.pos[acts.layers[0]])
    assert again.n_layers == acts.n_layers


def test_cache_is_reused_not_recomputed():
    a = get_activations("qwen-7b", "rte", 60, synthetic=True)
    b = get_activations("qwen-7b", "rte", 60, synthetic=True)
    assert np.array_equal(a.p_yes, b.p_yes)


def test_paired_models_share_item_order(run):
    assert run.acts_a.item_ids == run.acts_b.item_ids
    assert np.array_equal(run.acts_a.labels, run.acts_b.labels)


def test_split_is_group_aware():
    items = synthetic_items("truthfulqa", 200)
    train, test = split_items(items, 0.4)
    tr = {i.group for i in train}
    te = {i.group for i in test}
    assert not (tr & te), "a claim's twin must not straddle the split"
    assert len(train) + len(test) == len(items)


def test_train_test_masks_are_disjoint(run):
    assert not (run.tr & run.te).any()
    assert (run.tr | run.te).all()


# -- metrics ----------------------------------------------------------------

def test_eight_cell_counts_and_row2():
    v1 = np.array([1, 1, 1, 0, 0, 1])
    v2 = np.array([1, 1, 0, 0, 1, 1])
    gt = np.array([1, 0, 0, 0, 1, 0])
    t = metrics.eight_cell(v1, v2, gt)
    assert sum(t.counts.values()) == len(gt)
    assert t.counts["TTF"] == 2                     # items 2 and 6
    assert t.row2_count == 2
    assert t.n_false == 4
    assert t.row2_rate == pytest.approx(0.5)
    assert t.router_fire_rate == pytest.approx(2 / 6)


def test_row2_mask_is_both_true_and_gt_false():
    v1 = np.array([1, 1, 0]); v2 = np.array([1, 1, 1]); gt = np.array([0, 1, 0])
    assert metrics.row2_mask(v1, v2, gt).tolist() == [True, False, False]


def test_wilson_interval_brackets_the_point_estimate():
    lo, hi = metrics.wilson_interval(5, 50)
    assert lo < 0.1 < hi and 0 <= lo and hi <= 1


def test_auroc_signed_is_orientation_free():
    s = np.array([0.1, 0.2, 0.8, 0.9]); y = np.array([1, 1, 0, 0])
    assert metrics.auroc_signed(s, y) == pytest.approx(1.0)
    assert metrics.auroc(s, y) == pytest.approx(0.0)


# -- alignment --------------------------------------------------------------

def test_linear_map_recovers_a_known_transform():
    rng = np.random.default_rng(0)
    Xa = rng.normal(size=(500, 20))
    W = rng.normal(size=(20, 12))
    Xb = Xa @ W + 0.01 * rng.normal(size=(500, 12))
    m = align.fit_map(Xa, Xb)
    assert m.r2(Xa, Xb) > 0.95


def test_cka_is_one_for_a_rotation():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 10))
    Q, _ = np.linalg.qr(rng.normal(size=(10, 10)))
    assert align.linear_cka(X, X @ Q) == pytest.approx(1.0, abs=1e-6)


def test_cka_is_low_for_independent_data():
    rng = np.random.default_rng(0)
    assert align.linear_cka(rng.normal(size=(300, 10)),
                            rng.normal(size=(300, 10))) < 0.3


# -- probes -----------------------------------------------------------------

def test_ccs_probe_recovers_a_planted_direction():
    rng = np.random.default_rng(0)
    d, n = 16, 400
    w = rng.normal(size=d)
    y = rng.integers(0, 2, n)
    base = rng.normal(size=(n, d))
    Xp = base + np.outer(2 * y - 1, w)
    Xn = base - np.outer(2 * y - 1, w)
    p = CCSProbe(d)
    p.fit(Xp, Xn, epochs=300, ntries=3)
    p.resolve_sign(Xp, Xn, y)
    assert metrics.auroc(p.belief(Xp, Xn), y) > 0.9


def test_linear_direction_separates_a_planted_class():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 8))
    y = (X[:, 0] + 0.2 * rng.normal(size=300) > 0).astype(int)
    d = LinearDirection.fit(X, y)
    assert metrics.auroc(d.score(X), y) > 0.9


# -- synthetic world as a positive control ----------------------------------

def test_probes_beat_chance_on_the_synthetic_world(run):
    gt = run.labels_test
    assert metrics.auroc(run.native_belief("a"), gt) > 0.7
    assert metrics.auroc(run.native_belief("b"), gt) > 0.7


def test_transport_preserves_most_of_the_probe(run):
    gt = run.labels_test
    native = metrics.auroc(run.native_belief("b"), gt)
    transfer = metrics.auroc(run.transported_belief("a"), gt)
    assert native - transfer < 0.15


def test_row_2_exists_and_errors_are_correlated(run):
    v1, v2, gt = run.verdicts("native")
    t = metrics.eight_cell(v1, v2, gt)
    assert t.row2_count > 0, "the synthetic world plants a shared confound"
    assert t.error_correlation > 0, "Row 2 is correlated error by construction"


def test_aligned_features_keep_the_contrast_mean(run):
    """The confound lives in what the pos/neg difference cancels.

    Four blocks of equal width: diff and mean for each model. If this ever
    collapses to differences alone, Experiment 3 is asking a question the
    feature space cannot answer.
    """
    X = run.aligned_features("test")
    d_a = run.acts_a.pos[run.layer_a].shape[1]
    assert X.shape[1] == 4 * d_a


def test_layer_specs_resolve_per_model():
    """'final' means each model's own last layer, not a shared index."""
    from cmb.config import resolve_layer

    assert resolve_layer("final", 32) == 32
    assert resolve_layer("final", 64) == 64
    assert resolve_layer("-3", 32) == 30
    assert resolve_layer(0.6, 40) == 24
    assert resolve_layer("24", 32) == 24


def test_per_model_layer_override():
    """--layer-a and --layer-b can pick different sites in the two models."""
    r = PairRun(*PAIR, "boolq", "final", 0.5, 120, synthetic=True).build()
    assert r.layer_a == r.acts_a.n_layers
    assert r.layer_b == int(round(r.acts_b.n_layers * 0.5))
    assert r.layer_a != r.layer_b


def test_requested_layer_is_extracted_not_approximated():
    """A cache miss on a layer must re-extract rather than snap to a neighbour."""
    from cmb.extract import get_activations

    acts = get_activations("qwen-7b", "rte", 60, synthetic=True, extra_layers=(7,))
    assert 7 in acts.layers


def test_verdict_modes_agree_on_shape(run):
    for mode in ("native", "a_to_b", "b_to_a"):
        v1, v2, gt = run.verdicts(mode)
        assert len(v1) == len(v2) == len(gt) == int(run.te.sum())
        assert set(np.unique(v1)) <= {0, 1}


def test_synthetic_belief_flips_on_prominent_falsehoods():
    from cmb.synthetic import SyntheticModel

    a, b = SyntheticModel("qwen-7b"), SyntheticModel("llama-8b")
    items = synthetic_items("truthfulqa", 400)
    false_items = [i for i in items if i.label == 0]
    shared = [i for i in false_items
              if a.internal_belief(i.item_id, 0, i.dataset) == 1
              and b.internal_belief(i.item_id, 0, i.dataset) == 1]
    assert len(shared) > 0.05 * len(false_items), (
        "both models should read some popular falsehoods as true")
