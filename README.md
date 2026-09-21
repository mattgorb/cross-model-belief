# Cross-Model Belief Probes & the False-Agreement Failure Mode

Implementation of the research design in [`DESIGN.md`](DESIGN.md): two models,
each with an unsupervised CCS belief probe, connected by a linear alignment map.
Their agreement is a label-free oversight signal, and the signal's structural
blind spot is **false agreement** — both probes read a claim as *true* while the
ground truth is *false* (**Row 2**). The router stays silent exactly where it is
wrong.

The bet: is Row 2 a characterizable, generalizable region of the aligned
activation space? If yes, the method can detect its own blind spot. If no,
cross-model belief agreement has an irreducible cap and we report it.

**Read `DESIGN.md` first** — it is the spec, and every script cites the section
it implements.

## Quickstart

```bash
pip install -r requirements.txt
scripts/smoke.sh                 # whole pipeline on synthetic data, ~1 min, no GPU
scripts/run_all.sh --synthetic   # same, with all output
```

For a real run — GPU box, then save the vectors once, then iterate on the
analysis for free — follow [`docs/RUNBOOK.md`](docs/RUNBOOK.md):

```bash
export CMB_CACHE=/mnt/data/activations
scripts/extract.py --models qwen-7b,llama-8b --datasets truthfulqa --n 2000
scripts/extract.py --list
scripts/run_all.sh --pair cross-family --dataset truthfulqa
```

`--synthetic` swaps in a generative activation model (`cmb/synthetic.py`) whose
latents contain a truth direction, a *shared* prominence confound, a shared
content subspace and a per-model private subspace. It manufactures Row 2 on
purpose, so it is a positive control for the code: if a script cannot recover
planted structure there, the bug is in the implementation, not in the science.
It says nothing about real models.

## Layout

```
DESIGN.md                 the spec — read first
cmb/                      library
  config.py               paths, seeds, model registry, layer selection
  data.py                 dataset loaders -> Item(claim, label); manifest-pinned
  prompts.py              claim text + CCS contrast pairs (P / not-P invariant)
  models.py               HF backend: hidden states + output P(Yes)
  tokalign.py             span alignment + tokenizer compatibility (ablation)
  cache.py / extract.py   activation cache keyed (model, dataset, item, layer, pos|neg)
  probes.py               CCS belief probe; supervised direction for Exp 3/4
  align.py                ridge map A->B with held-out alpha selection; linear CKA
  metrics.py              sign-resolved AUROC, the 8-cell table, Row-2 rate
  synthetic*.py           the no-GPU testbed
experiments/
  common.py               PairRun: the shared fit-on-train / score-on-test pipeline
  exp0_baseline_gate.py   activations vs logprobs (run first)
  gate_a_probe_truth.py   truth or confidence, on the confident slice
  gate_b_transport.py     does the map carry the probe
  exp1_eight_cell.py      the 8-cell table and the Row-2 rate (headline)
  exp2_bidirectional.py   transport-robust hard-core false agreement
  exp3_separability.py    is Row 2 linearly separable (the bet)
  exp4_generalization_matrix.py   5x5 leave-one-dataset-out + held-out model pair
  exp5_layer_sweep.py     depth sweep of gates, Row 2, separability
tests/                    pytest, all synthetic
scripts/                  extract.py (warm the cache), run_all.sh, smoke.sh
docs/RUNBOOK.md           infra -> extraction -> experiments, end to end
reference/                the original single-file scripts this was refactored from
infra/                    Terraform for spot GPU instances
results/                  *.json per experiment + your RESULTS.md
```

## Reading a run

Each script prints a verdict and writes `results/<name>.json`. The gates exit
non-zero on failure and `run_all.sh` stops there, because a failed gate makes
everything downstream meaningless:

| step | question | failing means |
|---|---|---|
| Exp 0 | do activations beat output logprobs? | the internal-space framing is unjustified |
| Gate A | does the probe read truth, not confidence? | stop — nothing downstream is meaningful |
| Gate B | does the map carry the probe? | single-model story only; drop the cross-model framing |
| Exp 1 | how big is Row 2, and does it shrink with independence? | floor vs ceiling (§2.3) |
| Exp 3–4 | is Row 2 separable *and* generalizable? | either the result, or the honest cap |

## Choices worth knowing about

**The probe site defaults to the final layer**, set per model with `--layer`,
`--layer-a` and `--layer-b`. Each takes `final` (the default), a depth fraction
(`0.6`), a negative index (`-3`), or an absolute index (`24`). Mean-pooling the
last hidden state is what an embedding API returns, which is the setting the
linear-alignment result was established in; fractions keep the site comparable
across models of different depth, absolute indices do not. A requested layer
missing from the cache triggers re-extraction rather than snapping to a
neighbour, so the flag always means what it says.

```bash
python3 experiments/gate_b_transport.py --pair cross-family              # final/final
python3 experiments/gate_b_transport.py --layer-a final --layer-b 0.6    # per model
python3 experiments/exp5_layer_sweep.py --sweep 0.4,0.6,0.8,final
```

**Tokenizer differences need no special handling on the default path.** Each
claim becomes one pooled vector per model, so the map is fitted on item-level
pairs exactly as in the embedding-model setting — token counts never enter.
`cmb/tokalign.py` provides character-span alignment (the union-of-boundaries
method from the HELIX `lm_pertoken.py` work) as an *ablation*, for checking
that a span-fitted map reaches the same Gate B verdict, plus the HELIX
tokenizer-compatibility score, which is worth reporting next to CKA because it
predicts whether transport will work at all.

**The confident slice is the whole test (Gate A).** A probe that only separates
truth where output confidence already does is reading confidence. Gate A scores
on the top-50% confident items, where the natural high-confidence errors live.

**Probes and maps are refit per cell in Exp 4.** Probe, map and false-agreement
direction are three separately fitted objects. Only the *direction* crosses a
dataset boundary; otherwise a failed cell is unattributable.

**Features keep the contrast mean, not just the difference.** The belief probe
reads `pos - neg`, which cancels whatever the two halves share — including the
prominence confound that creates Row 2. Experiment 3's feature space therefore
carries both the difference and the mean; a difference-only space would be
engineered not to show the thing being looked for.

**Sign resolution is explicit.** CCS has no orientation, so one bit of label
information is unavoidable. It is spent on the train split in
`CCSProbe.resolve_sign` rather than hidden inside a `max(a, 1-a)` at scoring
time. `auroc_signed` still exists for diagnostics and is labeled as such.

**Splits are group-aware.** A TruthfulQA true/false pair or an MMLU
true/chosen-wrong pair never straddles the train/test boundary.

**Row 2 is small by construction** — it shrinks exactly where the system works
best. Rates are reported with Wilson intervals and the scripts warn below ~20
items. See the data-scarcity mitigations in DESIGN.md §4.

## Real runs

Open-weight models only (activations and logprobs are both required):
`Qwen2.5-{1.5B,7B,32B}-Instruct` and `Llama-3.1-8B-Instruct`. Extraction is
fp16 — never 4-bit, which perturbs exactly what the probe reads. All swept
layers come out of one forward pass and are cached together, so the expensive
32B pass runs once; point `CMB_CACHE` at persistent storage (the gp3 volume in
`infra/`) before starting it.

```bash
CMB_CACHE=/mnt/data/activations scripts/run_all.sh --pair same-family
python3 experiments/exp1_eight_cell.py --pair all --dataset truthfulqa --n 2000
```

`infra/main.tf` brings up a spot GPU box: `g5.2xlarge` for ≤8B, `g5.12xlarge`
for 32B fp16 sharded.

## Troubleshooting

*`RuntimeWarning: divide by zero encountered in matmul` on macOS* — spurious.
NumPy's Accelerate BLAS backend emits it for ordinary finite matmuls; the values
are correct. Reproduce with a bare `np.random.randn(5,24) @ np.random.randn(24,128)`.
Run with `python3 -W ignore` if it is noisy.

*Geometry of Truth fails to load* — the loader tries the HF mirrors. If they are
unreachable, clone `github.com/saprmarks/geometry_of_truth` and adapt
`_load_geometry_of_truth`, or work with TruthfulQA plus the Burns suite.

*Row-2 counts in single digits* — expected on small N. Raise `--n`, use
`--dataset truthfulqa` (confound-rich by construction), and read DESIGN.md §4.
