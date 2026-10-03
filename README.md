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
it implements. The write-up lives in [`paper/`](paper/) (skeleton, abstract, and
the formal treatment of the false-agreement algebra in `sections/theory.tex`, with the
framing and venue decisions in `paper/sections/PLAN.md`); the follow-on paper is
[`future_work/`](future_work/).

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
scripts/kickoff.sh --dry-run --pair cross-family --n 2000   # what it will do
scripts/kickoff.sh --pair cross-family --dataset truthfulqa --n 2000
```

`kickoff.sh` extracts what the pair needs, verifies the cache, then runs every
experiment in dependency order. The one-page plan — models, datasets, what each
step rules out — is [`docs/EXPERIMENT_PLAN.md`](docs/EXPERIMENT_PLAN.md).

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
  probes.py               CCS belief probe; mass-mean / LR baselines; Exp 3/4 direction
  align.py                ridge map A->B with held-out alpha selection; linear CKA
  metrics.py              sign-resolved AUROC, the 8-cell table, Row-2 rate,
                          the false-agreement identity + Frechet bounds + N_eff
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
  exp6_post_training.py   base vs instruct: does post-training break transport
tests/                    pytest, all synthetic
scripts/                  kickoff.sh (extract + run everything), extract.py
                          (warm the cache), run_all.sh, smoke.sh
docs/EXPERIMENT_PLAN.md   one page: models, datasets, the plan, the costs
docs/RUNBOOK.md           infra -> extraction -> experiments, end to end
paper/                    the write-up: tmlr.tex on the TMLR template, with
                          sections/{abstract,theory,method}.tex, refs.bib, PLAN.md
future_work/              Paper 2 (predicting reliability from CKA) and beyond
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
| Exp 1 | how big is Row 2, where does it sit between its bounds, and does it shrink with independence? | floor vs ceiling (§2.3) |
| Exp 3–4 | is Row 2 separable *and* generalizable? | either the result, or the honest cap |
| Exp 6 | does post-training degrade transport at the last layer? | fit maps at the probe's own layer instead |

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

**The Row-2 rate is always reported with its decomposition.** A rate on its own
cannot say whether a pair is safe because the probes are *accurate* or because
their errors are *independent* — and only the second improves with better model
choices. Experiment 1 therefore prints the two false-positive rates, the error
correlation `rho`, `rho` as a fraction of its feasible maximum (raw phi cannot
reach 1 unless the marginals match, so it is not comparable across pairs), the
Fréchet bounds the rate must lie inside, and the detectable coverage
`1 - FA/p_strong` that a disagreement router buys. The identity behind all of it
is `FA = p1*p2 + rho*sqrt(p1(1-p1))*sqrt(p2(1-p2))` — exact, not a bound. See
DESIGN.md §2.4 and `paper/sections/theory.tex`.

**Transport is priced, not assumed.** Gate B reports `eps`, the transported
probe's false-positive inflation at a matched positive rate, in the same units
as the rate it inflates; Experiment 1 then reports false agreement twice, native
and transported, and recomputes `rho` on the transported readouts rather than
inheriting the native value — the map is fitted from the source model's geometry,
so it can inflate the correlation as well as the error rate. On the synthetic
backend the planted map is recoverable almost exactly, so `eps ≈ 0` there by
construction; a real pair is where the number means something.

**Three probe types, one instrument.** `--probe ccs` (default, unsupervised — the
case that matters), `--probe mass-mean` (the strongest causal baseline in Marks &
Tegmark) and `--probe lr`, all on the same activations at the same layer. The
labeled ones are baselines that bound what supervision would buy; they never
substitute for CCS, since only CCS answers whether belief is recoverable without
labels.

**Both poolings are cached, and the choice is made afterwards.** One forward
pass yields a mean over the prompt *and* the final (Yes/No) token, so `--pooling
mean|last` is a free switch rather than another GPU rental. Mean is the default
because it is what the linear-alignment / embedding-API setting assumes; `last`
is what the probing literature (Marks & Tegmark) reads. Files written before
this carry only the mean and must be re-extracted with `--refresh` to get both.

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
`Qwen3-{4B,8B,32B}` (+ `Qwen3-8B-Base`), `Qwen3.8-27B`, `gemma-4-12B{,-it}` and
`Llama-3.1-8B-Instruct` — three families, one of them (Llama) deliberately
pre-reasoning-era. Every other family is a hybrid thinking model, and the
pipeline runs them with **no chat template and thinking off**, so the probe reads
*pre-reasoning* belief and `p_yes` stays a real Yes/No logprob (DESIGN.md §6.1).
Extraction is
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
