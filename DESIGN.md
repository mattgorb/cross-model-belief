# DESIGN — Cross-Model Belief Probes and the False-Agreement Failure Mode

**Status:** research design (pre-experiment). Target: a paper.
**Author:** Matt Gorbett (with collaborator Suman Jana, Columbia).
**Last updated:** 2026-09-21.

This document is the self-contained specification for the project. It states the
idea, the failure mode we care about, what is and isn't novel, and the exact
experiments to run. It is written to be handed to an implementation agent or a
new collaborator with no other context.

---

## 0. One-paragraph summary

Unsupervised probes (CCS-style) read a linear "belief" direction out of a
model's activations — an internal true/false verdict on a claim. A linear
alignment map lets a probe *characterized* on one model be transferred to and
read on another model. Put two independently-trained models together, each with
a belief probe, and their agreement/disagreement is a cross-model oversight
signal — the internal-activation version of cross-model disagreement as a
label-free signal. The signal's structural blind spot is **false agreement**:
both models' probes read a claim as *true* while the ground truth is *false*
(call it **Row 2** of the joint table). No consistency-based method can catch
this from the inside, because agreement is the only thing it has and here
agreement is wrong. The research question is whether Row 2 is a *characterizable,
generalizable region* of the aligned activation space — i.e., whether
"untrustworthy agreement" has a stable signature that transfers across datasets
and model pairs. If yes, we have a label-free detector of the oversight blind
spot itself. If no, cross-model belief agreement is fundamentally capped and we
should say so.

---

## 1. Background and prior work

### 1.1 The author's prior results this builds on
- **Cross-model disagreement as a label-free reward signal** (CMP/CME, COLM
  2026, arXiv 2603.25450) and **CME-GRPO** (arXiv 2605.29009). Establishes that
  disagreement *between models* carries usable signal without labels — but in a
  cooperative regime, and as a *reward*, not a *certificate*.
- **Linear cross-model alignment** ("Ventriloquist LLMs" / HELIX, Gorbett &
  Jana, arXiv 2603.18908). Across 34 model pairs, supervised linear alignment
  preserves classification and OOD-detection performance across embedding-model
  pairs with minimal degradation; linear CKA 0.595–0.881 across independently
  trained models. Tokenizer compatibility and model size are the primary
  determinants of alignment success. Originally paired with homomorphic
  encryption (TenSEAL/CKKS) for privacy-preserving inference.

This project reads the disagreement signal from **activations** rather than
output logprobs, and uses the **linear map** as the transport mechanism between
models.

### 1.2 The field this sits in
- **CCS — Contrast-Consistent Search** (Burns et al. 2022, arXiv 2212.03827).
  Unsupervised probe: build contrast pairs `x+`/`x-` (a claim asserted true vs
  false), fit a linear probe `p = σ(wᵀφ + b)` with
  `loss = consistency[p(x+) − (1 − p(x−))]² + confidence·min(p(x+), p(x−))²`.
  No labels. Recovers the model's internal *belief* about a claim.
- **CCS finds prominent features, not necessarily truth** (Farquhar et al. 2023,
  arXiv 2312.10029). Two theorems show the consistency structure is not specific
  to knowledge — arbitrary binary features of the questions can achieve optimal
  CCS loss. Empirically, when a competing prominent feature is injected, CCS
  follows the distractor. **This is the central weakness and it bites within a
  single benchmark, not only across benchmarks.**
- **Geometry of Truth** (Marks & Tegmark 2024). Curated true/false statement
  datasets built to study the truth direction directly — the closest existing
  data to what this project needs.
- Related CCS follow-ups: contrast pairs drive most of CCS's performance
  (Emmons; LessWrong); optimization-target ablations (Fry et al., arXiv
  2311.00488); contrastive-eigenproblem view (arXiv 2511.02089).

### 1.3 The scalable-oversight framing (and its honest limit)
Cross-model agreement lowers error by ensembling and *routes* attention via
disagreement. It does **not** solve scalable oversight in the strong sense
(supervising a model more capable than the overseer). N sub-frontier models
voting is a better classifier, not oversight of a super-overseer system. We
frame this project as **reliability infrastructure with a quantified
false-agreement rate**, and as a *router* (disagreement says where to spend
labels), not as an oracle that certifies truth. Overclaiming "agreement →
truth" is the one thing that will not survive review.

---

## 2. The core idea

Three components, in dependency order.

1. **Belief probe.** For a target model, fit an unsupervised CCS probe on its own
   activations over contrast pairs. The probe outputs the model's internal
   true/false verdict for a claim. (Known caveat: this may be a *prominence*
   direction, not a *truth* direction — Gate A below is designed to catch that.)

2. **Cross-model transport.** Fit a linear map `M: A → B` on paired activations
   (same inputs through both models). A probe *characterized* on model A — where
   we can validate it against labels/interventions — can then be read on model B
   through the map. The value is amortization of characterization: understand the
   instrument once, reuse it on models where native validation is impossible.
   **Caveat:** transport reaches only as far as the aligned subspace; failure
   modes living in A-specific structure the map doesn't carry do not transfer,
   and silently. Measuring *how far* the characterization survives transfer is
   part of the contribution, not an afterthought.

3. **Cross-model agreement as oversight.** Two independent models, each with a
   belief probe (native or transported). For each claim, each model votes
   true/false. Agreement/disagreement is the oversight signal. Disagreement is a
   **router**: flag the item, spend an expensive labeled check. This is the
   activation-space form of the author's disagreement work.

### 2.1 The failure mode we actually study: false agreement (Row 2)

With two models and ground truth, every item falls in one of eight cells:

| # | m1 says | m2 says | ground truth |
|---|---------|---------|--------------|
| 1 | True    | True    | True   (good, nothing to do) |
| **2** | **True** | **True** | **False  ← THE FAILURE WE CARE ABOUT** |
| 3 | True    | False   | True   (router fires) |
| 4 | True    | False   | False  (router fires) |
| 5 | False   | False   | True   (router fires) |
| 6 | False   | False   | False  (good) |
| 7 | False   | True    | True   (router fires) |
| 8 | False   | True    | False  (router fires) |

- Rows 3–8 involve disagreement → the router fires → they are handled.
- Row 1, Row 6: correct agreement → fine.
- **Row 2: confident, correlated, wrong agreement.** The router stays silent.
  Both probes say true, nothing flags it. This is the case cross-model
  consistency *cannot* see, and it is made *more* likely, not less, when both
  models share a prominence confound (e.g., a popular falsehood phrased in its
  familiar form fools both probes identically) or share a training-data error.

Row 2 is the ceiling on the whole approach. The project is about measuring it and
asking whether it is *detectable* without labels.

**Two causes, and they must be kept apart.** (a) The probe faithfully reads a
belief the two models *share*, and that belief is wrong — a real limit of
cross-model oversight. (b) The probe *fails*, following familiarity or prominence
instead of truth ("bats are blind") — an instrument problem, not an oversight
result. Gate A is the first line of defence; separating the two properly needs an
**attribution control**: adversarial items where the confound and the truth come
apart (TruthfulQA above all, plus familiar-vs-defamiliarized paraphrases of the
same false claim). If Row-2 items lie along the prominence direction recovered
from those pairs, that is evidence for (b); if they do not, for (a).

### 2.2 The research bet

> Is Row 2 a **characterizable, generalizable direction/region** in the aligned
> activation space?

- **If Row 2 separates and the separation generalizes** (across held-out
  datasets and held-out model pairs): we have a label-free detector of
  untrustworthy agreement — the oversight method can detect its own blind spot.
  This is the foundable/publishable contribution.
- **If Row 2 does not generalize** (each false agreement is idiosyncratic): the
  method is ensembling with an irreducible blind spot; we report the cap
  honestly (still publishable, less exciting).

### 2.3 Floor vs. ceiling (how to read the headline number)
- **Floor (good):** Row-2 rate is low and *drops* as models get more independent
  (cross-family < same-family). Agreement is trustworthy and improves with better
  model choices; the propagated error is boundable.
- **Ceiling (bad):** Row-2 rate stays high regardless of independence — shared
  training errors keep both probes wrong together. Irreducible blind spot.

### 2.4 The algebra of false agreement

Restrict to the GT-false items and let `p_i = P(probe i says True)` be each
probe's false-positive rate. False agreement is `FA = P(both say True)`, and with
`rho` the phi correlation between the two false-positive indicators:

```
FA = p1*p2 + rho * sqrt(p1(1-p1)) * sqrt(p2(1-p2))
```

This is an **identity**, not an approximation: the covariance of two indicators is
`P(both) - p1*p2`. Its content is that FA is governed by how *correlated* the
errors are, not only by how accurate each probe is — two accurate probes with
correlated errors can be worse than two mediocre probes with independent ones,
which is why overseers cannot be selected on accuracy alone.

Reference points, with the running example `p1 = 10%`, `p2 = 5%`:

| point | FA | reading |
|---|---|---|
| independence (`rho = 0`) | `p1*p2` = 0.5% | errors coincide by chance only — a *reference point*, not a bound |
| maximal overlap (upper) | `min(p1,p2)` = 5% | every error of the better probe is an error of the worse one; the second reader adds nothing (self-oversight) |
| disjoint errors (lower) | `max(0, p1+p2-1)` = 0 | needs anti-correlated errors; unrealistic for models sharing pretraining data |

These are the **Fréchet bounds**, so `rho` does *not* span `[-1, 1]` unless
`p1 = p2`: in the example, maximal overlap is `rho ≈ 0.69` and disjointness is
`rho ≈ -0.08`. Raw phi is therefore not comparable across pairs with different
marginals — report `rho` normalized by its feasible extreme, or report FA directly
next to its two bounds. Both are in `cmb.metrics.false_agreement`.

The same 2×2 table gives the positive claim and the blind spot at once. On the
false items the mass splits into agree-wrong (FA), disagree, agree-right. The
disagree cell is what the router can flag; FA is what it cannot. The stronger
probe's **detectable coverage** is `1 - FA/p_strong` — 90% under independence, 0%
under maximal overlap. One experiment reports both. Caveat that survives all of
it: disagreement says one of the two readers is wrong, never which one.

**Transport-adjusted.** When probe 2 is carried over the map instead of fitted
natively, `p2' = p2 + eps`, where `eps` is the Gate B degradation measured as
realized false-positive inflation at a matched positive rate (same units as
`p2`, so it substitutes straight into the identity). Report FA **twice**, native
and transported; the gap is the cost of the map. Transport plausibly inflates
`rho` as well, since the map is fitted from the source model's geometry, so
recompute `rho` on the transported readouts rather than reusing the native value.
Experiment 1 prints both rows; Gate B prints `eps`.

**Bidirectional intersection** (Experiment 2): compute the FA set under A→B and
under B→A, and treat the intersection as the robust set. Items flagged in one
direction only are transport noise; report the size of that gap.

**N-model remark.** For N overseers with equal error variance and mean pairwise
correlation `rho_bar`, the effective number of independent overseers is
`N_eff = N / (1 + (N-1) rho_bar) -> 1/rho_bar`. This is the **correlation floor**:
once `rho_bar > 0`, adding models stops helping — `rho_bar = 0.8` caps the pool at
1.25 effective readers, and it takes `rho_bar ≈ 0.3` for 20 models to be worth
about 3. This is standard ensemble theory (bias–variance–covariance); cite it,
do not claim it. Implemented as `cmb.metrics.n_eff`, developed properly in
`future_work/paper2-representational-similarity.md`.

**Honest framing.** The identity and the floor are borrowed. What is ours is
applying them to probe-based cross-model oversight, adding the transport term, and
measuring `rho` directly on internal readouts. Keep this light in the paper —
framing that motivates the measurement, not a theorem. The formal version is
`paper/sections/theory.tex`.

---

## 3. What is novel vs. known (be honest in the paper)

**Known / not a contribution:**
- CCS and its prominence-vs-truth weakness (Burns; Farquhar).
- Linear cross-model representational alignment (author's own prior work).
- "Disagreement finds errors" at the output level (author's own prior work).
- Cross-model / mutual verification *as a concept*.

**Candidate contributions (what to defend):**
1. **Measuring the false-agreement (Row-2) rate as a function of cross-model
   independence**, in activation space via the map — same-family vs cross-family.
   Nobody has this number for the internal-belief setting.
2. **Testing whether the Row-2 region is separable and generalizable** — a
   label-free signature of untrustworthy agreement, with transfer measured across
   datasets and model pairs. This is the novel unit; the mechanism (two probes
   agreeing) is not.
3. **Reporting false agreement natively and under transport**, with the map's
   cost `eps` measured in the same units as the false-positive rate it inflates,
   and `rho` recomputed on the transported readouts rather than inherited from
   the native fit (§2.4).
4. **Quantifying how much of a probe's *characterization* survives linear
   transport** — the gap between transferred failure-profile and native
   failure-profile, i.e., the instrument's cross-model blind spot, measured.

**Weakly novel alone (do not lead with):** "read disagreement from activations
instead of logprobs." Only interesting if it *beats* the output-space baseline
(the author's existing paper is the baseline) — see Experiment 0.

---

## 4. Experiments

All experiments cache activations to disk keyed by `(model, dataset, item_id,
layer, pos/neg)` so the expensive 32B pass runs once. Seed everything. Extract
activations in **fp16** (never 4-bit — quantization perturbs exactly what the
probe reads). Default probe layer = `0.6 × depth`; sweep later.

**Probe types — same data, same activations, same layer** (`--probe`):
`ccs` (unsupervised, the case that matters), `mass-mean` (the strongest causal
baseline in Marks & Tegmark), `lr` (cheap, expected to be largely redundant with
mass-mean — report it to show that rather than assert it). Optionally **VINC-S**,
which varies the *amount* of supervision inside one method and so gives a
supervision axis instead of three unrelated probes. **Native first:** fit native
probes on each model wherever labels allow and use transport only where they do
not; transport is a measured cost, not the default path. The **weak→strong**
direction (fit where labels exist, read where they do not) is where `eps` is
largest, and reporting it honestly is a contribution even if it is large.

### Experiment 0 — Is the activation signal even worth it? (baseline gate)
Compare activation-space disagreement/belief against **output-logprob**
disagreement (the author's existing method) on the same items. If activations do
not beat logprobs, the internal-space framing is not justified — report and stop
escalating. Cheap; run first.

### Gate A — Does the probe read truth, not confidence?
For each model, native CCS probe. Metric: AUROC of the belief score predicting
**correctness**, computed **on the confident slice** (top 50% of items by the
model's own output confidence). Baseline: raw output confidence predicting
correctness on the same slice.
- **Pass:** probe AUROC > confidence-baseline AUROC on the confident slice (the
  confident-*wrong* items are the whole test — natural high-confidence errors).
- **Fail:** probe ≤ baseline on the slice → the probe is reading confidence, not
  truth. Report and halt escalation; no downstream result is meaningful.

### Gate B — Does the map carry the probe?
Fit probe on A, transport to B via `M`, test on B. Report transfer AUROC and
linear CKA(A, B).
- **Pass:** transfer AUROC within ~0.05 of native-B AUROC.
- **Fail:** transfer collapses → the cross-model framing is dead weight for this
  pair; fall back to native probes and say so.

### Experiment 1 — The 8-cell table and the Row-2 rate (headline)
On a labeled test split, fill all eight cells for each model pair.
- **Headline metric:** `Row-2 rate = P(m1 says True AND m2 says True | GT = False)`.
- Report per pair, **same-family (Qwen-7B ↔ Qwen-32B)** vs **cross-family
  (Qwen-7B ↔ Llama-8B)**. Does Row 2 drop cross-family? (floor vs ceiling)
- Also report the **§2.4 decomposition**: each probe's false-positive rate, the
  error correlation `rho` and `rho` normalized by its feasible maximum, the
  Fréchet bounds, and the detectable coverage `1 - FA/p_strong`. The rate alone
  cannot distinguish "accurate probes" from "independent probes", and that
  distinction is the finding.
- And report FA **twice**, native and transported, with `eps` and the inflated
  `rho` — the gap is the price of the map (§2.4).

**Prediction, written down before the run:** same-family pairs sit near the
Fréchet upper bound; cross-family pairs show partial decorrelation but do not
reach independence, because they share web-scale pretraining data.

### Experiment 2 — Bidirectional hard-core false agreement
Compute the Row-2 set using `A→B` transport and again using `B→A` transport.
Report the **intersection** — false agreements robust to transport direction.
This is the hardest, most trustworthy-looking wrong set and the right target for
Experiment 3.

### Experiment 3 — Is Row 2 separable and does it generalize? (the bet)
Within the `{both say True}` subset, fit a **linear classifier** to separate
`GT-True` from `GT-False` in the aligned activation space. This asks: does
"true-because-true" look different internally from "true-because-shared-confound"?
- Within-dataset train/test split first (does it separate at all?).
- Then generalization — see Experiment 4.

**Data-scarcity note:** Row 2 shrinks as independence and N grow — the thing we
characterize is rarest exactly where the system is best. Mitigations:
(a) characterize at **small N on confound-rich data** (TruthfulQA, popular
falsehoods), then transfer the direction to larger-N settings;
(b) **enrich Row 2 deliberately** with popular-falsehood datasets and adversarial
paraphrase pairs (familiar vs. defamiliarized phrasing of the same false claim);
(c) borrow structure — if Row-2 errors are prominence-driven, get the
**prominence direction** from familiar-vs-defamiliarized pairs directly and test
whether Row-2 cases lie along it, rather than fitting the rare class from scratch.

### Experiment 4 — Generalization matrix (the honesty test)
Leave-one-dataset-out, **full 5×5 AUROC matrix** for the false-agreement
direction. Fit the direction on dataset *i*, test on dataset *j*, all pairs.
- Diagonal = in-distribution (expected high, near-useless).
- **Off-diagonal is the result.** The *pattern* across the matrix (graceful
  degradation with domain distance vs. cliff at domain boundaries vs. flat) is
  the scientific finding.
- **IMDB → TruthfulQA is the designated hardest cell and the headline honesty
  check** (sentiment confound → factual misconception; maximally different).
- Also test transfer to a **held-out model pair**.

**Attribution control (critical).** Probe, map, and false-agreement direction are
three separately fitted objects. To make Experiment 4 interpretable, fit probes
and maps **per cell on a train split**, so the *only* object transferred across
datasets is the **false-agreement direction**. Otherwise a failure is
unattributable (was it the probe, the map, or the direction that didn't
generalize?).

### Experiment 5 — Layer sweep
Repeat Gate A, Experiment 1, and Experiment 3 at several layers (e.g.,
`{0.4, 0.5, 0.6, 0.7, 0.8} × depth`). Generalizing truth structure and prominence
artifacts often live at different depths; nearly free once activations are cached.

### Experiment 6 — Post-training and transport (control)
One model, two checkpoints (base and instruct): same family, size and tokenizer,
so the only variable is post-training. Fit the probe on both, transport between
them, and compare the transport gap at the **last layer** against a mid-depth
site. Marks & Tegmark probe middle layers (layer 13 of 40 on LLaMA-13B) and do not
report the last layer; the sleeper-agent probes are also middle-layer, while HELIX
and the platonic-representation argument point at the last layer for the *map*.
If post-training rewrites the last layer more, the `final` default is buying map
quality at the cost of transport stability — in which case fit maps at several
layers so each map matches its probe's layer. Base↔instruct is also the most
favourable transport there is, so its `eps` is a lower bound on what every other
pair pays.

---

## 5. Datasets

All need ground-truth true/false labels. Build CCS contrast pairs (claim asserted
true vs. false; negate the *claim* to preserve the P/¬P invariant).

| dataset | role | why |
|---|---|---|
| **Geometry of Truth** (Marks & Tegmark) | primary | native curated T/F statements; the canonical truth-direction data |
| **TruthfulQA** | Row-2 enrichment | built from popular misconceptions → natural false-agreement distribution |
| **Burns CCS suite subset**: BoolQ, IMDB, RTE | matrix cells | the standard CCS-replication data; IMDB is the sentiment-confound extreme |
| **MMLU (hard subjects)** | general coverage | see conversion note |

Hold **one dataset out entirely** for the generalization test.

**MMLU conversion note.** MMLU is 4-way; CCS is binary. Do **not** expand to
1-true/3-false (skews 3:1 and muddies Row 2). Keep **two claims per question**:
the true claim and the model's *chosen wrong* claim. The chosen wrong answer is
the model's actual error, which is what Row 2 is about. Prefer native T/F
datasets (Geometry of Truth, TruthfulQA) as primary; use MMLU only for coverage.

---

## 6. Models

Open-weight only (activations *and* logprobs are both required), three families,
one variable per pair wherever the model zoo allows it.

| key | HF id | family | depth | notes |
|---|---|---|---|---|
| `qwen3-4b` | `Qwen/Qwen3-4B` | qwen | 36 | held-out pair |
| `qwen3-8b` | `Qwen/Qwen3-8B` | qwen | 36 | **the anchor** |
| `qwen3-32b` | `Qwen/Qwen3-32B` | qwen | 64 | same-family scale |
| `qwen3-8b-base` | `Qwen/Qwen3-8B-Base` | qwen | 36 | Exp 6 control |
| `qwen38-27b` | `Qwen/Qwen3.8-27B` | qwen | 64 | newest Qwen generation; native VLM |
| `gemma4-12b` | `google/gemma-4-12B-it` | gemma | 48 | third family |
| `gemma4-12b-base` | `google/gemma-4-12B` | gemma | 48 | second Exp 6 control |
| `llama-8b` | `meta-llama/Llama-3.1-8B-Instruct` | llama | 32 | non-reasoning; gated |
| `llama-8b-base` | `meta-llama/Llama-3.1-8B` | llama | 32 | gated |

Why this set: **Qwen 3** is the only current family shipping small dense
checkpoints *with* matching bases, so the anchor and the ladder live there.
**Qwen 3.8** is the newest generation but its smallest open dense model is 27B
(no 4B/8B/14B this generation), so it enters as a cross-generation pair rather
than as the anchor. **Gemma 4** gives a third family that is reasoning-era like
Qwen 3 and ships base + instruct, which makes it the cleanest cross-family
contrast. **Llama 3.1-8B** is still the newest *dense* small Llama — Llama 4 is
MoE with no 8B dense — and it earns its place precisely by being
pre-reasoning-era: it is the only non-reasoning point on the board.

| pair | `--pair` | the variable | confound |
|---|---|---|---|
| qwen3-8b ↔ qwen3-32b | `same-family` | scale | none (shared tokenizer) |
| qwen3-8b ↔ gemma4-12b | `cross-family` | family | 8B vs 12B |
| qwen3-8b ↔ llama-8b | `cross-era` | family **and** reasoning era | size-matched, so read it as an upper bound on decorrelation |
| qwen3-8b ↔ qwen38-27b | `cross-generation` | generation | scale too |
| qwen3-4b ↔ gemma4-12b | `held-out` | — | never used to fit anything |
| qwen3-8b-base ↔ qwen3-8b | `post-training` | post-training only | Exp 6 |

Expect lower CKA across tokenizers; per the author's HELIX finding, tokenizer
compatibility and size gap drive alignment success, so read a weak cross-family
result as *possibly the map, not the probe* — Gate B disambiguates.

**Hardware.** Only the 4B/8B models fit an A10G (24GB) in fp16. `gemma4-12b`
(~24GB) is borderline and `qwen3-32b` / `qwen38-27b` (~54–64GB) are not: those
need the `g5.12xlarge` (4× A10G).

### 6.1 Reasoning models — what we do and why

Every family here except Llama is a **hybrid thinking model**: thinking is on by
default in Qwen 3.8 and configurable in Qwen 3 and Gemma 4. That is a real design
question for a belief probe, not a detail.

**The policy: no chat template, no thinking, everywhere** (`USE_CHAT_TEMPLATE =
False`, `THINKING = False` in `cmb/config.py`). Extraction uses raw
completion-style prompts, so no `<think>` block is ever injected or generated.
Three reasons:

1. **It is the only reading comparable across the board.** A non-reasoning model
   has no trace to read; holding the prompt format fixed keeps the pair contrast
   about the models rather than about their chat templates.
2. **It keeps `p_yes` meaningful.** Exp 0's baseline and Gate A's slice selector
   are a single-token Yes/No logprob. With thinking on, the immediate next token
   is the start of a trace, not a verdict, and that baseline disappears.
3. **It is deterministic and cheap.** One forward pass per half. Generating a
   trace is variable-length sampling — hours of GPU, and a different number every
   seed.

**What this means for the claim:** the probe reads **pre-reasoning belief** — what
the model represents about the claim before any serialized deliberation. That is
the right object for this paper (the probe is meant to see what the model does not
say), but it must be stated as such, because a reasoning model's *considered*
answer can differ from it. Say "pre-reasoning internal belief" in the paper, not
"the model's belief".

**One consequence for Exp 6.** In a hybrid model, post-training is also where
thinking behaviour is installed, so base ↔ instruct now measures "did reasoning
training move the geometry", not just instruction tuning. Worth saying out loud
rather than treating the control as clean.

### Experiment 7 — belief after reasoning (specified, not yet implemented)

The open question §6.1 leaves: does false agreement change once the models
actually reason? Design, for when it is worth the GPU time:

- Generate a thinking trace per item (thinking on, temperature 0 where the family
  allows it), then pool activations over the **final answer span** instead of the
  prompt, and refit probes there.
- Compare FA and `rho` against the pre-reasoning numbers on the same items. Both
  directions are plausible and it is genuinely open which wins: independent traces
  may *decorrelate* errors (different reasoning paths, so FA drops), or both
  models may reason their way into the same familiar misconception, *inflating*
  `rho` — the confound-driven Row 2 getting worse, not better, with deliberation.
- Cost is the blocker: variable-length generation for every item and every model,
  against one forward pass today. Needs a generation path in `cmb/models.py`,
  trace caching, and a decision about sampling variance (the same item can land in
  Row 2 on one trace and not on another, which makes FA itself stochastic).

---

## 7. Metrics and decision rules

- **AUROC (sign-resolved).** CCS sign is arbitrary; take the orientation that
  predicts better. Report on full set and on the confident slice.
- **Row-2 rate** = `P(both True | GT False)`. Primary headline, reported with a
  Wilson interval.
- **The §2.4 decomposition**: `p1`, `p2`, `rho`, `rho` over its feasible maximum,
  the Fréchet bounds, and detectable coverage `1 - FA/p_strong`. A rate without
  its bounds does not say whether the pair is near independence or near maximal
  overlap.
- **`eps`**, the transport degradation as false-positive inflation at matched
  positive rate, and FA reported native *and* transported.
- **Error correlation** between probe verdicts over all items (drives
  amplification) — distinct from `rho`, which is measured on the false items only.
- **CKA(A,B)** as map-quality context.
- **Separability AUROC** for the false-agreement direction, in-distribution and
  transferred (Experiments 3–4).

**Overall decision tree:**
1. Exp 0 fails → internal-space framing unjustified; reconsider.
2. Gate A fails → probe reads confidence, not truth → stop; nothing downstream is
   meaningful.
3. Gate A passes, Gate B fails → probe real, transport dead → single-model story
   only; drop the cross-model framing.
4. Gates pass, Row 2 separable + generalizes (Exp 3–4) → **the result**: a
   label-free detector of untrustworthy agreement. Write the paper.
5. Gates pass, Row 2 does *not* generalize → report the honest cap on cross-model
   belief agreement.

---

## 8. Threats to validity (state these in the paper)
- **Prominence, not truth.** Gate A on the confident-wrong slice is the guard; if
  it barely passes, every downstream claim is fragile.
- **Contamination / teaching-to-the-test.** If the benchmark is public and the
  metric matters, a participant can contaminate or tune the probe to read
  "honest." Rotating/secret splits fight this but tension with the
  "agreed public benchmark" simplification — acknowledge it.
- **Transport blind spot.** Characterization survives transport only within the
  aligned subspace; Experiment 4 measures the shortfall but cannot eliminate it.
- **Independence of process ≠ independence of knowledge.** Independently
  generated data breaks pipeline collusion but not shared training-data errors;
  Row 2 partly reflects the latter and cannot be removed by careful pipelines.
- **Pre-reasoning belief, not belief.** Every family but Llama is a hybrid
  thinking model and we probe with thinking off (§6.1). The claim is therefore
  about what a model represents *before* deliberating, which is the right object
  for a probe but is not the same as its considered answer. Exp 7 is the test we
  have not run, and until it is run this is a scope limit on the result, not a
  footnote.
- **Pre-reasoning belief, not belief.** Every family but Llama is a hybrid
  thinking model and we probe with thinking off (§6.1). The claim is therefore
  about what a model represents *before* deliberating, which is the right object
  for a probe but is not the same as its considered answer. Exp 7 is the test we
  have not run, and until it is run this is a scope limit on the result, not a
  footnote.
- **Not scalable oversight in the strong sense.** This is reliability/ensembling
  + routing. Frame accordingly.

---

## 9. Infrastructure

- HuggingFace `transformers`, fp16, `device_map="auto"`.
- **Cache activations** to `activations_cache/` keyed by
  `(model, dataset, item_id, layer, pos|neg)` as `.npy`/parquet. Re-runs and the
  32B pass must never recompute.
- AWS (see `infra/`): dev + ≤8B on `g5.2xlarge` (1× A10G, ~$0.40–0.60/hr spot);
  32B fp16 sharded on `g5.12xlarge` (4× A10G, ~$2–2.50/hr spot). Spot + a large
  gp3 EBS volume holding the HF cache and activations; detach/reattach the volume
  across spot instances to preserve the expensive 32B extraction. Do **not**
  4-bit quantize for activation extraction.

---

## 10. Suggested milestones
1. Plumbing: model loader, contrast-pair builder, activation cache, CCS fit,
   ridge map, AUROC/CKA. (`src/` already has `probe_validation.py` and `w2s.py`
   as reference implementations to refactor from.)
2. Exp 0 + Gate A on Qwen-7B and Llama-8B, Geometry of Truth + TruthfulQA.
3. Gate B + Experiment 1 (8-cell table, Row-2 rate) both pairs.
4. Experiment 2 (bidirectional intersection).
5. Experiment 3 (separability), then Experiment 4 (5×5 matrix + held-out pair).
6. Experiment 5 (layer sweep) and Experiment 6 (base vs instruct control).
7. Probe-type comparison (`--probe ccs|mass-mean|lr`) at the chosen layer.
8. Write `results/RESULTS.md` classifying the outcome per the decision tree, then
   the paper itself — skeleton, theory section and writing plan are in `paper/`.

---

## 11. References
- Burns et al. 2022, *Discovering Latent Knowledge Without Supervision* (CCS), arXiv 2212.03827
- Farquhar et al. 2023, *Challenges with Unsupervised LLM Knowledge Discovery*, arXiv 2312.10029
- Marks & Tegmark 2024, *The Geometry of Truth*
- Fry et al. 2023, *Comparing Optimization Targets for CCS*, arXiv 2311.00488
- *LLM Probing with Contrastive Eigenproblems*, arXiv 2511.02089
- Gorbett & Jana, *Characterizing Linear Alignment Across LMs* / "Ventriloquist LLMs" (HELIX), arXiv 2603.18908
- Gorbett et al., *Cross-Model Disagreement as a Label-Free Reward Signal* (CMP/CME), arXiv 2603.25450
- Gorbett et al., *CME-GRPO*, arXiv 2605.29009
- Bürger et al. 2025, truth-direction consistency, ACL Findings
- EleutherAI, *VINC-S* (supervision-interpolating probe)
- Anthropic 2024, *Simple Probes Can Catch Sleeper Agents*
- Beigi et al., *ARA*, arXiv 2602.01750
- Bergen et al. / Goodfire, arXiv 2609.19101
- LLM-judge self-preference and perplexity bias (Zheng et al.; Panickssery et al.)
- Fréchet 1951 (bounds on a joint with fixed marginals); Ueda & Nakano 1996 /
  Brown et al. 2005 (bias–variance–covariance, the correlation floor)

Full BibTeX in `paper/refs.bib` — **the arXiv ids above need verifying before
submission**.

---

## 12. Where the paper lives

- `paper/` — `main.tex` (title, skeleton), `abstract.tex`, `sections/theory.tex` (the §2.4
  algebra, formally), `refs.bib`, and `PLAN.md`: framing, the two causes of false
  agreement and the attribution control, probe-type and layer decisions, the
  section plan, the citation checklist, and the venue (TMLR in December; ICML in
  February as the alternative; arXiv either way).
- `future_work/paper2-representational-similarity.md` — predicting `rho` from
  CKA, the `FA(kappa)` tradeoff curve, and the full N-model floor. Paper 1 is its
  prerequisite, because the CKA→`rho` fit needs measured `rho`.
