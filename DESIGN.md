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
3. **Quantifying how much of a probe's *characterization* survives linear
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
- Also report **error correlation** between the two models' probe verdicts; this
  drives the amplification factor (coverage bought per label).

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

Size ladder + cross-family, all open-weight (need activations + logprobs):

- `Qwen/Qwen2.5-1.5B-Instruct`
- `Qwen/Qwen2.5-7B-Instruct`
- `Qwen/Qwen2.5-32B-Instruct`
- `meta-llama/Llama-3.1-8B-Instruct`

Pairs:
- **Same-family:** Qwen-7B ↔ Qwen-32B (clean map, shared tokenizer).
- **Cross-family:** Qwen-7B ↔ Llama-8B (different tokenizers → expect lower CKA;
  per the author's HELIX finding, tokenizer compatibility and size gap drive
  alignment success, so read a weak cross-family result as *possibly the map, not
  the probe* — Gate B disambiguates).

---

## 7. Metrics and decision rules

- **AUROC (sign-resolved).** CCS sign is arbitrary; take the orientation that
  predicts better. Report on full set and on the confident slice.
- **Row-2 rate** = `P(both True | GT False)`. Primary headline.
- **Error correlation** between probe verdicts (drives amplification).
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
6. Experiment 5 (layer sweep). Write `results/RESULTS.md` classifying the outcome
   per the decision tree.

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
