# RESULTS

Runs of 2026-09-28/29, on the activation cache extracted 2026-09-27/28.
Every number below is on **held-out data**; the protocol for each is stated with it.

**Decision-tree classification (DESIGN.md §7): rule 4, qualified.** Gates pass,
Row 2 is separable, and the separation survives an unseen dataset — but weakly on
the dataset that matters, not at all into BoolQ, and the practical gain is a
prioritisation of the blind spot rather than closure of it.

## Configuration

| | |
|---|---|
| date | 2026-09-28 / 29 |
| models | 10 extracted; results here use qwen3-1.7b, qwen3-8b, qwen3-32b, gemma4-31b, llama-8b |
| datasets | geometry_of_truth (5030), truthfulqa (1634), boolq (3270), imdb (5000), rte (277) |
| N per dataset | all items (`--n all`) |
| layer | final (per model: 36 / 64 / 32 / 60 / 32) |
| pooling | mean |
| probe | **mass-mean** (see "Probe choice") |
| commit | see `git log` |

## Probe choice: CCS does not work here

CCS reads truth on simple factual statements and fails on confound-rich data.
Gate A on TruthfulQA, where 35–38% of confident items are wrong:

| probe | qwen3-8b edge | llama-8b edge |
|---|---|---|
| CCS | −0.03 to +0.08 | −0.06 to −0.17 |
| **mass-mean** | **+0.10 to +0.14** | **+0.15 to +0.17** |
| LR | +0.32 to +0.34 | +0.25 to +0.26 |

(`edge` = probe AUROC − output-confidence AUROC, on the confident slice.)
CCS scores 0.99 on Geometry of Truth and 0.60 on TruthfulQA: it finds *a* salient
binary feature, which coincides with truth on easy data and with familiarity on
adversarial data — the Farquhar et al. critique, reproduced. LR is stronger than
mass-mean but unconstrained (4096-dim fit on ~1000 items); mass-mean is the
causal baseline and is what the results below use. LR belongs in the paper as an
upper bound.

## Gates

| gate | metric | value | verdict |
|---|---|---|---|
| Exp 0 | probe vs logprob AUROC (GoT) | 0.990 vs 0.979 | **PASS** |
| Exp 0 | silent-error rate | 0.067 → 0.014 at higher coverage | **PASS** |
| Gate A | probe − confidence on confident slice (TruthfulQA) | +0.104 / +0.158 | **PASS** |
| Gate B | native − transfer AUROC, GoT, same-family | 0.000 / 0.000 | **PASS** |
| Gate B | same, TruthfulQA, cross-family, final layer | −0.136 / −0.000 | **FAIL one direction** |

Gate A has no power on Geometry of Truth (models are confidently wrong on 1% of
items, so there is nothing to beat) — it must be read on TruthfulQA.

Gate B is configuration-dependent: it passes both directions at 0.75 depth with
last-token pooling (−0.004 / +0.045) and fails one direction at final/mean. The
probe prefers the final layer; transport prefers 0.75. Unresolved.

## Experiment 1 — the headline

Native probes, each model on its own activations. TruthfulQA.

| pair | Row-2 rate | ρ/ρ_max | detectable coverage |
|---|---|---|---|
| **same-family** qwen3-8b \| qwen3-32b | **0.269** | +0.66 | 0.21 |
| cross qwen3-8b \| gemma4-12b | 0.211 | +0.22 | 0.38 |
| cross qwen3-1.7b \| olmo3-7b | 0.171 | +0.60 | 0.24 |
| cross qwen3-8b \| llama-8b | 0.168 | +0.50 | 0.33 |
| cross qwen3-32b \| gemma4-31b | 0.080 | −0.15 | 0.69 |

**same-family 0.269 vs cross-family 0.157 → FLOOR.** False agreement drops with
model independence, so agreement is boundable rather than irreducible.

**The pre-registered prediction holds** (pre-registered in DESIGN.md §4): same-family sits near
the Fréchet upper bound (ρ/ρ_max +0.66), cross-family partly decorrelates
(+0.29 mean) and *never reaches independence* — ρ is positive in four of five
pairs. The one negative value (qwen3-32b | gemma4-31b, −0.15) is the cell to
distrust; it did not reproduce under a different pooling, where the same pair
gave +0.16.

## Experiment 3 — separability

Inside the {both probes say True} subset, fit a linear direction to separate
GT-true from GT-false. TruthfulQA, in-distribution.

| pair | separability | shuffled control | confidence baseline |
|---|---|---|---|
| qwen3-8b \| qwen3-32b | 0.903 | 0.514 | 0.600 |
| qwen3-8b \| llama-8b | 0.899 | 0.508 | 0.637 |

Separable, and neither control explains it.

## Experiment 4 — generalization

**Pairwise matrix (fit on one dataset, test on another): uninformative.**
Off-diagonal 0.658 / 0.623 across two pairs, but every above-chance cell rests on
5–30 Row-2 items, and the two well-powered columns (TruthfulQA, BoolQ) sit at
chance. Reported for completeness; do not draw the conclusion from it.

**Leave-one-dataset-out (fit on the other four, test on the held-out one).**
Probes, map and direction are all fitted on the pool; the held-out set is unseen
by all three.

| pair | mean AUROC | shuffled | TruthfulQA fold | n Row-2 (test) |
|---|---|---|---|---|
| qwen3-8b \| qwen3-32b | 0.716 | 0.504 | **0.614** | 220 |
| qwen3-32b \| gemma4-31b | 0.681 | 0.511 | **0.661** | 359 |

The signature transfers to an unseen dataset, replicated across two structurally
different pairs. Two qualifications: **BoolQ is a well-powered null** (0.560 vs
0.518, n=1237), so transfer is not uniform; and the GoT fold's shuffled control
came out at 0.637, which should be ~0.5 — that fold is not trustworthy and the
cause is not yet understood.

**Held-out model pair**: the direction does not transfer to an unseen pair
(0.375–0.613, mean ≈0.46). Not pursued further — fitting a direction per pair
costs seconds, so the deployable claim is a *per-pair* detector.

## Experiment 5 — operating points

What the detector buys an operator. TruthfulQA held out entirely; budget is a
fraction of the *agree-true* set, which is where the router is silent.

**qwen3-8b | qwen3-32b** — router catches 213/817 = 26.1% of false claims free.
Silent set 742 items, 220 false (base rate 29.6%), direction AUROC 0.614.

| budget | routed | Row-2 found | recall | precision | total caught |
|---|---|---|---|---|---|
| 5% | 37 | 25 | 11.4% | **67.6%** | 29.1% |
| 10% | 74 | 43 | 19.5% | 58.1% | 31.3% |
| 20% | 148 | 67 | 30.5% | 45.3% | 34.3% |
| 50% | 371 | 132 | 60.0% | 35.6% | 42.2% |

**qwen3-1.7b | qwen3-32b (weak→strong)** — router catches 308/817 = 37.7%.
Silent set 700 items, 260 false (base rate 37.1%), direction AUROC 0.531.

| budget | routed | Row-2 found | recall | precision | total caught |
|---|---|---|---|---|---|
| 5% | 35 | 20 | 7.7% | **57.1%** | 40.1% |
| 10% | 70 | 32 | 12.3% | 45.7% | 41.6% |
| 20% | 140 | 55 | 21.2% | 39.3% | 44.4% |
| 50% | 350 | 135 | 51.9% | 38.6% | 54.2% |

**The trade-off between them is the interesting part.** The weak overseer
disagrees far more often, so free router coverage nearly doubles (37.7% vs
26.1%) — but what remains hidden is *less detectable* (AUROC 0.531 vs 0.614,
1.5× precision lift vs 2.3×). A weaker overseer buys coverage with more checks
and cruder targeting of the residue.

**The honest summary**: at a 5% budget the detector's flags are right 68% of the
time against a 30% base rate. It prioritises the blind spot well and closes it
slowly — reaching 60% of Row 2 costs a 50% budget at 1.2× lift.

## Threats, as they actually bit

- **Prominence, not truth.** This is not hypothetical here: CCS fails Gate A on
  TruthfulQA, which is exactly the predicted failure. The results depend on
  mass-mean, which is supervised — so the label-free framing in DESIGN.md §2.2
  does not survive, though the oversight story does (fit once, deploy unlabeled).
- **Transport blind spot.** Gate B passes at one configuration and fails at
  another; the probe and the map prefer different depths.
- **Small positive classes.** Row 2 is rare wherever the models are good, which
  is what made the pairwise matrix uninterpretable. Any future cell needs its
  Row-2 count reported next to it.
- **Independence of process ≠ of knowledge.** ρ is positive in nearly every
  pair; no pair reached independence.

## Not done

Exp 2 (bidirectional — not applicable while verdicts are native), Exp 6
(post-training control), the probe-comparison table as a formal experiment, and
the mirrored feature space (A→B instead of B→A) as a robustness check.

---

# The full sweep (49 pairs)

Everything above was five hand-picked pairs. This section is the sweep over every
sensible (overseer, target) pair — 49 ordered pairs across 10 models and 5
datasets, 587 + 585 rows, no failures. **It overturns several of the conclusions
above, and those are marked.**

Raw data: [`sweep_base_all.csv`](sweep_base_all.csv) (one row per pair x dataset)
and [`sweep_loo_all.csv`](sweep_loo_all.csv) (one row per pair x held-out
dataset). Regenerate with `scripts/sweep.py`, summarise with
`scripts/summarize_sweep.py`.

Pairs are *ordered*: the first model is the overseer, and it is never larger than
the target, because a big model watching a small one is not the oversight
question. Two simplifications relative to the single runs above: the ridge
strength is fixed at 0.1 rather than searched per cell, and the map is fitted on
the scored dataset alone rather than on all of them pooled. Both make the
transport numbers pessimistic.

*AUROC, throughout: pick one item with the property and one without; AUROC is how
often the score ranks the right one higher. 0.5 is a coin flip.*

## A. How often are both models wrong and agreeing?

For every claim that is actually false, the probes either disagree (a router
catches it) or **both say true** — both wrong, nobody flags it. `FA` is the size
of that blind spot as a share of false claims. `rho/rho_max` says whether they
are wrong for the *same reason* or by coincidence, scaled by the maximum the
marginals allow. `coverage` is the share of the **target's** errors that
disagreement catches.

| dataset | FA same | FA cross | rho/max same | rho/max cross | coverage same | coverage cross |
|---|---|---|---|---|---|---|
| geometry_of_truth | 0.044 | 0.045 | **0.473** | 0.285 | 0.403 | 0.512 |
| imdb | 0.097 | 0.061 | **0.624** | 0.337 | 0.384 | 0.577 |
| boolq | 0.124 | 0.127 | 0.210 | 0.204 | 0.494 | 0.528 |
| truthfulqa | 0.159 | 0.131 | 0.201 | 0.206 | 0.522 | 0.596 |
| rte | 0.219 | 0.235 | 0.418 | 0.481 | 0.362 | 0.373 |

**This overturns the FLOOR result above.** Across 49 pairs the blind spot does
*not* reliably shrink cross-family: three of five datasets move by less than
0.02. The five-pair version was not robust.

What does hold is subtler and is the better claim: errors are less correlated
cross-family (GoT 0.285 vs 0.473, IMDB 0.337 vs 0.624), and coverage of the
target's errors is higher cross-family on **all five** datasets. The rate stays
flat because cross-family pairs have higher *individual* error rates that cancel
their lower correlation — the identity's two terms trading off, which is the
argument for reporting the decomposition instead of the rate.

## B. Is the probe reading truth, or re-reading confidence?

On the half of items the model is most confident about, probe AUROC minus the
model's own confidence AUROC. Positive means the activations say something the
output does not.

| model | boolq | GoT | imdb | rte | truthfulqa |
|---|---|---|---|---|---|
| qwen38-27b | 0.221 | 0.310 | — | 0.400 | **0.323** |
| gemma4-31b | 0.057 | 0.275 | — | −0.030 | **0.350** |
| qwen3-1.7b | 0.138 | 0.268 | 0.073 | −0.105 | **0.209** |
| olmo3-7b | 0.055 | 0.006 | 0.089 | 0.032 | **0.227** |
| qwen3-8b-base | 0.067 | 0.192 | −0.016 | −0.072 | **0.207** |
| llama-8b | −0.009 | −0.003 | 0.001 | 0.000 | **0.158** |
| qwen3-8b | −0.033 | 0.242 | −0.032 | 0.049 | **0.104** |
| gemma4-12b | 0.021 | 0.043 | −0.011 | 0.015 | **0.079** |
| qwen3-32b | 0.007 | 0.077 | −0.004 | 0.122 | **0.049** |

TruthfulQA is the only column with no negatives: all ten models pass. That is
where models are confidently wrong most often (42% of confident items), so there
is work for a probe to do. On BoolQ and IMDB confidence is already a good
predictor and the probe cannot beat it. The instrument works where it is needed.

## C. Does the map carry the probe?

Fit the probe on one model, read it on the other through the linear map. `gap` is
the AUROC lost in transit; the tolerance is 0.05; `pass rate` requires both
directions under it.

| kind | dataset | gap a→b | gap b→a | worst | pass rate | CKA | map R² |
|---|---|---|---|---|---|---|---|
| same | geometry_of_truth | 0.049 | 0.034 | 0.072 | **0.67** | 0.757 | 0.786 |
| cross | geometry_of_truth | 0.032 | 0.049 | 0.084 | **0.61** | 0.788 | 0.792 |
| same | imdb | 0.060 | 0.063 | 0.095 | 0.46 | 0.499 | 0.476 |
| cross | imdb | 0.109 | 0.169 | 0.222 | 0.36 | 0.514 | 0.479 |
| cross | rte | 0.063 | 0.049 | 0.103 | 0.34 | 0.528 | 0.170 |
| same | truthfulqa | 0.045 | 0.037 | 0.073 | 0.28 | 0.407 | 0.252 |
| cross | truthfulqa | 0.042 | 0.069 | 0.097 | 0.18 | 0.441 | 0.268 |
| both | boolq | 0.14–0.19 | 0.14–0.16 | 0.24 | **0.00** | 0.42 | 0.40 |

Transport works on Geometry of Truth and mostly fails elsewhere — **zero of 126
BoolQ pairs pass**. Same family barely helps. Read this as the worst case: with
the map fitted on all datasets pooled (4x the rows) spot checks did far better.

## D. Does the detector work on a dataset it never saw?

Probes, map and direction are all fitted on four datasets; the fifth is unseen by
all three. `shuffled` is the same procedure with permuted labels — what a
detector that learned nothing would score. **`margin` is the real number**, and
`ctrl sd` is how much the control itself moves: a margin under ~2 sd is not
measurable.

| held out | kind | AUROC | shuffled | margin | ctrl sd | folds | median Row-2 |
|---|---|---|---|---|---|---|---|
| geometry_of_truth | cross | 0.909 | 0.447 | **+0.463** | 0.144 | 90 | 642 |
| | same | 0.889 | 0.479 | **+0.410** | 0.159 | 36 | 645 |
| rte | cross | 0.705 | 0.510 | +0.195 | 0.061 | 90 | 49 |
| imdb | same | 0.641 | 0.500 | +0.140 | 0.078 | 22 | 708 |
| | cross | 0.634 | 0.509 | +0.125 | 0.063 | 61 | 762 |
| truthfulqa | same | 0.552 | 0.486 | +0.066 | 0.044 | 35 | 363 |
| | cross | 0.560 | 0.495 | +0.065 | 0.047 | 89 | 373 |
| boolq | cross | 0.535 | 0.500 | +0.035 | 0.025 | 88 | 180 |
| | same | 0.526 | 0.494 | +0.032 | 0.023 | 36 | 109 |

Decisive on Geometry of Truth, moderate on RTE and IMDB, and **barely present on
TruthfulQA** (+0.065 against a control that wobbles ±0.047) — which is the
dataset where the blind spot matters. Same-family and cross-family are
indistinguishable throughout: the dataset decides, not the pairing.

## E. What does an operator get?

Two models read a claim. Disagreement routes it to an expensive check, free. When
both say true nobody flags it, so a budget of extra checks is spent there,
prioritised by the detector. `base rate` is how often a *randomly chosen*
agreeing item is false — the bar to beat. `prec@k` is the share of checks that
find a real error at that budget.

| held out | router (free) | base rate | prec@5% | prec@10% | prec@20% | caught@10% | caught@20% |
|---|---|---|---|---|---|---|---|
| geometry_of_truth | 0.395 | 0.234 | **0.938** | 0.889 | 0.747 | 0.495 | 0.563 |
| rte | 0.284 | 0.327 | 0.685 | 0.632 | 0.563 | 0.356 | 0.412 |
| truthfulqa | 0.339 | 0.428 | 0.510 | 0.501 | 0.490 | 0.393 | 0.444 |
| imdb | 0.291 | 0.298 | 0.508 | 0.480 | 0.444 | 0.340 | 0.382 |
| boolq | 0.459 | 0.171 | 0.175 | 0.186 | 0.189 | 0.478 | 0.498 |

Lift over random checking at a 5% budget: GoT **4.0x**, RTE 2.1x, IMDB 1.7x,
TruthfulQA 1.2x, BoolQ 1.0x (nothing).

Two things worth saying out loud. On GoT, 5% more checks find a real error 94% of
the time against 23% at random. On TruthfulQA it is 51% against 43% — barely a
coin. And the base rates themselves are the uncomfortable part: **when these
models agree that a TruthfulQA claim is true, it is false 43% of the time.**
Agreement is not much of a safety signal there to begin with.

## F. What predicts whether the detector works

Correlation with the margin from table D, across all 49 pairs:

| candidate explanation | corr with margin |
|---|---|
| how common false agreement is | **−0.636** |
| how often agreement is wrong (base rate) | −0.391 |
| how correlated the two models' errors are | +0.098 |
| how similar the two models look (CKA) | −0.071 |

**The rarer the blind spot, the easier it is to detect.** Where both models are
usually right, being wrong together is unusual and stands out; where they are
wrong together 43% of the time, that *is* the normal state and there is no
anomaly left to find.

This kills two hypotheses formed earlier in the project. Error correlation does
**not** predict detectability (+0.098), and neither does representational
similarity (−0.071) — which undercuts the CKA-predicts-rho premise of the Paper 2
sketch in `future_work/`. That sketch needs rewriting or dropping.

## The bottom line

The measurement is solid and replicates: false agreement is real, it is 4–24% of
false claims depending on the dataset, and its decomposition into error rates and
error correlation behaves as the identity says it must.

The detector is the weaker half. It works where the blind spot is small and rare
(4x lift on Geometry of Truth) and barely works where it is large and common
(1.2x on TruthfulQA) — the opposite of where an operator would want it. The
paper's claim should be the bound plus this asymmetry, not a detector.
