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

**The pre-registered prediction holds** (paper/PLAN.md §3): same-family sits near
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
