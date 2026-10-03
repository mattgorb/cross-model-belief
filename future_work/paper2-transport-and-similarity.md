# Paper 2 — Transportability and independence are the same quantity

*Status: the measurements exist (`results/sweep_base_all.csv`,
`results/pair_table.csv`), the write-up does not. Paper 1 is the prerequisite
only in that it defines the decomposition this paper's quantities sit in.*

This replaces the earlier sketch, which proposed that CKA predicts error
correlation and that the two effects trade off to give an interior optimum in
false agreement. The first half is right and strongly supported. The second is
wrong, and what it is wrong about is the more interesting result.

## The finding

Across $49$ ordered model pairs, measured per pair and averaged over five
datasets:

| relationship | correlation |
|---|---|
| CKA → error correlation (ρ/ρ_max) | **+0.749** |
| CKA → transfer loss | −0.442 |
| **transfer loss → error correlation** | **−0.764** |
| CKA → false agreement rate | −0.028 |

Splitting the pairs at the median transfer loss makes it concrete:

| | ρ/ρ_max | CKA | FA |
|---|---|---|---|
| transports well | 0.498 | 0.615 | 0.121 |
| transports badly | 0.134 | 0.451 | 0.128 |

**You cannot have both.** The representational overlap that lets a probe
validated on one model be read on another is the same overlap that makes the two
models fail together. Pairs that transport well have errors four times as
correlated as pairs that do not. These are not two properties to be traded off by
tuning; they are two measurements of one thing, correlated at $-0.76$.

**And false agreement does not move.** $0.121$ against $0.128$ across that split.
Similarity raises the correlation term of the identity and lowers the individual
error rates, and the two cancel. There is no interior optimum in this range —
which is the same cancellation that flattened the family-diversity contrast in
Paper 1, now driven by a continuous variable instead of a binary one.

## Why this is a paper

1. **It is a constraint on method design, not a parameter to tune.** Anyone
   proposing to amortize probe validation across models — characterize once,
   reuse where labels are unavailable — is proposing to do it precisely on the
   pairs that make the worst ensemble. That tension has not been stated, and it
   follows from measurements rather than from argument.
2. **It is prospective and label-free.** CKA needs activations on shared inputs
   and no ground truth. Fitting the CKA→ρ relation once, on labelled pairs, lets a
   practitioner read a reliability floor off a candidate overseer before deploying
   it. The $+0.749$ is what makes that usable; the per-dataset spread
   ($+0.29$ to $+0.70$) is what bounds the claim.
3. **The transport measurement is itself new.** Transfer loss — the AUROC a probe
   gives up when read on another model through a linear map — is a
   task-relevant measure of representational compatibility, and it disagrees with
   CKA in individual cases (a pair at CKA $0.49$ whose transported probe loses
   $0.004$). Per dataset it ranges from $0.027$ on curated factual statements to
   $0.23$ on BoolQ, where no pair of $126$ stays inside tolerance.

## What is already measured

Everything in the tables above, plus: transport by dataset and direction, the
strong-to-weak asymmetry (reading the larger model's probe on the smaller costs
$0.094$ against $0.115$ the other way, matching the asymmetry reported for linear
alignment), map $R^2$ and CKA per cell, and the logit-distance test of
\citet{nielsen2026logit} in Paper 1's appendix.

## What is not

- **The identifiability connection.** Nielsen et al. prove linear
  representational similarity is controlled by a logit-based distributional
  distance, not by KL. Paper 1's appendix tests a one-dimensional proxy and finds
  it uninformative; the real test needs full next-token distributions over a
  corpus, which would be the first extraction to add.
- **The N-model floor.** `N_eff = N/(1 + (N−1)ρ̄)` is a remark in Paper 1
  (implemented as `cmb.metrics.n_eff`). With ρ̄ now predictable from CKA, the
  floor becomes something a practitioner can estimate for a candidate pool
  without labels.
- **Predictors beyond CKA.** Tokenizer compatibility and size gap are known to
  drive alignment success; `cmb/tokalign.py` computes the compatibility score.
  Whether either adds anything over CKA is untested.

## Paper 3 (sketch, unchanged)

Swarms, where ρ is not a fixed property of a pair: agents reading one another's
outputs should become more correlated over rounds, so the effective number of
independent overseers decays during the interaction. The static relation above is
the initial condition.
