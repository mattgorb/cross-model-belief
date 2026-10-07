> **Note (added on restore, not part of the original):** this is the original Paper 2 proposal, kept as written. The later measurements agree with its first half (CKA does predict error correlation, +0.749) and contradict its second (no interior optimum in false agreement: 0.121 against 0.128 across the transport split). See `paper2-transport-and-similarity.md` for the version written against the data. Both are kept.

# Paper 2 — Predicting oversight reliability from representational similarity

*Status: future work. Nothing in this file is implemented in `cmb/`; Paper 1
(`../DESIGN.md`, `../paper/`) is the prerequisite, because the CKA→ρ relationship
can only be fitted once ρ has been measured on labeled pairs.*

## The gap this closes

Paper 1 measures false agreement **after the fact**, and doing so needs ground
truth: ρ is an error correlation, and errors are only visible against labels. That
is a poor fit for the deployment question, which is asked *before* a labeled
evaluation exists — *is this candidate overseer independent enough of the model it
will oversee?*

Paper 2 asks whether **linear CKA between two models predicts their error
correlation ρ**. CKA needs only activations on shared inputs: nothing is fitted,
nothing is transported, and no ground truth is required. Fit the CKA→ρ
relationship **once**, across many labeled model pairs; afterwards anyone can
compute CKA for a candidate overseer and read off its reliability floor before
deploying it.

## The theoretical part: two terms pulling against each other

With CKA (write it κ) as the single variable, the two quantities from Paper 1's
theory move in opposite directions:

- **transport error ε(κ)** — *decreases* with similarity. Similar models admit a
  good alignment map, so a probe characterized on one survives the trip to the
  other.
- **error correlation ρ(κ)** — *increases* with similarity. Similar models fail
  together, which is exactly what false agreement is made of.

Substituting both into the false-agreement identity
(`../paper/sections/theory.tex`, Prop. 1, with the transport adjustment of §2.5) makes
false agreement a **curve in κ with an interior optimum**:

    FA(κ) = p₁ · p₂'(κ) + ρ(κ) · √(p₁(1−p₁)) · √(p₂'(κ)(1−p₂'(κ))),
    where p₂'(κ) = p₂ + ε(κ)

The operating point to select for is therefore models **similar enough to
transport between, and different enough to fail independently**. This is a
formalized *empirical tradeoff*, not a theorem: the shapes of ε(κ) and ρ(κ) are
measured, and the optimum is wherever they cross.

## Contents

1. **The CKA→ρ fit.** Many model pairs (families × scales), ρ measured per pair as
   in Paper 1 Exp 1, κ computed from shared-input activations. Report the fit, its
   spread, and — the useful artifact — a **predicted reliability floor from κ
   alone**, with intervals.
2. **The FA(κ) curve** and its interior optimum, with ε(κ) and ρ(κ) shown
   separately so the tradeoff is visible rather than asserted.
3. **The full N-model correlation floor.** Paper 1 carries `N_eff = N / (1 + (N−1)ρ̄)`
   as a remark only (`../paper/sections/theory.tex` §2.6, implemented as `cmb.metrics.n_eff`);
   here it is developed properly — how ρ̄ behaves as the pool grows, whether
   adding a *family* rather than a *model* moves the floor, and what the cap
   `1/ρ̄` means for realistic overseer pools.
4. **Predictors beyond CKA.** Tokenizer compatibility and size gap already
   predict alignment success (HELIX, and `cmb/tokalign.py` computes the
   compatibility score); test whether they add anything over κ.

## Why it stands on its own

The deliverable is label-free and prospective: a practitioner picks an overseer
with a number they can compute, instead of discovering the blind spot after
deployment. Paper 1 bounds how much oversight agreement *can* provide; Paper 2
says which pair to choose.

## Paper 3 (sketch)

**Swarms**, where ρ is no longer a fixed property of a model pair: agents read one
another's outputs, so correlation *rises during the interaction* and the effective
number of independent overseers decays over rounds. Paper 2's static κ→ρ
relationship is the initial condition for that dynamic, which is why it has to
come first.
