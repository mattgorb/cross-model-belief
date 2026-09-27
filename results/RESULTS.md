# RESULTS

Fill this in as runs land. The point of the template is that the outcome gets
classified against the DESIGN.md §7 decision tree *before* the narrative is
written, so a negative result stays legible instead of being reframed.

## Run configuration

| | |
|---|---|
| date | |
| models | |
| datasets | |
| N per dataset | |
| layer | |
| probe (`--probe`) | |
| commit | |

## Decision-tree classification

Circle one (DESIGN.md §7):

1. **Exp 0 fails** — activations do not beat logprobs; internal-space framing unjustified.
2. **Gate A fails** — the probe reads confidence, not truth. Stop.
3. **Gate A passes, Gate B fails** — probe real, transport dead. Single-model story.
4. **Gates pass, Row 2 separable and generalizes** — the result: a label-free
   detector of untrustworthy agreement.
5. **Gates pass, Row 2 does not generalize** — the honest cap on cross-model
   belief agreement.

**Outcome: _____**

## Gates

| gate | metric | value | verdict |
|---|---|---|---|
| Exp 0 | probe edge over logprob | | |
| Exp 0 | silent-error reduction | | |
| Gate A | probe − confidence AUROC on the confident slice (per model) | | |
| Gate B | native − transfer AUROC, both directions | | |
| Gate B context | linear CKA, map R² | | |
| Gate B | `eps` — FP inflation at matched positive rate, both directions | | |

## Experiment 1 — the headline

| pair | kind | Row-2 rate | 95% CI | error corr | CKA |
|---|---|---|---|---|---|
| Qwen-7B ↔ Qwen-32B | same-family | | | | |
| Qwen-7B ↔ Llama-8B | cross-family | | | | |

Floor or ceiling (§2.3)? Does Row 2 drop cross-family?

### The decomposition (DESIGN.md §2.4)

A rate without its bounds does not distinguish "accurate probes" from
"independent probes". Fill both rows per pair:

| pair | p1 | p2 | FA | p1·p2 (indep.) | Fréchet [lo, hi] | rho | rho / rho_max | coverage |
|---|---|---|---|---|---|---|---|---|
| same-family | | | | | | | | |
| cross-family | | | | | | | | |

Prediction to check (paper/PLAN.md §3): same-family near the upper bound;
cross-family partly decorrelated but **not** independent. Was it right?

### The cost of transport

| pair | direction | eps | rho native → transported | FA native → transported |
|---|---|---|---|---|
| | A→B | | | |
| | B→A | | | |

Does the map inflate `rho` on top of the error rate, and by how much?

## Experiment 2 — transport-robust core

Row-2 count per transport direction, the intersection, and the Jaccard. A low
Jaccard means the map rather than shared belief is picking the set.

## Experiments 3–4 — the bet

In-distribution separability against both controls (shuffled labels, output
confidence), then the 5×5 matrix. **The off-diagonal is the result**; report
IMDB → TruthfulQA explicitly, and the held-out model pair with its anchor-map R².

## Experiment 5 — layers

Where the belief signal peaks vs where false-agreement structure peaks. If they
differ, say so and do not tune one at the other's depth.

## Experiment 6 — post-training control

Transport gap at `final` vs the mid-depth site, base ↔ instruct. If the last layer
is worse, maps belong at the probe's own layer and every table needs its layer
stated. This pair's `eps` is also the floor every other pair inherits.

## Probe types

Same activations, same layer: CCS against mass-mean and LR. Does the unsupervised
case reach the supervised baselines, and is LR redundant with mass-mean as
expected?

| probe | AUROC (native) | Row-2 rate | rho / rho_max |
|---|---|---|---|
| ccs | | | |
| mass-mean | | | |
| lr | | | |

## Threats to validity as they actually bit

Work through DESIGN.md §8 — prominence vs truth, contamination, the transport
blind spot, independence of process vs of knowledge — and state which ones
constrain *this* run's claims.
