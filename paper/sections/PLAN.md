# Paper 1 — writing plan

The spec for the *science* is [`../../DESIGN.md`](../../DESIGN.md). This file holds the
decisions that are about the *paper*: framing, what goes in which section, the
methodological additions that the experiments have to support, and the venue.

Title and abstract live in [`../tmlr.tex`](../tmlr.tex) / [`abstract.tex`](abstract.tex);
the formal treatment of the false-agreement algebra is
[`theory.tex`](theory.tex), and every quantity in it is implemented in
[`../../cmb/metrics.py`](../cmb/metrics.py).

## 1. Framing

**This is a scalable-oversight and evaluation paper, not an interpretability
paper.** The probe is the *instrument*; the subject is how reliable cross-model
agreement is as an oversight signal. Keep the interpretability machinery in
Method, and keep the claims about oversight reliability.

Do **not** make diversity-as-a-defense or security claims in the body — they are
Future Work. Likewise the theory section is framing that motivates a measurement
(see the scope note at the top of `theory.tex`): the identity is elementary and
the correlation floor is borrowed from ensemble theory. What is ours is the
application to probe-based cross-model oversight, the transport term, and
measuring ρ on internal readouts.

## 2. The two causes of false agreement — keep them separate

| cause | what is happening | what it implies |
|---|---|---|
| (a) shared wrong belief | the probe faithfully reads a belief both models hold, and that belief is false | a real limit of cross-model oversight |
| (b) probe failure | the probe follows familiarity/prominence instead of truth (*"bats are blind"*) | an instrument problem, not an oversight result |

Separating these needs an **attribution control**, and it is what makes the
result interpretable:

- Gate A (`../../experiments/gate_a_probe_truth.py`) is the first line: a probe that
  only works where output confidence already works is case (b) everywhere.
- Include **adversarial items where the confound and the truth come apart** —
  TruthfulQA above all, plus familiar-vs-defamiliarized paraphrases of the same
  false claim (DESIGN.md §4, Experiment 3 mitigation (c)). If Row-2 items lie
  along the prominence direction recovered from those pairs, that is evidence for
  (b); if they do not, for (a).

## 3. Method decisions the experiments must support

- **Probe types, on the same data and the same activations** (`--probe`):
  - `ccs` — unsupervised, the case that matters;
  - `mass-mean` — the strongest causal baseline in Marks & Tegmark;
  - `lr` — cheap, and probably redundant with mass-mean; report it to show that.
  - Optional: **VINC-S**, which varies the *amount* of supervision inside one
    method, giving a supervision axis rather than three unrelated probes.
- **Native first.** Fit native probes on each model wherever labels allow, and
  use transport only where they do not. Transport is a measured cost (ε), not
  the default path.
- **Clean contrasts, one variable each.** Same family at different sizes
  (Qwen-7B ↔ Qwen-32B) *versus* different families at the same size
  (Qwen-7B ↔ Llama-8B). Stated prediction, to be written down before the run:
  same-family pairs sit near the Fréchet upper bound; cross-family pairs show
  partial decorrelation but do **not** reach independence, because the models
  share web-scale pretraining data.
- **Weak→strong direction.** Fit where labels exist, read where they do not. ε is
  largest here, and reporting it honestly is a contribution even if it is large.
- **Layer choice.** Marks & Tegmark probe middle layers (layer 13 of 40 on
  LLaMA-13B) and do not report the last layer; the sleeper-agent probes are also
  middle-layer. HELIX and the platonic-representation argument point at the last
  layer for the *map*. Resolution: sweep layers (Exp 5), and fit maps at several
  layers so each map matches its probe's layer rather than forcing one site.
- **Post-training check.** Run the same probe on base and instruct checkpoints of
  one model and ask whether post-training degrades transport at the last layer
  (`../../experiments/exp6_post_training.py`).

## 4. Section plan

| section | source |
|---|---|
| Introduction | DESIGN.md §0, §2; the four-step argument in `tmlr.tex` |
| The Algebra of False Agreement | `theory.tex` (DESIGN.md §2.4) |
| Method | DESIGN.md §2, §5, §6; probe types above |
| Experiments | DESIGN.md §4 — Exp 0, Gates A/B, Exp 1–6 |
| Results | `../../results/RESULTS.md` + `results/*.json` |
| Related Work | §5 below |
| Discussion / Limitations | DESIGN.md §8, unabridged |
| Future Work | `../../future_work/paper2-representational-similarity.md` |

## 5. Citation checklist

Burns et al. (CCS) · Farquhar et al. (CCS critique) · Marks & Tegmark (COLM
2024) · Bürger et al. (ACL Findings 2025, truth-direction consistency) ·
contrastive eigenproblems (arXiv 2511.02089) · EleutherAI VINC-S · Anthropic
*Simple probes can catch sleeper agents* · ARA (Beigi et al., arXiv 2602.01750) ·
Bergen et al. / Goodfire (arXiv 2609.19101) · LLM-judge self-preference and
perplexity bias · Fréchet (bounds) · ensemble bias–variance–covariance for the
correlation floor · the author's own HELIX / CMP-CME / CME-GRPO line.

Entries are in [`refs.bib`](refs.bib); the arXiv ids came from DESIGN.md §11 and
**every one needs verifying before submission**.

## 6. Venue and timeline

- **Target: TMLR, submitted in December.** Reviews arrive about 4 weeks after
  submission; the median decision is about 76 days. An accepted TMLR paper is
  eligible for the NeurIPS/ICLR/ICML Journal-to-Conference track, though
  selection there is selective.
- **Alternative: ICML, February.**
- **Post to arXiv either way**, at submission time.
