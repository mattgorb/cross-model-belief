# Experiment plan — one page

The full spec is [`../DESIGN.md`](../DESIGN.md); the step-by-step operational
guide is [`RUNBOOK.md`](RUNBOOK.md). This page is the short version: what is
being measured, on what, in what order, and what each step can rule out.

## The question

Two models, each with a belief probe read from its own activations. Their
agreement is a label-free oversight signal. The signal has one structural blind
spot — **false agreement**: both probes read a claim as *true* while the ground
truth is *false*. Disagreement routing stays silent exactly there.

**Headline number:** `FA = P(both say True | ground truth False)`.

It is reported with its decomposition, because a small rate can mean two very
different things (DESIGN.md §2.4):

```
FA = p1*p2 + rho * sqrt(p1(1-p1)) * sqrt(p2(1-p2))
```

`p_i` are the probes' false-positive rates and `rho` their error correlation. Low
FA because the probes are *accurate* is a different finding from low FA because
their errors are *independent* — only the second improves when you pick better
model pairs. So every run also prints the Fréchet bounds FA must sit inside,
`rho` normalized by its feasible maximum, and the detectable coverage
`1 - FA/p_strong` a router buys.

## Models

Open-weight only — activations *and* output logprobs are both required. Three
families, and one of them deliberately pre-reasoning-era.

| key | HF id | depth | role |
|---|---|---|---|
| `qwen3-8b` | `Qwen/Qwen3-8B` | 36 | **the anchor**; in every pair |
| `qwen3-32b` | `Qwen/Qwen3-32B` | 64 | same-family, 4× the scale |
| `qwen3-4b` | `Qwen/Qwen3-4B` | 36 | held-out pair |
| `qwen3-8b-base` | `Qwen/Qwen3-8B-Base` | 36 | Exp 6 control |
| `qwen38-27b` | `Qwen/Qwen3.8-27B` | 64 | newest Qwen generation (native VLM) |
| `gemma4-12b` | `google/gemma-4-12B-it` | 48 | third family, reasoning-era, has a base |
| `llama-8b` | `meta-llama/Llama-3.1-8B-Instruct` | 32 | the only non-reasoning model (gated) |

Qwen 3 carries the ladder because it is the only current family with small dense
checkpoints *and* matching bases; Qwen 3.8's smallest open dense model is 27B, so
it enters as a cross-generation pair. Llama 3.1-8B is still the newest dense small
Llama (Llama 4 is MoE), and its being pre-reasoning is the point.

**Pairs, one variable each where possible:**

| pair | `--pair` | the variable |
|---|---|---|
| qwen3-8b ↔ qwen3-32b | `same-family` | scale, shared tokenizer |
| qwen3-8b ↔ gemma4-12b | `cross-family` | family (both reasoning-era) |
| qwen3-8b ↔ llama-8b | `cross-era` | family **and** reasoning era — size-matched, so an upper bound on decorrelation |
| qwen3-8b ↔ qwen38-27b | `cross-generation` | generation |
| qwen3-4b ↔ gemma4-12b | `held-out` | nothing is ever fitted on it |
| qwen3-8b-base ↔ qwen3-8b | `post-training` | post-training only (Exp 6) |

**Prediction, on the record before the run:** same-family sits near the Fréchet
upper bound; cross-family decorrelates partly but does **not** reach
independence, because both models saw overlapping web-scale pretraining data.

**Fits an A10G (24GB, fp16)?** 4B and 8B yes; `gemma4-12b` is borderline at
~24GB; `qwen3-32b` and `qwen38-27b` need the `g5.12xlarge` (4× A10G).

## Reasoning

All three families except Llama are hybrid thinking models. **We run with no chat
template and thinking off** — extraction uses raw completion-style prompts, so no
`<think>` block is ever generated.

- It is the only reading comparable across reasoning and non-reasoning families.
- It keeps `p_yes` (Exp 0's baseline, Gate A's slice selector) a real single-token
  Yes/No logprob; with thinking on, the next token starts a trace instead.
- It is one forward pass per half, deterministic. Traces are variable-length
  sampling.

So the probe measures **pre-reasoning belief** — what the model represents before
deliberating. That is the right object here (the probe is for what the model does
*not* say), but the paper has to call it that. Whether reasoning makes false
agreement better or worse is a real open question, specified as Exp 7 in
DESIGN.md §6.1 and not yet implemented: independent traces might decorrelate the
errors, or both models might reason into the same familiar misconception and make
`rho` worse.

Note for Exp 6: in a hybrid model, post-training is also where thinking is
installed, so base ↔ instruct measures that too — the control is not as clean as
it was with a non-reasoning model.

## Datasets

All need true/false labels; each claim becomes a CCS contrast pair (the claim
asserted, and negated).

| dataset | role |
|---|---|
| Geometry of Truth | primary — curated T/F statements, the canonical truth-direction data |
| TruthfulQA | **false-agreement enrichment** — built from popular misconceptions, so it produces the failure naturally |
| BoolQ, RTE | matrix cells, standard CCS-replication data |
| IMDB | the sentiment-confound extreme; `IMDB → TruthfulQA` is the designated hardest generalization cell |
| MMLU (hard subjects) | coverage only; two claims per question (true + the model's *chosen wrong* answer) |

Start with **TruthfulQA** — it is where the phenomenon is densest, and Row 2 is
small by construction. `N = 2000` per dataset is the working default.

## The plan, in dependency order

Each step is a gate or a result, and the early ones are cheap on purpose: three
of them can kill the project before the expensive passes.

| step | question | if it fails |
|---|---|---|
| **Exp 0** | do activations beat output logprobs? | the internal-space framing is unjustified — stop escalating |
| **Gate A** | does the probe read truth, not confidence? (scored on the top-50% confident items) | the instrument is broken; nothing downstream means anything |
| **Gate B** | does the linear map carry the probe? (transfer AUROC within 0.05 of native; also reports `eps`) | single-model story only; drop the cross-model framing |
| **Exp 1** | how big is FA, where does it sit between its bounds, does it shrink with independence? | *the headline* — floor (boundable) vs ceiling (irreducible) |
| **Exp 2** | which false agreements survive both transport directions? | the robust set; a low Jaccard means the map is picking it, not shared belief |
| **Exp 3** | is the false-agreement region linearly separable? | *the bet* — a label-free detector of the method's own blind spot |
| **Exp 4** | does that separation generalize? (5×5 leave-one-dataset-out + held-out pair) | the off-diagonal is the result; failure is the honest cap |
| **Exp 5** | at which depth do belief and false-agreement structure live? | if they differ, do not tune one at the other's layer |
| **Exp 6** | does post-training degrade transport at the last layer? | fit maps at the probe's own layer instead of defaulting to `final` |
| **Exp 7** | does belief after a reasoning trace differ? | *specified, not implemented* — needs generation; see DESIGN.md §6.1 |

Probe types run on the same cached activations at no extra cost: `--probe ccs`
(unsupervised, the case that matters), `mass-mean`, `lr`.

## Cost and time

- Extraction is the only expensive phase, and it is designed to happen **once**:
  3 forward passes per item, every sweep layer out of one pass, cached as `.npz`
  keyed `(model, dataset, n)`.
- ≤8B models: `g5.2xlarge` (1× A10G), ~$0.40–0.60/hr spot. 32B fp16 sharded:
  `g5.12xlarge` (4× A10G), ~$2–2.50/hr spot.
- Everything after extraction reads the cache, runs in seconds, needs no GPU, and
  can be done on a laptop with the `.npz` files copied down.
- fp16 only. **Never 4-bit** — quantization perturbs exactly what the probe reads.

## Kick it off

```bash
scripts/smoke.sh                       # locally, no GPU, ~1 min: does the code work

cd infra                               # bring up the box (see RUNBOOK.md §1)
cp terraform.tfvars.example terraform.tfvars   # fill in key_name + my_ip
terraform init && terraform apply
ssh ubuntu@$(terraform output -raw public_ip)

# on the box — HF_HOME and CMB_CACHE are already pointed at /mnt/data
export HF_TOKEN=...                    # Llama-3.1 is gated
git clone <this repo> && cd cross-model-belief && pip install -r requirements.txt
scripts/kickoff.sh --pair cross-family --dataset truthfulqa --n 2000
```

`scripts/kickoff.sh --dry-run` prints the extraction and experiment plan without
running it. Then write up `results/RESULTS.md` against the DESIGN.md §7 decision
tree and carry the numbers into [`../paper/`](../paper/).
