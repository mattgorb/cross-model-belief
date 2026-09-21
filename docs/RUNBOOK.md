# Runbook — from nothing to a Row-2 number

Three phases, in order. Phase 2 is the only expensive one, and it is designed to
happen exactly once.

## 0. Check the code without a GPU (locally, 1 min)

```bash
pip install -r requirements.txt
scripts/smoke.sh
```

Runs the tests and the whole pipeline on the synthetic backend. If this fails,
fix it before paying for a GPU.

## 1. Bring up the box

```bash
cd infra
terraform init
terraform apply -var "key_name=YOUR_EC2_KEY" -var "my_ip=$(curl -s ifconfig.me)/32"
ssh ubuntu@$(terraform output -raw ssh | awk '{print $2}')
```

Defaults to `g5.2xlarge` (1× A10G, 24GB) — enough for everything up to 8B. For
the 32B pass:

```bash
terraform apply -var "instance_type=g5.12xlarge" ...   # 4× A10G, 96GB
```

Both are spot. The root volume is 300GB gp3 and holds the HF model cache *and*
the activation cache, so a spot interruption costs you the instance but not the
extraction — provided you put the cache on the volume (next step) and detach
rather than destroy.

On the box:

```bash
git clone <this repo> && cd cross-model-belief
pip install -r requirements.txt
export HF_TOKEN=...                       # Llama-3.1 is gated
export HF_HOME=/mnt/data/hf               # model weights on the big volume
export CMB_CACHE=/mnt/data/activations    # activations on the big volume
```

`CMB_CACHE` is the important one. Leave it unset and the vectors land in the
repo, which is not where you want 300GB of activations.

## 2. Save the vectors (the expensive pass)

```bash
scripts/extract.py --models qwen-7b,llama-8b --datasets truthfulqa --n 2000
scripts/extract.py --list
```

What it writes: one `.npz` per `(model, dataset, n)` under
`$CMB_CACHE/<model>/<dataset>_n<N>.npz`, containing

| key | shape | what |
|---|---|---|
| `pos_l<i>`, `neg_l<i>` | `[N, d]` | mean-pooled hidden states, one array per extracted layer, for the Yes and No halves of the contrast pair |
| `item_ids` | `[N]` | stable ids, checked on load so a cache can never silently misalign |
| `labels` | `[N]` | ground truth |
| `p_yes` | `[N]` | the model's out-loud P(claim is true) — the Exp 0 baseline and the Gate A slice selector |
| `n_layers` | scalar | model depth, so a layer fraction resolves from the cache alone |

Every layer in the sweep comes out of **one** forward pass, so the file holds
`{0.4, 0.5, 0.6, 0.7, 0.8, final} × depth` and the layer sweep later is free.
Add more with `--layers -3,12`.

Cost: 3 forward passes per item (Yes half, No half, P(Yes)). At N=2000 that is
~6k passes per model-dataset. Size is roughly `N × d × n_layers × 2 × 4` bytes —
about 350MB for a 7B at N=2000, ~500MB for the 32B. Budget the 32B pass
separately and confirm it landed (`--list`) before letting the instance go.

Then the 32B, on the bigger box:

```bash
scripts/extract.py --models qwen-32b --datasets truthfulqa --n 2000
```

## 3. Run the experiments (no GPU needed)

Everything from here reads the cache, so it runs in seconds and can be done on
your laptop if you copy the `.npz` files down.

```bash
scripts/run_all.sh --pair cross-family --dataset truthfulqa
```

Or one at a time, in dependency order:

```bash
python3 experiments/exp0_baseline_gate.py --pair cross-family --n 2000
python3 experiments/gate_a_probe_truth.py --pair cross-family --n 2000
python3 experiments/gate_b_transport.py   --pair cross-family --n 2000
python3 experiments/exp1_eight_cell.py    --pair all --n 2000      # the headline
python3 experiments/exp2_bidirectional.py --pair cross-family --n 2000
python3 experiments/exp3_separability.py  --pair cross-family --n 2000
python3 experiments/exp4_generalization_matrix.py --pair cross-family --n 2000 --held-out-pair
python3 experiments/exp5_layer_sweep.py   --pair cross-family --n 2000
```

`--n` must match what you extracted — it is part of the cache filename. A
mismatch re-extracts rather than reusing, which on a real model means an
accidental hour.

`run_all.sh` stops at the first failed gate. That is deliberate: a probe that
reads confidence rather than truth makes every downstream number meaningless.
Use `--keep-going` to see them anyway.

Results land in `results/*.json` plus the printed verdicts. Write up the outcome
in `results/RESULTS.md`, classified against the DESIGN.md §7 decision tree.

## Gotchas

- **Extraction is unbatched** (one item per forward pass). Fine for N≈2000,
  slow past that; batching is the first optimization if you scale up.
- **`--n` is in the cache key.** Extract at the largest N you intend to use;
  experiments can subset but cannot grow into a smaller cache.
- **Don't 4-bit quantize for extraction.** It perturbs exactly what the probe
  reads (DESIGN.md §4). The scripts load fp16 and do not offer a flag for it.
- **The map is currently fitted on the labeled train split only** (~0.6×N pairs
  against d≈4096). Fitting it on a larger unlabeled corpus is the open
  improvement; until then read Gate B's map R² as a conditioning warning.
- **Check `--list` before terminating a spot box.** It is the only thing
  standing between you and re-running the 32B pass.
