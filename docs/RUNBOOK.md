# Runbook — from nothing to a Row-2 number

Three phases, in order. Phase 2 is the only expensive one, and it is designed to
happen exactly once. For *what* is being run and why, read the one-page
[experiment plan](EXPERIMENT_PLAN.md) first.

In a hurry, on a box that is already up: `scripts/kickoff.sh --pair cross-family
--dataset truthfulqa --n 2000` does phases 2 and 3 in one command (and
`--dry-run` prints the plan first).

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
cp terraform.tfvars.example terraform.tfvars    # fill in key_name and my_ip
terraform init
terraform apply
ssh ubuntu@$(terraform output -raw public_ip)
```

Needs AWS credentials in the environment (`aws configure`, or `AWS_PROFILE=...`)
and an EC2 keypair that already exists in the region.

Defaults to `g5.2xlarge` (1× A10G, 24GB) — enough for everything up to 8B. For
the 32B pass:

```bash
terraform apply -var "instance_type=g5.12xlarge"   # 4× A10G, 96GB
```

Both are spot. Two resources with deliberately different lifetimes:

| resource | what | lifetime |
|---|---|---|
| `aws_instance.gpu` | the spot box | disposable |
| `aws_ebs_volume.data` | `/mnt/data` — HF weights + activation cache | **survives the box** (`prevent_destroy`) |

So a spot interruption costs the instance and not the vectors. To swap the box
(e.g. up to the 32B instance) while keeping the extraction:

```bash
terraform destroy -target=aws_instance.gpu
terraform apply -var "instance_type=g5.12xlarge"
```

The volume is found by *volume id* and formatted only if it has no filesystem, so
reattaching is safe; on g5 the ephemeral instance store also appears as an NVMe
device, and the lookup deliberately avoids it. To reuse a volume from a torn-down
state: `terraform output data_volume_id`, then
`-var "existing_data_volume_id=vol-..." -var "availability_zone=<its AZ>"`.

On the box — cloud-init has already mounted `/mnt/data` and exported `HF_HOME`
and `CMB_CACHE` into `~/.bashrc`, so a fresh login has them:

```bash
git clone <this repo> && cd cross-model-belief
pip install -r requirements.txt
export HF_TOKEN=...                       # Llama-3.1 is gated
echo "$CMB_CACHE"                         # expect /mnt/data/activations
df -h /mnt/data                           # expect the 300GB volume, not the root disk
```

`CMB_CACHE` is the important one. If it is empty, cloud-init did not finish
(`sudo tail /var/log/cloud-init-output.log`) — fix that before extracting, or the
vectors land on the disposable root volume.

## 2. Save the vectors (the expensive pass)

```bash
scripts/extract.py --models qwen3-8b,gemma4-12b --datasets truthfulqa --n 2000
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
scripts/extract.py --models qwen3-32b --datasets truthfulqa --n 2000
```

## 3. Run the experiments (no GPU needed)

Everything from here reads the cache, so it runs in seconds and can be done on
your laptop if you copy the `.npz` files down.

```bash
scripts/kickoff.sh --pair cross-family --dataset truthfulqa --n 2000
```

That is phases 2 and 3 together: it resolves which models the pair needs from
`cmb/config.py`, extracts them, verifies the cache, then runs every experiment in
dependency order. `--dry-run` prints the plan and touches nothing.

Off a warm cache, `scripts/run_all.sh --pair cross-family --dataset truthfulqa`
runs just the experiments. Or one at a time, in dependency order:

```bash
python3 experiments/exp0_baseline_gate.py --pair cross-family --n 2000
python3 experiments/gate_a_probe_truth.py --pair cross-family --n 2000
python3 experiments/gate_b_transport.py   --pair cross-family --n 2000
python3 experiments/exp1_eight_cell.py    --pair all --n 2000      # the headline
python3 experiments/exp2_bidirectional.py --pair cross-family --n 2000
python3 experiments/exp3_separability.py  --pair cross-family --n 2000
python3 experiments/exp4_generalization_matrix.py --pair cross-family --n 2000 --held-out-pair
python3 experiments/exp5_layer_sweep.py   --pair cross-family --n 2000
python3 experiments/exp6_post_training.py --n 2000                 # base vs instruct
```

Add `--probe mass-mean` (or `--probe lr`) to any of these to run the labeled
baselines on the same cached activations; the cache is probe-independent, so this
costs nothing extra. Exp 6 needs the base checkpoints extracted as well
(`qwen3-8b-base`), which is one more model-sized pass.

`--n` must match what you extracted — it is part of the cache filename. A
mismatch re-extracts rather than reusing, which on a real model means an
accidental hour.

`run_all.sh` stops at the first failed gate. That is deliberate: a probe that
reads confidence rather than truth makes every downstream number meaningless.
Use `--keep-going` to see them anyway.

Results land in `results/*.json` plus the printed verdicts. Write up the outcome
in `results/RESULTS.md`, classified against the DESIGN.md §7 decision tree, then
carry the numbers into `paper/` (the results sentence in `paper/sections/abstract.tex` is
deliberately left blank until Exp 1 has a number).

## Gotchas

- **One backend per model, freed before the next.** `extract.py` loads a model
  once and reuses it across that model's datasets. Loading per dataset leaked the
  previous copy's VRAM, so the second load found the card full, silently offloaded
  layers to CPU ("Some parameters are on the meta device") and then OOMed
  mid-pass. If you see that message, weights are spilling to host RAM and the
  pass will be slow or die.
- **Extraction is unbatched** (one item per forward pass). Fine for N≈2000,
  slow past that; batching is the first optimization if you scale up.
- **`--n` is in the cache key.** Extract at the largest N you intend to use;
  experiments can subset but cannot grow into a smaller cache.
- **Don't 4-bit quantize for extraction.** It perturbs exactly what the probe
  reads (DESIGN.md §4). The scripts load fp16 and do not offer a flag for it.
- **The map is currently fitted on the labeled train split only** (~0.6×N pairs
  against d≈4096). Fitting it on a larger unlabeled corpus is the open
  improvement; until then read Gate B's map R² as a conditioning warning.
- **`CUDNN_STATUS_SUBLIBRARY_LOADING_FAILED` on a very new architecture.** Run
  with `CMB_DISABLE_CUDNN=1`. Extraction is forward matmuls, so cuDNN is not
  needed; the failure is a mismatch between the installed cuDNN build and an op
  path the model wants, not a real dependency. Seen on `qwen38-27b` with the
  torch 2.7 / CUDA 12.8 DLAMI.
- **`CUDNN_STATUS_SUBLIBRARY_LOADING_FAILED` on a very new architecture.** Run
  with `CMB_DISABLE_CUDNN=1`. Extraction is forward matmuls, so cuDNN is not a
  real dependency here; the failure is a mismatch between the installed cuDNN
  build and an op path the model wants. Seen on `qwen38-27b` with the torch 2.7
  / CUDA 12.8 DLAMI.
- **Check `--list` before terminating a spot box.** It is the only thing
  standing between you and re-running the 32B pass.
