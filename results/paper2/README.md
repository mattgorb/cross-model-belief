# Paper 2 results — transport, the linear map, representational similarity

These are the runs where model B's activations are pushed into A's space by the
fitted ridge map. Paper 1 uses none of them: the map is out of Paper 1, measured
rather than assumed. On the same 228 cells, with the same probe and folds, the
map *cost* 0.012 detector AUROC (paired t = 5.90) and lost on every held-out
dataset, so Paper 1's detector concatenates both models' features in their own
coordinates instead.

| file | what |
|---|---|
| `sweep_loo_lr16.csv` | LOO detection through the map, LR probe, 16 models (505 of 611 cells; resumes) |
| `sweep_loo_all.csv` | LOO through the map, mass-mean, 16 models, 609 cells |
| `sweep_loo_lr.csv` | earlier LOO through the map, LR, 10 models, 94 cells |
| `sweep_loo_lr_w0.csv`, `sweep_loo_lr_w1.csv` | per-fold splits of the above |
| `sweep_loo_mm_capped.csv` | mass-mean through the map, capped direction rows |

Its Paper 1 counterpart is `../sweep_loo_nomap16_lr.csv` (and the `_mm`/`_ccs`
variants): identical except the map, so the pair is a clean A/B on the map alone.

Note `../sweep_base_all.csv` and `../pair_table.csv` stay in Paper 1's folder --
they are Paper 1's main tables -- but their `cka`, `map_r2_*` and `transfer_*`
columns belong to this paper and Paper 1 does not report them.
