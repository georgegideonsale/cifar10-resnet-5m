# Script conventions

George's strategies (`strat_<N>`) all follow the conventions below. Teammates do not have to, but it makes
folders easier to compare. The code headers say "spec: strategy.md"; in this repo that write-up is the
`README.md` of each strategy folder.

## Files per strategy

```
strat_<N>_config.yaml   every hyperparameter
strat_<N>_model.py      build_model(cfg) -> nn.Module   (tools/count_params.py imports it)
strat_<N>_train.py      training entry point
strat_<N>_infer.py      checkpoint -> submission.csv
logs/                   stdout of the training run
outputs*/               results of one run (outputs_s<seed>/ when several seeds were trained)
```

## Training script

```
python strat_<N>_train.py --config <yaml> --data <npz> --out <dir> [--seed S] [--smoke] [--resume ckpt_last.pt] [--time-budget-min M]
```

- First action: build the model, count all parameters, print them and stop if the total is above 5,000,000.
- `--smoke`: 1-2 epochs on a few hundred images, full code path, all outputs. Runs on a CPU in a few minutes.
- `--resume`: continue from `ckpt_last.pt` (model, optimizer, schedule, EMA, RNG states).
- `--time-budget-min M` (older scripts): the LR schedule follows progress q = max(epoch/E, elapsed/M),
  so the schedule always finishes inside the time budget.
- `--seed` (strat_25 and later): overrides the seed in the config. Older scripts read the seed from the config.
- Data: the packed `.npz` is loaded once and moved to the GPU. Normalization uses training-set statistics.
- Training uses the 45,000 non-VAL5K images unless the config sets `data.full_train: true`.
- Mixed precision: fp16 with a GradScaler on a T4, bf16 on an A100.

## Outputs

| File | Content |
|---|---|
| `submission.csv` | `id,label`, 2,000 rows in `sample_submission.csv` order |
| `summary.json` | parameter count, VAL5K accuracy of every weight/TTA variant, chosen variant, epochs, train time, GPU, versions, seed, config and data SHA-256 |
| `metrics.jsonl` | one line per epoch: loss, train accuracy, VAL5K accuracy (every few epochs), LR, epoch time |
| `model.pt` | state_dict of the submitted weights |
| `*_probs_*.npy`, `*_logits.npy` | saved predictions (only kept where another strategy needs them) |

## Weight and TTA names

- `live`: the weights at the end of training. `ema`: exponential moving average of the weights (decay 0.999).
- `none`: one view. `flip2`: average of the image and its horizontal flip.
- `tta6`: 6 views with weights 0.25 (image), 0.25 (flip), and 0.125 each for 1-pixel shifts (-1,-1) and
  (+1,+1) of the image and of the flip.
- `agg`, `agg_flip2` (strat_38 only): rotation-aggregated prediction of the joint class-rotation head,
  averaged over the 4 rotations; `agg_flip2` also averages the flip.

Which variant is submitted is fixed in the config (`eval.submit_weights`, `eval.submit_tta`) before training.
The early strategies (strat_1, strat_6) picked the best of live/EMA x none/flip2 on VAL5K, which makes their
VAL5K number about 0.1 pp optimistic.

## Data file hashes

`summary.json` stores `data_sha256`, the hash of the `.npz` file used. Runs on Kaggle kernels show `696bab5e...`
and runs on NYU HPC show `5f3f740e...`. The Kaggle kernels rebuilt the `.npz` with `tools/pack_data.py` and
checked that every array hash matched our local file before training, so both contain the same images and the
same VAL5K split. Only the zip container differs.
