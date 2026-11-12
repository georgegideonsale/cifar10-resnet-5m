# How our scripts are laid out

All of George's strategies (`strat_<N>`) follow the conventions on this page. Teammates don't have to, though
it does make folders easier to compare. You'll see "spec: strategy.md" in the code headers; in this repo that
write-up is the `README.md` in each strategy folder.

## Files in a strategy folder

```
strat_<N>_config.yaml   every hyperparameter
strat_<N>_model.py      build_model(cfg) -> nn.Module   (tools/count_params.py imports it)
strat_<N>_train.py      training entry point
strat_<N>_infer.py      checkpoint -> submission.csv
logs/                   stdout of the training run
outputs*/               results of one run (outputs_s<seed>/ when several seeds were trained)
```

## The training script

```
python strat_<N>_train.py --config <yaml> --data <npz> --out <dir> [--seed S] [--smoke] [--resume ckpt_last.pt] [--time-budget-min M]
```

Before anything else, the script builds the model, counts all of its parameters, prints the count, and stops if
the total is over 5,000,000.

The options:

- `--smoke` runs 1-2 epochs on a few hundred images. It goes through the full code path and writes all the
  outputs, and it finishes on a CPU in a few minutes.
- `--resume` picks up from `ckpt_last.pt` (model, optimizer, schedule, EMA, RNG states).
- `--time-budget-min M` (older scripts only) makes the LR schedule follow progress q = max(epoch/E, elapsed/M),
  so the schedule always finishes inside the time budget.
- `--seed` (strat_25 and later) overrides the seed in the config. Older scripts take the seed from the config.

A few other things hold for every script. The packed `.npz` gets loaded once and moved to the GPU, and
normalization uses training-set statistics. Training runs on the 45,000 non-VAL5K images unless the config sets
`data.full_train: true`. For mixed precision we use fp16 with a GradScaler on a T4 and bf16 on an A100.

## What a run writes

- `submission.csv`: `id,label`, 2,000 rows in `sample_submission.csv` order.
- `summary.json`: the parameter count, VAL5K accuracy of every weight/TTA variant, the chosen variant, epochs,
  train time, GPU, versions, seed, and the config and data SHA-256.
- `metrics.jsonl`: one line per epoch with loss, train accuracy, VAL5K accuracy (every few epochs), LR and
  epoch time.
- `model.pt`: state_dict of the submitted weights.
- `*_probs_*.npy`, `*_logits.npy`: saved predictions, only kept where another strategy needs them.

## Weight and TTA names

The weights are either `live` (the weights at the end of training) or `ema` (an exponential moving average of
the weights, decay 0.999). The test-time views are named like this:

- `none` is a single view, and `flip2` averages the image with its horizontal flip.
- `tta6` uses 6 views with weights 0.25 (image), 0.25 (flip), and 0.125 each for 1-pixel shifts (-1,-1) and
  (+1,+1) of the image and of the flip.
- `agg` and `agg_flip2` (strat_38, strat_40) are the rotation-aggregated prediction of the joint
  class-rotation head, averaged over the 4 rotations. `agg_flip2` averages the flip as well.

The config fixes which variant gets submitted (`eval.submit_weights`, `eval.submit_tta`) before training
starts. There are two exceptions. The early strategies (strat_1, strat_6) took the best of live/EMA x
none/flip2 on VAL5K, so their VAL5K numbers are about 0.1 pp optimistic. And strat_40 went in with `flip2`
rather than its configured `agg_flip2`, which we chose on VAL5K after training (its README has the details).

## Data file hashes

`summary.json` records `data_sha256`, the hash of the `.npz` file a run used. Runs on Kaggle kernels show
`696bab5e...` and runs on NYU HPC show `5f3f740e...`. The Kaggle kernels rebuilt the `.npz` with
`tools/pack_data.py` and checked, before training, that every array hash matched our local file. So both
versions hold the same images and the same VAL5K split, and only the zip container is different.
