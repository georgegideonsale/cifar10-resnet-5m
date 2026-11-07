# strat_50: strat_43 (Cutout) trained on all 50,000 images, 75-minute budget

| | |
|---|---|
| Author | George Gideon Sale (`gs4602`) |
| Parent | [strat_43](../strat_43_rm400_sla_cutout) (one change: train on all 50,000 images, as [strat_42](../strat_42_rm400_sla_full) did for strat_40) |
| Status | submitted (seed 0) |
| Kaggle | `strat_50_cutout_full_tta6.csv`, submitted 2026-11-07 04:27 UTC, public LB 0.894, Kaggle ID 58079004 |
| Parameters | 4,338,034 |
| Local accuracy | none: VAL5K is part of the training data |

## Idea

strat_42 (strat_40 on all 50k) gave our best public score. This run applies the same full fit to the Cutout
variant (strat_43), to see whether the two changes add up on the public test set.

## Model and recipe

Code and config of [strat_43](../strat_43_rm400_sla_cutout) with `full_train: true`. Because of the final
deadline, the run used a 75-minute wall-clock budget: the learning-rate schedule (warm-up, then cosine to 0) is
driven by the elapsed fraction of the budget, so the schedule completes in full. It finished 245 of the 400
planned epochs.

| Setting | Value |
|---|---|
| Training images | 50,000 |
| Time budget | `--time-budget-min 75` (245 epochs completed) |
| Seed | 0 |
| Hardware | NYU HPC, A100-SXM4-40GB, 75.0 min of training |

## Inference and results

EMA weights, class head, `tta6`, fixed before training. Public LB 0.894. The VAL5K values in the log are not a
test score: those images were in the training set.

## Reproduce

```bash
cd strategies/strat_50_cutout_full
python strat_50_infer.py --config strat_50_config.yaml --data ../../packed/cifar_comp_v1.npz --out repro --ckpt outputs_s0/model.pt --tta tta6
python strat_50_train.py --config strat_50_config.yaml --data ../../packed/cifar_comp_v1.npz --out outputs_s0 --seed 0 --time-budget-min 75
```

A from-scratch rerun on different hardware stops at a different epoch count, because the budget is in minutes.

## Files

| File | Content |
|---|---|
| `strat_50_*.py`, `strat_50_config.yaml` | model, training, inference, config |
| `logs/s50_0_25890.out` | full training log of seed 0 (Slurm job 25890) |
| `outputs_s0/model.pt` | trained weights (EMA) |
| `outputs_s0/summary.json`, `outputs_s0/metrics.jsonl` | run summary and per-epoch metrics |
| `outputs_s0/submission.csv` | the submitted file, SHA-256 `89e4647dc4998d6f…` |
