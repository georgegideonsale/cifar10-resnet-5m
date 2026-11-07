# strat_51: wider, shallower ResNet (84-168-336, 2-2-2) on all 50,000 images, 75-minute budget

| | |
|---|---|
| Author | George Gideon Sale (`gs4602`) |
| Parent | [strat_42](../strat_42_rm400_sla_full) (one change: widths 84/168/336 with 2 blocks per stage instead of 64/128/256 with 3) |
| Status | submitted (seed 0) |
| Kaggle | `strat_51_wide84_full_tta6.csv`, submitted 2026-11-07 04:28 UTC, public LB 0.897, Kaggle ID 58079007 |
| Parameters | 4,794,854 (under the 5,000,000 limit) |
| Local accuracy | none: VAL5K is part of the training data |

## Idea

Spend the remaining parameter budget on width instead of depth: a wider, shallower network of the same type,
trained with the strat_42 recipe on all 50,000 images.

## Model

Post-activation ResNet with basic blocks (y = ReLU(S(x) + F(x))), stem 84 channels, three stages of 2 blocks with
widths 84/168/336, global average pooling, class head `fc` (336 -> 10) and joint rotation head `fc_joint`
(336 -> 40). Total 4,794,854 parameters, both heads counted.

## Training recipe

The [strat_42](../strat_42_rm400_sla_full) recipe (RegMixup 20, SLA+SD, TrivialAugmentWide, random erasing 0.25,
SGD 0.2 Nesterov, cosine, EMA 0.999) on all 50,000 images, with a 75-minute wall-clock budget that drives the
learning-rate schedule. It finished 154 of the 400 planned epochs (the wider first stage is about 1.6 times slower
per epoch than strat_42).

| Setting | Value |
|---|---|
| Time budget | `--time-budget-min 75` (154 epochs completed) |
| Seed | 0 |
| Hardware | NYU HPC, A100-SXM4-40GB, 75.0 min of training |

## Inference and results

EMA weights, class head, `tta6`, fixed before training. Public LB 0.897. The VAL5K values in the log are not a
test score: those images were in the training set.

## Reproduce

```bash
cd strategies/strat_51_wide84_full
python strat_51_infer.py --config strat_51_config.yaml --data ../../packed/cifar_comp_v1.npz --out repro --ckpt outputs_s0/model.pt --tta tta6
python strat_51_train.py --config strat_51_config.yaml --data ../../packed/cifar_comp_v1.npz --out outputs_s0 --seed 0 --time-budget-min 75
```

## Files

| File | Content |
|---|---|
| `strat_51_*.py`, `strat_51_config.yaml` | model, training, inference, config |
| `logs/s51_0_25891.out` | full training log of seed 0 (Slurm job 25891) |
| `outputs_s0/model.pt` | trained weights (EMA) |
| `outputs_s0/summary.json`, `outputs_s0/metrics.jsonl` | run summary and per-epoch metrics |
| `outputs_s0/submission.csv` | the submitted file, SHA-256 `6826ef666d41cb9b…` |
