# strat_26: deeper ResNet + RegMixup + SAM, 400 epochs

| | |
|---|---|
| Author | George Gideon Sale (`georgesale`) |
| Parent | strat_13 (ResNet + RegMixup, 200 epochs, not in this repo) |
| Status | submitted (seed 0) |
| Kaggle | `3_strat_26_s0_flip.csv`, submitted 2026-11-02 23:52 UTC, public LB 0.898, Kaggle ID 57924531 |
| Parameters | 4,501,570 |
| Local accuracy | VAL5K 97.62 (seed 0, EMA, flip2); 3-seed mean 97.65 |

## Idea

Stack every factor that looked positive in shorter runs: a deeper and slightly narrower ResNet (4 blocks per
stage), RegMixup, sharpness-aware minimization (SAM) and 400 epochs. This changes several factors at once on
purpose, to see how far the combination goes.

## Model

Post-activation ResNet as in [strat_11](../strat_11_s3_recipe_400ep), but with widths 56/112/224 and 4 basic
blocks per stage (stem 3x3 conv 3 -> 56). 4,501,570 parameters.

## Training recipe

As [strat_25](../strat_25_rm_400), with these changes:

| Setting | Value |
|---|---|
| Model | widths 56, 112, 224; blocks 4, 4, 4 |
| SAM | rho = 0.05: each step computes the gradient, moves the weights by rho x g / ‖g‖, recomputes the gradient there and applies it at the original weights. BN running statistics are not updated in the second pass. The mixed batch is the same in both passes. |
| Hardware and time | NYU HPC, A100-SXM4-40GB, 75.8 min of training (SAM doubles the cost per step) |

Unchanged: RegMixup (Beta(20, 20)), SGD Nesterov, LR 0.2 with 2-epoch warm-up and cosine decay, batch 256,
weight decay 5e-4, label smoothing 0.1, crop + flip + TrivialAugmentWide + random erasing 0.25, EMA 0.999,
bf16, 400 epochs.

## Inference

EMA weights, flip2 (fixed in the config).

## Results

| Seed 0 variant | VAL5K |
|---|---:|
| live, one view | 97.38 |
| live, flip2 | 97.60 |
| EMA, one view | 97.38 |
| **EMA, flip2 (submitted)** | **97.62** |
| EMA, tta6 | 97.74 |

| Seed | VAL5K (EMA, flip2) |
|---|---:|
| 0 | 97.62 |
| 1 | 97.56 |
| 2 | 97.78 |

Public LB 0.898 (seed 0). The 3-seed mean is 0.12 pp above strat_25, inside the noise, for 2.3x the training time.

## Reproduce

```bash
cd strategies/strat_26_rm_sam_deep_400
python strat_26_infer.py --config strat_26_config.yaml --data ../../packed/cifar_comp_v1.npz --out repro --ckpt outputs_s0/model.pt
sha256sum repro/submission.csv outputs_s0/submission.csv
```

Checked on 2026-11-03 on a CPU (108 s): byte-identical to the submitted file.

From scratch (GPU, seed 0):

```bash
python strat_26_train.py --config strat_26_config.yaml --data ../../packed/cifar_comp_v1.npz --out outputs_s0 --seed 0
```

## Files

| File | Content |
|---|---|
| `strat_26_*.py`, `strat_26_config.yaml` | model, training, inference, config |
| `logs/26_rm_sam_deep_400_s0_17020.out` | training log of seed 0 (Slurm job 17020) |
| `outputs_s0/model.pt` | submitted weights (EMA) |
| `outputs_s0/submission.csv` | the submitted file |
| `outputs_s0/summary.json`, `outputs_s0/metrics.jsonl` | run summary and per-epoch metrics |

## References

Foret et al. 2021 (SAM); Pinto et al. 2022 (RegMixup); plus the strat_11 references.
