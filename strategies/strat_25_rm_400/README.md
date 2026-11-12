# strat_25: ResNet recipe + RegMixup, 400 epochs

The plan here was to combine the two best ingredients we had so far: the ResNet + TrivialAugment recipe of
[strat_11](../strat_11_s3_recipe_400ep) and RegMixup from [strat_6](../strat_6_wrn16x5_regmixup_s1), trained
for 400 epochs. We trained three seeds to measure seed noise and submitted seed 0.

Author is George Gideon Sale (`gs4602`). The parent is strat_13, the same recipe at 200 epochs (not in this
repo). Seed 0 went in as `1_strat_25_s0_flip.csv` on 2026-11-02 23:51 UTC (Kaggle ID 57924507) and scored
0.893 on the public LB. VAL5K is 97.54 for seed 0 (EMA, flip2), and the 3-seed mean is 97.53.

## setup

The model is the post-activation ResNet of strat_11 (stem 64, widths 64/128/256, 3 basic blocks per stage),
4,327,754 parameters.

Training:

- 45,000 training images (VAL5K held out), 400 epochs.
- Batch: 256 clean images per step, plus their RegMixup copies (175 steps per epoch).
- SGD, momentum 0.9, Nesterov, weight decay 5e-4.
- LR: linear warm-up to 0.2 over 2 epochs, then cosine decay to 0.
- Loss: 0.5 x CE(clean) + 0.5 x CE(mixed), label smoothing 0.1.
- RegMixup: lambda ~ Beta(20, 20), one draw per step, with partner images from a random permutation of the batch.
- Augmentation (per image): random crop 32 with 4-pixel zero padding, horizontal flip, TrivialAugmentWide,
  normalize, random erasing p = 0.25.
- EMA of the weights, decay 0.999.
- bf16 autocast, channels_last.
- Seeds 0 (submitted), 1, 2.

It ran on NYU HPC on an NVIDIA A100-SXM4-40GB: 32.9 min of training for seed 0, with 11 data-loader workers.

Inference uses the EMA weights and averages the image with its horizontal flip (fixed in the config before
training).

## results

| Seed 0 variant | VAL5K |
|---|---:|
| live, one view | 97.40 |
| live, flip2 | 97.46 |
| EMA, one view | 97.48 |
| EMA, flip2 (submitted) | 97.54 |
| EMA, tta6 | 97.60 |

| Seed | VAL5K (EMA, flip2) |
|---|---:|
| 0 | 97.54 |
| 1 | 97.58 |
| 2 | 97.48 |

Public LB 0.893 (seed 0). Seeds 1 and 2 aren't in this repo because they weren't submitted.

## reproducing

```bash
cd strategies/strat_25_rm_400
python strat_25_infer.py --config strat_25_config.yaml --data ../../packed/cifar_comp_v1.npz --out repro --ckpt outputs_s0/model.pt
sha256sum repro/submission.csv outputs_s0/submission.csv
```

We checked this on 2026-11-03 on a CPU (104 s), and 1 of 2,000 rows differs from the submitted file. The
training run predicted on the GPU in bf16 while the CPU runs in fp32, and that image is a near-tie between two
classes. The other 1,999 labels are identical.

From scratch (GPU, seed 0):

```bash
python strat_25_train.py --config strat_25_config.yaml --data ../../packed/cifar_comp_v1.npz --out outputs_s0 --seed 0
```

## files

| File | Content |
|---|---|
| `strat_25_*.py`, `strat_25_config.yaml` | model, training, inference, config |
| `logs/25_rm_400_s0_17017.out` | training log of seed 0 (Slurm job 17017) |
| `outputs_s0/model.pt` | submitted weights (EMA); also the teacher of [strat_33](../strat_33_rm400_ba) |
| `outputs_s0/submission.csv` | the submitted file |
| `outputs_s0/summary.json`, `outputs_s0/metrics.jsonl` | run summary and per-epoch metrics |

References: Pinto et al. 2022 (RegMixup), plus the strat_11 references.
