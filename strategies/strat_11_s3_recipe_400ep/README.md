# strat_11: ResNet-20-style network with TrivialAugment, 400 epochs

| | |
|---|---|
| Author | George Gideon Sale (`georgesale`); model and recipe from Tarun's S3 notebook |
| Parent | strat_10 (the same recipe at 200 epochs, VAL5K 96.70, not submitted, not in this repo) |
| Status | submitted |
| Kaggle | `submission_strat_11.csv`, submitted 2026-11-01 06:17 UTC, public LB 0.882, Kaggle ID 57862406 |
| Parameters | 4,327,754 |
| Local accuracy | VAL5K 96.98 (EMA weights, flip TTA) |

## Idea

Tarun's S3 recipe (post-activation ResNet, TrivialAugment, random erasing, cosine schedule) scored better on
the public leaderboard than our WRN baseline. strat_10 ported it as-is at 200 epochs; strat_11 changes one
factor: 400 epochs instead of 200.

## Model

Post-activation ResNet with basic blocks, the reference block of the competition rules, y = ReLU(S(x) + F(x)):

- Stem: 3x3 convolution 3 -> 64, BN, ReLU.
- 3 stages of 3 basic blocks each (conv3x3-BN-ReLU-conv3x3-BN, add the skip, ReLU), widths 64, 128, 256.
  The first block of stages 2 and 3 uses stride 2 and a 1x1 convolution + BN skip; all other skips are identity.
- Global average pooling, linear layer 256 -> 10.
- 4,327,754 parameters in total.

## Training recipe

| Setting | Value |
|---|---|
| Training images | 45,000 (VAL5K held out) |
| Epochs | 400 |
| Batch size | 256, last partial batch dropped (175 steps per epoch) |
| Optimizer | SGD, momentum 0.9, Nesterov, weight decay 5e-4 |
| Learning rate | linear warm-up to 0.2 over 2 epochs, then cosine decay to 0 |
| Loss | cross-entropy, label smoothing 0.1 |
| Augmentation (per image, torchvision v2) | random crop 32 with 4-pixel zero padding, horizontal flip, TrivialAugmentWide, normalize, random erasing p = 0.25 |
| Regularization | EMA of the weights, decay 0.999 |
| Precision | fp16 autocast with GradScaler, channels_last |
| Seed | 0 |
| Hardware and time | Kaggle Tesla T4, 103.4 min of training (4 data-loader workers) |

## Inference

EMA weights, average of the image and its horizontal flip. Fixed in the config before training.

## Results

| Variant | VAL5K |
|---|---:|
| live, one view | 97.00 |
| live, flip2 | 97.02 |
| EMA, one view | 97.00 |
| **EMA, flip2 (submitted)** | **96.98** |
| EMA, tta6 | 97.16 |

400 epochs gave +0.28 pp over strat_10 (96.70) on one seed, which is about one standard error. The public
score was higher than strat_9 (88.2 vs 87.6) although VAL5K was lower (96.98 vs 97.52). That single difference
is within the public noise, but all later runs on this recipe kept the pattern: at equal VAL5K, this recipe
scores higher on the shifted test set than the WRN recipe.

## Reproduce

```bash
cd strategies/strat_11_s3_recipe_400ep
python strat_11_infer.py --config strat_11_config.yaml --data ../../packed/cifar_comp_v1.npz --out repro --ckpt outputs/model.pt
sha256sum repro/submission.csv outputs/submission.csv
```

Checked on 2026-11-03 on a CPU (104 s): byte-identical to the submitted file.

From scratch (GPU):

```bash
python strat_11_train.py --config strat_11_config.yaml --data ../../packed/cifar_comp_v1.npz --out outputs
```

## Files

| File | Content |
|---|---|
| `strat_11_*.py`, `strat_11_config.yaml` | model, training, inference, config |
| `logs/train_stdout.log` | training log |
| `outputs/model.pt` | submitted weights (EMA) |
| `outputs/submission.csv` | the submitted file |
| `outputs/summary.json`, `outputs/metrics.jsonl` | run summary and per-epoch metrics |

## References

He et al. 2016 (ResNet); Müller & Hutter 2021 (TrivialAugment); Zhong et al. 2020 (random erasing);
Szegedy et al. 2016 (label smoothing). Model and recipe: Tarun's S3 notebook (team member).
