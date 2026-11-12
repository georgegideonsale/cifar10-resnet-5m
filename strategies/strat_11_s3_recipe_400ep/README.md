# strat_11: ResNet-20-style network with TrivialAugment, 400 epochs

George Gideon Sale (`gs4602`) ran this; the model and recipe come from Tarun's S3 notebook. Submitted as
`submission_strat_11.csv` on 2026-11-01 06:17 UTC (Kaggle ID 57862406), public LB 0.882. The network has
4,327,754 parameters and gets 96.98 on VAL5K (EMA weights, flip TTA).

## Why this run

Tarun's S3 recipe (post-activation ResNet, TrivialAugment, random erasing, cosine schedule) scored better on the
public leaderboard than our WRN baseline. The parent, strat_10, ported it as-is at 200 epochs and got 96.70 on
VAL5K; it wasn't submitted and isn't in this repo. strat_11 changes one factor: 400 epochs instead of 200.

## Model

A post-activation ResNet with basic blocks, i.e. the reference block of the competition rules,
y = ReLU(S(x) + F(x)):

- Stem: 3x3 convolution 3 -> 64, BN, ReLU.
- 3 stages of 3 basic blocks each (conv3x3-BN-ReLU-conv3x3-BN, add the skip, ReLU), with widths 64, 128, 256.
  The first block of stages 2 and 3 uses stride 2 and a 1x1 convolution + BN skip. All other skips are identity.
- Global average pooling, then a linear layer 256 -> 10.

4,327,754 parameters in total.

## Recipe

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

At inference we use the EMA weights and average the image with its horizontal flip. That choice was fixed in the
config before training.

## Results

| Variant | VAL5K |
|---|---:|
| live, one view | 97.00 |
| live, flip2 | 97.02 |
| EMA, one view | 97.00 |
| EMA, flip2 (submitted) | 96.98 |
| EMA, tta6 | 97.16 |

Going to 400 epochs gave +0.28 pp over strat_10 (96.70) on one seed, which is about one standard error.

The public score was higher than strat_9 (88.2 vs 87.6) even though VAL5K was lower (96.98 vs 97.52). That
single difference is within the public noise. Still, every later run on this recipe kept the same pattern: at
equal VAL5K, this recipe scores higher on the shifted test set than the WRN recipe.

## Reproduce

```bash
cd strategies/strat_11_s3_recipe_400ep
python strat_11_infer.py --config strat_11_config.yaml --data ../../packed/cifar_comp_v1.npz --out repro --ckpt outputs/model.pt
sha256sum repro/submission.csv outputs/submission.csv
```

Checked on 2026-11-03 on a CPU (104 s): byte-identical to the submitted file.

To train from scratch (GPU):

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
Szegedy et al. 2016 (label smoothing). The model and recipe are from Tarun's S3 notebook (Tarun is on the team).
