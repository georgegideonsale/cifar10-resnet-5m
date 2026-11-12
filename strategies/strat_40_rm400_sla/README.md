# strat_40: RegMixup, 400 epochs, plus rotation labels with self-distillation (SLA+SD)

strat_25 (RegMixup, 400 epochs) and [strat_38](../strat_38_sla_sd) (SLA+SD) each gave a gain on their own, so
here we stack them: the strat_25 recipe with the SLA+SD loss and the joint rotation head of strat_38. The parent
is [strat_25](../strat_25_rm_400) and that's the one change; everything else is equal to strat_25. Run by George
Gideon Sale (`gs4602`).

It reached VAL5K 98.12 (seed 0, EMA, class head, flip2), our highest, and 0.905 on the public LB.

## Model

The same network as [strat_38](../strat_38_sla_sd): post-activation ResNet (stem 64, widths 64/128/256, 3 basic
blocks per stage) with two linear heads on the pooled 256-d feature:

- `fc`: 256 -> 10, the class head.
- `fc_joint`: 256 -> 40, index = class x 4 + rotation.

Total 4,338,034 parameters. Both heads are part of the model and are counted.

## Training recipe

| Setting | Value |
|---|---|
| Training images | 45,000 (VAL5K held out) |
| Epochs | 400 |
| Batch size | 256 images per step, 175 steps per epoch |
| RegMixup | alpha 20: each step uses the clean batch and one mixed batch, loss weighted 0.5 / 0.5 |
| SLA+SD loss | on clean and mixed images, each at 4 rotations (2,048 rotated inputs per step): CE(joint head, 40-way label) + 1.0 x CE(class head, upright image) + 1.0 x KL(aggregated joint ‖ class head), temperature 1, target detached |
| Optimizer | SGD, momentum 0.9, Nesterov, weight decay 5e-4 |
| Learning rate | linear warm-up to 0.2 over 2 epochs, then cosine decay to 0 |
| Label smoothing | 0.1 |
| Augmentation (per image, before rotation) | random crop 32 with 4-pixel zero padding, horizontal flip, TrivialAugmentWide, normalize, random erasing p = 0.25 |
| Regularization | EMA of the weights, decay 0.999. No SAM, no teacher. |
| Precision | bf16 autocast, channels_last |
| Seeds | 0 (submitted), 2 |
| Hardware and time | NYU HPC, A100-SXM4-40GB, 110.2 min of training (seed 0) |

## Which inference we used

Before training, the config fixed `agg_flip2` (rotation-aggregated joint head). On VAL5K, though, the class head
with flip averaging (`flip2`) scored higher on both seeds (seed 0: 98.12 vs 97.56, seed 2: 97.86 vs 97.72). So
the submitted file uses `flip2`: EMA weights, class head, mean of the softmax of the image and its horizontal
flip. We made that choice on VAL5K only, never on test images.

## Results

| Seed 0 variant | VAL5K |
|---|---:|
| EMA, class head, one view | 97.88 |
| EMA, class head, flip2 (submitted) | 98.12 |
| EMA, class head, tta6 | 98.06 |
| EMA, joint head aggregated, flip2 | 97.56 |

| Seed | VAL5K (EMA, flip2) | VAL5K (EMA, tta6) | VAL5K (EMA, agg_flip2) |
|---|---:|---:|---:|
| 0 | 98.12 | 98.06 | 97.56 |
| 2 | 97.86 | 98.06 | 97.72 |

Seed 2 wasn't submitted, so its files aren't in this repo.

Both Kaggle submissions are from seed 0:

| File | Submitted (UTC) | Kaggle ID | Public LB |
|---|---|---|---:|
| `strat_40_s0_flip2.csv` | 2026-11-03 21:44 | 57961146 | 0.905 |
| `strat_40_s0_agg_flip2.csv` | 2026-11-05 22:40 | 58036904 | 0.905 |

The same `agg_flip2` file was uploaded again by mistake at 23:52 UTC (ID 58038132).

Against strat_25 (3 seeds, mean VAL5K 97.53), our 2-seed mean of 97.99 is +0.46 pp. On the public LB, 0.905 is
about equal to strat_38 (0.903). The public score sits where our earlier ResNet runs predict from VAL5K, so the
extra public gain we saw for strat_38 didn't show up here (public standard error about 0.9 pp).

## Reproduce

```bash
cd strategies/strat_40_rm400_sla
python strat_40_infer.py --config strat_40_config.yaml --data ../../packed/cifar_comp_v1.npz --out repro --ckpt outputs_s0/model.pt --tta flip2
sha256sum repro/submission.csv outputs_s0_flip2/submission.csv
```

That's the command that made the submitted file, on a CPU (about 2 minutes), from `outputs_s0/model.pt`.

To train from scratch (GPU, seed 0):

```bash
python strat_40_train.py --config strat_40_config.yaml --data ../../packed/cifar_comp_v1.npz --out outputs_s0 --seed 0
```

## Files

- `strat_40_*.py`, `strat_40_config.yaml`: model (with the joint head), training, inference, config.
- `logs/s40_0_19965.out`: training log of seed 0 (Slurm job 19965).
- `outputs_s0/model.pt`: trained weights (EMA).
- `outputs_s0/summary.json`, `outputs_s0/metrics.jsonl`: run summary and per-epoch metrics.
- `outputs_s0/submission.csv`: the `agg_flip2` file written by the training run (the config default),
  submitted as `strat_40_s0_agg_flip2.csv`.
- `outputs_s0_flip2/submission.csv`: the submitted file (flip2 inference).
- `outputs_s0_flip2/infer_summary.json`: settings of the inference run that made it.

## References

F. Pinto, H. Yang, S. Lim, P. Torr, P. Dokania. Using Mixup as a Regularizer Can Surprisingly Improve Accuracy
and Out-of-Distribution Robustness (RegMixup). NeurIPS 2022.
H. Lee, S. J. Hwang, J. Shin. Self-supervised Label Augmentation via Input Transformations. ICML 2020.
Plus the strat_11 references.
