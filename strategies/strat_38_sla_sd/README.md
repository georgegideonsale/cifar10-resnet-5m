# strat_38: rotation label augmentation with self-distillation (SLA+SD)

Written up by George Gideon Sale (`gs4602`). The idea of using SLA came from Bhagyashree's recipe. The parent is
strat_12, which is the strat_11 recipe at 200 epochs on an A100 and isn't in this repo. We submitted seed 1.

## Why we tried it

Self-supervised label augmentation (Lee et al. 2020) shows every training image at 4 rotations (0, 90, 180,
270 degrees) and adds a second head that predicts the joint label (class, rotation) out of 40. The usual 10-way
head learns from the unrotated image, and also, by self-distillation, from the joint head's prediction
aggregated over the rotations. We expected the extra self-supervised task to give features that hold up better
under the test shift. Bhagyashree's 90.9 public recipe used it, and it was the one factor of that recipe we
hadn't tested yet.

## Model

It's the post-activation ResNet of [strat_11](../strat_11_s3_recipe_400ep) (stem 64, widths 64/128/256, 3 basic
blocks per stage) with two linear heads on the pooled 256-d feature:

- `fc`: 256 -> 10, the class head.
- `fc_joint`: 256 -> 40, index = class x 4 + rotation (adds 10,280 parameters).

That makes 4,338,034 parameters in total. Both heads are part of the submitted model and both are counted.

## Setup

- Data: 45,000 training images (VAL5K held out). Each augmented image is used at all 4 rotations in the same
  step.
- 200 epochs, batch size 256 images per step (1,024 rotated inputs), 175 steps per epoch.
- Loss: CE(joint head, 40-way label on all 4 rotations) + 1.0 x CE(class head, unrotated image) + 1.0 x
  KL(aggregated joint ‖ class head), temperature 1, aggregated target detached.
- Aggregation: for each image, the mean over rotations k of the joint logit [class, k] at rotation k.
- Optimizer: SGD, momentum 0.9, Nesterov, weight decay 5e-4.
- Learning rate: linear warm-up to 0.2 over 2 epochs, then cosine decay to 0.
- Label smoothing 0.1.
- Augmentation (per image, before rotation): random crop 32 with 4-pixel zero padding, horizontal flip,
  TrivialAugmentWide, normalize, random erasing p = 0.25.
- Regularization: EMA of the weights, decay 0.999. No mixup, no SAM, no teacher.
- Precision: bf16 autocast, channels_last.
- Seeds 0, 1 (submitted) and 2.
- Hardware: NYU HPC, A100-SXM4-40GB, 29.0 min of training (seed 1).

Inference is `agg_flip2` with EMA weights, fixed in the config before training. We run the joint head on the 4
rotations of the image and of its horizontal flip, aggregate over rotations, and average the softmax of the two
views.

## Results

| Seed 1 variant | VAL5K |
|---|---:|
| EMA, class head, one view | 97.22 |
| EMA, class head, flip2 | 97.38 |
| EMA, class head, tta6 | 97.48 |
| EMA, joint head aggregated, one view | 97.28 |
| EMA, joint head aggregated, flip2 (submitted) | 97.36 |

| Seed | VAL5K (EMA, agg_flip2) | VAL5K (EMA, class head, flip2) |
|---|---:|---:|
| 0 | 97.20 | 97.40 |
| 1 | 97.36 | 97.38 |
| 2 | 97.36 | 97.42 |

So the submitted model is VAL5K 97.36 (seed 1, EMA, rotation-aggregated + flip). Seeds 0 and 2 weren't
submitted, which is why their files aren't in this repo.

The submission was `strat_38_s1_agg_flip2.csv`, sent on 2026-11-02 23:48 UTC (Kaggle ID 57924465). It scored
0.903 on the public LB, the best of our submissions at the time, even though its VAL5K is below
strat_25/26/33. At equal VAL5K this strategy scores about 1.2 pp higher on the public test set than our earlier
ResNet runs. With only one public submission that offset is still uncertain (public standard error about
0.9 pp).

## Reproduce

```bash
cd strategies/strat_38_sla_sd
mkdir -p repro        # strat_38_infer.py does not create the output folder
python strat_38_infer.py --config strat_38_config.yaml --data ../../packed/cifar_comp_v1.npz --out repro --ckpt outputs_s1/model.pt
sha256sum repro/submission.csv outputs_s1/submission.csv
```

We checked it on 2026-11-03 on a CPU and the result was byte-identical to the submitted file. The
rotation-aggregated inference does 8 forward passes per image, so on a CPU it takes about 15 minutes.

From scratch (GPU, seed 1):

```bash
python strat_38_train.py --config strat_38_config.yaml --data ../../packed/cifar_comp_v1.npz --out outputs_s1 --seed 1
```

## Files

| File | Content |
|---|---|
| `strat_38_*.py`, `strat_38_config.yaml` | model (with the joint head), training, inference, config |
| `logs/s38_1_19824.out` | training log of seed 1 (Slurm job 19824) |
| `outputs_s1/model.pt` | submitted weights (EMA) |
| `outputs_s1/submission.csv` | the submitted file |
| `outputs_s1/summary.json`, `outputs_s1/metrics.jsonl` | run summary and per-epoch metrics |

## References

H. Lee, S. J. Hwang, J. Shin. Self-supervised Label Augmentation via Input Transformations. ICML 2020.
Plus the strat_11 references.
