# strat_43: strat_40 with Cutout-16 in place of random erasing

Cutout (DeVries and Taylor, 2017) is a common strong regularizer on CIFAR-10. Here we test whether a fixed 16x16
Cutout on every image helps the strat_40 recipe more than the random erasing it used before. So the one change
from the parent, [strat_40](../strat_40_rm400_sla), is that Cutout 16x16 at p = 1 replaces random erasing
p = 0.25.

George Gideon Sale (`gs4602`) ran it. Seed 0 was submitted.

## What changed

The network is the same as in [strat_40](../strat_40_rm400_sla#model): post-activation ResNet 3-3-3, widths
64/128/256, class head and joint rotation head, 4,338,034 parameters.

Training is identical to [strat_40](../strat_40_rm400_sla#training-recipe) apart from the augmentation. Per
image we do random crop 32 with 4-pixel zero padding, horizontal flip, TrivialAugmentWide, normalize, then
Cutout 16x16 with p = 1 (centre uniform over the image, clipped at the borders, fill 0 after normalization).
Other details of this run:

- 45,000 training images (VAL5K held out).
- Seed 0.
- RunPod, NVIDIA A100 80GB PCIe, PyTorch 2.4.1, 116.7 min of training.

Inference was fixed before training: EMA weights, class head, `tta6` (identity, flip, 1-pixel shifts).

## Results

| VAL5K (EMA) | strat_43 seed 0 | strat_40 seed 0 | strat_40 seed 2 |
|---|---:|---:|---:|
| one view | 97.78 | 97.88 | 97.60 |
| flip2 | 97.76 | 98.12 | 97.86 |
| tta6 (submitted) | 97.84 | 98.06 | 98.06 |

Cutout gave no gain over random erasing here: -0.22 pp on tta6, inside the VAL5K noise. The submitted model
is VAL5K 97.84 (seed 0, EMA, class head, tta6).

On Kaggle it's `strat_43_s0_cutout_tta6.csv`, submitted 2026-11-07 04:27 UTC, Kaggle ID 58079001, public LB
0.897.

## Reproduce

```bash
cd strategies/strat_43_rm400_sla_cutout
python strat_43_infer.py --config strat_43_config.yaml --data ../../packed/cifar_comp_v1.npz --out repro --ckpt outputs_s0/model.pt --tta tta6
python strat_43_train.py --config strat_43_config.yaml --data ../../packed/cifar_comp_v1.npz --out outputs_s0 --seed 0
```

## Files

- `strat_43_*.py`, `strat_43_config.yaml`: model, training, inference, config.
- `logs/s43_0_runpod.out`: full training log of seed 0.
- `outputs_s0/model.pt`: trained weights (EMA).
- `outputs_s0/summary.json`, `outputs_s0/metrics.jsonl`: run summary and per-epoch metrics.
- `outputs_s0/submission.csv`: the submitted file, SHA-256 `0933478b055f17f5…`.

## References

T. DeVries, G. W. Taylor. Improved Regularization of Convolutional Neural Networks with Cutout. arXiv 1708.04552, 2017.
Plus the [strat_40](../strat_40_rm400_sla#references) references.
