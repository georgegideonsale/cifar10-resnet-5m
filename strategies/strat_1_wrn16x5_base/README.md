# strat_1: pre-activation WRN-16-5 baseline

| | |
|---|---|
| Author | George Gideon Sale (`georgesale`) |
| Parent | none |
| Status | submitted |
| Kaggle | `submission_strat_1.csv`, submitted 2026-10-31 03:04 UTC, public LB 0.866, Kaggle ID 57824717 |
| Parameters | 4,289,754 |
| Local accuracy | VAL5K 96.46 (live weights, flip TTA) |

## Idea

A first full pipeline: a wide residual network close to the 5M limit, a standard CIFAR recipe (SGD, label
smoothing, crop + flip + Cutout, EMA) and a training length set by a time budget on a free Kaggle T4.

## Model

Pre-activation WRN-16-5 (Zagoruyko & Komodakis 2016; pre-activation blocks from He et al. 2016):

- 3x3 stem convolution, 16 channels.
- 3 stages of 2 pre-activation basic blocks (BN-ReLU-conv3x3-BN-ReLU-conv3x3), widths 80, 160, 320,
  strides 1, 2, 2. Identity skip inside a stage, 1x1 convolution skip at the first block of a stage.
- BN-ReLU, global average pooling, linear layer 320 -> 10.
- 4,289,754 parameters in total. The training script asserts this number.

## Training recipe

| Setting | Value |
|---|---|
| Training images | 45,000 (VAL5K held out) |
| Epochs | 200 (set at the end of epoch 3 from an 80-minute budget, capped at 200) |
| Batch size | 512, last partial batch dropped (87 steps per epoch) |
| Optimizer | SGD, momentum 0.9, Nesterov, weight decay 5e-4 on conv and linear weights (none on BN and biases) |
| Learning rate | linear warm-up 0.04 -> 0.4 over 3 epochs, then linear decay to 0 |
| Loss | cross-entropy, label smoothing 0.1 |
| Augmentation (GPU, per batch) | reflect-pad 4 + random 32x32 crop, horizontal flip p = 0.5, Cutout 16x16 (p = 1) |
| Regularization | EMA of the weights, decay 0.999, started after warm-up |
| Precision | fp16 autocast with GradScaler, channels_last |
| Seed | 0 |
| Hardware and time | Kaggle Tesla T4, 48.0 min of training |

## Inference

Live weights (not EMA), average of the image and its horizontal flip. This variant was the best of
live/EMA x none/flip on VAL5K, so the VAL5K number has a small optimistic bias (about 0.1 pp).

## Results

| Variant | VAL5K |
|---|---:|
| live, one view | 96.16 |
| **live, flip2 (submitted)** | **96.46** |
| live, tta6 | 96.62 |
| EMA, one view | 95.38 |
| EMA, flip2 | 95.42 |

Public LB 0.866.

## Reproduce

From the repo root, after `python tools/pack_data.py`:

```bash
cd strategies/strat_1_wrn16x5_base
python strat_1_infer.py --config strat_1_config.yaml --data ../../packed/cifar_comp_v1.npz --out repro --ckpt outputs/model.pt
sha256sum repro/submission.csv outputs/submission.csv
```

Checked on 2026-11-03 on a CPU (90 s): the rebuilt file is byte-identical to the submitted one.

From scratch (GPU):

```bash
python strat_1_train.py --config strat_1_config.yaml --data ../../packed/cifar_comp_v1.npz --out outputs --time-budget-min 80
```

## Files

| File | Content |
|---|---|
| `strat_1_config.yaml` | all hyperparameters |
| `strat_1_model.py` | WRN-16-5 (`build_model`) |
| `strat_1_train.py` | training, VAL5K evaluation, TTA, writes all outputs |
| `strat_1_infer.py` | checkpoint -> `submission.csv` |
| `logs/train_stdout.log` | training log |
| `outputs/model.pt` | submitted weights (live) |
| `outputs/submission.csv` | the submitted file |
| `outputs/summary.json`, `outputs/metrics.jsonl` | run summary and per-epoch metrics |

## Notes

- EMA weights were worse than the live weights in this run (95.42 vs 96.46 with flip2).
- The large gap between VAL5K (96.5) and public (86.6) shows how strong the test shift is.

## References

He et al. 2016 (ResNet, identity mappings); Zagoruyko & Komodakis 2016 (WRN); DeVries & Taylor 2017 (Cutout);
Szegedy et al. 2016 (label smoothing).
