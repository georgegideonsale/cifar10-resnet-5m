# strat_1: pre-activation WRN-16-5 baseline

Author: George Gideon Sale (`gs4602`). This one has no parent strategy. It was
submitted as `submission_strat_1.csv` on 2026-10-31 03:04 UTC (Kaggle ID 57824717) and got 0.866 on the public
LB. The network has 4,289,754 parameters, and locally it reaches 96.46 on VAL5K (live weights, flip TTA).

## Idea

We wanted a first full pipeline end to end: a wide residual network close to the 5M limit, a standard CIFAR
recipe (SGD, label smoothing, crop + flip + Cutout, EMA), and a training length set by a time budget on a free
Kaggle T4.

## The network

It's a pre-activation WRN-16-5 (Zagoruyko & Komodakis 2016, with the pre-activation blocks from He et al. 2016):

- a 3x3 stem convolution with 16 channels,
- 3 stages of 2 pre-activation basic blocks each (BN-ReLU-conv3x3-BN-ReLU-conv3x3), with widths 80, 160, 320
  and strides 1, 2, 2. Skips are identity inside a stage, and a 1x1 convolution at the first block of a stage,
- then BN-ReLU, global average pooling and a linear layer 320 -> 10.

That comes to 4,289,754 parameters in total, and the training script asserts this number.

## How we trained it

We trained on 45,000 images, holding VAL5K out. The run was 200 epochs: the script set this at the end of epoch
3 from an 80-minute budget, capped at 200. Batch size was 512 with the last partial batch dropped, so 87 steps
per epoch.

The rest of the recipe:

- Optimizer: SGD, momentum 0.9, Nesterov, weight decay 5e-4 on conv and linear weights (none on BN and biases).
- Learning rate: linear warm-up 0.04 -> 0.4 over 3 epochs, then linear decay to 0.
- Loss: cross-entropy with label smoothing 0.1.
- Augmentation (on the GPU, per batch): reflect-pad 4 + random 32x32 crop, horizontal flip with p = 0.5, and
  Cutout 16x16 (p = 1).
- EMA of the weights with decay 0.999, started after warm-up.
- fp16 autocast with GradScaler, channels_last.
- Seed 0.

Training took 48.0 min on a Kaggle Tesla T4.

For inference we used the live weights (not EMA) and averaged the image with its horizontal flip. We picked this
because it was the best of live/EMA x none/flip on VAL5K, which means the VAL5K number has a small optimistic
bias (about 0.1 pp).

## Results

| Variant | VAL5K |
|---|---:|
| live, one view | 96.16 |
| live, flip2 (submitted) | 96.46 |
| live, tta6 | 96.62 |
| EMA, one view | 95.38 |
| EMA, flip2 | 95.42 |

The public LB score was 0.866.

Two things stood out. The EMA weights were worse than the live weights in this run (95.42 vs 96.46 with flip2).
And the large gap between VAL5K (96.5) and public (86.6) shows how strong the test shift is.

## Reproducing the submission

From the repo root, after `python tools/pack_data.py`:

```bash
cd strategies/strat_1_wrn16x5_base
python strat_1_infer.py --config strat_1_config.yaml --data ../../packed/cifar_comp_v1.npz --out repro --ckpt outputs/model.pt
sha256sum repro/submission.csv outputs/submission.csv
```

We checked this on 2026-11-03 on a CPU (90 s), and the rebuilt file is byte-identical to the submitted one.

To train from scratch (GPU):

```bash
python strat_1_train.py --config strat_1_config.yaml --data ../../packed/cifar_comp_v1.npz --out outputs --time-budget-min 80
```

## What's in the folder

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

## References

He et al. 2016 (ResNet, identity mappings); Zagoruyko & Komodakis 2016 (WRN); DeVries & Taylor 2017 (Cutout);
Szegedy et al. 2016 (label smoothing).
