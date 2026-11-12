# cifar10-resnet-5m

This is team Overfitting101's repo for the NYU Tandon Deep Learning (Fall 2026) Kaggle competition,
*CIFAR-10 ResNet Architecture Challenge* (`cifar-10-res-net-architecture-challenge-dl`). It has our code,
configs, logs and checkpoints.

The task is to classify a distribution-shifted CIFAR-10 test set (2,000 images, 32x32) with a residual CNN of at
most 5,000,000 parameters, trained from scratch on the 50,000 training images the course staff gave us. Scoring
is plain accuracy, and the final ranking is on the private half of the test set.

The team is George Gideon Sale (`gs4602`), Tarun (`tk3701`) and Bhagyashree Patil (`bap9622`).

We tried to make every strategy folder reproducible by itself. Each one has the exact code behind a Kaggle
submission, its config, the training log, the checkpoint, the parameter count and the commands to rebuild the
`submission.csv`. If you want to add one of yours, [CONTRIBUTING.md](CONTRIBUTING.md) explains how.

## What we submitted

| Strategy | Model | Params | VAL5K | Public LB | Submitted (UTC) |
|---|---|---:|---:|---:|---|
| [strat_1](strategies/strat_1_wrn16x5_base) | Pre-act WRN-16-5, Cutout, 200 ep | 4,289,754 | 96.46 | 86.6 | 2026-10-31 |
| [strat_9](strategies/strat_9_s6_tta6) | strat_6 + 6-view TTA (no training) | 4,289,754 | 97.52 | 87.6 | 2026-11-01 |
| [strat_11](strategies/strat_11_s3_recipe_400ep) | ResNet 3-3-3, TrivialAugment, 400 ep | 4,327,754 | 96.98 | 88.2 | 2026-11-01 |
| [strat_25](strategies/strat_25_rm_400) | strat_11 recipe + RegMixup | 4,327,754 | 97.54 | 89.3 | 2026-11-02 |
| [strat_26](strategies/strat_26_rm_sam_deep_400) | ResNet 4-4-4 + RegMixup + SAM | 4,501,570 | 97.62 | 89.8 | 2026-11-02 |
| [strat_33](strategies/strat_33_rm400_ba) | strat_25 + born-again distillation | 4,327,754 | 97.64 | 89.6 | 2026-11-02 |
| [strat_38](strategies/strat_38_sla_sd) | ResNet 3-3-3 + rotation labels + self-distillation | 4,338,034 | 97.36 | 90.3 | 2026-11-02 |
| [strat_40](strategies/strat_40_rm400_sla) | strat_25 recipe + rotation labels + self-distillation | 4,338,034 | 98.12 | 90.5 | 2026-11-03 |
| [strat_42](strategies/strat_42_rm400_sla_full) | strat_40 trained on all 50,000 images, 6-view TTA | 4,338,034 | n/a (full fit) | **91.0** | 2026-11-07 |
| [strat_43](strategies/strat_43_rm400_sla_cutout) | strat_40 with Cutout-16 in place of random erasing | 4,338,034 | 97.84 | 89.7 | 2026-11-07 |
| [strat_50](strategies/strat_50_cutout_full) | strat_43 on all 50,000 images, 75-min budget | 4,338,034 | n/a (full fit) | 89.4 | 2026-11-07 |
| [strat_51](strategies/strat_51_wide84_full) | wider ResNet 84-168-336, 2-2-2, all 50,000 images, 75-min budget | 4,794,854 | n/a (full fit) | 89.7 | 2026-11-07 |

The team leader picked our two final selections on Kaggle: strat_42 (public 0.910) and Bhagyashree Patil's
PreAct ResNet submission (public 0.909; notebook [cifar10_hpc.ipynb](cifar10_hpc.ipynb), notes in
[README_cifar10_hpc.md](README_cifar10_hpc.md), file [submission.csv](submission.csv)). There's more on that in
[SUBMISSIONS.md](SUBMISSIONS.md#final-selections).

[strategies/strat_42_rm400_sla_full/strat_42_reproduce.ipynb](strategies/strat_42_rm400_sla_full/strat_42_reproduce.ipynb)
rebuilds our final submission from its checkpoint and prints the parameter count, the validation accuracy of the
recipe and the Kaggle scores.

[strat_6](strategies/strat_6_wrn16x5_regmixup_s1) never got submitted on its own. We kept it because strat_9 is
built from its saved predictions.

Every submitted file, with its SHA-256 and Kaggle submission ID, is in [SUBMISSIONS.md](SUBMISSIONS.md).

### Reading the numbers

- VAL5K is a fixed hold-out of 5,000 training images (500 per class, seed 0), made by `tools/pack_data.py`.
  Apart from the full fits (strat_42, strat_50, strat_51), every strategy above trains on the remaining 45,000
  images, so their VAL5K numbers can be compared. The full fits train on the VAL5K images too, which means they
  have no local score. For reference, the strat_42 recipe gets 98.06 on VAL5K (strat_40, 6-view TTA).
  The standard error is about 0.2 pp, so a difference under about 0.5 pp needs a second seed.
- The public leaderboard only uses 1,000 of the 2,000 test images. One image is worth 0.1 pp and the standard
  error is about 0.9 pp, so small public differences are mostly noise.
- Because the test set is shifted, public scores land 7-10 pp below VAL5K.

## Repository layout

```
README.md            this file
CONTRIBUTING.md      how to add a strategy (steps and checklist)
RULES.md             competition rules that every strategy must satisfy, and our rule decisions
SUBMISSIONS.md       every Kaggle submission: date, member, file, SHA-256, score
docs/workflow.md     the command-line and output conventions our scripts follow
templates/strategy/  README template for a new strategy
tools/               pack_data.py, count_params.py, validate_submission.py
packed/              cifar_comp_v1.json (hashes of the packed data and of the VAL5K split)
data/                competition data (not in git, see data/README.md)
strategies/          one folder per strategy
cifar10_hpc.ipynb    Bhagyashree's PreAct ResNet notebook (README_cifar10_hpc.md, submission.csv)
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Put the competition data in `data/` ([data/README.md](data/README.md) has the steps) and pack it once:

```bash
python tools/pack_data.py
```

That writes `packed/cifar_comp_v1.npz` (every image as a uint8 array, plus the VAL5K indices) and rewrites
`packed/cifar_comp_v1.json`. Check that `val_idx_sha256` still reads
`e2dfe1ecb7968a0303636292c48734fec18e706d6ab3309ca2c3c182b35dd609`; if so, your VAL5K split is the same as ours.
Don't worry if the `sha256` of the `.npz` file itself is different on your machine. That comes from zip
metadata, and the arrays inside are the same.

## Rebuilding a submission from its checkpoint

Inference takes about 2 minutes on a CPU (15 minutes for strat_38). Here's strat_25 as an example:

```bash
cd strategies/strat_25_rm_400
python strat_25_infer.py --config strat_25_config.yaml --data ../../packed/cifar_comp_v1.npz \
    --out repro --ckpt outputs_s0/model.pt
python ../../tools/validate_submission.py repro/submission.csv
```

The README in each strategy folder has the exact commands for that strategy, and says whether the CPU result
matched the submitted file byte for byte.

## Training from scratch

The strategy READMEs also list the training command, the GPU it ran on and how long it took. For example:

```bash
cd strategies/strat_25_rm_400
python strat_25_train.py --config strat_25_config.yaml --data ../../packed/cifar_comp_v1.npz --out outputs_s0 --seed 0
```

The first thing training does is print the full parameter count. It stops if that's above 5,000,000.

## How the final submission gets made

Whatever we pick has to come from a strategy folder in this repo, so someone can reproduce it:

1. Run `python tools/validate_submission.py strategies/<folder>/<outputs>/submission.csv` and make sure it prints `PASS`.
2. Upload exactly that file to Kaggle.
3. Add a row to [SUBMISSIONS.md](SUBMISSIONS.md) with its SHA-256 (`sha256sum <file>`) and the score.

## Citations

- K. He, X. Zhang, S. Ren, J. Sun. Deep Residual Learning for Image Recognition. CVPR 2016.
- K. He, X. Zhang, S. Ren, J. Sun. Identity Mappings in Deep Residual Networks. ECCV 2016.
- S. Zagoruyko, N. Komodakis. Wide Residual Networks. BMVC 2016.
- T. DeVries, G. W. Taylor. Improved Regularization of Convolutional Neural Networks with Cutout. 2017.
- C. Szegedy et al. Rethinking the Inception Architecture for Computer Vision (label smoothing). CVPR 2016.
- S. G. Müller, F. Hutter. TrivialAugment: Tuning-free Yet State-of-the-Art Data Augmentation. ICCV 2021.
- Z. Zhong et al. Random Erasing Data Augmentation. AAAI 2020.
- F. Pinto, H. Yang, S.-N. Lim, P. Torr, P. Dokania. Using Mixup as a Regularizer Can Surprisingly Improve
  Accuracy and Out-of-Distribution Robustness (RegMixup). NeurIPS 2022.
- P. Foret, A. Kleiner, H. Mobahi, B. Neyshabur. Sharpness-Aware Minimization for Efficiently Improving
  Generalization. ICLR 2021.
- T. Furlanello, Z. Lipton, M. Tschannen, L. Itti, A. Anandkumar. Born-Again Neural Networks. ICML 2018.
- G. Hinton, O. Vinyals, J. Dean. Distilling the Knowledge in a Neural Network. 2015.
- H. Lee, S. J. Hwang, J. Shin. Self-supervised Label Augmentation via Input Transformations (SLA+SD). ICML 2020.
- The post-activation ResNet and the TrivialAugment recipe we use from strat_11 onward come from Tarun's S3 notebook.

We don't use pretrained weights or external datasets anywhere. The only libraries are PyTorch, torchvision,
NumPy, pandas, Pillow and PyYAML.

## License

MIT, see [LICENSE](LICENSE). The competition data isn't part of this repo, and the license doesn't cover it.
