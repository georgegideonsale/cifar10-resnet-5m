# cifar10-resnet-5m

Code, configs, logs and checkpoints of team **Overfitting101** for the NYU Tandon Deep Learning (Fall 2026)
Kaggle competition *CIFAR-10 ResNet Architecture Challenge* (`cifar-10-res-net-architecture-challenge-dl`).

The task: classify a distribution-shifted CIFAR-10 test set (2,000 images, 32x32) with a residual CNN that has
at most 5,000,000 parameters, trained from scratch on the 50,000 training images provided by the course staff.
The metric is accuracy. The final ranking uses the private half of the test set.

Team members: George Gideon Sale (`georgesale`), Tarun (`tarun802`), Bhagyashree Patil (`bhagyashreepatil25`).

Every strategy in this repo is meant to be reproducible on its own: the exact code that produced a Kaggle
submission, its config, the training log, the checkpoint, the parameter count and the commands to rebuild the
`submission.csv`. See [CONTRIBUTING.md](CONTRIBUTING.md) to add yours.

## Submitted strategies

| Strategy | Model | Params | VAL5K | Public LB | Submitted (UTC) |
|---|---|---:|---:|---:|---|
| [strat_1](strategies/strat_1_wrn16x5_base) | Pre-act WRN-16-5, Cutout, 200 ep | 4,289,754 | 96.46 | 86.6 | 2026-10-31 |
| [strat_9](strategies/strat_9_s6_tta6) | strat_6 + 6-view TTA (no training) | 4,289,754 | 97.52 | 87.6 | 2026-11-01 |
| [strat_11](strategies/strat_11_s3_recipe_400ep) | ResNet 3-3-3, TrivialAugment, 400 ep | 4,327,754 | 96.98 | 88.2 | 2026-11-01 |
| [strat_25](strategies/strat_25_rm_400) | strat_11 recipe + RegMixup | 4,327,754 | 97.54 | 89.3 | 2026-11-02 |
| [strat_26](strategies/strat_26_rm_sam_deep_400) | ResNet 4-4-4 + RegMixup + SAM | 4,501,570 | 97.62 | 89.8 | 2026-11-02 |
| [strat_33](strategies/strat_33_rm400_ba) | strat_25 + born-again distillation | 4,327,754 | 97.64 | 89.6 | 2026-11-02 |
| [strat_38](strategies/strat_38_sla_sd) | ResNet 3-3-3 + rotation labels + self-distillation | 4,338,034 | 97.36 | **90.3** | 2026-11-02 |

[strat_6](strategies/strat_6_wrn16x5_regmixup_s1) was not submitted itself. It is here because strat_9 is built
from its saved predictions.

All submitted files, their SHA-256 and the Kaggle submission IDs are listed in [SUBMISSIONS.md](SUBMISSIONS.md).

**How to read the numbers**

- VAL5K is a fixed hold-out of 5,000 training images (500 per class, seed 0) made by `tools/pack_data.py`.
  Every strategy above trains on the other 45,000 images, so the VAL5K numbers are comparable.
  Standard error is about 0.2 pp, so differences below about 0.5 pp need a second seed.
- The public leaderboard uses 1,000 of the 2,000 test images: one image is 0.1 pp and the standard error is
  about 0.9 pp. Small public differences are mostly noise.
- The test set is shifted, so public scores sit 7-10 pp below VAL5K.

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
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Download the competition data into `data/` (see [data/README.md](data/README.md)), then pack it once:

```bash
python tools/pack_data.py
```

This writes `packed/cifar_comp_v1.npz` (all images as uint8 arrays plus the VAL5K indices) and rewrites
`packed/cifar_comp_v1.json`. Check that `val_idx_sha256` is still
`e2dfe1ecb7968a0303636292c48734fec18e706d6ab3309ca2c3c182b35dd609`. If it is, you have the same VAL5K split.
The `sha256` of the `.npz` file itself can differ between machines (zip metadata); the arrays are the same.

## Reproduce a submission from its checkpoint

Inference runs on a CPU in a few minutes. Example for strat_38:

```bash
cd strategies/strat_38_sla_sd
python strat_38_infer.py --config strat_38_config.yaml --data ../../packed/cifar_comp_v1.npz \
    --out repro --ckpt outputs_s1/model.pt
python ../../tools/validate_submission.py repro/submission.csv
```

Each strategy README gives the exact commands for that strategy and says whether the CPU result matched the
submitted file byte for byte.

## Retrain from scratch

Each strategy README gives the training command, the GPU it ran on and the training time. Example:

```bash
cd strategies/strat_25_rm_400
python strat_25_train.py --config strat_25_config.yaml --data ../../packed/cifar_comp_v1.npz --out outputs_s0 --seed 0
```

Training prints the full parameter count first and stops if it is above 5,000,000.

## Making the final submission

The submission we pick must come from a strategy folder in this repo, so that it can be reproduced:

1. Run `python tools/validate_submission.py strategies/<folder>/<outputs>/submission.csv`. It must print `PASS`.
2. Upload exactly that file on Kaggle.
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
- The post-activation ResNet and the TrivialAugment recipe used from strat_11 on come from Tarun's S3 notebook.

No pretrained weights and no external datasets are used anywhere. The only libraries are PyTorch, torchvision,
NumPy, pandas, Pillow and PyYAML.

## License

MIT, see [LICENSE](LICENSE). The competition data is not part of this repo and is not covered by this license.
