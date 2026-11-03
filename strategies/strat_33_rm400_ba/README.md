# strat_33: born-again distillation from strat_25

| | |
|---|---|
| Author | George Gideon Sale (`georgesale`) |
| Parent | [strat_25](../strat_25_rm_400) |
| Status | submitted (seed 0) |
| Kaggle | `4_strat_33_s0_flip.csv`, submitted 2026-11-02 23:52 UTC, public LB 0.896, Kaggle ID 57924537 |
| Parameters | 4,327,754 (student; only the student is used at inference) |
| Local accuracy | VAL5K 97.64 (seed 0, EMA, flip2); 3-seed mean 97.59 |

## Idea

Born-again networks (Furlanello et al. 2018): train a student with the same architecture as the teacher on the
same data, using the teacher's soft predictions as an extra target. The teacher is our own strat_25 seed 0
model, trained from scratch on the same 45,000 images, so this is allowed (Rules §5). One factor changes
compared with strat_25: the distillation loss.

## Model

Student and teacher are the post-activation ResNet of strat_25 (stem 64, widths 64/128/256, 3 basic blocks per
stage), 4,327,754 parameters each. The teacher is frozen during training and is not part of the submitted
model. The training script asserts that the teacher also has at most 5M parameters.

## Training recipe

As [strat_25](../strat_25_rm_400), plus distillation:

| Setting | Value |
|---|---|
| Teacher | `../strat_25_rm_400/outputs_s0/model.pt` (strat_25 seed 0, EMA weights; SHA-256 starts with `61c1e3e88a55`) |
| Distillation loss | loss = 0.5 x (strat_25 RegMixup loss) + 0.5 x T² x KL(teacher ‖ student) at temperature T = 4, on the same (clean + mixed) batch |
| Hardware and time | NYU HPC, A100-SXM4-40GB, 40.8 min of training (seed 0) |

Unchanged: RegMixup (Beta(20, 20)), SGD Nesterov, LR 0.2 with 2-epoch warm-up and cosine decay, batch 256,
weight decay 5e-4, label smoothing 0.1, crop + flip + TrivialAugmentWide + random erasing 0.25, EMA 0.999,
bf16, 400 epochs.

## Inference

EMA weights of the student, flip2 (fixed in the config).

## Results

| Seed 0 variant | VAL5K |
|---|---:|
| live, one view | 97.40 |
| live, flip2 | 97.66 |
| EMA, one view | 97.50 |
| **EMA, flip2 (submitted)** | **97.64** |
| EMA, tta6 | 97.64 |

| Seed | VAL5K (EMA, flip2) |
|---|---:|
| 0 | 97.64 |
| 1 | 97.58 |
| 2 | 97.56 |

Public LB 0.896 (seed 0). The 3-seed mean is 0.06 pp above strat_25: distillation from one same-size teacher
gave no measurable gain.

## Reproduce

```bash
cd strategies/strat_33_rm400_ba
python strat_33_infer.py --config strat_33_config.yaml --data ../../packed/cifar_comp_v1.npz --out repro --ckpt outputs_s0/model.pt
sha256sum repro/submission.csv outputs_s0/submission.csv
```

Checked on 2026-11-03 on a CPU (105 s): byte-identical to the submitted file.

From scratch (GPU, seed 0). Run it from this folder, because the config finds the teacher by a relative path:

```bash
cd strategies/strat_33_rm400_ba
python strat_33_train.py --config strat_33_config.yaml --data ../../packed/cifar_comp_v1.npz --out outputs_s0 --seed 0
```

## Files

| File | Content |
|---|---|
| `strat_33_*.py`, `strat_33_config.yaml` | model, training, inference, config |
| `logs/33_rm400_ba_s0_17115.out` | training log of seed 0 (Slurm job 17115), includes the teacher hash |
| `outputs_s0/model.pt` | submitted student weights (EMA) |
| `outputs_s0/submission.csv` | the submitted file |
| `outputs_s0/summary.json`, `outputs_s0/metrics.jsonl` | run summary and per-epoch metrics |

## References

Furlanello et al. 2018 (born-again networks); Hinton et al. 2015 (knowledge distillation); plus the strat_25
references.
