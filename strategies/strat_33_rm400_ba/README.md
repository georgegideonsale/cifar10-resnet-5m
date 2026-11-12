# strat_33: born-again distillation from strat_25

Run by George Gideon Sale (`gs4602`). Parent is [strat_25](../strat_25_rm_400). Status: submitted (seed 0).

Seed 0 scored VAL5K 97.64 (EMA, flip2) and 0.896 on the public LB. Over 3 seeds the VAL5K mean is 97.59.

## Idea

This is a born-again network (Furlanello et al. 2018). We train a student with the same architecture as the
teacher, on the same data, and give it the teacher's soft predictions as an extra target. The teacher is our
own strat_25 seed 0 model, trained from scratch on the same 45,000 images, so this is allowed (Rules §5).
Compared with strat_25 only one factor changes: the distillation loss.

## The network

Student and teacher are both the post-activation ResNet of strat_25 (stem 64, widths 64/128/256, 3 basic blocks
per stage), with 4,327,754 parameters each. The teacher is frozen during training and isn't part of the
submitted model, so only the student's 4,327,754 parameters are used at inference. The training script also
asserts that the teacher has at most 5M parameters.

## Training

Same as [strat_25](../strat_25_rm_400), with distillation added:

- Teacher: `../strat_25_rm_400/outputs_s0/model.pt` (strat_25 seed 0, EMA weights; SHA-256 starts with
  `61c1e3e88a55`).
- Loss: loss = 0.5 x (strat_25 RegMixup loss) + 0.5 x T² x KL(teacher ‖ student) at temperature T = 4, on the
  same (clean + mixed) batch.
- Hardware: NYU HPC, A100-SXM4-40GB, 40.8 min of training for seed 0.

Nothing else moved: RegMixup (Beta(20, 20)), SGD Nesterov, LR 0.2 with 2-epoch warm-up and cosine decay,
batch 256, weight decay 5e-4, label smoothing 0.1, crop + flip + TrivialAugmentWide + random erasing 0.25,
EMA 0.999, bf16, 400 epochs.

For inference we take the student's EMA weights with flip2 (fixed in the config).

## Results

| Seed 0 variant | VAL5K |
|---|---:|
| live, one view | 97.40 |
| live, flip2 | 97.66 |
| EMA, one view | 97.50 |
| EMA, flip2 (submitted) | 97.64 |
| EMA, tta6 | 97.64 |

| Seed | VAL5K (EMA, flip2) |
|---|---:|
| 0 | 97.64 |
| 1 | 97.58 |
| 2 | 97.56 |

We submitted seed 0 as `4_strat_33_s0_flip.csv` on 2026-11-02 23:52 UTC (Kaggle ID 57924537) and it got 0.896
on the public LB. The 3-seed mean is just 0.06 pp above strat_25, so distilling from one same-size teacher gave
no measurable gain.

## How to reproduce

```bash
cd strategies/strat_33_rm400_ba
python strat_33_infer.py --config strat_33_config.yaml --data ../../packed/cifar_comp_v1.npz --out repro --ckpt outputs_s0/model.pt
sha256sum repro/submission.csv outputs_s0/submission.csv
```

We checked this on 2026-11-03 on a CPU (105 s) and got a file byte-identical to the submitted one.

Training from scratch (GPU, seed 0) has to run from this folder, because the config finds the teacher by a
relative path:

```bash
cd strategies/strat_33_rm400_ba
python strat_33_train.py --config strat_33_config.yaml --data ../../packed/cifar_comp_v1.npz --out outputs_s0 --seed 0
```

## Files

- `strat_33_*.py`, `strat_33_config.yaml`: model, training, inference, config.
- `logs/33_rm400_ba_s0_17115.out`: training log of seed 0 (Slurm job 17115). It includes the teacher hash.
- `outputs_s0/model.pt`: the submitted student weights (EMA).
- `outputs_s0/submission.csv`: the submitted file.
- `outputs_s0/summary.json`, `outputs_s0/metrics.jsonl`: run summary and per-epoch metrics.

## References

Furlanello et al. 2018 (born-again networks); Hinton et al. 2015 (knowledge distillation); plus the strat_25
references.
