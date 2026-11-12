# strat_6: WRN-16-5 with RegMixup, 300 epochs (seed 1)

By George Gideon Sale (`gs4602`). We never submitted this run directly, but [strat_9](../strat_9_s6_tta6)
depends on it: strat_9's submission is built from the probabilities saved here.

It's the strat_1 recipe plus RegMixup. It came to us via strat_3, which is seed 0 of the same recipe and isn't
in this repo. Same 4,289,754 parameters as strat_1. On VAL5K it gets 97.32 (live, flip2) and 97.52 (live,
tta6, the number strat_9 uses).

## What we changed

We added RegMixup (Pinto et al. 2022) to the strat_1 recipe and trained longer. RegMixup keeps the clean
cross-entropy and adds a mixup cross-entropy as a regularizer, which was reported to help both accuracy and
robustness to distribution shift.

The model is the same pre-activation WRN-16-5 as [strat_1](../strat_1_wrn16x5_base). Relative to strat_1, the
differences are:

- RegMixup: each batch is the clean images plus a mixed copy (lambda ~ Beta(20, 20), one draw per batch), and
  loss = 0.5 x clean CE + 0.5 x mixup CE.
- 300 epochs (time budget 185 min, cap 300).
- Seed 1.

Training took 150.7 min on a Kaggle Tesla T4. Everything else (WRN-16-5, SGD Nesterov 0.4 peak with linear
decay, batch 512, label smoothing 0.1, crop + flip + Cutout 16, EMA 0.999, fp16) is unchanged. See
`strat_6_config.yaml`.

## Results

| Variant | VAL5K |
|---|---:|
| live, one view | 97.06 |
| live, flip2 | 97.32 |
| live, tta6 | 97.52 |
| EMA, one view | 96.50 |
| EMA, flip2 | 97.00 |
| EMA, tta6 | 97.26 |

Against strat_1 (flip2), VAL5K is +0.86 pp. Keep in mind that two things changed at once here (RegMixup, and 300
instead of 200 epochs), and the seed changed too. This run wasn't submitted; its saved 6-view probabilities are
the submission of strat_9.

## Running it

```bash
cd strategies/strat_6_wrn16x5_regmixup_s1
python strat_6_infer.py --config strat_6_config.yaml --data ../../packed/cifar_comp_v1.npz --out repro --ckpt outputs/model.pt
```

That rebuilds the flip2 file, `outputs/submission.csv`. We checked it on 2026-11-03 on a CPU (96 s): 1 of 2,000
rows differs. It's a near-tie, and the training run predicted on the GPU in fp16. The 6-view probabilities,
`outputs/test_probs_live_tta6.npy` and `outputs/val_probs_live_tta6.npy`, are written by the training script.

From scratch (GPU):

```bash
python strat_6_train.py --config strat_6_config.yaml --data ../../packed/cifar_comp_v1.npz --out outputs
```

## Files

| File | Content |
|---|---|
| `strat_6_*.py`, `strat_6_config.yaml` | model, training, inference, config |
| `logs/train_stdout.log` | training log |
| `outputs/model.pt` | live weights |
| `outputs/test_probs_live_tta6.npy`, `outputs/val_probs_live_tta6.npy` | 6-view softmax averages, input of strat_9 |
| `outputs/submission.csv` | flip2 predictions (not submitted) |
| `outputs/summary.json`, `outputs/metrics.jsonl` | run summary and per-epoch metrics |

References: Pinto et al. 2022 (RegMixup), plus the strat_1 references.
