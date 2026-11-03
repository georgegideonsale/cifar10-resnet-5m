# strat_6: WRN-16-5 with RegMixup, 300 epochs (seed 1)

| | |
|---|---|
| Author | George Gideon Sale (`georgesale`) |
| Parent | strat_1 recipe plus RegMixup (via strat_3, seed 0 of the same recipe, not in this repo) |
| Status | not submitted; **dependency of [strat_9](../strat_9_s6_tta6)** |
| Parameters | 4,289,754 |
| Local accuracy | VAL5K 97.32 (live, flip2); 97.52 (live, tta6, used by strat_9) |

## Idea

Add RegMixup (Pinto et al. 2022) to the strat_1 recipe and train longer. RegMixup keeps the clean
cross-entropy and adds a mixup cross-entropy as a regularizer, which was reported to help both accuracy and
robustness to distribution shift.

## Model

Same pre-activation WRN-16-5 as [strat_1](../strat_1_wrn16x5_base), 4,289,754 parameters.

## Training recipe

As strat_1, with these changes:

| Setting | Value |
|---|---|
| RegMixup | each batch is the clean images plus a mixed copy (lambda ~ Beta(20, 20), one draw per batch); loss = 0.5 x clean CE + 0.5 x mixup CE |
| Epochs | 300 (time budget 185 min, cap 300) |
| Seed | 1 |
| Hardware and time | Kaggle Tesla T4, 150.7 min of training |

Everything else (WRN-16-5, SGD Nesterov 0.4 peak with linear decay, batch 512, label smoothing 0.1,
crop + flip + Cutout 16, EMA 0.999, fp16) is unchanged. See `strat_6_config.yaml`.

## Results

| Variant | VAL5K |
|---|---:|
| live, one view | 97.06 |
| live, flip2 | 97.32 |
| live, tta6 | 97.52 |
| EMA, one view | 96.50 |
| EMA, flip2 | 97.00 |
| EMA, tta6 | 97.26 |

Compared with strat_1 (flip2), VAL5K is +0.86 pp. Two things changed at once here (RegMixup, and 300 instead of
200 epochs), plus the seed. This run was not submitted; its saved 6-view probabilities are the submission of
strat_9.

## Reproduce

```bash
cd strategies/strat_6_wrn16x5_regmixup_s1
python strat_6_infer.py --config strat_6_config.yaml --data ../../packed/cifar_comp_v1.npz --out repro --ckpt outputs/model.pt
```

This rebuilds the flip2 file `outputs/submission.csv`. Checked on 2026-11-03 on a CPU (96 s): 1 of 2,000 rows
differs (a near-tie; the training run predicted on the GPU in fp16). The 6-view probabilities `outputs/test_probs_live_tta6.npy`
and `outputs/val_probs_live_tta6.npy` are written by the training script.

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

## References

Pinto et al. 2022 (RegMixup); plus the strat_1 references.
