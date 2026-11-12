# strat_42: strat_40 trained on all 50,000 images (full fit)

strat_40 is our strongest recipe on VAL5K, but it holds out 5,000 images for validation. This run keeps the
same recipe and trains on those 5,000 images too, so the model sees 11% more data. That's the only change from
the parent, [strat_40](../strat_40_rm400_sla). Every setting was fixed in advance from strat_40, so no choice
depends on the held-out images.

Author: George Gideon Sale (`gs4602`). We submitted seed 0, and it got 0.910 on the public LB, our team's best
public score.

## Model

The same network as [strat_40](../strat_40_rm400_sla): post-activation ResNet (stem 64, widths 64/128/256,
3 basic blocks per stage, y = ReLU(S(x) + F(x))) with a class head `fc` (256 -> 10) and a joint rotation head
`fc_joint` (256 -> 40). Total 4,338,034 parameters, both heads counted.

## What's different in training

Everything is as in [strat_40](../strat_40_rm400_sla#training-recipe), except:

- Training images: 50,000 (all of them; no held-out split).
- 195 steps per epoch (batch 256), so 78,000 steps in 400 epochs.
- Seed 0.
- Hardware: RunPod, NVIDIA RTX PRO 5000 Blackwell, PyTorch 2.8.0, 132.5 min of training.

The config fixes `tta6` before training: EMA weights, class head, mean of the softmax over 6 views (identity,
horizontal flip, and 1-pixel shifts of both). We picked it from the strat_40 results on VAL5K (2-seed mean:
tta6 98.06 > flip2 97.99 > agg_flip2 97.64), never from test images. The training run writes the submitted
`submission.csv` itself at the end.

## Results

There's no local accuracy for this run, because VAL5K is part of the training data. For reference, the recipe
scores VAL5K 98.06 (EMA, tta6) on both strat_40 seeds.

| | Value |
|---|---:|
| Public LB | 0.910 (our team's best public score) |
| strat_40 seed 0, same inference (tta6), for comparison | VAL5K 98.06, public not measured with tta6 |
| strat_40 seed 0, flip2 | public 0.905 |

The file was `strat_42_s0_fullfit_tta6.csv`, submitted 2026-11-07 00:14 UTC (Kaggle ID 58073314).

The log prints VAL5K values near 99.9. Those aren't a test score, since the images were in the training set.
The public gain over strat_40 (+0.5 pp) is inside the public noise (standard error about 0.9 pp).

## Running it

From the checkpoint. A GPU is fastest; a CPU also works, but tta6 runs 6 views and is slower than flip2.

```bash
cd strategies/strat_42_rm400_sla_full
python strat_42_infer.py --config strat_42_config.yaml --data ../../packed/cifar_comp_v1.npz --out repro --ckpt outputs_s0/model.pt --tta tta6
sha256sum repro/submission.csv outputs_s0/submission.csv
```

From scratch (GPU, seed 0):

```bash
python strat_42_train.py --config strat_42_config.yaml --data ../../packed/cifar_comp_v1.npz --out outputs_s0 --seed 0
```

## Files

| File | Content |
|---|---|
| `strat_42_*.py`, `strat_42_config.yaml` | model, training, inference, config (code identical to strat_40) |
| `logs/s42_0_runpod.out` | full training log of seed 0 |
| `outputs_s0/model.pt` | trained weights (EMA), SHA-256 `529685e8d1e4f9da…` |
| `outputs_s0/summary.json`, `outputs_s0/metrics.jsonl` | run summary and per-epoch metrics |
| `outputs_s0/submission.csv` | the submitted file, SHA-256 `642a6a786d549d0f…` |

## References

See [strat_40](../strat_40_rm400_sla#references) (RegMixup, SLA+SD) and the strat_11 references.
