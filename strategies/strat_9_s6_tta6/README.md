# strat_9: strat_6 with 6-view test-time augmentation

| | |
|---|---|
| Author | George Gideon Sale (`georgesale`) |
| Parent | [strat_6](../strat_6_wrn16x5_regmixup_s1) |
| Status | submitted |
| Kaggle | `submission_strat_9.csv`, submitted 2026-11-01 02:19 UTC, public LB 0.876, Kaggle ID 57857631 |
| Parameters | 4,289,754 (the single strat_6 network) |
| Local accuracy | VAL5K 97.52 (live weights, tta6) |

## Idea

No training. strat_6 already saved the softmax outputs of six test-time views. On VAL5K the 6-view average was
0.20 pp better than the flip average (97.52 vs 97.32), so this strategy submits the 6-view prediction of the
same network.

## Model and training

Exactly [strat_6](../strat_6_wrn16x5_regmixup_s1): pre-activation WRN-16-5, RegMixup, 300 epochs, seed 1,
live weights.

## Inference

`tta6`: weighted average of softmax outputs over 6 views of each test image: the image (0.25), its horizontal
flip (0.25), and 1-pixel shifts (-1,-1) and (+1,+1) of the image and of the flip (0.125 each). Each view is one
forward pass of the same network; no labels and no statistics of the test set are used.

## Results

| Variant (strat_6 network) | VAL5K |
|---|---:|
| live, flip2 | 97.32 |
| **live, tta6 (submitted)** | **97.52** |

Public LB 0.876. Note that this VAL5K comparison was made after training, on the same 5,000 images, so the
0.20 pp gain is within noise.

## Reproduce

```bash
python tools/pack_data.py                        # once, from the repo root
python strategies/strat_9_s6_tta6/strat_9_make.py
sha256sum strategies/strat_9_s6_tta6/outputs/submission.csv
```

The script reads `strategies/strat_6_wrn16x5_regmixup_s1/outputs/{test,val}_probs_live_tta6.npy`, prints the
VAL5K accuracy (0.9752) and writes `outputs/submission.csv`. Checked on 2026-11-03: the output is
byte-identical to the submitted file. To rebuild the probabilities themselves, retrain strat_6.

## Files

| File | Content |
|---|---|
| `strat_9_make.py` | saved probabilities -> `submission.csv` |
| `outputs/submission.csv` | the submitted file |

## References

Test-time augmentation by averaging flips and shifts is standard practice; see the strat_6 references for the
network.
