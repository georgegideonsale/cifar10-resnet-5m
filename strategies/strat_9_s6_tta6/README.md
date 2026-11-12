# strat_9: strat_6 with 6-view test-time augmentation

There's no training in this one. [strat_6](../strat_6_wrn16x5_regmixup_s1) (the parent) already saved the
softmax outputs of six test-time views. On VAL5K the 6-view average was 0.20 pp better than the flip average
(97.52 vs 97.32), so we submitted the 6-view prediction of the same network.

- Author: George Gideon Sale (`gs4602`)
- Submitted: `submission_strat_9.csv`, 2026-11-01 02:19 UTC, Kaggle ID 57857631
- Public LB: 0.876
- Parameters: 4,289,754 (the single strat_6 network)
- VAL5K: 97.52 (live weights, tta6)

The network and its training are exactly [strat_6](../strat_6_wrn16x5_regmixup_s1): pre-activation WRN-16-5,
RegMixup, 300 epochs, seed 1, live weights.

## What tta6 does

`tta6` is a weighted average of softmax outputs over 6 views of each test image: the image (0.25), its
horizontal flip (0.25), and 1-pixel shifts (-1,-1) and (+1,+1) of the image and of the flip (0.125 each). Each
view is one forward pass of the same network. No labels and no statistics of the test set are used.

## Results

| Variant (strat_6 network) | VAL5K |
|---|---:|
| live, flip2 | 97.32 |
| live, tta6 (submitted) | 97.52 |

Public LB was 0.876. One caveat: we made this VAL5K comparison after training, on the same 5,000 images, so the
0.20 pp gain is within noise.

## How to reproduce

```bash
python tools/pack_data.py                        # once, from the repo root
python strategies/strat_9_s6_tta6/strat_9_make.py
sha256sum strategies/strat_9_s6_tta6/outputs/submission.csv
```

The script reads `strategies/strat_6_wrn16x5_regmixup_s1/outputs/{test,val}_probs_live_tta6.npy`, prints the
VAL5K accuracy (0.9752) and writes `outputs/submission.csv`. We checked it on 2026-11-03 and the output is
byte-identical to the submitted file. If you want to rebuild the probabilities themselves, you'll need to retrain
strat_6.

## Files

| File | Content |
|---|---|
| `strat_9_make.py` | saved probabilities -> `submission.csv` |
| `outputs/submission.csv` | the submitted file |

## References

Averaging flips and shifts at test time is standard practice. For the network, see the strat_6 references.
