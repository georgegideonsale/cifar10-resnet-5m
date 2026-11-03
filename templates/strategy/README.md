# <id>: <one-line title>

| | |
|---|---|
| Author | <name> (`<NYU net ID>`) |
| Parent | <id of the strategy this one changes, or "none"> |
| Status | <submitted / not submitted / dependency of another strategy> |
| Kaggle | <file name on Kaggle>, submitted <YYYY-MM-DD HH:MM UTC>, public LB <score>, Kaggle ID <id> |
| Parameters | <total count, all parameters used at inference> |
| Local accuracy | <VAL5K or your own split, with the TTA used> |

## Idea

Two to four sentences: what you changed compared to the parent and why you expected it to help.

## Model

Architecture (block type, widths, depth, head), with the residual connection described, and the parameter
count. Cite the paper or code it comes from.

## Training recipe

| Setting | Value |
|---|---|
| Training images | 45,000 (VAL5K held out) / 50,000 |
| Epochs | |
| Batch size | |
| Optimizer | e.g. SGD, momentum 0.9, Nesterov, weight decay 5e-4 |
| Learning rate | peak, warm-up, schedule |
| Loss | e.g. cross-entropy with label smoothing 0.1 |
| Augmentation | in the order applied |
| Regularization | mixup / SAM / distillation / EMA ... |
| Precision | fp32 / fp16 / bf16 |
| Seed | |
| Hardware and time | GPU, minutes |

## Inference

Which weights and which test-time augmentation produce the submitted file.

## Results

| Variant | Local accuracy |
|---|---:|
| | |

Public LB: <score>. Anything else you measured.

## Reproduce

From the checkpoint:

```bash
<exact command that writes submission.csv from model.pt>
python tools/validate_submission.py <path>/submission.csv
```

From scratch:

```bash
<exact training command>
```

Say whether the rebuilt file matches the submitted file exactly (`sha256sum`).

## Files

| File | Content |
|---|---|
| | |

## Notes

Limitations, known issues, what you would try next.

## References

Papers and code used (with license if code was copied).
