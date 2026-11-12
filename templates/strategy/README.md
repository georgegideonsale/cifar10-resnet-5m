# <id>: <one-line title>

- Author: <name> (`<NYU net ID>`)
- Parent: <id of the strategy this one changes, or "none">
- Status: <submitted / not submitted / dependency of another strategy>
- Kaggle: <file name on Kaggle>, submitted <YYYY-MM-DD HH:MM UTC>, public LB <score>, Kaggle ID <id>
- Parameters: <total count, all parameters used at inference>
- Local accuracy: <VAL5K or your own split, with the TTA used>

## Idea

A few sentences (two to four is plenty) on what you changed relative to the parent, and why you thought it
would help.

## Model

Describe the architecture: block type, widths, depth and head, including how the residual connection works,
plus the parameter count. Cite the paper or code it's based on.

## Training recipe

Fill in each line (delete the hints in brackets):

- Training images: 45,000 (VAL5K held out) / 50,000
- Epochs:
- Batch size:
- Optimizer: (e.g. SGD, momentum 0.9, Nesterov, weight decay 5e-4)
- Learning rate: (peak, warm-up, schedule)
- Loss: (e.g. cross-entropy with label smoothing 0.1)
- Augmentation: (in the order applied)
- Regularization: (mixup / SAM / distillation / EMA ...)
- Precision: (fp32 / fp16 / bf16)
- Seed:
- Hardware and time: (GPU, minutes)

## Inference

Which weights and which test-time augmentation produced the submitted file.

## Results

| Variant | Local accuracy |
|---|---:|
| | |

Public LB: <score>. Add anything else you measured.

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

Say whether the rebuilt file matches the submitted one exactly (`sha256sum`).

## Files

List each file in the folder and what's in it, for example:

- `<file>`: <content>

## Notes

Limitations, known issues, and what you'd try next.

## References

Papers and code you used (with the license, if you copied code).
