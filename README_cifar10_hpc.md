# CIFAR-10 NYU HPC Submission Notebook

This README explains the supplied [`cifar10_hpc.ipynb`](/Users/bhagyashreepatil/Downloads/DL-Kaggle-1/MAIN-GOOD-OUTPUT/cifar10_hpc.ipynb) notebook and the [`submission.csv`](https://github.com/georgegideonsale/cifar10-resnet-5m/blob/bap9622/submission.csv) file.

The notebook trains a CIFAR-10 classifier for the NYU Kaggle competition. It uses only the course-provided training images. It first compares two training lengths on a fixed validation split, chooses the better one, retrains that one model on all 50,000 labelled images, predicts the 2,000 unlabelled test images, and writes a Kaggle submission file.

The code does not load pretrained weights. It does not use external images, external labels, pseudo-labels, test labels, manual test-image labels, test-time fine-tuning, or a prediction ensemble. It submits one trained model.

## What the completed notebook ran

The saved notebook output shows that the code completed two matched validation runs and then completed the final full-data run.

| Run name | Training set | Epochs | Trainable parameters | Validation accuracy |
|---|---:|---:|---:|---:|
| `sla_e200` | 45,000 images | 200 | 4,706,674 | 97.74% |
| `sla_e300` | 45,000 images | 300 | 4,706,674 | 97.82% |

The notebook selected `sla_e300` because it achieved the higher validation accuracy. It then trained `final_all50k_sla_e300` on all 50,000 labelled images for 300 epochs.

The final retraining log shows approximately 10.2 seconds per epoch on an NVIDIA A100-SXM4-40GB GPU. The 97.82% number comes from the fixed local validation set. It is useful for selecting a model, but it does not guarantee the Kaggle public or private leaderboard score.

## Required data files

The notebook expects the competition data in one directory with this structure:

```text
data/
├── train_labels.csv
├── test.csv
├── sample_submission.csv
├── train_images/
└── test_images/
```

The code checks the input files before it trains:

- `train_labels.csv` must contain `id`, `filename`, and `label`.
- `test.csv` must contain `id` and `filename`.
- `sample_submission.csv` must contain exactly two columns: `id` and `label`.
- The IDs in `test.csv` and `sample_submission.csv` must match in the same order.
- Every image must be RGB and exactly 32 × 32 pixels.
- The training array must contain 50,000 images.
- The test array must contain 2,000 images.
- Training labels must be integers from 0 through 9.

The notebook searches for the data in this order:

1. The directory in `CIFAR10_DATA_DIR`.
2. `/data`.
3. `~/data`.
4. `./data` in the current notebook directory.

For the recorded NYU run, the notebook found the files in `/home/bap9622/data`.

The code opens each PNG image once, converts it to RGB, stores the pixels in NumPy arrays, and saves a compressed cache named `arrays.npz`. Later runs use that cache instead of reopening all image files.

## Paths, files, and environment variables

Run this notebook in an NYU OOD JupyterLab session with a GPU allocation. The code stops if CUDA is unavailable. That prevents an accidental multi-hour CPU run.

| Item | Default location |
|---|---|
| Input data | `CIFAR10_DATA_DIR`, `/data`, `~/data`, or `./data` |
| Working directory | `$SCRATCH/cifar10_lean_sla`; if `$SCRATCH` is unavailable, `~/cifar10_lean_sla` |
| Image cache | `<working directory>/arrays.npz` |
| Run folders | `<working directory>/runs/` |
| Checkpoint | `<working directory>/runs/<run-name>_s<seed>/checkpoint.pt` |
| Final model weights | `<working directory>/runs/<run-name>_s<seed>/model.pt` |
| Final submission | `~/submission.csv` |

Use these settings when the automatic data search does not find the files:

```python
%env CIFAR10_DATA_DIR=/home/bap9622/data
%env CIFAR10_WORK_DIR=/scratch/bap9622/cifar10_lean_sla
```

Use this assignment in the last notebook cell so the code saves the upload file as `submission.csv`:

```python
SUBMISSION_PATH = Path.home() / "submission.csv"
```

## Data split and model selection

The notebook creates one fixed stratified validation split before it trains the candidate models.

- It uses NumPy random seed `1123` to create the split.
- It selects 500 images from each CIFAR-10 class for validation.
- The validation set contains exactly 5,000 images.
- The training portion contains the remaining 45,000 images.

This class-balanced split lets the code compare the 200-epoch and 300-epoch schedules on the same data. The notebook does not look at the Kaggle test images while it makes this choice.

The default configuration uses one training seed:

```python
SEEDS = (123,)
```

You can change it to `(123, 456, 789)` to measure seed-to-seed variation. That change trains more validation models and selects the schedule by mean validation accuracy. It does not create a test-time ensemble.

After the comparison, the code trains the selected single configuration again on all 50,000 labelled images. It uses `FINAL_SEED = 123` for that final training run.

## Network architecture

The model is a CIFAR-sized pre-activation residual network. It uses three residual stages and two output heads.

| Part of the network | Design |
|---|---|
| Input | Normalized RGB image, shape `3 × 32 × 32` |
| Stem | 3 × 3 convolution, 64 output channels, stride 1 |
| Stage 1 | 4 residual blocks, 64 channels |
| Stage 2 | 4 residual blocks, 128 channels; first block downsamples by stride 2 |
| Stage 3 | 3 residual blocks, 256 channels; first block downsamples by stride 2 |
| Block order | BatchNorm → ReLU → 3 × 3 convolution → BatchNorm → ReLU → 3 × 3 convolution |
| Skip path | Identity when shapes match; 1 × 1 projection when channel count or resolution changes |
| Final feature layer | BatchNorm, ReLU, adaptive global average pooling |
| Main classification head | Linear layer from 256 features to 10 CIFAR-10 classes |
| SLA head | Linear layer from 256 features to 40 joint class-and-rotation outputs |
| Total trainable parameters | 4,706,674 |

The notebook counts every trainable parameter, including the SLA head, and stops if the count exceeds 5,000,000.

The code also contains an optional Squeeze-and-Excitation block. The default run keeps it disabled. Enabling it creates a separate candidate instead of silently changing the base model.

## SLA: supervised learning with rotations

SLA adds a second supervised task to the network.

For every training image, the code creates four views:

```text
0° rotation
90° rotation
180° rotation
270° rotation
```

The regular classification head predicts one of the ten CIFAR-10 classes from the unrotated image. The SLA head predicts both the object class and the rotation. It has 40 outputs because it represents every possible pair:

```text
10 object classes × 4 rotation classes = 40 joint classes
```

The joint target uses this formula:

```text
joint_target = class_label × 4 + rotation_index
```

For example, class `3` rotated by 180° has rotation index `2`, so its joint target is `3 × 4 + 2 = 14`.

The code trains both heads with cross-entropy loss and label smoothing. It then reads the four rotation-aligned class distributions from the SLA head, averages them, and treats that average as a detached teacher distribution. The ordinary class head learns to match it with KL divergence.

The complete loss is:

```text
total loss = SLA joint cross-entropy
           + main-class cross-entropy
           + self-distillation weight × KL divergence
```

The default self-distillation weight is `1.0`. The default temperature is `1.0`.

## Image preparation and training augmentation

The default candidate uses the `trivial` augmentation recipe. For every training image, the data loader applies these steps:

1. Convert the cached RGB array back into a PIL image.
2. Apply a random 32 × 32 crop from an image with four pixels of reflection padding.
3. Apply a random horizontal flip.
4. Apply one randomly selected `TrivialAugmentWide` operation.
5. Convert the image to a tensor.
6. Normalize it with CIFAR-10 channel statistics.
7. Apply a 12 × 12 Cutout region by setting that tensor region to zero.

The normalization values are:

```python
MEAN = (0.4914, 0.4822, 0.4465)
STD  = (0.2470, 0.2435, 0.2616)
```

The evaluation pipeline does not apply random crop, random flip, TrivialAugmentWide, or Cutout. It converts the original test image to a tensor and normalizes it.

## Optimizer and learning-rate schedule

The default training recipe uses the following settings:

| Setting | Value |
|---|---:|
| Batch size | 256 |
| Evaluation batch size | 128 |
| Epochs in compared runs | 200 and 300 |
| Warm-up | 5 epochs |
| Optimizer | SGD with Nesterov momentum |
| Momentum | 0.9 |
| Initial learning rate | 0.2 |
| Weight decay | `5e-4` for weights; `0` for biases and normalization parameters |
| Label smoothing | 0.1 |
| Schedule | Linear warm-up followed by cosine decay for every optimizer step |
| Precision | Automatic mixed precision with float16 autocast |
| Memory format | Channels-last |
| Matrix math | TF32 enabled on supported NVIDIA GPUs |
| SGD implementation | Fused SGD when PyTorch and the GPU support it; regular SGD otherwise |

The code calculates the learning rate as:

```text
learning rate = 0.2 × warm-up fraction
```

during warm-up. After warm-up, it follows a cosine curve down to zero by the last training step.

Each SLA batch contains 256 original images, but the model processes four rotations per image for the SLA task. The feature extractor therefore processes 1,024 rotated images for each SLA training step.

## Checkpoints and restart behavior

Each candidate stores its files in a separate directory. For example:

```text
<work directory>/runs/sla_e300_s123/
├── checkpoint.pt
└── model.pt
```

The notebook saves a checkpoint every 10 epochs and after the final epoch. A checkpoint contains:

- The model weights.
- The optimizer state.
- The learning-rate scheduler state.
- The automatic mixed-precision scaler state.
- The candidate configuration.
- The optional SWA model and update count when SWA is enabled.

When you rerun the notebook with the same configuration and `FORCE_RETRAIN = False`, it loads the checkpoint and continues after the last completed epoch. If the saved configuration differs from the active configuration, the notebook stops. It does not silently load incompatible weights.

The lean notebook restores the model, optimizer, scheduler, and scaler correctly. It does not save every random-number-generator state, so a resumed run does not promise bit-for-bit identical future augmentation order compared with an uninterrupted run.

Use this only when you intentionally want to discard the saved checkpoint and start the candidate again:

```python
FORCE_RETRAIN = True
```

## Optional experiments already included in the notebook

The notebook contains several optional candidates. All are disabled by default. Enable one at a time, compare it against the same validation split, and keep it only if it improves the measured validation result.

| Flag | Candidate name | What changes |
|---|---|---|
| `RUN_TEMPERATURE_2` | `sla_e300_t2` | Uses self-distillation temperature 2.0 |
| `RUN_SWA_300` | `sla_e300_swa` | Starts Stochastic Weight Averaging at epoch 220 and refreshes BatchNorm statistics afterward |
| `RUN_SLA_CP` | `sla_cp_e200` | Adds sampled RGB channel permutations to the rotation target; the joint head grows to 240 outputs |
| `RUN_AUGMIX_JSD` | `sla_augmix_jsd_e200` | Creates two AugMix views and adds a Jensen–Shannon consistency loss |
| `RUN_SE` | `sla_se_e300` | Adds Squeeze-and-Excitation blocks to every residual block |

The code does not combine these changes automatically. That design keeps comparisons meaningful and avoids assuming that several individually plausible techniques will work better together.

## Final inference and test-time augmentation

The code uses the final all-50k model in evaluation mode. For each image view, it calculates two 10-class probability vectors:

1. The ordinary classification-head probabilities.
2. The class probabilities recovered from the SLA joint head by aligning the predicted rotation with each rotated input.

It adds those two probability vectors. It then averages the result across six fixed test-time views:

| View | Weight |
|---|---:|
| Original image | 0.25 |
| Horizontally flipped image | 0.25 |
| Original image shifted by −1 pixel vertically and horizontally | 0.125 |
| Original image shifted by +1 pixel vertically and horizontally | 0.125 |
| Flipped image shifted by −1 pixel vertically and horizontally | 0.125 |
| Flipped image shifted by +1 pixel vertically and horizontally | 0.125 |

The six weights sum to `1.0`. The model does not update any weight, BatchNorm statistic, or calibration parameter while it processes test images.

## Submission file

The code takes the class with the highest final probability for every test image and writes:

```csv
id,label
0,6
1,3
2,3
...
```

Before writing the file, it checks all of these conditions:

```python
assert submission.columns.tolist() == ["id", "label"]
assert len(submission) == len(sample) == 2_000
assert np.array_equal(submission.id.to_numpy(), sample.id.to_numpy())
assert submission.label.between(0, 9).all()
```

The supplied `submission_final.csv` passes the same format checks:

| Check | Result |
|---|---|
| Columns | Exactly `id,label` |
| Number of rows | 2,000 |
| ID order | `0` through `1999`, in order |
| Label range | Every label is an integer from `0` through `9` |
| Missing values | None |

The file contains these predicted-class counts:

```text
0: 176, 1: 214, 2: 207, 3: 201, 4: 197,
5: 210, 6: 181, 7: 206, 8: 208, 9: 200
```

These counts help catch obvious pipeline mistakes. They do not reveal the hidden labels and do not predict the final leaderboard score.

## How to run the notebook

1. Start an NYU OOD JupyterLab session with a GPU.
2. Place the competition files in `/data`, `~/data`, or a directory set in `CIFAR10_DATA_DIR`.
3. Run every setup cell from top to bottom. They configure paths, validate the input files, load or create the cache, define transforms, create the model, and define the training functions.
4. Run the candidate-training cell. It compares the 200- and 300-epoch configurations on the fixed validation split, selects the better one, and starts the all-50k final retraining run.
5. In the final cell, set `SUBMISSION_PATH = Path.home() / "submission.csv"`.
6. Run the final cell. It makes the test predictions, validates the submission format, and writes `~/submission.csv`.
7. Upload that CSV file to Kaggle.

## Rules followed by this workflow

- It trains only on the permitted labelled CIFAR-10 training set.
- It uses the unlabelled evaluation images only for final forward-pass prediction.
- It does not use evaluation labels or manual labels.
- It does not use external datasets or pretrained weights.
- It uses one final model rather than a test-time model ensemble.
- It keeps the model under the 5 million parameter limit.
- It writes exactly the required `id,label` CSV format.
