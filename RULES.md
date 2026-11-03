# Rules checklist

Sources: the Competition Rules, the Overview and the Data Description on the Kaggle competition page, and the
Kaggle general competition rules. When in doubt, the course staff decide (Rules §15). This file is a summary;
the Kaggle pages are the reference.

## Checklist for every strategy

- [ ] The model is a convolutional network with residual (skip) connections. (Rules §4)
- [ ] Total parameters at inference are at most 5,000,000: all layers, the classifier head, frozen parts and
      every ensemble member added together. The count is in the README. (Rules §4)
- [ ] Trained from scratch. No pretrained weights, no fine-tuning of a pretrained model, no external
      checkpoint or feature extractor. (Rules §5)
- [ ] Trained only on the 50,000 training images provided by the course staff, plus augmentations of those
      images. No CIFAR-10 test set, CIFAR-10.1/10.2, CINIC-10, CIFAR-100, ImageNet or any other images. (Rules §3)
- [ ] Test images are used only as inference input. No training, pseudo-labels, fine-tuning, BN statistics
      or normalization statistics from the test images. (Rules §3, §12)
- [ ] No manual labelling or human prediction of test images, no label lookups, no attempt to infer the
      hidden labels. (Rules §12)
- [ ] Knowledge distillation only from a teacher that we trained ourselves from scratch on the same
      training images. (Rules §5)
- [ ] Any public code that is used is cited in the README and has an OSI-approved license that does not
      restrict commercial use. (Rules §5, Kaggle rules §6c)
- [ ] The README has the training configuration (optimizer, learning rate and schedule, batch size, epochs),
      the augmentation and preprocessing, the parameter count and the commands to reproduce the
      submission. (Rules §13)
- [ ] The training log and the checkpoint of the submitted model are in the folder. Staff may ask for
      them. (Rules §13, §15)
- [ ] `tools/validate_submission.py` prints `PASS` for the submitted file: header `id,label`, 2,000 rows in
      the order of `sample_submission.csv`, integer labels 0-9. (Rules §7)

## Sharing

- Do not share code, weights, prediction files or results with other teams. (Rules §12, Kaggle rules §6a)
- During the competition, public code sharing is only allowed on the Kaggle forum or Kaggle notebooks of this
  competition (Kaggle rules §6b). **This repo stays private until the competition ends.** Only team members
  and course staff are added.
- One Kaggle account per person. The team has 5 submissions per day in total (reset at 00:00 UTC).

## Our decisions on unclear points

We took the strict reading where the rules leave room.

1. **Ensembles.** The 5M limit applies to the final model, so the members of an ensemble are summed.
2. **Test-time augmentation** (flips, small shifts, averaging the outputs of one model) is allowed. It uses no
   labels and does not train on test images. We use it only when it helps on VAL5K, and the README says which
   views were averaged.
3. **Distillation teacher size.** Every network we train, teachers included, stays at or below 5M parameters.
   The training scripts assert this for the teacher (the "rules Q9" comments in the code).
4. **Test-time BN adaptation** or any statistic computed from test images is not allowed.
5. **Images from a generative model** trained on our data are not used. The rules allow augmentation of the
   training images only.
6. **Normalization** uses training-set statistics only: mean (0.4914, 0.4822, 0.4465),
   std (0.2470, 0.2435, 0.2616).
7. **Model selection** uses VAL5K (part of the training data) and public leaderboard scores only, never
   agreement or confidence on the test images.
