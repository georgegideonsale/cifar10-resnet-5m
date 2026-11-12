# Rules checklist

This is our summary of the Competition Rules, the Overview and the Data Description on the Kaggle competition
page, plus Kaggle's general competition rules. The Kaggle pages are the actual reference, and if something is
unclear the course staff have the final say (Rules §15).

## Check these for every strategy

- [ ] The model is a convolutional network with residual (skip) connections. (Rules §4)
- [ ] Total parameters at inference are at most 5,000,000, counting all layers, the classifier head, frozen
      parts and every ensemble member together. The README states the count. (Rules §4)
- [ ] Trained from scratch: no pretrained weights, no fine-tuning of a pretrained model, no external
      checkpoint or feature extractor. (Rules §5)
- [ ] Trained only on the 50,000 training images from the course staff, plus augmentations of those images.
      That rules out the CIFAR-10 test set, CIFAR-10.1/10.2, CINIC-10, CIFAR-100, ImageNet and any other
      images. (Rules §3)
- [ ] Test images are only ever inference input. Nothing is trained on them, and no pseudo-labels,
      fine-tuning, BN statistics or normalization statistics come from them. (Rules §3, §12)
- [ ] Nobody labels or predicts test images by hand, looks up labels, or tries to infer the hidden labels.
      (Rules §12)
- [ ] Knowledge distillation only uses a teacher we trained ourselves, from scratch, on the same training
      images. (Rules §5)
- [ ] Any public code we use is cited in the README and has an OSI-approved license that doesn't restrict
      commercial use. (Rules §5, Kaggle rules §6c)
- [ ] The README gives the training configuration (optimizer, learning rate and schedule, batch size, epochs),
      the augmentation and preprocessing, the parameter count, and the commands to reproduce the
      submission. (Rules §13)
- [ ] The folder has the training log and the checkpoint of the submitted model, since staff may ask for
      them. (Rules §13, §15)
- [ ] `tools/validate_submission.py` prints `PASS` for the submitted file, meaning header `id,label`, 2,000 rows
      in the order of `sample_submission.csv`, and integer labels 0-9. (Rules §7)

## Sharing

- We don't share code, weights, prediction files or results with other teams. (Rules §12, Kaggle rules §6a)
- While the competition runs, public code sharing is only allowed on the Kaggle forum or in Kaggle notebooks for
  this competition (Kaggle rules §6b). So the repo was private (team members and course staff only) until the
  competition ended. We made it public after the deadline on 2026-11-07.
- Each person has one Kaggle account. As a team we get 5 submissions per day in total, resetting at 00:00 UTC.

## Where the rules were unclear

Wherever the rules left room for interpretation, we went with the strict reading.

1. Ensembles: the 5M limit is on the final model, so we add up the members of an ensemble.
2. Test-time augmentation (flips, small shifts, averaging the outputs of one model) is allowed. It uses no
   labels and doesn't train on test images. We only use it when it helps on VAL5K, and the README says which
   views were averaged.
3. Distillation teacher size: every network we train stays at or below 5M parameters, teachers included. The
   training scripts assert this for the teacher (look for the "rules Q9" comments in the code).
4. Test-time BN adaptation, or any statistic computed from test images, is not allowed.
5. We don't use images from a generative model trained on our data, because the rules only allow augmentation
   of the training images.
6. Normalization uses training-set statistics only: mean (0.4914, 0.4822, 0.4465),
   std (0.2470, 0.2435, 0.2616).
7. Model selection uses VAL5K (which is part of the training data) and public leaderboard scores only. We never
   select on agreement or confidence on the test images.
