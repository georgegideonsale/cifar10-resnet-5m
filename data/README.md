# Data

The competition data is not stored in this repo (the Kaggle terms do not allow redistribution).
Download it from the competition page, or with the Kaggle CLI after accepting the rules:

```bash
kaggle competitions download -c cifar-10-res-net-architecture-challenge-dl
unzip cifar-10-res-net-architecture-challenge-dl.zip -d data/
```

Expected contents:

```
data/
  train_images/          50,000 PNG files, 32x32 RGB
  train_labels.csv       id,filename,label
  test_images/           2,000 PNG files, 32x32 RGB
  test.csv               id,filename (order = sample_submission.csv)
  sample_submission.csv  id,label
```

Then run `python tools/pack_data.py` from the repo root. Do not edit anything in `data/`.
