# data

We don't keep the competition data in this repo, since the Kaggle terms don't allow redistributing it. You can
get it from the competition page, or with the Kaggle CLI once you've accepted the rules:

```bash
kaggle competitions download -c cifar-10-res-net-architecture-challenge-dl
unzip cifar-10-res-net-architecture-challenge-dl.zip -d data/
```

After unzipping, the folder should look like this:

```
data/
  train_images/          50,000 PNG files, 32x32 RGB
  train_labels.csv       id,filename,label
  test_images/           2,000 PNG files, 32x32 RGB
  test.csv               id,filename (order = sample_submission.csv)
  sample_submission.csv  id,label
```

Next, run `python tools/pack_data.py` from the repo root. Please leave everything in `data/` as it is.
