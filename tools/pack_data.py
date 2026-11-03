"""Pack data/ into one .npz for fast upload to Colab. Reads data/, never writes to it.

Contents of packed/cifar_comp_v1.npz:
  x_train  uint8 (50000, 32, 32, 3)   y_train int64 (50000,)
  x_test   uint8 (2000, 32, 32, 3)    test_ids int64 (2000,)  (order = test.csv = sample_submission.csv)
  val_idx  int64 (5000,)              fixed VAL5K split: 500 per class, seed 0
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


def load(folder, files):
    return np.stack([np.asarray(Image.open(folder / f).convert("RGB")) for f in files])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "packed" / "cifar_comp_v1.npz"))
    ap.add_argument("--val-per-class", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data-dir", default=str(ROOT / "data"), help="folder with the competition files (Kaggle kernels: /kaggle/input/...)")
    a = ap.parse_args()

    d = Path(a.data_dir)
    tr = pd.read_csv(d / "train_labels.csv")
    te = pd.read_csv(d / "test.csv")
    ss = pd.read_csv(d / "sample_submission.csv")
    assert (te.id.values == ss.id.values).all(), "test.csv order != sample_submission order"

    x_train = load(d / "train_images", tr.filename)
    y_train = tr.label.to_numpy(np.int64)
    x_test = load(d / "test_images", te.filename)
    test_ids = te.id.to_numpy(np.int64)

    rng = np.random.default_rng(a.seed)
    val_idx = np.sort(np.concatenate([
        rng.choice(np.flatnonzero(y_train == c), a.val_per_class, replace=False) for c in range(10)
    ]))

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, x_train=x_train, y_train=y_train, x_test=x_test, test_ids=test_ids, val_idx=val_idx)
    meta = {
        "file": out.name,
        "sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
        "bytes": out.stat().st_size,
        "x_train": list(x_train.shape), "x_test": list(x_test.shape),
        "val_split": f"VAL5K seed={a.seed} per_class={a.val_per_class}",
        "val_idx_sha256": hashlib.sha256(val_idx.tobytes()).hexdigest(),
        "train_mean_rgb": np.round(x_train.reshape(-1, 3).mean(0) / 255, 4).tolist(),
        "train_std_rgb": np.round(x_train.reshape(-1, 3).std(0) / 255, 4).tolist(),
    }
    out.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
