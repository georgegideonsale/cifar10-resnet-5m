"""strat_9 file: strat_6 saved live 6-view TTA probabilities -> submission.csv. No training.

Usage: python strategies/strat_9_s6_tta6/strat_9_make.py
Input test_probs_live_tta6.npy and val_probs_live_tta6.npy are written by strat_6_train.py. They are not in strat_6_infer.py.
"""
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "strategies" / "strat_6_wrn16x5_regmixup_s1" / "outputs"
OUT = Path(__file__).resolve().parent / "outputs" / "submission.csv"

z = np.load(ROOT / "packed" / "cifar_comp_v1.npz")
val = np.load(SRC / "val_probs_live_tta6.npy")
print("VAL5K", (val.argmax(1) == z["y_train"][z["val_idx"]]).mean())
pred = np.load(SRC / "test_probs_live_tta6.npy").argmax(1)
OUT.parent.mkdir(exist_ok=True)
OUT.write_text("id,label\n" + "".join(f"{int(i)},{int(p)}\n" for i, p in zip(z["test_ids"], pred)), encoding="utf-8", newline="\n")
print("wrote", OUT)
