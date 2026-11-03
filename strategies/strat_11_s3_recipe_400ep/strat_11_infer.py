"""strat_11_s3_recipe_400ep inference: checkpoint -> submission.csv, test_logits.npy, val_logits.npy.

python strat_11_infer.py --config strat_11_config.yaml --data cifar_comp_v1.npz --out outputs --ckpt outputs/model.pt
       [--smoke]

--ckpt: model.pt (state_dict of the submitted EMA weights) or ckpt_last.pt (full training checkpoint, EMA is read).
Same preprocessing and the same views as the final evaluation in training. No BN update, no test statistics.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import strat_11_train as T  # noqa: E402  (same folder: shared data/preprocessing/TTA code)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--resume", default=None, help="unused in inference; kept for the entry-script contract")
    ap.add_argument("--time-budget-min", type=float, default=None, help="unused in inference")
    a = ap.parse_args()
    t0 = time.perf_counter()
    cfg, cfg_sha = T.load_cfg(a.config)
    T.seed_all(cfg["seed"])
    model, total = T.build_and_check(cfg)
    dev, gpu, _, _, _ = T.setup_device(cfg)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    obj = torch.load(a.ckpt, map_location="cpu", weights_only=False)  # own file (ckpt holds RNG states)
    sd = obj["ema"] if isinstance(obj, dict) and "ema" in obj else obj
    model.load_state_dict(sd)
    if cfg["train"]["channels_last"]:
        model = model.to(memory_format=torch.channels_last)
    model = model.to(dev).eval()

    D = T.load_data(a.data, cfg, a.smoke and dev.type != "cuda", dev, need_train=False)
    mean, std = T.norm_consts(cfg, dev)
    tta = cfg["eval"]["submit_tta"]
    res = {}
    for split, X in (("val", D["val_x"]), ("test", D["test_x"])):
        views = ["id", "flip"] if tta == "flip2" else T.TTA6_ORDER
        p = T.option_probs(T.predict_logits(model, X, views, cfg, mean, std), cfg)[tta]
        res[split] = p
        np.save(out / f"{split}_logits.npy", torch.log(p.clamp_min(1e-12)).cpu().numpy().astype(np.float32))
    val_acc = T.accuracy(res["val"], D["val_y"])
    T.write_submission(out / "submission.csv", D["test_ids"], res["test"].argmax(1).cpu().numpy())
    info = {"ckpt": str(a.ckpt), "weights": cfg["eval"]["submit_weights"], "tta": tta, "val_acc": val_acc,
            "n_val": D["n_val"], "params": total, "gpu": gpu, "torch_version": torch.__version__,
            "config_sha256": cfg_sha, "seconds": time.perf_counter() - t0}
    (out / "infer_summary.json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    print(f"INFER DONE: VAL5K={val_acc:.4f} -> {out / 'submission.csv'}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
