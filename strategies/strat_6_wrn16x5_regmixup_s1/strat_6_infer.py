"""strat_6_wrn16x5_regmixup_s1 inference: checkpoint -> submission.csv, test_logits.npy, val_logits.npy.

python strat_6_infer.py --config strat_6_config.yaml --data cifar_comp_v1.npz --out outputs --ckpt outputs/model.pt
       [--weights auto|live|ema] [--tta auto|none|flip2] [--smoke]

--ckpt: model.pt (state_dict of the submitted weights) or ckpt_last.pt (full training checkpoint).
--tta auto: read chosen.tta from summary.json next to the checkpoint (else none).
--weights auto: for ckpt_last.pt, read chosen.weights from summary.json (else ema if present).
Same preprocessing as training: train mean/std, eval mode, no BN update, no test statistics.
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
import strat_6_train as T  # noqa: E402  (same folder: shared data/preprocessing/TTA code)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--weights", default="auto", choices=["auto", "live", "ema"])
    ap.add_argument("--tta", default="auto", choices=["auto", "none", "flip2"])
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    t0 = time.perf_counter()
    cfg, cfg_sha = T.load_cfg(a.config)
    T.seed_all(cfg["seed"])
    model, total = T.build_and_check(cfg)
    dev, gpu, amp_dtype, _, amp_name = T.setup_device(cfg)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    ck_path = Path(a.ckpt)
    summ_path = ck_path.parent / "summary.json"
    chosen = json.loads(summ_path.read_text(encoding="utf-8")).get("chosen", {}) if summ_path.exists() else {}
    obj = torch.load(ck_path, map_location="cpu", weights_only=False)  # own file (ckpt holds RNG states)
    if isinstance(obj, dict) and "model" in obj and "state" in obj:
        w = a.weights if a.weights != "auto" else chosen.get("weights", "ema" if obj["ema"] is not None else "live")
        sd = obj["ema"] if w == "ema" else obj["model"]
        if sd is None:
            raise SystemExit("checkpoint has no EMA weights")
    else:
        w, sd = "model.pt", obj
    tta = a.tta if a.tta != "auto" else chosen.get("tta", "none")
    model.load_state_dict(sd)
    if cfg["train"]["channels_last"]:
        model = model.to(memory_format=torch.channels_last)
    model = model.to(dev).eval()
    print(f"infer: ckpt={ck_path} weights={w} tta={tta} device={dev} gpu={gpu} amp={amp_name}", flush=True)

    D = T.load_data(a.data, cfg, a.smoke, dev, need_train=False)
    mean, std = T.norm_consts(cfg, dev)
    views = ["id", "flip"] if tta == "flip2" else ["id"]
    res = {}
    for split, X in (("val", D["val_x"]), ("test", D["test_x"])):
        L = T.predict_logits(model, X, views, cfg, mean, std, amp_dtype)
        if tta == "none":
            logits = L["id"]
            pred = logits.argmax(1)
        else:
            p = T.option_probs(L, cfg)[tta]
            logits = torch.log(p.clamp_min(1e-12))
            pred = p.argmax(1)
        res[split] = (logits, pred)
    val_acc = (res["val"][1] == D["val_y"]).float().mean().item()
    np.save(out / "test_logits.npy", res["test"][0].cpu().numpy().astype(np.float32))
    np.save(out / "val_logits.npy", res["val"][0].cpu().numpy().astype(np.float32))
    T.write_submission(out / "submission.csv", D["test_ids"], res["test"][1].cpu().numpy())
    info = {"ckpt": str(ck_path), "weights": w, "tta": tta, "val_acc": val_acc, "n_val": D["n_val"],
            "params": total, "gpu": gpu, "amp": amp_name, "torch_version": torch.__version__,
            "config_sha256": cfg_sha, "seconds": time.perf_counter() - t0}
    (out / "infer_summary.json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    print(f"INFER DONE: VAL5K={val_acc:.4f} -> {out / 'submission.csv'}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
