"""strat_50_cutout_full inference: state_dict -> submission.csv, test_logits.npy, val_logits.npy.

python strat_50_infer.py --config strat_50_config.yaml --data cifar_comp_v1.npz --out outputs --ckpt outputs_s0/model.pt [--tta flip2|tta6|agg|agg_flip2]

Several --ckpt paths form an ensemble: softmax averaged over members, parameter sum asserted <= 5M.
Same preprocessing and views as the final evaluation in training. No BN update, no test statistics.
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
import strat_50_train as T  # noqa: E402  (same folder: shared data/preprocessing/TTA code)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ckpt", required=True, nargs="+")
    ap.add_argument("--tta", default=None, choices=["flip2", "tta6", "agg", "agg_flip2"])
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--resume", default=None, help="unused in inference; kept for the entry-script contract")
    ap.add_argument("--time-budget-min", type=float, default=None, help="unused in inference")
    a = ap.parse_args()
    t0 = time.perf_counter()
    cfg, cfg_sha = T.load_cfg(a.config)
    T.seed_all(cfg["seed"])
    dev, gpu, _, _, _ = T.setup_device(cfg)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tta = a.tta or cfg["eval"]["submit_tta"]
    views = ["id", "flip"] if tta == "flip2" else T.TTA6_ORDER
    D = T.load_data(a.data, cfg, a.smoke and dev.type != "cuda", dev, need_train=False)
    mean, std = T.norm_consts(cfg, dev)
    total = 0
    acc = {"val": 0, "test": 0}
    for path in a.ckpt:
        model = T.build_model(cfg["model"])
        total += sum(p.numel() for p in model.parameters())
        assert total <= 5_000_000, f"param limit exceeded: {total}"
        model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
        if cfg["train"]["channels_last"]:
            model = model.to(memory_format=torch.channels_last)
        model = model.to(dev).eval()
        for split, X in (("val", D["val_x"]), ("test", D["test_x"])):
            if tta.startswith("agg"):  # SLA joint head, aggregated over rotations (same code as the training eval)
                A = {v: torch.softmax(l, 1) for v, l in T.predict_agg_logits(model, X, ["id", "flip"], cfg, mean, std).items()}
                p = A["id"] if tta == "agg" else 0.5 * (A["id"] + A["flip"])
            else:
                p = T.option_probs(T.predict_logits(model, X, views, cfg, mean, std), cfg)[tta]
            acc[split] = acc[split] + p
    for split in acc:
        acc[split] = acc[split] / len(a.ckpt)
        np.save(out / f"{split}_probs.npy", acc[split].cpu().numpy().astype(np.float32))
        np.save(out / f"{split}_logits.npy", torch.log(acc[split].clamp_min(1e-12)).cpu().numpy().astype(np.float32))
    val_acc = T.accuracy(acc["val"], D["val_y"])
    T.write_submission(out / "submission.csv", D["test_ids"], acc["test"].argmax(1).cpu().numpy())
    info = {"ckpt": a.ckpt, "members": len(a.ckpt), "tta": tta, "val_acc": val_acc, "n_val": D["n_val"],
            "final_params": total, "gpu": gpu, "torch_version": torch.__version__, "config_sha256": cfg_sha,
            "seconds": time.perf_counter() - t0}
    (out / "infer_summary.json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    print(f"INFER DONE: members={len(a.ckpt)} tta={tta} params={total} VAL5K={val_acc:.4f} -> {out / 'submission.csv'}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
