"""strat_11_s3_recipe_400ep train entry point (contract: docs/workflow.md, spec: strategy.md).

python strat_11_train.py --config strat_11_config.yaml --data cifar_comp_v1.npz --out outputs
       [--smoke] [--resume outputs/ckpt_last.pt] [--time-budget-min M]

Recipe ported from the teammate notebook docs/10-31-2026 - Tarun/kaggle_train_tarun.ipynb (version 25fd8d1).
Differences from it: our VAL5K split, fp16 on T4, evaluation every 10 epochs, our output contract.
"""
import argparse
import contextlib
import copy
import hashlib
import json
import math
import os
import platform
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from strat_11_model import build_model  # noqa: E402

# View name -> (flip, dy, dx). Same 6 views as strat_3 to strat_9, so the 6-view reads compare.
VIEWS = {
    "id": (False, 0, 0),
    "flip": (True, 0, 0),
    "s_m": (False, -1, -1),
    "s_p": (False, 1, 1),
    "fs_m": (True, -1, -1),
    "fs_p": (True, 1, 1),
}
TTA6_ORDER = ["id", "flip", "s_m", "s_p", "fs_m", "fs_p"]


# ----------------------------------------------------------------------------- setup
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--ckpt", default=None, help="unused in training; kept for the entry-script contract")
    ap.add_argument("--resume", default=None)
    ap.add_argument("--time-budget-min", type=float, default=None)
    return ap.parse_args()


def sha256_file(path, chunk=1 << 24):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def load_cfg(path):
    raw = Path(path).read_bytes()
    return yaml.safe_load(raw.decode("utf-8")), hashlib.sha256(raw).hexdigest()


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # also seeds all CUDA devices


def setup_device(cfg):
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = bool(cfg["train"]["cudnn_benchmark"])
        name = cfg["train"]["amp_dtype"]
        amp_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}[name]
        return torch.device("cuda"), torch.cuda.get_device_name(0), amp_dtype, amp_dtype == torch.float16, name
    return torch.device("cpu"), "cpu", None, False, "none"


def build_and_check(cfg):
    model = build_model(cfg["model"])
    total = sum(p.numel() for p in model.parameters())
    print(f"PARAMS total={total:,} (all parameters, trainable+frozen)", flush=True)
    assert total <= 5_000_000, f"param limit exceeded: {total}"
    assert total <= cfg["param_limit"], f"param limit exceeded: {total}"
    assert total == cfg["expected_params"], f"param count {total} != spec {cfg['expected_params']}"
    return model, total


def load_data(npz_path, cfg, cpu_smoke, dev, need_train=True):
    """Train = not in val_idx. Val and test are uint8 NHWC tensors on dev; train stays on the CPU for the workers."""
    with np.load(npz_path) as z:
        x_train, y_train = z["x_train"], z["y_train"]
        x_test, test_ids, val_idx = z["x_test"], z["test_ids"], z["val_idx"]
    mask = np.ones(len(x_train), dtype=bool)
    mask[val_idx] = False
    if cfg["data"]["full_train"]:
        mask[:] = True
    train_idx = np.nonzero(mask)[0]
    if cpu_smoke:
        train_idx = train_idx[: cfg["smoke"]["cpu_n_train"]]
        val_idx = val_idx[: cfg["smoke"]["cpu_val_n"]]
    d = {
        "val_x": torch.from_numpy(np.ascontiguousarray(x_train[val_idx])).to(dev),
        "val_y": torch.from_numpy(y_train[val_idx].astype(np.int64)).to(dev),
        "test_x": torch.from_numpy(np.ascontiguousarray(x_test)).to(dev),
        "test_ids": test_ids.astype(np.int64),
        "n_val": len(val_idx),
    }
    if need_train:
        d["train_x"] = torch.from_numpy(np.ascontiguousarray(x_train[train_idx])).permute(0, 3, 1, 2).contiguous()
        d["train_y"] = torch.from_numpy(y_train[train_idx].astype(np.int64))
        d["n_train"] = len(train_idx)
    return d


def norm_consts(cfg, dev):
    mean = torch.tensor(cfg["data"]["mean"], device=dev).view(1, 3, 1, 1)
    std = torch.tensor(cfg["data"]["std"], device=dev).view(1, 3, 1, 1)
    return mean, std


# ----------------------------------------------------------------------------- augmentation (CPU, per image)
def train_transform(cfg):
    """uint8 CHW -> crop + flip, TrivialAugmentWide, float, normalize, random erasing (fill 0 = the mean colour)."""
    a = cfg["augment"]
    t = [v2.RandomCrop(32, padding=a["crop_padding"], padding_mode=a["pad_mode"]), v2.RandomHorizontalFlip(),
         v2.TrivialAugmentWide(), v2.ToDtype(torch.float32, scale=True),
         v2.Normalize(cfg["data"]["mean"], cfg["data"]["std"])]
    if a["random_erasing"] > 0:
        t.append(v2.RandomErasing(p=a["random_erasing"], value=0))
    return v2.Compose(t)


class ImageArrayDataset(Dataset):
    def __init__(self, x, y, transform):
        self.x, self.y, self.transform = x, y, transform

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return self.transform(self.x[i]), self.y[i]


# ----------------------------------------------------------------------------- inference
def amp_ctx(dev, amp_dtype):
    """Training autocast on CUDA only. CPU (smoke) runs in float32."""
    if amp_dtype is None:
        return contextlib.nullcontext()
    return torch.autocast(device_type=dev.type, dtype=amp_dtype)


def make_view(x, view, pad):
    flip, dy, dx = VIEWS[view]
    if flip:
        x = x.flip(3)
    if dy or dx:
        H, W = x.shape[2], x.shape[3]
        xp = F.pad(x, (pad, pad, pad, pad), mode="reflect")
        x = xp[:, :, pad + dy: pad + dy + H, pad + dx: pad + dx + W]
    return x.contiguous(memory_format=torch.channels_last)


@torch.no_grad()
def predict_logits(model, x_u8, views, cfg, mean, std):
    """Full fp32, eval mode, no BN update, no test statistics. Returns {view: float32 logits on device}."""
    model.eval()
    bs, pad = cfg["eval"]["batch_size"], cfg["eval"]["tta6_shift_pad"]
    out = {v: [] for v in views}
    for i in range(0, x_u8.shape[0], bs):
        xb = x_u8[i: i + bs].permute(0, 3, 1, 2).float().div_(255.0)
        xb = (xb - mean) / std
        for v in views:
            out[v].append(model(make_view(xb, v, pad)).float())
    return {v: torch.cat(t) for v, t in out.items()}


def option_probs(logits, cfg):
    """{view: logits} -> {'none', 'flip2', 'tta6' (if all views)}: averaged softmax probabilities (float32)."""
    P = {v: torch.softmax(l, dim=1) for v, l in logits.items()}
    res = {"none": P["id"], "flip2": 0.5 * (P["id"] + P["flip"])}
    if all(v in P for v in TTA6_ORDER):
        res["tta6"] = sum(w * P[v] for w, v in zip(cfg["eval"]["tta6_weights"], TTA6_ORDER))
    return res


def accuracy(probs, y):
    return (probs.argmax(1) == y).float().mean().item()


def write_submission(path, ids, labels):
    lines = ["id,label"] + [f"{int(i)},{int(l)}" for i, l in zip(ids, labels)]
    tmp = str(path) + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(lines) + "\n")
    os.replace(tmp, path)


def atomic_save(obj, path):
    tmp = str(path) + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


# ----------------------------------------------------------------------------- EMA
class EMA:
    """Parameters and float buffers (BN statistics) are averaged; integer buffers are copied. One fused lerp per step."""

    def __init__(self, model, decay):
        self.model = copy.deepcopy(model).eval().requires_grad_(False)
        self.decay = decay
        pairs = list(zip(list(self.model.parameters()) + list(self.model.buffers()),
                         list(model.parameters()) + list(model.buffers())))
        self.ema_float = [e for e, m in pairs if e.is_floating_point()]
        self.model_float = [m for e, m in pairs if e.is_floating_point()]
        self.other = [(e, m) for e, m in pairs if not e.is_floating_point()]

    @torch.no_grad()
    def update(self):
        torch._foreach_lerp_(self.ema_float, self.model_float, 1 - self.decay)
        for e, m in self.other:
            e.copy_(m)


# ----------------------------------------------------------------------------- main
def main():
    args = parse_args()
    t_setup = time.perf_counter()
    cfg, cfg_sha = load_cfg(args.config)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    seed_all(cfg["seed"])

    # First action: build the model, count ALL params, assert.
    model, total_params = build_and_check(cfg)

    dev, gpu, amp_dtype, use_scaler, amp_name = setup_device(cfg)
    tr, ev, kl, sm = cfg["train"], cfg["eval"], cfg["kill"], cfg["smoke"]
    smoke = args.smoke
    cpu_smoke = smoke and dev.type != "cuda"
    gpu_smoke = smoke and dev.type == "cuda"
    bs = sm["cpu_batch_size"] if cpu_smoke else tr["batch_size"]
    E_final = sm["cpu_epochs"] if cpu_smoke else tr["epochs"]
    E = E_final
    if gpu_smoke:
        E = sm["gpu_epochs"]  # stops early with the full-run schedule, so t_epoch and the projection are real
    eval_every = 1 if smoke else ev["every"]
    ckpt_every = 1 if smoke else cfg["checkpoint"]["every"]
    kill_on = not smoke
    workers = 0 if cpu_smoke else min(tr["workers"], os.cpu_count() or 1)
    M = args.time_budget_min
    print(f"seed={cfg['seed']} device={dev} gpu={gpu} amp={amp_name} scaler={use_scaler} smoke={smoke} "
          f"workers={workers} epochs={E_final} time_budget_min={M}", flush=True)

    data_sha = sha256_file(args.data)
    D = load_data(args.data, cfg, cpu_smoke, dev)
    mean, std = norm_consts(cfg, dev)
    n_train = D["n_train"]
    gen = torch.Generator()
    gen.manual_seed(cfg["seed"])
    loader = DataLoader(ImageArrayDataset(D["train_x"], D["train_y"], train_transform(cfg)), batch_size=bs,
                        shuffle=True, drop_last=tr["drop_last"], num_workers=workers,
                        pin_memory=dev.type == "cuda", persistent_workers=workers > 0, generator=gen)
    spe = len(loader)
    warmup_epochs = 1 if cpu_smoke else tr["warmup_epochs"]
    W = warmup_epochs * spe
    total_steps = E_final * spe
    print(f"data: train={n_train} val={D['n_val']} test={D['test_x'].shape[0]} steps/epoch={spe} "
          f"warmup_steps={W} total_steps={total_steps} data_sha256={data_sha[:12]}", flush=True)

    if tr["channels_last"]:
        model = model.to(memory_format=torch.channels_last)
    model = model.to(dev)
    opt = torch.optim.SGD(model.parameters(), lr=tr["lr"], momentum=tr["momentum"], nesterov=tr["nesterov"],
                          weight_decay=tr["weight_decay"])
    scaler = torch.amp.GradScaler("cuda") if use_scaler else None
    ema = EMA(model, tr["ema_decay"])
    st = {"step": 0, "epoch": 0, "elapsed": 0.0, "t_w": None, "epoch_times": [], "history": [], "done": False,
          "status": "running", "kill_reason": None}

    if args.resume:
        ck = torch.load(args.resume, map_location=dev, weights_only=False)  # own file (holds RNG states)
        if ck["smoke"] != smoke:
            raise SystemExit("resume: smoke flag differs from the checkpoint")
        if ck["config_sha256"] != cfg_sha:
            print("WARN: config sha256 differs from the checkpoint", flush=True)
        model.load_state_dict(ck["model"])
        ema.model.load_state_dict(ck["ema"])
        opt.load_state_dict(ck["optimizer"])
        if scaler is not None and ck["scaler"] is not None:
            scaler.load_state_dict(ck["scaler"])
        st = ck["state"]
        gen.set_state(ck["rng"]["loader"].cpu())
        random.setstate(ck["rng"]["python"])
        np.random.set_state(ck["rng"]["numpy"])
        torch.set_rng_state(ck["rng"]["torch"].cpu())
        if dev.type == "cuda" and ck["rng"]["cuda"] is not None:
            torch.cuda.set_rng_state(ck["rng"]["cuda"].cpu())
        print(f"RESUMED from {args.resume}: epoch={st['epoch']} step={st['step']} elapsed={st['elapsed']:.1f}s",
              flush=True)
        with open(out / "metrics.jsonl", "w", encoding="utf-8") as fh:  # rebuild from the checkpoint history
            for rec in st["history"]:
                fh.write(json.dumps(rec) + "\n")
    else:
        (out / "metrics.jsonl").write_text("", encoding="utf-8")

    t0 = time.perf_counter()
    el0 = st["elapsed"]

    def elapsed_now():
        return el0 + (time.perf_counter() - t0)

    def lr_q(step):
        """Linear warm-up, then cosine to zero. --time-budget-min: progress q = max(steps, time), stop at q >= 1."""
        if step < W:
            return tr["lr"] * (step + 1) / W, 0.0
        q = (step - W) / max(1, total_steps - W)
        if M is not None and st["t_w"] is not None:
            den = M * 60.0 - st["t_w"]
            q = max(q, (elapsed_now() - st["t_w"]) / den if den > 0 else 1.0)
        q = min(q, 1.0)
        return tr["lr"] * 0.5 * (1 + math.cos(math.pi * q)), q

    def save_ckpt():
        st["elapsed"] = elapsed_now()
        atomic_save({
            "model": model.state_dict(), "ema": ema.model.state_dict(), "optimizer": opt.state_dict(),
            "scaler": scaler.state_dict() if scaler else None, "state": st, "smoke": smoke,
            "config_sha256": cfg_sha, "steps_per_epoch": spe, "warmup_steps": W,
            "rng": {"loader": gen.get_state(), "python": random.getstate(), "numpy": np.random.get_state(),
                    "torch": torch.get_rng_state(),
                    "cuda": torch.cuda.get_rng_state() if dev.type == "cuda" else None},
        }, out / "ckpt_last.pt")

    print(f"setup {time.perf_counter() - t_setup:.1f}s", flush=True)
    exit_code = 0
    while not st["done"] and st["epoch"] < E:
        epoch = st["epoch"] + 1
        model.train()
        t_ep = time.perf_counter()
        loss_sum = torch.zeros((), device=dev)
        correct = torch.zeros((), device=dev, dtype=torch.long)
        nonfinite = torch.zeros((), device=dev, dtype=torch.long)
        seen, steps_ep, lr, q = 0, 0, 0.0, 0.0
        for xb, yb in loader:
            lr, q = lr_q(st["step"])
            if q >= 1.0:
                st["done"] = True
                break
            for g in opt.param_groups:
                g["lr"] = lr
            xb = xb.to(dev, non_blocking=True)
            if tr["channels_last"]:
                xb = xb.contiguous(memory_format=torch.channels_last)
            yb = yb.to(dev, non_blocking=True)
            with amp_ctx(dev, amp_dtype):
                logits = model(xb)
                loss = F.cross_entropy(logits, yb, label_smoothing=tr["label_smoothing"])
            opt.zero_grad(set_to_none=True)
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                opt.step()
            ema.update()
            lf = loss.detach().float()
            fin = torch.isfinite(lf)
            loss_sum += torch.where(fin, lf, torch.zeros_like(lf)) * yb.numel()
            nonfinite += (~fin).long()
            correct += (logits.detach().argmax(1) == yb).sum()
            seen += yb.numel()
            steps_ep += 1
            st["step"] += 1
            if st["step"] == W:
                st["t_w"] = elapsed_now()
        if steps_ep == 0:
            break
        if dev.type == "cuda":
            torch.cuda.synchronize()
        ep_time = time.perf_counter() - t_ep
        st["epoch"] = epoch
        st["epoch_times"].append(ep_time)
        st["elapsed"] = elapsed_now()
        rec = {"epoch": epoch, "steps": steps_ep, "train_loss": loss_sum.item() / max(seen, 1),
               "train_acc": correct.item() / max(seen, 1), "val_acc": None, "val_acc_live": None, "lr": lr, "q": q,
               "epoch_seconds": ep_time, "elapsed_train_s": st["elapsed"], "nonfinite": int(nonfinite.item()),
               "global_step": st["step"]}
        if scaler is not None:
            rec["grad_scale"] = scaler.get_scale()

        killed = None
        ids = kl["t_epoch_epochs"]
        t_epoch = None
        if epoch >= max(ids):
            t_epoch = float(np.mean([st["epoch_times"][e - 1] for e in ids]))
        if epoch == max(ids):
            projected = E_final * t_epoch / 60.0
            rec["projected_min"] = projected
            print(f"PROJECTED: t_epoch={t_epoch:.2f}s full run {projected:.1f} min (limit {kl['max_projected_min']})",
                  flush=True)
            if projected > kl["max_projected_min"] and M is None:
                killed = f"throughput: projected {projected:.1f} min > {kl['max_projected_min']} min"
        kill_epoch = math.ceil(kl["val_check_frac"] * E_final)
        if epoch % eval_every == 0 or epoch == kill_epoch or epoch == E:
            rec["val_acc"] = accuracy(predict_logits(ema.model, D["val_x"], ["id"], cfg, mean, std)["id"], D["val_y"])
            rec["val_acc_live"] = accuracy(predict_logits(model, D["val_x"], ["id"], cfg, mean, std)["id"], D["val_y"])
        if killed is None and rec["nonfinite"] >= kl["max_nonfinite_per_epoch"]:
            killed = f"non-finite losses: {rec['nonfinite']} in epoch {epoch}"
        if killed is None and epoch == kill_epoch and rec["val_acc"] < kl["val_min_acc"]:
            killed = f"VAL5K(ema, none)={rec['val_acc']:.4f} < {kl['val_min_acc']} at epoch {epoch}"
        if killed is not None and not kill_on:
            print("KILL CHECK (disabled in smoke): " + killed, flush=True)
            killed = None

        recent = float(np.mean(st["epoch_times"][-3:]))
        eta = max(E - epoch, 0) * recent
        with open(out / "metrics.jsonl", "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec) + "\n")
        st["history"].append(rec)
        va = "-" if rec["val_acc"] is None else f"{rec['val_acc']:.4f}"
        vl = "-" if rec["val_acc_live"] is None else f"{rec['val_acc_live']:.4f}"
        print(f"ep {epoch}/{E} | loss {rec['train_loss']:.4f} | train {rec['train_acc']:.4f} | val ema {va} live {vl} | "
              f"lr {lr:.5f} q {q:.4f} | {ep_time:.1f}s | elapsed {st['elapsed']:.0f}s | ETA {eta / 60:.1f}m | "
              f"nf {rec['nonfinite']}", flush=True)
        if killed is not None:
            st["done"], st["status"], st["kill_reason"] = True, "killed", killed
            print("KILL: " + killed, flush=True)
            exit_code = 2
        if epoch % ckpt_every == 0 or st["done"]:
            save_ckpt()

    if st["status"] == "running":
        st["status"] = "completed"
    st["done"] = True
    save_ckpt()
    train_seconds = st["elapsed"]
    full_epochs = st["step"] // spe
    short = full_epochs < E_final
    print(f"TRAIN END: status={st['status']} epochs={st['epoch']} of {E_final} train_seconds={train_seconds:.0f} "
          f"short={short}", flush=True)

    # ------------------------------------------------------------------ final evaluation (no selection)
    t_fe = time.perf_counter()
    nets = {"live": model, "ema": ema.model}
    probs = {}
    for w, net in nets.items():
        for split, X in (("val", D["val_x"]), ("test", D["test_x"])):
            views = ["id", "flip"] if (cpu_smoke and split == "test") else TTA6_ORDER  # CPU time only
            for o, p in option_probs(predict_logits(net, X, views, cfg, mean, std), cfg).items():
                probs[(w, split, o)] = p
    vy = D["val_y"]
    n_correct = {(w, o): int((probs[(w, "val", o)].argmax(1) == vy).sum().item())
                 for w in nets for o in ("none", "flip2", "tta6")}
    val_acc_all = {f"{w}_{o}": c / D["n_val"] for (w, o), c in n_correct.items()}
    weights, tta = ev["submit_weights"], ev["submit_tta"]
    val_acc = val_acc_all[f"{weights}_{tta}"]
    print("VAL5K: " + " ".join(f"{k}={v:.4f}" for k, v in val_acc_all.items()), flush=True)
    print(f"SUBMIT: weights={weights} tta={tta} VAL5K={val_acc:.4f} (fixed before training, no selection)", flush=True)

    for (w, split, o), p in probs.items():
        np.save(out / f"{split}_probs_{w}_{o}.npy", p.cpu().numpy().astype(np.float32))
    for split in ("val", "test"):
        logits = torch.log(probs[(weights, split, tta)].clamp_min(1e-12))
        np.save(out / f"{split}_logits.npy", logits.cpu().numpy().astype(np.float32))
    write_submission(out / "submission.csv", D["test_ids"], probs[(weights, "test", tta)].argmax(1).cpu().numpy())
    atomic_save(nets[weights].state_dict(), out / "model.pt")
    final_eval_s = time.perf_counter() - t_fe

    summary = {
        "strategy": "strat_11_s3_recipe_400ep", "status": st["status"], "kill_reason": st["kill_reason"],
        "params": total_params, "val_acc": val_acc, "val_split": "VAL5K" if not cpu_smoke else f"VAL5K[:{D['n_val']}]",
        "val_acc_all": val_acc_all, "val_correct_all": {f"{w}_{o}": c for (w, o), c in n_correct.items()},
        "chosen": {"weights": weights, "tta": tta},
        "logits_note": "test_logits/val_logits: log of the averaged softmax of the submitted weights and views.",
        "epochs": st["epoch"], "planned_epochs": E_final, "full_epochs": full_epochs, "steps": st["step"],
        "steps_per_epoch": spe, "warmup_steps": W, "short_run": short, "train_seconds": train_seconds,
        "train_minutes": train_seconds / 60.0, "final_eval_seconds": final_eval_s,
        "epoch_seconds_mean": float(np.mean(st["epoch_times"])) if st["epoch_times"] else None,
        "time_budget_min": M, "n_train": n_train, "n_val": D["n_val"], "full_train": cfg["data"]["full_train"],
        "batch_size": bs, "workers": workers, "smoke": smoke, "resumed": bool(args.resume), "gpu": gpu, "amp": amp_name,
        "grad_scaler": use_scaler, "channels_last": bool(tr["channels_last"]), "torch_version": torch.__version__,
        "numpy_version": np.__version__, "python_version": platform.python_version(), "seed": cfg["seed"],
        "config_sha256": cfg_sha, "data_sha256": data_sha,
        "peak_gpu_mem_gb": torch.cuda.max_memory_allocated() / 1e9 if dev.type == "cuda" else None,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"DONE: outputs in {out} (final eval {final_eval_s:.1f}s)", flush=True)
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
