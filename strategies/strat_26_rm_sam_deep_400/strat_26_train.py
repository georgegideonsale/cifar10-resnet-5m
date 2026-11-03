"""strat_26_rm_sam_deep_400 train entry point (contract: docs/workflow.md, spec: strategy.md).

python strat_26_train.py --config strat_26_config.yaml --data cifar_comp_v1.npz --out outputs_s0 --seed 0
       [--smoke] [--resume outputs_s0/ckpt_last.pt] [--time-budget-min M]

Base recipe: strat_10 (port of docs/10-31-2026 - Tarun/kaggle_train_tarun.ipynb, version 25fd8d1).
Added options, off unless the config sets them: RegMixup, MixUp, CutMix, SAM, distillation from a teacher, arch wrn.
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
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from strat_26_model import build_model  # noqa: E402

STRATEGY = "strat_26_rm_sam_deep_400"
# View name -> (flip, dy, dx). Same 6 views as strat_3 to strat_11, so the 6-view reads compare.
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
    ap.add_argument("--seed", type=int, default=None, help="overrides the config seed")
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
    final = total * cfg["members"]
    print(f"PARAMS one member={total:,} members={cfg['members']} final model={final:,} (all parameters)", flush=True)
    assert final <= 5_000_000, f"param limit exceeded: {final}"
    assert final <= cfg["param_limit"], f"param limit exceeded: {final}"
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


# ----------------------------------------------------------------------------- augmentation
def train_transform(cfg):
    """uint8 CHW -> crop + flip, policy, float, normalize, random erasing (fill 0 = the mean colour)."""
    a = cfg["augment"]
    t = [v2.RandomCrop(32, padding=a["crop_padding"], padding_mode=a["pad_mode"]), v2.RandomHorizontalFlip()]
    if a["policy"] == "trivial":
        t.append(v2.TrivialAugmentWide())
    elif a["policy"] == "randaug":
        t.append(v2.RandAugment(num_ops=2, magnitude=9))
    else:
        assert a["policy"] == "none", a["policy"]
    t += [v2.ToDtype(torch.float32, scale=True), v2.Normalize(cfg["data"]["mean"], cfg["data"]["std"])]
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


def cutmix_batch(x, lam):
    """Paste one random box from a shuffled batch. Returns images, perm and lam corrected to the true box area."""
    B, _, H, W = x.shape
    pm = torch.randperm(B, device=x.device)
    cut = math.sqrt(1.0 - lam)
    h, w = int(H * cut), int(W * cut)
    cy, cx = random.randrange(H), random.randrange(W)
    y0, y1 = max(cy - h // 2, 0), min(cy + h // 2, H)
    x0, x1 = max(cx - w // 2, 0), min(cx + w // 2, W)
    out = x.clone()
    out[:, :, y0:y1, x0:x1] = x[pm][:, :, y0:y1, x0:x1]
    return out, pm, 1.0 - (y1 - y0) * (x1 - x0) / (H * W)


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


# ----------------------------------------------------------------------------- image-change grid (VAL5K only)
def g_sharp(a, f):
    k = (torch.tensor([[1, 1, 1], [1, 5, 1], [1, 1, 1.]], device=a.device).view(1, 1, 3, 3) / 13).repeat(3, 1, 1, 1)
    out = a.clone()
    out[:, :, 1:-1, 1:-1] = F.conv2d(a, k, padding=1, groups=3)[:, :, 1:-1, 1:-1]
    return (out + f * (a - out)).clamp(0, 1)


def g_contrast(a, f):
    m = a.mean(1, keepdim=True).mean((2, 3), keepdim=True)
    return (m + f * (a - m)).clamp(0, 1)


GRID = {"sharp2": lambda a: g_sharp(a, 2.0), "contrast1.2": lambda a: g_contrast(a, 1.2),
        "all_up": lambda a: g_sharp(g_contrast((a * 1.05).clamp(0, 1), 1.1), 1.5)}


@torch.no_grad()
def grid_read(net, val_x, val_y, mean, std):
    """flip2 accuracy under mild image changes, same grid as strat_10_shift_grid.py."""
    net.eval()
    X = val_x.permute(0, 3, 1, 2).float() / 255
    res = {}
    for name, f in GRID.items():
        c = 0
        for i in range(0, len(X), 500):
            b = (f(X[i:i + 500]) - mean) / std
            c += ((net(b).softmax(1) + net(b.flip(3)).softmax(1)).argmax(1) == val_y[i:i + 500]).sum().item()
        res[name] = c / len(X)
    return res


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


@contextlib.contextmanager
def frozen_bn(model):
    """BN running statistics stay as they are in the second SAM pass."""
    norms = [m for m in model.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]
    momenta = [m.momentum for m in norms]
    for m in norms:
        m.momentum = 0.0
    try:
        yield
    finally:
        for m, v in zip(norms, momenta):
            m.momentum = v


# ----------------------------------------------------------------------------- main
def main():
    args = parse_args()
    t_setup = time.perf_counter()
    cfg, cfg_sha = load_cfg(args.config)
    seed = cfg["seed"] if args.seed is None else args.seed
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    seed_all(seed)

    # First action: build the model, count ALL params, assert.
    model, total_params = build_and_check(cfg)

    dev, gpu, amp_dtype, use_scaler, amp_name = setup_device(cfg)
    tr, ev, kl, sm, mx, kdc = cfg["train"], cfg["eval"], cfg["kill"], cfg["smoke"], cfg["mix"], cfg["kd"]
    smoke = args.smoke
    cpu_smoke = smoke and dev.type != "cuda"
    gpu_smoke = smoke and dev.type == "cuda"
    bs = sm["cpu_batch_size"] if cpu_smoke else tr["batch_size"]
    E_final = sm["cpu_epochs"] if cpu_smoke else tr["epochs"]
    E = sm["gpu_epochs"] if gpu_smoke else E_final  # GPU smoke stops early with the full-run schedule
    eval_every = 1 if smoke else ev["every"]
    ckpt_every = 1 if smoke else cfg["checkpoint"]["every"]
    kill_on = not smoke
    workers = 0 if cpu_smoke else min(tr["workers"], os.cpu_count() or 1)
    M = args.time_budget_min
    sam_rho = float(cfg["sam_rho"])
    assert not (sam_rho > 0 and use_scaler), "SAM needs bf16 or fp32"
    assert sum(1 for k in ("regmixup_alpha", "mixup_alpha", "cutmix_alpha") if mx[k] > 0) <= 1, "one mix option at most"
    assert (kdc["ckpt"] is not None) == (kdc["alpha"] > 0), "kd: set ckpt and alpha > 0 together"
    print(f"seed={seed} device={dev} gpu={gpu} amp={amp_name} scaler={use_scaler} smoke={smoke} workers={workers} "
          f"epochs={E_final} sam={sam_rho} mix={mx} kd={kdc['ckpt']} time_budget_min={M}", flush=True)

    data_sha = sha256_file(args.data)
    D = load_data(args.data, cfg, cpu_smoke, dev)
    mean, std = norm_consts(cfg, dev)
    n_train = D["n_train"]
    gen = torch.Generator()
    gen.manual_seed(seed)
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
    params = [p for p in model.parameters() if p.requires_grad]

    teacher = None
    if kdc["ckpt"] is not None:  # our own run, same images, at most 5M (rules Q9)
        tcfg = {"arch": kdc["arch"], "widths": kdc["widths"], "blocks": kdc["blocks"], "stem": kdc.get("stem", 16),
                "num_classes": cfg["model"]["num_classes"]}
        teacher = build_model(tcfg)
        assert sum(p.numel() for p in teacher.parameters()) <= 5_000_000, "teacher above 5M (rules Q9)"
        teacher.load_state_dict(torch.load(kdc["ckpt"], map_location="cpu", weights_only=True))
        teacher = teacher.eval().requires_grad_(False).to(dev).to(memory_format=torch.channels_last)
        print(f"teacher {kdc['ckpt']} sha256={sha256_file(kdc['ckpt'])[:12]}", flush=True)

    st = {"step": 0, "epoch": 0, "elapsed": 0.0, "t_w": None, "epoch_times": [], "history": [], "done": False,
          "status": "running", "kill_reason": None}

    if args.resume:
        ck = torch.load(args.resume, map_location=dev, weights_only=False)  # own file (holds RNG states)
        if ck["smoke"] != smoke:
            raise SystemExit("resume: smoke flag differs from the checkpoint")
        if ck["config_sha256"] != cfg_sha or ck["steps_per_epoch"] != spe:
            raise SystemExit("resume: config or steps per epoch differ from the checkpoint")
        if ck["seed"] != seed:  # the checkpoint holds the RNG of its own seed; pass the same --seed
            raise SystemExit(f"resume: seed {seed} differs from the checkpoint seed {ck['seed']}")
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
            "config_sha256": cfg_sha, "seed": seed, "steps_per_epoch": spe, "warmup_steps": W,
            "rng": {"loader": gen.get_state(), "python": random.getstate(), "numpy": np.random.get_state(),
                    "torch": torch.get_rng_state(),
                    "cuda": torch.cuda.get_rng_state() if dev.type == "cuda" else None},
        }, out / "ckpt_last.pt")

    ls = tr["label_smoothing"]

    def ce(logits, y):
        return F.cross_entropy(logits, y, label_smoothing=ls)

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
            # One mix choice per step, so both SAM passes see the same batch.
            x_in, y2, lam = xb, None, 1.0
            if mx["regmixup_alpha"] > 0:
                lam = random.betavariate(mx["regmixup_alpha"], mx["regmixup_alpha"])
                pm = torch.randperm(xb.shape[0], device=dev)
                x_in, y2 = torch.cat([xb, lam * xb + (1.0 - lam) * xb[pm]]), yb[pm]
            elif mx["mixup_alpha"] > 0 and random.random() < mx["mix_prob"]:
                lam = random.betavariate(mx["mixup_alpha"], mx["mixup_alpha"])
                pm = torch.randperm(xb.shape[0], device=dev)
                x_in, y2 = lam * xb + (1.0 - lam) * xb[pm], yb[pm]
            elif mx["cutmix_alpha"] > 0 and random.random() < mx["mix_prob"]:
                x_in, pm, lam = cutmix_batch(xb, random.betavariate(mx["cutmix_alpha"], mx["cutmix_alpha"]))
                y2 = yb[pm]
            soft = None
            if teacher is not None:
                with torch.no_grad(), amp_ctx(dev, amp_dtype):
                    soft = F.log_softmax(teacher(x_in).float() / kdc["temperature"], 1)

            def loss_of(output):
                if mx["regmixup_alpha"] > 0:
                    B = yb.shape[0]
                    loss = 0.5 * ce(output[:B], yb) + 0.5 * (lam * ce(output[B:], yb) + (1.0 - lam) * ce(output[B:], y2))
                elif y2 is not None:
                    loss = lam * ce(output, yb) + (1.0 - lam) * ce(output, y2)
                else:
                    loss = ce(output, yb)
                if soft is not None:
                    t = kdc["temperature"]
                    kd = F.kl_div(F.log_softmax(output.float() / t, 1), soft, reduction="batchmean", log_target=True)
                    loss = (1 - kdc["alpha"]) * loss + kdc["alpha"] * t * t * kd
                return loss

            with amp_ctx(dev, amp_dtype):
                logits = model(x_in)
                loss = loss_of(logits)
            opt.zero_grad(set_to_none=True)
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            elif sam_rho > 0:
                loss.backward()
                with torch.no_grad():
                    moved = [p for p in params if p.grad is not None]
                    saved = [p.detach().clone() for p in moved]  # exact restore: add then subtract drifts by rounding
                    grads = [p.grad for p in moved]
                    norm = torch.linalg.vector_norm(torch.stack(torch._foreach_norm(grads)))
                    e = torch._foreach_mul(grads, sam_rho / (norm + 1e-12))
                    torch._foreach_add_(moved, e)
                opt.zero_grad(set_to_none=True)
                with frozen_bn(model), amp_ctx(dev, amp_dtype):
                    second = loss_of(model(x_in))
                second.backward()
                with torch.no_grad():
                    for p, w0 in zip(moved, saved):
                        p.copy_(w0)
                opt.step()
            else:
                loss.backward()
                opt.step()
            ema.update()
            lf = loss.detach().float()
            fin = torch.isfinite(lf)
            loss_sum += torch.where(fin, lf, torch.zeros_like(lf)) * yb.numel()
            nonfinite += (~fin).long()
            correct += (logits.detach()[: yb.shape[0]].argmax(1) == yb).sum()
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
        if epoch == max(ids):
            t_epoch = float(np.mean([st["epoch_times"][e - 1] for e in ids]))
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
    if st["status"] == "killed":  # no submission file from a killed run
        (out / "summary.json").write_text(json.dumps({"strategy": STRATEGY, "status": "killed", "seed": seed,
                                                      "kill_reason": st["kill_reason"]}, indent=2) + "\n", encoding="utf-8")
        return exit_code

    # ------------------------------------------------------------------ final evaluation (no selection)
    t_fe = time.perf_counter()
    nets = {"live": model, "ema": ema.model}
    probs = {}
    for w, net in nets.items():
        for split, X in (("val", D["val_x"]), ("test", D["test_x"])):
            # CPU time only; a run that submits tta6 needs all views.
            views = ["id", "flip"] if (cpu_smoke and split == "test" and ev["submit_tta"] != "tta6") else TTA6_ORDER
            for o, p in option_probs(predict_logits(net, X, views, cfg, mean, std), cfg).items():
                probs[(w, split, o)] = p
    vy = D["val_y"]
    n_correct = {(w, o): int((probs[(w, "val", o)].argmax(1) == vy).sum().item())
                 for w in nets for o in ("none", "flip2", "tta6")}
    val_acc_all = {f"{w}_{o}": c / D["n_val"] for (w, o), c in n_correct.items()}
    weights, tta = ev["submit_weights"], ev["submit_tta"]
    val_acc = val_acc_all[f"{weights}_{tta}"]
    grid = grid_read(nets[weights], D["val_x"], vy, mean, std)
    print("VAL5K: " + " ".join(f"{k}={v:.4f}" for k, v in val_acc_all.items()), flush=True)
    print("GRID: " + " ".join(f"{k}={v:.4f}" for k, v in grid.items()), flush=True)
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
        "strategy": STRATEGY, "status": st["status"], "kill_reason": st["kill_reason"],
        "params": total_params, "members": cfg["members"], "final_params": total_params * cfg["members"],
        "val_acc": val_acc,
        "val_split": "VAL5K in train: no valid score" if cfg["data"]["full_train"]
                     else ("VAL5K" if not cpu_smoke else f"VAL5K[:{D['n_val']}]"),
        "val_acc_all": val_acc_all, "val_correct_all": {f"{w}_{o}": c for (w, o), c in n_correct.items()},
        "grid": grid, "chosen": {"weights": weights, "tta": tta},
        "logits_note": "test_logits/val_logits: log of the averaged softmax of the submitted weights and views.",
        "epochs": st["epoch"], "planned_epochs": E_final, "full_epochs": full_epochs, "steps": st["step"],
        "steps_per_epoch": spe, "warmup_steps": W, "short_run": short, "train_seconds": train_seconds,
        "train_minutes": train_seconds / 60.0, "final_eval_seconds": final_eval_s,
        "epoch_seconds_mean": float(np.mean(st["epoch_times"])) if st["epoch_times"] else None,
        "time_budget_min": M, "n_train": n_train, "n_val": D["n_val"], "full_train": cfg["data"]["full_train"],
        "batch_size": bs, "workers": workers, "smoke": smoke, "resumed": bool(args.resume), "gpu": gpu, "amp": amp_name,
        "grad_scaler": use_scaler, "channels_last": bool(tr["channels_last"]), "torch_version": torch.__version__,
        "numpy_version": np.__version__, "python_version": platform.python_version(), "seed": seed,
        "config_sha256": cfg_sha, "data_sha256": data_sha, "sam_rho": sam_rho, "mix": mx, "kd": kdc,
        "peak_gpu_mem_gb": torch.cuda.max_memory_allocated() / 1e9 if dev.type == "cuda" else None,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"DONE: outputs in {out} (final eval {final_eval_s:.1f}s)", flush=True)
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
