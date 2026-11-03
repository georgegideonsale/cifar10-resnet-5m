"""strat_1_wrn16x5_base train entry point (contract: docs/workflow.md, spec: strategy.md).

python strat_1_train.py --config strat_1_config.yaml --data cifar_comp_v1.npz --out outputs
       [--smoke] [--resume outputs/ckpt_last.pt] [--time-budget-min 80] [--stop-after-epoch K]

--stop-after-epoch K: test hook only. Save ckpt_last.pt after epoch K and exit without final outputs
(simulates a VM loss, so --resume can be checked against an uninterrupted run).
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

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from strat_1_model import build_model  # noqa: E402

# View name -> (flip, dy, dx). Shift = crop at offset (pad+dy, pad+dx) of the reflect-padded image.
# Flip is applied first, then the shift.
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
    ap.add_argument("--resume", default=None)
    ap.add_argument("--time-budget-min", type=float, default=None)
    ap.add_argument("--stop-after-epoch", type=int, default=None)
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
        dev = torch.device("cuda")
        gpu = torch.cuda.get_device_name(0)
        torch.backends.cudnn.benchmark = bool(cfg["train"]["cudnn_benchmark"])
        name = cfg["train"]["amp_dtype"]
        if name == "auto":
            name = "bf16" if torch.cuda.is_bf16_supported() and torch.cuda.get_device_capability(0)[0] >= 8 else "fp16"
        amp_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}[name]
        use_scaler = amp_dtype == torch.float16
    else:
        dev, gpu, amp_dtype, use_scaler, name = torch.device("cpu"), "cpu", None, False, "none"
    return dev, gpu, amp_dtype, use_scaler, name


def build_and_check(cfg):
    model = build_model(cfg["model"])
    total = sum(p.numel() for p in model.parameters())
    print(f"PARAMS total={total:,} (all parameters, trainable+frozen)", flush=True)
    assert total <= 5_000_000, f"param limit exceeded: {total}"
    assert total <= cfg["param_limit"], f"param limit exceeded: {total}"
    assert total == cfg["expected_params"], f"param count {total} != spec {cfg['expected_params']}"
    return model, total


def load_data(npz_path, cfg, smoke, dev, need_train=True):
    """Load the npz once. Train = not in val_idx. Returns uint8 NHWC tensors on dev."""
    with np.load(npz_path) as z:
        x_train, y_train = z["x_train"], z["y_train"]
        x_test, test_ids, val_idx = z["x_test"], z["test_ids"], z["val_idx"]
    sm = cfg["smoke"]
    mask = np.ones(len(x_train), dtype=bool)
    mask[val_idx] = False
    if cfg["data"]["full_train"]:
        mask[:] = True
    train_idx = np.nonzero(mask)[0]
    if smoke:
        train_idx = train_idx[: sm["n_train"]]
        val_idx = val_idx[: sm["val_n"]]
    d = {
        "val_x": torch.from_numpy(np.ascontiguousarray(x_train[val_idx])).to(dev),
        "val_y": torch.from_numpy(y_train[val_idx].astype(np.int64)).to(dev),
        "test_x": torch.from_numpy(np.ascontiguousarray(x_test)).to(dev),
        "test_ids": test_ids.astype(np.int64),
        "n_val": len(val_idx),
    }
    if need_train:
        p = cfg["data"]["pad"]
        xp = np.pad(x_train[train_idx], ((0, 0), (p, p), (p, p), (0, 0)), mode=cfg["data"]["pad_mode"])
        d["train_xpad"] = torch.from_numpy(np.ascontiguousarray(xp)).to(dev)  # (N, 40, 40, 3) uint8
        d["train_y"] = torch.from_numpy(y_train[train_idx].astype(np.int64)).to(dev)
        d["n_train"] = len(train_idx)
    return d


def norm_consts(cfg, dev):
    mean = torch.tensor(cfg["data"]["mean"], device=dev).view(1, 3, 1, 1)
    std = torch.tensor(cfg["data"]["std"], device=dev).view(1, 3, 1, 1)
    return mean, std


# ----------------------------------------------------------------------------- augmentation (GPU, batch)
def augment_batch(xpad, idx, cfg, mean, std):
    """Random 32x32 crop of the reflect-padded image, flip p=0.5, /255, normalize, Cutout 16 (fill 0)."""
    a = cfg["augment"]
    dev = xpad.device
    B, P, S = idx.numel(), xpad.shape[1], a["crop_size"]
    ar = torch.arange(S, device=dev)
    dy = torch.randint(0, P - S + 1, (B,), device=dev)
    dx = torch.randint(0, P - S + 1, (B,), device=dev)
    flip = torch.rand(B, device=dev) < a["flip_p"]
    rows = dy[:, None] + ar[None, :]
    cols = dx[:, None] + torch.where(flip[:, None], (S - 1) - ar[None, :], ar[None, :])
    x = xpad[idx[:, None, None], rows[:, :, None], cols[:, None, :]]  # (B, S, S, 3) uint8
    x = x.permute(0, 3, 1, 2).float().div_(255.0)  # NCHW view with channels_last strides
    x = (x - mean) / std
    half = a["cutout_size"] // 2
    cy = torch.randint(0, S, (B,), device=dev)
    cx = torch.randint(0, S, (B,), device=dev)
    on = torch.rand(B, device=dev) < a["cutout_p"]
    r_in = (ar[None, :] >= (cy - half)[:, None]) & (ar[None, :] <= (cy + half - 1)[:, None])
    c_in = (ar[None, :] >= (cx - half)[:, None]) & (ar[None, :] <= (cx + half - 1)[:, None])
    m = r_in[:, :, None] & c_in[:, None, :] & on[:, None, None]
    x = x.masked_fill(m[:, None, :, :], a["cutout_fill"])
    return x.contiguous(memory_format=torch.channels_last)


# ----------------------------------------------------------------------------- inference
def amp_ctx(dev, amp_dtype):
    """fp16/bf16 autocast on CUDA only. CPU (smoke) runs in float32."""
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
def predict_logits(model, x_u8, views, cfg, mean, std, amp_dtype):
    """eval mode, no BN update, no test statistics. Returns {view: float32 logits on device}."""
    model.eval()
    dev = x_u8.device
    bs, pad = cfg["eval"]["batch_size"], cfg["eval"]["tta6_shift_pad"]
    out = {v: [] for v in views}
    for i in range(0, x_u8.shape[0], bs):
        xb = x_u8[i: i + bs].permute(0, 3, 1, 2).float().div_(255.0)
        xb = (xb - mean) / std
        for v in views:
            with amp_ctx(dev, amp_dtype):
                out[v].append(model(make_view(xb, v, pad)).float())
    return {v: torch.cat(t) for v, t in out.items()}


def option_probs(logits, cfg):
    """{view: logits} -> {'none', 'flip2', 'tta6' (if all views)}: averaged softmax probabilities (float32)."""
    P = {v: torch.softmax(l, dim=1) for v, l in logits.items()}
    res = {"none": P["id"], "flip2": 0.5 * (P["id"] + P["flip"])}
    if all(v in P for v in TTA6_ORDER):
        res["tta6"] = sum(w * P[v] for w, v in zip(cfg["eval"]["tta6_weights"], TTA6_ORDER))
    return res


def accuracy(logits_or_probs, y):
    return (logits_or_probs.argmax(1) == y).float().mean().item()


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
    def __init__(self, model, decay):
        self.model = copy.deepcopy(model).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.decay = decay
        self._bind(model)

    def _bind(self, model):
        self.ep, self.mp = list(self.model.parameters()), list(model.parameters())
        self.eb, self.mb = list(self.model.buffers()), list(model.buffers())

    @torch.no_grad()
    def update(self):
        torch._foreach_lerp_(self.ep, self.mp, 1.0 - self.decay)
        for e, m in zip(self.eb, self.mb):  # BN running stats + num_batches_tracked from the live model
            e.copy_(m)


# ----------------------------------------------------------------------------- main
def main():
    args = parse_args()
    t_setup = time.perf_counter()
    cfg, cfg_sha = load_cfg(args.config)
    smoke = args.smoke
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    seed = cfg["seed"]
    seed_all(seed)

    # First action: build the model, count ALL params, assert.
    model, total_params = build_and_check(cfg)

    dev, gpu, amp_dtype, use_scaler, amp_name = setup_device(cfg)
    tr, sch, ev, kl, sm = cfg["train"], cfg["schedule"], cfg["eval"], cfg["kill"], cfg["smoke"]
    bs = sm["batch_size"] if smoke else tr["batch_size"]
    warmup_epochs = sm["warmup_epochs"] if smoke else tr["warmup_epochs"]
    eval_every = sm["eval_every"] if smoke else ev["every"]
    ckpt_every = sm["ckpt_every"] if smoke else cfg["checkpoint"]["every"]
    kill_on = (not smoke) or sm["kill_rules"]
    M = args.time_budget_min if args.time_budget_min is not None else sch["time_budget_min"]
    assert sch["e_rule_epoch"] <= warmup_epochs or smoke, "E must be known before the decay phase starts"
    print(f"seed={seed} device={dev} gpu={gpu} amp={amp_name} scaler={use_scaler} smoke={smoke} "
          f"time_budget_min={M}", flush=True)

    data_sha = sha256_file(args.data)
    D = load_data(args.data, cfg, smoke, dev)
    mean, std = norm_consts(cfg, dev)
    n_train = D["n_train"]
    spe = n_train // bs if tr["drop_last"] else math.ceil(n_train / bs)
    W = warmup_epochs * spe
    print(f"data: train={n_train} val={D['n_val']} test={D['test_x'].shape[0]} steps/epoch={spe} "
          f"warmup_steps={W} data_sha256={data_sha[:12]}", flush=True)

    if tr["channels_last"]:
        model = model.to(memory_format=torch.channels_last)
    model = model.to(dev)
    decay, no_decay = [], []
    for _, p in model.named_parameters():
        (decay if p.ndim > 1 else no_decay).append(p)  # conv + Linear weights | BN weight/bias + Linear bias
    opt = torch.optim.SGD(
        [{"params": decay, "weight_decay": tr["weight_decay"]}, {"params": no_decay, "weight_decay": 0.0}],
        lr=tr["warmup_lr_start"], momentum=tr["momentum"], nesterov=tr["nesterov"], dampening=tr["dampening"])
    scaler = torch.amp.GradScaler("cuda", init_scale=tr["grad_scaler_init_scale"]) if use_scaler else None
    crit = nn.CrossEntropyLoss(label_smoothing=tr["label_smoothing"])
    params = list(model.parameters())

    compile_on = bool(tr["compile"]) and dev.type == "cuda"
    train_model = torch.compile(model) if compile_on else model

    ema = None
    st = {"step": 0, "epoch": 0, "E": (sm["epochs"] if smoke else None), "t_epoch": None, "t_w": None,
          "elapsed": 0.0, "epoch_times": [], "history": [], "done": False, "status": "running",
          "kill_reason": None}

    if args.resume:
        ck = torch.load(args.resume, map_location=dev, weights_only=False)
        if ck["smoke"] != smoke:
            raise SystemExit("resume: smoke flag differs from the checkpoint")
        if ck["config_sha256"] != cfg_sha:
            print("WARN: config sha256 differs from the checkpoint", flush=True)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])
        if scaler is not None and ck["scaler"] is not None:
            scaler.load_state_dict(ck["scaler"])
        if ck["ema"] is not None:
            ema = EMA(model, tr["ema_decay"])
            ema.model.load_state_dict(ck["ema"])
        st = ck["state"]
        random.setstate(ck["rng"]["python"])
        np.random.set_state(ck["rng"]["numpy"])
        # map_location=dev moves the RNG ByteTensors to the GPU; set_rng_state needs CPU ByteTensors.
        torch.set_rng_state(ck["rng"]["torch"].cpu())
        if dev.type == "cuda" and ck["rng"]["cuda"] is not None:
            torch.cuda.set_rng_state(ck["rng"]["cuda"].cpu())
        print(f"RESUMED from {args.resume}: epoch={st['epoch']} step={st['step']} E={st['E']} "
              f"elapsed={st['elapsed']:.1f}s done={st['done']}", flush=True)
        with open(out / "metrics.jsonl", "w", encoding="utf-8") as fh:  # rebuild from the checkpoint history
            for rec in st["history"]:
                fh.write(json.dumps(rec) + "\n")
    else:
        (out / "metrics.jsonl").write_text("", encoding="utf-8")

    def log_metrics(rec):
        st["history"].append(rec)
        with open(out / "metrics.jsonl", "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec) + "\n")

    def save_ckpt():
        st["elapsed"] = elapsed_now()  # periodic eval time counts as training-loop time
        atomic_save({
            "model": model.state_dict(), "ema": ema.model.state_dict() if ema else None,
            "optimizer": opt.state_dict(), "scaler": scaler.state_dict() if scaler else None,
            "state": st, "smoke": smoke, "config_sha256": cfg_sha, "steps_per_epoch": spe, "warmup_steps": W,
            "rng": {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
                    "cuda": torch.cuda.get_rng_state() if dev.type == "cuda" else None},
        }, out / "ckpt_last.pt")

    def lr_q(step, elapsed):
        """LR and progress q. Warmup: linear per step. Then lr_peak * (1 - q)."""
        if step < W:
            frac = step / (W - 1) if W > 1 else 1.0
            return tr["warmup_lr_start"] + (tr["lr_peak"] - tr["warmup_lr_start"]) * frac, 0.0
        # A non-positive denominator means the step or time budget is already used up: q term = 1 (stop).
        den_s, den_t = st["E"] * spe - W, M * 60.0 - st["t_w"]
        q_step = (step - W) / den_s if den_s > 0 else 1.0
        q_time = (elapsed - st["t_w"]) / den_t if den_t > 0 else 1.0
        q = min(max(max(q_step, q_time), 0.0), 1.0)
        return tr["lr_peak"] * (1.0 - q), q

    def sync():
        if dev.type == "cuda":
            torch.cuda.synchronize()

    def evaluate_no_tta():
        acc_live = accuracy(predict_logits(model, D["val_x"], ["id"], cfg, mean, std, amp_dtype)["id"], D["val_y"])
        acc_ema = None
        if ema is not None:
            acc_ema = accuracy(predict_logits(ema.model, D["val_x"], ["id"], cfg, mean, std, amp_dtype)["id"],
                               D["val_y"])
        model.train()
        return acc_live, acc_ema

    print(f"setup {time.perf_counter() - t_setup:.1f}s", flush=True)
    model.train()
    t0 = time.perf_counter()
    el0 = st["elapsed"]

    def elapsed_now():
        return el0 + (time.perf_counter() - t0)

    exit_code = 0
    interrupted = False
    while not st["done"]:
        if st["step"] >= W:
            if lr_q(st["step"], elapsed_now())[1] >= 1.0:
                st["done"] = True
                break
        epoch = st["epoch"] + 1
        t_ep = time.perf_counter()
        loss_sum = torch.zeros((), device=dev)
        correct = torch.zeros((), device=dev, dtype=torch.long)
        nonfinite = torch.zeros((), device=dev, dtype=torch.long)
        gnorm = torch.zeros((), device=dev)
        seen, steps_ep, lr, q = 0, 0, 0.0, 0.0
        perm = torch.randperm(n_train, device=dev)
        for i in range(spe):
            lr, q = lr_q(st["step"], elapsed_now())
            if q >= 1.0:
                st["done"] = True
                break
            for g in opt.param_groups:
                g["lr"] = lr
            idx = perm[i * bs: (i + 1) * bs]
            x = augment_batch(D["train_xpad"], idx, cfg, mean, std)
            y = D["train_y"][idx]
            with amp_ctx(dev, amp_dtype):
                logits = train_model(x)
                loss = crit(logits, y)
            opt.zero_grad(set_to_none=True)
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
            else:
                loss.backward()
            grads = [p.grad for p in params if p.grad is not None]
            gnorm = torch.linalg.vector_norm(torch.stack(torch._foreach_norm(grads)))
            if scaler is not None:
                scaler.step(opt)
                scaler.update()
            else:
                opt.step()
            lf = loss.detach().float()
            fin = torch.isfinite(lf)
            loss_sum += torch.where(fin, lf, torch.zeros_like(lf)) * idx.numel()
            nonfinite += (~fin).long()
            correct += (logits.detach().argmax(1) == y).sum()
            seen += idx.numel()
            steps_ep += 1
            st["step"] += 1
            if st["step"] == W:  # end of warmup: EMA starts as a copy of the live model
                st["t_w"] = elapsed_now()
                ema = EMA(model, tr["ema_decay"])
            elif st["step"] > W:
                ema.update()
        if steps_ep == 0:
            break
        sync()
        ep_time = time.perf_counter() - t_ep
        st["epoch"] = epoch
        st["epoch_times"].append(ep_time)
        st["elapsed"] = elapsed_now()
        if st["E"] is not None and st["step"] >= st["E"] * spe:
            st["done"] = True
        rec = {"epoch": epoch, "steps": steps_ep, "train_loss": loss_sum.item() / max(seen, 1),
               "train_acc": correct.item() / max(seen, 1), "val_acc": None, "val_acc_ema": None,
               "lr": lr, "q": q, "epoch_seconds": ep_time, "elapsed_train_s": st["elapsed"],
               "grad_norm": gnorm.item(), "nonfinite": int(nonfinite.item()), "global_step": st["step"]}
        if scaler is not None:
            rec["grad_scale"] = scaler.get_scale()

        # Epoch-count rule (set once at the end of e_rule_epoch).
        killed = None
        if not smoke and st["E"] is None and epoch == sch["e_rule_epoch"]:
            ids = sch["t_epoch_epochs"]
            st["t_epoch"] = float(np.mean([st["epoch_times"][e - 1] for e in ids]))
            budget = M * 60.0 - sch["e_reserve_s"]
            e_raw = sch["e_rule_epoch"] + math.floor((budget - st["elapsed"]) / st["t_epoch"])
            st["E"] = min(sch["e_max"], sch["e_round"] * math.floor(e_raw / sch["e_round"]))
            # Stored in this epoch's metrics line (contract: one line per epoch).
            rec.update({"E_set": True, "E": st["E"], "E_raw": e_raw, "t_epoch": st["t_epoch"], "budget_s": budget})
            print(f"E SET: E={st['E']} (E_raw={e_raw}) t_epoch={st['t_epoch']:.2f}s budget={budget:.0f}s", flush=True)
            if st["E"] < sch["e_min"]:
                killed = f"throughput: E={st['E']} < {sch['e_min']} (t_epoch={st['t_epoch']:.1f}s). " \
                         f"Check AMP, channels_last and the data path."
                if not kill_on:
                    print("KILL CHECK (disabled): " + killed, flush=True)
                    killed = None

        kill_epoch = math.ceil(kl["val_check_frac"] * st["E"]) if st["E"] is not None else None
        if epoch % eval_every == 0 or epoch == kill_epoch:
            rec["val_acc"], rec["val_acc_ema"] = evaluate_no_tta()
        if killed is None and rec["nonfinite"] >= kl["max_nonfinite_per_epoch"]:
            killed = f"non-finite losses: {rec['nonfinite']} in epoch {epoch}"
        if killed is None and epoch == kill_epoch and rec["val_acc"] < kl["val_min_acc"]:
            killed = f"VAL5K(live, none)={rec['val_acc']:.4f} < {kl['val_min_acc']} at epoch {epoch}"
        if killed is not None and not kill_on:
            print("KILL CHECK (disabled in smoke): " + killed, flush=True)
            killed = None

        # Progress line + ETA.
        if st["E"] is not None:
            recent = float(np.mean(st["epoch_times"][-3:]))
            eta = min(max(st["E"] - epoch, 0) * recent, max(M * 60.0 - st["elapsed"], 0.0))
            eta_s = f"{eta / 60:.1f}m"
        else:
            eta_s = "n/a"
        rec["eta_s"] = eta if st["E"] is not None else None
        log_metrics(rec)
        va = "-" if rec["val_acc"] is None else f"{rec['val_acc']:.4f}"
        ve = "-" if rec["val_acc_ema"] is None else f"{rec['val_acc_ema']:.4f}"
        print(f"ep {epoch}/{st['E'] or '?'} | loss {rec['train_loss']:.4f} | train {rec['train_acc']:.4f} | "
              f"val {va} ema {ve} | lr {lr:.5f} q {q:.4f} | {ep_time:.1f}s | elapsed {st['elapsed']:.0f}s | "
              f"ETA {eta_s} | gn {rec['grad_norm']:.3f} | nf {rec['nonfinite']}", flush=True)

        if killed is not None:
            st["done"], st["status"], st["kill_reason"] = True, "killed", killed
            print("KILL: " + killed, flush=True)
            exit_code = 2
        if (epoch % ckpt_every == 0 or (not smoke and epoch == sch["e_rule_epoch"]) or st["done"]
                or epoch == args.stop_after_epoch):
            save_ckpt()
        if args.stop_after_epoch is not None and epoch == args.stop_after_epoch and not st["done"]:
            interrupted = True
            break

    if interrupted:
        print(f"STOPPED after epoch {st['epoch']} (--stop-after-epoch). ckpt_last.pt saved. No final outputs.",
              flush=True)
        return 0

    if st["status"] == "running":
        st["status"] = "completed"
    st["done"] = True
    save_ckpt()
    train_seconds = st["elapsed"]
    full_epochs = st["step"] // spe
    short = full_epochs < kl["short_frac"] * st["E"] if st["E"] else True
    print(f"TRAIN END: status={st['status']} epochs={st['epoch']} full_epochs={full_epochs} E={st['E']} "
          f"effective_epochs={st['step'] / spe:.2f} train_seconds={train_seconds:.0f} short={short}", flush=True)

    # ------------------------------------------------------------------ final evaluation and selection
    t_fe = time.perf_counter()
    nets = {"live": model, "ema": ema.model if ema is not None else model}
    if ema is None:
        print("WARN: EMA never started (stopped before end of warmup). EMA = live weights.", flush=True)
    probs, raw = {}, {}
    for w, net in nets.items():
        for split, X in (("val", D["val_x"]), ("test", D["test_x"])):
            views = TTA6_ORDER if (split == "val" or not smoke or sm["tta6_test"]) else ["id", "flip"]
            L = predict_logits(net, X, views, cfg, mean, std, amp_dtype)
            raw[(w, split)] = L["id"]
            for o, p in option_probs(L, cfg).items():
                probs[(w, split, o)] = p
    vy = D["val_y"]
    n_correct = {(w, o): int((probs[(w, "val", o)].argmax(1) == vy).sum().item())
                 for w in nets for o in ("none", "flip2", "tta6")}
    val_acc_all = {f"{w}_{o}": n_correct[(w, o)] / D["n_val"] for (w, o) in n_correct}
    weights = "ema" if n_correct[("ema", "none")] >= n_correct[("live", "none")] else "live"
    tta = "flip2" if n_correct[(weights, "flip2")] > n_correct[(weights, "none")] else "none"
    val_acc = val_acc_all[f"{weights}_{tta}"]
    print("VAL5K: " + " ".join(f"{k}={v:.4f}" for k, v in val_acc_all.items()), flush=True)
    print(f"CHOSEN: weights={weights} tta={tta} VAL5K={val_acc:.4f} "
          f"(best-of-4 selection adds about +0.1 pp optimistic bias, estimate)", flush=True)

    for (w, split, o), p in probs.items():
        np.save(out / f"{split}_probs_{w}_{o}.npy", p.cpu().numpy().astype(np.float32))

    def chosen_logits(split):
        if tta == "none":
            return raw[(weights, split)]
        return torch.log(probs[(weights, split, tta)].clamp_min(1e-12))

    np.save(out / "test_logits.npy", chosen_logits("test").cpu().numpy().astype(np.float32))
    np.save(out / "val_logits.npy", chosen_logits("val").cpu().numpy().astype(np.float32))
    test_pred = probs[(weights, "test", tta)].argmax(1).cpu().numpy()
    write_submission(out / "submission.csv", D["test_ids"], test_pred)
    final_sd = nets[weights].state_dict()
    atomic_save(final_sd, out / "model.pt")
    final_eval_s = time.perf_counter() - t_fe

    summary = {
        "strategy": "strat_1_wrn16x5_base", "status": st["status"], "kill_reason": st["kill_reason"],
        "params": total_params, "val_acc": val_acc, "val_split": "VAL5K" if not smoke else f"VAL5K[:{D['n_val']}]",
        "val_acc_all": val_acc_all, "val_correct_all": {f"{w}_{o}": c for (w, o), c in n_correct.items()},
        "chosen": {"weights": weights, "tta": tta},
        "selection_note": "Best-of-4 selection on VAL5K adds about +0.1 pp optimistic bias (estimate). "
                          "tta6 is a diagnostic only, never used for the submission.",
        "logits_note": "test_logits/val_logits: raw logits if tta=none, else log of the averaged softmax.",
        "epochs": st["epoch"], "full_epochs": full_epochs, "E": st["E"], "t_epoch": st["t_epoch"],
        "effective_epochs": st["step"] / spe, "steps": st["step"], "steps_per_epoch": spe, "warmup_steps": W,
        "short_run": short, "train_seconds": train_seconds, "final_eval_seconds": final_eval_s,
        "epoch1_seconds": st["epoch_times"][0] if st["epoch_times"] else None,
        "time_budget_min": M, "n_train": n_train, "n_val": D["n_val"], "full_train": cfg["data"]["full_train"],
        "batch_size": bs, "smoke": smoke, "resumed": bool(args.resume),
        "gpu": gpu, "amp": amp_name, "grad_scaler": use_scaler, "compile": compile_on,
        "channels_last": bool(tr["channels_last"]), "torch_version": torch.__version__,
        "numpy_version": np.__version__, "python_version": platform.python_version(), "seed": seed,
        "config_sha256": cfg_sha, "data_sha256": data_sha,
        "peak_gpu_mem_gb": torch.cuda.max_memory_allocated() / 1e9 if dev.type == "cuda" else None,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"DONE: outputs in {out} (final eval {final_eval_s:.1f}s)", flush=True)
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
