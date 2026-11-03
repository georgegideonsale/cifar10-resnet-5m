"""Hard gate: count parameters of one or more model files. Exit 1 if the sum > 5,000,000.

Each model file must expose build_model(cfg: dict) -> torch.nn.Module.
Counts ALL parameters (trainable + frozen). Several files = ensemble; the counts are summed.
Usage: python tools/count_params.py <model.py> [<model2.py> ...] [--config cfg.yaml]
"""
import argparse
import importlib.util
import sys
from pathlib import Path

LIMIT = 5_000_000


def load_cfg(path):
    if not path:
        return {}
    import yaml
    return yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="+")
    ap.add_argument("--config")
    a = ap.parse_args()
    cfg = load_cfg(a.config)
    total = 0
    for i, m in enumerate(a.models):
        spec = importlib.util.spec_from_file_location(f"model_{i}", m)
        mod = importlib.util.module_from_spec(spec)
        sys.path.insert(0, str(Path(m).resolve().parent))
        spec.loader.exec_module(mod)
        model = mod.build_model(cfg.get("model", cfg))
        n_all = sum(p.numel() for p in model.parameters())
        n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
        n_buf = sum(b.numel() for b in model.buffers())
        print(f"{m}: params={n_all:,} trainable={n_train:,} frozen={n_all - n_train:,} buffers={n_buf:,}")
        if n_all != n_train:
            print("WARN: frozen parameters present. They still count against the limit.")
        total += n_all
    print(f"TOTAL={total:,} LIMIT={LIMIT:,} HEADROOM={LIMIT - total:,}")
    print("RESULT:", "PASS" if total <= LIMIT else "FAIL")
    return 0 if total <= LIMIT else 1


if __name__ == "__main__":
    sys.exit(main())
