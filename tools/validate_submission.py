"""Validate a submission CSV. Exit 0 = PASS, 1 = FAIL. Run before every submission request.

Hard checks: header exactly id,label; one row per test id; same order as sample_submission.csv;
labels are integers 0-9; no blanks, no BOM, no extra columns.
Soft checks (warnings): file name, predicted class balance, agreement with another submission.
"""
import argparse
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--compare", help="another submission.csv: print agreement rate")
    a = ap.parse_args()
    path = Path(a.csv)
    errors, warns = [], []

    raw = path.read_bytes()
    if raw.startswith(b"\xef\xbb\xbf"):
        errors.append("file starts with a UTF-8 BOM")
    lines = raw.decode("utf-8-sig").replace("\r\n", "\n").rstrip("\n").split("\n")
    if lines[0].strip() != "id,label":
        errors.append(f"header is {lines[0]!r}, expected 'id,label'")

    ids, labels = [], []
    for i, line in enumerate(lines[1:], 2):
        parts = line.split(",")
        if len(parts) != 2:
            errors.append(f"line {i}: expected 2 columns, got {len(parts)}")
            continue
        try:
            ids.append(int(parts[0]))
            lab = int(parts[1])
        except ValueError:
            errors.append(f"line {i}: non-integer value {line!r}")
            continue
        if not 0 <= lab <= 9:
            errors.append(f"line {i}: label {lab} outside 0-9")
        labels.append(lab)

    ref = [int(l.split(",")[0]) for l in
           (ROOT / "data" / "sample_submission.csv").read_text().strip().split("\n")[1:]]
    if len(ids) != len(ref):
        errors.append(f"{len(ids)} rows, expected {len(ref)}")
    elif ids != ref:
        if sorted(ids) == sorted(ref):
            errors.append("ids are complete but NOT in sample_submission.csv order")
        else:
            errors.append(f"ids do not match test ids ({len(set(ids) ^ set(ref))} differ)")
    if len(set(ids)) != len(ids):
        errors.append("duplicate ids")

    if path.name != "submission.csv":
        warns.append(f"file name is {path.name}; rules ask for submission.csv")
    if labels:
        cnt = Counter(labels)
        share = {c: cnt.get(c, 0) / len(labels) for c in range(10)}
        bad = {c: round(s, 3) for c, s in share.items() if s < 0.04 or s > 0.16}
        print("predicted class share:", {c: round(s, 3) for c, s in share.items()})
        if bad:
            warns.append(f"class share outside 4-16% (possible bug or collapse): {bad}")

    if a.compare and labels:
        other = [int(l.split(",")[1]) for l in Path(a.compare).read_text().strip().split("\n")[1:]]
        if len(other) == len(labels):
            agree = sum(x == y for x, y in zip(labels, other)) / len(labels)
            print(f"agreement with {a.compare}: {agree:.4f} ({round((1-agree)*len(labels))} images differ)")

    for w in warns:
        print("WARN:", w)
    for e in errors[:20]:
        print("FAIL:", e)
    print("RESULT:", "FAIL" if errors else "PASS", f"({len(labels)} rows)")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
