#!/usr/bin/env python
"""Pick the annotation image set, stratified by estimated difficulty.

    python scripts/select_pilot.py --split pilot --n 20
    python scripts/select_pilot.py --split test  --n 500

Writes `<split>_manifest.json` next to the KTVIC data; `annotate.py` reads it.

**Why stratify** (research log ). If all 20 pilot images are easy, the team
agrees on them, calibration looks finished, and the disagreements then surface
on image 200 when it is too late to fix the guideline. The pilot has to contain
hard cases. But not too many, or the first day is demoralising — hence
8 easy / 8 medium / 4 hard.

**The pilot is a subset of the final test set**, so nothing annotated is wasted.

Difficulty is estimated from the image and its reference captions, with no model
involved:

| Signal | Meaning |
|---|---|
| distinct content words across the 5 captions | scene complexity — more things named, more to annotate |
| caption length | annotators wrote more because there was more |
| Laplacian variance | blur; low variance forces UNCERTAIN verdicts |
| mean brightness + spread | dark or blown-out images are hard |
| `xanh` in any caption | the Vietnamese colour ambiguity is present |
| crowd words (`nhóm`, `chợ`, `đông`, `nhiều người`) | the crowded case from the guideline |

This is a **heuristic for sampling**, not a measurement, and it is never
reported as difficulty. Its only job is to make the pilot diverse.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
from pathlib import Path
from typing import Any

DATA = Path(os.environ.get("NCS_DATA", Path.home() / "ncs-data"))
KTVIC = DATA / "datasets" / "ktvic"

CROWD_WORDS = ("nhóm", "chợ", "đông", "nhiều người", "đám", "phố", "giao thông")
STOPWORDS = {
    "có", "một", "hai", "ba", "và", "của", "ở", "trên", "dưới", "trong", "với",
    "là", "đang", "cái", "chiếc", "con", "người", "những", "các", "này", "đó",
    "được", "cùng", "bên", "phía", "ra", "vào", "cho", "từ", "đến", "khi",
}
TOKEN = re.compile(r"[^\W\d_]+", re.UNICODE)


def load_ktvic(split_file: str) -> list[dict[str, Any]]:
    """Return [{image_id, file_name, captions:[...]}] from a KTVIC json."""
    path = KTVIC / split_file
    if not path.exists():
        raise SystemExit(
            f"missing {path}\nRun: bash scripts/datasets/download_ktvic.sh --yes"
        )
    data = json.loads(path.read_text(encoding="utf-8"))

    images: dict[Any, dict[str, Any]] = {}
    if isinstance(data, dict) and "images" in data:
        for img in data["images"]:
            key = img.get("id", img.get("image_id"))
            images[key] = {
                "image_id": str(key),
                "file_name": img.get("file_name") or img.get("filename") or "",
                "captions": [],
                "segment_captions": [],
            }
        for ann in data.get("annotations", data.get("captions", [])):
            key = ann.get("image_id", ann.get("id"))
            if key in images:
                if ann.get("caption"):
                    images[key]["captions"].append(ann["caption"])
                if ann.get("segment_caption"):
                    images[key]["segment_captions"].append(ann["segment_caption"])
    else:  # flat list of records
        records = data if isinstance(data, list) else next(
            (v for v in data.values() if isinstance(v, list)), []
        )
        for rec in records:
            key = rec.get("image_id", rec.get("id"), )
            entry = images.setdefault(
                key,
                {
                    "image_id": str(key),
                    "file_name": rec.get("file_name") or rec.get("filename") or "",
                    "captions": [],
                    "segment_captions": [],
                },
            )
            if rec.get("caption"):
                entry["captions"].append(rec["caption"])
            if rec.get("segment_caption"):
                entry["segment_captions"].append(rec["segment_caption"])

    return [v for v in images.values() if v["file_name"]]


def image_signals(path: Path) -> dict[str, float]:
    """Blur and exposure. Returns empty dict when the file is unreadable."""
    try:
        import numpy as np
        from PIL import Image

        with Image.open(path) as im:
            grey = np.asarray(im.convert("L"), dtype="float32")
        # Laplacian variance: the standard cheap blur proxy.
        kernel_response = (
            grey[:-2, 1:-1] + grey[2:, 1:-1] + grey[1:-1, :-2] + grey[1:-1, 2:]
            - 4 * grey[1:-1, 1:-1]
        )
        return {
            "sharpness": float(kernel_response.var()),
            "brightness": float(grey.mean()),
            "contrast": float(grey.std()),
        }
    except Exception:
        return {}


def score(entry: dict[str, Any], signals: dict[str, float]) -> tuple[float, list[str]]:
    """Higher = harder. Also returns the reasons, so a choice is auditable."""
    captions = entry["captions"]
    joined = " ".join(captions).lower()

    content = {
        t for t in TOKEN.findall(joined) if t not in STOPWORDS and len(t) > 2
    }
    mean_len = (sum(len(c.split()) for c in captions) / len(captions)) if captions else 0

    difficulty = 0.0
    reasons: list[str] = []

    if len(content) >= 22:
        difficulty += 2; reasons.append(f"{len(content)} từ nội dung")
    elif len(content) >= 14:
        difficulty += 1; reasons.append(f"{len(content)} từ nội dung")

    if mean_len >= 14:
        difficulty += 1; reasons.append(f"caption dài ({mean_len:.0f} từ)")

    if any(word in joined for word in CROWD_WORDS):
        difficulty += 2; reasons.append("cảnh đông người")

    if re.search(r"\bxanh\b(?!\s*(dương|lam|lá|lục))", joined):
        difficulty += 1; reasons.append("có `xanh` chưa rõ nghĩa")

    if signals:
        if signals["sharpness"] < 80:
            difficulty += 2; reasons.append("ảnh mờ")
        if signals["brightness"] < 60 or signals["brightness"] > 200:
            difficulty += 1; reasons.append("quá tối/quá sáng")
        if signals["contrast"] < 35:
            difficulty += 1; reasons.append("tương phản thấp")

    return difficulty, reasons


def bucket(value: float) -> str:
    if value <= 1:
        return "dễ"
    if value <= 3:
        return "trung bình"
    return "khó"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="pilot", help="manifest name to write")
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--source", default="test_data.json", help="KTVIC json to draw from")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--quota", default="", help="e.g. 'dễ=8,trung bình=8,khó=4' (default: 40/40/20%%)"
    )
    parser.add_argument(
        "--within", default="", help="only pick from this existing manifest (keeps the pilot inside test)"
    )
    args = parser.parse_args()

    entries = load_ktvic(args.source)
    print(f"KTVIC {args.source}: {len(entries)} images")

    if args.within:
        allowed = {
            img["image_id"]
            for img in json.loads((KTVIC / args.within).read_text(encoding="utf-8"))["images"]
        }
        entries = [e for e in entries if e["image_id"] in allowed]
        print(f"  restricted to {args.within}: {len(entries)} images")

    images_dir = KTVIC / "images"
    have_images = images_dir.exists() and any(images_dir.iterdir())
    if not have_images:
        print("  ⚠ no images yet — stratifying by captions only, without sharpness/brightness")

    scored = []
    for entry in entries:
        signals = image_signals(images_dir / entry["file_name"]) if have_images else {}
        value, reasons = score(entry, signals)
        scored.append({**entry, "score": value, "difficulty": bucket(value), "why": reasons})

    groups: dict[str, list[dict]] = {"dễ": [], "trung bình": [], "khó": []}
    for item in scored:
        groups[item["difficulty"]].append(item)
    print("  distribution:", {k: len(v) for k, v in groups.items()})

    if args.quota:
        quota = {
            k.strip(): int(v)
            for k, v in (part.split("=") for part in args.quota.split(","))
        }
    else:
        quota = {
            "dễ": round(args.n * 0.4),
            "trung bình": round(args.n * 0.4),
            "khó": args.n - round(args.n * 0.4) * 2,
        }
    print("  quota:", quota)

    rng = random.Random(args.seed)
    picked: list[dict] = []
    shortfall: dict[str, int] = {}
    for name, want in quota.items():
        pool = groups[name][:]
        rng.shuffle(pool)
        take = pool[:want]
        picked.extend(take)
        if len(take) < want:
            shortfall[name] = want - len(take)

    # Backfill honestly: say what was substituted rather than silently returning
    # a set that does not match the requested stratification.
    if shortfall:
        print(f"  ⚠ shortfall at {shortfall} — backfilling with the closest images by difficulty")
        chosen = {i["image_id"] for i in picked}
        rest = sorted(
            (i for i in scored if i["image_id"] not in chosen),
            key=lambda i: i["score"],
            reverse=True,
        )
        picked.extend(rest[: sum(shortfall.values())])

    rng.shuffle(picked)  # so annotators do not meet all the easy ones first

    out = KTVIC / f"{args.split}_manifest.json"
    out.write_text(
        json.dumps(
            {
                "split": args.split,
                "n": len(picked),
                "source": args.source,
                "seed": args.seed,
                "quota": quota,
                "shortfall": shortfall,
                "note": (
                    "difficulty is a sampling heuristic (caption content words, "
                    "length, blur, exposure, `xanh`, crowd words). It is NOT a "
                    "measurement and must not be reported as one."
                ),
                "images": [
                    {
                        "image_id": i["image_id"],
                        "file_name": i["file_name"],
                        "difficulty": i["difficulty"],
                        "why": i["why"],
                        "captions": i["captions"],
                    }
                    for i in picked
                ],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"\nWrote {out}  ({len(picked)} images)")
    for item in picked[:8]:
        why = ", ".join(item["why"]) or "—"
        print(f"  [{item['difficulty']:<10}] {item['file_name']:<28} {why}")
    if len(picked) > 8:
        print(f"  … and {len(picked) - 8} more images")
    print(f"\nNext: python scripts/annotate.py --annotator <name> --split {args.split}")


if __name__ == "__main__":
    main()
