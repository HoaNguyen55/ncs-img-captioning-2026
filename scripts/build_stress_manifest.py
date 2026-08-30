#!/usr/bin/env python
"""Select 50 HARD test images for the challenge set.

    python scripts/build_stress_manifest.py \\
        --annotations ~/ncs-data/datasets/ktvic/test_data.json \\
        --out ~/ncs-data/datasets/ktvic/stress50_manifest.json

"Hard" must be MEASURABLE from the reference captions themselves, not a
feeling. Four signals, each aimed at one failure mode of the paper:

* **counting** (`hai/ba/bốn/năm/nhiều/vài/mấy/đông`): counting propositions
  are the probe type that fails most often.
* **colour, especially `xanh`**: the blue/green axis is a measured blind spot
  of both the data and the checker (grue).
* **gender-marked person nouns** (`phụ nữ/đàn ông/cô gái/chàng
  trai/em bé/cậu bé/cô bé/bà/ông`): the gender-fabrication axis.
* **entity richness**: many distinct nouns across the 5 captions = a crowded
  scene, where object hallucination is most likely.

Image score = sum of the four signals normalised to [0..1]. There is no
"85%" promised up front here — this set exists to find where the model BREAKS
and to report that honestly.

The manifest stores each image's per-signal scores, so the results table can
separate "broke on counting" from "broke on colour".
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

COUNT_WORDS = re.compile(
    r"\b(hai|ba|bốn|năm|sáu|bảy|nhiều|vài|mấy|đông|một số|một vài)\b")
COLOUR_WORDS = re.compile(
    r"\b(xanh|đỏ|vàng|trắng|đen|nâu|hồng|tím|cam|xám)\b")
GENDERED = re.compile(
    r"\b(phụ nữ|đàn ông|cô gái|chàng trai|em bé|cậu bé|cô bé|bé trai|bé gái"
    r"|người bà|người ông|bà cụ|ông cụ|cô|chú|anh|chị)\b")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--n", type=int, default=50)
    args = parser.parse_args()

    from rescap.chair import objects_in

    data = json.loads(Path(args.annotations).expanduser().read_text(encoding="utf-8"))
    file_names = {
        str(img.get("id", img.get("image_id"))): img.get("file_name") or img.get("filename")
        for img in data.get("images", [])
    }
    captions: dict[str, list[str]] = {}
    for ann in data.get("annotations", []):
        if ann.get("caption"):
            captions.setdefault(str(ann["image_id"]), []).append(ann["caption"])

    raw = []
    for image_id, caps in captions.items():
        text = " . ".join(c.lower() for c in caps)
        nouns = set()
        for c in caps:
            nouns.update(objects_in(c))
        raw.append({
            "image_id": image_id, "file_name": file_names.get(image_id),
            "counting": len(COUNT_WORDS.findall(text)),
            "colour": len(COLOUR_WORDS.findall(text)),
            "xanh": len(re.findall(r"\bxanh\b", text)),
            "gendered": len(GENDERED.findall(text)),
            "entity_richness": len(nouns),
        })

    signals = ("counting", "colour", "gendered", "entity_richness")
    maxima = {s: max((r[s] for r in raw), default=1) or 1 for s in signals}
    for r in raw:
        # `xanh` adds half an extra colour signal: exactly the measured grue blind spot.
        r["score"] = round(
            sum(r[s] / maxima[s] for s in signals)
            + 0.5 * (r["xanh"] / (maxima["colour"] or 1)), 4)

    raw.sort(key=lambda r: (-r["score"], r["image_id"]))
    chosen = raw[: args.n]

    out = Path(args.out).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "purpose": "50-image challenge set — selected by measurable difficulty signals",
        "signals": {s: f"max={maxima[s]}" for s in signals},
        "images": chosen,
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    n = len(chosen)
    print(f"  {n} images chosen out of {len(raw)} test images")
    for s in signals + ("xanh",):
        cover = sum(1 for r in chosen if r[s] > 0)
        print(f"  has signal {s:<16}: {cover}/{n} images")
    print(f"  wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
