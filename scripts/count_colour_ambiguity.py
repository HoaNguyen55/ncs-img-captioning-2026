#!/usr/bin/env python
"""Count how often `xanh` is left ambiguous in real Vietnamese captions.

    python scripts/count_colour_ambiguity.py
    python scripts/count_colour_ambiguity.py --source train_data.json --examples 20

research log  branch C. The paper claims Vietnamese `xanh` spans both blue and
green and so does not map onto `blue`/`green`. Right now that claim rests on
assertion. This turns it into a measurement taken from the corpus we actually
evaluate on.

**The number that matters** is the share of `xanh` mentions that stay
unresolved. High share -> colour ambiguity is a real property of the data and
the paper can lean on it. Low share -> annotators disambiguate by habit, the
claim is overstated, and we soften it. Either result is publishable; inventing
the first one is not.

Four buckets, not two:

| Bucket | Example | Hue known? |
|---|---|---|
| resolved | `xanh dương`, `xanh lá`, `xanh rêu` | yes |
| intensity only | `xanh đậm`, `xanh nhạt` | **no** |
| bare | `xanh` | no |
| not in lexicon | `xanh mướt`, `xanh biếc` | unjudged, excluded |

The intensity bucket is the point. `xanh đậm` looks specific and is not: it fixes
brightness while leaving blue-versus-green open. A pipeline that treats any
modifier as disambiguation would score it correct either way. Parsing comes from
`rescap.vi.color`, the same code the pipeline uses, so this measures the system's
own behaviour rather than a second opinion about it.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rescap.vi.color import Xanh, parse_color  # noqa: E402

DATA = Path(os.environ.get("NCS_DATA", Path.home() / "ncs-data"))
KTVIC = DATA / "datasets" / "ktvic"

# `xanh` plus everything attached to it, so `xanh nước biển` is read whole
# rather than as `xanh` followed by noise.
XANH_PHRASE = re.compile(
    r"\bxanh(?:\s+(?:dương|lam|da\s+trời|nước\s+biển|biển|lá(?:\s+cây)?|lục|"
    r"rêu|ngọc|non|đen|đậm|nhạt|thẫm|sáng|tối|lơ|xao|um|mướt|biếc))*",
    re.IGNORECASE,
)


def captions_from(path: Path, field: str = "caption") -> list[str]:
    """Pull caption strings out of a KTVIC-shaped json.

    KTVIC ships **two versions of every caption**: `caption` (raw) and
    `segment_caption` (word-segmented, compounds joined by `_`). Harvesting both
    counts every sentence twice and mixes two tokenizations in one tally, so the
    field is chosen explicitly. Raw is the default -- `xanh dương` is one colour
    phrase either way, but the segmented text writes compounds as `đóng_cửa`,
    and the regex should not have to reason about that.
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    out: list[str] = []

    def harvest(records):
        for rec in records:
            if isinstance(rec, dict) and rec.get(field):
                out.append(rec[field])

    if isinstance(data, dict):
        harvest(data.get("annotations", []))
        harvest(data.get("captions", []))
        if not out:
            for value in data.values():
                if isinstance(value, list):
                    harvest(value)
    elif isinstance(data, list):
        harvest(data)
    return out


BUCKETS = ("trơ", "chỉ có mức độ", "đã rõ", "chưa có trong từ điển")


def bucket(phrase: str) -> str:
    reading = parse_color(phrase)
    if reading.xanh_value in (Xanh.BLUE, Xanh.GREEN):
        return "đã rõ"
    if reading.xanh_value is Xanh.UNRESOLVED:
        # Unresolved hue. Split on whether a modifier created a false sense of
        # detail: `xanh đậm` fixes brightness and leaves blue-vs-green open.
        return "chỉ có mức độ" if reading.modifier else "trơ"
    # The lexicon does not know this phrase -- `xanh mướt`, `xanh biếc`,
    # `xanh xao`. Counting it as bare would be wrong (`xanh mướt` is green in
    # practice) and counting it as resolved would be wrong too. Surface it for
    # a human instead: an unknown phrase is a gap in the lexicon, and silently
    # bucketing it would hide exactly the thing worth finding.
    return "chưa có trong từ điển"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="train_data.json")
    parser.add_argument("--also", nargs="*", default=["test_data.json"])
    parser.add_argument("--examples", type=int, default=8)
    parser.add_argument(
        "--field", default="caption", choices=["caption", "segment_caption"],
        help="KTVIC ships both; counting both double-counts every sentence",
    )
    args = parser.parse_args()

    files = [KTVIC / args.source] + [KTVIC / name for name in args.also]
    files = [f for f in files if f.exists()]
    if not files:
        raise SystemExit(
            f"không thấy file nào trong {KTVIC}\n"
            "Chạy: bash scripts/datasets/download_ktvic.sh --yes"
        )

    captions: list[str] = []
    for path in files:
        found = captions_from(path, args.field)
        print(f"{path.name}: {len(found)} caption")
        captions.extend(found)

    counts: Counter[str] = Counter()
    phrases: Counter[str] = Counter()
    samples: dict[str, list[str]] = {name: [] for name in BUCKETS}
    captions_with_xanh = 0

    for caption in captions:
        hits = [m.group(0) for m in XANH_PHRASE.finditer(caption)]
        if hits:
            captions_with_xanh += 1
        for phrase in hits:
            normalised = " ".join(phrase.lower().split())
            name = bucket(normalised)
            counts[name] += 1
            phrases[normalised] += 1
            if len(samples[name]) < args.examples:
                samples[name].append(caption.strip())

    total = sum(counts.values())
    if not total:
        raise SystemExit("không tìm thấy `xanh` nào — kiểm tra lại đường dẫn dữ liệu")

    print(f"\n{'='*66}")
    print(f"Tổng caption: {len(captions)}   Có chứa `xanh`: {captions_with_xanh} "
          f"({captions_with_xanh/len(captions)*100:.1f}%)")
    print(f"Tổng lượt nhắc `xanh`: {total}")
    print("=" * 66)

    for name in BUCKETS:
        n = counts[name]
        bar = "█" * round(n / total * 40)
        print(f"  {name:<16} {n:>6}  {n/total*100:>5.1f}%  {bar}")

    # The headline ratio is computed over phrases the lexicon can actually
    # judge. Unknown phrases are neither evidence for nor against the claim,
    # so folding them in either direction would bias the number we publish.
    unresolved = counts["trơ"] + counts["chỉ có mức độ"]
    judged = unresolved + counts["đã rõ"]
    print(f"\n  >>> CHƯA XÁC ĐỊNH ĐƯỢC MÀU: {unresolved}/{judged} "
          f"= {unresolved/judged*100:.1f}%  <<<")
    print("      (mẫu số chỉ gồm cụm từ điển đọc được — con số này đi vào bài báo)")
    if counts["chưa có trong từ điển"]:
        print(f"      ⚠ {counts['chưa có trong từ điển']} cụm chưa có trong từ "
              f"điển, đã loại khỏi mẫu số — cần người rà rồi bổ sung vào "
              f"rescap/vi/color.py")

    print("\nCác cụm hay gặp nhất:")
    for phrase, n in phrases.most_common(12):
        print(f"  {n:>5}×  {phrase}   [{bucket(phrase)}]")

    for name in ("trơ", "chỉ có mức độ", "chưa có trong từ điển"):
        if samples[name]:
            print(f"\nVí dụ — {name}:")
            for caption in samples[name]:
                print(f"  · {caption}")

    out = KTVIC / "xanh_counts.json"
    out.write_text(
        json.dumps(
            {
                "files": [f.name for f in files],
                "field": args.field,
                "captions_total": len(captions),
                "captions_with_xanh": captions_with_xanh,
                "mentions_total": total,
                "buckets": dict(counts),
                "unresolved_share": unresolved / judged,
                "judged_denominator": judged,
                "phrases": dict(phrases.most_common(50)),
                "note": (
                    "`chỉ có mức độ` (xanh đậm/nhạt) counts as UNRESOLVED: a "
                    "brightness modifier does not settle blue vs green."
                ),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nĐã ghi {out}")


if __name__ == "__main__":
    main()
