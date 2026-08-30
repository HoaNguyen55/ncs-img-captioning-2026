#!/usr/bin/env python
"""So sánh ngữ liệu VI–EN cho hai hiện tượng bài tuyên bố (Q-corpus, 22/08).

    python scripts/compare_langs.py \\
        --coco ~/ncs-data/datasets/karpathy/dataset_coco.json \\
        --out data/results/lang_compare.json

Hai tuyên bố của bài cần chỗ dựa so sánh thay vì khẳng định đơn phương:
(1) `xanh` tiếng Việt mơ hồ lam/lục — tiếng Anh tách blue/green từ gốc từ vựng;
(2) áp lực danh xưng giới tính: khi nhắc người, chú thích viên hai ngôn ngữ
    chọn danh từ mang giới tính với tần suất bao nhiêu.

Phía VI dùng ĐÚNG bộ mã pipeline (`rescap.vi.color.parse_color`,
`rescap.vi.lexicon.GENDERED_NOUNS`) trên toàn bộ 21.635 chú thích KTVIC
(train + test) — không có bộ đếm thứ hai. Phía EN dùng danh sách từ khai báo
ngay trong file này, in vào artifact để tái tạo được.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rescap.vi.color import Xanh, parse_color  # noqa: E402
from rescap.vi.lexicon import GENDERED_NOUNS  # noqa: E402
from count_colour_ambiguity import XANH_PHRASE, bucket  # noqa: E402  (cùng thư mục scripts/)

DATA = Path(os.environ.get("NCS_DATA", Path.home() / "ncs-data"))
KTVIC = DATA / "datasets" / "ktvic"

EN_GENDERED = (
    "man", "men", "woman", "women", "boy", "boys", "girl", "girls",
    "lady", "ladies", "guy", "guys", "gentleman", "gentlemen", "male", "female",
)
EN_NEUTRAL = (
    "person", "people", "someone", "somebody", "individual", "human",
    "pedestrian", "child", "children", "kid", "kids", "adult", "adults",
)
_EN_G = re.compile(r"\b(" + "|".join(EN_GENDERED) + r")\b")
_EN_N = re.compile(r"\b(" + "|".join(EN_NEUTRAL) + r")\b")
_XANH = re.compile(r"xanh")


def ktvic_captions() -> list[str]:
    caps = []
    for name in ("train_data.json", "test_data.json"):
        d = json.loads((KTVIC / name).read_text(encoding="utf-8"))
        caps += [a["caption"].lower() for a in d["annotations"]]
    return caps


def vi_stats(caps: list[str]) -> dict:
    gendered_terms = sorted(GENDERED_NOUNS, key=len, reverse=True)
    total = xanh_mentions = xanh_unresolved = 0
    person = gendered = 0
    for c in caps:
        total += 1
        has_gender = any(g in c for g in gendered_terms)
        if has_gender or "người" in c or "em bé" in c or "trẻ em" in c or "đứa bé" in c:
            person += 1
            gendered += bool(has_gender)
        # đúng phương pháp count_colour_ambiguity.py (số 55,4% của bài): cụm `xanh` đọc
        # trọn, nhóm "chưa có trong từ điển" bị LOẠI khỏi mẫu số
        for m in XANH_PHRASE.finditer(c):
            b = bucket(m.group(0))
            if b == "chưa có trong từ điển":
                continue
            xanh_mentions += 1
            if b in ("trơ", "chỉ có mức độ"):
                xanh_unresolved += 1
    return {
        "captions": total,
        "person_mentions": person,
        "gendered_person": gendered,
        "gendered_share": round(gendered / person, 4),
        "xanh_mentions": xanh_mentions,
        "xanh_unresolved": xanh_unresolved,
        "xanh_unresolved_share": round(xanh_unresolved / xanh_mentions, 4),
    }


def en_stats(coco_path: Path) -> dict:
    d = json.loads(coco_path.expanduser().read_text(encoding="utf-8"))
    total = person = gendered = blue = green = 0
    for img in d["images"]:
        for s in img["sentences"]:
            c = s["raw"].lower()
            total += 1
            g = _EN_G.search(c) is not None
            if g or _EN_N.search(c):
                person += 1
                gendered += g
            blue += len(re.findall(r"\bblue\b", c))
            green += len(re.findall(r"\bgreen\b", c))
    return {
        "captions": total,
        "person_mentions": person,
        "gendered_person": gendered,
        "gendered_share": round(gendered / person, 4),
        "blue_mentions": blue,
        "green_mentions": green,
        # blue/green là hai từ vị rời nhau: không tồn tại dạng trần dùng chung
        "blue_green_merged_share": 0.0,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--coco", default="~/ncs-data/datasets/karpathy/dataset_coco.json")
    ap.add_argument("--out", default="data/results/lang_compare.json")
    args = ap.parse_args()

    vi = vi_stats(ktvic_captions())
    en = en_stats(Path(args.coco))
    result = {
        "vi_ktvic": vi,
        "en_coco_karpathy": en,
        "en_lexicon": {"gendered": EN_GENDERED, "neutral": EN_NEUTRAL},
        "note": (
            "VI dùng rescap.vi.color.parse_color + GENDERED_NOUNS của pipeline; "
            "EN dùng danh sách khai báo trên. So sánh mô tả ngữ liệu, không phải "
            "phán xét chú thích viên."
        ),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"VI KTVIC : {vi['captions']:,} caption · giới tính khi nhắc người "
          f"{100*vi['gendered_share']:.1f}% · xanh chưa phân định "
          f"{100*vi['xanh_unresolved_share']:.1f}% ({vi['xanh_unresolved']}/{vi['xanh_mentions']})")
    print(f"EN COCO  : {en['captions']:,} caption · giới tính khi nhắc người "
          f"{100*en['gendered_share']:.1f}% · blue/green tách từ vựng, mơ hồ 0%")
    print(f"đã ghi {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
