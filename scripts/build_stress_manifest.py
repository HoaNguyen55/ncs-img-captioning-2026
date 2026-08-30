#!/usr/bin/env python
"""Chọn 50 ảnh test KHÓ cho bộ thử thách .

    python research/scripts/build_stress_manifest.py \\
        --annotations ~/ncs-data/datasets/ktvic/test_data.json \\
        --out ~/ncs-data/datasets/ktvic/stress50_manifest.json

"Khó" phải ĐO ĐƯỢC từ chính chú thích tham chiếu, không phải cảm giác. Bốn
tín hiệu, mỗi cái nhắm đúng một chế độ lỗi của bài:

* **đếm** (`hai/ba/bốn/năm/nhiều/vài/mấy/đông`): mệnh đề đếm là loại probe
  dễ sai nhất.
* **màu, đặc biệt `xanh`**: trục lam/lục là vùng mù đã đo của cả dữ liệu lẫn
  bộ kiểm (grue).
* **người có giới tính trong danh từ** (`phụ nữ/đàn ông/cô gái/chàng
  trai/em bé/cậu bé/cô bé/bà/ông`): chiều bịa giới tính.
* **độ giàu thực thể**: nhiều danh từ khác nhau giữa 5 chú thích = cảnh
  đông đúc, chỗ ảo giác vật thể dễ xảy ra nhất.

Điểm ảnh = tổng bốn tín hiệu đã chuẩn hoá [0..1]. Không có "85%" hứa trước
nào ở đây — bộ này tồn tại để tìm chỗ mô hình GÃY và báo cáo trung thực.

Manifest ghi kèm điểm từng tín hiệu mỗi ảnh, nên bảng kết quả phân rã được
"gãy vì đếm" khác "gãy vì màu".
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
        # `xanh` cộng thêm nửa tín hiệu màu: đúng vùng mù grue đã đo.
        r["score"] = round(
            sum(r[s] / maxima[s] for s in signals)
            + 0.5 * (r["xanh"] / (maxima["colour"] or 1)), 4)

    raw.sort(key=lambda r: (-r["score"], r["image_id"]))
    chosen = raw[: args.n]

    out = Path(args.out).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "purpose": "bộ 50 ảnh thử thách  — chọn theo tín hiệu đo được",
        "signals": {s: f"max={maxima[s]}" for s in signals},
        "images": chosen,
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    n = len(chosen)
    print(f"  {n} ảnh chọn từ {len(raw)} ảnh test")
    for s in signals + ("xanh",):
        cover = sum(1 for r in chosen if r[s] > 0)
        print(f"  có tín hiệu {s:<16}: {cover}/{n} ảnh")
    print(f"  đã ghi {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
