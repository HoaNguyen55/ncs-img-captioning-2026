#!/usr/bin/env python
"""Phân tích bộ 50 ảnh thử thách  trên preds đã có — CPU thuần.

    python scripts/stress50_analysis.py \\
        --manifest ~/ncs-data/datasets/ktvic/stress50_manifest.json \\
        --preds "zero-shot=~/ncs-data/results/zeroshot-short.preds.json" \\
                "VSPS=~/ncs-data/results/vsps-short.preds.json" \\
        --out data/stress50/short.json

Không cần GPU: mọi hệ đã sinh caption đủ 558 ảnh test, 50 ảnh khó là tập
con — chỉ đọc preds từ đĩa và chấm lại. Ba con số mỗi hệ, phân rã theo
từng tín hiệu khó (đếm / màu / giới tính / độ giàu thực thể):

* **CHAIR_i trên tập con** — cận trên ảo giác vật thể, đúng bộ từ điển
  CHAIR-vi của bảng chính (`rescap.chair`).
* **vật thể nhắc/caption** — độ chi tiết, để CHAIR không thắng bằng im lặng.
* **bịa giới tính** — % caption dùng danh từ có giới tính mà KHÔNG chú
  thích tham chiếu nào của ảnh đó dùng (cùng regex với build_stress_manifest).

Nhóm "có tín hiệu X" lấy thẳng từ manifest (điểm từng tín hiệu đã ghi kèm
lúc chọn ảnh), nên bảng phân rã tái tạo được từ hai file đầu vào.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

GENDERED = re.compile(
    r"\b(phụ nữ|đàn ông|cô gái|chàng trai|em bé|cậu bé|cô bé|bé trai|bé gái"
    r"|người bà|người ông|bà cụ|ông cụ|cô|chú|anh|chị)\b")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--preds", nargs="+", required=True,
                        help="tên=đường-dẫn preds.json ({image_id: caption})")
    parser.add_argument("--split", default="test_data.json")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    from evaluate import references
    from rescap.chair import objects_in

    manifest = json.loads(Path(args.manifest).expanduser().read_text(encoding="utf-8"))
    images = manifest["images"]
    ids = [str(r["image_id"]) for r in images]
    signal_of = {str(r["image_id"]): r for r in images}
    refs = references(args.split)

    # vàng-từ-chú-thích của TỪNG ảnh: vật thể và danh từ giới tính mà ít
    # nhất một trong 5 chú thích tham chiếu nhắc tới.
    gold_objects: dict[str, set] = {}
    gold_gender: dict[str, set] = {}
    for i in ids:
        caps = refs[str(i)]
        objs, gens = set(), set()
        for c in caps:
            objs.update(objects_in(c))
            gens.update(GENDERED.findall(c.lower()))
        gold_objects[i] = objs
        gold_gender[i] = gens

    groups = {
        "tất cả 50": ids,
        "đếm": [i for i in ids if signal_of[i]["counting"] > 0],
        "màu": [i for i in ids if signal_of[i]["colour"] > 0],
        "xanh (grue)": [i for i in ids if signal_of[i]["xanh"] > 0],
        "giới tính": [i for i in ids if signal_of[i]["gendered"] > 0],
    }

    result = {"manifest": str(args.manifest), "groups": {g: len(v) for g, v in groups.items()},
              "systems": {}}
    for spec in args.preds:
        name, _, path = spec.partition("=")
        preds = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
        preds = {str(k): (v[0] if isinstance(v, list) else v) for k, v in preds.items()}
        missing = [i for i in ids if i not in preds]
        if missing:
            raise SystemExit(f"{name}: thiếu {len(missing)} ảnh của manifest — "
                             f"preds phải phủ đủ 558 (vd {missing[:3]})")

        per_image = {}
        for i in ids:
            cap = (preds[i] or "").lower()
            objs = objects_in(cap)
            halluc = [o for o in objs if o not in gold_objects[i]]
            gens = set(GENDERED.findall(cap))
            per_image[i] = {
                "mentions": len(objs),
                "halluc": len(halluc),
                "gender_fab": bool(gens - gold_gender[i]),
            }

        def agg(sub):
            rows = [per_image[i] for i in sub]
            m = sum(r["mentions"] for r in rows)
            h = sum(r["halluc"] for r in rows)
            return {
                "n": len(sub),
                "chair_i": round(h / m, 3) if m else None,
                "mentions_per_caption": round(m / len(sub), 2),
                "gender_fab_pct": round(
                    100 * sum(r["gender_fab"] for r in rows) / len(sub), 1),
            }

        result["systems"][name.strip()] = {g: agg(sub) for g, sub in groups.items()}

    out = Path(args.out).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    for name, by_group in result["systems"].items():
        print(f"\n== {name} ==")
        for g, a in by_group.items():
            print(f"  {g:<12} n={a['n']:>2}  CHAIR_i={a['chair_i']}  "
                  f"vật thể/cap={a['mentions_per_caption']}  bịa giới tính={a['gender_fab_pct']}%")
    print(f"\nđã ghi {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
