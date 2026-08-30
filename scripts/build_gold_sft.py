#!/usr/bin/env python
"""Dựng dữ liệu SFT từ CHÚ THÍCH VÀNG của KTVIC — đối chứng P1 .

    python research/scripts/build_gold_sft.py \\
        --annotations ~/ncs-data/datasets/ktvic/train_data.json \\
        --out ~/ncs-data/stage2_gold

Vì sao tồn tại: câu hỏi tự nhiên nhất của phản biện là *"có 3.769 ảnh kèm chú
thích người viết — sao không SFT thẳng trên đó?"*. Bài phải có con số trả lời.
Dự đoán trung thực được ghi trước ở SFT-vàng có thể THẮNG chưng cất ở
CIDEr chế độ ngắn (nó học đúng văn phong tham chiếu); cái nó không học được
là chi tiết + phòng hộ, và Bảng 2 đo đúng chỗ đó.

Để so sánh công bằng với nhánh chưng cất (cùng số ảnh, cùng epoch, cùng
prompt):

* MỘT chú thích mỗi ảnh (chú thích đầu tiên theo thứ tự file — KTVIC không
  đánh dấu chú thích "chính"), không phải cả ~5 — nhánh chưng cất cũng chỉ có
  một câu trả lời mỗi ảnh cho mỗi phong cách.
* Prompt là ĐÚNG prompt ngắn của đánh giá (`PROMPT_SHORT` của
  build_dpo_data.py) — mô hình được hỏi lúc chấm y như lúc học.
* KHÔNG có biến thể chi tiết: chú thích vàng là một câu; bịa thêm phong cách
  chi tiết từ nó là sáng tác dữ liệu. Ở chế độ chi tiết, mô hình này trả lời
  bằng những gì nó còn giữ từ pretraining — và đó chính là phép đo.

`dpo.jsonl` không được tạo: chú thích vàng không có phán quyết nên không tồn
tại cặp ưu tiên — chạy `train_stage2.py --stage sft` là đủ.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_dpo_data import PROMPT_SHORT  # cùng một prompt, không chép tay lại


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", required=True,
                        help="train_data.json của KTVIC")
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=0,
                        help="giới hạn số ảnh (0 = tất cả) — chỉ để thử")
    args = parser.parse_args()

    data = json.loads(Path(args.annotations).expanduser().read_text(encoding="utf-8"))
    file_names = {
        str(img.get("id", img.get("image_id"))): img.get("file_name") or img.get("filename")
        for img in data.get("images", [])
    }

    first_caption: dict[str, str] = {}
    n_captions = 0
    for ann in data.get("annotations", []):
        image_id = str(ann["image_id"])
        caption = (ann.get("caption") or "").strip()
        if not caption:
            continue
        n_captions += 1
        first_caption.setdefault(image_id, caption)

    rows = [
        {"image_id": image_id, "file_name": file_names.get(image_id),
         "prompt": PROMPT_SHORT, "response": caption, "variant": "gold"}
        for image_id, caption in sorted(first_caption.items())
        if file_names.get(image_id)
    ]
    dropped = len(first_caption) - len(rows)
    if args.limit:
        rows = rows[: args.limit]

    rng = random.Random(args.seed)
    rng.shuffle(rows)

    dst = Path(args.out).expanduser()
    dst.mkdir(parents=True, exist_ok=True)
    with (dst / "sft.jsonl").open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    report = {
        "source": str(args.annotations), "seed": args.seed,
        "baseline": "gold_sft (P1, )",
        "images_with_caption": len(first_caption),
        "captions_total": n_captions,
        "captions_used": len(rows),
        "policy": "one first-caption per image, PROMPT_SHORT, no detailed variant",
        "dropped_no_file_name": dropped,
    }
    (dst / "_build_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"  {len(rows)} ví dụ SFT-vàng (từ {n_captions} chú thích / "
          f"{len(first_caption)} ảnh, lấy 1 câu đầu mỗi ảnh)")
    if dropped:
        print(f"  ⚠ {dropped} ảnh bị bỏ vì không tra được file_name")
    print(f"  -> {dst/'sft.jsonl'}\n"
          f"  chạy: train_stage2.py --stage sft --data {dst} --epochs 2")
    return 0


if __name__ == "__main__":
    sys.exit(main())
