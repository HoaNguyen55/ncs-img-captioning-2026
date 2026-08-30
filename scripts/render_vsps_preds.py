#!/usr/bin/env python
"""Dựng caption VSPS (không huấn luyện) từ bản ghi stage1_test — hàng
"VSPS" của Bảng 1/2.

    python research/scripts/render_vsps_preds.py \\
        --records ~/ncs-data/stage1_test \\
        --out-dir ~/ncs-data/results

    # rồi chấm bằng đúng bộ chấm của mọi hàng khác:
    python research/scripts/evaluate.py --predictions \\
        ~/ncs-data/results/vsps-detailed.preds.json \\
        --name vsps-detailed --prompt detailed --also-syllable

Đi qua ĐÚNG đường dựng của dữ liệu huấn luyện (`build_for_image` trong
build_dpo_data.py): chi tiết = bậc A (mọi mệnh đề qua chọn lọc, phần chưa
chắc có rào), ngắn = biến thể short (tối đa 2 mệnh đề SUPPORTED ưu tiên
nhất). Không viết bộ dựng thứ hai — hai bộ dựng là hai nguồn lệch số.

`evaluate.py` từ chối chấm tập con (đúng), nên ảnh nào VSPS không nói được
gì sẽ nhận chuỗi RỖNG và bị chấm như im lặng — đó là hành vi thật của
pipeline, không phải lỗi; số lượng in ra và ghi vào *.stats.json để bài
báo công bố kèm.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_dpo_data import build_for_image
from evaluate import references


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", required=True,
                        help="thư mục bản ghi stage1 của tập test")
    parser.add_argument("--split", default="test_data.json")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--allow-partial", action="store_true",
                        help="dựng dù chưa đủ 558 bản ghi (chỉ để xem trước; "
                             "file thiếu ảnh sẽ bị evaluate.py từ chối)")
    args = parser.parse_args()

    expected = {str(i) for i in references(args.split)}
    records_dir = Path(args.records).expanduser()
    files = sorted(records_dir.glob("*.json"))
    print(f"{len(files)} bản ghi trong {records_dir} · tập test cần {len(expected)} ảnh")

    stats = Counter()
    detailed: dict[str, str] = {}
    short: dict[str, str] = {}
    for f in files:
        record = json.loads(f.read_text(encoding="utf-8"))
        image_id = str(record.get("image_id"))
        if image_id not in expected:
            stats["ngoai_tap_test"] += 1
            continue
        sft, sft_short, _pairs = build_for_image(record, stats)
        detailed[image_id] = (sft or {}).get("response") or ""
        if not detailed[image_id]:
            stats["chi_tiet_rong"] += 1
        # Ngắn: không có mệnh đề SUPPORTED nào lọt top thì lùi về bậc A —
        # VSPS thà nói câu có rào còn hơn im; im hẳn chỉ khi A cũng rỗng.
        short[image_id] = (sft_short or {}).get("response") or detailed[image_id]
        if not sft_short:
            stats["ngan_lui_ve_A" if short[image_id] else "ngan_rong"] += 1

    missing = expected - set(detailed)
    if missing and not args.allow_partial:
        raise SystemExit(
            f"mới có {len(detailed)}/{len(expected)} ảnh test "
            f"(thiếu vd {sorted(missing)[:3]}) — chờ các mảnh VSPS-test chạy "
            f"xong rồi dựng lại, hoặc --allow-partial để xem trước.")

    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, preds in (("vsps-detailed", detailed), ("vsps-short", short)):
        p = out_dir / f"{name}.preds.json"
        p.write_text(json.dumps(preds, ensure_ascii=False, indent=1),
                     encoding="utf-8")
        print(f"  đã ghi {p} ({len(preds)} ảnh)")
    (out_dir / "vsps-preds.stats.json").write_text(
        json.dumps(dict(stats), ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8")

    for k in ("chi_tiet_rong", "ngan_lui_ve_A", "ngan_rong"):
        if stats[k]:
            print(f"  {k}: {stats[k]} ảnh")
    print(f"  còn thiếu: {len(missing)} ảnh" if missing else "  đủ toàn bộ tập test")
    return 0


if __name__ == "__main__":
    sys.exit(main())
