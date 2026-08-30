#!/usr/bin/env python
"""Vẽ hình ví dụ định tính (P3, ): ảnh → mệnh đề → phán quyết → A/B/C.

    python research/scripts/plot_teaser.py \\
        --record research/backups/stage1/00000000833.json \\
        --image ~/ncs-data/datasets/ktvic/images/00000000833.jpg \\
        --ids P3,P4,P6,P15,P25 \\
        --out research/paper/figures/hinh_vidu_dinhtinh.png

Bài phương pháp mà không có một ví dụ cụ thể nào là điểm trừ đọc-hiểu lớn
. Hình này thay nửa trang văn xuôi của mục 3.4: người đọc thấy một ảnh
thật, năm mệnh đề với phán quyết ba trạng thái, và ba biến thể A/B/C dựng từ
đúng bản ghi đó — không có gì được viết tay.

Ba biến thể lấy từ CHÍNH `build_dpo_data.build_for_image` — cùng đường mã với
dữ liệu huấn luyện, nên hình không thể lệch khỏi những gì mô hình thực học.

Phán quyết được mã hoá KÉP (màu + ký hiệu ✓/?/✗) — bài in trắng đen vẫn đọc
được, bài học từ Hình 1.
"""

from __future__ import annotations

import argparse
import json
import sys
import textwrap
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

STYLE = {
    "SUPPORTED": dict(colour="#2e933c", symbol="✓", label="SUPPORTED"),
    "UNCERTAIN": dict(colour="#d9a404", symbol="?", label="UNCERTAIN"),
    "REJECTED": dict(colour="#c0392b", symbol="✗", label="REJECTED"),
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--ids", required=True,
                        help="mệnh đề đưa lên hình, phẩy ngăn cách (chọn tay "
                             "cho dễ đọc — hình ghi rõ là trích)")
    parser.add_argument("--out", default="research/paper/figures/hinh_vidu_dinhtinh.png")
    args = parser.parse_args()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.image as mpimg
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch

    from build_dpo_data import build_for_image, verdict_of

    record = json.loads(Path(args.record).expanduser().read_text(encoding="utf-8"))
    props = {str(p.get("id")): p for p in record.get("propositions") or []}
    ids = [i.strip() for i in args.ids.split(",") if i.strip()]
    missing = [i for i in ids if i not in props]
    if missing:
        raise SystemExit(f"không thấy mệnh đề {missing} trong bản ghi")

    stats: Counter[str] = Counter()
    sft, _, pairs = build_for_image(record, stats)
    if not sft:
        raise SystemExit("bản ghi này không dựng được bậc A — chọn bản ghi khác")
    variant_a = sft["response"]
    variant_b = next((p["rejected"] for p in pairs
                      if p["pair_type"].startswith("A>B")), None)
    variant_c = next((p["rejected"] for p in pairs
                      if "rejected_included" in p["pair_type"]), None)
    if not variant_c:
        variant_c = next((p["rejected"] for p in pairs
                          if "uncertain_asserted" in p["pair_type"]), None)
    n_total = len(record.get("propositions") or [])

    fig = plt.figure(figsize=(7.0, 6.2))
    # hàng trên: ảnh trái, mệnh đề phải; hàng dưới: ba biến thể
    ax_img = fig.add_axes([0.015, 0.565, 0.44, 0.40])
    ax_img.imshow(mpimg.imread(Path(args.image).expanduser()))
    ax_img.set_xticks([]); ax_img.set_yticks([])
    for side in ax_img.spines.values():
        side.set_visible(False)
    ax_img.set_title("ảnh KTVIC", fontsize=8, pad=3)

    ax_p = fig.add_axes([0.47, 0.565, 0.52, 0.40])
    ax_p.set_xlim(0, 1); ax_p.set_ylim(0, 1); ax_p.axis("off")
    ax_p.set_title(f"trích {len(ids)} trong {n_total} mệnh đề đã kiểm",
                   fontsize=8, pad=3)
    y = 0.94
    for pid in ids:
        prop = props[pid]
        verdict = verdict_of(prop) or "UNCERTAIN"
        st = STYLE[verdict]
        ax_p.add_patch(FancyBboxPatch(
            (0.0, y - 0.055), 0.24, 0.095,
            boxstyle="round,pad=0.008", linewidth=0.9,
            edgecolor=st["colour"], facecolor=st["colour"], alpha=0.9))
        ax_p.text(0.12, y - 0.007, f"{st['symbol']} {st['label']}",
                  ha="center", va="center", fontsize=5.6, color="white",
                  fontweight="bold")
        ax_p.text(0.27, y - 0.007, str(prop.get("text_vi", "")),
                  ha="left", va="center", fontsize=7.6)
        note = ""
        if verdict == "UNCERTAIN" and "màu" not in str(prop.get("text_vi")) \
                and any(w in str(prop.get("text_vi", "")).lower()
                        for w in ("xanh", "đỏ", "vàng", "nâu", "trắng", "đen")):
            note = "cả hai mô hình cùng dò màu, trả lời không nhất quán → hạ bậc"
        if note:
            ax_p.text(0.27, y - 0.055, note, ha="left", va="center",
                      fontsize=6.2, style="italic", color="#666666")
        y -= 0.185

    boxes = [
        ("A — rào đón  (biến thể khảo sát, Mục 5)", variant_a, "#7a7a7a", "-"),
        ("B — loại bỏ không chắc  (GIÁM SÁT SFT — hệ chính)", variant_b, "#2e933c", "-"),
        ("C — khẳng định không kiểm chứng  (đối chứng âm)", variant_c, "#c0392b", "--"),
    ]
    y0 = 0.545
    for title, text, colour, ls in boxes:
        ax = fig.add_axes([0.015, y0 - 0.165, 0.97, 0.155])
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
        ax.add_patch(FancyBboxPatch(
            (0.002, 0.03), 0.994, 0.94, boxstyle="round,pad=0.004",
            linewidth=1.3, edgecolor=colour, facecolor="white", linestyle=ls))
        ax.text(0.012, 0.90, title, fontsize=7.2, fontweight="bold",
                color=colour, va="top")
        wrapped = textwrap.fill(text or "(không dựng được)", width=118)
        ax.text(0.012, 0.62, wrapped, fontsize=6.6, va="top", family="sans-serif")
        y0 -= 0.175

    fig.text(0.015, 0.008,
             "Ba biến thể dựng bằng luật từ phán quyết — bậc B (loại bỏ không chắc, "
             "ngân sách 9) là giám sát SFT của hệ chính; A/C dùng cho "
             "khảo sát Mục 5.",
             fontsize=6.6, style="italic", color="#444444")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300)
    fig.savefig(out.with_suffix(".pdf"))
    print(f"  đã ghi {out} và {out.with_suffix('.pdf')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
