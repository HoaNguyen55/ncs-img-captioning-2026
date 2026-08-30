#!/usr/bin/env python
"""Vẽ Hình 4b — ví dụ định tính zero-shot vs hệ đề xuất .

    python research/scripts/plot_qualitative.py \\
        --ids 2595,7240 \\
        --zs ~/ncs-data/results/zeroshot-detailed.preds.json \\
        --ours ~/ncs-data/results/off4090sft-detailed.preds.json \\
        --out research/paper/figures/hinh4b_dinhtinh.png

Trưởng nhóm duyệt phương án A : caption NGUYÊN VĂN của cả hai hệ, không
tuyển chọn hay sửa văn phong — hệ đề xuất đánh đổi trôi chảy lấy trung thực và
hình nói thẳng điều đó. Dòng phán quyết dưới mỗi caption (vật thể không căn
cứ, danh xưng bịa, rào đón) tính bằng ĐÚNG bộ `rescap.chair.objects_in` và
regex giới tính của stress-50 — không có bộ đếm thứ hai. Tiêu chí chọn ảnh in
ngay trong chú thích để phản biện không thể tố cherry-pick.
"""

from __future__ import annotations

import argparse
import json
import sys
import textwrap
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluate import KTVIC, file_names, references
from rescap.chair import objects_in
from stress50_analysis import GENDERED

HEDGES = ("có vẻ", "dường như")
CJK = __import__("re").compile(r"[一-鿿]+")
# DejaVu không có glyph CJK; matplotlib ≥3.6 fallback theo danh sách họ font —
# caption zero-shot rò "引擎" phải hiện đúng mặt chữ, không phải ô vuông.
FONTS = ["DejaVu Sans", "Noto Sans CJK JP"]


def load_preds(path: str) -> dict[str, str]:
    preds = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    return {str(k): (v[0] if isinstance(v, list) else v) or ""
            for k, v in preds.items()}


def verdict(cap: str, gold_obj: set, gold_gen: set) -> tuple[str, bool]:
    """(dòng phán quyết, sạch?) — cùng luật đếm với Bảng 1/2."""
    low = cap.lower()
    halluc = sorted(set(objects_in(low)) - gold_obj)
    fabricated = sorted(set(GENDERED.findall(low)) - gold_gen)
    cjk = CJK.findall(cap)
    hedge = [h for h in HEDGES if h in low]
    if not halluc and not fabricated and not cjk:
        parts = ["✓ 0 vật thể không căn cứ"]
        if hedge:
            parts.append("rào đón: " + ", ".join(f"“{h}”" for h in hedge))
        return " · ".join(parts), True
    parts = []
    if halluc:
        parts.append("✗ vật thể không căn cứ: " + ", ".join(halluc))
    if fabricated:
        parts.append("danh xưng bịa: " + ", ".join(f"“{g}”" for g in fabricated))
    if cjk:
        parts.append("lẫn ký tự CJK: " + ", ".join(cjk))
    if hedge:
        parts.append("rào đón: " + ", ".join(f"“{h}”" for h in hedge))
    return " · ".join(parts), False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ids", default="2595,7240")
    parser.add_argument("--zs", required=True)
    parser.add_argument("--ours", required=True)
    parser.add_argument("--split", default="test_data.json")
    parser.add_argument("--out", default="research/paper/figures/hinh4b_dinhtinh.png")
    args = parser.parse_args()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    ids = [i.strip() for i in args.ids.split(",") if i.strip()]
    zs, ours = load_preds(args.zs), load_preds(args.ours)
    refs = references(args.split)
    names = file_names(args.split)

    fig = plt.figure(figsize=(11.5, 3.9 * len(ids)))
    grid = fig.add_gridspec(len(ids), 2, width_ratios=[1, 1.9],
                            hspace=0.32, wspace=0.05)

    for row, image_id in enumerate(ids):
        gold_obj, gold_gen = set(), set()
        for c in refs[image_id]:
            gold_obj |= set(objects_in(c))
            gold_gen |= set(GENDERED.findall(c.lower()))

        ax_img = fig.add_subplot(grid[row, 0])
        ax_img.imshow(Image.open(KTVIC / "images" / names[image_id]).convert("RGB"))
        ax_img.set_title(f"ảnh KTVIC {image_id}", fontsize=9)
        ax_img.axis("off")

        ax = fig.add_subplot(grid[row, 1])
        ax.axis("off")
        y = 0.98
        for label, cap, colour in (
                ("Zero-shot (chi tiết)", zs[image_id], "#c0392b"),
                ("Hệ đề xuất — SFT chưng cất (chi tiết)", ours[image_id], "#2e933c")):
            line, clean = verdict(cap, gold_obj, gold_gen)
            ax.text(0.0, y, label, fontsize=9.5, fontweight="bold",
                    color=colour, va="top", transform=ax.transAxes)
            y -= 0.075
            body = textwrap.fill(cap.strip(), width=86)
            ax.text(0.0, y, body, fontsize=8, va="top", fontfamily=FONTS,
                    transform=ax.transAxes, wrap=True)
            y -= 0.052 * (body.count("\n") + 1) + 0.02
            ax.text(0.0, y, line, fontsize=8, fontfamily=FONTS,
                    color="#2e933c" if clean else "#c0392b",
                    style="italic", va="top", transform=ax.transAxes)
            y -= 0.105

    fig.suptitle(
        "Đầu ra NGUYÊN VĂN, không tuyển chọn văn phong. Tiêu chí chọn ảnh (chạy máy trên toàn bộ 558 ảnh test): hệ đề xuất\n"
        "tối đa 1 vật thể không căn cứ, không bịa danh xưng, văn mạch lạc; zero-shot bịa rõ. Phán quyết tính bằng đúng bộ CHAIR-vi\n"
        "của Bảng 1/2 — vàng rút từ chú thích tham chiếu nên là CẬN TRÊN: “bầu trời” (ảnh 7240) có trong ảnh nhưng không chú\n"
        "thích nào nhắc (xem Mục 6). Dòng phán quyết áp cho CẢ HAI hệ, kể cả lỗi của hệ đề xuất.",
        fontsize=8, y=0.03, va="bottom", style="italic", color="#444444")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    print(f"  đã ghi {out} và {out.with_suffix('.pdf')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
