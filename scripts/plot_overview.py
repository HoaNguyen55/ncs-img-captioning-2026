#!/usr/bin/env python
"""Hình 1a — TỔNG QUÁT pipeline chia theo PHA (yêu cầu nhóm 24/08 tối).

    python scripts/plot_overview.py --out research/paper/figures/hinh1a_tongquat.png

Tập dữ liệu → PHA 1 (tiền xử lý & huấn luyện) ⇒ kết quả pha → PHA 2 (suy luận).
Chi tiết module nằm ở Hình 1b (hinh1_quytrinh).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="research/paper/figures/hinh1a_tongquat.png")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    fig, ax = plt.subplots(figsize=(11.5, 4.6))
    ax.set_xlim(0, 100); ax.set_ylim(0, 42); ax.axis("off")

    C_DL = "#f2f2f2"; C_DLE = "#666666"          # dữ liệu
    C_P1 = "#eaf2f6"; C_P1E = "#1f6f8b"          # pha 1
    C_KQ = "#eef7ee"; C_KQE = "#2e933c"          # kết quả pha 1
    C_P2 = "#fdf1e7"; C_P2E = "#c96f2f"          # pha 2

    def box(x, y, w, h, text, fc, ec, fs=8.6, bold=False, lw=1.6):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.35",
                                    fc=fc, ec=ec, lw=lw))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=fs, fontweight="bold" if bold else "normal",
                color="#1a1a1a")

    def arrow(x1, y1, x2, y2, color="#555", lw=1.6, double=False):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2),
                                     arrowstyle="-|>", mutation_scale=14,
                                     color=color, lw=lw))
        if double:
            ax.add_patch(FancyArrowPatch((x1, y1 - 1.2), (x2, y2 - 1.2),
                                         arrowstyle="-|>", mutation_scale=14,
                                         color=color, lw=lw))

    # ---- TẬP DỮ LIỆU (trái) ----
    box(1, 22, 15, 14,
        "ĐẦU VÀO PHA 1:\nTẬP DỮ LIỆU\nKTVIC train 3.700 ảnh\n(không dùng\nchú thích vàng)",
        C_DL, C_DLE, fs=7.8, bold=True)

    # ---- PHA 1 ----
    ax.text(19, 39.5, "PHA 1 — TIỀN XỬ LÝ & HUẤN LUYỆN (chạy MỘT lần)",
            fontsize=10, fontweight="bold", color=C_P1E)
    box(19, 22, 17, 14,
        "SINH & KIỂM CHỨNG\nmệnh đề có cấu trúc\nba trạng thái + lùi bậc\ntrung tính (M1–M3b)",
        C_P1, C_P1E, fs=8)
    box(38, 22, 17, 14,
        "CHỌN & KẾT XUẤT GIÁM SÁT\nngân sách 9 · im lặng\nLM-có-ràng-buộc + khuôn\nkho-đóng (M6–M8)",
        C_P1, C_P1E, fs=8)
    box(57, 22, 14, 14, "HUẤN LUYỆN\nSFT QLoRA 4-bit\n2 phong cách\n1×RTX 4090",
        C_P1, C_P1E, fs=8)
    arrow(16, 29, 19, 29); arrow(36, 29, 38, 29); arrow(55, 29, 57, 29)

    # ---- KẾT QUẢ PHA 1 ----
    box(74, 22, 25, 14,
        "KẾT QUẢ PHA 1\n• Kho giám sát đã kiểm chứng\n  (7.177 ví dụ, có truy vết)\n• MÔ HÌNH CHƯNG CẤT 7B\n  (adapter 170MB)",
        C_KQ, C_KQE, fs=8, bold=True, lw=2.2)
    arrow(71, 29, 74, 29, color=C_KQE)

    # ---- PHA 2 ----
    # vách ngăn tách cứng hai pha
    ax.plot([0, 100], [18.2, 18.2], color="#999999", lw=1.2, ls=(0, (6, 3)))
    ax.text(99, 19.0, "ranh giới hai pha — chỉ MÔ HÌNH đi qua", fontsize=7.5,
            ha="right", style="italic", color="#777777")
    ax.text(19, 16.5, "PHA 2 — SUY LUẬN (tách biệt hoàn toàn: mỗi lần chạy nhận MỘT ảnh, MỘT lượt sinh)",
            fontsize=10, fontweight="bold", color=C_P2E)
    box(19, 2, 12, 11, "ĐẦU VÀO PHA 2:\nMỘT ảnh\nđơn lẻ", C_DL, C_DLE, fs=7.8, bold=True)
    box(35, 2, 22, 11, "MÔ HÌNH CHƯNG CẤT\n(bf16 + adapter gộp)",
        C_P2, C_P2E, fs=8.6)
    box(61, 2, 38, 11,
        "ĐẦU RA: mô tả NGẮN (~0,8s) hoặc CHI TIẾT (~1,9s)\nđánh giá: KTVIC test 558 ảnh · COCO-2014 2.500 ảnh\n(xuyên ngôn ngữ, giao thức CHAIR 80 lớp)",
        C_P2, C_P2E, fs=8)
    arrow(31, 7.5, 35, 7.5); arrow(57, 7.5, 61, 7.5)
    # mũi tên mô hình từ kết quả pha 1 xuống pha 2
    ax.plot([86.5, 86.5, 46, 46], [22, 18.5, 18.5, 13], color=C_KQE,
            ls="--", lw=1.6)
    ax.add_patch(FancyArrowPatch((46, 14.5), (46, 13), arrowstyle="-|>",
                                 mutation_scale=14, color=C_KQE, lw=1.6,
                                 linestyle="--"))

    fig.tight_layout()
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300); fig.savefig(out.with_suffix(".pdf"))
    print(f"  đã ghi {out} và {out.with_suffix('.pdf')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
