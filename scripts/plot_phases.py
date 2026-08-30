#!/usr/bin/env python
"""Draw the three per-phase detail figures (team request, night of 24/08):

    hinh2_pha1a  — Phase 1a: verified supervision generation (M1–M8)
    hinh3_pha1b  — Phase 1b: SFT distillation
    hinh4_pha2   — Phase 2: single-image inference

    python scripts/plot_phases.py --outdir research/paper/figures
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

C_MAIN = "#eaf2f6"; C_EDGE = "#1f6f8b"
C_EXT = "#fdf1e7"; C_EEDG = "#c96f2f"
C_DIS = "#eef7ee"; C_DEDG = "#2e933c"
C_DL = "#f2f2f2"; C_DLE = "#666666"


def _helpers(ax):
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    def box(x, y, w, h, text, fc, ec, fs=8.6, bold=False, ls="-", lw=1.5):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.35",
                                    fc=fc, ec=ec, lw=lw, linestyle=ls))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=fs, fontweight="bold" if bold else "normal",
                color="#1a1a1a")

    def arrow(x1, y1, x2, y2, color="#555", lw=1.5, ls="-"):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                     mutation_scale=13, color=color, lw=lw,
                                     linestyle=ls))
    return box, arrow


def ve_pha1a(out: Path):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(11.5, 4.9))
    ax.set_xlim(0, 100); ax.set_ylim(0, 44); ax.axis("off")
    box, arrow = _helpers(ax)
    y1, h = 31, 9
    box(1, y1, 10, h, "TẬP DỮ LIỆU\n(từng ảnh\ntrong tập)", C_DL, C_DLE, fs=7.8, bold=True)
    box(13, y1, 15, h, "M1–M2\nSinh mệnh đề\ncó cấu trúc", C_MAIN, C_EDGE)
    box(30, y1, 17, h, "M3 — Kiểm chứng ba trạng thái\nSUPPORTED · UNCERTAIN · REJECTED\n(probe khẳng định/phủ định,\nkiểm màu kép)", C_MAIN, C_EDGE, fs=6.6)
    box(49, y1, 15, h, "GIỚI HẠN PHÁN QUYẾT\nchỉ-hạ (đếm · suy luận\n· giới tính)", C_EXT, C_EEDG, fs=7.2, ls="solid")
    box(66, y1, 15, h, "M3b ★\nLÙI BẬC trung tính\n(nới giới hạn bằng\nbằng chứng đã lưu)", C_EXT, C_EEDG, fs=7.4, ls="solid")
    box(83, y1, 16, h, "M6\nChọn mệnh đề\n(ngân sách 9)", C_MAIN, C_EDGE)
    for xa, xb in ((11, 13), (28, 30), (47, 49), (64, 66), (81, 83)):
        arrow(xa, y1 + h / 2, xb, y1 + h / 2)
    y2 = 16
    box(83, y2, 16, h, "M7 ★\nKẾT XUẤT LAI: LM-có-\nràng-buộc (650 ảnh vượt\nmọi ràng buộc) + khuôn\nkho-đóng (fallback) · im lặng", C_EXT, C_EEDG, fs=7.6, ls="solid")
    arrow(91, y1, 91, y2 + h)
    box(58, y2, 22, h, "VỆ SINH GIÁM SÁT ★\nlọc mệnh đề rác · động từ\ntrang phục · rơi đại từ ·\nlọc khẩu ngữ", C_EXT, C_EEDG, fs=7.6, ls="solid")
    arrow(83, y2 + h / 2, 80, y2 + h / 2)
    box(33, y2, 22, h, "M8\nHậu kiểm tiếng Việt\n+ BẢNG TRUY VẾT\n(cụm ← mệnh đề ← phán quyết)", C_MAIN, C_EDGE, fs=7.4)
    arrow(58, y2 + h / 2, 55, y2 + h / 2)
    box(1, y2 - 1, 29, h + 2,
        "KHO GIÁM SÁT ĐÃ KIỂM CHỨNG\n7.177 ví dụ (3.617 ngắn + 3.560 chi tiết)\nba biến thể A/B/C — giám sát = bậc B",
        C_DIS, C_DEDG, fs=7.6, bold=True, lw=2.2)
    arrow(33, y2 + h / 2, 30, y2 + h / 2, color=C_DEDG)
    ax.text(1, 42.5, "PHA 1a — TIỀN XỬ LÝ: từ tập dữ liệu đến kho giám sát (chạy MỘT lần)",
            fontsize=10, fontweight="bold", color=C_EDGE)
    ax.text(99, 1.2, "★ = chặng đóng góp của bài (đều thuộc hệ chính)", fontsize=8,
            ha="right", color=C_EEDG)
    fig.tight_layout(); fig.savefig(out, dpi=300); fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig); print(f"  wrote {out}")


def ve_pha1b(out: Path):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9.0, 3.2))
    ax.set_xlim(0, 100); ax.set_ylim(0, 26); ax.axis("off")
    box, arrow = _helpers(ax)
    y, h = 4, 14
    box(1, y, 24, h, "KHO GIÁM SÁT\nĐÃ KIỂM CHỨNG\n(kết quả Pha 1a)", C_DIS, C_DEDG, fs=8.4, bold=True, lw=2.2)
    box(29, y, 30, h, "SFT QLoRA 4-bit NF4\nr=16 · α=32 · lr 1e-4 · 2 epoch\nbatch 1 × tích lũy 8 · 512 token thị giác\nHAI phong cách: ngắn + chi tiết", C_MAIN, C_EDGE, fs=7.8)
    box(63, y, 17, h, "3 HẠT GIỐNG\nđộc lập\n(dải dung sai\nđăng ký trước)", C_EXT, C_EEDG, fs=7.8, ls="solid")
    box(84, y, 15, h, "MÔ HÌNH\nCHƯNG CẤT 7B\n(adapter 170MB)", C_DIS, C_DEDG, fs=8.2, bold=True, lw=2.2)
    arrow(25, y + h / 2, 29, y + h / 2); arrow(59, y + h / 2, 63, y + h / 2); arrow(80, y + h / 2, 84, y + h / 2)
    ax.text(1, 23.5, "PHA 1b — HUẤN LUYỆN: chưng cất kho giám sát vào trọng số (1×RTX 4090; DPO bị loại qua khảo sát liều lượng, Mục 5)",
            fontsize=10, fontweight="bold", color=C_DEDG)
    fig.tight_layout(); fig.savefig(out, dpi=300); fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig); print(f"  wrote {out}")


def ve_pha2(out: Path):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9.0, 3.2))
    ax.set_xlim(0, 100); ax.set_ylim(0, 26); ax.axis("off")
    box, arrow = _helpers(ax)
    y, h = 4, 14
    box(1, y, 14, h, "ĐẦU VÀO:\nMỘT ảnh\nđơn lẻ", C_DL, C_DLE, fs=8.4, bold=True)
    box(19, y, 20, h, "PROMPT cố định\nngắn (một câu)\nhoặc chi tiết", C_MAIN, C_EDGE, fs=8.2)
    box(43, y, 24, h, "MÔ HÌNH CHƯNG CẤT\nbf16 + adapter gộp\nMỘT lượt sinh greedy\n(không kiểm chứng lúc chạy)", C_DIS, C_DEDG, fs=7.8, bold=True, lw=2.2)
    box(71, y, 28, h, "ĐẦU RA: mô tả tiếng Việt\nngắn ~0,79–0,99s · chi tiết ~1,45–1,96s\n(so 10–15s/ảnh của khung đa tác tử)", C_MAIN, C_EDGE, fs=7.8)
    arrow(15, y + h / 2, 19, y + h / 2); arrow(39, y + h / 2, 43, y + h / 2); arrow(67, y + h / 2, 71, y + h / 2)
    ax.text(1, 23.5, "PHA 2 — SUY LUẬN: mỗi lần chạy nhận đúng MỘT ảnh; chi phí trung thực đã trả trước ở Pha 1",
            fontsize=10, fontweight="bold", color=C_EEDG)
    fig.tight_layout(); fig.savefig(out, dpi=300); fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig); print(f"  wrote {out}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--outdir", default="research/paper/figures")
    args = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    d = Path(args.outdir); d.mkdir(parents=True, exist_ok=True)
    ve_pha1a(d / "hinh2_pha1a.png")
    ve_pha1b(d / "hinh3_pha1b.png")
    ve_pha2(d / "hinh4_pha2.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
