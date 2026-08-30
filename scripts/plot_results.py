#!/usr/bin/env python
"""Vẽ Hình 3 — cột so sánh kết quả sau huấn luyện, hai cụm chế độ.

    python scripts/plot_results.py \\
        --short  "zero-shot=~/ncs-data/results/zs-short.json" \\
                 "chưng cất=~/ncs-data/results/official-short.json" \\
        --detailed "zero-shot=~/ncs-data/results/zs-detailed.json" \\
                   "chưng cất=~/ncs-data/results/official-detailed.json" \\
        --out research/paper/figures/hinh3_ketqua.png

Hình 3 là yêu cầu bắt buộc của trưởng nhóm (18/08). Đọc thẳng file kết quả
của `evaluate.py` nên tái tạo được từ đĩa; KHÔNG nhận số gõ tay — số gõ tay
là chỗ số-ước lẻn vào bài.

* Cụm trái (chế độ ngắn): CIDEr mức từ, kèm vạch tham chiếu GRIT 136,0
  (hệ có giám sát, số công bố) — thước đo tham vọng, không phải của ta.
* Cụm phải (chế độ chi tiết): CHAIR_i (thấp = tốt, ghi rõ là CẬN TRÊN) và
  số vật thể nhắc/caption (độ chi tiết — cao = tốt). Hai cột cạnh nhau vì
  bài học vòng 1: CHAIR_i giảm bằng cách CÂM ĐI là thất bại, phải nhìn cả
  hai cùng lúc.

Mã hoá kép (màu + hoa văn gạch) — in trắng đen vẫn phân biệt được.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

GRIT_CIDER = 136.0  # số công bố của GRIT trên KTVIC, thang x100

PALETTE = [("#9aa5b1", ""), ("#1f6f8b", "//"), ("#2e933c", "xx"), ("#d1495b", "..")]


def load(spec: str):
    name, _, path = spec.partition("=")
    data = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    word = next((r for r in data.get("results", [])
                 if r.get("segmenter") == "rdrsegmenter"), None)
    if word is None:
        raise SystemExit(f"{path}: không có hàng rdrsegmenter — số không so được")
    chair = data.get("chair") or {}
    return name.strip(), {
        "CIDEr": word["scores_scaled_x100"].get("CIDEr"),
        "CHAIR_i": (chair.get("CHAIR_i") or 0) * 100,
        "mentions": chair.get("mentions_per_caption"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--short", nargs="+", required=True,
                        help="tên=đường-dẫn kết quả chế độ ngắn, theo thứ tự vẽ")
    parser.add_argument("--detailed", nargs="+", required=True)
    parser.add_argument("--out", default="research/paper/figures/hinh3_ketqua.png")
    args = parser.parse_args()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    short = [load(s) for s in args.short]
    detailed = [load(s) for s in args.detailed]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.4, 3.6),
                                   gridspec_kw={"width_ratios": [1, 1.4]})

    # --- cụm ngắn: CIDEr ---
    xs = range(len(short))
    for i, (name, m) in enumerate(short):
        colour, hatch = PALETTE[i % len(PALETTE)]
        ax1.bar(i, m["CIDEr"], color=colour, hatch=hatch, edgecolor="white",
                width=0.62)
        ax1.text(i, m["CIDEr"] + 1.2, f"{m['CIDEr']:.1f}", ha="center",
                 fontsize=8)
    ax1.axhline(GRIT_CIDER, color="#555", lw=1, ls="--")
    ax1.text(len(short) - 0.5, GRIT_CIDER + 1.5,
             f"GRIT (có giám sát) {GRIT_CIDER:.0f}", ha="right", fontsize=7,
             color="#555")
    ax1.set_xticks(list(xs), [n for n, _ in short], fontsize=8)
    ax1.set_ylabel("CIDEr (mức từ, ×100)")
    ax1.set_title("(a) Chế độ ngắn — 558 ảnh", fontsize=9)
    ax1.grid(axis="y", alpha=0.3)

    # --- cụm chi tiết: CHAIR_i + vật thể/caption, hai trục ---
    width = 0.38
    ax2b = ax2.twinx()
    for i, (name, m) in enumerate(detailed):
        colour, hatch = PALETTE[i % len(PALETTE)]
        ax2.bar(i - width / 2, m["CHAIR_i"], width=width, color=colour,
                hatch=hatch, edgecolor="white")
        ax2.text(i - width / 2, m["CHAIR_i"] + 1, f"{m['CHAIR_i']:.1f}",
                 ha="center", fontsize=7.5)
        ax2b.bar(i + width / 2, m["mentions"], width=width, color=colour,
                 hatch=hatch, edgecolor="white", alpha=0.55)
        ax2b.text(i + width / 2, m["mentions"] + 0.12, f"{m['mentions']:.2f}",
                  ha="center", fontsize=7.5)
    ax2.set_xticks(range(len(detailed)), [n for n, _ in detailed], fontsize=8)
    ax2.set_ylabel("CHAIR_i % (cận trên; thấp = tốt) — cột trái")
    ax2b.set_ylabel("vật thể nhắc/caption (cao = chi tiết) — cột phải")
    ax2.set_title("(b) Chế độ chi tiết — trung thực và độ chi tiết cùng lúc",
                  fontsize=9)
    ax2.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300)
    fig.savefig(out.with_suffix(".pdf"))
    print(f"  đã ghi {out} và {out.with_suffix('.pdf')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
