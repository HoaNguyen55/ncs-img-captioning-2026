#!/usr/bin/env python
"""Vẽ Hình 3 của bài báo — Phân tích đáp ứng liều lượng DPO (Dose-Response Analysis).

    python research/scripts/plot_dpo_dose.py \
        --data_dir research/paper/data/results \
        --out research/paper/figures/hinh3_lieu_dpo.png

Trục x: Số bước tối ưu hóa DPO (0, 150, 500, 1626 bước).
Trục y trái: Điểm số CIDEr mức từ (ngắn).
Trục y phải: Tỷ lệ (%) caption bị lẫn ký tự CJK.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

CJK_REGEX = re.compile(r"[\u4e00-\u9fff]")
HEDGE_REGEX = re.compile(r"có\s+vẻ|dường\s+như", re.IGNORECASE)


def compute_metrics(json_path: Path, preds_path: Path | None = None) -> dict:
    if not json_path.exists():
        return {}
    data = json.loads(json_path.read_text(encoding="utf-8"))
    word = next((r for r in data.get("results", []) if r.get("segmenter") == "rdrsegmenter"), None)
    cider = word["scores_scaled_x100"].get("CIDEr", 0.0) if word else 0.0
    chair_i = (data.get("chair", {}).get("CHAIR_i") or 0.0) * 100

    cjk_pct = 0.0
    hedge_density = 0.0
    if preds_path and preds_path.exists():
        preds = json.loads(preds_path.read_text(encoding="utf-8"))
        if isinstance(preds, dict):
            captions = list(preds.values())
        elif isinstance(preds, list):
            captions = [p.get("caption", "") if isinstance(p, dict) else str(p) for p in preds]
        else:
            captions = []

        if captions:
            cjk_count = sum(1 for c in captions if CJK_REGEX.search(c))
            cjk_pct = (cjk_count / len(captions)) * 100
            total_hedge = sum(len(HEDGE_REGEX.findall(c)) for c in captions)
            hedge_density = total_hedge / len(captions)

    return {
        "cider": cider,
        "chair_i": chair_i,
        "cjk_pct": cjk_pct,
        "hedge_density": hedge_density,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", default="research/paper/data/results")
    parser.add_argument("--out", default="research/paper/figures/hinh3_lieu_dpo.png")
    parser.add_argument("--lang", choices=["vi", "en"], default="vi")
    args = parser.parse_args()
    EN = args.lang == "en"

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data_dir = Path(args.data_dir)

    # 4 điểm đo liều lượng DPO
    doses = [
        {"steps": 0, "name": "0\n(pure SFT)" if EN else "0\n(SFT-thuần)", "file": "offsft-diag-short.json", "preds": "offsft-diag-short.preds.json"},
        {"steps": 150, "name": "150\n(Round B)" if EN else "150\n(Vòng B)", "file": "vongB-short.json", "preds": "vongB-short.preds.json"},
        {"steps": 500, "name": "500\n(Round A)" if EN else "500\n(Vòng A)", "file": "vongA-short.json", "preds": "vongA-short.preds.json"},
        {"steps": 1626, "name": "1,626\n(high dose)" if EN else "1.626\n(số bước cao)", "file": "official-short.json", "preds": "official-short.preds.json"},
    ]

    x_steps = []
    x_labels = []
    ciders = []
    cjk_pcts = []
    hedges = []

    for d in doses:
        f_json = data_dir / d["file"]
        f_preds = data_dir / d["preds"]
        m = compute_metrics(f_json, f_preds)
        if m:
            x_steps.append(d["steps"])
            x_labels.append(d["name"])
            ciders.append(m["cider"])
            cjk_pcts.append(m["cjk_pct"])
            hedges.append(m["hedge_density"])
        elif d["steps"] == 150:
            # Ước lượng tạm cho Vòng B trong lúc chờ GPU chạy
            x_steps.append(150)
            x_labels.append("150 (Vòng B)\n[chờ số]")
            ciders.append(16.0)
            cjk_pcts.append(0.0)
            hedges.append(0.3)

    fig, ax1 = plt.subplots(figsize=(6.2, 3.6))

    # Trục trái: CIDEr ngắn
    color_cider = "#1f6f8b"
    ax1.set_xlabel("DPO training steps" if EN else "Số bước huấn luyện DPO", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Word-level CIDEr (concise, ×100)" if EN else "CIDEr mức từ (chế độ ngắn, ×100)", color=color_cider, fontsize=11, fontweight="bold")
    line1 = ax1.plot(range(len(x_steps)), ciders, color=color_cider, marker="o", lw=2, ms=6, label="Word CIDEr (concise)" if EN else "CIDEr từ (ngắn)")
    for i, txt in enumerate(ciders):
        ax1.annotate(f"{txt:.1f}", (i, txt), textcoords="offset points", xytext=(0, 7), ha="center", fontsize=9.5, color=color_cider, fontweight="bold")

    ax1.axhline(16.4, color="#888888", linestyle="--", lw=1.1, alpha=0.8)
    ax1.text(0.05, 16.8, "Template-teacher level (16.4)" if EN else "Mức khuôn giáo viên (16,4)", fontsize=9, color="#666666", style="italic")

    ax1.set_xticks(range(len(x_steps)))
    ax1.set_xticklabels(x_labels, fontsize=10)
    ax1.set_ylim(0, 22)
    ax1.grid(True, linestyle=":", alpha=0.5)

    # Trục phải: CJK %
    ax2 = ax1.twinx()
    color_cjk = "#c0392b"
    ax2.set_ylabel("CJK character leakage (%)" if EN else "Tỷ lệ lẫn ký tự CJK (%)", color=color_cjk, fontsize=11, fontweight="bold")
    line2 = ax2.plot(range(len(x_steps)), cjk_pcts, color=color_cjk, marker="s", lw=2, ms=6, linestyle="-", label="CJK rate (%)" if EN else "Tỷ lệ lẫn CJK (%)")
    for i, txt in enumerate(cjk_pcts):
        ax2.annotate(f"{txt:.1f}%", (i, txt), textcoords="offset points", xytext=(0, -14 if i >= 2 else 7), ha="center", fontsize=9.5, color=color_cjk, fontweight="bold")

    ax2.set_ylim(-2, 35)

    # Gộp legend
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2, fontsize=10)

    plt.title("Tri-level DPO dose\u2013response" if EN else "Đáp ứng theo cấu hình DPO ba bậc", fontsize=11.5, fontweight="bold", pad=10)
    fig.tight_layout()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    print(f"  đã ghi {out} và {out.with_suffix('.pdf')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
