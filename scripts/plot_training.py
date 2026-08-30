#!/usr/bin/env python
"""Draw Figure 2 of the paper — SFT and DPO training curves, Vietnamese labels.

    python scripts/plot_training.py \\
        --sft ~/ncs-data/runs/sft/metrics.jsonl \\
        --dpo ~/ncs-data/runs/dpo/metrics.jsonl \\
        --out research/paper/figures/hinh2_huan_luyen.png

Figure 2 is a hard requirement from the team lead (18/08): the paper must show
the training process — loss, epochs, and post-training results. The paper is
written entirely in Vietnamese, so **every label on the figure is Vietnamese
too** — an axis reading "ordering accuracy" in the middle of a Vietnamese paper
is exactly the kind of thing a reviewer circles immediately.

Reads the `metrics.jsonl` that `train_stage2.py` writes, so the figure can be
rebuilt from disk without re-running anything. Outputs a 300 dpi PNG (enough
for the IEEE print version) plus a same-named PDF for LaTeX.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def load(path: str | None) -> list[dict]:
    if not path:
        return []
    p = Path(path).expanduser()
    if not p.exists():
        print(f"  ⚠ {p} not found — skipping this part")
        return []
    return [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sft", default=None)
    parser.add_argument("--dpo", default=None)
    parser.add_argument("--out", default="research/paper/figures/hinh2_huan_luyen.png")
    parser.add_argument("--lang", choices=["vi", "en"], default="vi")
    args = parser.parse_args()
    EN = args.lang == "en"

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # DejaVu (matplotlib's default) covers Vietnamese diacritics — no font install needed.
    sft = [e for e in load(args.sft) if e.get("stage") == "sft" and "loss" in e]
    # metrics.jsonl ends with a summary line (summary_per_pair_type) that has no
    # "step" — only plot the per-step lines; the summary belongs to the appendix table.
    dpo = [e for e in load(args.dpo)
           if e.get("stage") == "dpo" and "step" in e and "loss" in e]
    if not sft and not dpo:
        raise SystemExit("no metrics to plot — run train_stage2.py first")

    n_panels = (1 if sft else 0) + (2 if dpo else 0)
    fig, axes = plt.subplots(1, n_panels, figsize=(4.2 * n_panels, 3.4))
    if n_panels == 1:
        axes = [axes]
    axes = list(axes)

    if sft:
        ax = axes.pop(0)
        steps = [e.get("step", i) for i, e in enumerate(sft, 1)]
        ax.plot(steps, [e["loss"] for e in sft], "-o", ms=3,
                color="#1f6f8b", label="training loss" if EN else "mất mát (loss)")
        ax.set_xlabel("training step" if EN else "bước huấn luyện", fontsize=11)
        ax.set_ylabel("loss" if EN else "mất mát", fontsize=11)
        ax.set_title("(a) Supervised fine-tuning (SFT)" if EN else "(a) Tinh chỉnh có giám sát (SFT)", fontsize=11.5)
        acc = [(e.get("step", i), e["mean_token_accuracy"])
               for i, e in enumerate(sft, 1) if "mean_token_accuracy" in e]
        if acc:
            ax2 = ax.twinx()
            ax2.plot(*zip(*acc), "-s", ms=3, color="#d1495b",
                     label="token accuracy" if EN else "độ chính xác token")
            ax2.set_ylabel("token accuracy" if EN else "độ chính xác token", fontsize=11)
            ax2.set_ylim(0, 1)
            lines = ax.get_lines() + ax2.get_lines()
            ax.legend(lines, [l.get_label() for l in lines],
                      loc="center right", fontsize=10)
        ax.grid(alpha=0.3)

    if dpo:
        steps = [e["step"] for e in dpo]
        ax = axes.pop(0)
        ax.plot(steps, [e["loss"] for e in dpo], "-o", ms=3, color="#1f6f8b")
        ax.set_xlabel("bước huấn luyện")
        ax.set_ylabel("mất mát DPO")
        ax.set_title("(b) Tối ưu ưu tiên (DPO) — mất mát")
        ax.grid(alpha=0.3)

        ax = axes.pop(0)
        ax.plot(steps, [e["ordering_accuracy"] for e in dpo], "-o", ms=3,
                color="#2e933c", label="tỷ lệ đúng thứ tự")
        ax.set_xlabel("bước huấn luyện")
        ax.set_ylabel("tỷ lệ đúng thứ tự")
        ax.set_ylim(-0.05, 1.05)
        ax2 = ax.twinx()
        ax2.plot(steps, [e["reward_margin"] for e in dpo], "-s", ms=3,
                 color="#d1495b", label="biên thưởng")
        ax2.set_ylabel("biên thưởng")
        ax2.axhline(0, color="#999", lw=0.6, ls="--")
        ax.set_title("(c) DPO — thứ tự ưu tiên 3 bậc")
        # Only labelled lines: the zero axhline has no label and would show up
        # in the legend as the literal string "_child1".
        lines = [l for l in ax.get_lines() + ax2.get_lines()
                 if not l.get_label().startswith("_")]
        ax.legend(lines, [l.get_label() for l in lines], loc="lower right", fontsize=8)
        ax.grid(alpha=0.3)

    fig.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300)
    fig.savefig(out.with_suffix(".pdf"))
    print(f"  wrote {out} and {out.with_suffix('.pdf')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
