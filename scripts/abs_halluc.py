#!/usr/bin/env python
"""Tính "vật thể ảo tuyệt đối/caption" + CHAIR_s + tỷ lệ CJK cho mọi hệ — hàng
mới của Bảng 1/2 .

    python research/scripts/abs_halluc.py \\
        --results research/paper/data/results \\
        --out research/paper/data/results/abs_halluc_summary.json

CHAIR_i là tỷ lệ (ảo/tổng nhắc) nên hệ nói ít bị phạt nặng hơn hệ nói nhiều dù
bịa ít hơn về tuyệt đối; hàng "vật thể ảo tuyệt đối/caption" =
n_hallucinated_mentions / n_captions của CÙNG bộ chấm `rescap.chair.chair` —
không có bộ đếm thứ hai, chỉ đổi mẫu số. Chạy lại từ *.preds.json trên đĩa nên
mọi số trong hai bảng tái tạo được bằng một lệnh.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

CJK = re.compile(r"[一-鿿]")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", default="data/results")
    parser.add_argument("--split", default="test_data.json")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    from evaluate import references
    from rescap.chair import chair

    refs = references(args.split)
    results_dir = Path(args.results).expanduser()
    summary: dict[str, dict] = {}
    for f in sorted(results_dir.glob("*.preds.json")):
        name = f.name.removesuffix(".preds.json")
        preds = json.loads(f.read_text(encoding="utf-8"))
        preds = {str(k): (v[0] if isinstance(v, list) else v) or ""
                 for k, v in preds.items()}
        r = chair(preds, refs, strict_ids=False)
        n = len(preds)
        n_cjk = sum(1 for c in preds.values() if CJK.search(c))
        summary[name] = {
            "n_captions": n,
            "halluc_per_caption": round(r.n_hallucinated_mentions / n, 3),
            "n_hallucinated_mentions": r.n_hallucinated_mentions,
            "mentions_per_caption": round(r.mentions_per_caption, 2),
            "chair_i": round(r.chair_i, 3) if r.chair_i is not None else None,
            "chair_s": round(r.chair_s, 3) if r.chair_s is not None else None,
            "cjk_captions": n_cjk,
            "cjk_pct": round(100 * n_cjk / n, 1),
        }
        s = summary[name]
        print(f"{name:32s} ảo/cap={s['halluc_per_caption']:.3f}  "
              f"CHAIR_i={s['chair_i']:.1%}  CHAIR_s={s['chair_s']:.1%}  "
              f"CJK={s['cjk_captions']}/{n}")

    out = Path(args.out) if args.out else results_dir / "abs_halluc_summary.json"
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2,
                              sort_keys=True), encoding="utf-8")
    print(f"\nđã ghi {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
