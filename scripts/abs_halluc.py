#!/usr/bin/env python
"""Compute "absolute hallucinated objects/caption" + CHAIR_s + CJK rate for
every system — the new row of Tables 1/2.

    python scripts/abs_halluc.py \\
        --results data/results \\
        --out data/results/abs_halluc_summary.json

CHAIR_i is a ratio (hallucinated/total mentions), so a terse system gets
punished harder than a talkative one even when it fabricates less in absolute
terms; the "absolute hallucinated objects/caption" row =
n_hallucinated_mentions / n_captions from the SAME scorer
`rescap.chair.chair` — no second counter, only the denominator changes.
Recomputed from the *.preds.json on disk, so every number in both tables can
be rebuilt with one command.
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
        print(f"{name:32s} halluc/cap={s['halluc_per_caption']:.3f}  "
              f"CHAIR_i={s['chair_i']:.1%}  CHAIR_s={s['chair_s']:.1%}  "
              f"CJK={s['cjk_captions']}/{n}")

    out = Path(args.out) if args.out else results_dir / "abs_halluc_summary.json"
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2,
                              sort_keys=True), encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
