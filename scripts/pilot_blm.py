#!/usr/bin/env python
"""PILOT B-LM: kết xuất giám sát bằng LM-có-ràng-buộc trên 100 ảnh.

    python research/scripts/pilot_blm.py --in /root/stage1_bnguoi \
        --out /root/blm_pilot.json --n 100

Nền: build_dpo_data gọi realize() KHÔNG truyền model → toàn bộ giám sát rơi
về khuôn luật (mở "Có một…") — trần văn phong của mọi hệ hiện tại. Đường
constrained_lm có sẵn trong realize.py: prompt §6.1 + vòng sửa vi phạm +
đối chiếu cụm↔mệnh đề. Pilot đo trên 100 ảnh (seed 42): văn ra sao, tỷ lệ
rơi-về-khuôn, tỷ lệ vi phạm ràng buộc, độ dài.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in", dest="src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--images", default="/root/ncs-data/datasets/ktvic/images")
    args = ap.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from PIL import Image
    from rescap.pipeline.realize import RealizeConfig, realize
    from rescap.pipeline.select import SelectionConfig, select
    from rescap.pipeline.verify import verdict_name
    from rescap.vlm.registry import get_vlm
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from build_dpo_data import clean_props
    from collections import Counter as _Counter

    files = sorted(Path(args.src).glob("*.json"))
    random.seed(args.seed)
    random.shuffle(files)

    print("nạp qwen2.5-vl-7b (bộ kết xuất)…", flush=True)
    vlm = get_vlm("qwen2.5-vl-7b").load()

    out_path = Path(args.out)
    results = json.loads(out_path.read_text(encoding="utf-8")) if out_path.exists() else {}
    stats = {"n": 0, "fallback_template": 0, "violation_spans": 0,
             "captions_with_violation": 0, "non_vi_retry": 0, "errors": 0}
    t0 = time.time()
    picked = 0
    for f in files:
        if picked >= args.n:
            break
        rec = json.loads(f.read_text(encoding="utf-8"))
        rid = f.stem
        props = rec.get("propositions") or []
        entities = rec.get("entities") or []
        props = clean_props(list(props), _Counter())  # vệ sinh vòng 1 quên
        supported = [p for p in props if verdict_name(p) == "SUPPORTED"]
        if len(supported) < 3:
            continue
        picked += 1
        if rid in results:
            continue
        img_path = Path(args.images) / str(rec.get("file_name"))
        image = Image.open(img_path).convert("RGB")
        try:
            sel = select(supported, entities, config=SelectionConfig(budget=9))
            keep = {str(i) for i in (getattr(sel, "selected_ids", []) or [])}
            chosen = [p for p in supported if str(p.get("id")) in keep] or supported[:6]
            res = realize(chosen, entities, model=vlm, image=image,
                          config=RealizeConfig(strategy="constrained_lm", no_strip=True, max_retries=3))
            cap = res.caption
            text = cap.get("text_vi") if isinstance(cap, dict) else getattr(cap, "text_vi", None)
            st = res.stats
            viol = getattr(st, "violation_spans", None)
            if viol is None:
                viol = getattr(st, "violations", 0) or 0
            try:
                viol = int(viol)
            except (TypeError, ValueError):
                viol = len(viol) if hasattr(viol, "__len__") else 0
            fell = bool(getattr(st, "fell_back_to_template", False))
            results[rid] = {
                "caption": text, "n_props": len(chosen),
                "fell_back_to_template": fell, "violations": viol,
                "non_vi_retries": getattr(st, "non_vietnamese_retries", 0),
                "generator": getattr(st, "strategy", "?"),
            }
            stats["n"] += 1
            stats["fallback_template"] += fell
            stats["violation_spans"] += viol
            stats["captions_with_violation"] += bool(viol)
            stats["non_vi_retry"] += bool(getattr(st, "non_vietnamese_retries", 0))
        except Exception as e:
            stats["errors"] += 1
            results[rid] = {"error": f"{type(e).__name__}: {e}"[:200]}
        if stats["n"] % 10 == 0 and stats["n"]:
            out_path.write_text(json.dumps(results, ensure_ascii=False, indent=1),
                                encoding="utf-8")
            rate = (time.time() - t0) / max(stats["n"], 1)
            print(f"  {stats['n']}/{args.n} · {rate:.1f}s/ảnh · fallback "
                  f"{stats['fallback_template']} · vi phạm {stats['captions_with_violation']}",
                  flush=True)
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False))
    print("=== PILOT BLM DONE ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
