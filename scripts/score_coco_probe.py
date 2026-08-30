#!/usr/bin/env python
""" (nhật ký NC) — chấm CHAIR chuẩn trên COCO-2014 cho probe xuyên ngôn ngữ.

    python research/scripts/score_coco_probe.py \\
        --preds-dir ~/ncs-data/coco_probe_out \\
        --out research/paper/data/results/coco_probe_scores.json

GIAO THỨC ĐÓNG BĂNG TRƯỚC KHI THẤY KẾT QUẢ (commit trước khi probe xong — đó
là toàn bộ giá trị của file này):
1. Vũ trụ vật thể = ĐÚNG 80 lớp COCO (Rohrbach 2018). Vật thể ngoài 80 lớp
   không được đếm — cả phía nhắc lẫn phía vàng.
2. Vàng(ảnh) = lớp trong instances_val2014 ∪ lớp nhắc trong 5 caption tham
   chiếu Karpathy (khớp EN qua synonyms.txt gốc của Rohrbach, + số nhiều s/es).
3. Phát hiện vật thể trong caption TIẾNG VIỆT: từ điển coco80_vi.json (đã
   commit trước), khớp cụm dài nhất trước, biên là ký tự không phải chữ; các
   ghi chú `_rui_ro` trong từ điển bị bỏ qua khi khớp.
4. Chỉ số cho từng (hệ × chế độ): số lớp nhắc/caption · CHAIR_i (lớp ảo /
   lớp nhắc, gộp toàn tập) · CHAIR_s (% caption có ≥1 lớp ảo) · vật thể ảo
   tuyệt đối/caption (trung bình số lớp ảo phân biệt).
5. Gộp đủ 2 shard; thiếu ảnh nào báo ảnh đó; không loại mẫu hậu nghiệm.
"""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from collections import defaultdict
from pathlib import Path

HOME = Path.home()
PROBE_DATA = Path(__file__).resolve().parents[1] / "paper" / "data" / "coco_probe"
ANN = HOME / "ncs-data" / "datasets" / "coco_probe" / "annotations"
KARPATHY = HOME / "ncs-data" / "datasets" / "karpathy" / "dataset_coco.json"

_LETTER = r"a-zA-ZÀ-ỹ"


def load_synonyms() -> dict[str, str]:
    """EN synonym → tên lớp COCO, kèm số nhiều đơn giản."""
    mapping: dict[str, str] = {}
    for line in (PROBE_DATA / "chair_synonyms.txt").read_text().splitlines():
        parts = [p.strip().lower() for p in line.split(",") if p.strip()]
        if not parts:
            continue
        cls = parts[0]
        for p in parts:
            mapping[p] = cls
            mapping[p + "s"] = cls
            mapping[p + "es"] = cls
    return mapping


def load_vi_terms() -> list[tuple[str, str]]:
    """[(cụm VI, lớp COCO)] — cụm dài xếp trước; bỏ ghi chú _rui_ro."""
    d = json.loads((PROBE_DATA / "coco80_vi.json").read_text(encoding="utf-8"))
    pairs = []
    for cls, terms in d.items():
        if cls.startswith("_"):
            continue
        for t in terms:
            if isinstance(t, str) and not t.startswith("_rui_ro"):
                pairs.append((t.lower(), cls))
    return sorted(pairs, key=lambda x: -len(x[0]))


def vi_classes(caption: str, vi_terms) -> set[str]:
    text = " " + unicodedata.normalize("NFC", caption.lower()) + " "
    found: set[str] = set()
    for term, cls in vi_terms:
        if cls in found:
            continue
        for m in re.finditer(re.escape(term), text):
            a, b = m.start() - 1, m.end()
            if not re.match(f"[{_LETTER}]", text[a]) and not re.match(f"[{_LETTER}]", text[b]):
                found.add(cls)
                break
    return found


def en_classes(caption: str, syn: dict[str, str]) -> set[str]:
    text = " " + re.sub(f"[^{_LETTER}]", " ", caption.lower()) + " "
    found: set[str] = set()
    # cụm nhiều từ trước
    for phrase, cls in syn.items():
        if " " in phrase and f" {phrase} " in text:
            found.add(cls)
    for tok in text.split():
        if tok in syn:
            found.add(syn[tok])
    return found


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--preds-dir", required=True)
    ap.add_argument("--manifest", default=str(PROBE_DATA / "manifest.json"))
    ap.add_argument("--out", default="research/paper/data/results/coco_probe_scores.json")
    args = ap.parse_args()

    syn = load_synonyms()
    vi_terms = load_vi_terms()
    man = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    want = {i["cocoid"] for i in man["images"]}

    # vàng 1 — instance thật
    inst = json.loads((ANN / "instances_val2014.json").read_text())
    cat_name = {c["id"]: c["name"] for c in inst["categories"]}
    gold_inst: dict[int, set[str]] = defaultdict(set)
    for a in inst["annotations"]:
        if a["image_id"] in want:
            gold_inst[a["image_id"]].add(cat_name[a["category_id"]])
    # vàng 2 — caption tham chiếu Karpathy
    kar = json.loads(KARPATHY.read_text())
    gold_ref: dict[int, set[str]] = defaultdict(set)
    for img in kar["images"]:
        if img["cocoid"] in want:
            for s in img["sentences"]:
                gold_ref[img["cocoid"]] |= en_classes(s["raw"], syn)
    gold = {i: gold_inst[i] | gold_ref[i] for i in want}

    preds_dir = Path(args.preds_dir).expanduser()
    results = {}
    for system in ("zeroshot", "distill"):
        for mode in ("short", "detailed"):
            rows: dict[int, str] = {}
            for f in sorted(preds_dir.glob(f"{system}-{mode}.shard*.jsonl")):
                for line in f.read_text(encoding="utf-8").splitlines():
                    r = json.loads(line)
                    rows[r["cocoid"]] = r["caption"]
            missing = sorted(want - set(rows))
            n = len(rows)
            if not n:
                continue
            men_total = hal_total = hal_caps = 0
            hal_per_cap = []
            for cid, cap in rows.items():
                mentioned = vi_classes(cap, vi_terms)
                halluc = mentioned - gold[cid]
                men_total += len(mentioned)
                hal_total += len(halluc)
                hal_caps += bool(halluc)
                hal_per_cap.append(len(halluc))
            results[f"{system}-{mode}"] = {
                "n_captions": n, "n_missing": len(missing),
                "missing_sample": missing[:5],
                "mentions_per_caption": round(men_total / n, 3),
                "chair_i": round(hal_total / max(men_total, 1), 4),
                "chair_s": round(hal_caps / n, 4),
                "abs_halluc_per_caption": round(sum(hal_per_cap) / n, 3),
            }
            r = results[f"{system}-{mode}"]
            print(f"{system:9s} {mode:9s}: nhắc {r['mentions_per_caption']:.2f}/cap · "
                  f"CHAIR_i {100*r['chair_i']:.1f}% · CHAIR_s {100*r['chair_s']:.1f}% · "
                  f"ảo tuyệt đối {r['abs_halluc_per_caption']:.2f}/cap"
                  + (f"  (THIẾU {r['n_missing']} ảnh)" if r["n_missing"] else ""))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "protocol": " (nhật ký NC) đóng băng trước khi thấy kết quả; commit trước khi probe xong",
        "gold": "instances_val2014 ∪ ref-caption (synonyms.txt Rohrbach)",
        "results": results,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"đã ghi {out}")
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
