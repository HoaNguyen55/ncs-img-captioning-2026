#!/usr/bin/env python
""" (research log) — standard CHAIR scoring on COCO-2014 for the cross-lingual probe.

    python scripts/score_coco_probe.py \\
        --preds-dir ~/ncs-data/coco_probe_out \\
        --out data/results/coco_probe_scores.json

PROTOCOL FROZEN BEFORE SEEING RESULTS (committed before the probe finished — that
is the entire value of this file):
1. Object universe = EXACTLY the 80 COCO classes (Rohrbach 2018). Objects outside
   the 80 classes are not counted — on the mention side or the gold side.
2. Gold(image) = classes in instances_val2014 ∪ classes mentioned in the 5
   Karpathy reference captions (EN matching via Rohrbach's original synonyms.txt,
   + simple s/es plurals).
3. Object detection in the VIETNAMESE caption: the coco80_vi.json lexicon
   (committed beforehand), longest-phrase-first matching, boundaries are
   non-letter characters; `_rui_ro` notes in the lexicon are skipped when matching.
4. Metrics per (system × mode): classes mentioned/caption · CHAIR_i (invented
   classes / mentioned classes, pooled over the set) · CHAIR_s (% captions with
   ≥1 invented class) · absolute invented objects/caption (mean distinct
   invented classes).
5. Merge both shards in full; report every missing image by name; no post-hoc
   sample exclusion.
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
    """EN synonym → COCO class name, with simple plurals."""
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
    """[(VI phrase, COCO class)] — longest phrases first; skip _rui_ro notes."""
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
    # multi-word phrases first
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
    ap.add_argument("--out", default="data/results/coco_probe_scores.json")
    args = ap.parse_args()

    syn = load_synonyms()
    vi_terms = load_vi_terms()
    man = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    want = {i["cocoid"] for i in man["images"]}

    # gold 1 — real instances
    inst = json.loads((ANN / "instances_val2014.json").read_text())
    cat_name = {c["id"]: c["name"] for c in inst["categories"]}
    gold_inst: dict[int, set[str]] = defaultdict(set)
    for a in inst["annotations"]:
        if a["image_id"] in want:
            gold_inst[a["image_id"]].add(cat_name[a["category_id"]])
    # gold 2 — Karpathy reference captions
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
            print(f"{system:9s} {mode:9s}: mentions {r['mentions_per_caption']:.2f}/cap · "
                  f"CHAIR_i {100*r['chair_i']:.1f}% · CHAIR_s {100*r['chair_s']:.1f}% · "
                  f"absolute invented {r['abs_halluc_per_caption']:.2f}/cap"
                  + (f"  (MISSING {r['n_missing']} images)" if r["n_missing"] else ""))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "protocol": " (research log) frozen before seeing results; committed before the probe finished",
        "gold": "instances_val2014 ∪ ref-caption (synonyms.txt Rohrbach)",
        "results": results,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
