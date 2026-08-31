#!/usr/bin/env python
"""Object-level precision, recall, and F1 over unique classes per image (Table VII).

    python scripts/object_prf.py                       # KTVIC test split, all *.preds.json in data/results
    python scripts/object_prf.py --coco-preds ~/ncs-data/coco_score5000   # add the COCO-2014 block

Definitions (Section IV-C of the paper): with M_i the set of object classes mentioned
in the caption of image i and G_i its gold classes, precision = sum|M_i & G_i| / sum|M_i|,
recall = mean over images with non-empty G_i of |M_i & G_i| / |G_i|, F1 = harmonic mean.
KTVIC uses the same 113-entry object lexicon and reference-derived gold as CHAIR-vi;
COCO uses the frozen instance-plus-reference gold of score_coco_probe.py. Because the
sets count unique classes rather than mentions, precision is close to but not identical
to 1 - CHAIR_i.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))


def prf(tp: int, mentions: int, recalls: list[float]) -> tuple[float, float, float]:
    p = tp / mentions if mentions else 0.0
    r = statistics.mean(recalls) if recalls else 0.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return 100 * p, 100 * r, 100 * f


def ktvic_block(results_dir: Path, split: str) -> None:
    from evaluate import references
    from rescap.chair import objects_in

    refs = references(split)
    gold = {str(k): set().union(*[set(objects_in(r)) for r in rs]) for k, rs in refs.items()}
    print(f"KTVIC ({split}, {len(gold)} images) -- object precision / recall / F1 (%)")
    for f in sorted(results_dir.glob("*-detailed.preds.json")):
        preds = json.loads(f.read_text(encoding="utf-8"))
        preds = {str(k): (v[0] if isinstance(v, list) else v) or "" for k, v in preds.items()}
        tp = mentions = 0
        recalls: list[float] = []
        for k, cap in preds.items():
            m = set(objects_in(cap))
            g = gold.get(k, set())
            tp += len(m & g)
            mentions += len(m)
            if g:
                recalls.append(len(m & g) / len(g))
        p, r, f1 = prf(tp, mentions, recalls)
        name = f.name.removesuffix("-detailed.preds.json")
        print(f"  {name:26s} true/cap={tp/len(preds):.2f}  P={p:5.1f}  R={r:5.1f}  F1={f1:5.1f}")


def coco_block(preds_dir: Path, manifest: Path) -> None:
    import score_coco_probe as sc

    syn = sc.load_synonyms()
    vi_terms = sc.load_vi_terms()
    want = {i["cocoid"] for i in json.loads(manifest.read_text(encoding="utf-8"))["images"]}
    inst = json.loads((sc.ANN / "instances_val2014.json").read_text())
    cat = {c["id"]: c["name"] for c in inst["categories"]}
    gold_inst: dict[int, set[str]] = defaultdict(set)
    for a in inst["annotations"]:
        if a["image_id"] in want:
            gold_inst[a["image_id"]].add(cat[a["category_id"]])
    gold_ref: dict[int, set[str]] = defaultdict(set)
    for img in json.loads(sc.KARPATHY.read_text())["images"]:
        if img["cocoid"] in want:
            for s in img["sentences"]:
                gold_ref[img["cocoid"]] |= sc.en_classes(s["raw"], syn)
    gold = {i: gold_inst[i] | gold_ref[i] for i in want}
    print(f"\nCOCO-2014 ({len(want)} images, instance + reference gold) -- object precision / recall / F1 (%)")
    for system in ("zeroshot", "distill"):
        rows: dict[int, str] = {}
        for f in sorted(preds_dir.glob(f"{system}-detailed.shard*.jsonl")):
            for line in f.read_text(encoding="utf-8").splitlines():
                r = json.loads(line)
                rows[r["cocoid"]] = r["caption"]
        if not rows:
            continue
        tp = mentions = 0
        recalls: list[float] = []
        for cid, cap in rows.items():
            m = sc.vi_classes(cap, vi_terms)
            tp += len(m & gold[cid])
            mentions += len(m)
            if gold[cid]:
                recalls.append(len(m & gold[cid]) / len(gold[cid]))
        p, r, f1 = prf(tp, mentions, recalls)
        label = "VSPS" if system == "distill" else "zero-shot"
        print(f"  {label:26s} n={len(rows)}  true/cap={tp/len(rows):.2f}  P={p:5.1f}  R={r:5.1f}  F1={f1:5.1f}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default=str(ROOT / "data" / "results"))
    ap.add_argument("--split", default="test_data.json")
    ap.add_argument("--coco-preds", default=None, help="directory with {zeroshot,distill}-detailed.shard*.jsonl")
    ap.add_argument("--coco-manifest", default=str(ROOT / "data" / "coco_probe" / "manifest_full5000.json"))
    args = ap.parse_args()
    ktvic_block(Path(args.results).expanduser(), args.split)
    if args.coco_preds:
        coco_block(Path(args.coco_preds).expanduser(), Path(args.coco_manifest))
    return 0


if __name__ == "__main__":
    sys.exit(main())
