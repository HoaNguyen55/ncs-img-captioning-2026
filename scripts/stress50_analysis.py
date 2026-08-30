#!/usr/bin/env python
"""Analyse the 50-image challenge set  on existing preds — pure CPU.

    python scripts/stress50_analysis.py \\
        --manifest ~/ncs-data/datasets/ktvic/stress50_manifest.json \\
        --preds "zero-shot=~/ncs-data/results/zeroshot-short.preds.json" \\
                "VSPS=~/ncs-data/results/vsps-short.preds.json" \\
        --out data/stress50/short.json

No GPU needed: every system already captioned all 558 test images, and the 50
hard images are a subset — just read the preds from disk and re-score. Three
numbers per system, broken down by each hard signal (counting / colour /
gender / entity richness):

* **CHAIR_i on the subset** — the object-hallucination upper bound, using the
  main table's exact CHAIR-vi lexicon (`rescap.chair`).
* **objects mentioned/caption** — detail, so CHAIR cannot win by silence.
* **gender fabrication** — % of captions using a gendered noun that NO
  reference caption of that image uses (same regex as build_stress_manifest).

The "has signal X" groups come straight from the manifest (per-signal scores
were recorded when the images were picked), so the breakdown table is
reproducible from the two input files.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

GENDERED = re.compile(
    r"\b(phụ nữ|đàn ông|cô gái|chàng trai|em bé|cậu bé|cô bé|bé trai|bé gái"
    r"|người bà|người ông|bà cụ|ông cụ|cô|chú|anh|chị)\b")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--preds", nargs="+", required=True,
                        help="name=path to preds.json ({image_id: caption})")
    parser.add_argument("--split", default="test_data.json")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    from evaluate import references
    from rescap.chair import objects_in

    manifest = json.loads(Path(args.manifest).expanduser().read_text(encoding="utf-8"))
    images = manifest["images"]
    ids = [str(r["image_id"]) for r in images]
    signal_of = {str(r["image_id"]): r for r in images}
    refs = references(args.split)

    # caption-derived gold for EACH image: the objects and gendered nouns that at
    # least one of the 5 reference captions mentions.
    gold_objects: dict[str, set] = {}
    gold_gender: dict[str, set] = {}
    for i in ids:
        caps = refs[str(i)]
        objs, gens = set(), set()
        for c in caps:
            objs.update(objects_in(c))
            gens.update(GENDERED.findall(c.lower()))
        gold_objects[i] = objs
        gold_gender[i] = gens

    groups = {
        "all 50": ids,
        "counting": [i for i in ids if signal_of[i]["counting"] > 0],
        "colour": [i for i in ids if signal_of[i]["colour"] > 0],
        "xanh (grue)": [i for i in ids if signal_of[i]["xanh"] > 0],
        "gender": [i for i in ids if signal_of[i]["gendered"] > 0],
    }

    result = {"manifest": str(args.manifest), "groups": {g: len(v) for g, v in groups.items()},
              "systems": {}}
    for spec in args.preds:
        name, _, path = spec.partition("=")
        preds = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
        preds = {str(k): (v[0] if isinstance(v, list) else v) for k, v in preds.items()}
        missing = [i for i in ids if i not in preds]
        if missing:
            raise SystemExit(f"{name}: missing {len(missing)} manifest images — "
                             f"preds must cover all 558 (e.g. {missing[:3]})")

        per_image = {}
        for i in ids:
            cap = (preds[i] or "").lower()
            objs = objects_in(cap)
            halluc = [o for o in objs if o not in gold_objects[i]]
            gens = set(GENDERED.findall(cap))
            per_image[i] = {
                "mentions": len(objs),
                "halluc": len(halluc),
                "gender_fab": bool(gens - gold_gender[i]),
            }

        def agg(sub):
            rows = [per_image[i] for i in sub]
            m = sum(r["mentions"] for r in rows)
            h = sum(r["halluc"] for r in rows)
            return {
                "n": len(sub),
                "chair_i": round(h / m, 3) if m else None,
                "mentions_per_caption": round(m / len(sub), 2),
                "gender_fab_pct": round(
                    100 * sum(r["gender_fab"] for r in rows) / len(sub), 1),
            }

        result["systems"][name.strip()] = {g: agg(sub) for g, sub in groups.items()}

    out = Path(args.out).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    for name, by_group in result["systems"].items():
        print(f"\n== {name} ==")
        for g, a in by_group.items():
            print(f"  {g:<12} n={a['n']:>2}  CHAIR_i={a['chair_i']}  "
                  f"objects/cap={a['mentions_per_caption']}  gender fab={a['gender_fab_pct']}%")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
