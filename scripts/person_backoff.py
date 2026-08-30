#!/usr/bin/env python
"""B-PERSON: re-verify person entities with a neutral head noun (backoff).

    python scripts/person_backoff.py \
        --in /root/stage1_records --out /root/stage1_person --shard 0 --of 2

Baseline figures (24/08): 74% of images with people lose the WHOLE subject
cluster — the entity "một người phụ nữ" going UNCERTAIN (suspected: the gender
half of the noun) drags every attached attribute down with it, even though the
generator produced plenty of material (e.g. 7240 P1→P3/P4).

Principle (evidence-proportional specificity, pushed down into the verification
layer): an UNCERTAIN person proposition is re-asked ONCE with the head noun
neutralized ("một người phụ nữ" → "một người") using EXACTLY stage-1's verify
machinery (double negation + colour cross-check, same config). If it passes, the
proposition survives with the neutral text (a `backoff` trace keeps the original
text); if not, it is left as-is. The source records are IMMUTABLE — results go to
a NEW directory, and images with nothing changed are copied verbatim.
Restart-safe: a stem already present in --out is skipped.
"""

from __future__ import annotations

import argparse
import copy
import json
import re
import time
from pathlib import Path

# GENDERED person head nouns (to neutralize); "đứa trẻ/em bé" mark age only — keep.
GENDERED = (
    "người phụ nữ", "người đàn ông", "cô gái", "chàng trai", "cậu bé",
    "cô bé", "cậu con trai", "cô con gái", "phụ nữ", "đàn ông",
    "con trai", "con gái", "cô", "anh", "chị", "bà", "ông",
)
_GENDER_RE = re.compile(
    r"\b(" + "|".join(sorted(GENDERED, key=len, reverse=True)) + r")\b"
)
PERSON_HINT = GENDERED + ("người", "đứa trẻ", "em bé", "trẻ em")


def neutralise(text: str) -> str:
    # drop the repeated subject pronoun ("một người phụ nữ CÔ ẤY đội mũ" — M1
    # often generates this) before neutralizing, lest it become "một người ấy đội mũ".
    out = re.sub(r"\b(cô|anh|chị|ông|bà|em|họ)\s+ấy\b", " ", text)
    out = _GENDER_RE.sub("người", out)
    out = re.sub(r"\bngười(\s+người)+\b", "người", out)
    out = re.sub(r"\s{2,}", " ", out).strip()
    return out


def is_person_entity_prop(p: dict) -> bool:
    if p.get("type") != "entity":
        return False
    t = str(p.get("text_vi", "")).lower()
    return any(h in t for h in PERSON_HINT)


def subject_entity_id(p: dict) -> str | None:
    s = p.get("subject") or {}
    v = s.get("entity_id")
    return str(v) if v is not None else None


def neutralise_prop(p: dict) -> dict:
    q = copy.deepcopy(p)
    for key in ("text_vi",):
        if q.get(key):
            q[key] = neutralise(str(q[key]))
    subj = q.get("subject") or {}
    for key in ("text_vi", "head_noun_vi"):
        if subj.get(key):
            subj[key] = neutralise(str(subj[key]))
    # clear the old verdict so verify scores again from scratch
    q.pop("verification", None)
    q.pop("contradicts", None)
    q.pop("evidence", None)
    return q


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in", dest="src", required=True)
    ap.add_argument("--out", dest="dst", required=True)
    ap.add_argument("--images", default=None,
                    help="KTVIC image directory (default $NCS_DATA/datasets/ktvic/images)")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--of", type=int, default=1)
    ap.add_argument("--verifier", default="vintern-1b")
    ap.add_argument("--generator", default="qwen2.5-vl-7b",
                    help="the colour cross-checker, as in stage-1; 'none' to disable")
    ap.add_argument("--verifier-tiles", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    import os
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from rescap.pipeline.verify import verdict_name, verify
    from rescap.vlm.registry import get_vlm
    from PIL import Image

    ncs = Path(os.environ.get("NCS_DATA", Path.home() / "ncs-data"))
    img_dir = Path(args.images) if args.images else ncs / "datasets" / "ktvic" / "images"
    src, dst = Path(args.src), Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)

    files = sorted(src.glob("*.json"))[args.shard :: args.of]
    if args.limit:
        files = files[: args.limit]
    done = {p.stem for p in dst.glob("*.json")}
    todo = [f for f in files if f.stem not in done]
    print(f"shard {args.shard}/{args.of}: {len(files)} records, {len(todo)} to go", flush=True)

    ver_model = gen_model = None
    stats = {"images": 0, "with_cluster": 0, "probed_props": 0,
             "flipped_entity": 0, "flipped_attached": 0, "still_uncertain": 0,
             "copied_unchanged": 0, "errors": 0}
    t0 = time.time()

    for n, f in enumerate(todo, 1):
        rec = json.loads(f.read_text(encoding="utf-8"))
        props = rec.get("propositions") or []
        by_id = {str(p.get("id")): p for p in props}
        stats["images"] += 1

        person_unc = [p for p in props
                      if is_person_entity_prop(p) and verdict_name(p) == "UNCERTAIN"]
        # cluster = UNCERTAIN person entity + UNCERTAIN propositions attached to the same entity_id
        cluster_ids = {subject_entity_id(p) for p in person_unc} - {None}
        attached = [p for p in props
                    if subject_entity_id(p) in cluster_ids
                    and verdict_name(p) == "UNCERTAIN"
                    and str(p.get("id")) not in {str(q.get("id")) for q in person_unc}]

        targets = person_unc + attached
        # only re-ask propositions whose text neutralization ACTUALLY changes
        targets = [p for p in targets
                   if neutralise(str(p.get("text_vi", ""))) != str(p.get("text_vi", ""))]
        if not targets:
            (dst / f.name).write_text(json.dumps(rec, ensure_ascii=False), encoding="utf-8")
            stats["copied_unchanged"] += 1
            continue

        if ver_model is None:
            print(f"loading {args.verifier} …", flush=True)
            ver_model = get_vlm(args.verifier).load()
            if hasattr(ver_model, "max_tiles"):
                ver_model.max_tiles = args.verifier_tiles
            if args.generator != "none":
                print(f"loading {args.generator} (colour cross-check) …", flush=True)
                gen_model = get_vlm(args.generator).load()

        img_path = img_dir / str(rec.get("file_name"))
        if not img_path.exists():
            stats["errors"] += 1
            (dst / f.name).write_text(json.dumps(rec, ensure_ascii=False), encoding="utf-8")
            continue
        image = Image.open(img_path).convert("RGB")

        stats["with_cluster"] += 1
        copies = [neutralise_prop(p) for p in targets]
        entities = rec.get("entities") or []
        # The GENDER CEILING (verify.py ~2614) reads entity.gender.value from the
        # REGISTRY, not from the proposition text — diagnosis 24/08: "Supported by
        # visual evidence. Capped at UNCERTAIN". A neutral proposition asserts NO
        # gender, so the registry handed to verify must be neutral too: head
        # "người" + gender khong_xac_dinh → the ceiling has no reason left to press.
        entities_probe = []
        for e in entities:
            e2 = copy.deepcopy(e)
            txt = str(e2.get("text_vi") or "") + " " + str(e2.get("head_noun_vi") or "")
            if _GENDER_RE.search(txt.lower()):
                for key in ("text_vi", "head_noun_vi"):
                    if e2.get(key):
                        e2[key] = neutralise(str(e2[key]))
                g = e2.get("gender") or {}
                e2["gender"] = {"value": "khong_xac_dinh",
                                "evidence": g.get("evidence"),
                                "backoff_neutralised": True}
            entities_probe.append(e2)
        try:
            verify(ver_model, image, entities_probe, copies, colour_verifier=gen_model)
        except Exception as e:
            stats["errors"] += 1
            print(f"  ! verify error {f.stem}: {type(e).__name__}", flush=True)
            (dst / f.name).write_text(json.dumps(rec, ensure_ascii=False), encoding="utf-8")
            continue
        stats["probed_props"] += len(copies)

        flipped_entity_ids: set[str] = set()
        for orig, cop in zip(targets, copies):
            if verdict_name(cop) == "SUPPORTED":
                pid = str(orig.get("id"))
                original_text = orig.get("text_vi")
                by_id[pid].update({
                    "text_vi": cop.get("text_vi"),
                    "subject": cop.get("subject"),
                    "verification": cop.get("verification"),
                    "evidence": cop.get("evidence"),
                    "backoff": {"kind": "gender_neutral",
                                "original_text_vi": original_text},
                })
                if is_person_entity_prop(orig):
                    stats["flipped_entity"] += 1
                    eid = subject_entity_id(orig)
                    if eid:
                        flipped_entity_ids.add(eid)
                else:
                    stats["flipped_attached"] += 1
            else:
                stats["still_uncertain"] += 1
        # entity whose existence proposition flipped → neutral head in the registry
        for e in rec.get("entities") or []:
            if str(e.get("id")) in flipped_entity_ids:
                for key in ("text_vi", "head_noun_vi"):
                    if e.get(key):
                        e[key] = neutralise(str(e[key]))

        (dst / f.name).write_text(json.dumps(rec, ensure_ascii=False), encoding="utf-8")
        if n % 25 == 0:
            rate = (time.time() - t0) / n
            print(f"  {n}/{len(todo)} · {rate:.1f}s/image · ~{int((len(todo)-n)*rate/60)} min left "
                  f"· flipped {stats['flipped_entity']}+{stats['flipped_attached']}", flush=True)

    print(json.dumps(stats, ensure_ascii=False))
    print("=== BACKOFF NGUOI DONE ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
