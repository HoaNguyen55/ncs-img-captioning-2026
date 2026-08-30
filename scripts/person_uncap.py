#!/usr/bin/env python
""" (research log)b — B-PERSON OFFLINE path: lift the gender ceiling using SAVED evidence.

    python scripts/person_uncap.py \
        --in research/backups/stage1 --out ~/ncs-data/stage1_person

Finding (diagnosis 24/08): stage-1 records keep the ENTIRE probe evidence
(affirming/negating + samples + grounding_score), and the UNCERTAIN verdict of a
person proposition is typically "supported by visual evidence" OVERRIDDEN by the
gender ceiling (verify.py ~2614) + the INFERENCE tag generate.py attaches to the
gendered noun itself (inference_type="none" — no other inference involved).

The neutralized proposition ("một người") asserts NO gender → every reason for
the ceiling vanishes → the ORIGINAL verifier's correct verdict is SUPPORTED.
Offline un-capping = absolute fidelity to the original evidence: no re-asking,
no sampling noise.

UNCAP CRITERIA (pre-registered, machine-readable, NO parsing of explanation text):
 1. verification.status == UNCERTAIN;
 2. neutralization ACTUALLY changes the text (it carries a gendered head);
 3. epistemic INFERENCE due to the noun alone: external_knowledge.inference_type
    in {"none", None} (a real inference KEEPS the ceiling);
 4. the original evidence is unanimous: the affirming probe polarity=affirms with
    EVERY sample affirming, the negating probe polarity=refutes with EVERY sample
    denying, visual_grounding.grounding_score >= 0.9.
All 4 met → neutral text + status SUPPORTED + trace `backoff{kind:
"gender_uncap", original_text_vi, grounding_score}`. The entity registry of
un-capped existence propositions is also neutralized: head + gender khong_xac_dinh.
Other records are copied verbatim. The input directory is immutable.
"""

from __future__ import annotations

import argparse
import copy
import json
import re
from collections import Counter
from pathlib import Path

GENDERED = (
    "người phụ nữ", "người đàn ông", "cô gái", "chàng trai", "cậu bé",
    "cô bé", "cậu con trai", "cô con gái", "phụ nữ", "đàn ông",
    "con trai", "con gái", "cô", "anh", "chị", "bà", "ông",
)
_GENDER_RE = re.compile(
    r"\b(" + "|".join(sorted(GENDERED, key=len, reverse=True)) + r")\b"
)


def neutralise(text: str) -> str:
    out = re.sub(r"\b(cô|anh|chị|ông|bà|em|họ)\s+ấy\b", " ", text)
    out = _GENDER_RE.sub("người", out)
    out = re.sub(r"\bngười(\s+người)+\b", "người", out)
    out = re.sub(r"\s{2,}", " ", out).strip()
    return out


_YES = ("có", "đúng", "phải", "yes")
_NO = ("không", "sai", "no")


def _all_answers(probe: dict, expected: str) -> bool:
    samples = probe.get("samples") or [probe.get("answer")]
    if not samples:
        return False
    heads = _YES if expected == "yes" else _NO
    for s in samples:
        t = str(s or "").strip().lower().rstrip(".!")
        if not any(t.startswith(h) for h in heads):
            return False
    return True


def evidence_unanimous(p: dict) -> tuple[bool, float | None]:
    ev = p.get("evidence") or {}
    probes = ev.get("probes") or []
    affirm = next((q for q in probes if q.get("polarity") == "affirms"), None)
    refute = next((q for q in probes if q.get("polarity") == "refutes"), None)
    if not affirm or not refute:
        return False, None
    g = ((ev.get("visual_grounding") or {}).get("grounding_score"))
    if g is None or float(g) < 0.9:
        return False, g
    ok = _all_answers(affirm, "yes") and _all_answers(refute, "no")
    return ok, float(g)


def eligible(p: dict) -> tuple[bool, float | None]:
    v = p.get("verification") or {}
    if v.get("status") != "UNCERTAIN":
        return False, None
    text = str(p.get("text_vi") or "")
    if neutralise(text) == text:
        return False, None
    ek = ((p.get("evidence") or {}).get("external_knowledge") or {})
    if ek.get("inference_type") not in (None, "none"):
        return False, None
    return evidence_unanimous(p)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in", dest="src", required=True)
    ap.add_argument("--out", dest="dst", required=True)
    ap.add_argument(
        "--types", default="entity,attribute,action,counting",
        help="Only un-cap these types. relation/spatial_relation EXCLUDED BY DEFAULT: "
             "the least trusted class per the (research log) audit (42.9%% human-rejected) and "
             "the planner does not yet control relation chains — keep the ceiling until "
             "image-region anchoring lands (the extended version).",
    )
    args = ap.parse_args()
    allowed_types = {t.strip() for t in args.types.split(",") if t.strip()}

    src = Path(args.src).expanduser()
    dst = Path(args.dst).expanduser()
    dst.mkdir(parents=True, exist_ok=True)

    stats: Counter[str] = Counter()
    for f in sorted(src.glob("*.json")):
        rec = json.loads(f.read_text(encoding="utf-8"))
        stats["records"] += 1
        flipped_entity_ids: set[str] = set()
        changed = False
        for p in rec.get("propositions") or []:
            if str(p.get("type")) not in allowed_types:
                continue
            ok, g = eligible(p)
            stats["candidates"] += bool((p.get("verification") or {}).get("status") == "UNCERTAIN"
                                        and neutralise(str(p.get("text_vi") or "")) != str(p.get("text_vi") or ""))
            if not ok:
                continue
            original = p.get("text_vi")
            p["text_vi"] = neutralise(str(original))
            subj = p.get("subject") or {}
            for key in ("text_vi", "head_noun_vi"):
                if subj.get(key):
                    subj[key] = neutralise(str(subj[key]))
            v = p.get("verification") or {}
            v["status"] = "SUPPORTED"
            v["explanation_vi"] = (
                "Gỡ trần giới tính ( (nhật ký NC)b): bằng chứng gốc nhất trí "
                f"(grounding {g:.2f}); mệnh đề trung tính không khẳng định "
                "giới tính nên các lý do trần không còn áp dụng."
            )
            p["verification"] = v
            p["backoff"] = {"kind": "gender_uncap",
                            "original_text_vi": original,
                            "grounding_score": g}
            changed = True
            stats["uncapped"] += 1
            stats[f"uncapped:{p.get('type')}"] += 1
            if p.get("type") == "entity":
                eid = (p.get("subject") or {}).get("entity_id")
                if eid:
                    flipped_entity_ids.add(str(eid))
        for e in rec.get("entities") or []:
            if str(e.get("id")) in flipped_entity_ids:
                for key in ("text_vi", "head_noun_vi"):
                    if e.get(key):
                        e[key] = neutralise(str(e[key]))
                gd = e.get("gender") or {}
                e["gender"] = {"value": "khong_xac_dinh",
                               "evidence": gd.get("evidence"),
                               "backoff_neutralised": True}
        if changed:
            stats["records_changed"] += 1
        (dst / f.name).write_text(json.dumps(rec, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(dict(stats), ensure_ascii=False, indent=1))
    print("=== UNCAP NGUOI DONE ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
