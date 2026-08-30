#!/usr/bin/env python
""" (nhật ký NC)b — B-NGƯỜI đường OFFLINE: gỡ trần giới tính bằng bằng chứng ĐÃ LƯU.

    python scripts/person_uncap.py \
        --in research/backups/stage1 --out ~/ncs-data/stage1_person

Phát hiện (chẩn đoán 24/08): bản ghi stage-1 lưu TRỌN bằng chứng probe
(khẳng định/phủ định + samples + grounding_score) và verdict KHÔNG CHẮC của
mệnh đề người thường là "Được hỗ trợ bởi bằng chứng thị giác" bị ĐÈ bởi trần
giới tính (verify.py ~2614) + thẻ INFERENCE mà generate.py gắn cho chính danh
từ giới tính (inference_type="none" — không có suy luận nào khác).

Mệnh đề trung tính hóa ("một người") KHÔNG khẳng định giới tính → mọi lý do
trần biến mất → verdict đúng của bộ kiểm GỐC là ỦNG HỘ. Gỡ trần offline =
trung thành tuyệt đối với bằng chứng gốc, không hỏi lại, không nhiễu lấy mẫu.

TIÊU CHÍ GỠ (đăng ký trước, máy-đọc-được, KHÔNG parse văn bản giải thích):
 1. verification.status == UNCERTAIN;
 2. trung tính hóa THẬT SỰ đổi văn bản (mang đầu giới tính);
 3. epistemic INFERENCE chỉ do danh từ: external_knowledge.inference_type
    trong {"none", None} (có suy luận thật thì GIỮ NGUYÊN trần);
 4. bằng chứng gốc nhất trí: probe khẳng định polarity=affirms với MỌI sample
    khẳng định, probe phủ định polarity=refutes với MỌI sample phủ nhận,
    visual_grounding.grounding_score >= 0.9.
Đạt đủ 4 → text trung tính + status SUPPORTED + vết `backoff{kind:
"gender_uncap", original_text_vi, grounding_score}`. Registry thực thể của
các mệnh đề tồn tại được gỡ cũng trung tính hóa đầu + gender khong_xac_dinh.
Bản ghi khác chép nguyên. Thư mục vào bất biến.
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
        help="Chỉ gỡ trần các loại này. MẶC ĐỊNH LOẠI relation/spatial_relation: "
             "lớp tin cậy thấp nhất theo audit (nhật ký NC) (42,9%% người bác) và "
             "planner chưa kiểm soát chuỗi quan hệ — giữ trần cho tới khi có "
             "neo vùng ảnh (bản mở rộng).",
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
