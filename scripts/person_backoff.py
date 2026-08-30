#!/usr/bin/env python
"""B-NGƯỜI: backoff kiểm chứng thực thể người bằng đầu trung tính.

    python scripts/person_backoff.py \
        --in /root/stage1_records --out /root/stage1_person --shard 0 --of 2

Số nền (24/08): 74% ảnh có người mất NGUYÊN CỤM chủ thể — thực thể "một người
phụ nữ" bị KHÔNG CHẮC (nghi do vế giới tính trong danh từ) kéo mọi thuộc tính
treo chết theo, dù bộ sinh tạo đủ nguyên liệu (vd 7240 P1→P3/P4).

Nguyên tắc (đặc-hiệu-theo-bằng-chứng, áp xuống tầng kiểm chứng): mệnh đề
người KHÔNG CHẮC được hỏi lại MỘT lần với danh từ đầu trung tính hóa
("một người phụ nữ" → "một người") bằng ĐÚNG bộ máy verify của stage-1
(phủ định kép + kiểm màu chéo, cùng cấu hình). Đậu thì mệnh đề sống với văn
bản trung tính (giữ vết `backoff` kèm văn bản gốc); không đậu giữ nguyên.
Bản ghi gốc BẤT BIẾN — kết quả ghi ra thư mục MỚI, ảnh không có gì đổi được
chép nguyên. Kháng lặp: stem đã có trong --out thì bỏ qua.
"""

from __future__ import annotations

import argparse
import copy
import json
import re
import time
from pathlib import Path

# Đầu người mang GIỚI TÍNH (trung tính hóa); "đứa trẻ/em bé" chỉ tuổi — giữ.
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
    # rơi đại từ lặp chủ ngữ ("một người phụ nữ CÔ ẤY đội mũ" — M1 hay sinh
    # kiểu này) trước khi trung tính hóa, kẻo thành "một người ấy đội mũ".
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
    # xoá verdict cũ để verify chấm lại từ trắng
    q.pop("verification", None)
    q.pop("contradicts", None)
    q.pop("evidence", None)
    return q


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in", dest="src", required=True)
    ap.add_argument("--out", dest="dst", required=True)
    ap.add_argument("--images", default=None,
                    help="thư mục ảnh KTVIC (mặc định $NCS_DATA/datasets/ktvic/images)")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--of", type=int, default=1)
    ap.add_argument("--verifier", default="vintern-1b")
    ap.add_argument("--generator", default="qwen2.5-vl-7b",
                    help="bộ kiểm màu chéo, như stage-1; 'none' để tắt")
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
    print(f"mảnh {args.shard}/{args.of}: {len(files)} bản ghi, còn {len(todo)}", flush=True)

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
        # cụm = thực thể người KHÔNG CHẮC + mệnh đề treo cùng entity_id KHÔNG CHẮC
        cluster_ids = {subject_entity_id(p) for p in person_unc} - {None}
        attached = [p for p in props
                    if subject_entity_id(p) in cluster_ids
                    and verdict_name(p) == "UNCERTAIN"
                    and str(p.get("id")) not in {str(q.get("id")) for q in person_unc}]

        targets = person_unc + attached
        # chỉ hỏi lại mệnh đề mà trung tính hóa THẬT SỰ đổi văn bản
        targets = [p for p in targets
                   if neutralise(str(p.get("text_vi", ""))) != str(p.get("text_vi", ""))]
        if not targets:
            (dst / f.name).write_text(json.dumps(rec, ensure_ascii=False), encoding="utf-8")
            stats["copied_unchanged"] += 1
            continue

        if ver_model is None:
            print(f"nạp {args.verifier} …", flush=True)
            ver_model = get_vlm(args.verifier).load()
            if hasattr(ver_model, "max_tiles"):
                ver_model.max_tiles = args.verifier_tiles
            if args.generator != "none":
                print(f"nạp {args.generator} (kiểm màu chéo) …", flush=True)
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
        # TRẦN GIỚI TÍNH (verify.py ~2614) đọc entity.gender.value từ REGISTRY,
        # không phải văn bản mệnh đề — chẩn đoán 24/08: "Supported by visual
        # evidence. Capped at UNCERTAIN". Mệnh đề trung tính KHÔNG khẳng định
        # giới tính nên registry đưa vào verify cũng phải trung tính: đầu
        # "người" + gender khong_xac_dinh → trần không còn lý do đè.
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
            print(f"  ! verify lỗi {f.stem}: {type(e).__name__}", flush=True)
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
        # thực thể có mệnh đề tồn tại lật → đầu trung tính trong registry
        for e in rec.get("entities") or []:
            if str(e.get("id")) in flipped_entity_ids:
                for key in ("text_vi", "head_noun_vi"):
                    if e.get(key):
                        e[key] = neutralise(str(e[key]))

        (dst / f.name).write_text(json.dumps(rec, ensure_ascii=False), encoding="utf-8")
        if n % 25 == 0:
            rate = (time.time() - t0) / n
            print(f"  {n}/{len(todo)} · {rate:.1f}s/ảnh · còn ~{int((len(todo)-n)*rate/60)} phút "
                  f"· lật {stats['flipped_entity']}+{stats['flipped_attached']}", flush=True)

    print(json.dumps(stats, ensure_ascii=False))
    print("=== BACKOFF NGUOI DONE ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
