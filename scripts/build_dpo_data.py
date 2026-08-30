#!/usr/bin/env python
"""Turn Stage 1 verdict records into tri-level preference data for Stage 2.

    python scripts/build_dpo_data.py --in ~/ncs-data/stage1 --out ~/ncs-data/stage2

Writes `sft.jsonl` and `dpo.jsonl`. Nothing here calls a model: Stage 1 already
paid for the verdicts, and this is the symbolic step that turns them into
supervision.

**The three rungs.** Ordinary preference data has two — good and bad. The
verifier gives three, and the middle one is where the contribution lives:

    A  SUPPORTED asserted + UNCERTAIN hedged     full coverage, nothing overclaimed
    B  SUPPORTED asserted, UNCERTAIN dropped     honest, but says less
    C  UNCERTAIN asserted flatly, or REJECTED    says more than the image supports

A > B says **hedge rather than stay silent** -- the detail is worth keeping when
it is marked as uncertain. B > C says **stay silent rather than overclaim**. Both
directions are needed: training on B > C alone teaches the model that the safest
caption is the shortest one, which loses exactly the detail the paper is about.

That ordering is why the third verdict has to exist. Collapse UNCERTAIN into
REJECTED and A disappears; collapse it into SUPPORTED and C does.

**Pairs, not rankings.** DPO consumes pairs, so the three rungs yield
(A,B), (B,C) and (A,C). The last is the easiest to learn from and the least
informative -- it is kept because dropping it would bias the mix toward
hard pairs, and reported separately so its share is visible.

**What gets skipped, loudly.** An image whose verdicts cannot produce a given
rung yields no pair for it, and the count is reported. A builder that quietly
emitted only the images that worked would describe a cleaner dataset than the
one Stage 1 actually produced.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

PROMPT = "Mô tả chi tiết bức ảnh này bằng tiếng Việt."
#: Same wording as evaluate.py's short-mode prompt ON PURPOSE: round 1 trained
#: only the detailed style, so short mode leaned entirely on prompting a model
#: whose finetuning pulled the other way. Training a short variant under the
#: same instruction the benchmark uses lets one model serve both tables.
PROMPT_SHORT = (
    "Mô tả bức ảnh này bằng MỘT câu tiếng Việt ngắn gọn, "
    "giống chú thích ảnh. Không liệt kê, không giải thích."
)


def verdict_of(proposition: dict) -> str | None:
    """One shared parser (`pipeline.verify.verdict_name`)."""
    from rescap.pipeline.verify import verdict_name

    return verdict_name(proposition)


def as_asserted(props: list[dict]) -> list[dict]:
    """Copies with the verdict rewritten to SUPPORTED.

    `realize.py` refuses to state an UNCERTAIN proposition flatly -- it unions
    verdict-derived UNCERTAIN into the hedge set whatever the caller passes.
    That is right for the production path and it is exactly why rung C could not
    be built: the overclaiming caption came out identical to the hedged one, and
    all 17 A>C pairs were dropped as duplicates.

    So the negative is constructed deliberately, by relabelling the copies. The
    originals are untouched, and the relabelled ones never leave this function's
    caller -- they exist only to render a caption we want the model to rank
    lower.
    """
    import copy

    out = []
    for prop in props:
        clone = copy.deepcopy(prop)
        clone.setdefault("verification", {})["status"] = "SUPPORTED"
        out.append(clone)
    return out


#: (nhật ký NC): ngân sách lựa chọn cho dữ liệu giám sát. 5 = công thức chính thức
#: (mặc định của SelectionConfig — "nút vặn chi tiết/rủi ro"). Ghi đè bằng
#: --budget để dựng biến thể nhiều-chi-tiết; mặc định giữ nguyên hành vi cũ.
SELECT_BUDGET: list[int | None] = [None]
# (nhật ký NC) (gói câu chữ): bật khuôn kết xuất xoay + lọc mệnh đề rác.
STYLE_VARIATION: list[bool] = [False]
CLEAN_PROPS: list[bool] = [False]
CURRENT_STYLE_SEED: list[int] = [42]


def _malformed(text: str) -> bool:
    """Mệnh đề rác theo (nhật ký NC): lặp từ liền kề, hoặc predicate nuốt nguyên mệnh đề."""
    words = text.split()
    if any(a.lower() == b.lower() for a, b in zip(words, words[1:])):
        return True
    return False


def clean_props(props: list[dict], stats: Counter) -> list[dict]:
    """Lọc rác trước build ( (nhật ký NC) #2): lặp từ · predicate quá dài · trùng lặp."""
    seen: set[str] = set()
    kept: list[dict] = []
    for p in props:
        text = str(p.get("text_vi") or "")
        pred = str(((p.get("predicate") or {}).get("lemma_vi")) or "")
        if _malformed(text):
            stats["clean:dup_word"] += 1
            continue
        # Vết bug M2 (audit (nhật ký NC), vd 4962 P23/P35): predicate nuốt nguyên một
        # mệnh đề CÓ CHỦ NGỮ RIÊNG ("một người đang nắm chặt..."). Predicate dài
        # nhưng không mở bằng danh ngữ vô định + đang/đã là hợp lệ, giữ nguyên.
        import re as _re
        _emb = r"(?:^|\s)(một số|một vài|nhiều|vài|các|những|một|hai|ba|bốn|năm)\s+(?:\S+\s+){1,6}?(đang|đã)\s"
        m_pred = _re.search(_emb, " " + pred)
        m_text = _re.search(_emb, text)
        # match ở ĐẦU text là chủ ngữ hợp lệ của mệnh đề hành động — chỉ bắt
        # NP-vô-định + đang/đã NHÚNG GIỮA chuỗi (vết bug M2 nuốt mệnh đề).
        if m_pred or (m_text and m_text.start() > 0) or len(pred.split()) > 12:
            stats["clean:pred_clause"] += 1
            continue
        key = text.lower().strip()
        if key and key in seen:
            stats["clean:dup_text"] += 1
            continue
        seen.add(key)
        kept.append(p)
    return kept


def _select(props: list[dict], entities: list[dict], stats: Counter, label: str) -> list[dict]:
    """Run M6 and return the propositions it kept, in its order."""
    if not props:
        return []
    from rescap.pipeline.select import SelectionConfig, select

    try:
        if SELECT_BUDGET[0] is None:
            result = select(props, entities)
        else:
            result = select(props, entities,
                            config=SelectionConfig(budget=SELECT_BUDGET[0]))
    except Exception as e:
        stats[f"select_failed:{label}:{type(e).__name__}"] += 1
        return list(props)
    keep = {str(i) for i in (getattr(result, "selected_ids", []) or [])}
    chosen = [p for p in props if str(p.get("id")) in keep]
    stats[f"selected:{label}"] += len(chosen)
    stats[f"dropped_by_selection:{label}"] += len(props) - len(chosen)
    return chosen or list(props)[:6]


import re as _re_verb

# (nhật ký NC): hợp đúng động từ trang phục theo danh từ — luật kho-đóng, bảo toàn sự
# thật (đội mũ/nón · đeo kính/túi/đồng hồ/khẩu trang · đi giày/dép). Vết: 10/6403
# ví dụ giám sát v2 còn "mặc mũ"; output B-rộng 7240 "mặc một chiếc mũ".
_WEAR_FIXES = (
    (_re_verb.compile(r"\bmặc(\s+(?:một|hai|vài|nhiều)?\s*(?:chiếc|cái)?\s*)(mũ|nón)\b"), r"đội\1\2"),
    (_re_verb.compile(r"\bmặc(\s+(?:một|hai|vài|nhiều)?\s*(?:chiếc|cái)?\s*)(kính|túi|đồng hồ|khẩu trang)\b"), r"đeo\1\2"),
    (_re_verb.compile(r"\bmặc(\s+(?:một|hai|vài|nhiều)?\s*(?:đôi|chiếc)?\s*)(giày|dép)\b"), r"đi\1\2"),
)


_PRONOUN_REMNANT = _re_verb.compile(r"\b(cô|anh|chị|ông|bà|em|họ)\s+ấy\b")


def fix_wear_verbs(text: str) -> str:
    for pat, rep in _WEAR_FIXES:
        text = pat.sub(rep, text)
    # Tàn dư đại từ sau trung tính hóa ( (nhật ký NC)b): chủ ngữ đã thành "người"
    # nhưng text mệnh đề (kể cả mệnh đề neo trên thực thể QUẦN ÁO) còn
    # "cô ấy" → caption "một người ... cô ấy mặc ...". Rơi toàn bộ đại từ
    # lặp — tiếng Việt cho phép lược chủ ngữ lặp (đúng luật kết xuất §6),
    # và văn giám sát vốn kiêng đại từ (luật 7 cấm mở câu bằng đại từ).
    text = _PRONOUN_REMNANT.sub(" ", text)
    text = _re_verb.sub(r"\s{2,}", " ", text)
    text = _re_verb.sub(r"\s+([,.;:!?])", r"\1", text).strip()
    return text


def realise(props, entities, hedged_ids, stats: Counter, label: str) -> str | None:
    """Render one variant. Returns None when there is nothing to say."""
    if not props:
        return None
    from rescap.pipeline.realize import RealizeConfig, realize

    try:
        cfg = None
        # v2 (bài học bvan42-v1): khuôn xoay CHỈ cho chi tiết — chế độ ngắn phải
        # giữ văn cô đọng chuẩn KTVIC, khuôn "Bức ảnh cho thấy" làm BLEU-4 sập.
        if STYLE_VARIATION[0] and label != "short":
            cfg = RealizeConfig(style_variation=True, seed=CURRENT_STYLE_SEED[0])
        result = realize(props, entities, hedged_ids=list(hedged_ids), config=cfg)
    except Exception as e:
        stats[f"realise_failed:{label}:{type(e).__name__}"] += 1
        return None
    caption = getattr(result.caption, "text_vi", None) or getattr(
        result.caption, "text", None
    )
    if isinstance(result.caption, dict):
        caption = result.caption.get("text_vi")
    caption = (caption or "").strip()
    if caption and CLEAN_PROPS[0]:
        fixed = fix_wear_verbs(caption)
        if fixed != caption:
            stats["clean:wear_verb"] += 1
        caption = fixed
    return caption or None


def build_for_image(record: dict, stats: Counter):
    """Return `(sft_example, dpo_pairs)` for one Stage 1 record."""
    props = record.get("propositions") or []
    entities = record.get("entities") or []
    if CLEAN_PROPS[0]:
        props = clean_props(list(props), stats)
        # (nhật ký NC)b: thực thể đã trung tính hóa (backoff) → mọi mệnh đề treo trên
        # nó phải rơi đại từ lặp giới tính trong text ("... cô ấy mặc quần
        # trắng ..."), kẻo caption thành "một người ... cô ấy ...".
        neutral_ids = {str(e.get("id")) for e in entities
                       if (e.get("gender") or {}).get("backoff_neutralised")}
        if neutral_ids:
            import re as _re_pn
            _pn = _re_pn.compile(r"\b(cô|anh|chị|ông|bà|em|họ)\s+ấy\b")
            for p in props:
                if str(((p.get("subject") or {}).get("entity_id"))) in neutral_ids:
                    for holder, key in ((p, "text_vi"),
                                        (p.get("predicate") or {}, "lemma_vi")):
                        val = holder.get(key)
                        if val and _pn.search(val):
                            holder[key] = _re_pn.sub(
                                r"\s{2,}", " ", _pn.sub(" ", val)).strip()
                            stats["clean:pronoun_drop"] += 1
    try:
        CURRENT_STYLE_SEED[0] = int(str(record.get("image_id")).strip() or 42)
    except (TypeError, ValueError):
        CURRENT_STYLE_SEED[0] = 42
    by_verdict: dict[str, list[dict]] = {"SUPPORTED": [], "UNCERTAIN": [], "REJECTED": []}
    for p in props:
        v = verdict_of(p)
        if v in by_verdict:
            by_verdict[v].append(p)

    supported = by_verdict["SUPPORTED"]
    uncertain = by_verdict["UNCERTAIN"]
    rejected = by_verdict["REJECTED"]

    # SELECTION FIRST. Handing every verified proposition to the realiser
    # produced a 71-word caption carrying 42 claims, and it read like it:
    # `Nhiều người nhiều người một số người mặc áo màu đỏ có vẻ nhiều người
    # mặc áo màu xanh dương…`. M6 exists between verification and realisation
    # precisely to choose what goes in, and skipping it meant SFT would have
    # trained on that.
    chosen_a = _select(supported + uncertain, entities, stats, "A")
    hedged_a = [
        str(p.get("id")) for p in chosen_a
        if str(p.get("id")) in {str(q.get("id")) for q in uncertain}
    ]
    # Rung A: everything checkable that selection kept, unresolved parts marked.
    a = realise(chosen_a, entities, hedged_a, stats, "A")
    # Rung B: only what verified. Honest, and quieter.
    b = realise(_select(supported, entities, stats, "B"), entities, [], stats, "B")
    # Rung C: overclaiming. Two ways to be wrong, and they are different errors,
    # so whichever is available is used and which one is recorded.
    # BOTH failure modes get a negative, not whichever exists first. The old
    # first-available logic preferred uncertain_asserted, and with UNCERTAIN at
    # 60-75% of verdicts the rejected_included pair almost never existed -- so
    # DPO carried no direct anti-hallucination signal, and round 1 showed it:
    # CIDEr rose (+9.4) while CHAIR_i did not improve in short mode. The two
    # negatives teach different lessons and both belong in the mix.
    c_flat = None
    if uncertain:
        # UNCERTAIN stated as fact -- the overconfidence failure. Selected from
        # the SAME pool as rung A so the pair differs in hedging alone.
        c_flat = realise(as_asserted(chosen_a), entities, [], stats, "C_flat")
    c_rej = None
    if rejected:
        # REJECTED content included -- the hallucination failure: the caption
        # names things the image refuted. Relabelled because assert_selected
        # rightly raises on rejected content everywhere except here.
        c_rej = realise(
            supported + as_asserted(rejected[:4]), entities, [], stats, "C_rejected"
        )

    image_id = record.get("image_id")
    file_name = record.get("file_name")

    sft = None
    if a:
        sft = {
            "image_id": image_id, "file_name": file_name,
            "prompt": PROMPT, "response": a,
            "n_supported": len(supported), "n_uncertain": len(uncertain),
        }
        stats["sft"] += 1
    else:
        stats["skipped_no_rung_A"] += 1

    # Short variant: the top few SUPPORTED propositions in selection's own
    # priority order, realised as one short caption. KTVIC references are one
    # sentence, so this is the style Table 1 is scored against.
    sft_short = None
    top_ids = [str(p.get("id")) for p in chosen_a
               if verdict_of(p) == "SUPPORTED"][:2]
    if top_ids:
        short_caption = realise(
            [p for p in chosen_a if str(p.get("id")) in top_ids],
            entities, [], stats, "short",
        )
        if short_caption:
            sft_short = {
                "image_id": image_id, "file_name": file_name,
                "prompt": PROMPT_SHORT, "response": short_caption,
                "variant": "short",
            }
            stats["sft_short"] += 1

    pairs: list[dict] = []

    def add(chosen, rejected_text, kind):
        if not chosen or not rejected_text or chosen == rejected_text:
            stats[f"pair_unavailable:{kind}"] += 1
            return
        pairs.append({
            "image_id": image_id, "file_name": file_name, "prompt": PROMPT,
            "chosen": chosen, "rejected": rejected_text, "pair_type": kind,
        })
        stats[f"pair:{kind}"] += 1

    add(a, b, "A>B_hedge_beats_silence")
    add(b, c_flat, "B>C_silence_beats_overclaim:uncertain_asserted")
    add(a, c_flat, "A>C_hedge_beats_overclaim:uncertain_asserted")
    add(b, c_rej, "B>C_silence_beats_overclaim:rejected_included")
    add(a, c_rej, "A>C_hedge_beats_overclaim:rejected_included")
    return sft, sft_short, pairs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in", dest="src", required=True)
    parser.add_argument("--out", dest="dst", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--min-supported", type=int, default=1,
        help="bỏ ảnh có ít hơn ngần này mệnh đề SUPPORTED (0 = giữ hết)",
    )
    parser.add_argument(
        "--budget", type=int, default=None,
        help=" (nhật ký NC): ghi đè ngân sách lựa chọn (mặc định 5 của công thức chính "
             "thức). Nâng lên để caption giám sát giữ nhiều mệnh đề SUPPORTED "
             "hơn — nút vặn chi tiết/rủi ro của formulation/10 §3.1.",
    )
    parser.add_argument(
        "--detailed-variant", choices=("A", "B"), default="A",
        help=" (nhật ký NC) (quyết định nhóm 22/08): 'B' xuất bậc IM LẶNG (chỉ mệnh đề "
             "SUPPORTED, không từ phòng hộ) làm giám sát chi tiết thay bậc A. "
             "Công thức chính thức hiện hành giữ mặc định 'A'.",
    )
    parser.add_argument(
        "--style-variation", action="store_true",
        help=" (nhật ký NC): xoay khuôn kết xuất kho-đóng (mở câu + từ nối) tất định "
             "theo image_id — cùng sự thật, văn đa dạng hơn.",
    )
    parser.add_argument(
        "--clean-props", action="store_true",
        help=" (nhật ký NC): lọc mệnh đề rác trước build (lặp từ liền kề · predicate "
             "nuốt nguyên mệnh đề · trùng văn bản trong ảnh).",
    )
    parser.add_argument(
        "--no-verification", action="store_true",
        help="ABLATION P2 : coi MỌI mệnh đề là SUPPORTED — chưng cất "
             "không qua kiểm chứng. A trùng B nên không tồn tại cặp ưu tiên "
             "nào: dpo.jsonl sẽ RỖNG, và đó chính là kết quả — không có phán "
             "quyết thì không có bậc ưu tiên để học.",
    )
    args = parser.parse_args()
    if args.budget is not None:
        SELECT_BUDGET[0] = args.budget
        print(f"⚠ ngân sách lựa chọn ghi đè: {args.budget} (chính thức: 5)")
    if args.style_variation:
        STYLE_VARIATION[0] = True
        print("⚠ (nhật ký NC): khuôn kết xuất xoay (style_variation) BẬT")
    if args.clean_props:
        CLEAN_PROPS[0] = True
        print("⚠ (nhật ký NC): lọc mệnh đề rác (clean_props) BẬT")
    if args.no_verification:
        print("⚠⚠ CHẾ ĐỘ KHÔNG KIỂM CHỨNG (ablation P2) — mọi mệnh đề coi là "
              "SUPPORTED, dữ liệu này KHÔNG được dùng cho mô hình chính ⚠⚠")

    src = Path(args.src)
    dst = Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)

    files = sorted(f for f in src.glob("*.json") if not f.name.startswith("_"))
    if not files:
        raise SystemExit(f"không thấy bản ghi giai đoạn 1 nào trong {src}")
    print(f"{len(files)} bản ghi giai đoạn 1")

    stats: Counter[str] = Counter()
    sft_rows: list[dict] = []
    dpo_rows: list[dict] = []

    for path in files:
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except Exception as e:
            stats[f"unreadable:{type(e).__name__}"] += 1
            continue
        if args.no_verification:
            record = dict(record)
            record["propositions"] = as_asserted(record.get("propositions") or [])
        n_supported = sum(1 for p in record.get("propositions") or []
                          if verdict_of(p) == "SUPPORTED")
        if n_supported < args.min_supported:
            stats["skipped_too_few_supported"] += 1
            continue
        sft, sft_short, pairs = build_for_image(record, stats)
        if args.detailed_variant == "B" and sft:
            # Bậc B nằm sẵn trong các cặp ưu tiên: rejected của A>B, hoặc
            # chosen của B>C. Không có cặp nào và ảnh không có mệnh đề
            # UNCERTAIN → bậc A vốn đã thuần SUPPORTED, giữ nguyên.
            b_text = next((p["rejected"] for p in pairs
                           if p["pair_type"].startswith("A>B")), None) or \
                     next((p["chosen"] for p in pairs
                           if p["pair_type"].startswith("B>C")), None)
            if b_text:
                sft = dict(sft, response=b_text, variant_detailed="B")
                stats["sft_detailed_B"] += 1
            elif sft.get("n_uncertain", 0) == 0:
                sft = dict(sft, variant_detailed="B_equals_A")
                stats["sft_detailed_B_equals_A"] += 1
            else:
                stats["skipped_no_rung_B"] += 1
                sft = None
        if args.no_verification:
            for row in ([sft] if sft else []) + ([sft_short] if sft_short else []):
                row["ablation"] = "no_verification"
        if sft:
            sft_rows.append(sft)
        if sft_short:
            sft_rows.append(sft_short)
        dpo_rows.extend(pairs)

    rng = random.Random(args.seed)
    rng.shuffle(sft_rows)
    rng.shuffle(dpo_rows)

    for name, rows in (("sft.jsonl", sft_rows), ("dpo.jsonl", dpo_rows)):
        with (dst / name).open("w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    report = {
        "source": str(src), "records": len(files),
        "ablation": "no_verification" if args.no_verification else None,
        "sft_examples": len(sft_rows), "dpo_pairs": len(dpo_rows),
        "counts": dict(stats.most_common()),
        "seed": args.seed,
    }
    (dst / "_build_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"\n  SFT  : {len(sft_rows):>6} ví dụ  -> {dst/'sft.jsonl'}")
    print(f"  DPO  : {len(dpo_rows):>6} cặp     -> {dst/'dpo.jsonl'}")
    print("\n  theo loại cặp:")
    for key, n in sorted(stats.items()):
        if key.startswith("pair:"):
            print(f"     {key[5:]:<48} {n:>6}")
    dropped = {k: v for k, v in stats.items()
               if k.startswith(("skipped", "pair_unavailable", "realise_failed", "unreadable"))}
    if dropped:
        print("\n  bỏ qua — nói rõ để không đọc nhầm là đã bao phủ hết:")
        for key, n in sorted(dropped.items()):
            print(f"     {key:<48} {n:>6}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
