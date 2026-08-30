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


#: (research log): selection budget for the supervision data. 5 = official recipe
#: (SelectionConfig default — the "detail/risk knob"). Override with
#: --budget to build the high-detail variant; the default keeps the old behaviour.
SELECT_BUDGET: list[int | None] = [None]
# (research log) (wording bundle): enable rotating rendering templates + junk proposition filtering.
STYLE_VARIATION: list[bool] = [False]
CLEAN_PROPS: list[bool] = [False]
CURRENT_STYLE_SEED: list[int] = [42]


def _malformed(text: str) -> bool:
    """Junk proposition per (research log): adjacent duplicated word, or a predicate swallowing a whole clause."""
    words = text.split()
    if any(a.lower() == b.lower() for a, b in zip(words, words[1:])):
        return True
    return False


def clean_props(props: list[dict], stats: Counter) -> list[dict]:
    """Filter junk before the build ( (research log) #2): duplicated words · overlong predicate · duplicates."""
    seen: set[str] = set()
    kept: list[dict] = []
    for p in props:
        text = str(p.get("text_vi") or "")
        pred = str(((p.get("predicate") or {}).get("lemma_vi")) or "")
        if _malformed(text):
            stats["clean:dup_word"] += 1
            continue
        # M2 bug trace (audit (research log), e.g. 4962 P23/P35): a predicate swallowing a
        # whole clause WITH ITS OWN SUBJECT ("một người đang nắm chặt..."). A long predicate
        # that does not open with an indefinite noun phrase + đang/đã is valid, keep it.
        import re as _re
        _emb = r"(?:^|\s)(một số|một vài|nhiều|vài|các|những|một|hai|ba|bốn|năm)\s+(?:\S+\s+){1,6}?(đang|đã)\s"
        m_pred = _re.search(_emb, " " + pred)
        m_text = _re.search(_emb, text)
        # a match at the START of the text is the valid subject of an action
        # proposition — only catch an indefinite NP + đang/đã EMBEDDED MID-string
        # (the M2 clause-swallowing bug trace).
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

# (research log): agree the clothing verb with its noun — closed-inventory rule, truth-
# preserving (đội mũ/nón · đeo kính/túi/đồng hồ/khẩu trang · đi giày/dép). Trace: 10/6403
# v2 supervision examples still had "mặc mũ"; wide-B output 7240 "mặc một chiếc mũ".
_WEAR_FIXES = (
    (_re_verb.compile(r"\bmặc(\s+(?:một|hai|vài|nhiều)?\s*(?:chiếc|cái)?\s*)(mũ|nón)\b"), r"đội\1\2"),
    (_re_verb.compile(r"\bmặc(\s+(?:một|hai|vài|nhiều)?\s*(?:chiếc|cái)?\s*)(kính|túi|đồng hồ|khẩu trang)\b"), r"đeo\1\2"),
    (_re_verb.compile(r"\bmặc(\s+(?:một|hai|vài|nhiều)?\s*(?:đôi|chiếc)?\s*)(giày|dép)\b"), r"đi\1\2"),
)


_PRONOUN_REMNANT = _re_verb.compile(r"\b(cô|anh|chị|ông|bà|em|họ)\s+ấy\b")


def fix_wear_verbs(text: str) -> str:
    for pat, rep in _WEAR_FIXES:
        text = pat.sub(rep, text)
    # Pronoun remnants after neutralization ( (research log)b): the subject has become
    # "người" but the proposition text (including propositions anchored on a CLOTHING
    # entity) still has "cô ấy" → caption "một người ... cô ấy mặc ...". Drop every
    # repeated pronoun — Vietnamese allows dropping a repeated subject (per rendering
    # rule §6), and the supervision style avoids pronouns anyway (rule 7 forbids
    # opening a sentence with one).
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
        # v2 (lesson from bvan42-v1): rotating templates for the detailed style ONLY —
        # short mode must keep the concise KTVIC-style prose; the "Bức ảnh cho thấy"
        # template tanks BLEU-4.
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
        # (research log)b: an entity neutralized by backoff → every proposition hanging
        # off it must drop the repeated gendered pronoun in its text ("... cô ấy mặc
        # quần trắng ..."), lest the caption become "một người ... cô ấy ...".
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
        help="skip images with fewer SUPPORTED propositions than this (0 = keep all)",
    )
    parser.add_argument(
        "--budget", type=int, default=None,
        help=" (research log): override the selection budget (the official recipe's "
             "default of 5). Raise it so supervision captions keep more SUPPORTED "
             "propositions — the detail/risk knob of formulation/10 §3.1.",
    )
    parser.add_argument(
        "--detailed-variant", choices=("A", "B"), default="A",
        help=" (research log) (team decision 22/08): 'B' emits the SILENT rung (SUPPORTED "
             "propositions only, no hedging words) as the detailed supervision instead of "
             "rung A. The current official recipe keeps the default 'A'.",
    )
    parser.add_argument(
        "--style-variation", action="store_true",
        help=" (research log): rotate closed-inventory rendering templates (sentence "
             "openers + connectives) deterministically by image_id — same facts, more "
             "varied prose.",
    )
    parser.add_argument(
        "--clean-props", action="store_true",
        help=" (research log): filter junk propositions before the build (adjacent "
             "duplicated words · predicate swallowing a whole clause · duplicate text "
             "within an image).",
    )
    parser.add_argument(
        "--no-verification", action="store_true",
        help="ABLATION P2 : treat EVERY proposition as SUPPORTED — distillation "
             "without verification. A equals B so no preference pair exists: "
             "dpo.jsonl will be EMPTY, and that is exactly the result — without "
             "verdicts there are no preference rungs to learn.",
    )
    args = parser.parse_args()
    if args.budget is not None:
        SELECT_BUDGET[0] = args.budget
        print(f"⚠ selection budget overridden: {args.budget} (official: 5)")
    if args.style_variation:
        STYLE_VARIATION[0] = True
        print("⚠ (research log): rotating rendering templates (style_variation) ON")
    if args.clean_props:
        CLEAN_PROPS[0] = True
        print("⚠ (research log): junk proposition filtering (clean_props) ON")
    if args.no_verification:
        print("⚠⚠ NO-VERIFICATION MODE (ablation P2) — every proposition treated as "
              "SUPPORTED, this data must NOT be used for the main model ⚠⚠")

    src = Path(args.src)
    dst = Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)

    files = sorted(f for f in src.glob("*.json") if not f.name.startswith("_"))
    if not files:
        raise SystemExit(f"no Stage 1 records found in {src}")
    print(f"{len(files)} Stage 1 records")

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
            # Rung B already lives inside the preference pairs: the rejected of A>B,
            # or the chosen of B>C. No such pair and the image has no UNCERTAIN
            # proposition → rung A was already pure SUPPORTED, keep it.
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

    print(f"\n  SFT  : {len(sft_rows):>6} examples -> {dst/'sft.jsonl'}")
    print(f"  DPO  : {len(dpo_rows):>6} pairs    -> {dst/'dpo.jsonl'}")
    print("\n  by pair type:")
    for key, n in sorted(stats.items()):
        if key.startswith("pair:"):
            print(f"     {key[5:]:<48} {n:>6}")
    dropped = {k: v for k, v in stats.items()
               if k.startswith(("skipped", "pair_unavailable", "realise_failed", "unreadable"))}
    if dropped:
        print("\n  skipped — stated openly so this is not misread as full coverage:")
        for key, n in sorted(dropped.items()):
            print(f"     {key:<48} {n:>6}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
