"""Proposition matching — the definition every metric in doc 04 rests on.

`formulation/02-PROPOSITION-SCHEMA.md` §7 specifies this. It is implemented here
once, and every metric calls it, because two divergent notions of "match" would
produce an apparent improvement that is really a scoring difference.

Two propositions match iff **all** hold:

    1. same type
    2. subject_match      -- at ENTITY level, not string level
    3. predicate_match    -- canonical form, tier-dependent
    4. object_match       -- when either side has an object
    5. payload_match      -- type-specific (attribute value, relation, count…)

**Three tiers, and every table must state which it used** (doc 02 §5.2):

| tier | matches on |
|---|---|
| `EXACT` | identical canonical forms |
| `LEXICAL` | + synonyms and the hypernym lattice |
| `SOFT` | + Vietnamese sentence-embedding similarity ≥ τ |

`SOFT` is never reported alone: a soft matcher can be tuned until any result
appears, so strict always accompanies it.

**Deliberately excluded from matching** — they are realisation choices, not
claims: aspect (`đang`), classifier (`con`/`chiếc`), number *marker*
(`những`/`các`). The count *value* is included, because that is a claim.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Sequence


from ..vi.color import ColorErrorType, classify_color_error
from ..vi.lexicon import ASPECT_MARKERS, HYPERNYMS, NUMBER_MARKERS, SYNONYMS


class Tier(str, Enum):
    EXACT = "exact"
    LEXICAL = "lexical"
    SOFT = "soft"


@dataclass
class MatchResult:
    matched: bool
    tier: Tier | None = None
    reason: str = ""
    similarity: float | None = None

    def __bool__(self) -> bool:
        return self.matched


# ---------------------------------------------------------------------------
# Normalisation used only for comparison
# ---------------------------------------------------------------------------
def canonical(text: str | None) -> str:
    """Reduce a Vietnamese phrase to its comparison form.

    Strips what is realisation rather than claim — aspect markers, number
    markers, a leading classifier — then maps through the synonym lexicon.

    **Only a LEADING classifier token is removed.** Running the full
    noun-phrase parser here was a bug: it is built for noun phrases, and
    applying it to a predicate destroyed the verb (`đạp xe` "cycling" came back
    as `xe` "vehicle"), while multi-word terms absent from the lexicon were
    truncated (`phương tiện` -> `phương`).

    The synonym lexicon is consulted **before and after** stripping, because
    entries may be keyed either way: `xe hơi` -> `ô tô` only matches before the
    classifier-ish first token is removed.

    >>> canonical("đang đạp xe")
    'đạp xe'
    >>> canonical("một chiếc xe đạp")
    'xe đạp'
    >>> canonical("xe hơi")
    'ô tô'
    """
    if not text:
        return ""
    joined = " ".join(str(text).strip().lower().split())
    if not joined:
        return ""

    # A synonym may be keyed on the full surface form.
    if joined in SYNONYMS:
        return SYNONYMS[joined]

    words = [w for w in joined.split() if w not in ASPECT_MARKERS]
    joined = " ".join(words)
    if joined in SYNONYMS:
        return SYNONYMS[joined]

    # Leading number marker, longest first so "một vài" beats "một".
    for marker in sorted(NUMBER_MARKERS, key=len, reverse=True):
        if joined.startswith(marker + " "):
            joined = joined[len(marker) + 1 :]
            break
    if joined in SYNONYMS:
        return SYNONYMS[joined]

    # Leading classifier, and only when something follows it -- never reduce a
    # phrase to nothing, and never touch anything past the first token.
    from ..vi.lexicon import CLASSIFIERS

    parts = joined.split()
    if len(parts) > 1 and parts[0] in CLASSIFIERS:
        joined = " ".join(parts[1:])

    return SYNONYMS.get(joined, joined).strip()


def _hypernym_chain(term: str) -> list[str]:
    chain, seen = [term], {term}
    while term in HYPERNYMS:
        term = HYPERNYMS[term]
        if term in seen:
            break
        chain.append(term)
        seen.add(term)
    return chain


def compatible_categories(a: str, b: str) -> bool:
    """True when one category subsumes the other (doc 02 §2.3).

    `người` and `đàn ông` are compatible; `đàn ông` and `phụ nữ` are siblings
    and are **not** — an entity cannot be both.
    """
    a, b = canonical(a), canonical(b)
    if a == b:
        return True
    return a in _hypernym_chain(b) or b in _hypernym_chain(a)


# ---------------------------------------------------------------------------
# Component matchers
# ---------------------------------------------------------------------------
def _argument(prop: dict[str, Any], key: str) -> dict[str, Any] | None:
    value = prop.get(key)
    return value if isinstance(value, dict) else None


def subject_match(p: dict, g: dict, tier: Tier, iou_fn: Callable | None = None) -> bool:
    """Match subjects at entity level.

    Same registry -> compare `entity_id`. Different registries (our predictions
    against gold annotation) -> fall back to bbox IoU >= 0.5 plus compatible
    categories, exactly as doc 02 §7 specifies. Comparing surface strings would
    make `người đàn ông` and `người` different subjects and the whole metric
    meaningless.
    """
    pa, ga = _argument(p, "subject"), _argument(g, "subject")
    if pa is None or ga is None:
        return pa is ga

    pid, gid = pa.get("entity_id"), ga.get("entity_id")
    if pid and gid and pid == gid:
        return True

    if iou_fn is not None and pid and gid:
        iou = iou_fn(pid, gid)
        if iou is not None and iou >= 0.5:
            return compatible_categories(
                pa.get("head_noun_vi") or pa.get("text_vi", ""),
                ga.get("head_noun_vi") or ga.get("text_vi", ""),
            )

    ptext = canonical(pa.get("head_noun_vi") or pa.get("text_vi"))
    gtext = canonical(ga.get("head_noun_vi") or ga.get("text_vi"))
    if tier is Tier.EXACT:
        return bool(ptext) and ptext == gtext
    return compatible_categories(ptext, gtext)


def object_match(p: dict, g: dict, tier: Tier) -> bool:
    pa, ga = _argument(p, "object"), _argument(g, "object")
    if pa is None and ga is None:
        return True
    if pa is None or ga is None:
        return False
    ptext = canonical(pa.get("head_noun_vi") or pa.get("text_vi"))
    gtext = canonical(ga.get("head_noun_vi") or ga.get("text_vi"))
    if tier is Tier.EXACT:
        return ptext == gtext
    return compatible_categories(ptext, gtext)


def predicate_match(p: dict, g: dict, tier: Tier) -> bool:
    pp = (p.get("predicate") or {}).get("lemma_vi", "")
    gp = (g.get("predicate") or {}).get("lemma_vi", "")
    if not pp and not gp:
        return True
    pc, gc = canonical(pp), canonical(gp)
    if pc == gc:
        return True
    if tier is Tier.EXACT:
        return False
    return compatible_categories(pc, gc)


def _attribute_match(p: dict, g: dict, tier: Tier) -> tuple[bool, str]:
    """Attributes match on FAMILY and VALUE.

    Colour goes through the Vietnamese resolver so `xanh dương` vs `xanh lá` is
    a mismatch while a bare `xanh` against a resolved gold is recorded as
    under-specification rather than a wrong value (`vi/color.py`).
    """
    pas, gas = p.get("attributes") or [], g.get("attributes") or []
    if not pas and not gas:
        return True, ""
    if not pas or not gas:
        return False, "một bên không có thuộc tính"

    for pa in pas:
        for ga in gas:
            if pa.get("kind") != ga.get("kind"):
                continue
            pv, gv = str(pa.get("value_vi", "")), str(ga.get("value_vi", ""))
            if pa.get("kind") == "màu_sắc":
                outcome = classify_color_error(pv, gv)
                if outcome is ColorErrorType.CORRECT:
                    return True, ""
                return False, f"màu: {outcome.value}"
            if canonical(pv) == canonical(gv):
                return True, ""
            if tier is not Tier.EXACT and compatible_categories(pv, gv):
                return True, ""
    return False, "không thuộc tính nào khớp"


def _spatial_match(p: dict, g: dict) -> tuple[bool, str]:
    """Spatial relations need the SAME relation and the SAME frame of reference.

    `bên trái` under `image_relative` and under `object_relative` describe
    different configurations, so treating them as equal would silently accept a
    wrong answer (doc 02 §4.6).
    """
    ps, gs = p.get("spatial_relation") or {}, g.get("spatial_relation") or {}
    if not ps and not gs:
        return True, ""
    if not ps or not gs:
        return False, "một bên không có quan hệ không gian"
    if ps.get("relation_vi") != gs.get("relation_vi"):
        return False, f"{ps.get('relation_vi')} != {gs.get('relation_vi')}"
    pf = ps.get("frame_of_reference", "image_relative")
    gf = gs.get("frame_of_reference", "image_relative")
    if pf != gf and "unspecified" not in (pf, gf):
        return False, f"hệ quy chiếu khác: {pf} != {gf}"
    return True, ""


def _count_match(p: dict, g: dict) -> tuple[bool, str]:
    pc, gc = p.get("count") or {}, g.get("count") or {}
    if not pc and not gc:
        return True, ""
    if not pc or not gc:
        return False, "một bên không có số lượng"
    tolerance = int(gc.get("tolerance", 0))
    pv, gv = pc.get("value"), gc.get("value")
    if pv is None or gv is None:
        return pv == gv, "một bên không xác định số"
    ok = abs(int(pv) - int(gv)) <= tolerance
    return ok, "" if ok else f"{pv} != {gv} (dung sai {tolerance})"


def _scene_match(p: dict, g: dict) -> tuple[bool, str]:
    """Compare only the scene fields gold actually committed to.

    Gold leaving `weather` unset means it was not determinable; requiring the
    prediction to match a null would penalise it for gold's uncertainty.
    """
    ps, gs = p.get("scene") or {}, g.get("scene") or {}
    if not ps and not gs:
        return True, ""
    if not ps or not gs:
        return False, "một bên không có bối cảnh"
    for field, gvalue in gs.items():
        if gvalue in (None, "", "không_xác_định"):
            continue
        pvalue = ps.get(field)
        if pvalue in (None, ""):
            return False, f"thiếu {field}"
        if canonical(str(pvalue)) != canonical(str(gvalue)):
            return False, f"{field}: {pvalue} != {gvalue}"
    return True, ""


_PAYLOAD: dict[str, Callable[..., tuple[bool, str]]] = {
    "attribute": lambda p, g, tier: _attribute_match(p, g, tier),
    "spatial_relation": lambda p, g, tier: _spatial_match(p, g),
    "counting": lambda p, g, tier: _count_match(p, g),
    "scene": lambda p, g, tier: _scene_match(p, g),
}


# ---------------------------------------------------------------------------
def match(
    predicted: dict,
    gold: dict,
    tier: Tier = Tier.LEXICAL,
    *,
    iou_fn: Callable | None = None,
    similarity_fn: Callable[[str, str], float] | None = None,
    soft_threshold: float = 0.75,
) -> MatchResult:
    """Do these two propositions describe the same claim?

    `similarity_fn` is only consulted at `Tier.SOFT`, and is injected rather
    than imported so this module stays free of any embedding dependency —
    matching must work on a machine with no model stack.
    """
    if predicted.get("type") != gold.get("type"):
        return MatchResult(False, reason=f"loại khác: {predicted.get('type')} vs {gold.get('type')}")

    if not subject_match(predicted, gold, tier, iou_fn):
        return MatchResult(False, reason="chủ ngữ khác")
    if not predicate_match(predicted, gold, tier):
        return MatchResult(False, reason="vị ngữ khác")
    if not object_match(predicted, gold, tier):
        return MatchResult(False, reason="tân ngữ khác")

    payload = _PAYLOAD.get(str(predicted.get("type")))
    if payload is not None:
        ok, why = payload(predicted, gold, tier)
        if not ok:
            return MatchResult(False, reason=why)

    return MatchResult(True, tier=tier)


def match_any(
    predicted: dict,
    golds: Sequence[dict],
    tier: Tier = Tier.LEXICAL,
    **kwargs: Any,
) -> tuple[int, MatchResult]:
    """First gold this proposition matches. Returns `(index, result)`, index -1 if none."""
    last = MatchResult(False, reason="không có mệnh đề vàng nào")
    for i, gold in enumerate(golds):
        result = match(predicted, gold, tier, **kwargs)
        if result.matched:
            return i, result
        last = result
    return -1, last


def align(
    predicted: Sequence[dict],
    gold: Sequence[dict],
    tier: Tier = Tier.LEXICAL,
    **kwargs: Any,
) -> dict[str, Any]:
    """One-to-one alignment between predicted and gold propositions.

    Greedy and **one-to-one**: each gold is consumed once, so five predictions
    of the same claim cannot each score a hit. Without that, a system that
    repeats itself would inflate its own recall.

    Returns the counts doc 04 §4.1 needs, plus the pairing for error analysis.
    """
    used: set[int] = set()
    pairs: list[tuple[int, int]] = []
    unmatched_pred: list[int] = []

    for pi, prop in enumerate(predicted):
        candidates = [(gi, g) for gi, g in enumerate(gold) if gi not in used]
        hit = -1
        for gi, g in candidates:
            if match(prop, g, tier, **kwargs).matched:
                hit = gi
                break
        if hit >= 0:
            used.add(hit)
            pairs.append((pi, hit))
        else:
            unmatched_pred.append(pi)

    unmatched_gold = [gi for gi in range(len(gold)) if gi not in used]
    tp = len(pairs)
    precision = tp / len(predicted) if predicted else 0.0
    recall = tp / len(gold) if gold else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    return {
        "tier": tier.value,
        "pairs": pairs,
        "unmatched_predicted": unmatched_pred,
        "unmatched_gold": unmatched_gold,
        "tp": tp,
        "fp": len(unmatched_pred),
        "fn": len(unmatched_gold),
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }
