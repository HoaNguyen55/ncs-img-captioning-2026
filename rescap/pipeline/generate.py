"""M2 — candidate proposition generation.

    Image  ──►  P_candidate = {p1 … pn}   (verification.status is always None)

Implements `formulation/08-MODULE-GENERATION.md`. Three passes rather than the
design's five, per `FAIR2026-PLAN.md` §4 — the scene and adversarial passes were
cut for the 10-day schedule, which lowers proposition recall and is reported as
a ceiling rather than hidden.

**The invariant this module exists to protect:** the generator *proposes*, it
never *asserts*. Every proposition leaves here with `verification.status = None`
and an epistemic tag that caps what verdict it can later receive. A generator
that could mark its own output SUPPORTED would make verification decorative.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Sequence

from ..vi.classifier import classifier_for, parse_noun_phrase
from ..vi.color import needs_disambiguation
from ..vi.lexicon import GENDERED_NOUNS, NON_VISUAL_MARKERS, SPATIAL_RELATIONS
from ..vlm.base import VLM

# ---------------------------------------------------------------------------
# Vietnamese prompts. Issued in Vietnamese on purpose (formulation/08 §3.1):
# surface forms then enter the representation Vietnamese-native rather than
# translated, which is what H5 tests.
# ---------------------------------------------------------------------------
PROMPT_ENTITIES = (
    "Liệt kê tất cả các đối tượng nhìn thấy được trong ảnh. "
    "Mỗi đối tượng một dòng, chỉ ghi danh từ kèm loại từ, ví dụ: "
    "'một người đàn ông', 'hai con chó', 'một chiếc xe đạp'. "
    "Không mô tả, không suy đoán."
)

#: Which attributes may be asked of which entity. Asking every entity about
#: every attribute is not a harmless waste of probes -- it manufactures
#: hallucinations the verifier cannot catch.
#:
#: Measured on a real image: `trang phục` (clothing) was asked of `cánh đồng`
#: (a field), `ánh sáng` (sunlight) and `cây` (trees). The model answered
#: helpfully with the clothing it could actually see -- the women's white áo
#: dài -- and attached it to whichever entity had been asked about. The field's
#: clothing then verified as SUPPORTED, because the probe `is there a white áo
#: dài in the image` is TRUE. A nonsensical question invites a plausible answer,
#: and no amount of verification recovers from it: the claim is checkable and
#: the check passes.
ATTRIBUTES_BY_ANIMACY: dict[str, tuple[str, ...]] = {
    # people: clothing and posture are the point; material is not
    "người": ("màu sắc", "trang phục", "tư thế"),
    # animals: posture yes, clothing no
    "động vật": ("màu sắc", "kích thước", "tư thế"),
    # everything else: no clothing, no posture
    "vật": ("màu sắc", "kích thước", "chất liệu"),
    # mass/abstract referents -- light, sky, scenery. Size and material are as
    # meaningless here as clothing is on a field.
    "phi vật thể": ("màu sắc",),
}

PROMPT_ATTRIBUTES = (
    "Mô tả các đặc điểm nhìn thấy được của {subject} trong ảnh: {aspects}. "
    "CHỈ mô tả {subject}, không mô tả vật khác trong ảnh. "
    "Mỗi đặc điểm một dòng. Nếu màu là 'xanh', phải ghi rõ 'xanh dương' hoặc "
    "'xanh lá'. Nếu không nhìn rõ, ghi 'không rõ'."
)

PROMPT_ACTIONS = (
    "{subject} trong ảnh đang làm gì? Chỉ mô tả hành động NHÌN THẤY được. "
    "Không đoán mục đích, nghề nghiệp hay cảm xúc. Mỗi hành động một dòng."
)

PROMPT_RELATION = (
    "Quan hệ giữa {a} và {b} trong ảnh là gì? "
    "Nêu quan hệ vị trí (trên, dưới, bên cạnh, phía sau, gần) "
    "hoặc hành động (cầm, mặc, đẩy). Một dòng một quan hệ. "
    "Nếu không rõ, ghi 'không rõ'."
)


class Epistemic:
    """What kind of claim this is (formulation/08 §2)."""

    OBSERVATION = "OBSERVATION"
    INFERENCE = "INFERENCE"
    SPECULATION = "SPECULATION"


@dataclass
class GenerationStats:
    """Reported alongside the propositions -- silence about truncation reads as
    full coverage (formulation/07 §6)."""

    entities: int = 0
    probes: int = 0
    by_type: dict[str, int] = field(default_factory=dict)
    by_epistemic: dict[str, int] = field(default_factory=dict)
    truncations: list[str] = field(default_factory=list)
    parse_degraded: int = 0


_NOISE = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s*")

#: `Màu sắc: đen và nâu` — a short leading label, not part of the claim.
#: Bounded to four words so a real sentence containing a colon is not chopped.
_LABEL = re.compile(r"^([^:：]{1,30}?)\s*[:：]\s*(.+)$")

#: The model's own attribute labels, mapped onto the schema's kinds.
_ATTRIBUTE_KINDS = {
    "màu sắc": "màu_sắc", "màu": "màu_sắc",
    "kích thước": "kích_thước", "kích cỡ": "kích_thước",
    "chất liệu": "chất_liệu", "vật liệu": "chất_liệu",
    "trang phục": "trang_phục", "quần áo": "trang_phục",
    "tư thế": "tư_thế", "dáng": "tư_thế",
}

#: A value carrying no positive visual claim. Checked against the value, after
#: the label is removed: `Trang phục: không có` reads as informative until the
#: `Trang phục:` comes off, and then it plainly is not.
_NO_CLAIM = (
    "không rõ", "không xác định", "không biết", "không thấy", "không nhìn",
    "không có", "không được", "không thể", "chưa rõ", "chưa xác định",
    "n/a", "không áp dụng", "không liên quan",
)


def _lines(text: str) -> list[tuple[str | None, str]]:
    """Split a model answer into `(label, value)` candidate items.

    Instruction-tuned models answer a list prompt with labelled lines --
    `Màu sắc: Đen và nâu`. Keeping the label made the proposition read
    `một chiếc thuyền Màu sắc: Đen và nâu`, which then became the probe
    question `Trong ảnh, một chiếc thuyền có Màu sắc: Đen và nâu không` --
    not a Vietnamese sentence, so the verifier was answering something we did
    not mean to ask.

    The label also hid the no-claim filter, which matched the start of the
    line. `Trang phục: Không có trang phục trên thuyền` survived as a
    proposition asserting a boat's clothing. Filtering the value catches it.

    >>> _lines("Màu sắc: Đen và nâu")
    [('Màu sắc', 'Đen và nâu')]
    >>> _lines("- Chất liệu: Gỗ\\n- Trang phục: Không có")
    [('Chất liệu', 'Gỗ')]
    >>> _lines("người đàn ông đang đi bộ")
    [(None, 'người đàn ông đang đi bộ')]
    """
    out: list[tuple[str | None, str]] = []
    for raw in text.splitlines():
        line = _NOISE.sub("", raw).strip(" .;")
        if not line:
            continue

        label = None
        match = _LABEL.match(line)
        if match and len(match.group(1).split()) <= 4:
            label, line = match.group(1).strip(), match.group(2).strip(" .;")

        lowered = line.lower()
        if len(line) < 2 or any(lowered.startswith(m) for m in _NO_CLAIM):
            continue
        out.append((label, line))
    return out


#: Proposition types whose claim IS the entity's identity. Only these can be
#: made uncertain by an uncertain gender.
IDENTITY_TYPES = frozenset({"entity", "counting"})


def classify_epistemic(text: str, ptype: str | None = None) -> tuple[str, str]:
    """Tag the claim type. Returns `(epistemic, inference_type)`.

    This tag is **not** a verdict — it caps what verdict verification may later
    assign. A purpose claim can never become a visual fact however confident the
    model sounds (formulation/09 §5.1).

    **The gender rule applies to identity claims only** (research log ).
    `có một người phụ nữ` rests on inference; `áo dài màu trắng` about that same
    person does not -- the colour is visible whatever her gender. Applying it to
    both capped 1,263 propositions the image had already confirmed, 12% of
    everything generated, and took supported-per-image from 6.9 down to 4.1.

    Nothing is asserted about gender as a result: `realize.py` swaps `phụ nữ`
    for `người` when the gender is unverified, so the caption never states it.
    Verification declining to believe the colour as well was suppression twice
    over.
    """
    lowered = text.lower()
    for marker, kind in NON_VISUAL_MARKERS.items():
        if marker in lowered:
            return Epistemic.SPECULATION, kind
    if ptype is None or ptype in IDENTITY_TYPES:
        for noun, _ in GENDERED_NOUNS.items():
            if noun in lowered:
                # Gender asserted from a caption is inference until the verifier
                # finds visible evidence (formulation/02 §4.3).
                return Epistemic.INFERENCE, "none"
    return Epistemic.OBSERVATION, "none"


def _proposition(
    pid: str,
    ptype: str,
    subject: dict | None,
    text: str,
    *,
    predicate: str | None = None,
    obj: dict | None = None,
    attributes: list[dict] | None = None,
    spatial: dict | None = None,
    generator: str = "",
    confidence: float = 0.0,
) -> dict[str, Any]:
    epistemic, inference_type = classify_epistemic(text, ptype)
    return {
        "id": pid,
        "type": ptype,
        "subject": subject or {"text_vi": "", "is_entity": False},
        "predicate": {"lemma_vi": predicate or "có"},
        "object": obj,
        "attributes": attributes or [],
        "spatial_relation": spatial,
        "text_vi": text,
        "confidence": confidence,
        "generator": generator,
        "epistemic": epistemic,
        "evidence": {
            "external_knowledge": {
                "requires_inference": epistemic != Epistemic.OBSERVATION,
                "inference_type": inference_type,
            }
        },
        # INVARIANT: the generator asserts nothing.
        "verification": {"status": None},
    }


def _strip_diacritics(text: str) -> str:
    """Vietnamese text to bare ASCII letters, for use in identifiers.

    Decomposition rather than a hand-written translation table. Vietnamese has
    67 precomposed letters once every tone mark is counted, and the two strings
    a `str.maketrans` needs must match in length exactly — the earlier version
    here was two characters out and raised `ValueError` on the first accented
    head noun a real model produced. NFD decomposes each letter into its base
    plus combining marks, which are then dropped, so there is no length to keep
    in step.

    `đ` is the exception: it is its own letter, not `d` plus a mark, so NFD
    leaves it intact and it is mapped explicitly.

    >>> _strip_diacritics("người phụ nữ")
    'nguoi phu nu'
    >>> _strip_diacritics("chiếc xe đạp")
    'chiec xe dap'
    """
    lowered = text.lower().replace("đ", "d")
    decomposed = unicodedata.normalize("NFD", lowered)
    return "".join(c for c in decomposed if not unicodedata.combining(c))


def _entity_id(head_noun: str, index: int) -> str:
    ascii_name = re.sub(r"[^a-z]+", "_", _strip_diacritics(head_noun)).strip("_")
    return f"{ascii_name or 'entity'}_{index}"


def build_entities(model: VLM, image: Any, stats: GenerationStats) -> list[dict]:
    """Pass G2 — the entity registry. Everything else references it by id."""
    answer = model.describe(image, PROMPT_ENTITIES, max_new_tokens=200)
    entities: list[dict] = []
    for _label, line in _lines(answer.text):
        phrase = parse_noun_phrase(line)
        head = phrase.head_noun
        if not head:
            stats.parse_degraded += 1
            continue
        classifier, known = classifier_for(head)
        if not known:
            stats.truncations.append(f"unknown classifier for {head!r}")
        gender = GENDERED_NOUNS.get(head)
        entities.append(
            {
                "id": _entity_id(head, len(entities) + 1),
                "category_vi": head,
                "classifier": phrase.classifier or classifier,
                "number": {
                    "value": phrase.numeral,
                    "marker": phrase.number_marker,
                    "countable": True,
                },
                # Neutral by default. Guessing gender is a hallucination, so the
                # generator records only what the surface form claimed and lets
                # verification decide (formulation/02 §4.3).
                "gender": {
                    "value": gender or "khong_xac_dinh",
                    "evidence": "not_determinable" if not gender else "inferred_from_clothing",
                },
                "confidence": answer.confidence,
                "detector": model.name,
                "surface": line,
            }
        )
    stats.entities = len(entities)
    return entities


def generate(
    model: VLM,
    image: Any,
    *,
    max_entities: int = 12,
    max_pairs: int = 12,
    self_consistency_k: int = 1,
) -> tuple[list[dict], list[dict], GenerationStats]:
    """Run the candidate-generation passes.

    Returns `(entities, propositions, stats)`. Every proposition has
    `verification.status is None` — asserted by the caller in `assert_clean`.
    """
    stats = GenerationStats()
    entities = build_entities(model, image, stats)

    if len(entities) > max_entities:
        stats.truncations.append(
            f"truncated to {max_entities}/{len(entities)} objects — coverage is NOT complete"
        )
        entities = entities[:max_entities]

    props: list[dict] = []
    counter = [0]

    def new_id() -> str:
        counter[0] += 1
        return f"P{counter[0]}"

    for entity in entities:
        ref = {
            "entity_id": entity["id"],
            "text_vi": entity["surface"],
            "head_noun_vi": entity["category_vi"],
        }

        # existence
        props.append(
            _proposition(
                new_id(), "entity", ref, entity["surface"],
                predicate="có", generator=model.name, confidence=entity["confidence"],
            )
        )
        # count, when the model committed to a number
        if entity["number"].get("value"):
            p = _proposition(
                new_id(), "counting", ref,
                f"{entity['surface']}", generator=model.name,
            )
            p["count"] = {
                "value": entity["number"]["value"],
                "classifier": entity["classifier"],
                "entity_category_vi": entity["category_vi"],
                "exact": True,
            }
            props.append(p)

        # G3 — attributes
        animacy = animacy_of(entity["category_vi"])
        aspects = ATTRIBUTES_BY_ANIMACY[animacy]
        answer = model.probe(
            image,
            PROMPT_ATTRIBUTES.format(
                subject=entity["surface"], aspects=", ".join(aspects)
            ),
            k=self_consistency_k,
            max_new_tokens=120,
        )
        stats.probes += 1
        for label, line in _lines(answer.text):
            # The label is the model's own name for the attribute; using it
            # beats guessing `trạng_thái` for everything that is not a colour.
            kind = _ATTRIBUTE_KINDS.get((label or "").lower().strip())
            if kind is None:
                kind = "màu_sắc" if _is_color(line) else "trạng_thái"
            # An attribute value is folded into the middle of a sentence, so
            # it must not carry the capital the model put at the start of its
            # line: `có vẻ Đứng trên một khung gỗ`. Only the first character is
            # lowered, so `áo dài truyền thống Việt Nam` keeps its proper noun.
            #
            # `value_vi` is also the value alone -- `đen và nâu`, not `màu đen
            # và nâu`. Both the question builder and the realiser prepend their
            # own `màu`; storing it here too produced `màu màu đen và nâu`.
            # The attribute pass echoes the subject too: PROMPT_ATTRIBUTES names
            # it, so the answer comes back `bảng hiệu có màu vàng` and the
            # caption read `Có một cái bảng hiệu bảng hiệu có màu vàng`. Same
            # strip as the action and relation passes -- this was the last of
            # the four places `_lines` feeds.
            value = _decapitalise(strip_subject_echo(line, entity))
            attribute = {"kind": kind, "value_vi": value}
            if kind == "màu_sắc" and needs_disambiguation(line):
                attribute["color_disambiguation"] = {
                    "raw": line, "resolved": "xanh_không_xác_định",
                    "resolution_source": "unresolved",
                }
            props.append(
                _proposition(
                    new_id(), "attribute", ref,
                    f"{entity['surface']} {value}",
                    predicate="có", attributes=[attribute],
                    generator=model.name, confidence=answer.confidence,
                )
            )

        # G3 — actions
        answer = model.probe(
            image,
            PROMPT_ACTIONS.format(subject=entity["surface"]),
            k=self_consistency_k,
            max_new_tokens=120,
        )
        stats.probes += 1
        for _label, line in _lines(answer.text):
            action = _decapitalise(strip_subject_echo(line, entity))
            props.append(
                _proposition(
                    new_id(), "action", ref,
                    f"{entity['surface']} {action}",
                    predicate=action, generator=model.name,
                    confidence=answer.confidence,
                )
            )

    # G4 — pairwise relations, truncated and logged
    pairs = [
        (a, b)
        for i, a in enumerate(entities)
        for b in entities[i + 1 :]
    ]
    if len(pairs) > max_pairs:
        stats.truncations.append(
            f"truncated to {max_pairs}/{len(pairs)} relation pairs — coverage is NOT complete"
        )
        pairs = pairs[:max_pairs]

    for a, b in pairs:
        answer = model.probe(
            image,
            PROMPT_RELATION.format(a=a["surface"], b=b["surface"]),
            k=self_consistency_k,
            max_new_tokens=80,
        )
        stats.probes += 1
        aref = {"entity_id": a["id"], "text_vi": a["surface"], "head_noun_vi": a["category_vi"]}
        bref = {"entity_id": b["id"], "text_vi": b["surface"], "head_noun_vi": b["category_vi"]}
        for _label, line in _lines(answer.text):
            # Same echo the action pass strips: PROMPT_RELATION names both
            # entities, so the model restates them and the proposition came out
            # `một người phụ nữ Người phụ nữ đứng phía trước một số đồ`.
            # Stripped for BOTH ends, since either can be echoed.
            line = _decapitalise(strip_subject_echo(strip_subject_echo(line, a), b))
            relation = _spatial_term(line)
            if relation:
                p = _proposition(
                    new_id(), "spatial_relation", aref,
                    f"{a['surface']} {line} {b['surface']}",
                    obj=bref, generator=model.name, confidence=answer.confidence,
                )
                p["spatial_relation"] = {
                    "relation_vi": relation,
                    # 2-D images: default to the viewer's frame, and say so
                    # rather than leaving it ambiguous (formulation/02 §4.6).
                    "frame_of_reference": "image_relative",
                }
                props.append(p)
            else:
                props.append(
                    _proposition(
                        new_id(), "relation", aref,
                        f"{a['surface']} {line} {b['surface']}",
                        predicate=line, obj=bref,
                        generator=model.name, confidence=answer.confidence,
                    )
                )

    for p in props:
        stats.by_type[p["type"]] = stats.by_type.get(p["type"], 0) + 1
        stats.by_epistemic[p["epistemic"]] = stats.by_epistemic.get(p["epistemic"], 0) + 1

    assert_clean(props)
    return entities, props, stats


#: Referents with no shape or substance of their own.
_INCORPOREAL = frozenset({
    "ánh sáng", "ánh nắng", "nắng", "bóng", "bóng râm", "bầu trời", "trời",
    "mây", "sương", "khói", "gió", "mưa", "không khí", "cảnh", "phong cảnh",
    "khung cảnh", "nền", "hậu cảnh", "tiền cảnh", "màu", "ánh",
})


def animacy_of(head_noun: str) -> str:
    """Which attribute set this entity can be asked about.

    Derived from the classifier the lexicon assigns, because Vietnamese
    classifiers already encode the distinction: `người` for humans, `con` for
    animals. That is the language doing the work rather than a second
    hand-maintained list drifting out of step with the first.

    >>> animacy_of("phụ nữ")
    'người'
    >>> animacy_of("chó")
    'động vật'
    >>> animacy_of("ánh sáng")
    'phi vật thể'
    >>> animacy_of("xe đạp")
    'vật'
    """
    from ..vi.lexicon import GENDERED_NOUNS, NOUN_CLASSIFIER

    noun = head_noun.strip().lower()
    if noun in _INCORPOREAL:
        return "phi vật thể"
    if noun in GENDERED_NOUNS or NOUN_CLASSIFIER.get(noun) in ("người", "cô", "chàng", "bé", "em", "đứa"):
        return "người"
    if NOUN_CLASSIFIER.get(noun) == "con":
        return "động vật"
    return "vật"


def _decapitalise(text: str) -> str:
    """Lower a line-initial capital so the value can sit inside a sentence.

    Only the first character, and only when the word is not written all in
    capitals -- an acronym should survive intact.

    >>> _decapitalise("Áo dài trắng")
    'áo dài trắng'
    >>> _decapitalise("áo dài truyền thống Việt Nam")
    'áo dài truyền thống Việt Nam'
    >>> _decapitalise("ATM")
    'ATM'
    """
    if not text:
        return text
    first = text.split()[0]
    if len(first) > 1 and first.isupper():
        return text
    return text[0].lower() + text[1:]


def strip_subject_echo(line: str, entity: dict) -> str:
    """Drop the subject the model repeated back at the start of its answer.

    `PROMPT_ACTIONS` asks "{subject} trong ảnh đang làm gì?" and an
    instruction-tuned model answers helpfully by restating it: for
    `một chiếc xe máy` it returns `Xe máy đang đỗ trên đường`. The caption
    planner then prepends the subject again, giving
    `Một chiếc xe máy Xe máy đang đỗ trên đường`.

    Only a prefix is removed, and only when it matches this entity's own head
    noun or surface -- a second mention later in the clause is real content
    (`xe máy đỗ cạnh một xe máy khác`) and stays.

    >>> e = {"category_vi": "xe máy", "surface": "một chiếc xe máy"}
    >>> strip_subject_echo("Xe máy đang đỗ trên đường", e)
    'đang đỗ trên đường'
    >>> strip_subject_echo("đang đỗ trên đường", e)
    'đang đỗ trên đường'
    >>> strip_subject_echo("Xe máy đỗ cạnh một xe máy khác", e)
    'đỗ cạnh một xe máy khác'
    >>> g = {"category_vi": "người", "surface": "nhiều người"}
    >>> strip_subject_echo("một số người đang cầm sản phẩm", g)
    'đang cầm sản phẩm'
    """
    head = str(entity.get("category_vi") or "").strip().lower()
    surface = str(entity.get("surface") or "").strip().lower()
    candidates = [c for c in (surface, head) if c]
    # Also the surface without its leading numeral/classifier: the answer echoes
    # `xe máy`, not `một chiếc xe máy`.
    for c in list(candidates):
        parts = c.split()
        for cut in (1, 2):
            if len(parts) > cut:
                candidates.append(" ".join(parts[cut:]))

    lowered = line.lower()
    best = ""
    for c in candidates:
        if len(c) > len(best) and lowered.startswith(c):
            best = c

    if not best and head:
        # Surface match is not enough: the model refers to one group as
        # `nhiều người` and then `một số người`, and neither string is a prefix
        # of the other, so the echo survived and the caption read
        # `Nhiều người nhiều người một số người đang cầm sản phẩm`.
        #
        # Compare HEAD NOUNS instead. A leading noun phrase whose head is this
        # entity's head is the same referent said differently, whatever
        # quantifier it carries.
        phrase = parse_noun_phrase(line)
        if phrase.head_noun and phrase.head_noun.lower() == head:
            marker = phrase.head_noun.lower()
            cut = lowered.find(marker)
            if 0 <= cut <= 24:  # a leading NP, not a mention deep in the clause
                rest = line[cut + len(marker):].lstrip(" ,.:;-–—")
                if rest:
                    return rest

    if not best:
        return line
    rest = line[len(best):].lstrip(" ,.:;-–—")
    # Never strip down to nothing: a bare echo carries no action, but returning
    # an empty string would silently drop the proposition instead of showing it.
    return rest or line


def _is_color(text: str) -> bool:
    """Is this attribute value a colour?

    Requires the colour word near the start. Matching anywhere in the string
    made `kích thước của cây cỏ và cây xanh: Không rõ` -- a size answer -- come
    back tagged as a colour, because `xanh` appears inside the entity's own
    name.
    """
    from ..vi.lexicon import COLORS

    head = " ".join(text.lower().split()[:3])
    return any(c in head for c in COLORS) or "xanh" in head or head.startswith("màu")


def _spatial_term(text: str) -> str | None:
    lowered = " ".join(text.lower().split())
    for term in sorted(SPATIAL_RELATIONS, key=len, reverse=True):
        if term.replace("_", " ") in lowered:
            return term
    return None


def assert_clean(props: Sequence[dict]) -> None:
    """The generator must not have asserted anything.

    A hard error, not a warning: a proposition arriving at verification already
    marked SUPPORTED would bypass the entire mechanism this project is about.
    """
    guilty = [
        p["id"] for p in props if (p.get("verification") or {}).get("status") is not None
    ]
    if guilty:
        raise RuntimeError(
            "generator asserted a verdict, which it must never do: " + ", ".join(guilty)
        )
