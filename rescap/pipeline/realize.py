"""M6 + M7 — Vietnamese caption planning and grounded realisation.

    P*  ──►  [① order → ② group → ③ refer/link]  ──►  [④ realise → ⑤ enforce]  ──►  GeneratedCaption

Implements `formulation/11-MODULE-CAPTION.md`. Stages ①–③ are the
proposition-to-text planner (M6, §3–§5); ④–⑤ are realisation (M7, §6–§7).

**The invariant this module exists to protect:** Facts(C) ⊆ P*. The realiser may
choose *how* to say things — order, connectives, pronouns, aspect — but may not
introduce *what* is said (§1). Enforcement (§7) is what turns that from an
instruction into a guarantee, and `ungrounded_spans` is what makes a failure to
enforce visible instead of silent.

Planning is symbolic and deterministic, which is the whole reason the provenance
record is exact rather than reconstructed: the planner knows which proposition
each span realises because it put it there (§2, §7.1 tier 1). Only stage ④ may
involve a language model, and its output is re-aligned against the planner
record before anything ships.

Two contract points where doc 11 and `configs/proposition_schema.json` disagree,
resolved here in favour of the schema because it is the data contract every
other module reads:

1. **Pure connectives are not emitted as spans.** §1.1 lists `và`, `là`,
   punctuation as `is_factual: false` material, but `GeneratedCaption.spans[]`
   requires `proposition_ids` with `minItems: 1`, and a bare connective realises
   no proposition. Doc 11 §10's own worked output agrees: it spans six fragments
   and omits `là`, the comma and every space. Such material is recorded on the
   plan and on `RealizationResult.exempt_spans` — the grounding detector needs
   those ranges to know they carry no claim — but stays out of the caption.
   A connective *is* emitted when it is bundled with the clause it introduces,
   with that clause's proposition id and `is_factual=false`.
2. **Everything outside `GeneratedCaption` stays outside it.** The object is
   `additionalProperties: false`, so stats, plan warnings and prevented
   hallucinations travel on `RealizationResult`, and only enforcement facts that
   fit `decoding` (`additionalProperties: true`) are stored in the caption.

Ablation A5 (`grounding_constraint=false`) disables the repair loop — no retry,
no strip, no template fallback. Alignment still runs: measuring is not
enforcing, `ungrounded_spans` is a reported result (§7), and suppressing it
would make A5 look artificially clean. Note that doc 04 §5.4 computes A5's claim
set with a parser, not with this record, so nothing here inflates that column.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Iterable, Sequence

from ..vi.classifier import build_noun_phrase, classifier_for
from ..vi.color import Xanh, parse_color
from ..vi.lexicon import (
    ASPECT_MARKERS,
    CLASSIFIERS,
    COLORS,
    GENDERED_NOUNS,
    NEUTRAL_PERSON,
    NOUN_CLASSIFIER,
    NUMBER_MARKERS,
    NUMERALS,
    PRONOUNS,
    PRONOUN_PLURAL,
)
from ..vlm.base import VLM

# ---------------------------------------------------------------------------
# Vietnamese surface inventory used by realisation
# ---------------------------------------------------------------------------
# Hedge markers for UNCERTAIN content admitted under the quota (doc 10 §6).
# A hedge is not a claim, so every span it governs is `is_factual: false` and
# therefore cannot inflate PGF (doc 04 §5.2).
HEDGE_MARKER = "có vẻ như"
HEDGE_MARKERS: tuple[str, ...] = ("có vẻ như", "có vẻ", "dường như", "trông giống", "khoảng")

# Connectives — the only tokens the planner may add (§5.4b). None asserts
# anything; `còn` in particular marks a topic switch, which is the transition
# this pipeline makes constantly.
CONNECTIVE_TOPIC_SHIFT = "còn"
CONNECTIVE_COORDINATION = "và"
COPULA_EXISTENTIAL = "là"
DISCOURSE_CONNECTIVES: tuple[str, ...] = (
    "bên cạnh đó", "trong khi đó", "trong khi", "còn", "và", "ngoài ra",
    "trong ảnh", "bức ảnh cho thấy", "cùng với đó",
)

# §5.4b extended ( (research log)): the template variants form ONE CLOSED STORE — none
# asserts anything extra about the image CONTENT ("trong ảnh"/"bức ảnh cho thấy" only
# assert that this is a picture). Rotated DETERMINISTICALLY by cfg.seed to reproduce.
STYLE_OPENERS: tuple[str, ...] = ("có", "trong ảnh có", "bức ảnh cho thấy")
STYLE_SHIFT_CONNECTIVES: tuple[str, ...] = ("còn", "bên cạnh đó", "ngoài ra", "cùng với đó")


def _style_pick(options: tuple[str, ...], cfg: "RealizeConfig", salt: int) -> str:
    """Deterministic style rotation — identical (seed, salt) → identical pick."""
    if not getattr(cfg, "style_variation", False):
        return options[0]
    return options[(int(getattr(cfg, "seed", 0)) * 31 + salt * 7) % len(options)]

# §11: P* = ∅ must never be filled with invention.
EMPTY_CAPTION_VI = "Không thể mô tả chi tiết bức ảnh này."

# Ordering policy of §3, as a sort key over proposition types.
_TYPE_ORDER: dict[str, int] = {
    "entity": 0,
    "counting": 1,
    "attribute": 2,
    "action": 3,
    "interaction": 4,
    "relation": 5,
    "spatial_relation": 6,
    "scene": 7,
}

# Verbal predicates that take their own object, so the attribute realises as a
# clause (`mặc áo đỏ`) rather than folding into the noun phrase (`ô tô màu
# trắng`). Anything else with a contentful predicate is handled generically.
_COPULAR_PREDICATES = {"có", "là", "mang", ""}

# Function words that carry no claim (§1.1). Used ONLY by the ungrounded-span
# detector: text left over after span alignment that consists entirely of these
# asserts nothing and must not be reported as a violation. Kept deliberately
# tight — spatial words (`trên`, `bên cạnh`, `phía sau`) are claims, not glue,
# and putting them here would silently exempt the hardest proposition type.
#
# KNOWN LOOSENESS, stated rather than hidden. The detector tokenises on
# whitespace, so the multi-word entries below can never match a token and are
# inert; the bare `người`, `anh`, `cô` do fire. They are here to stop an
# unplanned referring expression (`người này`, realising an entity the planner
# already introduced) being reported as a fresh claim — but the cost is that an
# LM which invents a SECOND, unpropositioned person realised as `một người`
# slips past, while `một phụ nữ` would be caught. The detector therefore
# under-reports bare-`người` additions, and the measured violation rate is a
# lower bound on that one pattern.
#
# A NUMERAL IS A CLAIM (§9.1 Count) and is deliberately absent below. Exempting
# the numerals made an invented count invisible: against the planner span
# `một chiếc ô tô`, the caption `ba chiếc ô tô` anchored on `ô tô` and left
# `ba` unreported, so a counting hallucination cost nothing. `một` stays exempt
# via NUMBER_MARKERS — there it is the indefinite article introducing one
# referent, not a count — and so does `không`, which in a caption is the
# negator far more often than the numeral zero.
_EXEMPT_WORDS: frozenset[str] = frozenset(
    set(CLASSIFIERS)
    | set(NUMBER_MARKERS)
    | {"không"}
    | set(ASPECT_MARKERS)
    | {
        "là", "và", "còn", "với", "của", "có", "cùng", "thì", "cũng", "rất",
        "một", "này", "đó", "ấy", "kia", "các", "được", "bị", "mà", "nhưng",
        # (research log): purely GRAMMATICAL glue — asserts no content at all.
        # Locative words (trên/dưới/trong/ngoài/giữa/phía/bên/ở) are excluded
        # ON PURPOSE: they are spatial claims and need a supporting proposition.
        "để", "từ", "đến", "khi", "lúc", "vì", "nên", "cho", "về", "theo",
        "như", "bằng", "đều", "cả", "nhau", "khác", "khá", "hơi", "lên",
        "xuống", "ra", "vào", "đi", "lại", "nữa", "rồi", "vừa", "hay",
        "hoặc", "trong đó", "sau đó", "ngoài ra", "bên cạnh đó",
        "anh ấy", "cô ấy", "ông ấy", "bà ấy", "người này", "người đó", "họ",
        "anh", "cô", "ông", "bà", "người",
    }
)

# Determiners: grammatically bound to the noun that follows them, so they are
# removed together with an ungrounded noun rather than left dangling.
_DETERMINERS: frozenset[str] = frozenset(set(CLASSIFIERS) | set(NUMBER_MARKERS) | set(NUMERALS))

# Gendered third-person forms. Emitting one without `evidence: clearly_visible`
# asserts something the image does not show (§5.2, §9.1).
_GENDERED_PRONOUNS: tuple[str, ...] = ("anh ấy", "cô ấy", "ông ấy", "bà ấy", "chị ấy")

# Nouns whose first syllable is spelled like a pronoun.
_PRONOUN_COMPOUNDS: dict[str, frozenset[str]] = {
    "họ": frozenset({"hàng", "tên"}),        # họ hàng = relatives, họ tên = full name
    "cô": frozenset({"gái", "giáo", "dâu"}),
    "anh": frozenset({"em", "trai"}),
    "bà": frozenset({"cụ", "ngoại", "nội"}),
    "ông": frozenset({"cụ", "ngoại", "nội"}),
}

# Age-marked address forms carry an age claim of their own (§5.2).
_AGE_MARKED_NOUNS: frozenset[str] = frozenset({"ông", "bà", "cụ", "lão"})

_PUNCTUATION = ".,;:!?…"


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass
class RealizeConfig:
    """Knobs of `generation_caption` in the pipeline config (doc 07 §7).

    Detail is NOT set here: it is set upstream by the selection budget B
    (§8). `max_chars` is a hard ceiling for the output medium, not a detail dial
    — exceeding it drops a proposition and re-realises rather than truncating
    mid-clause (§11).
    """

    strategy: str = "constrained_lm"          # constrained_lm | template
    grounding_constraint: bool = True         # False ← Ablation A5
    max_retries: int = 2
    fallback: str = "template"
    max_attributes_per_np: int = 2            # §4: a third becomes its own clause
    max_propositions_per_sentence: int = 4    # §4 rule of thumb: 2–4
    max_sentences: int = 3
    max_topic_shifts: int = 2                 # §5.5 constraint 2
    pronoun_distance_clauses: int = 2         # §5.3
    max_chars: int | None = 320
    no_strip: bool = False        # (research log): no mid-phrase cuts — clean or back to template
    temperature: float = 0.3
    seed: int = 42
    soft_threshold: float = 0.75              # §7.1 tier 3, baselines only
    style_variation: bool = False             # (research log): rotate closed-store templates by seed


# ---------------------------------------------------------------------------
# Plan representation
# ---------------------------------------------------------------------------
@dataclass
class PlannedSpan:
    """A caption fragment together with the proposition it realises.

    `char_start`/`char_end` are None until a realiser places the span in a
    string. `alignment` records HOW provenance was established (§7.1): the
    planner record is exact, the lexical anchor validates it, and `semantic` is
    a degraded tier that exists only for unconstrained baselines.
    """

    text: str
    proposition_ids: list[str]
    span_role: str
    is_factual: bool = True
    entity_id: str | None = None
    char_start: int | None = None
    char_end: int | None = None
    alignment: str = "planner"                # planner | lexical | semantic | none

    def as_schema(self) -> dict[str, Any]:
        """Project onto `GeneratedCaption.spans[]` — no extra keys, the object is
        `additionalProperties: false`."""
        if self.char_start is None or self.char_end is None:
            raise RuntimeError(
                f"span {self.text!r} has no character offsets; provenance would be "
                "unusable and a fabricated offset is worse than none"
            )
        return {
            "text": self.text,
            "char_start": self.char_start,
            "char_end": self.char_end,
            "proposition_ids": list(self.proposition_ids),
            "span_role": self.span_role,
            "is_factual": self.is_factual,
        }


@dataclass
class ExemptToken:
    """Material that realises no proposition: connectives, `là`, punctuation.

    Exempt from grounding (§1.1) and — because `proposition_ids` needs at least
    one entry — kept out of `GeneratedCaption.spans`. The detector still needs
    these ranges: without them every connective the planner emitted would be
    reported as an ungrounded claim.
    """

    text: str
    role: str                                  # discourse_connective | copula | punctuation
    glue: str = " "
    char_start: int | None = None
    char_end: int | None = None


Piece = PlannedSpan | ExemptToken


@dataclass
class Clause:
    """One clause of the discourse plan."""

    pieces: list[Piece] = field(default_factory=list)
    topic_entity_id: str | None = None
    subject_elided: bool = False
    starts_sentence: bool = False

    @property
    def spans(self) -> list[PlannedSpan]:
        return [p for p in self.pieces if isinstance(p, PlannedSpan)]

    @property
    def proposition_ids(self) -> list[str]:
        out: list[str] = []
        for span in self.spans:
            for pid in span.proposition_ids:
                if pid not in out:
                    out.append(pid)
        return out


@dataclass
class DiscoursePlan:
    """Output of M6 — stages ①–③."""

    clauses: list[Clause] = field(default_factory=list)
    ordered_ids: list[str] = field(default_factory=list)
    topic_order: list[str] = field(default_factory=list)
    hedged_ids: list[str] = field(default_factory=list)
    unplanned_ids: list[str] = field(default_factory=list)
    topic_shifts: int = 0
    warnings: list[str] = field(default_factory=list)

    @property
    def spans(self) -> list[PlannedSpan]:
        return [s for clause in self.clauses for s in clause.spans]


@dataclass
class RealizationStats:
    """Reported alongside the caption. Silence about a repair reads as a clean
    run, which is exactly the misreading doc 11 §7 forbids.

    Two scopes, and the distinction is not cosmetic. **Run-level** counters
    measure the whole effort of producing one caption (retries, regenerations,
    propositions dropped for length). **Caption-level** counters describe the
    caption that actually ships, and are reset by `begin_caption()` at the start
    of every planning attempt: the length-drop loop re-plans from scratch, so
    accumulating across discarded attempts would publish a prevented-
    hallucination rate two or three times the real one — and that rate is a
    reported result (§9.1, §11).
    """

    strategy: str = "template"
    # -- run-level -----------------------------------------------------------
    retries: int = 0
    regenerations: int = 0
    non_vietnamese_retries: int = 0
    enforcement_enabled: bool = True
    dropped_for_length: list[str] = field(default_factory=list)
    prompt_hash: str | None = None
    # -- caption-level -------------------------------------------------------
    stripped_spans: int = 0
    fell_back_to_template: bool = False
    alignment_degraded: bool = False
    ungrounded_count: int = 0
    topic_shifts: int = 0
    elisions: int = 0
    prevented_gender_hallucinations: int = 0
    prevented_age_claims: int = 0
    classifier_corrections: int = 0
    adjective_order_corrections: int = 0
    xanh_resolved: int = 0
    xanh_hedged: int = 0
    unrealised_ids: list[str] = field(default_factory=list)
    #: Propositions selection admitted as UNCERTAIN whose hedge marker is absent
    #: from the shipped text. The caption then ASSERTS uncertain content, which
    #: §9 layer 3 forbids, and nothing else in the record would show it.
    unhedged_uncertain: list[str] = field(default_factory=list)
    lexicon_warnings: list[str] = field(default_factory=list)
    #: How each shipped span got its provenance (§7.1): planner / lexical /
    #: semantic. doc 04 §5.4 credits the Full model with exact, parser-free
    #: spans; a span recovered by lexical anchor is NOT one, and publishing the
    #: breakdown is what keeps that disclosure honest.
    spans_by_alignment: dict[str, int] = field(default_factory=dict)

    # Deduplication keys. A guard that fires once per entity must be counted
    # once per entity: `_safe_head_noun` is called again whenever that entity is
    # referred to, and counting each call would inflate the rate.
    _guarded_gender: set[str] = field(default_factory=set, repr=False, compare=False)
    _guarded_age: set[str] = field(default_factory=set, repr=False, compare=False)
    _guarded_classifier: set[str] = field(default_factory=set, repr=False, compare=False)
    _seen_colors: set[str] = field(default_factory=set, repr=False, compare=False)

    def begin_caption(self) -> None:
        """Reset the caption-level counters before a fresh planning attempt."""
        self.stripped_spans = 0
        self.fell_back_to_template = False
        self.alignment_degraded = False
        self.ungrounded_count = 0
        self.topic_shifts = 0
        self.elisions = 0
        self.prevented_gender_hallucinations = 0
        self.prevented_age_claims = 0
        self.classifier_corrections = 0
        self.adjective_order_corrections = 0
        self.xanh_resolved = 0
        self.xanh_hedged = 0
        self.unrealised_ids = []
        self.unhedged_uncertain = []
        self.lexicon_warnings = []
        self.spans_by_alignment = {}
        self._guarded_gender.clear()
        self._guarded_age.clear()
        self._guarded_classifier.clear()
        self._seen_colors.clear()


@dataclass
class RealizationResult:
    """Everything M7 produced.

    `caption` is schema-valid on its own; anything the schema has no field for
    lives here rather than being smuggled into the caption object.
    """

    caption: dict[str, Any]
    plan: DiscoursePlan
    stats: RealizationStats
    log: list[str] = field(default_factory=list)
    exempt_spans: list[dict[str, Any]] = field(default_factory=list)

    @property
    def text_vi(self) -> str:
        return self.caption["text_vi"]

    @property
    def ungrounded_spans(self) -> list[str]:
        return list(self.caption.get("ungrounded_spans") or [])


# ---------------------------------------------------------------------------
# Text assembly — the only place character offsets are ever produced
# ---------------------------------------------------------------------------
class _Builder:
    """Accumulates the caption and stamps every piece with its offsets.

    Provenance is bookkeeping, not parsing (§7.1), and this class is where the
    bookkeeping happens: a span's offsets are recorded at the moment its text is
    appended, so `text[char_start:char_end] == text` holds by construction.
    Capitalisation is applied BEFORE the offsets are taken — capitalising
    afterwards would leave the recorded span text differing from the caption.
    """

    def __init__(self) -> None:
        self._parts: list[str] = []
        self._length = 0
        self.spans: list[PlannedSpan] = []
        self.exempt: list[ExemptToken] = []

    @property
    def length(self) -> int:
        return self._length

    def _append(self, text: str, glue: str) -> tuple[int, int]:
        if self._length and glue:
            self._parts.append(glue)
            self._length += len(glue)
        start = self._length
        self._parts.append(text)
        self._length += len(text)
        return start, self._length

    def add_span(self, span: PlannedSpan, *, glue: str = " ", capitalise: bool = False) -> PlannedSpan:
        text = _capitalise(span.text) if capitalise else span.text
        start, end = self._append(text, glue)
        placed = replace(span, text=text, char_start=start, char_end=end)
        self.spans.append(placed)
        return placed

    def add_exempt(self, token: ExemptToken, *, capitalise: bool = False) -> ExemptToken:
        text = _capitalise(token.text) if capitalise else token.text
        start, end = self._append(text, token.glue)
        placed = replace(token, text=text, char_start=start, char_end=end)
        self.exempt.append(placed)
        return placed

    def text(self) -> str:
        return "".join(self._parts)


def _capitalise(text: str) -> str:
    return text[:1].upper() + text[1:] if text else text


def nfc(text: str) -> str:
    """Normalise to NFC before any offset arithmetic.

    Vietnamese diacritics have both a precomposed and a decomposed encoding. A
    model that returns the decomposed form makes every `str.find` of a planner
    span miss and every recorded offset land in the wrong place — the provenance
    record would be wrong in a way no smoke test surfaces. Normalise once, at
    the boundary, before anything is measured.
    """
    return unicodedata.normalize("NFC", text)


# ---------------------------------------------------------------------------
# Entity-level helpers — the Vietnamese guards of §9.1
# ---------------------------------------------------------------------------
def _gender(entity: dict) -> tuple[str, str]:
    g = entity.get("gender") or {}
    return (
        str(g.get("value") or "khong_xac_dinh"),
        str(g.get("evidence") or "not_determinable"),
    )


def gender_verified(entity: dict) -> bool:
    """May this entity be referred to with a gendered noun or pronoun?

    Only `clearly_visible` counts. `inferred_from_clothing` is an inference, and
    in Vietnamese gender sits in the NOUN, so acting on it is a content error,
    not a style slip (§5.2, doc 02 §4.3).
    """
    value, evidence = _gender(entity)
    return value in ("nam", "nu") and evidence == "clearly_visible"


def _has_age_attribute(props: Sequence[dict], entity_id: str | None) -> bool:
    """Was an age band actually verified for this entity? (§9.1 Age)"""
    for p in props:
        subject = p.get("subject") or {}
        if entity_id is not None and subject.get("entity_id") != entity_id:
            continue
        for attribute in p.get("attributes") or []:
            if attribute.get("kind") == "độ_tuổi":
                return True
    return False


def _safe_head_noun(
    entity: dict,
    props: Sequence[dict],
    stats: RealizationStats,
    log: list[str],
) -> str:
    """The head noun this entity may be called, after the gender and age guards.

    A gendered noun without visible evidence is a hallucination, not a register
    choice, so the fallback is the neutral `người` (§9.1). This is the guard
    that makes the default output say `một người` where an ungoverned realiser
    would say `một người đàn ông`.
    """
    head = str(entity.get("category_vi") or "").strip().lower()
    if not head:
        return NEUTRAL_PERSON
    # `category_vi` is specified bare, but annotators write `người đàn ông`.
    bare = head[len(NEUTRAL_PERSON) + 1 :] if head.startswith(NEUTRAL_PERSON + " ") else head

    entity_key = str(entity.get("id") or bare)
    if bare in GENDERED_NOUNS and not gender_verified(entity):
        # Once per entity: this function runs again on every later mention, and
        # the count is published as a prevented-hallucination rate.
        if entity_key not in stats._guarded_gender:
            stats._guarded_gender.add(entity_key)
            stats.prevented_gender_hallucinations += 1
            log.append(
                f"prevented_gender_hallucination: {entity_key} {bare!r} -> {NEUTRAL_PERSON!r} "
                f"(evidence={_gender(entity)[1]})"
            )
        return NEUTRAL_PERSON
    if bare in _AGE_MARKED_NOUNS and not _has_age_attribute(props, entity.get("id")):
        if entity_key not in stats._guarded_age:
            stats._guarded_age.add(entity_key)
            stats.prevented_age_claims += 1
            log.append(f"prevented_age_claim: {entity_key} {bare!r} -> {NEUTRAL_PERSON!r}")
        return NEUTRAL_PERSON
    return bare


def _number_realisation(
    entity: dict,
    count_prop: dict | None,
) -> tuple[int | None, str | None, bool]:
    """Return `(count, marker, exact)` for the noun phrase.

    A numeral is a claim, so it is only asserted when a counting proposition
    survived selection and declared itself exact (§9.1 Count). Otherwise the
    approximate markers are used — `một vài`, `nhiều` — or the plain `một`,
    which here is the indefinite introduction of a single referent rather than a
    count claim.
    """
    number = entity.get("number") or {}
    if count_prop is not None:
        claim = count_prop.get("count") or {}
        value = claim.get("value")
        if value is not None and claim.get("exact", True):
            return int(value), None, True
        if value is not None:
            return int(value), None, False
    marker = number.get("marker")
    if marker:
        return None, str(marker).replace("_", " "), True
    value = number.get("value")
    if value in (None, 1):
        return None, "một", True
    return int(value), None, False


def _noun_phrase_text(
    head: str,
    count: int | None,
    marker: str | None,
    exact: bool,
    modifiers: Sequence[str],
    entity: dict,
    stats: RealizationStats,
) -> str:
    """Build NUMERAL + CLASSIFIER + NOUN + POST-nominal modifiers.

    Delegates to `vi.classifier.build_noun_phrase` so classifier selection and
    the post-nominal adjective rule have exactly one implementation, and so its
    warnings (unknown classifier) reach the stats instead of being lost.
    """
    phrase = build_noun_phrase(
        head, count=count, marker=marker, modifiers=list(modifiers), exact=exact
    )
    recorded = str(entity.get("classifier") or "").strip().lower()
    expected, known = classifier_for(head)
    entity_key = str(entity.get("id") or head)
    if known and recorded and recorded != expected:
        # A wrong classifier is a FLUENCY error, corrected from the lexicon and
        # counted — never treated as a factual violation (§11, doc 02 §4.1).
        # Counted once per entity: the same NP is rebuilt on later mentions.
        if entity_key not in stats._guarded_classifier:
            stats._guarded_classifier.add(entity_key)
            stats.classifier_corrections += 1
    elif not known and recorded:
        phrase.classifier = recorded

    # A classifier identical to (or a prefix of) its head noun would double the
    # noun: `một người người`, `một bé bé trai`. Vietnamese drops it there.
    if phrase.classifier and (
        phrase.classifier == phrase.head_noun
        or phrase.head_noun.startswith(phrase.classifier + " ")
    ):
        phrase.classifier = None

    stats.lexicon_warnings.extend(phrase.warnings)
    return phrase.surface()


def _definite_np(entity: dict, head: str) -> str:
    """`người đàn ông này`, `chiếc ô tô đó` — definite, so no numeral.

    The classifier stays: Vietnamese keeps it in a definite noun phrase, and
    dropping it yields the clipped `đàn ông này` that reads as translated text.
    """
    classifier, known = classifier_for(head)
    if not known:
        classifier = str(entity.get("classifier") or "").strip().lower()
    if classifier and (classifier == head or head.startswith(classifier + " ")):
        classifier = ""
    return " ".join(part for part in (classifier, head, "này") if part)


def _referring_expression(
    entity: dict,
    head: str,
    mention_index: int,
    clauses_since_mention: int,
    competitor_since: bool,
    cfg: RealizeConfig,
) -> tuple[str, str]:
    """Vietnamese referring expression for a repeat mention (§5).

    Returns `(text, form)` with form in {definite, pronoun}. First mentions are
    handled by the noun-phrase builder, not here: a new referent is introduced
    indefinitely, because a definite first mention presupposes knowledge the
    reader does not have (§5.4c).

    Follows the §5 table — 2nd mention definite, 3rd+ pronoun — with §5.3's
    distance limit overriding it back to a definite NP when the antecedent is
    far away or another entity of the same category has intervened. Vietnamese
    has no agreement to disambiguate competing antecedents, so an ambiguous
    pronoun is an ungrounded span: the reader cannot recover which proposition
    it realises.
    """
    definite = _definite_np(entity, head)
    if mention_index < 3:
        return definite, "definite"
    if clauses_since_mention > cfg.pronoun_distance_clauses or competitor_since:
        return definite, "definite"
    number = entity.get("number") or {}
    if (number.get("value") or 1) > 1:
        return PRONOUN_PLURAL, "pronoun"
    value, _ = _gender(entity)
    if gender_verified(entity):
        return PRONOUNS.get(value, PRONOUNS["khong_xac_dinh"]), "pronoun"
    # Neutral by default: a gendered pronoun here would assert unverified gender.
    return PRONOUNS["khong_xac_dinh"], "pronoun"


# ---------------------------------------------------------------------------
# ① Content ordering (§3)
# ---------------------------------------------------------------------------
@dataclass
class _Bucket:
    """Everything the plan knows about one entity."""

    entity_id: str
    entity: dict
    existence: dict | None = None
    count: dict | None = None
    np_attributes: list[dict] = field(default_factory=list)
    clausal_attributes: list[dict] = field(default_factory=list)
    actions: list[dict] = field(default_factory=list)
    relations: list[dict] = field(default_factory=list)
    salience: float | None = None

    @property
    def size(self) -> int:
        return (
            (1 if self.existence else 0)
            + (1 if self.count else 0)
            + len(self.np_attributes)
            + len(self.clausal_attributes)
            + len(self.actions)
            + len(self.relations)
        )


def _is_np_attribute(prop: dict) -> bool:
    """Does this attribute fold into the noun phrase, or become its own clause?

    `ô tô màu trắng` folds; `mặc áo đỏ` cannot, because it has its own verb and
    object. §3: attributes attach to the noun and precede the verb either way.
    """
    lemma = str((prop.get("predicate") or {}).get("lemma_vi") or "").strip().lower()
    if lemma not in _COPULAR_PREDICATES:
        return False
    return not (prop.get("object") or {})


def order_content(
    props: Sequence[dict],
    entities: Sequence[dict],
    cfg: RealizeConfig,
    stats: RealizationStats,
    log: list[str],
) -> tuple[list[_Bucket], list[dict], list[str]]:
    """Stage ① — Vietnamese information structure (§3).

    Returns `(buckets in topic order, scene propositions, unplanned ids)`.
    The policy is: topic entity, its attributes, its action, the setting, then
    secondary entities with their own attributes, then relations.
    """
    by_entity: dict[str, _Bucket] = {}
    registry = {str(e.get("id")): e for e in entities}
    scene_props: list[dict] = []
    unplanned: list[str] = []

    for prop in sorted(props, key=lambda p: (_TYPE_ORDER.get(str(p.get("type")), 9), _pid_key(p))):
        ptype = str(prop.get("type"))
        subject = prop.get("subject") or {}
        eid = subject.get("entity_id")
        if ptype == "scene" or (eid or "").startswith("scene"):
            scene_props.append(prop)
            continue
        if not eid or eid not in registry:
            # No entity to hang it on. Not realised, and said so — inventing a
            # referent to carry it would be exactly the failure this pipeline
            # is about.
            unplanned.append(str(prop.get("id")))
            continue
        bucket = by_entity.setdefault(eid, _Bucket(eid, registry[eid]))
        bucket.salience = registry[eid].get("salience")
        if ptype == "entity":
            if bucket.existence is None:
                bucket.existence = prop
            else:
                bucket.np_attributes.append(prop)
        elif ptype == "counting":
            bucket.count = prop
        elif ptype == "attribute":
            (bucket.np_attributes if _is_np_attribute(prop) else bucket.clausal_attributes).append(prop)
        elif ptype == "action":
            bucket.actions.append(prop)
        else:  # relation, spatial_relation, interaction
            bucket.relations.append(prop)

    if not by_entity:
        return [], scene_props, unplanned

    salient = [b for b in by_entity.values() if b.salience is not None]
    if not salient:
        log.append("topic_order_rule=proposition_count (no salience recorded on any entity)")
    ordered = sorted(
        by_entity.values(),
        # Salience first (§3); then how much P* has to say about the entity;
        # then id, so a tie never depends on dict ordering and runs reproduce.
        key=lambda b: (-(b.salience if b.salience is not None else 0.0), -b.size, b.entity_id),
    )
    stats.topic_shifts = max(0, len(ordered) - 1)
    return ordered, scene_props, unplanned


def _pid_key(prop: dict) -> tuple[int, str]:
    """Sort key over proposition ids: P2 before P10, deterministically."""
    pid = str(prop.get("id") or "")
    match = re.fullmatch(r"P(\d+)", pid)
    return (int(match.group(1)), pid) if match else (10**6, pid)


# ---------------------------------------------------------------------------
# Proposition → Vietnamese fragments
# ---------------------------------------------------------------------------
def _color_surface(
    attribute: dict,
    key: str,
    stats: RealizationStats,
    log: list[str],
) -> tuple[str, bool]:
    """Realise a colour value. Returns `(text, resolved)`.

    Bare `xanh` must never reach the output: it means both blue and green, and
    picking one here would manufacture a fact (§9.1 Colour, doc 02 §4.5). It is
    resolved from `color_disambiguation` when the verifier resolved it, and
    hedged otherwise — hedged spans are `is_factual: false`, so a vague colour
    cannot inflate PGF.
    """
    raw = str(attribute.get("value_vi") or "").strip()
    # Some extractors store the value WITH its nominal head (`màu vàng nhạt`).
    # Every caller supplies its own frame — `màu {surface}` or a bare modifier
    # — so a head kept here doubles: `màu màu vàng nhạt` in real output.
    if raw.startswith("màu "):
        raw = raw[len("màu "):].strip()
    reading = parse_color(raw)
    if reading.xanh_value is not Xanh.UNRESOLVED:
        return raw, True

    # Counted once per (proposition, colour): the same attribute is realised
    # again whenever the planner reconsiders the noun phrase it sits in.
    first_time = f"{key}:{raw}" not in stats._seen_colors
    stats._seen_colors.add(f"{key}:{raw}")

    disambiguation = attribute.get("color_disambiguation") or {}
    resolved = str(disambiguation.get("resolved") or "")
    if resolved in ("xanh_dương", "xanh_lá"):
        surface = resolved.replace("_", " ")
        if first_time:
            stats.xanh_resolved += 1
            log.append(
                f"xanh_resolved: {raw!r} -> {surface!r} via {disambiguation.get('resolution_source')}"
            )
        return surface, True

    if first_time:
        stats.xanh_hedged += 1
        log.append(f"xanh_hedged: {raw!r} left unresolved by verification, realised as a hedge")
    return raw, False


def _attribute_modifier(
    prop: dict,
    stats: RealizationStats,
    log: list[str],
) -> tuple[str, bool]:
    """A post-nominal modifier for an attribute that folds into the NP.

    Returns `(text, factual)`. Colour takes the nominal head `màu` here —
    `ô tô màu trắng` — because the bare adjective form is idiomatic on a
    garment inside a verb phrase but reads as clipped on a standalone referent.
    """
    parts: list[str] = []
    factual = True
    for attribute in prop.get("attributes") or []:
        if attribute.get("kind") == "màu_sắc":
            surface, resolved = _color_surface(attribute, str(prop.get("id")), stats, log)
            factual = factual and resolved
            parts.append(f"màu {surface}")
        else:
            value = str(attribute.get("value_vi") or "").strip()
            if value:
                parts.append(value)
    if not parts:
        # Fall back to the proposition's own canonical rendering rather than
        # dropping content that survived selection.
        parts.append(str(prop.get("text_vi") or "").strip())
    return " ".join(p for p in parts if p), factual


def _object_phrase(prop: dict, stats: RealizationStats, log: list[str]) -> tuple[str, bool]:
    """The object of a verbal attribute or action: `áo đỏ`, `xe đạp`.

    Bare on purpose — `mặc áo đỏ`, not `mặc một chiếc áo đỏ`. A numeral and
    classifier here would assert a count nobody verified.
    """
    obj = prop.get("object") or {}
    head = str(obj.get("head_noun_vi") or obj.get("text_vi") or "").strip()
    factual = True
    modifiers: list[str] = []
    for attribute in prop.get("attributes") or []:
        if attribute.get("kind") == "màu_sắc":
            surface, resolved = _color_surface(attribute, str(prop.get("id")), stats, log)
            factual = factual and resolved
            modifiers.append(surface)
        else:
            value = str(attribute.get("value_vi") or "").strip()
            if value:
                modifiers.append(value)
    # Adjectives FOLLOW the noun. This ordering is the rule, not a preference.
    return " ".join([p for p in [head] + modifiers if p]), factual


def _predicate_surface(prop: dict) -> str:
    """Verb phrase with its aspect marker.

    `đang` is emitted only for `tiếp_diễn`, or when the lemma already carries
    it. Aspect is a realisation choice and is never verified (doc 02 §4.4), so
    defaulting every action to the progressive would put an unasserted marker
    on every caption.
    """
    predicate = prop.get("predicate") or {}
    lemma = str(predicate.get("lemma_vi") or "").strip()
    if not lemma:
        return ""
    first = lemma.split()[0]
    if first in ASPECT_MARKERS:
        return lemma
    if predicate.get("aspect") == "tiếp_diễn":
        return f"đang {lemma}"
    return lemma


def _drop_leading_aspect(text: str) -> str:
    """Remove a repeated aspect marker in a serial-verb continuation.

    `đang ngồi trên ghế và đọc sách`, not `… và đang đọc sách`: the marker
    scopes over both verbs, and repeating it is a translation artefact.
    """
    parts = text.split(" ", 1)
    if len(parts) == 2 and parts[0] in ASPECT_MARKERS:
        return parts[1]
    return text


def _scene_surface(prop: dict) -> str:
    """Locative / setting phrase, realised clause-final (§3)."""
    scene = prop.get("scene") or {}
    place = str(scene.get("place_vi") or "").strip()
    if place:
        return place
    return str(prop.get("text_vi") or "").strip()


def _spatial_surface(prop: dict) -> str:
    relation = (prop.get("spatial_relation") or {}).get("relation_vi") or ""
    return str(relation).replace("_", " ").strip()


# ---------------------------------------------------------------------------
# ②③ Sentence planning, referring expressions, discourse linking (§4, §5)
# ---------------------------------------------------------------------------
class _Discourse:
    """Mention history — what makes referring expressions and ellipsis decidable."""

    def __init__(self) -> None:
        self.mentions: dict[str, int] = {}
        self.last_clause: dict[str, int] = {}
        self.categories_since: dict[str, list[str]] = {}
        self.clause_index = 0

    def note(self, entity_id: str, category: str) -> int:
        self.mentions[entity_id] = self.mentions.get(entity_id, 0) + 1
        self.last_clause[entity_id] = self.clause_index
        for other in self.categories_since:
            if other != entity_id:
                self.categories_since[other].append(category)
        self.categories_since[entity_id] = []
        return self.mentions[entity_id]

    def introduced(self, entity_id: str) -> bool:
        return entity_id in self.mentions

    def distance(self, entity_id: str) -> int:
        return self.clause_index - self.last_clause.get(entity_id, -99)

    def competitor(self, entity_id: str, category: str) -> bool:
        return category in self.categories_since.get(entity_id, [])


def plan_caption(
    props: Sequence[dict],
    entities: Sequence[dict],
    *,
    hedged_ids: Iterable[str] = (),
    config: RealizeConfig | None = None,
    stats: RealizationStats | None = None,
    log: list[str] | None = None,
) -> DiscoursePlan:
    """M6 — stages ①②③: order, group into clauses, choose referring expressions.

    Symbolic and deterministic (§2). Every span leaves here carrying the id of
    the proposition it realises, which is what makes the provenance record exact
    rather than reconstructed by a parser (§7.1 tier 1).
    """
    cfg = config or RealizeConfig()
    stats = stats if stats is not None else RealizationStats()
    log = log if log is not None else []
    hedged = {str(h) for h in hedged_ids}

    buckets, scene_props, unplanned = order_content(props, entities, cfg, stats, log)
    plan = DiscoursePlan(hedged_ids=sorted(hedged), unplanned_ids=list(unplanned))
    if not buckets and not scene_props:
        return plan

    discourse = _Discourse()
    props_in_sentence = 0
    sentence_count = 0
    topic_shifts = 0
    scene_used = False

    def hedge_for(prop: dict) -> bool:
        return str(prop.get("id")) in hedged

    def new_clause(topic: str | None, starts_sentence: bool) -> Clause:
        nonlocal props_in_sentence, sentence_count
        if starts_sentence:
            sentence_count += 1
            props_in_sentence = 0
        clause = Clause(topic_entity_id=topic, starts_sentence=starts_sentence)
        plan.clauses.append(clause)
        discourse.clause_index += 1
        return clause

    def emit(clause: Clause, span: PlannedSpan) -> None:
        nonlocal props_in_sentence
        clause.pieces.append(span)
        for pid in span.proposition_ids:
            if pid not in plan.ordered_ids:
                plan.ordered_ids.append(pid)
                props_in_sentence += 1

    for index, bucket in enumerate(buckets):
        entity = bucket.entity
        head = _safe_head_noun(entity, props, stats, log)
        anchor = bucket.existence or bucket.count
        if anchor is None and not (
            bucket.np_attributes or bucket.clausal_attributes or bucket.actions or bucket.relations
        ):
            continue

        # -- topic shift handling (§5.5 constraints 2 and 4) ------------------
        is_shift = index > 0
        if is_shift and topic_shifts >= cfg.max_topic_shifts:
            plan.warnings.append(
                f"skipping {bucket.entity_id}: more than {cfg.max_topic_shifts} topic shifts "
                "(§5.5 constraint 2)"
            )
            plan.unplanned_ids.extend(
                pid for pid in _bucket_ids(bucket) if pid not in plan.ordered_ids
            )
            continue

        fronted = _fronted_spatial(bucket, discourse) if is_shift else None
        starts_sentence = is_shift and (
            sentence_count < cfg.max_sentences
            and props_in_sentence >= cfg.max_propositions_per_sentence
        )
        clause = new_clause(bucket.entity_id, starts_sentence=starts_sentence or index == 0)
        if is_shift:
            topic_shifts += 1

        if is_shift and not clause.starts_sentence:
            clause.pieces.append(ExemptToken(",", "punctuation", glue=""))

        subject_role = "subject"
        if fronted is not None:
            # `phía sau anh là một chiếc ô tô` — the existential-locative is the
            # natural Vietnamese way to introduce a new referent at a known
            # location (§3, §4). The fronted phrase IS the spatial claim, so it
            # carries the spatial proposition's id rather than being exempt.
            other_id = str((fronted.get("object") or {}).get("entity_id"))
            other_entity = next((e for e in entities if str(e.get("id")) == other_id), {})
            other_head = _safe_head_noun(other_entity, props, stats, log)
            ref, _form = _referring_expression(
                other_entity,
                other_head,
                discourse.mentions.get(other_id, 1) + 1,
                discourse.distance(other_id),
                discourse.competitor(other_id, other_head),
                cfg,
            )
            discourse.note(other_id, other_head)
            emit(
                clause,
                PlannedSpan(
                    text=f"{_spatial_surface(fronted)} {ref}".strip(),
                    proposition_ids=[str(fronted.get("id"))],
                    span_role="spatial",
                    is_factual=not hedge_for(fronted),
                    entity_id=other_id,
                ),
            )
            clause.pieces.append(ExemptToken(COPULA_EXISTENTIAL, "copula"))
            bucket.relations = [r for r in bucket.relations if r is not fronted]
            subject_role = "object"
        elif is_shift:
            clause.pieces.append(ExemptToken(
                _style_pick(STYLE_SHIFT_CONNECTIVES, cfg, topic_shifts),
                "discourse_connective"))

        # An entity with nothing predicated of it needs the existential `có`:
        # a bare noun phrase is a fragment, not a Vietnamese sentence.
        if fronted is None and not (
            bucket.actions or bucket.clausal_attributes or bucket.relations
            or (index == 0 and scene_props)
        ):
            if index == 0:
                clause.pieces.append(ExemptToken(
                    _style_pick(STYLE_OPENERS, cfg, 0), "copula"))
            else:
                clause.pieces.append(ExemptToken("có", "copula"))

        # -- ③ referring expression for this entity --------------------------
        mention = discourse.mentions.get(bucket.entity_id, 0) + 1
        folded = bucket.np_attributes[: cfg.max_attributes_per_np]
        overflow = bucket.np_attributes[cfg.max_attributes_per_np :]

        if mention == 1:
            count, marker, exact = _number_realisation(entity, bucket.count)
            modifiers: list[str] = []
            hedged_modifiers: list[tuple[str, str]] = []
            np_ids = [str(anchor.get("id"))] if anchor is not None else []
            for attribute_prop in folded:
                if attribute_prop is anchor:
                    continue
                text, factual = _attribute_modifier(attribute_prop, stats, log)
                if not text:
                    continue
                if factual and not hedge_for(attribute_prop):
                    modifiers.append(text)
                    np_ids.append(str(attribute_prop.get("id")))
                else:
                    # An unresolved value trails the noun phrase as its own
                    # hedged span, so the existence claim stays assertable while
                    # the vague part cannot count towards PGF.
                    hedged_modifiers.append((text, str(attribute_prop.get("id"))))
            if anchor is not None:
                anchor_text, anchor_factual = _attribute_modifier(anchor, stats, log)
                if (anchor.get("attributes") or []) and anchor_text:
                    # Colour carried by the existence proposition itself
                    # (`có một chiếc ô tô màu trắng`) belongs in the same span:
                    # both fragments realise the same claim.
                    if anchor_factual and not hedge_for(anchor):
                        modifiers.insert(0, anchor_text)
                    else:
                        hedged_modifiers.append((anchor_text, str(anchor.get("id"))))
            if bucket.count is not None and str(bucket.count.get("id")) not in np_ids:
                np_ids.append(str(bucket.count.get("id")))
            if not np_ids:
                # No standalone existence proposition survived selection. The
                # subject NP is then the subject OF this entity's first
                # proposition, and that is its provenance — a span with no
                # proposition id is forbidden by the schema and would break the
                # explainability chain (doc 07 §5.1).
                np_ids = _bucket_ids(bucket)[:1]
            text = _noun_phrase_text(head, count, marker, exact, modifiers, entity, stats)
            if np_ids:
                emit(
                    clause,
                    PlannedSpan(
                        text=text,
                        proposition_ids=np_ids,
                        span_role=subject_role,
                        is_factual=not (anchor is not None and hedge_for(anchor)),
                        entity_id=bucket.entity_id,
                    ),
                )
            for hedged_text, pid in hedged_modifiers:
                emit(
                    clause,
                    PlannedSpan(
                        text=f"có vẻ {hedged_text}",
                        proposition_ids=[pid],
                        span_role="hedge",
                        is_factual=False,
                        entity_id=bucket.entity_id,
                    ),
                )
            folded_done = True
        else:
            ref, _form = _referring_expression(
                entity, head, mention, discourse.distance(bucket.entity_id),
                discourse.competitor(bucket.entity_id, head), cfg,
            )
            ref_ids = [str(anchor.get("id"))] if anchor is not None else _bucket_ids(bucket)[:1]
            if ref_ids:
                emit(
                    clause,
                    PlannedSpan(
                        text=ref,
                        proposition_ids=ref_ids,
                        span_role=subject_role,
                        entity_id=bucket.entity_id,
                    ),
                )
            folded_done = False
        discourse.note(bucket.entity_id, head)

        if not folded_done:
            overflow = bucket.np_attributes

        # How many continuation clauses this entity needs. Known up front so the
        # last one takes `và` and the others take a comma: `mặc áo đỏ, đội mũ và
        # đeo kính`. Repeating `và` reads as a machine listing (§4).
        continuations = (
            max(0, len(bucket.clausal_attributes) - 1)
            + max(0, len(bucket.actions) - 1)
            + len(overflow)
        )
        continuation = 0

        def coordinator() -> ExemptToken:
            nonlocal continuation
            continuation += 1
            if continuation >= continuations:
                return ExemptToken(CONNECTIVE_COORDINATION, "discourse_connective")
            return ExemptToken(",", "punctuation", glue="")

        def continuation_clause() -> Clause:
            """Open the clause that continues this entity's description.

            §4's clause-splitting limit is what stops the generator emitting
            `một người đàn ông cao, gầy, mặc áo đỏ, đội mũ xanh, đeo kính` —
            grammatical, faithful and unnatural. Past the per-sentence budget
            the description starts a new sentence, and a new sentence cannot
            elide its subject: §5.1's ellipsis rule is about ADJACENT clauses,
            and an opening `Và đeo kính.` has no subject at all.
            """
            if (
                anchor is not None
                and props_in_sentence >= cfg.max_propositions_per_sentence
                and sentence_count < cfg.max_sentences
            ):
                fresh = new_clause(bucket.entity_id, starts_sentence=True)
                reference, _form = _referring_expression(
                    entity, head, discourse.mentions.get(bucket.entity_id, 1) + 1,
                    discourse.distance(bucket.entity_id),
                    discourse.competitor(bucket.entity_id, head), cfg,
                )
                discourse.note(bucket.entity_id, head)
                emit(
                    fresh,
                    PlannedSpan(
                        text=reference,
                        proposition_ids=[str(anchor.get("id"))],
                        span_role="subject",
                        entity_id=bucket.entity_id,
                    ),
                )
                return fresh
            fresh = new_clause(bucket.entity_id, starts_sentence=False)
            fresh.subject_elided = True
            stats.elisions += 1
            fresh.pieces.append(coordinator())
            return fresh

        # -- ② attributes precede the verb, attached to the noun (§3) --------
        for position, attribute_prop in enumerate(bucket.clausal_attributes):
            text, factual = _clausal_attribute_text(attribute_prop, stats, log)
            if not text:
                continue
            if position:
                # Coordination, never bare juxtaposition: `mặc áo đỏ và đội mũ`.
                clause.pieces.append(coordinator())
            emit(
                clause,
                _maybe_hedge(
                    PlannedSpan(
                        text=text,
                        proposition_ids=[str(attribute_prop.get("id"))],
                        span_role="attribute",
                        is_factual=factual,
                        entity_id=bucket.entity_id,
                    ),
                    # An unresolved colour hedges the clause it sits in: the
                    # output must never carry a bare `xanh` as an assertion.
                    hedge_for(attribute_prop) or not factual,
                    clause,
                    plan,
                ),
            )

        # -- ③ action, with the subject elided in the adjacent clause (§5.1) --
        for position, action in enumerate(bucket.actions):
            text = _predicate_surface(action)
            if not text:
                continue
            target = clause
            if position > 0:
                # Same subject, adjacent clause: Vietnamese elides. Not eliding
                # is the hallmark of translated text (§5.1). The serial-verb
                # construction shares the aspect marker too — repeating `đang`
                # is the same translated register one level down (§4).
                target = continuation_clause()
                if target.subject_elided:
                    text = _drop_leading_aspect(text)
            obj_text, obj_factual = _object_phrase(action, stats, log)
            full = f"{text} {obj_text}".strip() if obj_text else text
            emit(
                target,
                _maybe_hedge(
                    PlannedSpan(
                        text=full,
                        proposition_ids=[str(action.get("id"))],
                        span_role="predicate",
                        is_factual=obj_factual,
                        entity_id=bucket.entity_id,
                    ),
                    hedge_for(action) or not obj_factual,
                    target,
                    plan,
                ),
            )
            clause = target

        # -- ② a third attribute becomes its own clause (§4) -----------------
        for attribute_prop in overflow:
            text, factual = _attribute_modifier(attribute_prop, stats, log)
            if not text:
                continue
            target = continuation_clause()
            emit(
                target,
                _maybe_hedge(
                    PlannedSpan(
                        # `có` heads a nominal property (`có màu đỏ`); Vietnamese
                        # adjectives are stative verbs and take no copula.
                        text=f"có {text}" if text.startswith("màu ") else text,
                        proposition_ids=[str(attribute_prop.get("id"))],
                        span_role="attribute",
                        is_factual=factual,
                        entity_id=bucket.entity_id,
                    ),
                    hedge_for(attribute_prop) or not factual,
                    target,
                    plan,
                ),
            )
            clause = target

        # -- ① setting is clause-final, on the topic clause only -------------
        if index == 0 and scene_props and not scene_used:
            for scene_prop in scene_props:
                text = _scene_surface(scene_prop)
                if not text:
                    continue
                emit(
                    clause,
                    _maybe_hedge(
                        PlannedSpan(
                            text=text,
                            proposition_ids=[str(scene_prop.get("id"))],
                            span_role="scene",
                            entity_id=None,
                        ),
                        hedge_for(scene_prop),
                        clause,
                        plan,
                    ),
                )
            scene_used = True

        # -- ⑥ relations between this entity and an introduced one -----------
        for relation in bucket.relations:
            span = _relation_span(relation, entities, props, discourse, stats, log, cfg)
            if span is None:
                plan.unplanned_ids.append(str(relation.get("id")))
                continue
            emit(clause, _maybe_hedge(span, hedge_for(relation), clause, plan))

    if scene_props and not scene_used:
        clause = plan.clauses[-1] if plan.clauses else new_clause(None, starts_sentence=True)
        for scene_prop in scene_props:
            text = _scene_surface(scene_prop)
            if text:
                emit(
                    clause,
                    PlannedSpan(
                        text=text,
                        proposition_ids=[str(scene_prop.get("id"))],
                        span_role="scene",
                    ),
                )

    plan.topic_order = [b.entity_id for b in buckets]
    plan.topic_shifts = topic_shifts
    stats.topic_shifts = topic_shifts

    planned = set(plan.ordered_ids)
    for prop in props:
        pid = str(prop.get("id"))
        if pid not in planned and pid not in plan.unplanned_ids:
            plan.unplanned_ids.append(pid)
    if plan.unplanned_ids:
        log.append(f"unplanned_propositions={sorted(set(plan.unplanned_ids))}")

    _guard_opening(plan, log)
    return plan


def _bucket_ids(bucket: _Bucket) -> list[str]:
    props = [bucket.existence, bucket.count]
    props += bucket.np_attributes + bucket.clausal_attributes + bucket.actions + bucket.relations
    return [str(p.get("id")) for p in props if p]


def _clausal_attribute_text(
    prop: dict, stats: RealizationStats, log: list[str]
) -> tuple[str, bool]:
    """`mặc áo đỏ` — a verbal attribute with its own object."""
    verb = _predicate_surface(prop)
    obj_text, factual = _object_phrase(prop, stats, log)
    if verb and obj_text:
        return f"{verb} {obj_text}", factual
    if verb:
        return verb, factual
    text, factual = _attribute_modifier(prop, stats, log)
    return text, factual


def _maybe_hedge(
    span: PlannedSpan, hedged: bool, clause: Clause, plan: DiscoursePlan
) -> PlannedSpan:
    """Wrap an UNCERTAIN proposition in a Vietnamese epistemic marker.

    The marker and everything it governs are `is_factual: false`, so hedged
    content cannot inflate PGF — letting it count would make hedging a free way
    to raise the metric (doc 10 §6, doc 04 §5.2).
    """
    if not hedged:
        return span
    clause.pieces.append(
        PlannedSpan(
            text=HEDGE_MARKER,
            proposition_ids=list(span.proposition_ids),
            span_role="hedge",
            is_factual=False,
            entity_id=span.entity_id,
        )
    )
    return replace(span, is_factual=False)


def _fronted_spatial(bucket: _Bucket, discourse: _Discourse) -> dict | None:
    """A spatial relation whose landmark is already on stage.

    Only fronted when THIS entity is the relation's subject. Fronting the
    converse would require inverting the relation (`phía sau` ↔ `phía trước`),
    and an inverted relation is a different claim.
    """
    for relation in bucket.relations:
        if str(relation.get("type")) != "spatial_relation":
            continue
        obj = relation.get("object") or {}
        other = obj.get("entity_id")
        if other and discourse.introduced(str(other)) and _spatial_surface(relation):
            return relation
    return None


def _relation_span(
    relation: dict,
    entities: Sequence[dict],
    props: Sequence[dict],
    discourse: _Discourse,
    stats: RealizationStats,
    log: list[str],
    cfg: RealizeConfig,
) -> PlannedSpan | None:
    """An in-situ relation clause: `ở phía sau một chiếc ô tô`, `cầm một cái ô`."""
    obj = relation.get("object") or {}
    other_id = str(obj.get("entity_id") or "")
    other = next((e for e in entities if str(e.get("id")) == other_id), None)
    if other is not None:
        head = _safe_head_noun(other, props, stats, log)
        if discourse.introduced(other_id):
            ref, _ = _referring_expression(
                other, head, discourse.mentions.get(other_id, 1) + 1,
                discourse.distance(other_id), discourse.competitor(other_id, head), cfg,
            )
        else:
            count, marker, exact = _number_realisation(other, None)
            ref = _noun_phrase_text(head, count, marker, exact, [], other, stats)
        discourse.note(other_id, head)
    else:
        ref = str(obj.get("text_vi") or "").strip()
    if not ref:
        return None

    if str(relation.get("type")) == "spatial_relation":
        surface = _spatial_surface(relation)
        if not surface:
            return None
        return PlannedSpan(
            text=f"ở {surface} {ref}",
            proposition_ids=[str(relation.get("id"))],
            span_role="spatial",
            entity_id=other_id or None,
        )
    verb = _predicate_surface(relation)
    if not verb:
        return None
    return PlannedSpan(
        text=f"{verb} {ref}",
        proposition_ids=[str(relation.get("id"))],
        span_role="object",
        entity_id=other_id or None,
    )


def _guard_opening(plan: DiscoursePlan, log: list[str]) -> None:
    """§5.5 constraint 6 — never open a caption with a pronoun.

    At the first mention there is no antecedent, so a pronoun there is an
    unresolvable reference. The planner introduces referents indefinitely, so
    this is a guard against a planning bug rather than a routine rewrite; it is
    logged when it fires so the bug is visible.
    """
    for clause in plan.clauses:
        for piece in clause.pieces:
            if isinstance(piece, ExemptToken):
                continue
            lowered = piece.text.strip().lower()
            if lowered in set(PRONOUNS.values()) | {PRONOUN_PLURAL} | set(_GENDERED_PRONOUNS):
                log.append(f"opening_pronoun_blocked: {piece.text!r}")
                plan.warnings.append("a caption must not open with a pronoun (§5.5)")
            return


# ---------------------------------------------------------------------------
# ④ Surface realisation — template
# ---------------------------------------------------------------------------
def realise_template(plan: DiscoursePlan) -> tuple[str, list[PlannedSpan], list[ExemptToken]]:
    """Deterministic realisation from the plan (§6).

    Zero hallucination by construction: every character of the output is either
    a planner span or a connective from the closed inventory. This is not merely
    the fallback — it is the existence proof that grounded realisation is
    achievable, which bounds how much observed hallucination can be blamed on
    realisation rather than on earlier stages (§6.0).
    """
    builder = _Builder()
    for index, clause in enumerate(plan.clauses):
        if index and clause.starts_sentence:
            builder.add_exempt(ExemptToken(".", "punctuation", glue=""))
        first_in_clause = True
        for piece in clause.pieces:
            capitalise = builder.length == 0 or (clause.starts_sentence and first_in_clause)
            if isinstance(piece, ExemptToken):
                builder.add_exempt(piece, capitalise=capitalise and piece.role != "punctuation")
            else:
                builder.add_span(piece, capitalise=capitalise)
            if piece.text.strip():
                first_in_clause = False
    if builder.length:
        builder.add_exempt(ExemptToken(".", "punctuation", glue=""))
    return builder.text(), builder.spans, builder.exempt


# ---------------------------------------------------------------------------
# ④ Surface realisation — constrained LM
# ---------------------------------------------------------------------------
# §6.1. A prompt change is a method change, so this text is hashed into
# `DocumentProvenance.modules[].prompt_hash` by the caller.
PROMPT_REALISE = """Bạn nhận được một danh sách các mệnh đề đã được kiểm chứng về một bức ảnh.

Nhiệm vụ: viết MỘT chú thích tiếng Việt tự nhiên, chi tiết, chỉ dựa trên
các mệnh đề này.

QUY TẮC BẮT BUỘC:
1. KHÔNG thêm bất kỳ thông tin nào không có trong danh sách.
2. KHÔNG suy đoán về mục đích, nghề nghiệp, cảm xúc, độ tuổi.
3. KHÔNG dùng đại từ chỉ giới tính nếu giới tính không được nêu rõ.
4. Tính từ đứng SAU danh từ (ví dụ: "áo đỏ", không phải "đỏ áo").
5. Dùng loại từ đúng (con, chiếc, cái, người, ...).
6. Có thể lược bỏ chủ ngữ lặp lại.
7. KHÔNG mở đầu chú thích bằng đại từ.
8. KHÔNG dùng "xanh" một mình: phải ghi rõ "xanh dương" hoặc "xanh lá".
9. Chỉ viết chú thích, không giải thích, không liệt kê lại mệnh đề.
10. KHÔNG dùng từ "này" trừ khi nhắc lại đối tượng đã nêu ở câu trước;
    câu đầu tiên tuyệt đối không dùng "này".
11. Mỗi câu phải hoàn chỉnh, có chủ ngữ rõ ràng.

Mệnh đề (theo thứ tự nên dùng):
{propositions}

Cách gọi tên đối tượng:
{references}
"""

PROMPT_REGENERATE = """Chú thích vừa viết: "{caption}"

Các cụm từ sau KHÔNG tương ứng với mệnh đề nào trong danh sách và VI PHẠM quy tắc 1:
{offending}

Hãy viết lại chú thích, loại bỏ hoàn toàn các cụm từ đó và không thêm thông tin mới.
"""

PROMPT_VIETNAMESE_ONLY = (
    "Chú thích phải viết hoàn toàn bằng tiếng Việt có dấu. Hãy viết lại bằng tiếng Việt."
)


def build_realisation_prompt(
    plan: DiscoursePlan,
    props: Sequence[dict],
    entities: Sequence[dict],
    stats: RealizationStats,
    log: list[str],
) -> str:
    """Render the §6.1 prompt for this plan.

    The proposition list is given in the planner's order and the referring
    expressions are given explicitly, because stage ④'s input is the discourse
    plan (§2), not the raw set — the LM is asked to realise a plan, not to
    re-plan.
    """
    by_id = {str(p.get("id")): p for p in props}
    lines = []
    for pid in plan.ordered_ids:
        prop = by_id.get(pid)
        if prop is None:
            continue
        marker = " [KHÔNG CHẮC — phải dùng 'có vẻ như']" if pid in plan.hedged_ids else ""
        lines.append(f"  {pid}: {prop.get('text_vi', '')}{marker}")

    references = []
    for entity_id in plan.topic_order:
        entity = next((e for e in entities if str(e.get("id")) == entity_id), None)
        if entity is None:
            continue
        head = _safe_head_noun(entity, props, stats, log)
        references.append(
            f"  - {head}: lần đầu gọi \"một {head}\"; các lần sau chỉ cần \"{head}\""
        )
    return PROMPT_REALISE.format(
        propositions="\n".join(lines) or "  (không có)",
        references="\n".join(references) or "  (không có)",
    )


def prompt_hash(prompt: str) -> str:
    """Stable hash for `DocumentProvenance.modules[].prompt_hash` (§6.1)."""
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16]


def _looks_vietnamese(text: str) -> bool:
    """Shallow check for §11's 'LM emits non-Vietnamese' row.

    Diacritics or a Vietnamese function word must appear. Deliberately crude and
    conservative: it only has to catch a wholesale switch to English, and a
    false negative costs one retry while a false positive would ship English.
    """
    lowered = text.lower()
    if any(ch in lowered for ch in "ăâđêôơưàáảãạèéẻẽẹìíỉĩịòóỏõọùúủũụỳýỷỹỵ"):
        return True
    return any(f" {w} " in f" {lowered} " for w in ("của", "một", "và", "người", "đang"))


# ---------------------------------------------------------------------------
# Post-editing guards (§11, §9.1) — fluency repairs, never content changes
# ---------------------------------------------------------------------------
def fix_adjective_order(text: str, stats: RealizationStats, log: list[str]) -> str:
    """`đỏ áo` -> `áo đỏ`. The most common MT artefact (§3, doc 03 §4).

    Only swaps a COLOUR immediately before a noun known to the classifier
    lexicon; anything less certain is left alone and the caption is reported as
    it was generated rather than silently rewritten.
    """
    tokens = text.split(" ")
    out: list[str] = []
    index = 0
    while index < len(tokens):
        current = tokens[index].strip(_PUNCTUATION).lower()
        following = tokens[index + 1].strip(_PUNCTUATION).lower() if index + 1 < len(tokens) else ""
        if current in COLORS and following in NOUN_CLASSIFIER:
            out.append(tokens[index + 1])
            out.append(tokens[index])
            stats.adjective_order_corrections += 1
            log.append(f"adjective_order_fixed: {current!r} {following!r} -> {following!r} {current!r}")
            index += 2
            continue
        out.append(tokens[index])
        index += 1
    return " ".join(out)


def fix_classifiers(text: str, stats: RealizationStats, log: list[str]) -> str:
    """Replace a classifier that does not agree with its noun.

    Grammatical agreement, so this is a fluency repair and never touches the
    claim set (§11, doc 02 §4.1).

    The head noun is matched LONGEST-FIRST over the following tokens. Vietnamese
    writes syllables apart, so taking one token would read `ô tô` (car,
    classifier `chiếc`) as `ô` (umbrella, classifier `cái`) and "correct" a
    right classifier into a wrong one.

    The lookup is an EXACT match on the window, never `classifier_for`, which
    also searches suffixes: on `một người mặc áo đỏ` it found `áo` inside the
    window `mặc áo đỏ`, declared the classifier of a noun three tokens away, and
    rewrote the subject to `một chiếc mặc áo đỏ` — destroying both the sentence
    and the span that carried the entity's proposition. A classifier governs the
    noun IMMEDIATELY after it or no noun at all.
    """
    tokens = text.split(" ")
    for index in range(len(tokens) - 1):
        current = tokens[index].strip(_PUNCTUATION).lower()
        if current not in CLASSIFIERS:
            continue
        noun, expected = "", ""
        for width in (3, 2, 1):
            candidate = " ".join(
                t.strip(_PUNCTUATION).lower() for t in tokens[index + 1 : index + 1 + width]
            )
            if candidate in NOUN_CLASSIFIER:
                noun, expected = candidate, NOUN_CLASSIFIER[candidate]
                break
        if not noun or expected == current or expected == noun:
            continue
        log.append(f"classifier_fixed: {current!r} -> {expected!r} before {noun!r}")
        tokens[index] = tokens[index].replace(current, expected)
        stats.classifier_corrections += 1
    return " ".join(tokens)


#: Gendered pronoun -> (gender it asserts, whether it also asserts an age band).
#: `ông ấy` / `bà ấy` are age-marked address forms: they claim an older
#: referent, so they need a verified age band on top of verified gender (§5.2).
_PRONOUN_CLAIMS: dict[str, tuple[str, bool]] = {
    "anh ấy": ("nam", False),
    "ông ấy": ("nam", True),
    "cô ấy": ("nu", False),
    "chị ấy": ("nu", False),
    "bà ấy": ("nu", True),
}


def fix_gendered_pronouns(
    text: str,
    entities: Sequence[dict],
    stats: RealizationStats,
    log: list[str],
    props: Sequence[dict] = (),
) -> str:
    """Replace a gendered pronoun that no entity's evidence licenses.

    Checked PER PRONOUN, not once for the whole caption. A single verified male
    entity does not license `cô ấy` for a second, unverified one, and it never
    licenses `ông ấy`, which asserts age as well as gender (§5.2, §9.1). Gating
    on `any(gender_verified(...))` would let both through — a hallucination the
    guard exists to prevent.

    Logged as a prevented hallucination (§11): the pronoun asserts something the
    image does not show.
    """
    out = text
    for pronoun in _GENDERED_PRONOUNS:
        value, needs_age = _PRONOUN_CLAIMS.get(pronoun, ("", True))
        gendered = [e for e in entities if gender_verified(e) and _gender(e)[0] == value]
        licensed = bool(gendered) and (
            not needs_age
            or any(_has_age_attribute(props, e.get("id")) for e in gendered)
        )
        if licensed:
            continue
        pattern = re.compile(rf"(?<!\w){re.escape(pronoun)}(?!\w)", re.IGNORECASE)
        if pattern.search(out):
            out = pattern.sub(PRONOUNS["khong_xac_dinh"], out)
            if gendered and needs_age:
                # Gender was fine; it is the age band the image never showed.
                stats.prevented_age_claims += 1
                log.append(f"prevented_age_claim: {pronoun!r} -> {PRONOUNS['khong_xac_dinh']!r}")
            else:
                stats.prevented_gender_hallucinations += 1
                log.append(
                    f"prevented_gender_hallucination: {pronoun!r} -> {PRONOUNS['khong_xac_dinh']!r}"
                )
    return out


def fix_opening_pronoun(
    text: str, plan: DiscoursePlan, stats: RealizationStats, log: list[str]
) -> str:
    """§5.5 constraint 6 — a caption may not open with a pronoun.

    At the first word there is no antecedent, so the pronoun refers to nothing a
    reader can recover; §5.3 makes an unrecoverable reference an ungrounded
    span. Repaired from the planner's own first subject, which is the referent
    the pronoun was standing in for. If the plan has no subject to restore, the
    text is left alone and stage ⑤ judges it — inventing a referent here would
    be exactly the failure the module prevents.
    """
    lowered = text.lower()
    pronouns = set(PRONOUNS.values()) | {PRONOUN_PLURAL} | set(_GENDERED_PRONOUNS)
    for pronoun in sorted(pronouns, key=len, reverse=True):
        if not lowered.startswith(pronoun):
            continue
        tail = text[len(pronoun) :]
        if tail and _WORD_CHAR.match(tail[0]):
            continue
        # Vietnamese writes syllables apart, so a pronoun-shaped first syllable
        # may be half of a noun: `họ hàng` (relatives) is not the pronoun `họ`.
        # Hand-built and therefore incomplete, like every list in `vi.lexicon`;
        # an unlisted compound costs one unnecessary repair, never a wrong claim.
        next_syllable = tail.strip().split(" ")[0].strip(_PUNCTUATION).lower() if tail.strip() else ""
        if next_syllable in _PRONOUN_COMPOUNDS.get(pronoun, frozenset()):
            continue
        subject = next((s.text for s in plan.spans if s.span_role == "subject"), None)
        if subject is None:
            return text
        log.append(f"opening_pronoun_repaired: {pronoun!r} -> {subject!r}")
        return _capitalise(subject) + tail
    return text


_BARE_XANH = re.compile(r"(?<!\w)(màu\s+)?xanh(?!\s*(?:dương|lam|lá|lục|nước biển|da trời|rêu))(?!\w)")


def fix_bare_xanh(
    text: str, props: Sequence[dict], stats: RealizationStats, log: list[str]
) -> str:
    """No bare `xanh` in the output: resolved, or hedged (§9.1 Colour).

    Resolution is only ever taken from `color_disambiguation` — the guard may
    not decide between blue and green on its own, because that would manufacture
    the very fact the schema refuses to guess. Unresolvable occurrences get an
    epistemic marker, and the span carrying them is marked `is_factual: false`
    downstream so it cannot inflate PGF.
    """
    resolved: set[str] = set()
    unresolved = False
    for prop in props:
        for attribute in prop.get("attributes") or []:
            if attribute.get("kind") != "màu_sắc":
                continue
            answer = str((attribute.get("color_disambiguation") or {}).get("resolved"))
            if answer in ("xanh_dương", "xanh_lá"):
                resolved.add(answer)
            elif parse_color(str(attribute.get("value_vi") or "")).xanh_value is Xanh.UNRESOLVED:
                # A bare `xanh` in P* that verification could NOT resolve. Its
                # presence means an occurrence in the text may belong to it, so
                # no resolution found elsewhere in P* may be applied globally.
                unresolved = True

    def substitute(match: re.Match[str]) -> str:
        prefix = match.group(1) or ""
        # One resolution, and nothing left unresolved that this occurrence could
        # be. Otherwise hedge: carrying a sibling proposition's blue over to an
        # unresolved green is manufacturing the very fact §9.1 refuses to guess.
        if len(resolved) == 1 and not unresolved:
            surface = next(iter(resolved)).replace("_", " ")
            stats.xanh_resolved += 1
            log.append(f"xanh_resolved: bare 'xanh' -> {surface!r} (from color_disambiguation)")
            return f"{prefix}{surface}"
        stats.xanh_hedged += 1
        log.append("xanh_hedged: bare 'xanh' unresolvable from P*, hedged instead of guessed")
        return f"có vẻ {prefix}xanh"

    return _BARE_XANH.sub(substitute, text)


# ---------------------------------------------------------------------------
# ⑤ Grounding verification (§7)
# ---------------------------------------------------------------------------
@dataclass
class Alignment:
    """Result of AlignSpansToPropositions (§7.1)."""

    spans: list[PlannedSpan] = field(default_factory=list)
    ungrounded: list[tuple[int, int, str]] = field(default_factory=list)
    unrealised_ids: list[str] = field(default_factory=list)
    degraded: bool = False


_WORD_CHAR = re.compile(r"\w", re.UNICODE)


def _find_word(haystack: str, needle: str, start: int = 0) -> int:
    """`str.find` that will not match inside a longer word.

    Vietnamese separates syllables with spaces, so a bare `str.find` locates the
    connective `là` inside `làm` — which then masks two characters of a genuine
    ungrounded claim and reports the remaining `m` as the violation. Every
    search over the caption goes through here for that reason.
    """
    position = haystack.find(needle, start)
    while position >= 0:
        before = position == 0 or not _WORD_CHAR.match(haystack[position - 1])
        end = position + len(needle)
        after = end >= len(haystack) or not _WORD_CHAR.match(haystack[end])
        if before and after:
            return position
        position = haystack.find(needle, position + 1)
    return -1


def _content_tokens(text: str) -> list[str]:
    return [t for t in re.split(r"\s+", text.strip()) if t and t.strip(_PUNCTUATION)]


def _strip_token(token: str) -> str:
    return token.strip(_PUNCTUATION + "\"'()").lower()


def align_spans(
    text: str,
    planned: Sequence[PlannedSpan],
    exempt: Sequence[ExemptToken],
    *,
    similarity_fn: Callable[[str, str], float] | None = None,
    soft_threshold: float = 0.75,
) -> Alignment:
    """Align caption spans to propositions (§7.1).

    Three tiers, in order: the planner record (exact), a lexical anchor
    validating it, and embedding similarity — which is used only when a
    similarity function is injected, and marks the alignment degraded, because
    it is the tier meant for unconstrained baselines where no planner record
    exists. Any text left over that is not exempt function material is an
    ungrounded span.
    """
    text = nfc(text)
    lowered = text.lower()
    alignment = Alignment()
    cursor = 0
    consumed: list[tuple[int, int]] = []

    for span in planned:
        needle = nfc(span.text).lower()
        if not needle.strip():
            continue
        position = _find_word(lowered, needle, cursor)
        if position < 0:
            position = _find_word(lowered, needle)
        if position >= 0:
            end = position + len(needle)
            alignment.spans.append(
                replace(span, text=text[position:end], char_start=position, char_end=end,
                        alignment="planner")
            )
            consumed.append((position, end))
            cursor = max(cursor, end)
            continue

        anchored = _lexical_anchor(text, lowered, needle, consumed)
        if anchored is not None:
            start, end = anchored
            alignment.spans.append(
                replace(span, text=text[start:end], char_start=start, char_end=end,
                        alignment="lexical")
            )
            consumed.append((start, end))
            cursor = max(cursor, end)
            continue

        if similarity_fn is not None:
            chunk = _semantic_anchor(text, span.text, consumed, similarity_fn, soft_threshold)
            if chunk is not None:
                start, end = chunk
                alignment.spans.append(
                    replace(span, text=text[start:end], char_start=start, char_end=end,
                            alignment="semantic")
                )
                consumed.append((start, end))
                alignment.degraded = True
                cursor = max(cursor, end)
                continue

        alignment.unrealised_ids.extend(span.proposition_ids)

    for token in exempt:
        needle = nfc(token.text).lower()
        if not needle.strip():
            continue
        position = _find_word(lowered, needle)
        while position >= 0:
            if not _overlaps(position, position + len(needle), consumed):
                consumed.append((position, position + len(needle)))
                break
            position = _find_word(lowered, needle, position + 1)

    alignment.spans.sort(key=lambda s: (s.char_start or 0, s.char_end or 0))
    alignment.ungrounded = _residual_claims(text, consumed)
    return alignment


def _overlaps(start: int, end: int, ranges: Sequence[tuple[int, int]]) -> bool:
    return any(start < b and a < end for a, b in ranges)


def _lexical_anchor(
    text: str, lowered: str, needle: str, consumed: Sequence[tuple[int, int]]
) -> tuple[int, int] | None:
    """Tier 2 — content words of the span appearing in the caption (§7.1).

    Requires at least half the content words, contiguous within a window, so an
    incidental single-word overlap cannot claim a span it does not realise.
    """
    words = [w for w in _content_tokens(needle) if _strip_token(w) not in _EXEMPT_WORDS]
    if not words:
        return None
    positions: list[tuple[int, int]] = []
    for word in words:
        found = _find_word(lowered, word)
        while found >= 0 and _overlaps(found, found + len(word), consumed):
            found = _find_word(lowered, word, found + 1)
        if found >= 0:
            positions.append((found, found + len(word)))
    if len(positions) * 2 < len(words):
        return None
    start = min(p[0] for p in positions)
    end = max(p[1] for p in positions)
    if end - start > len(needle) * 2 + 16:
        return None
    return start, end


def _semantic_anchor(
    text: str,
    span_text: str,
    consumed: Sequence[tuple[int, int]],
    similarity_fn: Callable[[str, str], float],
    threshold: float,
) -> tuple[int, int] | None:
    """Tier 3 — embedding similarity over clause chunks. Baselines only (§7.1)."""
    best: tuple[float, int, int] | None = None
    for match in re.finditer(r"[^,.;:!?]+", text):
        start, end = match.start(), match.end()
        if _overlaps(start, end, consumed):
            continue
        score = similarity_fn(match.group().strip(), span_text)
        if score >= threshold and (best is None or score > best[0]):
            best = (score, start, end)
    return (best[1], best[2]) if best else None


def _residual_claims(text: str, consumed: Sequence[tuple[int, int]]) -> list[tuple[int, int, str]]:
    """Text not covered by any span and not exempt function material.

    This is the detector behind `ungrounded ← { s : s.is_factual ∧ ids = ∅ }`
    (§7 line 2). Content words left uncovered are claims a reader would take the
    caption to assert, and nothing in P* put them there.
    """
    covered = sorted(consumed)
    gaps: list[tuple[int, int]] = []
    position = 0
    for start, end in covered:
        if start > position:
            gaps.append((position, start))
        position = max(position, end)
    if position < len(text):
        gaps.append((position, len(text)))

    out: list[tuple[int, int, str]] = []
    for start, end in gaps:
        segment = text[start:end]
        for match in re.finditer(r"[^,.;:!?]+", segment):
            chunk_start = start + match.start()
            chunk = match.group()
            tokens = list(re.finditer(r"\S+", chunk))
            run_start: int | None = None
            run_end = 0
            determiner_start: int | None = None
            for token in tokens:
                word = _strip_token(token.group())
                if not word or word in _EXEMPT_WORDS or word in HEDGE_MARKERS:
                    # A numeral or classifier immediately before ungrounded
                    # content is bound to it: leaving `chiếc` behind after
                    # removing `xe tải` produces a dangling determiner, which
                    # is worse than removing one token too many.
                    if word in _DETERMINERS and run_start is None:
                        if determiner_start is None:
                            determiner_start = chunk_start + token.start()
                    elif run_start is None:
                        determiner_start = None
                    continue
                if run_start is None:
                    run_start = determiner_start if determiner_start is not None else (
                        chunk_start + token.start()
                    )
                run_end = chunk_start + token.end()
            if run_start is not None:
                out.append((run_start, run_end, text[run_start:run_end]))
    return out


# ---------------------------------------------------------------------------
# Repair
# ---------------------------------------------------------------------------
_TIDY_PATTERNS: tuple[tuple[re.Pattern[str], int], ...] = (
    (re.compile(r"\s{2,}"), 1),              # collapse runs of spaces to one
    (re.compile(r"\s+(?=[,.;:!?])"), 0),     # space before punctuation
    (re.compile(r",\s*(?=[,.])"), 0),        # doubled punctuation after a strip
    (re.compile(r"^\s+"), 0),
    (re.compile(r"\s+$"), 0),
)


def _remove_ranges(
    text: str, ranges: Sequence[tuple[int, int]], spans: Sequence[PlannedSpan]
) -> tuple[str, list[PlannedSpan], bool]:
    """Delete character ranges and re-index the surviving spans.

    Right-to-left so earlier offsets stay valid while later ones shift. A span
    overlapping a removed range is reported as damaged rather than silently
    re-pointed — a wrong offset is worse than an admitted failure, and the
    caller falls back to template realisation.
    """
    out = text
    kept = [replace(s) for s in spans]
    damaged = False
    for start, end in sorted(ranges, key=lambda r: r[0], reverse=True):
        if start >= end:
            continue
        out = out[:start] + out[end:]
        width = end - start
        for span in kept:
            if span.char_start is None or span.char_end is None:
                continue
            if span.char_start >= end:
                span.char_start -= width
                span.char_end -= width
            elif span.char_end > start:
                damaged = True
    return out, kept, damaged


def _tidy(text: str, spans: Sequence[PlannedSpan]) -> tuple[str, list[PlannedSpan], bool]:
    """Clean up the whitespace and punctuation a strip leaves behind.

    Expressed as range removals so span offsets stay exact — rewriting the
    string with `re.sub` would invalidate every offset recorded so far.
    """
    out, kept, damaged = text, [replace(s) for s in spans], False
    for pattern, keep in _TIDY_PATTERNS:
        for _ in range(8):
            match = pattern.search(out)
            if match is None:
                break
            start, end = match.start() + keep, match.end()
            if start >= end:
                break
            out, kept, hurt = _remove_ranges(out, [(start, end)], kept)
            damaged = damaged or hurt
    if out and not out.endswith((".", "!", "?")):
        out = out + "."
    return out, kept, damaged


def looks_wellformed(text: str, spans: Sequence[PlannedSpan]) -> bool:
    """Grammaticality check for §7 line 13 — narrowly, the damage a strip causes.

    Not a Vietnamese grammar checker: it enumerates exactly the breakages that
    removing a span can produce, and returns False whenever it is unsure. The
    fallback is guaranteed-clean, so being conservative costs a little
    naturalness and never costs faithfulness.
    """
    stripped = text.strip()
    if not stripped or not spans:
        return False
    body = stripped.rstrip(".!?").strip()
    if not body:
        return False
    words = body.split()
    first, last = _strip_token(words[0]), _strip_token(words[-1])
    if first in set(DISCOURSE_CONNECTIVES) | {COPULA_EXISTENTIAL} | set(_PUNCTUATION):
        return False
    if last in set(DISCOURSE_CONNECTIVES) | {COPULA_EXISTENTIAL} | _DETERMINERS:
        return False  # a dangling determiner: `… là chiếc.`
    if first in set(PRONOUNS.values()) | {PRONOUN_PLURAL} | set(_GENDERED_PRONOUNS):
        return False  # §5.5 constraint 6
    if re.search(r"[,;:]\s*[,.;:]", stripped) or re.search(r"\.\s*\.", stripped):
        return False
    if re.search(r"(?:^|[,.;:])\s*(?:và|còn|là)\s*(?:[,.;:]|$)", " " + body.lower()):
        return False
    if any(len(_content_tokens(chunk)) == 0 for chunk in re.split(r"[,;]", body)):
        return False
    return True


def enforce_grounding(
    text: str,
    planned: Sequence[PlannedSpan],
    exempt: Sequence[ExemptToken],
    plan: DiscoursePlan,
    props: Sequence[dict],
    *,
    cfg: RealizeConfig,
    stats: RealizationStats,
    log: list[str],
    regenerate: Callable[[str, list[str]], str | None] | None = None,
    similarity_fn: Callable[[str, str], float] | None = None,
) -> tuple[str, list[PlannedSpan], list[str]]:
    """ALGORITHM EnforceGrounding (§7).

    Returns `(text, spans, ungrounded_texts)`. `ungrounded` is what was FOUND,
    and it is returned even after repair: the grounding-violation rate is a
    reported result measuring how hard the constraint is to satisfy, not
    something to hide (§7, §11 invariant).
    """
    retries_left = cfg.max_retries if regenerate is not None else 0
    found: list[str] = []
    alignment = align_spans(
        text, planned, exempt, similarity_fn=similarity_fn, soft_threshold=cfg.soft_threshold
    )

    while True:
        stats.alignment_degraded = stats.alignment_degraded or alignment.degraded
        if not alignment.ungrounded:
            break
        offending = [t for _, _, t in alignment.ungrounded]
        found.extend(t for t in offending if t not in found)
        if retries_left <= 0:
            break
        retries_left -= 1
        stats.retries += 1
        rewritten = regenerate(text, offending) if regenerate is not None else None
        if rewritten is None:
            break
        stats.regenerations += 1
        log.append(f"regenerated after ungrounded spans: {offending}")
        text = nfc(rewritten)
        alignment = align_spans(
            text, planned, exempt, similarity_fn=similarity_fn, soft_threshold=cfg.soft_threshold
        )

    if not alignment.ungrounded:
        stats.unrealised_ids = sorted(set(alignment.unrealised_ids))
        return text, alignment.spans, found

    # ▸ Exhausted retries: repair rather than ship a violation (§7 line 11).
    if getattr(cfg, "no_strip", False):
        # (research log): mid-phrase cuts make word salad ("cửa trắng, hàng có…") — the
        # B-LM variant forbids cutting: leftover violations go to the safe template, not patched.
        template_text, template_spans, _te = realise_template(plan)
        stats.fell_back_to_template = True
        log.append("template_fallback: no_strip — ungrounded remained after retries")
        return template_text, template_spans, found
    ranges = [(s, e) for s, e, _ in alignment.ungrounded]
    stripped_text, kept, damaged = _remove_ranges(text, ranges, alignment.spans)
    stripped_text, kept, hurt = _tidy(stripped_text, kept)
    stats.stripped_spans += len(ranges)
    log.append(f"stripped_ungrounded_spans={[t for _, _, t in alignment.ungrounded]}")

    if damaged or hurt or not looks_wellformed(stripped_text, kept):
        template_text, template_spans, _template_exempt = realise_template(plan)
        stats.fell_back_to_template = True
        log.append("template_fallback: stripping left the caption ill-formed")
        return template_text, template_spans, found

    stats.unrealised_ids = sorted(set(alignment.unrealised_ids))
    return stripped_text, kept, found


# ---------------------------------------------------------------------------
# Invariants
# ---------------------------------------------------------------------------
def assert_spans_exact(caption: dict[str, Any]) -> None:
    """Every span must actually be at the offsets it claims.

    A hard error, not a warning: doc 07 line 79 makes exact span provenance an
    interface invariant, and doc 04 §5.4 computes the Full model's PGF from
    these offsets without a parser. A silently shifted offset would corrupt the
    metric rather than merely annoy a reader — which is the failure mode NFC
    normalisation and right-to-left re-indexing are both there to prevent.
    """
    text = caption.get("text_vi") or ""
    guilty: list[str] = []
    for span in caption.get("spans") or []:
        start, end = span.get("char_start"), span.get("char_end")
        if not isinstance(start, int) or not isinstance(end, int):
            guilty.append(f"{span.get('text')!r}: missing character offsets")
            continue
        if text[start:end] != span.get("text"):
            guilty.append(f"{span.get('text')!r} != text_vi[{start}:{end}]={text[start:end]!r}")
        if not span.get("proposition_ids"):
            guilty.append(f"{span.get('text')!r}: no proposition_ids")
    if guilty:
        raise RuntimeError("span provenance is not exact: " + "; ".join(guilty))


def _verdict_of(prop: dict) -> str:
    """The bare verdict string, however `verification.status` was written.

    `pipeline.verify.Verdict` is a `str, Enum`, so a caller that assigns the
    enum MEMBER onto the document leaves `str(Verdict.REJECTED)` ==
    `'Verdict.REJECTED'` here, not `'REJECTED'`. Comparing the stringified form
    would then miss every rejected proposition and silently resurrect exactly
    the content verification refused — the failure `select._verdict_of` exists
    to prevent. Normalise through `.value` first.
    """
    raw = (prop.get("verification") or {}).get("status")
    raw = getattr(raw, "value", raw)
    return raw if isinstance(raw, str) else ""


def assert_selected(props: Sequence[dict]) -> None:
    """No REJECTED proposition may reach realisation (doc 07 §3 invariant 4)."""
    guilty = [str(p.get("id")) for p in props if _verdict_of(p) == "REJECTED"]
    if guilty:
        raise RuntimeError(
            "REJECTED propositions reached the realiser, which resurrects content "
            "verification already refused: " + ", ".join(guilty)
        )


# ---------------------------------------------------------------------------
# M7 entry point
# ---------------------------------------------------------------------------
def realize(
    propositions: Sequence[dict],
    entities: Sequence[dict],
    *,
    config: RealizeConfig | None = None,
    model: VLM | None = None,
    image: Any = None,
    hedged_ids: Iterable[str] = (),
    similarity_fn: Callable[[str, str], float] | None = None,
) -> RealizationResult:
    """Plan and realise a grounded Vietnamese caption from P*.

    `propositions` is P* — the selected set, already free of REJECTED content.
    Returns a `RealizationResult` whose `.caption` conforms to
    `GeneratedCaption` in `configs/proposition_schema.json`.

    `config.grounding_constraint=False` is Ablation A5: the repair loop is
    disabled. Alignment still runs, because the violation rate is a reported
    result and hiding it would make A5 look clean (§7).
    """
    cfg = config or RealizeConfig()
    if cfg.strategy not in ("constrained_lm", "template"):
        # §6.0's trained realiser is a separate method with its own checkpoint
        # and provenance; aliasing it to the prompted path would report one
        # method's numbers under another's name.
        raise ValueError(
            f"unknown realisation strategy {cfg.strategy!r}; "
            "'trained' (doc 11 §6.0) is not implemented in this module"
        )
    assert_selected(propositions)

    stats = RealizationStats(strategy=cfg.strategy, enforcement_enabled=cfg.grounding_constraint)
    log: list[str] = []
    working = list(propositions)

    # §9 layer 3: UNCERTAIN is never asserted. Selection normally passes the
    # admitted-hedge set in `hedged_ids`, but a caller that omits it would have
    # every UNCERTAIN proposition realised as a flat assertion — the three-way
    # verdict collapsing into a binary one at the last stage, invisibly. Union
    # the verdict-derived set in and say so, rather than trusting the caller.
    hedged_set = {str(h) for h in hedged_ids}
    from_verdict = {
        str(p.get("id")) for p in working if _verdict_of(p) == "UNCERTAIN"
    } - hedged_set
    if from_verdict:
        log.append(
            "hedged_from_verdict="
            f"{sorted(from_verdict)} (UNCERTAIN in P* but absent from hedged_ids)"
        )
        hedged_set |= from_verdict
    hedged_ids = sorted(hedged_set)

    if not working:
        # §11: never fabricate. Flag the image instead.
        log.append("empty_selection: P* is empty, emitting the fixed no-description caption")
        caption = _caption_dict(EMPTY_CAPTION_VI, [], [], [], cfg, stats, "template")
        return RealizationResult(caption, DiscoursePlan(), stats, log, [])

    while True:
        # Each attempt describes a different caption; only the last one ships.
        stats.begin_caption()
        plan = plan_caption(
            working, entities, hedged_ids=hedged_ids, config=cfg, stats=stats, log=log
        )
        if not plan.spans:
            # Nothing in P* could be attached to an entity in the registry.
            # The fixed no-description sentence is a message about the system,
            # not a claim about the image, so it carries no spans and skips
            # enforcement (§11).
            log.append("no_realisable_content: no proposition could be planned into a span")
            caption = _caption_dict(EMPTY_CAPTION_VI, [], [], [], cfg, stats, "template")
            return RealizationResult(caption, plan, stats, log, [])

        text, spans, exempt, generator = _realise(
            plan, working, entities, cfg, stats, log, model, image
        )
        text = nfc(text)

        if cfg.grounding_constraint:
            regenerator = _make_regenerator(
                plan, working, entities, cfg, stats, log, model, image
            ) if generator.startswith("constrained_lm") else None
            text, spans, ungrounded = enforce_grounding(
                text, spans, exempt, plan, working,
                cfg=cfg, stats=stats, log=log,
                regenerate=regenerator, similarity_fn=similarity_fn,
            )
        else:
            # A5 — measure, do not repair.
            alignment = align_spans(
                text, spans, exempt,
                similarity_fn=similarity_fn, soft_threshold=cfg.soft_threshold,
            )
            spans = alignment.spans
            ungrounded = [t for _, _, t in alignment.ungrounded]
            stats.alignment_degraded = alignment.degraded
            stats.unrealised_ids = sorted(set(alignment.unrealised_ids))
            log.append("ablation_a5_no_enforcement: grounding constraint disabled")

        if cfg.max_chars is not None and len(text) > cfg.max_chars and len(working) > 1:
            dropped = _drop_least_useful(working, plan, log)
            if dropped is not None:
                stats.dropped_for_length.append(dropped)
                working = [p for p in working if str(p.get("id")) != dropped]
                continue

        break

    if stats.fell_back_to_template and not generator.startswith("template"):
        # The text on the page is the template's, so `generator` must say so —
        # reporting the LM as the source of a caption it did not produce would
        # attribute the fallback's faithfulness to the wrong method.
        generator = f"template@fallback_from:{generator}"

    # Which tier established each span's provenance (§7.1). doc 04 §5.4 claims
    # the Full model's spans are exact and parser-free; a span recovered by
    # lexical anchor is a reconstruction, and the breakdown is what lets a
    # reader check that claim per document instead of taking it on trust.
    tiers: dict[str, int] = {}
    for span in spans:
        tiers[span.alignment] = tiers.get(span.alignment, 0) + 1
    stats.spans_by_alignment = dict(sorted(tiers.items()))
    if tiers.get("lexical"):
        log.append(f"spans_recovered_by_lexical_anchor={tiers['lexical']} (not planner-exact)")

    _check_hedges_survived(text, spans, plan, stats, log)
    # A proposition counts as unrealised only when NOTHING of it shipped. An
    # unaligned hedge marker carries its content proposition's id, so leaving it
    # in would report a realised claim as lost.
    shipped_ids = {pid for s in spans for pid in s.proposition_ids}
    stats.unrealised_ids = [pid for pid in stats.unrealised_ids if pid not in shipped_ids]
    if stats.unrealised_ids:
        # §11 'span alignment fails': the shipped caption realises fewer
        # propositions than the plan, so its span record is INCOMPLETE. Left
        # unsaid, a caption realising one of five propositions with no
        # ungrounded span reads as a perfect grounded caption, and doc 04 §5.4
        # would compute exact PGF over it. Publishing the loss is what lets that
        # image be excluded instead.
        log.append(f"provenance_incomplete: propositions with no span={stats.unrealised_ids}")

    hedged_texts = [s.text for s in spans if s.span_role == "hedge" or not s.is_factual]
    if ungrounded:
        # The one outcome the module must make impossible is a SILENT violation.
        log.append(f"grounding_violation={ungrounded}")
    stats.ungrounded_count = len(ungrounded)

    caption = _caption_dict(text, spans, ungrounded, hedged_texts, cfg, stats, generator)
    assert_spans_exact(caption)
    exempt_records = [
        {"text": t.text, "char_start": t.char_start, "char_end": t.char_end,
         "role": t.role, "is_factual": False}
        for t in exempt
        if t.char_start is not None
    ]
    return RealizationResult(caption, plan, stats, log, exempt_records)


def _check_hedges_survived(
    text: str,
    spans: Sequence[PlannedSpan],
    plan: DiscoursePlan,
    stats: RealizationStats,
    log: list[str],
) -> None:
    """Did every UNCERTAIN proposition keep its epistemic marker? (§9 layer 3)

    The planner emits `có vẻ như` before hedged content, but a constrained LM
    may drop it while keeping the content. The span still ships with
    `is_factual: false`, so PGF stays honest — but the SENTENCE now asserts what
    verification could not confirm, and that is the three-way verdict collapsing
    at the last stage with nothing in the record to show it. Detected and
    reported here; repairing the LM's wording is the caller's call, but it can
    no longer happen silently.
    """
    hedged = set(plan.hedged_ids)
    if not hedged:
        return
    lowered = nfc(text).lower()
    marker_positions = [
        match.start()
        for marker in HEDGE_MARKERS
        for match in re.finditer(re.escape(marker), lowered)
    ]
    missing: list[str] = []
    for pid in sorted(hedged):
        content = [
            s for s in spans
            if pid in s.proposition_ids and s.span_role != "hedge" and s.char_start is not None
        ]
        if not content:
            continue  # not realised at all — reported as unrealised instead
        if any(s.span_role == "hedge" and pid in s.proposition_ids for s in spans):
            continue
        start = min(int(s.char_start or 0) for s in content)
        if any(0 <= start - position <= 60 for position in marker_positions):
            continue
        missing.append(pid)
    if missing:
        stats.unhedged_uncertain = missing
        log.append(
            f"unhedged_uncertain={missing}: UNCERTAIN content shipped without "
            "an epistemic marker (§9 layer 3)"
        )


def _realise(
    plan: DiscoursePlan,
    props: Sequence[dict],
    entities: Sequence[dict],
    cfg: RealizeConfig,
    stats: RealizationStats,
    log: list[str],
    model: VLM | None,
    image: Any,
) -> tuple[str, list[PlannedSpan], list[ExemptToken], str]:
    """Stage ④. Returns `(text, spans, exempt, generator_name)`."""
    template_text, template_spans, template_exempt = realise_template(plan)
    if cfg.strategy == "template":
        return template_text, template_spans, template_exempt, "template"
    if model is None:
        # Honest degradation: the constrained-LM path needs a backbone, and
        # pretending we ran it would misreport the method.
        log.append("template_realisation: strategy=constrained_lm but no VLM was supplied")
        stats.fell_back_to_template = True
        stats.strategy = "template"
        return template_text, template_spans, template_exempt, "template"

    prompt = build_realisation_prompt(plan, props, entities, stats, log)
    stats.prompt_hash = prompt_hash(prompt)
    answer = model.describe(image, prompt, temperature=cfg.temperature, max_new_tokens=160)
    text = nfc(answer.text.strip().strip('"'))

    if not _looks_vietnamese(text):
        stats.non_vietnamese_retries += 1
        log.append("non_vietnamese_output: retrying with a stricter instruction (§11)")
        answer = model.describe(
            image, prompt + "\n" + PROMPT_VIETNAMESE_ONLY,
            temperature=cfg.temperature, max_new_tokens=160,
        )
        text = nfc(answer.text.strip().strip('"'))
        if not _looks_vietnamese(text):
            stats.fell_back_to_template = True
            log.append("template_fallback: model would not produce Vietnamese")
            return template_text, template_spans, template_exempt, "template"

    text = _post_edit(text, plan, props, entities, stats, log)
    return text, list(plan.spans), list(_plan_exempt(plan)), f"constrained_lm@{model.name}"


def _post_edit(
    text: str,
    plan: DiscoursePlan,
    props: Sequence[dict],
    entities: Sequence[dict],
    stats: RealizationStats,
    log: list[str],
) -> str:
    """Vietnamese fluency and hallucination guards (§9.1, §11).

    Applied BEFORE alignment so that every offset is measured on the final
    string; post-editing afterwards would invalidate the provenance record.
    """
    text = fix_adjective_order(text, stats, log)
    text = fix_classifiers(text, stats, log)
    text = fix_gendered_pronouns(text, entities, stats, log, props)
    text = fix_bare_xanh(text, props, stats, log)
    text = fix_opening_pronoun(text, plan, stats, log)
    return " ".join(text.split())


def _plan_exempt(plan: DiscoursePlan) -> list[ExemptToken]:
    return [p for clause in plan.clauses for p in clause.pieces if isinstance(p, ExemptToken)]


def _make_regenerator(
    plan: DiscoursePlan,
    props: Sequence[dict],
    entities: Sequence[dict],
    cfg: RealizeConfig,
    stats: RealizationStats,
    log: list[str],
    model: VLM | None,
    image: Any,
) -> Callable[[str, list[str]], str | None] | None:
    """Regenerate(P*, feedback = ungrounded) — §7 line 8, naming the offenders."""
    if model is None:
        return None
    base = build_realisation_prompt(plan, props, entities, stats, log)

    def regenerate(previous: str, offending: list[str]) -> str | None:
        feedback = PROMPT_REGENERATE.format(
            caption=previous,
            offending="\n".join(f'  - "{o}"' for o in offending),
        )
        answer = model.describe(
            image, base + "\n" + feedback, temperature=cfg.temperature, max_new_tokens=160
        )
        text = nfc(answer.text.strip().strip('"'))
        if not text or not _looks_vietnamese(text):
            return None
        return _post_edit(text, plan, props, entities, stats, log)

    return regenerate


def _drop_least_useful(
    props: Sequence[dict], plan: DiscoursePlan, log: list[str]
) -> str | None:
    """Which proposition to drop when the caption is too long (§11).

    `informativeness` is optional in the schema, so it is used only when every
    candidate has one; otherwise the last proposition in the planner's order
    goes, because §3's order is by decreasing discourse importance. The rule
    that fired is logged — an unexplained drop is an unexplained content loss.
    """
    candidates = [p for p in props if str(p.get("id")) in plan.ordered_ids]
    if not candidates:
        return None
    if all(p.get("informativeness") is not None for p in candidates):
        victim = min(candidates, key=lambda p: (float(p["informativeness"]), _pid_key(p)))
        log.append(f"length_drop rule=informativeness id={victim.get('id')}")
        return str(victim.get("id"))
    order = {pid: i for i, pid in enumerate(plan.ordered_ids)}
    victim = max(candidates, key=lambda p: order.get(str(p.get("id")), -1))
    log.append(f"length_drop rule=last_in_plan_order id={victim.get('id')} (informativeness absent)")
    return str(victim.get("id"))


def _caption_dict(
    text: str,
    spans: Sequence[PlannedSpan],
    ungrounded: Sequence[str],
    hedged: Sequence[str],
    cfg: RealizeConfig,
    stats: RealizationStats,
    generator: str,
) -> dict[str, Any]:
    """Project onto `GeneratedCaption`.

    Only the schema's own fields appear — the object is
    `additionalProperties: false`. Enforcement facts ride in `decoding`, which
    is the one open map, so a reader of the document alone can tell whether the
    constraint was applied and what it cost.
    """
    return {
        "text_vi": text,
        "spans": [s.as_schema() for s in spans],
        "ungrounded_spans": list(ungrounded),
        "hedged_spans": list(dict.fromkeys(hedged)),
        "generator": generator,
        "decoding": {
            "temperature": cfg.temperature,
            "seed": cfg.seed,
            "strategy": cfg.strategy,
            "grounding_constraint": cfg.grounding_constraint,
            "retries": stats.retries,
            "template_fallback": stats.fell_back_to_template,
            # `alignment_degraded` is the semantic tier only (§11): lexical
            # anchoring validates the planner record, semantic replaces it.
            "alignment_degraded": stats.alignment_degraded,
            "spans_by_alignment": dict(stats.spans_by_alignment),
            # Selected propositions that reached no span, and UNCERTAIN ones
            # shipped without their marker. Both are silent failures otherwise:
            # the first makes an under-realised caption score as a clean one,
            # the second asserts what verification refused to confirm.
            "unrealised_ids": list(stats.unrealised_ids),
            "unhedged_uncertain": list(stats.unhedged_uncertain),
            "prompt_hash": stats.prompt_hash,
        },
    }


def realize_document(
    doc: dict[str, Any],
    *,
    config: RealizeConfig | None = None,
    model: VLM | None = None,
    image: Any = None,
    similarity_fn: Callable[[str, str], float] | None = None,
) -> RealizationResult:
    """Realise the caption for a whole SVP document and store it in `doc`.

    Reads P* from `selection.selected_ids` and the hedge set from
    `selection.uncertain_admitted_ids`, so the caller cannot accidentally pass a
    proposition set that selection never approved.
    """
    selection = doc.get("selection") or {}
    selected = list(selection.get("selected_ids") or [])
    by_id = {str(p.get("id")): p for p in doc.get("propositions") or []}
    props = [by_id[pid] for pid in selected if pid in by_id]
    missing = [pid for pid in selected if pid not in by_id]
    result = realize(
        props,
        doc.get("entities") or [],
        config=config,
        model=model,
        image=image,
        hedged_ids=selection.get("uncertain_admitted_ids") or (),
        similarity_fn=similarity_fn,
    )
    if missing:
        result.log.append(f"selected ids absent from propositions: {missing}")
    doc["caption"] = result.caption
    return result


__all__ = [
    "RealizeConfig",
    "PlannedSpan",
    "ExemptToken",
    "Clause",
    "DiscoursePlan",
    "RealizationStats",
    "RealizationResult",
    "Alignment",
    "order_content",
    "plan_caption",
    "realise_template",
    "build_realisation_prompt",
    "prompt_hash",
    "align_spans",
    "enforce_grounding",
    "looks_wellformed",
    "assert_spans_exact",
    "assert_selected",
    "fix_adjective_order",
    "fix_classifiers",
    "fix_gendered_pronouns",
    "fix_bare_xanh",
    "fix_opening_pronoun",
    "gender_verified",
    "nfc",
    "realize",
    "realize_document",
    "EMPTY_CAPTION_VI",
    "PROMPT_REALISE",
]
