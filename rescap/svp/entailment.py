"""Entailment and specificity over structured visual propositions.

`formulation/02-PROPOSITION-SCHEMA.md` §6 defines both fields; selection
(`formulation/10` §2.3, §2.6) is the consumer that changes behaviour because of
them. They live in `rescap.svp` rather than in `rescap.pipeline` because they
are a property of the representation, not of one pipeline stage — and because
`svp/__init__` already promises "matching, alignment, entailment".

**Entailed propositions are reported, never deleted** (doc 08 §6). If
`có một người đàn ông` is later REJECTED because the gender is not visible,
`có một người` must still exist for the caption to fall back on. Dropping
anything is the selection objective's decision, taken with the whole set in
view; this module only says which pairs stand in an entailment relation.

**Direction convention:** `entails(a, b)` is true when **a is the more specific
claim** and b follows from it — P2 `có một người đàn ông` ⊨ P1 `có một người`.
Doc 02 §6's prose reads the other way round ("subject category of A subsumes
B's"), but its worked example (P2 ⊨ P1, P3 ⊨ P4) is what doc 10 §5's selection
trace was built from, so the example is authoritative here.

**Known gap — verb generality.** `vi/lexicon.py` owns every word list and other
modules "add no vocabulary of their own", and its `HYPERNYMS` lattice is
nouns-only. So `đạp xe ⊨ di chuyển` (doc 02 §6, P3 ⊨ P4) is **not derivable
here**; it is detected only when M3 declared it in `p["entails"]`. Closing the
gap means adding a verb lattice to `vi/lexicon.py`, not a private table here.
"""

from __future__ import annotations

from typing import Any

from ..vi.color import Xanh, parse_color
from .matching import _hypernym_chain, canonical

# Predicates that assert only that something is present. A proposition whose
# predicate is one of these carries no event content, which is what makes the
# existence rule below sound.
EXISTENTIAL_PREDICATES: frozenset[str] = frozenset({"có", "là", "xuất hiện"})

# Relation names returned by the component comparisons. `SAME` means the two
# slots make the same claim; `MORE_SPECIFIC` means a's slot claims strictly
# more than b's; `None` means no entailment can be read off this slot.
SAME = "same"
MORE_SPECIFIC = "more_specific"


# ---------------------------------------------------------------------------
# Category lattice
# ---------------------------------------------------------------------------
def hypernym_chain(term: str) -> list[str]:
    """`term` and every category above it, most specific first.

    Delegates to `matching._hypernym_chain` rather than re-walking `HYPERNYMS`:
    a second copy of the walk would eventually drift from the matcher's (its
    cycle guard in particular), and matching and entailment disagreeing about
    the lattice would make redundancy and precision incomparable.
    """
    return _hypernym_chain(canonical(term))


def hypernym_depth(term: str) -> int:
    """How far below the lattice root `term` sits. `người` -> 0, `đàn ông` -> 1."""
    if not term:
        return 0
    return len(hypernym_chain(term)) - 1


def subsumes(general: str, specific: str) -> bool:
    """True when `general` is a strict ancestor of `specific` (`người` ⊃ `đàn ông`).

    Strict: a term does not subsume itself, because equality is `SAME` and must
    stay distinguishable from `MORE_SPECIFIC` — otherwise two identical
    propositions would report as an entailment instead of as the duplicate they
    are, which is M3's business, not selection's.
    """
    general, specific = canonical(general), canonical(specific)
    if not general or not specific or general == specific:
        return False
    return general in _hypernym_chain(specific)


# ---------------------------------------------------------------------------
# Specificity  (doc 02 §6, doc 10 §2.3)
# ---------------------------------------------------------------------------
def specificity_level(p: dict[str, Any]) -> int:
    """Depth of `p` in the entailment hierarchy, σ(p).

    A **declared** `specificity_level` wins: M3 computed it with the entailment
    graph in view, which is strictly more information than this function has.
    The fallback is a structural proxy — how much the proposition commits to:

        subject depth in the category lattice
        + 1 if the predicate is more than bare existence
        + one per semantic attribute
        + 1 if it also relates the subject to an object or a location

    Reproduces doc 02 §6's table (P1=0, P2=1, P3=2). Note doc 10 §2.3 gives P4
    σ=1 where doc 02 §6 gives it σ=2; the proxy follows doc 02, and the
    disagreement changes only the ζ-weighted term of u(p), never a verdict.
    """
    declared = p.get("specificity_level")
    if isinstance(declared, int) and not isinstance(declared, bool) and declared >= 0:
        return declared

    _, category = _argument_key(p, "subject")
    level = hypernym_depth(category)

    predicate = canonical((p.get("predicate") or {}).get("lemma_vi"))
    if predicate and predicate not in EXISTENTIAL_PREDICATES:
        level += 1

    level += len(p.get("attributes") or [])

    # One bump for relating to a second thing, not two: an object and a spatial
    # relation on the same proposition are one commitment expressed twice.
    if p.get("object") or p.get("spatial_relation"):
        level += 1

    return level


# ---------------------------------------------------------------------------
# Slot comparisons
# ---------------------------------------------------------------------------
def _argument_key(p: dict[str, Any], slot: str) -> tuple[str | None, str]:
    """`(entity_id, canonical category)` for an argument slot, or `(None, "")`."""
    argument = p.get(slot)
    if not isinstance(argument, dict):
        return None, ""
    return (
        argument.get("entity_id"),
        canonical(argument.get("head_noun_vi") or argument.get("text_vi")),
    )


def _argument_relation(a: dict[str, Any], b: dict[str, Any], slot: str) -> str | None:
    """Compare one argument slot of `a` against `b`'s.

    **Distinct entity ids are distinct referents.** An image with two people
    makes `có một người đàn ông` about `nguoi_1` no evidence at all about
    `nguoi_2`, so entailment is refused rather than guessed. Deciding that two
    surface forms denote one referent is M3's coreference job (doc 02 §2.2);
    when it has run, both propositions carry the same `entity_id` and the
    category comparison below does the rest.
    """
    a_id, a_category = _argument_key(a, slot)
    b_id, b_category = _argument_key(b, slot)

    if a_id and b_id and a_id != b_id:
        return None
    if a_category == b_category:
        # Also the both-slots-absent case ("" == ""), which is `SAME` on
        # purpose: two intransitive actions agree about having no object.
        return SAME
    if a_category and b_category:
        if subsumes(b_category, a_category):
            return MORE_SPECIFIC
        if subsumes(a_category, b_category):
            return None  # a is the MORE GENERAL one; it cannot entail b
        if a_id and b_id:
            return SAME  # one referent, unrelated surface forms
    return None


def _predicate_relation(a: dict[str, Any], b: dict[str, Any]) -> str | None:
    """Compare predicates, including polarity.

    Differing polarity returns `None` rather than a direction: negation flips
    entailment, and `mặc áo đỏ` versus `không mặc áo đỏ` is a contradiction to
    be caught by the consistency constraint, never a redundancy.
    """
    predicate_a = a.get("predicate") or {}
    predicate_b = b.get("predicate") or {}
    if predicate_a.get("polarity", "positive") != predicate_b.get("polarity", "positive"):
        return None

    lemma_a = canonical(predicate_a.get("lemma_vi"))
    lemma_b = canonical(predicate_b.get("lemma_vi"))
    if lemma_a == lemma_b:
        return SAME
    if lemma_a and lemma_b in EXISTENTIAL_PREDICATES:
        return MORE_SPECIFIC  # doing something is more than merely being there
    if lemma_a and lemma_b and subsumes(lemma_b, lemma_a):
        return MORE_SPECIFIC  # see the module docstring: nouns-only lattice
    return None


def _attribute_value_relation(kind: str, value_a: str, value_b: str) -> str | None:
    """Compare two attribute values of the same family.

    Colour goes through the Vietnamese resolver so that `xanh lá ⊨ xanh` is
    recognised: a resolved green does entail the bare `xanh`. The reverse never
    holds — bare `xanh` covers blue *and* green, and letting it entail
    `xanh dương` would resolve the ambiguity silently, which is the one thing
    `vi/color.py` exists to prevent (doc 02 §4.5).
    """
    if canonical(value_a) == canonical(value_b):
        return SAME
    if kind == "màu_sắc":
        reading_a, reading_b = parse_color(value_a), parse_color(value_b)
        if reading_b.xanh_value is Xanh.UNRESOLVED and reading_a.xanh_value in (
            Xanh.BLUE,
            Xanh.GREEN,
        ):
            return MORE_SPECIFIC
        return None
    if subsumes(value_b, value_a):
        return MORE_SPECIFIC
    return None


def _attributes_relation(a: dict[str, Any], b: dict[str, Any]) -> str | None:
    """Attribute-subset rule: a ⊨ b when b's attributes are covered by a's."""
    attrs_a = [x for x in (a.get("attributes") or []) if isinstance(x, dict)]
    attrs_b = [x for x in (b.get("attributes") or []) if isinstance(x, dict)]

    if not attrs_b:
        return SAME if not attrs_a else MORE_SPECIFIC
    if not attrs_a:
        return None

    sharper = False
    for wanted in attrs_b:
        best: str | None = None
        for held in attrs_a:
            if held.get("kind") != wanted.get("kind"):
                continue
            relation = _attribute_value_relation(
                str(wanted.get("kind", "")),
                str(held.get("value_vi", "")),
                str(wanted.get("value_vi", "")),
            )
            if relation == SAME:
                best = SAME
                break
            if relation == MORE_SPECIFIC:
                best = MORE_SPECIFIC
        if best is None:
            return None
        sharper = sharper or best == MORE_SPECIFIC

    if sharper or len(attrs_a) > len(attrs_b):
        return MORE_SPECIFIC
    return SAME


def _payload_relation(a: dict[str, Any], b: dict[str, Any]) -> str | None:
    """Type-specific payloads. Equal or nothing.

    No specificity ordering is defined over spatial relations, counts or scene
    fields: `gần` is not a more precise `bên cạnh`, and `ba con chó` is not a
    refinement of `hai con chó` but a conflict for the contradiction detector.
    Inventing an ordering here would let redundancy silently delete a claim
    that disagrees with the one it is kept under.
    """
    spatial_a, spatial_b = a.get("spatial_relation") or {}, b.get("spatial_relation") or {}
    if bool(spatial_a) != bool(spatial_b):
        return None
    if spatial_a and (
        spatial_a.get("relation_vi") != spatial_b.get("relation_vi")
        or spatial_a.get("frame_of_reference", "image_relative")
        != spatial_b.get("frame_of_reference", "image_relative")
    ):
        return None

    count_a, count_b = a.get("count") or {}, b.get("count") or {}
    if bool(count_a) != bool(count_b):
        return None
    if count_a and count_a.get("value") != count_b.get("value"):
        return None

    scene_a, scene_b = a.get("scene") or {}, b.get("scene") or {}
    if {k: v for k, v in scene_a.items() if v} != {k: v for k, v in scene_b.items() if v}:
        return None

    return SAME


# ---------------------------------------------------------------------------
# The relation itself
# ---------------------------------------------------------------------------
def entailment_reason(a: dict[str, Any], b: dict[str, Any]) -> str | None:
    """Why `a` entails `b`, or None. The reason is logged, so name it.

    Rules, in the order doc 02 §6 lists them:

    | reason | example |
    |---|---|
    | `declared` | M3 wrote b's id into `a["entails"]` |
    | `subject_hypernymy` | P2 `có một người đàn ông` ⊨ P1 `có một người` |
    | `predicate_generality` | P3 `đạp xe` ⊨ P4 `di chuyển` (declared only) |
    | `attribute_subset` | `áo đỏ dài tay` ⊨ `áo đỏ` |

    **Entailment is compared within one proposition type**, which is stronger
    than pure logic and is deliberate. `người đàn ông đang đạp xe` (action) does
    logically imply `có một người` (entity) — doc 02 §6 says so — but treating
    that as *redundancy* would delete the entity proposition, and the entity
    proposition is what licenses the subject noun phrase at realisation. Doc 10
    §5 is explicit: its P* keeps P2 `có một người đàn ông` next to both P3
    (an action about that man) and P7 (a relation naming him as object), and
    blocks P1 by P2 alone. Inferring existence across types here drops P2 and
    leaves the caption with a dangling reference; a genuine cross-type
    entailment is still honoured when M3 declares it in `entails`.
    """
    a_id, b_id = a.get("id"), b.get("id")
    if a_id is not None and a_id == b_id:
        return None
    if b_id is not None and b_id in (a.get("entails") or []):
        return "declared"

    subject = _argument_relation(a, b, "subject")
    if subject is None:
        return None

    # An `attribute` proposition says nothing about whether an `action` one
    # holds; see the note above for the existence case, which is the only
    # cross-type rule pure logic would add and the one we refuse to infer.
    if a.get("type") != b.get("type"):
        return None

    predicate = _predicate_relation(a, b)
    if predicate is None:
        return None
    obj = _argument_relation(a, b, "object")
    if obj is None:
        return None
    attributes = _attributes_relation(a, b)
    if attributes is None:
        return None
    if _payload_relation(a, b) is None:
        return None

    for relation, reason in (
        (subject, "subject_hypernymy"),
        (predicate, "predicate_generality"),
        (obj, "object_hypernymy"),
        (attributes, "attribute_subset"),
    ):
        if relation == MORE_SPECIFIC:
            return reason

    # Every slot identical. That is a duplicate, which M3 merges (doc 08 §6);
    # calling it entailment here would let selection drop one of two records of
    # the same claim and report the drop as a redundancy saving.
    return None


def entails(a: dict[str, Any], b: dict[str, Any]) -> bool:
    """True when `a` is the more specific claim and `b` follows from it."""
    return entailment_reason(a, b) is not None


def entailment_pairs(props: list[dict[str, Any]]) -> dict[tuple[str, str], str]:
    """All entailments in a set, as `{(specific_id, general_id): reason}`.

    O(n²) and deliberately materialised once: selection re-scores candidate
    sets B times per step, and recomputing the lattice walks inside that loop
    dominated the runtime.
    """
    found: dict[tuple[str, str], str] = {}
    for a in props:
        for b in props:
            if a is b:
                continue
            reason = entailment_reason(a, b)
            if reason is not None:
                found[(str(a.get("id")), str(b.get("id")))] = reason
    return found


def maximal_ids(props: list[dict[str, Any]]) -> list[str]:
    """Ids of the entailment-maximal propositions — those nothing else entails.

    Offered for analysis and for the doc 02 §6 note that selection "may then
    keep only entailment-maximal propositions". Selection itself does **not**
    call this: pruning to the maximal set is a hard filter, and doc 10 §4.1
    lines 5-7 require the general parent to stay available in case the specific
    child loses to the budget or the hedge quota.
    """
    dominated = {general for _, general in entailment_pairs(props)}
    return [str(p.get("id")) for p in props if str(p.get("id")) not in dominated]
