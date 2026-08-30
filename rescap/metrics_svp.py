"""Proposition-level and factuality metrics — doc 04 groups 3, 4 and the ★ group 5.

    hallucination_rates()      formulation/04 §3.1-3.2   primary vs inherited
    vietnamese_factuality()    formulation/04 §3.3       the RQ4 instruments
    vietnamese_fluency()       formulation/04 §3.3       GRAMMAR, never factuality
    proposition_prf()          formulation/04 §4.1       P / R / F1
    verification_quality()     formulation/04 §4.2       accuracy, per-verdict F1, confusion
    pgf() / vcf()              formulation/04 §5.2       precision + coverage + harmonic mean

**Every match goes through `rescap.svp.matching`.** A second notion of "match"
living in the metrics would turn a scoring difference into an apparent
improvement, which is the failure doc 02 §7 exists to prevent.

Three rules this module enforces structurally rather than by convention:

1. **Primary and inherited hallucinations are never summed into the headline.**
   A hallucinated entity makes every attribute of it unmatched too; counting
   those independently multiplies one mistake into five and buries the
   attribute-level signal (§3.2, doc 09 §3.1).
2. **`xanh` confusion and `xanh` under-specification are separate numbers.**
   Blue asserted where gold is green is a hallucination; a bare `xanh` where
   gold is resolved is vague but *true*. Merging them would misattribute the fix
   (§3.3).
3. **No accessor returns grounding precision on its own.** `GroundedFactuality`
   stores counts, and the only ways out — `scores`, `as_row()`, `__str__` —
   carry precision, coverage and claim count together. A caption that says
   almost nothing scores 1.0 on precision, so precision alone is not a result
   (§5.3).

Classifier agreement, word order and count-classifier well-formedness are
computed here too, and they are **fluency**, not factuality: a system must not
be able to improve its "hallucination rate" by fixing grammar (§3.3).

No torch, no transformers, no embedding stack — this runs on a CPU box.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, NamedTuple, Sequence

from .svp.matching import Tier, align, match, match_any, subject_match
from .vi.classifier import check_agreement, parse_noun_phrase
from .vi.color import ColorErrorType, classify_color_error, parse_color
from .vi.lexicon import COLORS, GENDERED_NOUNS, XANH_AMBIGUOUS

# ---------------------------------------------------------------------------
# Fixed vocabularies
# ---------------------------------------------------------------------------
VERDICTS: tuple[str, str, str] = ("SUPPORTED", "UNCERTAIN", "REJECTED")

#: Schema proposition type -> the five hallucination buckets of doc 04 §3.1.
#: `action` and `scene` are DELIBERATELY absent: §3.1 defines a relation error as
#: a wrong predicate between two *correct entities* (two arguments), while an
#: action is single-argument, and doc 09 keeps channel 4 (action) apart from
#: channel 5 (relation, interaction). Folding them together is exactly the
#: silent merge this file exists to prevent, so they are reported on their own
#: schema-type rows instead and never enter a doc-04 bucket.
DOC04_BUCKET: dict[str, str] = {
    "entity": "object",
    "attribute": "attribute",
    "relation": "relation",
    "interaction": "relation",
    "spatial_relation": "spatial",
    "counting": "count",
}

#: Reported outside the five buckets, on their own rows, so they are visible.
UNBUCKETED_TYPES: tuple[str, ...] = ("action", "scene")

#: Type weights for the optional weighted variant (§5.3, safeguard 3).
#: **Stated, not tuned** — entity existence is cheap, relations and spatial
#: claims are hard. Tuning these to favour our system would be metric-gaming,
#: and the unweighted numbers are always reported alongside.
TYPE_WEIGHTS: dict[str, float] = {
    "entity": 0.5, "scene": 0.5,
    "attribute": 1.0,
    "action": 1.5, "relation": 1.5, "interaction": 1.5,
    "spatial_relation": 1.5, "counting": 1.5,
}
DEFAULT_TYPE_WEIGHT = 1.0

#: Where a REJECTED verdict records *why* (doc 03 §6.2: `contradicted_by_image`
#: vs `not_visually_determinable`). ONE canonical key, not a list of aliases —
#: accepting aliases invites gold producers to diverge and makes the split
#: incomparable across annotation batches.
#:
#: KNOWN SCHEMA GAP: `Verification` in proposition_schema.json v1.0.0 has no
#: such property and sets `additionalProperties: false`, so a schema-valid gold
#: file *cannot* carry the sub-label doc 03 §6.2 declares mandatory. Until the
#: schema is bumped, the F1_R split is reported as None with that reason rather
#: than guessed.
REJECT_SUBLABEL_FIELD = "reject_sublabel"
REJECT_SUBLABELS: tuple[str, str] = ("contradicted_by_image", "not_visually_determinable")

_SOFT_NOT_IMPLEMENTED = (
    "tier=soft requested but rescap.svp.matching.match() accepts `similarity_fn` "
    "and never consults it, so SOFT computes exactly LEXICAL. The number is a "
    "lexical one and must not be published in a soft column. Supplying a "
    "`similarity_fn` does NOT change this — the function is threaded through and "
    "dropped — so the warning stands whether or not one was passed."
)


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------
class PropositionSet(str, Enum):
    """Which set of the system's propositions a metric scores.

    Stated per report because the three answer different questions and doc 04
    uses different ones: §3.1 scores the caption's claims, §4.1 scores the
    candidate set (whose recall is the ceiling on everything downstream).
    """

    ALL = "all_propositions"          # candidate set P̂ (doc 08)
    SELECTED = "selected"             # P* (doc 10)
    CAPTION_CLAIMS = "caption_claims"  # P̂(C) — planner-recorded spans (doc 11 §7.1)


@dataclass
class EvalPair:
    """One image: the system's SVP document beside the gold SVP document.

    Both follow `configs/proposition_schema.json`. `parsed_claims` is the escape
    hatch of §5.4: baselines have no planner record, so their caption claims come
    from a Vietnamese proposition parser supplied by the caller. **We do not ship
    that parser** — writing one here would hide parser error inside the metric.
    Every report states which source it used, because the two are not comparable
    and the asymmetry favours our own system.
    """

    predicted: dict[str, Any]
    gold: dict[str, Any]
    parsed_claims: list[dict] | None = None
    image_id: str | None = None

    def key(self) -> str:
        for doc in (self.gold, self.predicted):
            ident = (doc.get("image") or {}).get("image_id")
            if ident:
                return str(ident)
        return self.image_id or "<unknown>"


def _props(doc: dict) -> list[dict]:
    return [p for p in (doc.get("propositions") or []) if isinstance(p, dict)]


def _entities(doc: dict) -> list[dict]:
    return [e for e in (doc.get("entities") or []) if isinstance(e, dict)]


def _selected(doc: dict) -> list[dict] | None:
    """P* — the propositions selection kept. None when no selection ran.

    None and [] mean different things: no selection stage at all versus a
    selection that chose nothing. PGF is undefined in the first case and 0 in
    the second, so they must not collapse.
    """
    selection = doc.get("selection")
    if not isinstance(selection, dict) or "selected_ids" not in selection:
        return None
    by_id = {p.get("id"): p for p in _props(doc)}
    return [by_id[i] for i in (selection.get("selected_ids") or []) if i in by_id]


def _verdict(prop: dict) -> str | None:
    """Delegates to `pipeline.verify.verdict_name`.

    This used to parse the field itself and returned `"Verdict.UNCERTAIN"` for
    an enum -- so every metric here read every in-memory verdict as an unknown
    label. Six implementations of this existed and three disagreed; there is
    now one.
    """
    from .pipeline.verify import verdict_name

    return verdict_name(prop)


def _is_adversarial(prop: dict) -> bool:
    """The literal doc 03 §10 marker, and *only* it.

    KNOWN SCHEMA GAP: `Proposition` in proposition_schema.json v1.0.0 sets
    `additionalProperties: false` and has no `adversarial` property, so a
    schema-valid gold file **cannot carry this marker** even though doc 03 §10
    declares it mandatory. Reading it alone therefore returns False for every
    adversarial proposition in schema-valid gold, which silently puts planted
    false claims into the §4.1 recall denominator and scores a system that
    hallucinates exactly the planted claim as a hit. Use `_is_false_gold`.
    """
    return bool(prop.get("adversarial"))


#: Signals that a GOLD proposition is false-by-construction, in priority order.
FALSE_GOLD_SIGNALS: tuple[str, str] = ("adversarial_marker", "rejected_verdict")


def _false_gold_signal(prop: dict) -> str | None:
    """Which signal marks this **gold** proposition as false-by-construction.

    Two signals, because the first cannot survive schema validation (see
    `_is_adversarial`) and silently disabling the doc 03 §10 exclusion is worse
    than naming the fallback:

    * `adversarial_marker` — the literal `adversarial: true` of doc 03 §10.
    * `rejected_verdict` — `verification.status == "REJECTED"`. Doc 03 §1.1's
      verdict table defines the gold REJECTED class as exactly "adversarial:
      plausible but false; and non-visual (purpose, identity)", both written by
      the same stage-4 construction (§10). A gold proposition the annotators
      rejected is one the system must not be credited for asserting, and one it
      must not be penalised for failing to generate — which is what §4.1's
      `G_non-adv` says.

    Applied only to gold. A predicted REJECTED proposition is the system's own
    verdict and means the opposite thing; `verification_quality` scores that.
    """
    if _is_adversarial(prop):
        return "adversarial_marker"
    if _verdict(prop) == "REJECTED":
        return "rejected_verdict"
    return None


def _is_false_gold(prop: dict) -> bool:
    return _false_gold_signal(prop) is not None


def _false_gold_notes(counts: dict[str, int]) -> list[str]:
    """Say which signal carried the §4.1 exclusion, so a reader can tell it ran."""
    if not any(counts.values()):
        return []
    if not counts.get("adversarial_marker"):
        return [
            f"{counts['rejected_verdict']} gold propositions were excluded as "
            "false-by-construction on `verification.status == REJECTED`: none carried "
            "`adversarial: true`, which proposition_schema.json v1.0.0 cannot represent "
            "(`Proposition.additionalProperties: false`). Doc 03 §10's exclusion is "
            "therefore applied through the verdict, not the marker."
        ]
    return [
        f"gold excluded as false-by-construction: {counts['adversarial_marker']} by "
        f"`adversarial: true`, {counts['rejected_verdict']} by a REJECTED gold verdict."
    ]


def _entity_id(prop: dict, role: str) -> str | None:
    argument = prop.get(role)
    if not isinstance(argument, dict):
        return None
    ident = argument.get("entity_id")
    return str(ident) if ident else None


def bbox_iou_fn(
    predicted_entities: Sequence[dict],
    gold_entities: Sequence[dict],
) -> Callable[[str, str], float | None]:
    """Build the `iou_fn` that `matching.subject_match` expects.

    Returns None — not 0.0 — when either side has no bbox. 0.0 would assert
    "these are different objects" on the strength of missing data; None lets the
    matcher fall through to its category test.
    """
    pred = {str(e.get("id")): e.get("bbox") for e in predicted_entities if e.get("id")}
    gold = {str(e.get("id")): e.get("bbox") for e in gold_entities if e.get("id")}

    def iou(predicted_id: str, gold_id: str) -> float | None:
        a, b = pred.get(predicted_id), gold.get(gold_id)
        if not a or not b or len(a) < 4 or len(b) < 4:
            return None
        ax, ay, aw, ah = (float(v) for v in a[:4])
        bx, by, bw, bh = (float(v) for v in b[:4])
        ix = max(0.0, min(ax + aw, bx + bw) - max(ax, bx))
        iy = max(0.0, min(ay + ah, by + bh) - max(ay, by))
        overlap = ix * iy
        union = aw * ah + bw * bh - overlap
        return overlap / union if union > 0 else None

    return iou


def _match_kwargs(
    pair: EvalPair,
    similarity_fn: Callable[[str, str], float] | None,
    soft_threshold: float,
) -> dict[str, Any]:
    return {
        "iou_fn": bbox_iou_fn(_entities(pair.predicted), _entities(pair.gold)),
        "similarity_fn": similarity_fn,
        "soft_threshold": soft_threshold,
    }


def _tier_notes(tier: Tier, similarity_fn: Callable | None) -> list[str]:
    """The SOFT warning does NOT depend on `similarity_fn`.

    `matching.match()` accepts the callable and never calls it, so a caller who
    supplies one gets lexical numbers in a column labelled soft — with the
    warning suppressed, which is worse than getting them without it. Gating the
    note on `similarity_fn is None` made the flag look like it did something.
    """
    if tier is Tier.SOFT:
        return [_SOFT_NOT_IMPLEMENTED]
    return []


def _rate(numerator: float, denominator: float) -> float | None:
    """None, never 0.0, on an empty denominator — 'nothing to measure' is not 'perfect'."""
    return numerator / denominator if denominator else None


def _mean_defined(values: Sequence[float | None]) -> float | None:
    """Mean over the values that exist. An undefined per-image score is skipped,
    never imputed as 0 or 1 — imputing either would invent a result."""
    present = [v for v in values if v is not None]
    return sum(present) / len(present) if present else None


def _fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"


# ---------------------------------------------------------------------------
# Caption claim set — P̂(C), doc 04 §5.2
# ---------------------------------------------------------------------------
#: Type of the synthetic claim that stands for a `caption.ungrounded_spans`
#: entry. No real proposition carries it, and `matching.match()` compares `type`
#: first, so such a claim can never match anything — which is the point: an
#: ungrounded span is factual caption text the planner could not trace to any
#: proposition (`pipeline/realize.py`: `ungrounded ← {s : s.is_factual ∧ ids = ∅}`).
UNGROUNDED_SPAN_TYPE = "__ungrounded_span__"

#: How P̂(C) was obtained (§5.4). The two are **not comparable** — planner spans
#: are exact while a parser injects its own error — and the asymmetry favours our
#: own system, so it is recorded per image rather than left to the table caption.
CLAIM_PROVENANCE_PLANNER = "planner_spans"
CLAIM_PROVENANCE_PARSER = "external_parser"


def _ungrounded_claim(text: str) -> dict:
    return {"type": UNGROUNDED_SPAN_TYPE, "text_vi": text, "ungrounded_span": True}


@dataclass
class ClaimSet:
    """The factual claims of one caption, with everything that was dropped.

    Hedged spans are excluded (§5.2): `có vẻ như X` does not assert X, so
    scoring it as an assertion would punish a system for being appropriately
    cautious — the behaviour RQ2 is trying to encourage.
    """

    claims: list[dict] = field(default_factory=list)
    source: PropositionSet | None = None
    provenance: str = CLAIM_PROVENANCE_PLANNER
    n_spans: int = 0
    n_hedged_excluded: int = 0
    n_nonfactual_excluded: int = 0
    n_spans_unresolved: int = 0
    n_ungrounded_spans: int = 0
    available: bool = True
    reason: str = ""

    @property
    def claim_source(self) -> str:
        """`source` and `provenance` together — the string every report prints."""
        source = self.source.value if self.source is not None else "none"
        return f"{source}:{self.provenance}"


def caption_claims(pair: EvalPair) -> ClaimSet:
    """Extract P̂(C) from the caption (§5.2), by planner record or supplied parse.

    Three exclusions, each for a stated reason:

    * `span_role = "hedge"`, anything listed in `caption.hedged_spans` — a hedge
      is not an assertion (§5.2). Tested BEFORE `is_factual`, because the
      realiser marks a hedge `is_factual: false` as well (`pipeline/realize.py`
      §7) and testing the other way round filed every hedge under
      `n_nonfactual_excluded`, leaving `n_hedged_excluded` at 0 for every
      caption — the count RQ2's hedging claim is read from.
    * `is_factual = false` — connectives and copulas carry no claim (schema).
    * spans whose `proposition_ids` point at a proposition not in the document —
      counted, never silently dropped, because that is a pipeline bug.

    And one **inclusion** that is easy to miss: `caption.ungrounded_spans` holds
    factual caption text the planner could NOT trace to any proposition. Those
    are claims the caption makes (§5.2 defines m over factual spans), and they
    are exactly what a grounding-constraint violation looks like — the thing
    ablation A5 turns off. Dropping them made PGF blind to the violation and
    scored a caption that hallucinated `một con chó` identically to one that did
    not, so they enter the claim set as claims nothing can match.

    A span with no `is_factual` flag is treated as factual: assuming the
    opposite would quietly exempt it from grounding.
    """
    if pair.parsed_claims is not None:
        # §5.4: parser error now enters the metric, and the report says so.
        return ClaimSet(
            claims=list(pair.parsed_claims),
            source=PropositionSet.CAPTION_CLAIMS,
            provenance=CLAIM_PROVENANCE_PARSER,
            n_spans=len(pair.parsed_claims),
            reason="claims supplied by an external Vietnamese parser (doc 04 §5.4)",
        )

    caption = pair.predicted.get("caption")
    if not isinstance(caption, dict):
        return ClaimSet(available=False, reason="tài liệu không có caption")
    spans = caption.get("spans")
    if not isinstance(spans, list) or not spans:
        return ClaimSet(
            available=False,
            reason=(
                "caption has no planner-recorded spans; supply `parsed_claims` "
                "for an unconstrained baseline (doc 04 §5.4)"
            ),
        )

    hedged_texts = {str(t).strip() for t in (caption.get("hedged_spans") or [])}
    by_id = {p.get("id"): p for p in _props(pair.predicted)}

    result = ClaimSet(source=PropositionSet.CAPTION_CLAIMS, n_spans=len(spans))
    seen: set[str] = set()
    counted_texts: set[str] = set()
    for span in spans:
        if not isinstance(span, dict):
            continue
        text = str(span.get("text", "")).strip()
        # Hedge first: the realiser also sets `is_factual: false` on a hedge.
        if span.get("span_role") == "hedge" or text in hedged_texts:
            result.n_hedged_excluded += 1
            continue
        if span.get("is_factual") is False:
            result.n_nonfactual_excluded += 1
            continue
        counted_texts.add(text)
        for pid in span.get("proposition_ids") or []:
            if pid not in by_id:
                result.n_spans_unresolved += 1
                continue
            # One claim per distinct proposition: the planner routinely splits
            # one proposition across a subject span and a predicate span, and
            # counting it twice would double-weight it in the mean of §5.2.
            if pid in seen:
                continue
            seen.add(pid)
            result.claims.append(by_id[pid])

    # Grounding-constraint violations: factual text with no proposition behind
    # it. Deduplicated against the spans already counted, so a span listed in
    # both places is one claim, not two.
    for raw in caption.get("ungrounded_spans") or []:
        text = str(raw).strip()
        if not text or text in hedged_texts or text in counted_texts:
            continue
        counted_texts.add(text)
        result.n_ungrounded_spans += 1
        result.claims.append(_ungrounded_claim(text))
    return result


def _claim_source(
    pair: EvalPair,
    source: PropositionSet,
) -> tuple[list[dict], str, str]:
    """Resolve the predicted set a metric scores. Returns `(props, source, reason)`."""
    if source is PropositionSet.ALL:
        return _props(pair.predicted), source.value, ""
    if source is PropositionSet.SELECTED:
        selected = _selected(pair.predicted)
        if selected is None:
            return [], source.value, "tài liệu không có khối selection"
        return selected, source.value, ""
    claims = caption_claims(pair)
    if not claims.available:
        return [], source.value, claims.reason
    return claims.claims, source.value, ""


# ---------------------------------------------------------------------------
# Entity alignment — the authority for the primary/inherited split
# ---------------------------------------------------------------------------
def _as_subject(entity: dict) -> dict:
    """Wrap an entity as a one-argument proposition so `subject_match` can judge it.

    Reuses doc 02 §7's rule (id equality -> bbox IoU -> compatible categories)
    instead of restating it here; a second entity-matching rule is exactly what
    this module must not contain.
    """
    return {
        "type": "entity",
        "subject": {
            "entity_id": entity.get("id"),
            "head_noun_vi": entity.get("category_vi", ""),
            "text_vi": entity.get("surface") or entity.get("category_vi", ""),
        },
    }


@dataclass
class EntityAlignment:
    """Predicted entities paired one-to-one with gold entities."""

    tier: str
    pairs: dict[str, str] = field(default_factory=dict)          # predicted id -> gold id
    hallucinated: set[str] = field(default_factory=set)          # no gold counterpart
    unmatched_gold: set[str] = field(default_factory=set)
    adjudicable: set[str] = field(default_factory=set)           # ids we could rule on at all
    abstained: bool = False                                      # no gold registry to rule with


def align_entities(pair: EvalPair, tier: Tier, **kwargs: Any) -> EntityAlignment:
    """Greedy one-to-one entity alignment. Each gold entity is consumed once, so
    five predictions of the same referent cannot each claim it.

    With an **empty gold entity registry** the alignment abstains: no pairs, no
    hallucinations, nothing adjudicable. Declaring every predicted entity
    hallucinated on the strength of an absent registry is an answer produced
    from missing data, and it lands downstream as a primary hallucination rate
    of 0 with everything filed as inherited.
    """
    predicted, gold = _entities(pair.predicted), _entities(pair.gold)
    iou_fn = kwargs.get("iou_fn")
    result = EntityAlignment(tier=tier.value)
    if not gold:
        result.abstained = True
        return result
    used: set[int] = set()

    for entity in predicted:
        pid = str(entity.get("id") or "")
        if not pid:
            continue
        result.adjudicable.add(pid)
        hit = -1
        for gi, gold_entity in enumerate(gold):
            if gi in used:
                continue
            if subject_match(_as_subject(entity), _as_subject(gold_entity), tier, iou_fn):
                hit = gi
                break
        if hit >= 0:
            used.add(hit)
            result.pairs[pid] = str(gold[hit].get("id") or "")
        else:
            result.hallucinated.add(pid)

    result.unmatched_gold = {
        str(g.get("id") or "") for i, g in enumerate(gold) if i not in used
    }
    return result


def _hallucinated_entities(
    pair: EvalPair,
    tier: Tier,
    alignment: EntityAlignment,
    **kwargs: Any,
) -> tuple[set[str], set[str]]:
    """`(hallucinated_ids, adjudicable_ids)` for the cascade rule of doc 09 §3.1.

    Two signals, and the second can **overturn** the first. Gold registry
    alignment leads: a hallucinated entity is one with no gold counterpart. Gold
    `entity` propositions are the second signal, and they are evidence in both
    directions — a gold proposition asserting the entity proves it exists even
    where the registry alignment said otherwise, which is the case whenever gold
    ships no entity registry at all. An add-only second signal turned that case
    into "every entity hallucinated", filed every claim as INHERITED, and printed
    a primary hallucination rate of 0 for a system that hallucinated freely.

    What neither signal can decide stays undecided rather than defaulting to
    primary, which would overstate the headline number.
    """
    hallucinated = set(alignment.hallucinated)
    adjudicable = set(alignment.adjudicable)

    gold_props = _props(pair.gold)
    for prop in _props(pair.predicted):
        if prop.get("type") != "entity":
            continue
        eid = _entity_id(prop, "subject")
        if not eid:
            continue
        adjudicable.add(eid)
        if eid in alignment.pairs:
            continue  # gold alignment already ruled: this entity exists
        matched = any(
            match(prop, g, tier, **kwargs).matched
            for g in gold_props
            if not _is_false_gold(g)
        )
        # The secondary signal must be able to CLEAR a flag, not only set one.
        # When gold asserts this entity in a proposition, the entity exists no
        # matter what the registry alignment concluded — and when gold ships no
        # entity registry at all, the alignment concluded nothing on evidence.
        # Leaving the flag on would file every downstream claim as INHERITED and
        # report a primary hallucination rate of 0 on a system that hallucinated.
        if matched:
            hallucinated.discard(eid)
        else:
            hallucinated.add(eid)
    return hallucinated, adjudicable


# ---------------------------------------------------------------------------
# Group 3 — hallucination rates (§3.1, §3.2)
# ---------------------------------------------------------------------------
@dataclass
class HallucinationRow:
    """One reported row. PRIMARY is the headline; the three never merge silently."""

    key: str
    n_claims: int = 0
    n_primary: int = 0
    n_inherited: int = 0
    n_undetermined: int = 0

    @property
    def primary_rate(self) -> float | None:
        return _rate(self.n_primary, self.n_claims)

    @property
    def inherited_rate(self) -> float | None:
        return _rate(self.n_inherited, self.n_claims)

    @property
    def combined_rate(self) -> float | None:
        """§3.2's Hall = Hall_primary + Hall_inherited. Reported, never the headline:
        alone it makes one entity mistake look catastrophic and hides attribute errors."""
        return _rate(self.n_primary + self.n_inherited, self.n_claims)

    @property
    def undetermined_rate(self) -> float | None:
        """Hallucinations whose cascade we could not establish. Non-zero here means
        the split above is incomplete, and the reader must be told."""
        return _rate(self.n_undetermined, self.n_claims)

    def as_row(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "n_claims": self.n_claims,
            "n_primary": self.n_primary,
            "n_inherited": self.n_inherited,
            "n_undetermined": self.n_undetermined,
            "primary_rate": self.primary_rate,
            "inherited_rate": self.inherited_rate,
            "combined_rate": self.combined_rate,
            "undetermined_rate": self.undetermined_rate,
        }


@dataclass
class HallucinationReport:
    tier: str
    source: str
    by_bucket: dict[str, HallucinationRow] = field(default_factory=dict)
    by_type: dict[str, HallucinationRow] = field(default_factory=dict)
    n_images: int = 0
    n_matched_adversarial: int = 0
    false_gold: dict[str, int] = field(
        default_factory=lambda: {s: 0 for s in FALSE_GOLD_SIGNALS}
    )
    skipped: list[tuple[str, str]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "tier": self.tier,
            "source": self.source,
            "n_images": self.n_images,
            "n_images_skipped": len(self.skipped),
            "skipped": self.skipped,
            "n_matched_adversarial": self.n_matched_adversarial,
            "false_gold_by_signal": self.false_gold,
            "by_bucket": {k: v.as_row() for k, v in self.by_bucket.items()},
            "by_type": {k: v.as_row() for k, v in self.by_type.items()},
            "notes": self.notes + _false_gold_notes(self.false_gold),
        }


def _ordered_gold(gold: Sequence[dict]) -> list[dict]:
    """True gold first, false-by-construction gold last.

    `align()` is greedy, and with a non-zero count `tolerance` a prediction can
    satisfy both the true count and its adversarial twin. Offering the true one
    first credits the prediction to the claim it actually satisfies instead of
    flagging a correct answer as a caught hallucination.

    **Not used by `verification_quality`.** There the gold REJECTED row is the
    measurement, and de-prioritising rejected gold could re-route a prediction
    onto a SUPPORTED gold and drain the row F1_R and the sub-label split rest on.
    """
    return [g for g in gold if not _is_false_gold(g)] + [g for g in gold if _is_false_gold(g)]


def hallucination_rates(
    pairs: Sequence[EvalPair],
    *,
    tier: Tier = Tier.LEXICAL,
    source: PropositionSet = PropositionSet.CAPTION_CLAIMS,
    similarity_fn: Callable[[str, str], float] | None = None,
    soft_threshold: float = 0.75,
) -> HallucinationReport:
    """Hallucination rate per type, split primary vs inherited (§3.1-3.2).

    A claim is hallucinated when no gold proposition matches it **or** when the
    gold it matches is adversarial — an adversarial gold is a claim the dataset
    built to be false, so matching one is a hallucination the annotators
    predicted, not a hit.

    It is INHERITED when its subject or object entity is itself hallucinated: a
    red shirt on a non-existent man is a cascade from an existence error, not an
    attribute error (doc 09 §3.1). Otherwise PRIMARY — the headline number.

    Matching here is **existential**, not one-to-one: §3.1 asks whether any gold
    proposition matches (`∄ g ∈ G`), and gold is not consumed. Using the greedy
    one-to-one `align()` — which §4.1 does need, to stop a repetitive system
    inflating its recall — would make two claims that legitimately match the same
    gold proposition come out as one hit and one hallucination, inflating the
    headline primary rate with a false positive.

    Corpus-level (micro): counts pool across images, because a per-image mean
    would weight a caption with two claims like one with twenty.
    """
    report = HallucinationReport(
        tier=tier.value, source=source.value, notes=_tier_notes(tier, similarity_fn)
    )

    def row(table: dict[str, HallucinationRow], key: str) -> HallucinationRow:
        return table.setdefault(key, HallucinationRow(key=key))

    for pair in pairs:
        claims, _, reason = _claim_source(pair, source)
        if reason:
            report.skipped.append((pair.key(), reason))
            continue
        report.n_images += 1

        kwargs = _match_kwargs(pair, similarity_fn, soft_threshold)
        alignment = align_entities(pair, tier, **kwargs)
        hallucinated_entities, adjudicable = _hallucinated_entities(
            pair, tier, alignment, **kwargs
        )

        # True gold first: `match_any` returns the FIRST match, so a claim that
        # satisfies a true gold is credited to it rather than to the adversarial
        # twin it may also satisfy under a count `tolerance`.
        gold = _ordered_gold(_props(pair.gold))
        for gold_prop in gold:
            signal = _false_gold_signal(gold_prop)
            if signal:
                report.false_gold[signal] += 1

        for claim in claims:
            ptype = str(claim.get("type") or "unknown")
            bucket = DOC04_BUCKET.get(ptype)
            rows = [row(report.by_type, ptype)]
            if bucket:
                rows.append(row(report.by_bucket, bucket))
            for target in rows:
                target.n_claims += 1

            gold_index, _ = match_any(claim, gold, tier, **kwargs)
            if gold_index >= 0:
                if not _is_false_gold(gold[gold_index]):
                    continue
                # Gold built this claim to be FALSE (doc 03 §10, or gold rejected
                # it outright). Matching one is a hallucination the annotators
                # predicted, never a hit.
                report.n_matched_adversarial += 1

            # Cascade. `entity` claims are the existence claims themselves, so
            # they can only ever be primary.
            subject_id = _entity_id(claim, "subject")
            object_id = _entity_id(claim, "object")
            involved = [i for i in (subject_id, object_id) if i]
            if ptype == "entity" or not involved:
                kind = "n_primary" if ptype == "entity" else "n_undetermined"
            elif any(i in hallucinated_entities for i in involved):
                kind = "n_inherited"
            elif all(i in adjudicable for i in involved):
                kind = "n_primary"
            else:
                kind = "n_undetermined"

            for target in rows:
                setattr(target, kind, getattr(target, kind) + 1)

    return report


# ---------------------------------------------------------------------------
# Group 3 — Vietnamese-specific factuality (§3.3)
# ---------------------------------------------------------------------------
_COLOR_TERMS: tuple[str, ...] = tuple(
    sorted(COLORS | {XANH_AMBIGUOUS}, key=len, reverse=True)
)


def extract_color_term(attribute: dict) -> str | None:
    """Pull the colour expression out of an attribute record.

    `color_disambiguation.raw` wins when present -- it is the field the schema
    reserves for the raw expression. Otherwise `value_vi`.

    Extraction itself is `vi.color.colour_term`. This used to iterate a SET of
    colour terms and return the first hit, so on `đen và nâu` it answered `nâu`
    while `pipeline.verify` answered `đen`: the verifier believed one colour and
    this scored another.
    """
    from .vi.color import colour_term

    disambiguation = attribute.get("color_disambiguation") or {}
    raw = disambiguation.get("raw") or attribute.get("value_vi") or ""
    return colour_term(raw)


def _color_attributes(prop: dict) -> list[dict]:
    return [
        a
        for a in (prop.get("attributes") or [])
        if isinstance(a, dict) and a.get("kind") == "màu_sắc"
    ]


@dataclass
class VietnameseFactualityReport:
    """The RQ4 instruments (§3.3). These are FACTUALITY rows.

    `xanh_confusion_rate` and `xanh_underspecification_rate` are separate fields
    and are never summed: the first is a hallucination (blue asserted where gold
    is green), the second is a true-but-vague answer. They have different causes
    and different fixes, so a single "xanh error rate" would misdirect the work.
    """

    tier: str
    n_color_pairs: int = 0
    n_color_comparable: int = 0
    n_color_unparseable: int = 0
    color_counts: dict[str, int] = field(default_factory=dict)
    n_gender_checked: int = 0
    n_gender_hallucinated: int = 0
    n_gender_gold_unrecorded: int = 0
    notes: list[str] = field(default_factory=list)

    @property
    def color_accuracy(self) -> float | None:
        return _rate(self.color_counts.get(ColorErrorType.CORRECT.value, 0), self.n_color_comparable)

    @property
    def color_error_rate(self) -> float | None:
        correct = self.color_counts.get(ColorErrorType.CORRECT.value, 0)
        return _rate(self.n_color_comparable - correct, self.n_color_comparable)

    @property
    def xanh_confusion_rate(self) -> float | None:
        """Blue asserted where gold is green, or the reverse — a hallucination."""
        return _rate(
            self.color_counts.get(ColorErrorType.XANH_CONFUSION.value, 0), self.n_color_comparable
        )

    @property
    def xanh_underspecification_rate(self) -> float | None:
        """Bare `xanh` where gold is resolved — vague, but TRUE. Not a hallucination."""
        return _rate(
            self.color_counts.get(ColorErrorType.XANH_UNDERSPECIFIED.value, 0),
            self.n_color_comparable,
        )

    @property
    def xanh_overspecification_rate(self) -> float | None:
        """Resolved prediction where gold itself could not tell. Neither a hit nor a
        hallucination against a known truth, so it gets its own row."""
        return _rate(
            self.color_counts.get(ColorErrorType.XANH_OVERSPECIFIED.value, 0),
            self.n_color_comparable,
        )

    @property
    def other_color_error_rate(self) -> float | None:
        return _rate(
            self.color_counts.get(ColorErrorType.OTHER_COLOR_ERROR.value, 0),
            self.n_color_comparable,
        )

    @property
    def gender_hallucination_rate(self) -> float | None:
        """Gendered noun asserted where gold records gender as not determinable.

        In Vietnamese gender sits in the noun (`đàn ông` / `phụ nữ`), so this is a
        CONTENT error, not a pronoun-agreement slip (§3.3).
        """
        return _rate(self.n_gender_hallucinated, self.n_gender_checked)

    def as_dict(self) -> dict[str, Any]:
        return {
            "tier": self.tier,
            "n_color_pairs": self.n_color_pairs,
            "n_color_comparable": self.n_color_comparable,
            "n_color_unparseable": self.n_color_unparseable,
            "color_accuracy": self.color_accuracy,
            "color_error_rate": self.color_error_rate,
            "xanh_confusion_rate": self.xanh_confusion_rate,
            "xanh_underspecification_rate": self.xanh_underspecification_rate,
            "xanh_overspecification_rate": self.xanh_overspecification_rate,
            "other_color_error_rate": self.other_color_error_rate,
            "color_counts": self.color_counts,
            "n_gender_checked": self.n_gender_checked,
            "n_gender_hallucinated": self.n_gender_hallucinated,
            "n_gender_gold_unrecorded": self.n_gender_gold_unrecorded,
            "gender_hallucination_rate": self.gender_hallucination_rate,
            "notes": self.notes,
        }


def _asserts_gender(entity: dict) -> bool:
    """Did the system commit to a gender for this entity?

    Three places it can leak in: the `gender.value` field, a gendered head noun
    (`đàn ông`), and the surface form the generator recorded. The neutral default
    `khong_xac_dinh` / `người` asserts nothing.
    """
    if (entity.get("gender") or {}).get("value") in ("nam", "nu"):
        return True
    category = str(entity.get("category_vi") or "").strip().lower()
    if category in GENDERED_NOUNS:
        return True
    surface = str(entity.get("surface") or "").lower()
    return any(noun in surface for noun in GENDERED_NOUNS)


def vietnamese_factuality(
    pairs: Sequence[EvalPair],
    *,
    tier: Tier = Tier.LEXICAL,
    similarity_fn: Callable[[str, str], float] | None = None,
    soft_threshold: float = 0.75,
) -> VietnameseFactualityReport:
    """Colour, `xanh` and gender rates (§3.3) — the Vietnamese-only failure modes.

    Colour pairs are collected on *correct* entities only: §3.3 defines a colour
    error as a wrong value on an entity that exists, so a colour claim about a
    hallucinated entity belongs to the inherited-hallucination row instead of
    being counted twice.
    """
    report = VietnameseFactualityReport(
        tier=tier.value, notes=_tier_notes(tier, similarity_fn)
    )
    counts: dict[str, int] = {t.value: 0 for t in ColorErrorType}

    for pair in pairs:
        kwargs = _match_kwargs(pair, similarity_fn, soft_threshold)
        alignment = align_entities(pair, tier, **kwargs)
        iou_fn = kwargs["iou_fn"]

        gold_color_props = [p for p in _props(pair.gold) if _color_attributes(p)]
        used: set[int] = set()

        for prop in _props(pair.predicted):
            predicted_attrs = _color_attributes(prop)
            if not predicted_attrs:
                continue
            subject_id = _entity_id(prop, "subject")
            if subject_id and subject_id not in alignment.pairs:
                continue  # colour on a hallucinated entity: an inherited error, counted there

            # Pair by SUBJECT only. A full `match()` fails precisely when the
            # colour differs, which is the case this metric exists to classify.
            partner = None
            for gi, gold_prop in enumerate(gold_color_props):
                if gi in used or _is_false_gold(gold_prop):
                    continue
                if subject_match(prop, gold_prop, tier, iou_fn):
                    partner = gold_prop
                    used.add(gi)  # one-to-one: a gold colour is compared once
                    break
            if partner is None:
                continue

            predicted_term = extract_color_term(predicted_attrs[0])
            gold_term = extract_color_term(_color_attributes(partner)[0])
            report.n_color_pairs += 1
            if predicted_term is None or gold_term is None:
                report.n_color_unparseable += 1
                continue
            outcome = classify_color_error(predicted_term, gold_term)
            counts[outcome.value] += 1
            if outcome is ColorErrorType.NOT_COMPARABLE:
                report.n_color_unparseable += 1
            else:
                report.n_color_comparable += 1

        gold_by_id = {str(e.get("id")): e for e in _entities(pair.gold)}
        for entity in _entities(pair.predicted):
            gold_id = alignment.pairs.get(str(entity.get("id")))
            gold_entity = gold_by_id.get(gold_id or "")
            if gold_entity is None:
                continue
            gold_gender = gold_entity.get("gender")
            if not isinstance(gold_gender, dict) or not gold_gender.get("evidence"):
                # Gold never recorded it, so "was this a guess?" is unanswerable.
                report.n_gender_gold_unrecorded += 1
                continue
            report.n_gender_checked += 1
            not_determinable = (
                gold_gender.get("evidence") == "not_determinable"
                or gold_gender.get("value") == "khong_xac_dinh"
            )
            if not_determinable and _asserts_gender(entity):
                report.n_gender_hallucinated += 1

    report.color_counts = counts
    return report


# ---------------------------------------------------------------------------
# Group 3 — Vietnamese fluency (§3.3, the grammatical rows)
# ---------------------------------------------------------------------------
@dataclass
class VietnameseFluencyReport:
    """Classifier agreement, adjective order, count well-formedness.

    **These are FLUENCY metrics and must never enter a hallucination rate**
    (§3.3). `ba cái chó` is bad Vietnamese, not a false statement about the
    image; letting it into a factuality number would let a system improve its
    "factuality" by fixing grammar, which would be a false claim in the paper.
    """

    n_classifier_checked: int = 0
    n_classifier_agree: int = 0
    n_classifier_unknown_noun: int = 0
    n_np_checked: int = 0
    n_np_word_order_violations: int = 0
    n_np_unchecked: int = 0
    n_count_checked: int = 0
    n_count_agree: int = 0
    n_count_unknown_noun: int = 0

    @property
    def classifier_agreement_rate(self) -> float | None:
        """Unknown nouns are excluded: `check_agreement` returns None for them, and
        counting a vocabulary gap as a grammar error would blame the lexicon."""
        return _rate(self.n_classifier_agree, self.n_classifier_checked)

    @property
    def word_order_violation_rate(self) -> float | None:
        """Pre-nominal adjectives (`đỏ áo`) — the classic MT artefact (doc 03 §4)."""
        return _rate(self.n_np_word_order_violations, self.n_np_checked)

    @property
    def count_classifier_agreement_rate(self) -> float | None:
        """NUMERAL + CLASSIFIER + NOUN well-formedness."""
        return _rate(self.n_count_agree, self.n_count_checked)

    def as_dict(self) -> dict[str, Any]:
        return {
            "note": "FLUENCY, not factuality (doc 04 §3.3) — never add to a hallucination rate",
            "classifier_agreement_rate": self.classifier_agreement_rate,
            "n_classifier_checked": self.n_classifier_checked,
            "n_classifier_unknown_noun": self.n_classifier_unknown_noun,
            "word_order_violation_rate": self.word_order_violation_rate,
            "n_np_checked": self.n_np_checked,
            "n_np_unchecked": self.n_np_unchecked,
            "count_classifier_agreement_rate": self.count_classifier_agreement_rate,
            "n_count_checked": self.n_count_checked,
            "n_count_unknown_noun": self.n_count_unknown_noun,
        }


def vietnamese_fluency(pairs: Sequence[EvalPair]) -> VietnameseFluencyReport:
    """Grammatical well-formedness of the system's Vietnamese (§3.3).

    Reference-free by construction: agreement and adjective order are properties
    of the output alone, so no gold document is consulted. `pairs` is still the
    argument type so callers hold one corpus object rather than two.
    """
    report = VietnameseFluencyReport()

    for pair in pairs:
        for entity in _entities(pair.predicted):
            head = str(entity.get("category_vi") or "")
            classifier = str(entity.get("classifier") or "")
            if head and classifier:
                agrees = check_agreement(classifier, head)
                if agrees is None:
                    report.n_classifier_unknown_noun += 1
                else:
                    report.n_classifier_checked += 1
                    report.n_classifier_agree += int(agrees)

            # Adjective order needs the surface noun phrase. `category_vi` is the
            # bare head noun by definition and carries no modifiers, so scoring it
            # would report a perfect rate while checking nothing.
            surface = entity.get("surface")
            if not surface:
                report.n_np_unchecked += 1
                continue
            report.n_np_checked += 1
            # NounPhrase exposes the violation only through its warning text;
            # if classifier.py rewords that message this line must change with it.
            if any(w.startswith("pre-nominal") for w in parse_noun_phrase(str(surface)).warnings):
                report.n_np_word_order_violations += 1

        for prop in _props(pair.predicted):
            count = prop.get("count")
            if prop.get("type") != "counting" or not isinstance(count, dict):
                continue
            noun = str(count.get("entity_category_vi") or "")
            classifier = str(count.get("classifier") or "")
            if not noun or not classifier or count.get("value") is None:
                report.n_count_unknown_noun += 1
                continue
            agrees = check_agreement(classifier, noun)
            if agrees is None:
                report.n_count_unknown_noun += 1
            else:
                report.n_count_checked += 1
                report.n_count_agree += int(agrees)

    return report


# ---------------------------------------------------------------------------
# Group 4 §4.1 — proposition precision / recall / F1
# ---------------------------------------------------------------------------
@dataclass
class PRF:
    """Counts first, rates derived. None on an empty denominator, never 0.0."""

    key: str = "overall"
    tp: int = 0
    fp: int = 0
    fn: int = 0
    n_predicted: int = 0
    n_gold_non_adversarial: int = 0
    n_matched_adversarial: int = 0

    @property
    def precision(self) -> float | None:
        return _rate(self.tp, self.n_predicted)

    @property
    def recall(self) -> float | None:
        """Denominator excludes gold that is false by construction (doc 03 §10):
        a system is not penalised for failing to generate a claim the dataset
        built to be false. See `_false_gold_signal` for how such gold is
        identified when the schema cannot carry the `adversarial` marker."""
        return _rate(self.tp, self.n_gold_non_adversarial)

    @property
    def f1(self) -> float | None:
        p, r = self.precision, self.recall
        if p is None or r is None or (p + r) == 0:
            return None
        return 2 * p * r / (p + r)

    def as_row(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "tp": self.tp, "fp": self.fp, "fn": self.fn,
            "n_predicted": self.n_predicted,
            "n_gold_non_adversarial": self.n_gold_non_adversarial,
            "n_matched_adversarial": self.n_matched_adversarial,
            "precision": self.precision, "recall": self.recall, "f1": self.f1,
        }


@dataclass
class PropositionReport:
    tier: str
    source: str
    overall: PRF = field(default_factory=PRF)
    by_type: dict[str, PRF] = field(default_factory=dict)
    n_images: int = 0
    false_gold: dict[str, int] = field(
        default_factory=lambda: {s: 0 for s in FALSE_GOLD_SIGNALS}
    )
    skipped: list[tuple[str, str]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "tier": self.tier,
            "source": self.source,
            "n_images": self.n_images,
            "skipped": self.skipped,
            "false_gold_by_signal": self.false_gold,
            "overall": self.overall.as_row(),
            "by_type": {k: v.as_row() for k, v in self.by_type.items()},
            "notes": self.notes + _false_gold_notes(self.false_gold),
        }


def proposition_prf(
    pairs: Sequence[EvalPair],
    *,
    tier: Tier = Tier.LEXICAL,
    source: PropositionSet = PropositionSet.ALL,
    similarity_fn: Callable[[str, str], float] | None = None,
    soft_threshold: float = 0.75,
) -> PropositionReport:
    """Proposition P / R / F1, overall and per type (§4.1).

    A prediction matching a **false-by-construction** gold proposition is a false
    positive, not a hit: those gold entries are false by construction, so
    crediting them would reward hallucinating exactly the claims the dataset
    planted. `false_gold_by_signal` in the report says which signal identified
    them, because the doc 03 §10 marker is unrepresentable in a schema-valid gold
    file and a silently inert exclusion looks identical to no adversarial data.

    Overall F1 is computed from overall P and R, never as the mean of per-type
    F1s — the two differ whenever the types are unevenly populated, and the mean
    lets a rare, easy type carry the headline.

    With `source=ALL` this is candidate-set recall, the ceiling on everything
    downstream (doc 08 §10). Every later number must be read against it.
    """
    report = PropositionReport(
        tier=tier.value, source=source.value, notes=_tier_notes(tier, similarity_fn)
    )

    def row(key: str) -> PRF:
        return report.by_type.setdefault(key, PRF(key=key))

    for pair in pairs:
        predicted, _, reason = _claim_source(pair, source)
        if reason:
            report.skipped.append((pair.key(), reason))
            continue
        report.n_images += 1

        kwargs = _match_kwargs(pair, similarity_fn, soft_threshold)
        gold = _ordered_gold(_props(pair.gold))
        for gold_prop in gold:
            signal = _false_gold_signal(gold_prop)
            if signal:
                report.false_gold[signal] += 1
        result = align(predicted, gold, tier, **kwargs)
        matched = {pi: gi for pi, gi in result["pairs"]}

        for index, prop in enumerate(predicted):
            ptype = str(prop.get("type") or "unknown")
            target = row(ptype)
            report.overall.n_predicted += 1
            target.n_predicted += 1

            gold_index = matched.get(index)
            if gold_index is not None and not _is_false_gold(gold[gold_index]):
                report.overall.tp += 1
                target.tp += 1
                continue
            if gold_index is not None:
                report.overall.n_matched_adversarial += 1
                target.n_matched_adversarial += 1
            report.overall.fp += 1
            target.fp += 1

        # Recall side: gold that is false by construction never counts, per
        # doc 03 §10 — a system is not penalised for failing to generate it.
        hit_gold = {gi for _, gi in result["pairs"]}
        for gi, gold_prop in enumerate(gold):
            if _is_false_gold(gold_prop):
                continue
            gtype = str(gold_prop.get("type") or "unknown")
            target = row(gtype)
            report.overall.n_gold_non_adversarial += 1
            target.n_gold_non_adversarial += 1
            if gi not in hit_gold:
                report.overall.fn += 1
                target.fn += 1

    return report


# ---------------------------------------------------------------------------
# Group 4 §4.2 — verification quality
# ---------------------------------------------------------------------------
@dataclass
class VerdictPRF:
    verdict: str
    tp: int = 0
    fp: int = 0
    fn: int = 0

    @property
    def precision(self) -> float | None:
        return _rate(self.tp, self.tp + self.fp)

    @property
    def recall(self) -> float | None:
        return _rate(self.tp, self.tp + self.fn)

    @property
    def f1(self) -> float | None:
        p, r = self.precision, self.recall
        if p is None or r is None or (p + r) == 0:
            return None
        return 2 * p * r / (p + r)

    def as_row(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict, "tp": self.tp, "fp": self.fp, "fn": self.fn,
            "precision": self.precision, "recall": self.recall, "f1": self.f1,
        }


@dataclass
class VerificationReport:
    """§4.2. The confusion matrix is indexed `confusion[gold][predicted]`.

    That orientation is load-bearing. §4.2's cost table reads along the GOLD row:
    cost₂ is gold SUPPORTED predicted REJECTED (lost detail), cost₃ is gold
    UNCERTAIN predicted SUPPORTED (a hallucination). A transposed matrix swaps
    those two and nothing downstream would catch it.
    """

    tier: str
    n_compared: int = 0
    n_correct: int = 0
    n_predicted_unverified: int = 0
    n_gold_unverified: int = 0
    n_unmatched_predicted: int = 0
    n_unmatched_gold: int = 0
    confusion: dict[str, dict[str, int]] = field(default_factory=dict)
    per_verdict: dict[str, VerdictPRF] = field(default_factory=dict)
    rejected_by_sublabel: dict[str, dict[str, Any]] | None = None
    rejected_sublabel_reason: str = ""
    skipped: list[tuple[str, str]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def accuracy(self) -> float | None:
        """Reported, but never alone: it hides the class RQ2 turns on (F1_UNCERTAIN)."""
        return _rate(self.n_correct, self.n_compared)

    def uncertain_row(self) -> dict[str, int]:
        """The gold-UNCERTAIN row — reported side by side against the binary
        Baseline C, which has no UNCERTAIN class at all (§4.2)."""
        return dict(self.confusion.get("UNCERTAIN", {}))

    def as_dict(self) -> dict[str, Any]:
        return {
            "tier": self.tier,
            "accuracy": self.accuracy,
            "n_compared": self.n_compared,
            "n_correct": self.n_correct,
            "n_predicted_unverified": self.n_predicted_unverified,
            "n_gold_unverified": self.n_gold_unverified,
            "n_unmatched_predicted": self.n_unmatched_predicted,
            "n_unmatched_gold": self.n_unmatched_gold,
            "n_images_skipped": len(self.skipped),
            "skipped": self.skipped,
            "confusion_gold_by_predicted": self.confusion,
            "gold_uncertain_row": self.uncertain_row(),
            "per_verdict": {k: v.as_row() for k, v in self.per_verdict.items()},
            "rejected_by_sublabel": self.rejected_by_sublabel,
            "rejected_sublabel_reason": self.rejected_sublabel_reason,
            "notes": self.notes,
        }


def _sublabel(prop: dict) -> str | None:
    value = (prop.get("verification") or {}).get(REJECT_SUBLABEL_FIELD)
    return str(value) if value in REJECT_SUBLABELS else None


def verification_quality(
    pairs: Sequence[EvalPair],
    *,
    tier: Tier = Tier.LEXICAL,
    source: PropositionSet = PropositionSet.ALL,
    similarity_fn: Callable[[str, str], float] | None = None,
    soft_threshold: float = 0.75,
) -> VerificationReport:
    """Verdict accuracy, per-verdict P/R/F1 and the confusion matrix (§4.2).

    Scored only on propositions present in **both** sets — a proposition the
    system never generated is a recall failure of generation (§4.1), and folding
    it in here would blame the verifier for it.

    A predicted proposition still carrying `status = None` is counted as
    unverified rather than mapped onto a verdict: the generator asserts nothing
    (doc 08), and inventing a verdict for it would fabricate the very number
    this metric measures.
    """
    report = VerificationReport(tier=tier.value, notes=_tier_notes(tier, similarity_fn))
    report.confusion = {g: {p: 0 for p in VERDICTS} for g in VERDICTS}
    report.per_verdict = {v: VerdictPRF(verdict=v) for v in VERDICTS}

    sublabel_counts: dict[str, dict[str, int]] = {
        s: {"n_gold": 0, "n_predicted_rejected": 0, "n_predicted_this_sublabel": 0}
        for s in REJECT_SUBLABELS
    }
    gold_sublabels_seen = 0
    gold_rejected_total = 0
    predicted_sublabels_missing = 0

    for pair in pairs:
        predicted, _, reason = _claim_source(pair, source)
        if reason:
            # Recorded, not dropped in silence: an image missing from the
            # denominator changes every rate below it.
            report.skipped.append((pair.key(), reason))
            continue
        kwargs = _match_kwargs(pair, similarity_fn, soft_threshold)
        # Document order, NOT `_ordered_gold`. Here the gold REJECTED row *is*
        # the measurement (§4.2, F1_R and its sub-label split); de-prioritising
        # rejected gold in a greedy alignment could re-route a prediction onto a
        # SUPPORTED gold and drain the very row being reported.
        gold = _props(pair.gold)
        result = align(predicted, gold, tier, **kwargs)
        report.n_unmatched_predicted += len(result["unmatched_predicted"])
        report.n_unmatched_gold += len(result["unmatched_gold"])

        for pi, gi in result["pairs"]:
            predicted_verdict = _verdict(predicted[pi])
            gold_verdict = _verdict(gold[gi])
            if gold_verdict is None:
                report.n_gold_unverified += 1
                continue
            if predicted_verdict is None:
                report.n_predicted_unverified += 1
                continue

            report.confusion[gold_verdict][predicted_verdict] += 1
            report.n_compared += 1
            report.n_correct += int(predicted_verdict == gold_verdict)

            if gold_verdict == "REJECTED":
                gold_rejected_total += 1
                gold_sublabel = _sublabel(gold[gi])
                if gold_sublabel is None:
                    continue
                gold_sublabels_seen += 1
                bucket = sublabel_counts[gold_sublabel]
                bucket["n_gold"] += 1
                if predicted_verdict == "REJECTED":
                    bucket["n_predicted_rejected"] += 1
                    predicted_sublabel = _sublabel(predicted[pi])
                    if predicted_sublabel is None:
                        predicted_sublabels_missing += 1
                    elif predicted_sublabel == gold_sublabel:
                        bucket["n_predicted_this_sublabel"] += 1

    for verdict in VERDICTS:
        row = report.per_verdict[verdict]
        row.tp = report.confusion[verdict][verdict]
        row.fp = sum(report.confusion[g][verdict] for g in VERDICTS if g != verdict)
        row.fn = sum(report.confusion[verdict][p] for p in VERDICTS if p != verdict)

    # F1_R split by sub-label (§4.2): rejecting a purpose claim and rejecting a
    # colour error are different competencies.
    if gold_sublabels_seen == 0:
        report.rejected_by_sublabel = None
        report.rejected_sublabel_reason = (
            f"no gold REJECTED proposition carries `verification.{REJECT_SUBLABEL_FIELD}` "
            f"({gold_rejected_total} were compared). Doc 03 §6.2 makes the sub-label "
            "mandatory but proposition_schema.json v1.0.0 has no field for it and sets "
            "additionalProperties:false, so the split cannot be computed."
        )
    else:
        report.rejected_by_sublabel = {}
        for sublabel, bucket in sublabel_counts.items():
            entry: dict[str, Any] = dict(bucket)
            entry["rejection_recall"] = _rate(bucket["n_predicted_rejected"], bucket["n_gold"])
            if predicted_sublabels_missing:
                entry["sublabel_accuracy"] = None
                entry["sublabel_accuracy_reason"] = (
                    f"{predicted_sublabels_missing} predicted REJECTED verdicts record no "
                    f"`{REJECT_SUBLABEL_FIELD}`, so agreement on the sub-label is not computable"
                )
            else:
                entry["sublabel_accuracy"] = _rate(
                    bucket["n_predicted_this_sublabel"], bucket["n_predicted_rejected"]
                )
            report.rejected_by_sublabel[sublabel] = entry

    return report


# ---------------------------------------------------------------------------
# ★ Group 5 — PGF and VCF (§5.2)
# ---------------------------------------------------------------------------
class Scores(NamedTuple):
    """Precision, coverage and their harmonic mean — always together.

    §5.3 rule 1: never report precision alone. `Có một người.` is one claim, it
    is grounded, and its precision is 1.0. Coverage is what makes terseness
    unprofitable, so it travels in the same value.
    """

    precision: float | None
    coverage: float | None
    harmonic_mean: float | None
    n_claims: int
    n_reference: int


def _harmonic(precision: float | None, coverage: float | None) -> float | None:
    if coverage is None:
        return None
    # A caption covering nothing scores 0 whatever its precision is — including
    # the degenerate empty caption whose precision is undefined. This is the
    # anti-terseness property of §5.3 and it must survive the None case.
    if coverage == 0.0:
        return 0.0
    if precision is None or (precision + coverage) == 0:
        return None
    return 2 * precision * coverage / (precision + coverage)


@dataclass(frozen=True)
class GroundedFactuality:
    """One caption's PGF or VCF (§5.2).

    **Stores counts, not scores.** The object has no `precision` attribute to
    read: `scores`, `as_row()` and `__str__` are the only ways out, and each
    yields precision, coverage, the harmonic mean and the claim count together.
    Python cannot forbid attribute access, so the safeguard is that no accessor
    exists which produces precision on its own (§5.3 rule 1) and none which
    produces factuality without the detail it was achieved at (rule 2).
    """

    metric: str                     # "PGF" | "VCF"
    tier: str
    image_id: str
    claim_source: str
    n_claims: int                   # m, after hedged and non-factual spans are dropped
    n_claims_grounded: int
    n_match_targets: int            # PGF: |P*_SUP|; VCF: |G_true|
    n_reference: int                # coverage denominator — PGF: |P*|; VCF: |G_true|
    n_reference_covered: int
    n_hedged_excluded: int = 0
    n_reference_hedged_uncovered: int = 0
    n_claims_matching_uncertain_gold: int = 0
    #: Claims that came from `caption.ungrounded_spans`. They count in `n_claims`
    #: and can never be grounded or cover a reference. Two readings, both stated
    #: because the direction of the error differs:
    #:   PGF — determinately ungrounded. That is what the field means, and this
    #:         is the grounding-constraint violation the metric exists to catch.
    #:   VCF — NOT determinately false. There is no proposition to match against
    #:         gold, so the claim is unverifiable rather than wrong. It is
    #:         counted in m (§5.2 defines m over factual spans) but this count is
    #:         published beside it so the undeterminable share of the precision
    #:         loss is never read as outright falsehood.
    n_claims_ungrounded_spans: int = 0
    #: Caption spans whose `proposition_ids` resolve to nothing — a pipeline bug
    #: that silently shrinks m if it is not reported.
    n_spans_unresolved: int = 0
    #: PGF only: selected propositions carrying no verdict yet (doc 08 leaves
    #: `status: None` until `assert_clean` runs). They stay in the coverage
    #: denominator |P*| but can never be a precision target, so a non-zero count
    #: caps PGF-P below 1 for a reason that is not the caption's fault.
    n_reference_unverified: int = 0
    w_claims: float = 0.0
    w_claims_grounded: float = 0.0
    w_reference: float = 0.0
    w_reference_covered: float = 0.0
    available: bool = True
    reason: str = ""

    @property
    def n_claims_checkable(self) -> int:
        """Precision denominator. Equals `n_claims` except for VCF.

        A claim from `caption.ungrounded_spans` has no proposition structure, so
        `match(p̂, g)` against gold is **undefined**, not false. Leaving it in
        VCF's denominator scores "we could not check this" as "this is wrong" —
        the collapse of undetermined into negative that this whole framework
        exists to prevent — and the resulting 0.6 is indistinguishable in a table
        from a genuine 0.6. It leaves the denominator and is published as
        `n_claims_ungrounded_spans`, from which a reader can reconstruct the
        pessimistic bound. `n_claims` stays whole: m is still the caption's claim
        count, so coverage and the §5.5 detail column are unaffected.

        PGF keeps every claim: there the span is *determinately* ungrounded
        (`is_factual ∧ ids = ∅`), which is precisely what PGF measures.
        """
        if self.metric == "VCF":
            return self.n_claims - self.n_claims_ungrounded_spans
        return self.n_claims

    @property
    def scores(self) -> Scores:
        """Unweighted precision, coverage and harmonic mean.

        Precision is None — not 1.0 — when the caption made no factual claim:
        an empty claim set has nothing to be right about, and 1.0 would hand a
        silent caption the top of the table.
        """
        precision = _rate(self.n_claims_grounded, self.n_claims_checkable)
        coverage = _rate(self.n_reference_covered, self.n_reference)
        return Scores(precision, coverage, _harmonic(precision, coverage),
                      self.n_claims, self.n_reference)

    @property
    def weighted_scores(self) -> Scores:
        """Type-weighted variant (§5.3 safeguard 3) using the STATED `TYPE_WEIGHTS`.
        Always reported beside the unweighted numbers, never instead of them."""
        precision = (self.w_claims_grounded / self.w_claims) if self.w_claims else None
        coverage = (self.w_reference_covered / self.w_reference) if self.w_reference else None
        return Scores(precision, coverage, _harmonic(precision, coverage),
                      self.n_claims, self.n_reference)

    def as_row(self) -> dict[str, Any]:
        unweighted, weighted = self.scores, self.weighted_scores
        return {
            "metric": self.metric,
            "tier": self.tier,
            "image_id": self.image_id,
            "claim_source": self.claim_source,
            "available": self.available,
            "reason": self.reason,
            "precision": unweighted.precision,
            "coverage": unweighted.coverage,
            self.metric.lower(): unweighted.harmonic_mean,
            "weighted_precision": weighted.precision,
            "weighted_coverage": weighted.coverage,
            f"{self.metric.lower()}_weighted": weighted.harmonic_mean,
            "n_claims": self.n_claims,
            "n_claims_checkable": self.n_claims_checkable,
            "n_claims_grounded": self.n_claims_grounded,
            "n_match_targets": self.n_match_targets,
            "n_reference": self.n_reference,
            "n_reference_covered": self.n_reference_covered,
            "n_hedged_excluded": self.n_hedged_excluded,
            "n_reference_hedged_uncovered": self.n_reference_hedged_uncovered,
            "n_claims_matching_uncertain_gold": self.n_claims_matching_uncertain_gold,
            "n_claims_ungrounded_spans": self.n_claims_ungrounded_spans,
            "n_spans_unresolved": self.n_spans_unresolved,
            "n_reference_unverified": self.n_reference_unverified,
        }

    def __str__(self) -> str:
        s = self.scores
        return (
            f"{self.metric}={_fmt(s.harmonic_mean)} "
            f"(P={_fmt(s.precision)}, coverage={_fmt(s.coverage)}, "
            f"{self.n_claims} claims vs {self.n_reference} ref, tier={self.tier})"
        )


@dataclass
class FactualityReport:
    """Corpus PGF or VCF. Micro pools counts; macro averages defined per-caption values."""

    metric: str
    tier: str
    per_image: list[GroundedFactuality] = field(default_factory=list)
    n_undefined: int = 0
    skipped: list[tuple[str, str]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def micro(self) -> GroundedFactuality:
        """Counts pooled over the corpus, so a two-claim caption does not weigh
        the same as a twenty-claim one."""
        total = GroundedFactuality(
            metric=self.metric, tier=self.tier, image_id="<corpus:micro>",
            claim_source=self.per_image[0].claim_source if self.per_image else "",
            n_claims=sum(r.n_claims for r in self.per_image),
            n_claims_grounded=sum(r.n_claims_grounded for r in self.per_image),
            n_match_targets=sum(r.n_match_targets for r in self.per_image),
            n_reference=sum(r.n_reference for r in self.per_image),
            n_reference_covered=sum(r.n_reference_covered for r in self.per_image),
            n_hedged_excluded=sum(r.n_hedged_excluded for r in self.per_image),
            n_reference_hedged_uncovered=sum(
                r.n_reference_hedged_uncovered for r in self.per_image
            ),
            n_claims_matching_uncertain_gold=sum(
                r.n_claims_matching_uncertain_gold for r in self.per_image
            ),
            n_claims_ungrounded_spans=sum(r.n_claims_ungrounded_spans for r in self.per_image),
            n_spans_unresolved=sum(r.n_spans_unresolved for r in self.per_image),
            n_reference_unverified=sum(r.n_reference_unverified for r in self.per_image),
            w_claims=sum(r.w_claims for r in self.per_image),
            w_claims_grounded=sum(r.w_claims_grounded for r in self.per_image),
            w_reference=sum(r.w_reference for r in self.per_image),
            w_reference_covered=sum(r.w_reference_covered for r in self.per_image),
        )
        return total

    @property
    def macro(self) -> Scores:
        """Mean over captions where the value is defined. Captions whose score is
        undefined are counted in `n_undefined`, never imputed as 0 or 1."""
        scores = [r.scores for r in self.per_image]
        return Scores(
            _mean_defined([s.precision for s in scores]),
            _mean_defined([s.coverage for s in scores]),
            _mean_defined([s.harmonic_mean for s in scores]),
            sum(r.n_claims for r in self.per_image),
            sum(r.n_reference for r in self.per_image),
        )

    @property
    def mean_claims_per_caption(self) -> float | None:
        """Detail. §5.3 rule 2 and §5.5: a factuality number without the claim
        count it was achieved at is not comparable across systems."""
        return _rate(sum(r.n_claims for r in self.per_image), len(self.per_image))

    def as_dict(self) -> dict[str, Any]:
        return {
            "metric": self.metric,
            "tier": self.tier,
            "n_images": len(self.per_image),
            "n_undefined": self.n_undefined,
            "mean_claims_per_caption": self.mean_claims_per_caption,
            "micro": self.micro.as_row(),
            "macro": self.macro._asdict(),
            "per_image": [r.as_row() for r in self.per_image],
            "skipped": self.skipped,
            "claim_sources": sorted({r.claim_source for r in self.per_image}),
            "notes": self.notes + self._provenance_notes()
            + [
                "PGF/VCF precision is never reported without coverage (doc 04 §5.3).",
                "Disclose the §5.4 parser asymmetry in this table's caption.",
            ],
        }

    def _provenance_notes(self) -> list[str]:
        """§5.4 requirement 9, computed rather than left to the caption writer."""
        notes: list[str] = []
        n_parser = sum(
            1 for r in self.per_image if r.claim_source.endswith(CLAIM_PROVENANCE_PARSER)
        )
        n_planner = len(self.per_image) - n_parser
        if n_parser and n_planner:
            notes.append(
                f"MIXED CLAIM SOURCES: {n_planner} images use planner-recorded spans "
                f"(exact) and {n_parser} use an external parser (parser error enters "
                "the metric). Doc 04 §5.4 — these are not comparable and the asymmetry "
                "favours the planner-recorded system; do not average them into one row."
            )
        elif n_parser:
            notes.append(
                f"all {n_parser} images use externally parsed claims: parser error is "
                "inside this number (doc 04 §5.4)."
            )
        n_ungrounded = sum(r.n_claims_ungrounded_spans for r in self.per_image)
        if n_ungrounded:
            notes.append(
                f"{n_ungrounded} claims come from `caption.ungrounded_spans`. PGF "
                "counts them and never grounds them — they ARE the grounding-constraint "
                "violation. VCF leaves them out of its precision denominator "
                "(`n_claims_checkable`), because with no proposition to match there is "
                "nothing to check against gold; they stay in `n_claims`, so coverage and "
                "the detail column are unaffected, and the pessimistic bound is "
                "n_claims_grounded / n_claims."
            )
        n_unresolved = sum(r.n_spans_unresolved for r in self.per_image)
        if n_unresolved:
            notes.append(
                f"{n_unresolved} caption spans point at proposition ids that are not in "
                "the document — a pipeline bug that shrinks m; the claim counts here are "
                "smaller than the caption's real claim count by that many."
            )
        return notes


def _weight(prop: dict) -> float:
    return TYPE_WEIGHTS.get(str(prop.get("type")), DEFAULT_TYPE_WEIGHT)


def _match_matrix(
    claims: Sequence[dict],
    references: Sequence[dict],
    tier: Tier,
    **kwargs: Any,
) -> list[list[bool]]:
    """`hits[i][j]` — does claim i match reference j?

    Existential, not one-to-one: §5.2 asks whether there *exists* a proposition
    the claim traces to, so two claims about the same proposition may both be
    grounded. That is right for grounding, and the coverage term is what stops a
    system from exploiting it by repeating one claim.
    """
    return [
        [bool(match(claim, reference, tier, **kwargs)) for reference in references]
        for claim in claims
    ]


def _grounded_factuality(
    metric: str,
    pair: EvalPair,
    claims: Sequence[dict],
    claim_source: str,
    match_targets: Sequence[dict],
    reference: Sequence[dict],
    tier: Tier,
    *,
    n_hedged_excluded: int = 0,
    n_spans_unresolved: int = 0,
    n_reference_unverified: int = 0,
    hedged_reference_ids: frozenset[str] = frozenset(),
    uncertain_gold: Sequence[dict] = (),
    **kwargs: Any,
) -> GroundedFactuality:
    """Shared core of PGF and VCF — same structure, different reference set (§5.1).

    Two matrices rather than one: precision matches against `match_targets`
    (PGF: P*_SUP) and coverage against `reference` (PGF: all of P*), and for PGF
    those sets differ. Deriving one from the other by identity would break the
    moment a caller passed equal-but-distinct dicts.
    """
    target_hits = _match_matrix(claims, match_targets, tier, **kwargs)
    hits = _match_matrix(claims, reference, tier, **kwargs)

    n_grounded = 0
    w_claims = w_grounded = 0.0
    for i, claim in enumerate(claims):
        weight = _weight(claim)
        # VCF cannot check an ungrounded span against gold, so it does not carry
        # weight in the precision denominator either — see
        # `GroundedFactuality.n_claims_checkable`. PGF weighs every claim.
        if not (metric == "VCF" and claim.get("type") == UNGROUNDED_SPAN_TYPE):
            w_claims += weight
        if any(target_hits[i]):
            n_grounded += 1
            w_grounded += weight

    n_covered = 0
    w_reference = w_covered = 0.0
    n_hedged_uncovered = 0
    for j, ref in enumerate(reference):
        weight = _weight(ref)
        w_reference += weight
        if any(hits[i][j] for i in range(len(claims))):
            n_covered += 1
            w_covered += weight
        elif str(ref.get("id")) in hedged_reference_ids:
            n_hedged_uncovered += 1

    n_uncertain_hit = 0
    if uncertain_gold:
        uncertain_hits = _match_matrix(claims, list(uncertain_gold), tier, **kwargs)
        n_uncertain_hit = sum(1 for row in uncertain_hits if any(row))

    return GroundedFactuality(
        metric=metric,
        tier=tier.value,
        image_id=pair.key(),
        claim_source=claim_source,
        n_claims=len(claims),
        n_claims_grounded=n_grounded,
        n_match_targets=len(match_targets),
        n_reference=len(reference),
        n_reference_covered=n_covered,
        n_hedged_excluded=n_hedged_excluded,
        n_reference_hedged_uncovered=n_hedged_uncovered,
        n_claims_matching_uncertain_gold=n_uncertain_hit,
        n_claims_ungrounded_spans=sum(
            1 for c in claims if c.get("type") == UNGROUNDED_SPAN_TYPE
        ),
        n_spans_unresolved=n_spans_unresolved,
        n_reference_unverified=n_reference_unverified,
        w_claims=w_claims,
        w_claims_grounded=w_grounded,
        w_reference=w_reference,
        w_reference_covered=w_covered,
    )


def pgf(
    pairs: Sequence[EvalPair],
    *,
    tier: Tier = Tier.LEXICAL,
    similarity_fn: Callable[[str, str], float] | None = None,
    soft_threshold: float = 0.75,
) -> FactualityReport:
    """PGF — Proposition-Grounded Factuality (§5.2). **Internal discipline.**

    *Does the caption say only what the system verified?* The reference set is
    P*, the system's own selected propositions. PGF answers a different question
    from VCF and the two must never be conflated: a system can be perfectly
    disciplined about a set of wrong propositions, which is precisely the
    reading "PGF high + VCF low" localises to verification (§5.1).

    Note the deliberate asymmetry of §5.2: precision matches against P*_SUP (a
    claim is grounded only in a SUPPORTED selected proposition) while coverage
    is over all of P*. One consequence is worth stating rather than engineering
    away — an UNCERTAIN proposition admitted as a hedge is excluded from the
    claim set, so it can never be covered and hedging costs coverage. That is
    §5.2 as written; `n_reference_hedged_uncovered` exposes how much of the gap
    it accounts for.
    """
    report = FactualityReport(
        metric="PGF", tier=tier.value, notes=_tier_notes(tier, similarity_fn)
    )

    for pair in pairs:
        selected = _selected(pair.predicted)
        if selected is None:
            report.skipped.append((pair.key(), "tài liệu không có khối selection (P* undefined)"))
            continue
        claim_set = caption_claims(pair)
        if not claim_set.available:
            report.skipped.append((pair.key(), claim_set.reason))
            continue

        supported = [p for p in selected if _verdict(p) == "SUPPORTED"]
        hedged_ids = frozenset(
            str(i)
            for i in ((pair.predicted.get("selection") or {}).get("uncertain_admitted_ids") or [])
        ) | frozenset(str(p.get("id")) for p in selected if _verdict(p) == "UNCERTAIN")

        result = _grounded_factuality(
            "PGF", pair, claim_set.claims, claim_set.claim_source,
            supported, selected, tier,
            n_hedged_excluded=claim_set.n_hedged_excluded,
            n_spans_unresolved=claim_set.n_spans_unresolved,
            # Reported, NOT skipped: `status: None` is a legitimate pre-
            # `assert_clean` state of the system's own P* (doc 08), and PGF is
            # a question about the system agreeing with itself.
            n_reference_unverified=sum(1 for p in selected if _verdict(p) is None),
            hedged_reference_ids=hedged_ids,
            **_match_kwargs(pair, similarity_fn, soft_threshold),
        )
        report.per_image.append(result)
        if result.scores.harmonic_mean is None:
            report.n_undefined += 1

    return report


def vcf(
    pairs: Sequence[EvalPair],
    *,
    tier: Tier = Tier.LEXICAL,
    similarity_fn: Callable[[str, str], float] | None = None,
    soft_threshold: float = 0.75,
) -> FactualityReport:
    """VCF — Verified Caption Factuality (§5.2). **External correctness.**

    *Is what the caption says actually true?* The reference set is G_true: gold
    propositions that are SUPPORTED and not adversarial. An adversarial gold is
    false by construction and a REJECTED gold is false outright, so neither can
    make a claim true.

    A claim matching a gold proposition the annotators marked UNCERTAIN is
    **not** counted as true — §4.2's cost₃ names asserting an undeterminable
    claim a hallucination. It is reported separately in
    `n_claims_matching_uncertain_gold` so the composition of the precision loss
    stays visible instead of being attributed to outright falsehood.

    An image whose gold carries verdicts on **some** propositions and not others
    is skipped, loudly. G_true is `status == SUPPORTED`, so an unverdicted gold
    proposition drops silently out of the coverage denominator: with three of
    four gold propositions unverdicted, `Có một người.` — §5.3's own worked
    example of the caption that must score 0.095 — came out at VCF 1.000.
    """
    report = FactualityReport(
        metric="VCF", tier=tier.value, notes=_tier_notes(tier, similarity_fn)
    )

    for pair in pairs:
        claim_set = caption_claims(pair)
        if not claim_set.available:
            report.skipped.append((pair.key(), claim_set.reason))
            continue

        gold = _props(pair.gold)
        unverdicted = [p for p in gold if _verdict(p) is None]
        if unverdicted:
            report.skipped.append((
                pair.key(),
                f"{len(unverdicted)}/{len(gold)} mệnh đề vàng không có "
                "`verification.status` — G_true would silently exclude them and "
                "inflate VCF coverage; gold verdicts are mandatory (doc 03 §1.1)",
            ))
            continue
        gold_true = [p for p in gold if not _is_false_gold(p) and _verdict(p) == "SUPPORTED"]
        gold_uncertain = [p for p in gold if not _is_false_gold(p) and _verdict(p) == "UNCERTAIN"]

        result = _grounded_factuality(
            "VCF", pair, claim_set.claims, claim_set.claim_source,
            gold_true, gold_true, tier,
            n_hedged_excluded=claim_set.n_hedged_excluded,
            n_spans_unresolved=claim_set.n_spans_unresolved,
            uncertain_gold=gold_uncertain,
            **_match_kwargs(pair, similarity_fn, soft_threshold),
        )
        report.per_image.append(result)
        if result.scores.harmonic_mean is None:
            report.n_undefined += 1

    return report


# ---------------------------------------------------------------------------
# The whole doc-04 table, at every tier
# ---------------------------------------------------------------------------
def evaluate(
    pairs: Sequence[EvalPair],
    *,
    tiers: Sequence[Tier] = (Tier.EXACT, Tier.LEXICAL),
    similarity_fn: Callable[[str, str], float] | None = None,
    soft_threshold: float = 0.75,
) -> dict[str, Any]:
    """Groups 3, 4 and 5 at each tier — the shape doc 04 §6 requires.

    Two tiers by default (requirement 2), and both of them are strict: EXACT and
    LEXICAL. **`Tier.SOFT` is deliberately not in the default** — it is not
    implemented anywhere below this module. `matching.match()` takes a
    `similarity_fn` and never calls it, so a soft run returns lexical numbers no
    matter what is passed; any report at that tier therefore carries the
    `_SOFT_NOT_IMPLEMENTED` note, and the numbers must not be published in a soft
    column until matching itself grows an embedding tier.

    Fluency is computed once, outside the tier loop, and kept in its own key —
    it is reference-free, and nesting it under a matching tier would suggest it
    is a factuality number (§3.3).
    """
    out: dict[str, Any] = {
        "n_images": len(pairs),
        "fluency_not_factuality": vietnamese_fluency(pairs).as_dict(),
        "by_tier": {},
    }
    for tier in tiers:
        kwargs = {"similarity_fn": similarity_fn, "soft_threshold": soft_threshold}
        out["by_tier"][tier.value] = {
            "hallucination": hallucination_rates(pairs, tier=tier, **kwargs).as_dict(),
            "vietnamese_factuality": vietnamese_factuality(pairs, tier=tier, **kwargs).as_dict(),
            "proposition_prf_candidate_set": proposition_prf(
                pairs, tier=tier, source=PropositionSet.ALL, **kwargs
            ).as_dict(),
            "verification": verification_quality(pairs, tier=tier, **kwargs).as_dict(),
            "pgf": pgf(pairs, tier=tier, **kwargs).as_dict(),
            "vcf": vcf(pairs, tier=tier, **kwargs).as_dict(),
        }
    return out


__all__ = [
    "EvalPair", "PropositionSet", "Tier",
    "ClaimSet", "caption_claims", "UNGROUNDED_SPAN_TYPE",
    "CLAIM_PROVENANCE_PLANNER", "CLAIM_PROVENANCE_PARSER", "FALSE_GOLD_SIGNALS",
    "EntityAlignment", "align_entities", "bbox_iou_fn",
    "HallucinationRow", "HallucinationReport", "hallucination_rates",
    "VietnameseFactualityReport", "vietnamese_factuality", "extract_color_term",
    "VietnameseFluencyReport", "vietnamese_fluency",
    "PRF", "PropositionReport", "proposition_prf",
    "VerdictPRF", "VerificationReport", "verification_quality",
    "Scores", "GroundedFactuality", "FactualityReport", "pgf", "vcf",
    "evaluate",
    "DOC04_BUCKET", "UNBUCKETED_TYPES", "TYPE_WEIGHTS", "VERDICTS",
    "REJECT_SUBLABEL_FIELD", "REJECT_SUBLABELS",
]
