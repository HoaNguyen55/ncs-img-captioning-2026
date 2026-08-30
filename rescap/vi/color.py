"""The `xanh` problem — Vietnamese colour resolution.

Vietnamese `xanh` denotes **both blue and green**. English has no equivalent
collapse, so no English-derived pipeline or metric can see this failure mode.
It is one of the paper's Vietnamese-specific contributions
(`formulation/02 §4.5`).

Three outcomes, kept distinct because they have different causes and different
fixes:

    xanh dương / xanh lam / xanh nước biển  ->  BLUE
    xanh lá   / xanh lục                    ->  GREEN
    xanh      (bare)                        ->  AMBIGUOUS  -- never resolved silently

And two error types that must never be merged in reporting:

    CONFUSION        blue asserted where gold is green (or vice versa)
                     -> a genuine colour hallucination
    UNDER-SPEC       bare `xanh` where gold is resolved
                     -> a different error, with a different cause, and NOT a
                        hallucination: the model said something true but vague
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .lexicon import COLOR_MODIFIERS, COLORS, XANH_AMBIGUOUS, XANH_BLUE, XANH_GREEN


class Xanh(str, Enum):
    """Resolution of a `xanh` expression."""

    BLUE = "xanh_dương"
    GREEN = "xanh_lá"
    UNRESOLVED = "xanh_không_xác_định"


class ColorErrorType(str, Enum):
    """How a predicted colour can differ from gold. Reported separately."""

    CORRECT = "correct"
    XANH_CONFUSION = "xanh_confusion"        # blue <-> green: a hallucination
    XANH_UNDERSPECIFIED = "xanh_underspec"   # bare `xanh` vs a resolved gold
    XANH_OVERSPECIFIED = "xanh_overspec"     # resolved prediction vs ambiguous gold
    OTHER_COLOR_ERROR = "other_color_error"  # e.g. red vs black
    NOT_COMPARABLE = "not_comparable"


@dataclass(frozen=True)
class ColorReading:
    """The result of parsing a colour expression."""

    raw: str
    canonical: str | None          # canonical colour term, or None if unparsed
    is_xanh: bool
    xanh_value: Xanh | None        # set only when is_xanh
    modifier: str | None           # đậm / nhạt / ...

    @property
    def resolved(self) -> bool:
        """True when this reading commits to a definite colour."""
        if not self.is_xanh:
            return self.canonical is not None
        return self.xanh_value in (Xanh.BLUE, Xanh.GREEN)


def parse_color(text: str) -> ColorReading:
    """Parse a Vietnamese colour expression.

    >>> parse_color("xanh dương").xanh_value
    <Xanh.BLUE: 'xanh_dương'>
    >>> parse_color("xanh").resolved
    False
    >>> parse_color("đỏ đậm").modifier
    'đậm'
    """
    raw = text.strip().lower()
    if not raw:
        return ColorReading(text, None, False, None, None)

    modifier = None
    tokens = raw.split()
    if tokens and tokens[-1] in COLOR_MODIFIERS:
        modifier = tokens[-1]
        tokens = tokens[:-1]
    core = " ".join(tokens)

    if core in XANH_BLUE:
        return ColorReading(text, "xanh dương", True, Xanh.BLUE, modifier)
    if core in XANH_GREEN:
        return ColorReading(text, "xanh lá", True, Xanh.GREEN, modifier)
    if core == XANH_AMBIGUOUS:
        # Deliberately NOT resolved. A generator that emits bare `xanh` has not
        # committed to a colour, and guessing here would manufacture a fact.
        return ColorReading(text, None, True, Xanh.UNRESOLVED, modifier)
    if core in COLORS:
        return ColorReading(text, core, False, None, modifier)
    return ColorReading(text, None, False, None, modifier)


def needs_disambiguation(text: str) -> bool:
    """True when the expression is a bare `xanh` and must be resolved or hedged."""
    return parse_color(text).xanh_value is Xanh.UNRESOLVED


def classify_color_error(predicted: str, gold: str) -> ColorErrorType:
    """Compare a predicted colour against gold.

    The four `xanh` outcomes are separated on purpose. Collapsing them would
    hide the phenomenon the paper is about: a system that says `xanh` where gold
    says `xanh lá` is **vague**, while one that says `xanh dương` there is
    **wrong**, and those call for different fixes.

    >>> classify_color_error("xanh dương", "xanh lá")
    <ColorErrorType.XANH_CONFUSION: 'xanh_confusion'>
    >>> classify_color_error("xanh", "xanh lá")
    <ColorErrorType.XANH_UNDERSPECIFIED: 'xanh_underspec'>
    """
    p, g = parse_color(predicted), parse_color(gold)

    if p.canonical is None and not p.is_xanh:
        return ColorErrorType.NOT_COMPARABLE
    if g.canonical is None and not g.is_xanh:
        return ColorErrorType.NOT_COMPARABLE

    if p.is_xanh and g.is_xanh:
        if p.xanh_value is Xanh.UNRESOLVED and g.xanh_value is Xanh.UNRESOLVED:
            return ColorErrorType.CORRECT       # both vague, equally so
        if p.xanh_value is Xanh.UNRESOLVED:
            return ColorErrorType.XANH_UNDERSPECIFIED
        if g.xanh_value is Xanh.UNRESOLVED:
            # Gold could not tell; the model committed. Not a hallucination
            # against a known truth -- flagged separately so it is never counted
            # as one.
            return ColorErrorType.XANH_OVERSPECIFIED
        return (
            ColorErrorType.CORRECT
            if p.xanh_value is g.xanh_value
            else ColorErrorType.XANH_CONFUSION
        )

    if p.is_xanh != g.is_xanh:
        return ColorErrorType.OTHER_COLOR_ERROR

    return (
        ColorErrorType.CORRECT
        if p.canonical == g.canonical
        else ColorErrorType.OTHER_COLOR_ERROR
    )


def color_error_rates(pairs: list[tuple[str, str]]) -> dict[str, float]:
    """Per-type colour error rates over (predicted, gold) pairs.

    Returns `xanh_confusion_rate` and `xanh_underspec_rate` **separately** —
    `formulation/04 §3.3` requires it, because merging them would attribute a
    vagueness problem to hallucination.
    """
    if not pairs:
        return {}

    counts: dict[str, int] = {t.value: 0 for t in ColorErrorType}
    for predicted, gold in pairs:
        counts[classify_color_error(predicted, gold).value] += 1

    comparable = len(pairs) - counts[ColorErrorType.NOT_COMPARABLE.value]
    denom = max(1, comparable)

    return {
        "n_pairs": len(pairs),
        "n_comparable": comparable,
        "accuracy": counts[ColorErrorType.CORRECT.value] / denom,
        "xanh_confusion_rate": counts[ColorErrorType.XANH_CONFUSION.value] / denom,
        "xanh_underspec_rate": counts[ColorErrorType.XANH_UNDERSPECIFIED.value] / denom,
        "xanh_overspec_rate": counts[ColorErrorType.XANH_OVERSPECIFIED.value] / denom,
        "other_color_error_rate": counts[ColorErrorType.OTHER_COLOR_ERROR.value] / denom,
        "counts": counts,
    }


def colour_term(value: str, *, canonicalise=None) -> str | None:
    """The longest colour expression inside a descriptive value, or None.

    **One implementation, because there were two and they disagreed.**
    `pipeline.verify` scanned left to right for the longest match;
    `metrics_svp` iterated a `set` of colour terms and returned the first hit,
    so its answer depended on set iteration order. On `đen và nâu` one said
    `đen` and the other `nâu` -- verification believing one colour while the
    metric scored a different one, on the axis this work is about.

    Left-to-right, longest match, deterministic.

    **Known limitation, stated rather than papered over:** a value naming two
    colours (`đen và nâu`) returns only the first. That is not a claim about
    which colour is right -- it is a reduction, and a genuinely two-coloured
    object is not represented. Recorded here so the caller can see it; handling
    it properly needs a multi-valued attribute, which the schema does not have.

    >>> colour_term("áo xanh dương")
    'xanh dương'
    >>> colour_term("quần bò") is None
    True
    >>> colour_term("đen và nâu")
    'đen'
    """
    text = canonicalise(value) if canonicalise else str(value or "").lower()
    tokens = text.split()
    best = ""
    for start in range(len(tokens)):
        for length in range(min(3, len(tokens) - start), 0, -1):
            candidate = " ".join(tokens[start : start + length])
            reading = parse_color(candidate)
            if (reading.canonical or reading.is_xanh) and len(candidate) > len(best):
                best = candidate
    return best or None
