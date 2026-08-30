"""Vietnamese classifiers (loại từ) and noun-phrase construction.

    NUMERAL + CLASSIFIER + NOUN + ADJECTIVE
      ba        con        chó      nâu        "three brown dogs"
      một      chiếc       áo       đỏ         "a red shirt"

Two rules this module exists to enforce:

1. **A classifier is grammatical agreement, not a property of the referent.**
   `ba con chó` vs `ba cái ghế` differ because dogs are animate, not because
   the dogs are "con-coloured". So a classifier error is a **fluency** error,
   never a factuality one — it is scored under Fluency in human evaluation and
   is excluded from attribute precision (`formulation/02 §4.1`).

2. **Adjectives follow the noun.** `áo đỏ`, never `đỏ áo`. Pre-nominal
   adjectives are the single most common machine-translation artefact in
   Vietnamese and are detected here as a data-quality signal
   (`formulation/03 §4`).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .lexicon import (
    CLASSIFIER_DEFAULT,
    CLASSIFIERS,
    COLOR_MODIFIERS,
    COLORS,
    NOUN_CLASSIFIER,
    NUMBER_MARKERS,
    NUMERALS,
)


#: Grammatical words that cannot sit inside a noun compound. Used only when
#: extending an unknown head noun to its second syllable: hitting one of these
#: means the noun ended at the first.
_NOT_IN_COMPOUND: frozenset[str] = frozenset({
    "ở", "của", "và", "với", "trong", "trên", "dưới", "ngoài", "giữa",
    "phía", "bên", "cạnh", "gần", "xa", "sau", "trước",
    "đang", "có", "là", "bị", "được", "cho", "từ", "đến", "về",
    "các", "những", "này", "kia", "đó", "khác", "nữa", "rất", "hơn", "nhất",
})


@dataclass
class NounPhrase:
    """A parsed Vietnamese noun phrase."""

    raw: str
    numeral: int | None = None
    number_marker: str | None = None
    classifier: str | None = None
    head_noun: str = ""
    modifiers: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def surface(self) -> str:
        """Realise back to Vietnamese, adjectives after the noun."""
        parts: list[str] = []
        if self.numeral is not None:
            parts.append(_numeral_word(self.numeral))
        elif self.number_marker:
            parts.append(self.number_marker)
        if self.classifier:
            parts.append(self.classifier)
        parts.append(self.head_noun)
        parts.extend(self.modifiers)  # POST-nominal: this ordering is the rule
        return " ".join(p for p in parts if p)


def _numeral_word(value: int) -> str:
    for word, number in NUMERALS.items():
        if number == value:
            return word
    return str(value)


def classifier_for(head_noun: str) -> tuple[str, bool]:
    """Return `(classifier, known)` for a head noun.

    `known=False` means we fell back to the default and the caller should log
    it — a silently wrong classifier is a fluency bug that is hard to trace
    later.
    """
    noun = head_noun.strip().lower()
    if noun in NOUN_CLASSIFIER:
        return NOUN_CLASSIFIER[noun], True
    # Multi-word nouns: try the last word ("xe đạp thể thao" -> "xe đạp"?).
    tokens = noun.split()
    for start in range(len(tokens)):
        candidate = " ".join(tokens[start:])
        if candidate in NOUN_CLASSIFIER:
            return NOUN_CLASSIFIER[candidate], True
    return CLASSIFIER_DEFAULT, False


def check_agreement(classifier: str, head_noun: str) -> bool | None:
    """Does this classifier agree with this noun?

    Returns None when the noun is not in the lexicon — **unknown is not
    disagreement**, and reporting it as an error would penalise vocabulary
    coverage as if it were a grammar mistake.
    """
    expected, known = classifier_for(head_noun)
    if not known:
        return None
    return classifier.strip().lower() == expected


_TOKEN = re.compile(r"[^\s]+")


def parse_noun_phrase(text: str) -> NounPhrase:
    """Parse `một chiếc áo đỏ` into its parts.

    Deliberately shallow: a real parser is out of scope, and this only has to
    handle the noun phrases our own generator emits plus common annotator
    forms. Anything it cannot parse comes back with the head noun set to the
    whole string and a warning, never a silent guess.

    >>> np = parse_noun_phrase("ba con chó nâu")
    >>> np.numeral, np.classifier, np.head_noun, np.modifiers
    (3, 'con', 'chó', ['nâu'])
    """
    raw = text.strip()
    tokens = _TOKEN.findall(raw.lower())
    phrase = NounPhrase(raw=raw)
    if not tokens:
        return phrase

    index = 0

    # 1. numeral or number marker (multi-word markers first: "một vài")
    if index + 1 < len(tokens) and f"{tokens[index]} {tokens[index + 1]}" in NUMBER_MARKERS:
        phrase.number_marker = f"{tokens[index]} {tokens[index + 1]}"
        index += 2
    elif tokens[index] in NUMERALS:
        phrase.numeral = NUMERALS[tokens[index]]
        if phrase.numeral == 1:
            phrase.number_marker = "một"
        index += 1
    elif tokens[index].isdigit():
        phrase.numeral = int(tokens[index])
        index += 1
    elif tokens[index] in NUMBER_MARKERS:
        phrase.number_marker = tokens[index]
        index += 1

    # 2. classifier
    if index < len(tokens) and tokens[index] in CLASSIFIERS:
        phrase.classifier = tokens[index]
        index += 1

    if index >= len(tokens):
        phrase.head_noun = raw
        phrase.warnings.append("no head noun found after numeral/classifier")
        return phrase

    # 3. head noun -- scan the WHOLE remainder, not just its prefix, and take
    # the longest lexicon match. Scanning only the prefix would mis-parse the
    # very artefact this module exists to catch: in `đỏ áo` the prefix search
    # finds nothing, falls back to the first token, and silently declares the
    # colour to be the head noun.
    remaining = tokens[index:]
    head_offset: int | None = None
    for start in range(len(remaining)):
        for length in range(min(3, len(remaining) - start), 0, -1):
            candidate = " ".join(remaining[start : start + length])
            if candidate in NOUN_CLASSIFIER:
                phrase.head_noun = candidate
                head_offset = start
                index += start + length
                break
        if head_offset is not None:
            break

    if head_offset is None:
        # Unknown noun. Do NOT blindly take the first token: in a malformed
        # phrase that token is often a colour, and declaring the colour to be
        # the head noun hides the very error we want to surface. Skip leading
        # colours, size/shape modifiers and stray classifiers first.
        skip = 0
        while skip < len(remaining) - 1 and (
            remaining[skip] in COLORS
            or remaining[skip] in COLOR_MODIFIERS
            or remaining[skip] in CLASSIFIERS
        ):
            skip += 1
        # Take the compound, not its first syllable. Vietnamese writes
        # multi-syllable nouns with spaces, so one token is usually half a word:
        # `bánh lái` (a rudder) truncated to `bánh` turned a boat's rudder into
        # a cake and put `một vài cái bánh` in a caption of an exhibition room.
        # `ánh sáng` became `ánh`, `khu rừng` became `khu`.
        #
        # Two syllables, because that is what the overwhelming majority of
        # Vietnamese compounds are, and because a third would start swallowing
        # the modifiers this function exists to separate out. Extension stops at
        # anything that cannot be inside a noun: a colour, a size word, a
        # classifier, a numeral, or a noun the lexicon already knows, since a
        # known noun begins a phrase of its own.
        span = 1
        if skip + 1 < len(remaining):
            nxt = remaining[skip + 1]
            if not (
                nxt in COLORS
                or nxt in COLOR_MODIFIERS
                or nxt in CLASSIFIERS
                or nxt in NUMERALS
                or nxt in NUMBER_MARKERS
                or nxt in NOUN_CLASSIFIER
                or nxt in _NOT_IN_COMPOUND
            ):
                span = 2

        phrase.head_noun = " ".join(remaining[skip : skip + span])
        head_offset = skip
        index += skip + span
        phrase.warnings.append(
            f"head noun {phrase.head_noun!r} not in the lexicon"
            + (f"; inferred by skipping {remaining[:skip]!r}" if skip else "")
            + ("; read as a compound" if span > 1 else "")
        )

    # 4. modifiers -- everything after the head noun, plus anything that
    # appeared BEFORE it (which is the word-order violation, recorded below).
    pre_nominal = remaining[:head_offset]
    phrase.modifiers = pre_nominal + tokens[index:]

    # Adjective order: Vietnamese places adjectives AFTER the noun. Anything
    # before it is the classic machine-translation artefact (formulation/03 §4).
    if pre_nominal:
        pre_colors = [t for t in pre_nominal if t in COLORS]
        phrase.warnings.append(
            f"pre-nominal modifier(s) {pre_nominal!r} before head noun "
            f"{phrase.head_noun!r}: Vietnamese places adjectives AFTER the noun "
            f"(expected {phrase.head_noun!r} {' '.join(pre_nominal)!r})"
            + (" — colour term, a common MT artefact" if pre_colors else "")
        )

    if phrase.classifier is not None:
        agrees = check_agreement(phrase.classifier, phrase.head_noun)
        if agrees is False:
            expected, _ = classifier_for(phrase.head_noun)
            phrase.warnings.append(
                f"classifier {phrase.classifier!r} does not agree with "
                f"{phrase.head_noun!r} (expected {expected!r}) — a FLUENCY error, "
                "not a factual one"
            )

    return phrase


def build_noun_phrase(
    head_noun: str,
    count: int | None = None,
    marker: str | None = None,
    modifiers: list[str] | None = None,
    exact: bool = True,
) -> NounPhrase:
    """Construct a well-formed Vietnamese noun phrase for realisation.

    When `exact=False` the count is realised approximately (`một vài`, `nhiều`)
    instead of asserting a number — required when counting confidence is low,
    because asserting "ba con chó" on two visible dogs is a counting
    hallucination (`formulation/02 §4.2`).
    """
    classifier, known = classifier_for(head_noun)
    phrase = NounPhrase(
        raw="",
        head_noun=head_noun,
        classifier=classifier,
        modifiers=list(modifiers or []),
    )
    if not known:
        phrase.warnings.append(
            f"no classifier known for {head_noun!r}; defaulted to "
            f"{CLASSIFIER_DEFAULT!r} — verify before publishing"
        )

    if count is not None and exact:
        phrase.numeral = count
    elif count is not None:
        phrase.number_marker = "một vài" if count <= 3 else "nhiều"
    elif marker:
        phrase.number_marker = marker

    phrase.raw = phrase.surface()
    return phrase


def strip_classifier(text: str) -> str:
    """Reduce a noun phrase to its bare head noun, for entity normalisation.

    `một chiếc xe đạp` -> `xe đạp`  (`formulation/02 §5.3`)
    """
    return parse_noun_phrase(text).head_noun
