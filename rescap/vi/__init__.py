"""`rescap.vi` — the Vietnamese language layer.

**Reusable by design.** Nothing here may import from `rescap.pipeline`: this
layer is meant to outlive the paper it was written for and serve later
Vietnamese vision-language work (`research/MODEL-SELECTION.md` §6).

What it handles, and why each matters for factuality rather than only fluency:

| Module | Phenomenon | Why it is not cosmetic |
|---|---|---|
| `lexicon` | all Vietnamese word lists, in one inspectable place | — |
| `color` | **`xanh` = blue AND green** | a wrong resolution is a colour hallucination invisible to English-derived metrics |
| `classifier` | loại từ, and post-nominal adjective order | classifier errors are FLUENCY; adjective order detects MT artefacts |
| `gender` | gender is **lexical in the noun**, not just pronominal | guessing it is a CONTENT error, so the default is neutral `người` |
| `segment` | whitespace separates syllables, not words | without segmentation every n-gram metric measures the wrong unit |
| `normalize` | NFC, synonyms, aspect stripping, canonical forms | makes two propositions comparable at all |
"""

from .classifier import (
    NounPhrase,
    build_noun_phrase,
    check_agreement,
    classifier_for,
    parse_noun_phrase,
    strip_classifier,
)
from .color import (
    ColorErrorType,
    ColorReading,
    Xanh,
    classify_color_error,
    color_error_rates,
    needs_disambiguation,
    parse_color,
)

__all__ = [
    # colour
    "Xanh", "ColorReading", "ColorErrorType",
    "parse_color", "needs_disambiguation", "classify_color_error", "color_error_rates",
    # classifiers / noun phrases
    "NounPhrase", "parse_noun_phrase", "build_noun_phrase",
    "classifier_for", "check_agreement", "strip_classifier",
]
