"""`rescap.svp` — the proposition layer: matching, alignment, entailment.

**Reusable.** No imports from `rescap.pipeline`, and no model dependency —
`matching.py` takes a similarity function by injection, so it runs on a machine
with no embedding stack.

The matching rule (`formulation/02` §7) lives here once. Every metric in doc 04
calls it, because two different notions of "match" would turn a scoring
difference into an apparent improvement.
"""

from .entailment import (
    entailment_pairs,
    entailment_reason,
    entails,
    hypernym_depth,
    maximal_ids,
    specificity_level,
    subsumes,
)
from .matching import (
    MatchResult,
    Tier,
    align,
    canonical,
    compatible_categories,
    match,
    match_any,
)

__all__ = [
    "Tier", "MatchResult", "match", "match_any", "align",
    "canonical", "compatible_categories",
    # entailment / specificity (formulation/02 §6)
    "entails", "entailment_reason", "entailment_pairs", "maximal_ids",
    "specificity_level", "hypernym_depth", "subsumes",
]
