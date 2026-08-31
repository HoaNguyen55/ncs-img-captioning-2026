#!/usr/bin/env python
"""Automatic error triage for generated captions.

    python scripts/error_analysis.py \
        --predictions <run>/results/test_predictions.json \
        --experiment A0_cnn_lstm_baseline \
        --out research/results/error_analysis

**This is a triage tool, not a labelling tool.** The categories below are
lexical heuristics: they surface *candidates* fast so a human can look at a few
hundred failures instead of five thousand. Every case is written out with
`verified: false`, and no count from this script belongs in a paper until a
human has confirmed a sample and the agreement rate is reported.

Known blind spots, by construction:
  * synonyms ("man"/"guy"/"person") are only partly handled by SYNONYMS below
  * plurals are stripped naively
  * no parser, so "relation" errors are detected only via preposition mismatch
  * a caption can legitimately mention something no reference happens to name
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Lexicons -- deliberately small and inspectable
# ---------------------------------------------------------------------------
NUMBER_WORDS = {
    "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
    "single", "pair", "couple", "several", "many", "few", "group", "crowd",
}

SPATIAL_WORDS = {
    "on", "in", "under", "above", "below", "behind", "front", "next", "beside",
    "near", "between", "over", "across", "against", "inside", "outside", "left",
    "right", "top", "bottom", "middle", "beneath", "onto", "into",
}

COLOR_WORDS = {
    "red", "blue", "green", "yellow", "black", "white", "brown", "orange",
    "purple", "pink", "grey", "gray", "golden", "silver", "dark", "light",
}

# A caption with none of these is very likely a fragment.
COMMON_VERBS = {
    "is", "are", "sits", "sitting", "stands", "standing", "walks", "walking",
    "runs", "running", "holds", "holding", "plays", "playing", "jumps",
    "jumping", "rides", "riding", "looks", "looking", "wears", "wearing",
    "eats", "eating", "lies", "lying", "climbs", "climbing", "swims",
    "swimming", "throws", "throwing", "catches", "catching", "watches",
    "watching", "talks", "talking", "flies", "flying",
}

STOPWORDS = {
    "a", "an", "the", "of", "and", "or", "with", "to", "at", "by", "from",
    "his", "her", "its", "their", "this", "that", "these", "those", "some",
    "while", "as", "it", "there", "he", "she", "they",
} | SPATIAL_WORDS | COMMON_VERBS

# Decoder artefacts, not claims. `<unk>` means the model failed to produce a
# word at all; counting it as a hallucinated object inflates the rate and
# misattributes a vocabulary-coverage problem to factuality.
NON_CLAIM_TOKENS = {"unk", "pad", "start", "end"}

SYNONYMS = {
    "man": "person", "woman": "person", "boy": "person", "girl": "person",
    "guy": "person", "lady": "person", "child": "person", "kid": "person",
    "male": "person", "female": "person", "people": "person", "men": "person",
    "women": "person", "player": "person", "puppy": "dog", "doggy": "dog",
    "kitten": "cat", "bike": "bicycle", "photo": "picture", "image": "picture",
}

CATEGORIES = [
    "object_hallucination",
    "missing_object",
    "wrong_attribute",
    "counting_error",
    "spatial_error",
    "semantic_mismatch",
    "repetition",
    "incomplete_caption",
    "correct_or_minor",
]

TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def normalise(token: str) -> str:
    """Crude lemmatisation: strip a trailing plural 's', then map synonyms."""
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        token = token[:-1]
    return SYNONYMS.get(token, token)


def content_words(text: str) -> set[str]:
    return {
        normalise(t)
        for t in tokenize(text)
        if t not in STOPWORDS and t not in NON_CLAIM_TOKENS and len(t) > 2
    }


# ---------------------------------------------------------------------------
def categorise(prediction: str, references: list[str]) -> tuple[list[str], dict[str, Any]]:
    """Return (categories, evidence) for one prediction.

    A caption can land in several categories -- they are not mutually exclusive,
    and forcing a single label would hide compound failures.
    """
    pred_tokens = tokenize(prediction)
    pred_content = content_words(prediction)

    ref_content_sets = [content_words(r) for r in references]
    ref_union = set().union(*ref_content_sets) if ref_content_sets else set()
    # "Consensus" words: named by more than half the references.
    ref_counts = Counter(w for s in ref_content_sets for w in s)
    threshold = max(1, len(references) // 2 + 1)
    ref_consensus = {w for w, c in ref_counts.items() if c >= threshold}

    categories: list[str] = []
    evidence: dict[str, Any] = {}

    # -- content overlap ----------------------------------------------------
    hallucinated = pred_content - ref_union
    if hallucinated:
        categories.append("object_hallucination")
        evidence["hallucinated"] = sorted(hallucinated)

    missing = ref_consensus - pred_content
    if missing:
        categories.append("missing_object")
        evidence["missing"] = sorted(missing)

    # -- attributes ---------------------------------------------------------
    pred_colors = {t for t in pred_tokens if t in COLOR_WORDS}
    ref_colors = {t for r in references for t in tokenize(r) if t in COLOR_WORDS}
    if pred_colors and not (pred_colors & ref_colors):
        categories.append("wrong_attribute")
        evidence["attribute"] = {"predicted": sorted(pred_colors), "references": sorted(ref_colors)}

    # -- counting -----------------------------------------------------------
    pred_numbers = {t for t in pred_tokens if t in NUMBER_WORDS}
    ref_numbers = {t for r in references for t in tokenize(r) if t in NUMBER_WORDS}
    if pred_numbers and not (pred_numbers & ref_numbers):
        categories.append("counting_error")
        evidence["counting"] = {"predicted": sorted(pred_numbers), "references": sorted(ref_numbers)}

    # -- spatial relations --------------------------------------------------
    pred_spatial = {t for t in pred_tokens if t in SPATIAL_WORDS}
    ref_spatial = {t for r in references for t in tokenize(r) if t in SPATIAL_WORDS}
    if pred_spatial and ref_spatial and not (pred_spatial & ref_spatial):
        categories.append("spatial_error")
        evidence["spatial"] = {"predicted": sorted(pred_spatial), "references": sorted(ref_spatial)}

    # -- fluency ------------------------------------------------------------
    bigrams = list(zip(pred_tokens, pred_tokens[1:]))
    repeated = [bg for bg, c in Counter(bigrams).items() if c > 1]
    if repeated or (len(pred_tokens) > 3 and len(set(pred_tokens)) / len(pred_tokens) < 0.6):
        categories.append("repetition")
        evidence["repeated_bigrams"] = [" ".join(bg) for bg in repeated]

    if len(pred_tokens) < 4 or not any(t in COMMON_VERBS for t in pred_tokens):
        categories.append("incomplete_caption")
        evidence["length"] = len(pred_tokens)

    # -- overall semantic agreement ----------------------------------------
    overlap = len(pred_content & ref_union) / max(1, len(pred_content | ref_union))
    evidence["jaccard_overlap"] = round(overlap, 3)
    if overlap < 0.15:
        categories.append("semantic_mismatch")

    if not categories:
        categories.append("correct_or_minor")

    return categories, evidence


# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True,
                        help="JSON: {image_id: {prediction, references}} or {image_id: caption}")
    parser.add_argument("--references", default=None,
                        help="separate JSON {image_id: [refs]} if not embedded above")
    parser.add_argument("--experiment", default="unknown")
    parser.add_argument("--out", default=None)
    parser.add_argument("--max-cases", type=int, default=200,
                        help="cases written per category (they are for human review)")
    args = parser.parse_args()

    data = json.loads(Path(args.predictions).read_text())
    external_refs = json.loads(Path(args.references).read_text()) if args.references else {}

    out_dir = Path(args.out) if args.out else Path(__file__).resolve().parents[1] / "results" / "error_analysis"
    out_dir = out_dir / args.experiment
    out_dir.mkdir(parents=True, exist_ok=True)

    buckets: dict[str, list[dict]] = defaultdict(list)
    counts: Counter[str] = Counter()
    total = 0

    for image_id, value in data.items():
        if isinstance(value, dict):
            prediction = value.get("prediction", "")
            references = value.get("references", [])
        else:
            prediction, references = value, external_refs.get(image_id, [])
        if not references:
            continue

        total += 1
        categories, evidence = categorise(prediction, references)
        for category in categories:
            counts[category] += 1
            if len(buckets[category]) < args.max_cases:
                buckets[category].append(
                    {
                        "image": image_id,
                        "ground_truth": references,
                        "prediction": prediction,
                        "model": args.experiment,
                        "experiment": args.experiment,
                        "error_category": category,
                        "all_categories": categories,
                        "evidence": evidence,
                        "analysis": "",        # <- human fills this in
                        "verified": False,     # <- flip to true after human review
                    }
                )

    for category, cases in buckets.items():
        (out_dir / f"{category}.json").write_text(json.dumps(cases, indent=2))

    summary = {
        "experiment": args.experiment,
        "predictions_file": str(Path(args.predictions).resolve()),
        "total_captions": total,
        "counts": dict(counts.most_common()),
        "rates": {k: round(v / max(1, total), 4) for k, v in counts.most_common()},
        "method": "lexical heuristics -- CANDIDATES ONLY, all cases verified=false",
        "caveat": (
            "Categories are not mutually exclusive; percentages sum above 100%. "
            "These counts are unverified triage output and must not be reported "
            "as measured error rates without human confirmation and an "
            "inter-annotator agreement figure."
        ),
        "known_over_flagging": (
            "`object_hallucination` compares against the REFERENCE CAPTIONS, not "
            "against the image. A word the model says that no reference happens "
            "to mention is flagged even when it is visibly correct -- e.g. "
            "predicting 'red shirt' where references say 'wearing red'. With ~5 "
            "references this is common, so the rate is an UPPER BOUND on true "
            "hallucination, not an estimate of it. Hand-verification on a "
            "sample is required, and the verified precision must be reported "
            "alongside the raw count. Measuring factuality properly needs the "
            "image or gold propositions -- which is the argument for the "
            "proposition-level evaluation in formulation/04 §3."
        ),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    print(f"\nError triage -- {args.experiment}  ({total} captions)")
    print("-" * 56)
    for category, count in counts.most_common():
        print(f"  {category:<24}{count:>6}  {count / max(1, total):>7.1%}")
    print("-" * 56)
    print(f"  cases written to {out_dir}")
    print("  ! heuristic triage: verify a sample by hand before reporting\n")

    try:
        import sys

        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from rescap.viz import plot_error_distribution

        plot_error_distribution(
            {k: v for k, v in counts.items() if k != "correct_or_minor"},
            name=f"error_distribution_{args.experiment}",
        )
    except Exception as exc:
        print(f"  (figure skipped: {exc})")


if __name__ == "__main__":
    main()
