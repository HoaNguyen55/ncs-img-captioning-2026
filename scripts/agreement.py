#!/usr/bin/env python
"""Inter-annotator agreement on the three-way verdict.

    python scripts/agreement.py --split pilot
    python scripts/agreement.py --split main --annotators an,binh,cuong

**A human evaluation without an agreement figure can be discounted entirely**,
and reasonably so: without it nobody can tell whether the verdicts measure the
image or measure the guideline being unclear.  budgets 100 double-annotated
images for exactly this, and the budget is wasted unless the number is computed.

**Cohen's κ, not raw agreement.** Three-way verdicts are not uniformly
distributed -- UNCERTAIN dominates -- so two annotators who both answer
UNCERTAIN most of the time agree ~60% by accident. κ subtracts that floor. Raw
agreement is printed beside it so the gap is visible rather than implied.

**Matching is by proposition text, not by position.** Annotators add
propositions in whatever order they see them, so pairing the nth of one list
with the nth of another compares unrelated claims and produces a κ that means
nothing. Only propositions both annotators actually wrote are scored, and the
count of unmatched ones is reported -- a pair who agree perfectly on ten shared
propositions while each writing thirty of their own have not agreed about the
image.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

DATA = Path(os.environ.get("NCS_DATA", Path.home() / "ncs-data"))
OUT_ROOT = DATA / "annotations"
VERDICTS = ("SUPPORTED", "UNCERTAIN", "REJECTED")


def normalise(text: str) -> str:
    """Compare claims by content, not by typing.

    Goes through the Vietnamese canonical form so `chiếc xe hơi` and `ô tô` are
    the same claim -- two annotators describing one thing in different words
    have agreed, and scoring them as a disagreement would understate κ.
    """
    from rescap.svp.matching import canonical

    cleaned = " ".join(str(text or "").lower().split()).strip(" .,;:")
    return canonical(cleaned) or cleaned


def cohens_kappa(pairs: list[tuple[str, str]]) -> tuple[float | None, float]:
    """`(kappa, raw_agreement)` over paired labels."""
    if not pairs:
        return None, 0.0
    n = len(pairs)
    observed = sum(1 for a, b in pairs if a == b) / n
    first = Counter(a for a, _ in pairs)
    second = Counter(b for _, b in pairs)
    expected = sum((first[v] / n) * (second[v] / n) for v in set(first) | set(second))
    if expected >= 1.0:
        # Both annotators used exactly one label. Agreement is total and
        # meaningless; reporting κ = 1 would dress that up as a result.
        return None, observed
    return (observed - expected) / (1 - expected), observed


def load(split: str, wanted: list[str] | None) -> dict[str, dict]:
    directory = OUT_ROOT / split
    if not directory.exists():
        raise SystemExit(f"annotation directory {directory} not found")
    out = {}
    for path in sorted(directory.glob("*.json")):
        name = path.stem
        if wanted and name not in wanted:
            continue
        out[name] = json.loads(path.read_text(encoding="utf-8"))
    if len(out) < 2:
        raise SystemExit(
            f"need at least 2 annotators, only found {sorted(out)}. "
            f"κ cannot be computed from one person."
        )
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="pilot")
    parser.add_argument("--annotators", default="", help="names, comma-separated")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    people = load(args.split, [a.strip() for a in args.annotators.split(",") if a.strip()])
    print(f"  {len(people)} annotators: {', '.join(sorted(people))}\n")

    report = {"split": args.split, "annotators": sorted(people), "pairs": []}
    for a, b in combinations(sorted(people), 2):
        images_a, images_b = people[a]["images"], people[b]["images"]
        shared_images = sorted(set(images_a) & set(images_b))

        pairs: list[tuple[str, str]] = []
        only_a = only_b = 0
        confusion: dict[tuple[str, str], int] = defaultdict(int)
        per_type: dict[str, list[tuple[str, str]]] = defaultdict(list)

        for image_id in shared_images:
            by_text_a = {normalise(p["text_vi"]): p for p in images_a[image_id]["propositions"]}
            by_text_b = {normalise(p["text_vi"]): p for p in images_b[image_id]["propositions"]}
            shared = set(by_text_a) & set(by_text_b)
            only_a += len(by_text_a) - len(shared)
            only_b += len(by_text_b) - len(shared)
            for key in shared:
                va, vb = by_text_a[key]["verdict"], by_text_b[key]["verdict"]
                pairs.append((va, vb))
                confusion[(va, vb)] += 1
                per_type[by_text_a[key].get("type", "?")].append((va, vb))

        kappa, observed = cohens_kappa(pairs)
        shown = "n/a (a single label only)" if kappa is None else f"{kappa:.3f}"
        print(f"  {a} ↔ {b}")
        print(f"    shared images    : {len(shared_images)}")
        print(f"    matched props    : {len(pairs)}")
        print(f"    only {a:<12}: {only_a}")
        print(f"    only {b:<12}: {only_b}")
        print(f"    raw agreement    : {observed*100:.1f}%")
        print(f"    **Cohen κ**      : {shown}")
        if kappa is not None:
            band = ("poor (<0.40)" if kappa < 0.40 else
                    "moderate (0.40–0.60)" if kappa < 0.60 else
                    "good (0.60–0.80)" if kappa < 0.80 else "very good (≥0.80)")
            print(f"    band             : {band}")
        if only_a + only_b > len(pairs):
            print(f"    ⚠ propositions seen by only ONE annotator ({only_a + only_b}) outnumber "
                  f"the matched ones ({len(pairs)}) — κ speaks only to the shared part; "
                  f"the two are looking at the image very differently")

        if confusion:
            print(f"\n    {'':<12}" + "".join(f"{v[:9]:>11}" for v in VERDICTS))
            for va in VERDICTS:
                row = "".join(f"{confusion.get((va, vb), 0):>11}" for vb in VERDICTS)
                print(f"    {va:<12}{row}")
        print()

        report["pairs"].append({
            "a": a, "b": b, "shared_images": len(shared_images),
            "matched_propositions": len(pairs), "only_a": only_a, "only_b": only_b,
            "raw_agreement": observed, "cohens_kappa": kappa,
            "confusion": {f"{k[0]}|{k[1]}": v for k, v in confusion.items()},
            "per_type_kappa": {
                t: cohens_kappa(v)[0] for t, v in per_type.items() if len(v) >= 10
            },
        })

    out = Path(args.out or DATA / "results" / f"agreement_{args.split}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
