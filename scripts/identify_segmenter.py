#!/usr/bin/env python
"""Identify which word segmenter produced KTVIC's `segment_caption` field.

    python scripts/identify_segmenter.py

**Why this blocks everything.** KTVIC's Table 3 (GRIT: CIDEr 136.0) is computed
on the *segmented* captions. Vietnamese writes whitespace between syllables, so
`người đàn ông` is three tokens but one word — and a different segmenter gives a
different token sequence, hence different BLEU/CIDEr. If we segment our output
differently from KTVIC, our numbers cannot sit anywhere near theirs, even as
context.

The test is direct: run each candidate segmenter over KTVIC's own raw captions
and see which one reproduces `segment_caption` exactly.

Outcomes:
  * one segmenter matches at a high rate  -> use it, record name + version
  * none matches                          -> use KTVIC's OWN segmented field as
                                             the reference and segment our
                                             predictions with the closest tool,
                                             stating the mismatch as a caveat
"""

from __future__ import annotations

import argparse
import difflib
import json
import os
import sys
from pathlib import Path


def load_pairs(path: Path, limit: int) -> list[tuple[str, str]]:
    """Extract (raw_caption, segmented_caption) pairs, whatever the JSON shape."""
    data = json.loads(path.read_text(encoding="utf-8"))
    records = []
    if isinstance(data, dict):
        for key in ("annotations", "captions", "data"):
            if isinstance(data.get(key), list):
                records = data[key]
                break
        else:
            for value in data.values():
                if isinstance(value, list) and value and isinstance(value[0], dict):
                    records = value
                    break
    elif isinstance(data, list):
        records = data

    pairs: list[tuple[str, str]] = []
    for record in records:
        if not isinstance(record, dict):
            continue
        raw = record.get("caption") or record.get("raw_caption")
        seg = record.get("segment_caption") or record.get("segmented_caption")
        if raw and seg:
            pairs.append((str(raw).strip(), str(seg).strip()))
        if len(pairs) >= limit:
            break
    return pairs


def get_segmenters() -> dict[str, tuple[callable, str]]:
    """Available segmenters as {name: (fn, version)}."""
    out: dict[str, tuple[callable, str]] = {}

    try:
        from importlib.metadata import version as pkg_version

        from pyvi import ViTokenizer

        try:
            v = pkg_version("pyvi")
        except Exception:
            v = "unknown"
        out["pyvi"] = (ViTokenizer.tokenize, v)
    except Exception as exc:
        print(f"  pyvi unavailable: {exc}")

    try:
        import underthesea

        out["underthesea"] = (
            lambda t: underthesea.word_tokenize(t, format="text"),
            getattr(underthesea, "__version__", "unknown"),
        )
    except Exception as exc:
        print(f"  underthesea unavailable: {exc}")

    try:
        import py_vncorenlp  # noqa: F401

        print("  py_vncorenlp is installed but needs an explicit model download;")
        print("  add it manually if neither of the above matches.")
    except Exception:
        pass

    return out


def normalise(text: str) -> str:
    """Compare on whitespace-collapsed lowercase; casing is not the question."""
    return " ".join(text.lower().split())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    default = Path(os.environ.get("NCS_DATA", Path.home() / "ncs-data"))
    parser.add_argument("--ktvic", default=str(default / "datasets" / "ktvic"))
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--show", type=int, default=5, help="mismatches to print")
    args = parser.parse_args()

    root = Path(args.ktvic)
    path = next(
        (root / n for n in ("train_data.json", "test_data.json") if (root / n).exists()),
        None,
    )
    if path is None:
        raise SystemExit(
            f"No KTVIC annotations under {root}.\n"
            "Run: bash scripts/datasets/download_ktvic.sh --yes"
        )

    pairs = load_pairs(path, args.limit)
    if not pairs:
        raise SystemExit(
            f"{path.name} has no (caption, segment_caption) pairs — inspect the "
            "JSON shape and extend load_pairs()."
        )
    print(f"Loaded {len(pairs)} caption pairs from {path.name}\n")

    print("Available segmenters:")
    segmenters = get_segmenters()
    if not segmenters:
        raise SystemExit("No segmenter installed. uv pip install pyvi underthesea")
    for name, (_, version) in segmenters.items():
        print(f"  {name} {version}")
    print()

    results: dict[str, dict] = {}
    for name, (segment, version) in segmenters.items():
        exact = 0
        similarity_total = 0.0
        mismatches: list[tuple[str, str, str]] = []
        for raw, gold in pairs:
            try:
                got = segment(raw)
            except Exception as exc:
                mismatches.append((raw, gold, f"<error {exc}>"))
                continue
            if normalise(got) == normalise(gold):
                exact += 1
            else:
                if len(mismatches) < args.show:
                    mismatches.append((raw, gold, got))
            similarity_total += difflib.SequenceMatcher(
                None, normalise(got), normalise(gold)
            ).ratio()

        results[name] = {
            "version": version,
            "exact_match_rate": exact / len(pairs),
            "mean_similarity": similarity_total / len(pairs),
            "mismatches": mismatches,
        }

    print("=" * 66)
    print(f"{'segmenter':<16}{'version':<12}{'exact':>10}{'similarity':>14}")
    print("-" * 66)
    for name, r in sorted(
        results.items(), key=lambda kv: -kv[1]["exact_match_rate"]
    ):
        print(
            f"{name:<16}{r['version']:<12}"
            f"{r['exact_match_rate']:>9.1%}{r['mean_similarity']:>14.3f}"
        )
    print("=" * 66)

    best = max(results.items(), key=lambda kv: kv[1]["exact_match_rate"])
    name, r = best

    print()
    if r["exact_match_rate"] >= 0.90:
        print(f"MATCH: {name} {r['version']} reproduces KTVIC's segmentation "
              f"({r['exact_match_rate']:.1%} exact).")
        print("Use it for every reported metric, and name it in every table.")
    elif r["exact_match_rate"] >= 0.50:
        print(f"PARTIAL: {name} {r['version']} matches {r['exact_match_rate']:.1%}.")
        print("Close but not identical. Recommended: score against KTVIC's OWN")
        print("`segment_caption` as the reference, segment our predictions with")
        print(f"{name}, and state the mismatch rate as a caveat in the paper.")
    else:
        print(f"NO MATCH — best is {name} at {r['exact_match_rate']:.1%}.")
        print("KTVIC likely used a tool we do not have (VnCoreNLP is the usual")
        print("suspect). Options, in order of preference:")
        print("  1. install py_vncorenlp and re-run this script")
        print("  2. use KTVIC's `segment_caption` as the reference and segment")
        print("     our predictions with the closest tool, reporting the caveat")
        print("  3. report our numbers ONLY against our own segmentation and")
        print("     never place them beside KTVIC's Table 3")

    if r["mismatches"]:
        print(f"\nExample mismatches for {name}:")
        for raw, gold, got in r["mismatches"][: args.show]:
            print(f"\n  raw  : {raw[:88]}")
            print(f"  gold : {gold[:88]}")
            print(f"  got  : {got[:88]}")

    out = Path(__file__).resolve().parents[1] / "results" / "segmenter_check.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "source": str(path),
                "n_pairs": len(pairs),
                "results": {
                    k: {kk: vv for kk, vv in v.items() if kk != "mismatches"}
                    for k, v in results.items()
                },
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
