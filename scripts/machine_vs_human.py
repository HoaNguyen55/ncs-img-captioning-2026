#!/usr/bin/env python
"""Compare the verifier's verdicts with human verdicts on the same images.

    python scripts/machine_vs_human.py \\
        --machine ~/ncs-data/stage1_calibration --split pilot_calibration

**Why this exists** : supported-per-image rose 2.4x after a
chain of bug fixes, and more SUPPORTED is only an improvement if the newly
supported propositions are actually TRUE. The person who produced that rise
should not be the one deciding whether it is real, so the pilot annotations
double as a spot check: humans verdict the same propositions independently, and
the disagreement pattern says which way the machine errs.

**The direction of each disagreement matters more than the count.**

* machine SUPPORTED / human REJECTED — the dangerous cell. These become
  training labels; each one teaches the generator something false.
* machine REJECTED / human SUPPORTED — lost coverage, costs detail, poisons
  nothing.
* either / human UNCERTAIN — often legitimate: the annotator could not tell
  from the image either, which is the verdict meaning exactly that.

Matching is by canonical proposition text (same as `agreement.py`): the humans
annotate freely rather than verdicting the machine's list, so only propositions
both sides expressed can be compared, and the unmatched counts are reported —
they measure *coverage* difference, not correctness.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

DATA = Path(os.environ.get("NCS_DATA", Path.home() / "ncs-data"))
VERDICTS = ("SUPPORTED", "UNCERTAIN", "REJECTED")


def normalise(text: str) -> str:
    from rescap.svp.matching import canonical

    cleaned = " ".join(str(text or "").lower().split()).strip(" .,;:")
    return canonical(cleaned) or cleaned


def machine_verdicts(directory: Path) -> dict[str, dict[str, str]]:
    """{image_stem: {canonical_text: verdict}} from Stage 1 records."""
    from rescap.pipeline.verify import verdict_name

    out: dict[str, dict[str, str]] = {}
    for path in sorted(directory.glob("*.json")):
        if path.name.startswith("_"):
            continue
        record = json.loads(path.read_text(encoding="utf-8"))
        image_id = str(record.get("image_id") or path.stem)
        for prop in record.get("propositions") or []:
            verdict = verdict_name(prop)
            if verdict:
                out.setdefault(image_id, {})[normalise(prop.get("text_vi"))] = verdict
    return out


def human_verdicts(split: str) -> dict[str, dict[str, dict[str, str]]]:
    """{annotator: {image_id: {canonical_text: verdict}}}"""
    directory = DATA / "annotations" / split
    if not directory.exists():
        raise SystemExit(
            f"no annotations yet in {directory} — "
            f"run annotate.py --split {split} first"
        )
    out: dict[str, dict[str, dict[str, str]]] = {}
    for path in sorted(directory.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        person: dict[str, dict[str, str]] = {}
        for image_id, record in (data.get("images") or {}).items():
            for prop in record.get("propositions") or []:
                person.setdefault(str(image_id), {})[normalise(prop["text_vi"])] = prop["verdict"]
        out[path.stem] = person
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--machine", required=True, help="directory of Stage 1 records")
    parser.add_argument("--split", default="pilot_calibration")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    machine = machine_verdicts(Path(args.machine))
    humans = human_verdicts(args.split)
    print(f"  machine: {len(machine)} images · humans: {', '.join(sorted(humans))}\n")

    report = {"split": args.split, "annotators": {}}
    for name, person in sorted(humans.items()):
        confusion: dict[tuple[str, str], int] = defaultdict(int)
        matched = machine_only = human_only = 0
        dangerous: list[dict] = []

        for image_id, human_props in person.items():
            machine_props = machine.get(image_id, {})
            shared = set(human_props) & set(machine_props)
            matched += len(shared)
            machine_only += len(set(machine_props) - shared)
            human_only += len(set(human_props) - shared)
            for key in shared:
                mv, hv = machine_props[key], human_props[key]
                confusion[(mv, hv)] += 1
                if mv == "SUPPORTED" and hv == "REJECTED" and len(dangerous) < 12:
                    dangerous.append({"image_id": image_id, "text": key})

        total = sum(confusion.values()) or 1
        agree = sum(n for (m, h), n in confusion.items() if m == h)
        false_support = confusion[("SUPPORTED", "REJECTED")]
        n_machine_supported = sum(
            n for (m, _), n in confusion.items() if m == "SUPPORTED"
        )

        print(f"  === machine ↔ {name} ===")
        print(f"    text-matched props   : {matched}"
              f"   (machine only: {machine_only} · human only: {human_only})")
        print(f"    verdict agreement    : {agree}/{total} = {agree/total*100:.1f}%")
        print(f"\n    {'machine \\\\ human':<14}" + "".join(f"{v[:9]:>11}" for v in VERDICTS))
        for mv in VERDICTS:
            row = "".join(f"{confusion.get((mv, hv), 0):>11}" for hv in VERDICTS)
            print(f"    {mv:<14}{row}")
        if n_machine_supported:
            rate = false_support / n_machine_supported
            print(f"\n    ⚠ DANGEROUS CELL — machine SUPPORTED, human REJECTED: "
                  f"{false_support}/{n_machine_supported} "
                  f"= {rate*100:.1f}% of everything the machine marked SUPPORTED")
            print(f"      (each of these is a FALSE thing that will be taught to the generator)")
        for d in dangerous[:5]:
            print(f"      · image {d['image_id']}: {d['text'][:70]}")
        print()

        report["annotators"][name] = {
            "matched": matched, "machine_only": machine_only, "human_only": human_only,
            "agreement": agree / total,
            "false_support": false_support,
            "false_support_rate_of_supported": (
                false_support / n_machine_supported if n_machine_supported else None
            ),
            "confusion": {f"{m}|{h}": n for (m, h), n in confusion.items()},
            "dangerous_examples": dangerous,
        }

    out = Path(args.out or DATA / "results" / f"machine_vs_human_{args.split}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
