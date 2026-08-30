#!/usr/bin/env python
"""Check that replaying a Stage 1 run reproduces the verdicts it recorded.

    python scripts/validate_replay.py --in ~/ncs-data/stage1

**Run this before trusting any number that came from a replay.** The harness
exists so a rule change can be scored on a CPU instead of six fleet-hours, and a
harness that quietly disagrees with its source is worse than no harness: it
produces confident numbers about a run that never happened.

That is not hypothetical. The first version reproduced 62.8% of its own source
and the gap turned out to be a real bug in `_answer_stability` -- string entropy
where polarity entropy was meant. The second reproduced 74.6% because the replay
returned an `Answer` with no `samples`, so every probe scored as perfectly
stable. The third reproduced 98.9% because a colour cross-check asks the SAME
question of two models and both answers landed under one key.

Each of those looked like a working tool until it was asked to reproduce
something known.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class _SizeOnly:
    """Fake image carrying just the real size for the geometric channel on replay.

    Found 19/08: the real run passes the image, so `geometric_check` normalises
    dx/dy by the image frame; replay passed None, so it fell back to
    box-based normalisation — 2.1% of verdicts shifted, concentrated exactly in
    spatial_relation/relation. The geometric channel only needs `.size`; the
    paths that need a real PIL image (region crops) still see this is not PIL
    and skip it as before.
    """

    def __init__(self, size):
        self.size = tuple(size)


def _load_sizes():
    import json as _json
    path = Path(__file__).resolve().parents[1] / "paper" / "data" / "image_sizes.json"
    try:
        return _json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def verdict_of(proposition: dict) -> str | None:
    """One shared parser (`pipeline.verify.verdict_name`)."""
    from rescap.pipeline.verify import verdict_name

    return verdict_name(proposition)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in", dest="src", required=True)
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument(
        "--min-fidelity", type=float, default=0.99,
        help="below this it counts as broken, not as 'close enough'",
    )
    args = parser.parse_args()

    from rescap.pipeline.verify import verify
    from rescap.vlm.replay import ReplayVLM

    sizes = _load_sizes()
    if not sizes:
        print("  ⚠ no image_sizes.json — the replayed geometric channel will lack the image frame")

    files = [f for f in sorted(Path(args.src).glob("*.json"))
             if not f.name.startswith("_")][: args.limit]
    if not files:
        raise SystemExit(f"no records found in {args.src}")

    same = diff = hits = misses = 0
    shifts: Counter[tuple[str, str]] = Counter()
    examples: list[dict] = []

    for path in files:
        record = json.loads(path.read_text(encoding="utf-8"))
        props = copy.deepcopy(record["propositions"])
        before = {p["id"]: verdict_of(p) for p in props}
        for p in props:
            p["verification"] = {"status": None}

        # Which model answered how many probes tells us which was the verifier
        # and which was the colour cross-check, without needing the run's config.
        models: Counter[str] = Counter(
            str(pr.get("model", ""))
            for p in record["propositions"]
            for pr in ((p.get("evidence") or {}).get("probes") or [])
        )
        if not models:
            continue
        verifier = models.most_common(1)[0][0]
        second = [m for m in models if m != verifier]

        primary = ReplayVLM(record["propositions"], model_name=verifier)
        colour = ReplayVLM(record["propositions"], model_name=second[0]) if second else None
        verify(primary,
               _SizeOnly(sizes[record["file_name"]]) if record.get("file_name") in sizes else None,
               record["entities"], props, colour_verifier=colour)

        hits += primary.hits
        misses += primary.misses
        for p in props:
            after = verdict_of(p)
            if after == before[p["id"]]:
                same += 1
            else:
                diff += 1
                shifts[(before[p["id"]], after)] += 1
                if len(examples) < 8:
                    examples.append({
                        "file": path.name, "id": p["id"], "type": p.get("type"),
                        "was": before[p["id"]], "now": after,
                        "text": str(p.get("text_vi"))[:60],
                    })

    total = same + diff
    fidelity = same / total if total else 0.0
    print(f"  {len(files)} images · {total} propositions")
    print(f"  probe hits   : {hits}/{hits + misses}"
          f" = {hits/(hits+misses)*100:.1f}%" if hits + misses else "")
    print(f"  VERDICTS REPRODUCED: {same}/{total} = {fidelity*100:.2f}%")
    if shifts:
        print("  remaining shifts:")
        for (was, now), n in shifts.most_common():
            print(f"    {was:>10} -> {now:<10} {n:>5}")
        for e in examples[:4]:
            print(f"    · {e['id']} [{e['type']}] {e['was']}->{e['now']}  {e['text']}")

    if fidelity < args.min_fidelity:
        print(f"\n  ⛔ BELOW THE {args.min_fidelity:.0%} THRESHOLD — do not use the replay "
              f"harness to report any number until the drift is tracked down.")
        return 1
    print(f"\n  ✅ Meets the {args.min_fidelity:.0%} threshold. The remaining drift must be "
          f"stated alongside every replay-derived result — it does NOT reproduce exactly.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
