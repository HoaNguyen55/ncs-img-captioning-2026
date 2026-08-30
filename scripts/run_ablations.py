#!/usr/bin/env python
"""Score the verification ablations by replaying a Stage 1 run.

    python scripts/run_ablations.py --in ~/ncs-data/stage1

Every ablation here changes only how recorded probe answers are SCORED, never
which probes get asked. So the whole table comes from one Stage 1 run replayed
under different configurations: seconds on a CPU instead of re-running 3,769
images once per condition, which is around forty fleet-hours.

**Validate the harness first.** `validate_replay.py` must pass on the same
records, and its residual has to be quoted with these numbers. Three successive
versions of the replay looked correct and were not; a table built on an
unchecked replay describes a run that never happened.

**What this cannot do.** A1 (`mode = none`) and ablations that change the
PROMPT or the proposition set need a real run -- different questions get asked,
and a question never asked has no recorded answer. Those are marked and skipped
rather than silently approximated.
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
    """A fake image carrying only the real size, for the geometric channel during replay.

    Finding 19/08: the real run passes the image, so `geometric_check` normalizes
    dx/dy by the image frame; replay passed None so it fell back to normalizing by
    the box — 2.1% of verdicts shifted, concentrated exactly in
    spatial_relation/relation. The geometric channel needs only `.size`; the paths
    needing a real PIL image (region crops) still see this is not PIL and skip as before.
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


def build_conditions():
    from rescap.pipeline.verify import Mode, VerificationConfig, Verdict

    return [
        ("A0 full", "three verdicts, every channel on",
         VerificationConfig()),
        ("A2 binary→REJECTED", "UNCERTAIN folded into REJECTED",
         VerificationConfig(mode=Mode.BINARY, binary_uncertain_to=Verdict.REJECTED)),
        ("A2 binary→SUPPORTED", "UNCERTAIN folded into SUPPORTED",
         VerificationConfig(mode=Mode.BINARY, binary_uncertain_to=Verdict.SUPPORTED)),
        ("A4 no contradiction", "channels 9(a)+9(b) off",
         VerificationConfig(contradiction_detection=False)),
        ("A7 no geometry", "channel 6 off",
         VerificationConfig(spatial_verification=False)),
        ("A4+A7 both off", "probe channel only",
         VerificationConfig(contradiction_detection=False, spatial_verification=False)),
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in", dest="src", required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    from rescap.pipeline.verify import verify
    from rescap.vlm.replay import ReplayVLM

    sizes = _load_sizes()
    if not sizes:
        print("  ⚠ no image_sizes.json — the replayed geometric channel will lack the image frame")

    files = [f for f in sorted(Path(args.src).glob("*.json"))
             if not f.name.startswith("_")]
    if args.limit:
        files = files[: args.limit]
    if not files:
        raise SystemExit(f"no records found in {args.src}")

    records = [json.loads(f.read_text(encoding="utf-8")) for f in files]
    print(f"{len(records)} images, replayed — no GPU used\n")

    rows = []
    for name, note, config in build_conditions():
        counts: Counter[str] = Counter()
        supported_per_image = []
        misses = 0
        for record in records:
            props = copy.deepcopy(record["propositions"])
            for p in props:
                p["verification"] = {"status": None}

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
            colour = (ReplayVLM(record["propositions"], model_name=second[0])
                      if second else None)

            verify(primary,
               _SizeOnly(sizes[record["file_name"]]) if record.get("file_name") in sizes else None,
               record["entities"], props,
                   config=copy.deepcopy(config), colour_verifier=colour)
            misses += primary.misses

            n_supported = 0
            for p in props:
                v = verdict_of(p)
                counts[v] += 1
                n_supported += v == "SUPPORTED"
            supported_per_image.append(n_supported)

        total = sum(counts.values()) or 1
        rows.append({
            "condition": name, "note": note,
            "SUPPORTED": counts["SUPPORTED"], "UNCERTAIN": counts["UNCERTAIN"],
            "REJECTED": counts["REJECTED"],
            "supported_pct": counts["SUPPORTED"] / total * 100,
            "supported_per_image": (sum(supported_per_image) / len(supported_per_image)
                                    if supported_per_image else 0.0),
            "probe_misses": misses,
        })

    header = (f"  {'condition':<24} {'SUP':>7} {'UNC':>7} {'REJ':>7} "
              f"{'%SUP':>7} {'SUP/img':>9}")
    print(header + "\n  " + "-" * (len(header) - 2))
    base = rows[0]
    for row in rows:
        delta = ""
        if row is not base:
            d = row["supported_per_image"] - base["supported_per_image"]
            delta = f"  ({d:+.2f})"
        print(f"  {row['condition']:<24} {row['SUPPORTED']:>7} {row['UNCERTAIN']:>7} "
              f"{row['REJECTED']:>7} {row['supported_pct']:>6.1f}% "
              f"{row['supported_per_image']:>9.2f}{delta}")
        if row["probe_misses"]:
            print(f"       ⚠ {row['probe_misses']} probes absent from the records — "
                  f"this condition CANNOT be replayed faithfully")

    print("\n  A1 (verification off) and every ablation changing the PROMPT or the")
    print("  proposition set are NOT here: they ask different questions, and a")
    print("  question never asked has no answer to replay. Those need real runs.")
    print("\n  ⚠ Alongside every number here: the replayer reproduces ~99.7% of the")
    print("    original verdicts, not 100%. Run validate_replay.py on the same records.")

    out = Path(args.out or Path(args.src).parent / "ablations_replay.json")
    out.write_text(json.dumps({"n_images": len(records), "rows": rows},
                              ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n  wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
