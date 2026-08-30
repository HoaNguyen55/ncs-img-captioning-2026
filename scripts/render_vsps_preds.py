#!/usr/bin/env python
"""Build VSPS captions (no training) from stage1_test records — the
"VSPS" row of Tables 1/2.

    python scripts/render_vsps_preds.py \\
        --records ~/ncs-data/stage1_test \\
        --out-dir ~/ncs-data/results

    # then score with the exact scorer used for every other row:
    python scripts/evaluate.py --predictions \\
        ~/ncs-data/results/vsps-detailed.preds.json \\
        --name vsps-detailed --prompt detailed --also-syllable

Goes through the EXACT construction path of the training data
(`build_for_image` in build_dpo_data.py): detailed = variant A (every
proposition that passes selection, uncertain parts hedged), short = the short
variant (at most 2 top-priority SUPPORTED propositions). No second builder is
written — two builders are two sources of number drift.

`evaluate.py` refuses to score a subset (rightly so), so any image VSPS has
nothing to say about gets an EMPTY string and is scored as silence — that is
the pipeline's real behaviour, not a bug; the counts are printed and written
to *.stats.json for the paper to publish alongside.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_dpo_data import build_for_image
from evaluate import references


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", required=True,
                        help="directory of stage1 records for the test set")
    parser.add_argument("--split", default="test_data.json")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--allow-partial", action="store_true",
                        help="build even without all 558 records (preview only; "
                             "a file with missing images is rejected by evaluate.py)")
    args = parser.parse_args()

    expected = {str(i) for i in references(args.split)}
    records_dir = Path(args.records).expanduser()
    files = sorted(records_dir.glob("*.json"))
    print(f"{len(files)} records in {records_dir} · test set needs {len(expected)} images")

    stats = Counter()
    detailed: dict[str, str] = {}
    short: dict[str, str] = {}
    for f in files:
        record = json.loads(f.read_text(encoding="utf-8"))
        image_id = str(record.get("image_id"))
        if image_id not in expected:
            stats["ngoai_tap_test"] += 1
            continue
        sft, sft_short, _pairs = build_for_image(record, stats)
        detailed[image_id] = (sft or {}).get("response") or ""
        if not detailed[image_id]:
            stats["chi_tiet_rong"] += 1
        # Short: if no SUPPORTED proposition makes the top, fall back to variant A —
        # VSPS would rather say a hedged sentence than stay silent; full silence only when A is empty too.
        short[image_id] = (sft_short or {}).get("response") or detailed[image_id]
        if not sft_short:
            stats["ngan_lui_ve_A" if short[image_id] else "ngan_rong"] += 1

    missing = expected - set(detailed)
    if missing and not args.allow_partial:
        raise SystemExit(
            f"only {len(detailed)}/{len(expected)} test images so far "
            f"(missing e.g. {sorted(missing)[:3]}) — wait for the VSPS-test shards "
            f"to finish and rebuild, or use --allow-partial to preview.")

    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, preds in (("vsps-detailed", detailed), ("vsps-short", short)):
        p = out_dir / f"{name}.preds.json"
        p.write_text(json.dumps(preds, ensure_ascii=False, indent=1),
                     encoding="utf-8")
        print(f"  wrote {p} ({len(preds)} images)")
    (out_dir / "vsps-preds.stats.json").write_text(
        json.dumps(dict(stats), ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8")

    for k in ("chi_tiet_rong", "ngan_lui_ve_A", "ngan_rong"):
        if stats[k]:
            print(f"  {k}: {stats[k]} images")
    print(f"  still missing: {len(missing)} images" if missing else "  full test set covered")
    return 0


if __name__ == "__main__":
    sys.exit(main())
