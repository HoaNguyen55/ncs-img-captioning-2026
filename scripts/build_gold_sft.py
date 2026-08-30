#!/usr/bin/env python
"""Build SFT data from KTVIC's GOLD CAPTIONS — the P1 baseline.

    python scripts/build_gold_sft.py \\
        --annotations ~/ncs-data/datasets/ktvic/train_data.json \\
        --out ~/ncs-data/stage2_gold

Why this exists: the most natural reviewer question is *"there are 3,769
images with human-written captions — why not SFT directly on those?"*. The
paper must have a number to answer with. The honest, pre-registered prediction
is that gold-SFT may BEAT distillation on concise-mode CIDEr (it learns the
exact reference style); what it cannot learn is detail + guardedness, and
Table 2 measures exactly that.

For a fair comparison with the distillation branch (same images, same epochs,
same prompt):

* ONE caption per image (the first caption in file order — KTVIC does not
  mark a "primary" caption), not all ~5 — the distillation branch likewise
  has only one answer per image per style.
* The prompt is the EXACT short evaluation prompt (`PROMPT_SHORT` from
  build_dpo_data.py) — the model is asked at scoring time just as at
  training time.
* NO detailed variant: a gold caption is a single sentence; inventing a
  detailed style from it would be fabricating data. In detailed mode this
  model answers with whatever it retains from pretraining — and that is
  precisely the measurement.

`dpo.jsonl` is not produced: gold captions carry no verdicts, so no
preference pairs exist — running `train_stage2.py --stage sft` is enough.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_dpo_data import PROMPT_SHORT  # the same prompt, not re-typed by hand


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", required=True,
                        help="KTVIC's train_data.json")
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=0,
                        help="limit the number of images (0 = all) — for quick trials only")
    args = parser.parse_args()

    data = json.loads(Path(args.annotations).expanduser().read_text(encoding="utf-8"))
    file_names = {
        str(img.get("id", img.get("image_id"))): img.get("file_name") or img.get("filename")
        for img in data.get("images", [])
    }

    first_caption: dict[str, str] = {}
    n_captions = 0
    for ann in data.get("annotations", []):
        image_id = str(ann["image_id"])
        caption = (ann.get("caption") or "").strip()
        if not caption:
            continue
        n_captions += 1
        first_caption.setdefault(image_id, caption)

    rows = [
        {"image_id": image_id, "file_name": file_names.get(image_id),
         "prompt": PROMPT_SHORT, "response": caption, "variant": "gold"}
        for image_id, caption in sorted(first_caption.items())
        if file_names.get(image_id)
    ]
    dropped = len(first_caption) - len(rows)
    if args.limit:
        rows = rows[: args.limit]

    rng = random.Random(args.seed)
    rng.shuffle(rows)

    dst = Path(args.out).expanduser()
    dst.mkdir(parents=True, exist_ok=True)
    with (dst / "sft.jsonl").open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    report = {
        "source": str(args.annotations), "seed": args.seed,
        "baseline": "gold_sft (P1, )",
        "images_with_caption": len(first_caption),
        "captions_total": n_captions,
        "captions_used": len(rows),
        "policy": "one first-caption per image, PROMPT_SHORT, no detailed variant",
        "dropped_no_file_name": dropped,
    }
    (dst / "_build_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"  {len(rows)} gold-SFT examples (from {n_captions} captions / "
          f"{len(first_caption)} images, first caption per image)")
    if dropped:
        print(f"  ⚠ {dropped} images dropped because file_name could not be resolved")
    print(f"  -> {dst/'sft.jsonl'}\n"
          f"  run: train_stage2.py --stage sft --data {dst} --epochs 2")
    return 0


if __name__ == "__main__":
    sys.exit(main())
