#!/usr/bin/env python
"""Stage 1 — run the VSPS pipeline over KTVIC and write the supervision it yields.

    # split across two machines
    python research/scripts/generate_stage1.py --shard 0 --of 2
    python research/scripts/generate_stage1.py --shard 1 --of 2

    # one machine, first 200 images, to check the rate before committing
    python research/scripts/generate_stage1.py --limit 200

This is the module research log  specifies. Its output is not captions --
those are a by-product. What Stage 2 needs is the **verdict record**: for every
image, every candidate proposition with its three-way verdict, its confidence
and the probes that produced it. That is the training signal, and it costs no
annotation time.

**Resumable, because it has to be.** ~3,500 images at 30-90 s each is 30-80
hours. An SSH drop, an OOM, a machine handed back -- any of these will happen
over that span. One file per image, and an image whose file already exists is
skipped, so re-running continues rather than restarting. Nothing is held in
memory that a crash would lose.

**Shardable, because there are two cards.** `--shard i --of n` takes every n-th
image, so the shards need no coordination and no shared state. Merging is
concatenation.

**What it does NOT do:** decide anything. No thresholds are tuned here, no
propositions are dropped for looking wrong. The verdicts are whatever the
verifier said, and the failures are recorded rather than skipped -- a run that
silently omitted its hard cases would report a factuality rate that belongs to
an easier dataset than the one we have.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
import traceback
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

DATA = Path(os.environ.get("NCS_DATA", Path.home() / "ncs-data"))
KTVIC = DATA / "datasets" / "ktvic"

_stop = False


def _handle_signal(signum, frame):  # pragma: no cover
    """Finish the image in flight, then stop. A half-written record is worse
    than a missing one, because the merge step cannot tell them apart."""
    global _stop
    _stop = True
    print("\n  (nhận tín hiệu dừng — xong ảnh hiện tại rồi thoát)", flush=True)


def load_manifest(source: str) -> list[dict]:
    """Images to process, from a manifest if there is one, else the raw split."""
    manifest = KTVIC / source
    if manifest.exists() and manifest.name.endswith("_manifest.json"):
        return json.loads(manifest.read_text(encoding="utf-8"))["images"]

    data = json.loads((KTVIC / source).read_text(encoding="utf-8"))
    captions: dict[str, list[str]] = {}
    for ann in data.get("annotations", []):
        if ann.get("caption"):
            captions.setdefault(str(ann.get("image_id")), []).append(ann["caption"])
    return [
        {
            "image_id": str(img.get("id", img.get("image_id"))),
            "file_name": img.get("file_name") or img.get("filename") or "",
            "captions": captions.get(str(img.get("id", img.get("image_id"))), []),
        }
        for img in data.get("images", [])
        if img.get("file_name") or img.get("filename")
    ]


def summarise(props: list[dict]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for p in props:
        status = (p.get("verification") or {}).get("status")
        counts[str(getattr(status, "value", status))] += 1
    return dict(counts)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="train_data.json")
    parser.add_argument("--out", default=str(DATA / "stage1"))
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--of", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0, help="0 = tất cả")
    parser.add_argument("--generator", default="qwen2.5-vl-7b")
    parser.add_argument("--verifier", default="vintern-1b")
    parser.add_argument("--max-entities", type=int, default=8)
    parser.add_argument("--max-pairs", type=int, default=8)
    parser.add_argument(
        "--verifier-tiles", type=int, default=4,
        help=(
            "InternVL tile budget for the verifier. Probe cost scales with the "
            "tile count: measured on a real KTVIC image, max_tiles=6 gives 7 "
            "tiles at 850 ms/probe and max_tiles=4 gives 3 at 323 ms. 4 is the "
            "default because the accuracy check behind the cut used one large "
            "object and does not establish that a single tile is enough for "
            "small ones."
        ),
    )
    parser.add_argument(
        "--no-colour-cross-check", action="store_true",
        help="tắt bộ kiểm chứng màu thứ hai (rẻ hơn, nhưng nhãn màu kém tin cậy)",
    )
    args = parser.parse_args()

    if not 0 <= args.shard < args.of:
        raise SystemExit(f"--shard phải trong [0, {args.of})")

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    from rescap.pipeline.generate import assert_clean, generate
    from rescap.pipeline.realize import realize
    from rescap.pipeline.select import select
    from rescap.pipeline.verify import verify
    from rescap.vlm.registry import get_vlm

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    items = load_manifest(args.source)
    items = items[args.shard :: args.of]
    if args.limit:
        items = items[: args.limit]

    done = {p.stem for p in out_dir.glob("*.json")}
    todo = [i for i in items if Path(i["file_name"]).stem not in done]
    print(f"mảnh {args.shard}/{args.of}: {len(items)} ảnh, "
          f"{len(items) - len(todo)} đã xong, còn {len(todo)}")
    if not todo:
        print("không còn gì để làm")
        return 0

    from PIL import Image

    print(f"nạp {args.generator} …", flush=True)
    gen_model = get_vlm(args.generator).load()
    print(f"nạp {args.verifier} …", flush=True)
    ver_model = get_vlm(args.verifier).load()
    if hasattr(ver_model, "max_tiles"):
        ver_model.max_tiles = args.verifier_tiles
        print(f"  ngân sách ô ảnh của bộ kiểm chứng: {args.verifier_tiles}")
    colour_verifier = None if args.no_colour_cross_check else gen_model

    totals: Counter[str] = Counter()
    failures: list[dict] = []
    started = time.time()

    for n, item in enumerate(todo, 1):
        if _stop:
            break
        name = item["file_name"]
        stem = Path(name).stem
        path = KTVIC / "images" / name
        if not path.exists():
            failures.append({"file": name, "stage": "input", "error": "không thấy file"})
            continue

        t0 = time.time()
        try:
            image = Image.open(path).convert("RGB")
            entities, props, gstats = generate(
                gen_model, image,
                max_entities=args.max_entities, max_pairs=args.max_pairs,
            )
            assert_clean(props)
            results, vstats = verify(
                ver_model, image, entities, props, colour_verifier=colour_verifier
            )
            selection = select(props, entities)
            selected_ids = set(getattr(selection, "selected_ids", []) or [])
            hedged_ids = list(getattr(selection, "hedged_ids", []) or [])
            chosen = [p for p in props if str(p.get("id")) in selected_ids]
            caption = None
            if chosen:
                realized = realize(chosen, entities, hedged_ids=hedged_ids)
                caption = realized.caption

            counts = summarise(props)
            totals.update(counts)

            # Written whole, then renamed: a crash mid-write would otherwise
            # leave a truncated file that the resume logic counts as done.
            record = {
                "image_id": item.get("image_id"),
                "file_name": name,
                "reference_captions": item.get("captions", []),
                "entities": entities,
                "propositions": props,
                "selected_ids": sorted(selected_ids),
                "hedged_ids": hedged_ids,
                "caption": caption,
                "verdicts": counts,
                "stats": {
                    "generation": vars(gstats),
                    "verification_notes": list(vstats.notes),
                    "colour_cross_checked": vstats.colour_cross_checked,
                    "colour_disagreements": vstats.colour_disagreements,
                    "probe_calls": vstats.probe_calls,
                    "seconds": round(time.time() - t0, 1),
                },
            }
            tmp = out_dir / f".{stem}.tmp"
            tmp.write_text(
                json.dumps(record, ensure_ascii=False, indent=1, default=str),
                encoding="utf-8",
            )
            tmp.rename(out_dir / f"{stem}.json")

        except Exception as e:  # keep going; a stopped run loses the whole shard
            failures.append({
                "file": name, "stage": "pipeline",
                "error": f"{type(e).__name__}: {e}",
                "traceback": traceback.format_exc()[-800:],
            })
            print(f"  [{n}/{len(todo)}] {name}  ✗ {type(e).__name__}: {e}", flush=True)
            continue

        elapsed = time.time() - started
        rate = elapsed / n
        left = (len(todo) - n) * rate
        print(
            f"  [{n}/{len(todo)}] {name}  {time.time()-t0:5.1f}s  {counts}"
            f"   còn ~{left/3600:.1f}h",
            flush=True,
        )

    report = {
        "shard": f"{args.shard}/{args.of}",
        "source": args.source,
        "processed": len(list(out_dir.glob("*.json"))),
        "verdicts_total": dict(totals),
        "failures": failures,
        "stopped_early": _stop,
        "generator": args.generator,
        "verifier": args.verifier,
        "colour_cross_check": not args.no_colour_cross_check,
        "seconds": round(time.time() - started, 1),
    }
    (out_dir / f"_report_shard{args.shard}of{args.of}.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    total = sum(totals.values()) or 1
    print(f"\n{'='*62}")
    print(f"  xử lý xong : {report['processed']} ảnh")
    print(f"  mệnh đề    : {total}")
    for status, n in totals.most_common():
        print(f"     {status:<12} {n:>7}  {n/total*100:>5.1f}%")
    if failures:
        print(f"  ✗ hỏng     : {len(failures)} ảnh — xem _report_shard*.json")
    print(f"  thời gian  : {report['seconds']/3600:.2f} h")
    print(f"{'='*62}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
