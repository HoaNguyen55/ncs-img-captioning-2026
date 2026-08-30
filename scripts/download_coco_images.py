#!/usr/bin/env python
"""Tải ảnh val2014 cho probe (nhật ký NC) (song song, resumable).

    python research/scripts/download_coco_images.py --out /root/coco_images
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import urllib.request
from pathlib import Path

URL = "http://images.cocodataset.org/val2014/{name}"


def fetch(name: str, out: Path) -> str | None:
    dst = out / name
    if dst.exists() and dst.stat().st_size > 1000:
        return None
    try:
        with urllib.request.urlopen(URL.format(name=name), timeout=60) as r:
            dst.write_bytes(r.read())
        return None
    except Exception as e:  # ghi lại, thử lại ở lần chạy sau
        return f"{name}: {e}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="research/paper/data/coco_probe/manifest.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    man = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    names = [i["filename"] for i in man["images"]]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    errs = []
    with cf.ThreadPoolExecutor(args.workers) as ex:
        for i, err in enumerate(ex.map(lambda n: fetch(n, out), names)):
            if err:
                errs.append(err)
            if (i + 1) % 500 == 0:
                print(f"  {i+1}/{len(names)}", flush=True)
    have = sum(1 for n in names if (out / n).exists())
    print(f"đủ {have}/{len(names)} ảnh; lỗi: {len(errs)}")
    for e in errs[:5]:
        print("  ", e)
    return 0 if have == len(names) else 1


if __name__ == "__main__":
    import sys

    sys.exit(main())
