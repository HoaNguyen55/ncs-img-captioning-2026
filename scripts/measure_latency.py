#!/usr/bin/env python
"""Đo độ trễ suy luận batch=1 cho claim "~N giây/ảnh" của §5b .

    python research/scripts/measure_latency.py \\
        --adapter /root/ncs-data/runs/off4090_sft \\
        --n 60 --out ~/ncs-data/results/latency_off4090sft.json

Cùng đường nạp model và vòng sinh với `evaluate.py` (bf16 + adapter, batch=1,
greedy) — đo cái hệ THẬT SỰ chạy khi đánh giá, không phải một cấu hình demo.
60 ảnh đầu của tập test theo thứ tự image_id, 3 ảnh warmup không tính (lần
sinh đầu chứa chi phí biên dịch/khởi tạo CUDA). Ghi trung vị/p90/trung bình
từng chế độ vào JSON để con số trong bài tái tạo được từ đĩa.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluate import KTVIC, PROMPTS, file_names, references


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--split", default="test_data.json")
    parser.add_argument("--n", type=int, default=60)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--max-vision-tokens", type=int, default=1024)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    import torch
    from PIL import Image
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    from rescap.vlm.base import dtype_kwarg

    P = 28 * 28
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-VL-7B-Instruct", attn_implementation="sdpa",
        device_map={"": 0}, **dtype_kwarg(torch.bfloat16))
    if args.adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()
    processor = AutoProcessor.from_pretrained(
        "Qwen/Qwen2.5-VL-7B-Instruct",
        min_pixels=256 * P, max_pixels=args.max_vision_tokens * P)

    names = file_names(args.split)
    ids = sorted(references(args.split), key=int)[: args.n]
    gpu = torch.cuda.get_device_name(0)

    result = {"adapter": args.adapter, "gpu": gpu, "n": len(ids),
              "warmup_excluded": args.warmup, "batch": 1, "modes": {}}
    for mode in ("short", "detailed"):
        max_new = 60 if mode == "short" else 160
        times: list[float] = []
        for k, image_id in enumerate(ids):
            image = Image.open(KTVIC / "images" / names[image_id]).convert("RGB")
            messages = [{"role": "user", "content": [
                {"type": "image"}, {"type": "text", "text": PROMPTS[mode]}]}]
            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
            inputs = processor(text=[text], images=[image],
                               return_tensors="pt").to(model.device)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.no_grad():
                model.generate(**inputs, max_new_tokens=max_new, do_sample=False)
            torch.cuda.synchronize()
            if k >= args.warmup:
                times.append(time.perf_counter() - t0)
        result["modes"][mode] = {
            "max_new_tokens": max_new,
            "mean_s": round(statistics.mean(times), 3),
            "median_s": round(statistics.median(times), 3),
            "p90_s": round(statistics.quantiles(times, n=10)[8], 3),
            "min_s": round(min(times), 3),
            "max_s": round(max(times), 3),
        }
        m = result["modes"][mode]
        print(f"{mode}: trung vị {m['median_s']}s · trung bình {m['mean_s']}s "
              f"· p90 {m['p90_s']}s ({len(times)} ảnh, {gpu})", flush=True)

    out = Path(args.out).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    print(f"đã ghi {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
