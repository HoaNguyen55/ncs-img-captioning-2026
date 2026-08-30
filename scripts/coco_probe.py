#!/usr/bin/env python
""" (nhật ký NC) — probe khái quát hóa xuyên ngôn ngữ trên COCO-2014 (Karpathy test, 2.500 ảnh).

    # zero-shot, cả hai chế độ
    python research/scripts/coco_probe.py --images-dir /root/coco_images --out-dir /root/probe_out
    # hệ chưng cất
    python research/scripts/coco_probe.py --adapter /root/off4090_sft_slim ...
    # chia 2 máy
    ... --shard 0 --of 2   |   ... --shard 1 --of 2

Sinh caption TIẾNG VIỆT trên ảnh COCO bằng ĐÚNG cấu hình evaluate.py (bf16,
sdpa, min/max_pixels, greedy, trần 60/160 token) — đo hệ thật của Bảng 1/2,
không phải một cấu hình demo. Thiết kế theo generate_stage1.py: resumable
(mỗi dòng jsonl một ảnh, ảnh có rồi thì bỏ qua) và shard không cần phối hợp.

Chấm điểm KHÔNG làm ở đây — score_coco_probe.py chạy trên máy local với
instances_val2014 + caption tham chiếu (vàng = hợp, đăng ký (nhật ký NC)).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from evaluate import PROMPTS  # noqa: E402  — cùng prompt với Bảng 1/2

P = 28 * 28
MAX_NEW = {"short": 60, "detailed": 160}


def load_model(adapter: str | None):
    import torch
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    from rescap.vlm.base import dtype_kwarg

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-VL-7B-Instruct", attn_implementation="sdpa",
        device_map={"": 0}, **dtype_kwarg(torch.bfloat16))
    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter)
        # Gộp adapter vào trọng số: cùng phép tính W+BA, bỏ overhead PEFT
        # mỗi token (đo thật: 11,8s/ảnh chưa gộp vs ~5,8s đã gộp ở chế độ
        # chi tiết). Khai báo trong artifact: caption distill sinh từ adapter
        # ĐÃ GỘP — tương đương toán học, greedy như cũ.
        model = model.merge_and_unload()
    model.eval()
    processor = AutoProcessor.from_pretrained(
        "Qwen/Qwen2.5-VL-7B-Instruct", min_pixels=256 * P, max_pixels=1024 * P)
    return model, processor


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", default="research/paper/data/coco_probe/manifest.json")
    ap.add_argument("--images-dir", required=True)
    ap.add_argument("--adapter", default=None, help="bỏ trống = zero-shot")
    ap.add_argument("--modes", default="short,detailed")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--of", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    import torch
    from PIL import Image

    system = "distill" if args.adapter else "zeroshot"
    man = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    images = sorted(man["images"], key=lambda x: x["cocoid"])[args.shard:: args.of]
    if args.limit:
        images = images[: args.limit]
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    model, processor = load_model(args.adapter)
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]

    for mode in modes:
        out = out_root / f"{system}-{mode}.shard{args.shard}of{args.of}.jsonl"
        done = set()
        if out.exists():
            for line in out.read_text(encoding="utf-8").splitlines():
                try:
                    done.add(json.loads(line)["cocoid"])
                except Exception:
                    pass
        todo = [i for i in images if i["cocoid"] not in done]
        print(f"[{system}/{mode}] {len(done)} có sẵn, {len(todo)} cần sinh", flush=True)
        t0 = time.time()
        with out.open("a", encoding="utf-8") as fh:
            for k, img in enumerate(todo):
                path = Path(args.images_dir) / img["filename"]
                image = Image.open(path).convert("RGB")
                messages = [{"role": "user", "content": [
                    {"type": "image"}, {"type": "text", "text": PROMPTS[mode]}]}]
                text = processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True)
                inputs = processor(text=[text], images=[image],
                                   return_tensors="pt").to(model.device)
                with torch.no_grad():
                    ids = model.generate(**inputs, max_new_tokens=MAX_NEW[mode],
                                         do_sample=False)
                caption = processor.batch_decode(
                    ids[:, inputs["input_ids"].shape[1]:],
                    skip_special_tokens=True)[0].strip()
                fh.write(json.dumps({"cocoid": img["cocoid"], "caption": caption},
                                    ensure_ascii=False) + "\n")
                fh.flush()
                if (k + 1) % 50 == 0:
                    rate = (time.time() - t0) / (k + 1)
                    print(f"  {k+1}/{len(todo)} · {rate:.2f}s/ảnh · còn "
                          f"~{rate*(len(todo)-k-1)/60:.0f} phút", flush=True)
        print(f"[{system}/{mode}] XONG → {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
