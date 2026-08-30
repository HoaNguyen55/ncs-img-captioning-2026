#!/usr/bin/env python
"""Đo VRAM suy luận dưới trần cấp phát ép — số cho bảng 5b .

    # bf16 không trần (số "card 24GB")
    python scripts/measure_vram.py --adapter ~/ncs-data/runs/off_dpo

    # 4-bit dưới trần 12GB rồi 8GB (số "card phổ thông")
    python scripts/measure_vram.py --four-bit --cap-gb 12 --adapter ...
    python scripts/measure_vram.py --four-bit --cap-gb 8  --adapter ...

Phương pháp đã duyệt ở `torch.cuda.set_per_process_memory_fraction`
giả lập card nhỏ trên card cụm, đo `torch.cuda.max_memory_allocated` (cấp
phát thật) và `max_memory_reserved` (allocator giữ — số một card rời phải
chứa). Bài báo ghi số RESERVED kèm mô tả phương pháp — trung thực rằng đây
là trần-cấp-phát, không phải card vật lý; khác biệt còn lại là bộ nhớ màn
hình/hệ thống mà card rời phải gánh thêm.

OOM dưới trần là MỘT KẾT QUẢ, không phải lỗi harness: in "KHÔNG lọt trần"
và thoát 0 — bảng 5b cần cả câu trả lời phủ định.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from evaluate import KTVIC, PROMPTS, file_names, references  # cùng đường nạp với bảng chính


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--four-bit", action="store_true")
    parser.add_argument("--cap-gb", type=float, default=0,
                        help="trần cấp phát (GB); 0 = không trần")
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--images", type=int, default=8)
    parser.add_argument("--max-vision-tokens", type=int, default=512)
    parser.add_argument("--max-new-tokens", type=int, default=60)
    parser.add_argument("--split", default="test_data.json")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    import torch
    from PIL import Image
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    from rescap.vlm.base import dtype_kwarg

    total_gb = torch.cuda.get_device_properties(0).total_memory / 2**30
    if args.cap_gb:
        # Trần đặt TRƯỚC khi nạp — nạp xong mới ép thì trọng số đã chiếm chỗ
        # và phép đo mất nghĩa.
        torch.cuda.set_per_process_memory_fraction(args.cap_gb / total_gb, 0)
    label = (f"{'4-bit' if args.four_bit else 'bf16'}"
             f"/{args.max_vision_tokens}vt"
             + (f" cap {args.cap_gb:g}GB" if args.cap_gb else " no cap"))
    print(f"  card {total_gb:.1f}GB · cấu hình: {label}")

    result = {"config": label, "four_bit": args.four_bit, "cap_gb": args.cap_gb,
              "max_vision_tokens": args.max_vision_tokens,
              "adapter": args.adapter, "device_total_gb": round(total_gb, 1)}
    out_path = Path(args.out or Path.home() / "ncs-data" / "results" /
                    f"vram_{label.replace(' ', '_').replace('/', '-')}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        P = 28 * 28
        load_kw = dict(attn_implementation="sdpa", device_map={"": 0},
                       **dtype_kwarg(torch.bfloat16))
        if args.four_bit:
            from transformers import BitsAndBytesConfig

            load_kw["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True)
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2.5-VL-7B-Instruct", **load_kw)
        if args.adapter:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, args.adapter)
        model.eval()
        processor = AutoProcessor.from_pretrained(
            "Qwen/Qwen2.5-VL-7B-Instruct",
            min_pixels=256 * P, max_pixels=args.max_vision_tokens * P)

        names = file_names(args.split)
        ids = sorted(references(args.split))[: args.images]
        prompt = PROMPTS["short"]
        times = []
        captions = {}
        for image_id in ids:
            image = Image.open(KTVIC / "images" / names[image_id]).convert("RGB")
            messages = [{"role": "user", "content": [
                {"type": "image"}, {"type": "text", "text": prompt}]}]
            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
            inputs = processor(text=[text], images=[image],
                               return_tensors="pt").to(model.device)
            t0 = time.time()
            with torch.no_grad():
                out_ids = model.generate(**inputs, max_new_tokens=args.max_new_tokens,
                                         do_sample=False)
            times.append(time.time() - t0)
            # Lưu caption để SO SÁNH giữa các cấu hình: cùng lượng tử hoá thì
            # trần cấp phát không được phép đổi đầu ra (sinh tất định);
            # 4-bit vs bf16 thì được phép lệch — và độ lệch là số đáng báo cáo.
            trimmed = out_ids[:, inputs.input_ids.shape[1]:]
            captions[image_id] = processor.batch_decode(
                trimmed, skip_special_tokens=True)[0].strip()

        result.update({
            "fits": True,
            "images_measured": len(ids),
            "max_allocated_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2),
            "max_reserved_gb": round(torch.cuda.max_memory_reserved() / 2**30, 2),
            "seconds_per_image": round(sum(times) / len(times), 2),
            "captions": captions,
        })
        print(f"  ✅ LỌT: cấp phát đỉnh {result['max_allocated_gb']}GB · "
              f"allocator giữ {result['max_reserved_gb']}GB · "
              f"{result['seconds_per_image']}s/ảnh ({len(ids)} ảnh)")
        print("  Số đưa vào bài: max_reserved (card rời phải chứa được ngần này).")
    except torch.cuda.OutOfMemoryError as e:
        result.update({"fits": False, "error": str(e)[:200]})
        print(f"  ⛔ KHÔNG lọt trần {args.cap_gb:g}GB — đây là kết quả, ghi vào 5b.")

    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    print(f"  đã ghi {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
