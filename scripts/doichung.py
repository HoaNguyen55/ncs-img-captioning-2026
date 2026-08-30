#!/usr/bin/env python
"""đối chứng NGOÀI trên cùng backbone/cùng test-558/cùng thước đo.

    python research/scripts/doichung.py --method selfcorrect --prompt short \
        --out $NCS_DATA/results/sc-short.preds.json
    python research/scripts/doichung.py --method vcd --prompt detailed \
        --out $NCS_DATA/results/vcd-detailed.preds.json

Hai phương pháp training-free từ văn liệu, chạy trên Qwen2.5-VL-7B zero-shot:

* selfcorrect — Self-Correction prompting (họ Self-Refine): sinh nháp, rồi
  yêu cầu chính mô hình rà và viết lại bỏ nội dung không chắc/không thấy rõ.
* vcd — Visual Contrastive Decoding (Leng et al., CVPR 2024): giải mã đối
  chiếu logits ảnh gốc với ảnh nhiễu khuếch tán;
  l = (1+α)·l_gốc − α·l_nhiễu, kèm ràng buộc hợp lý (plausibility) β trên
  phân phối gốc. Tham số theo bài gốc: α=1, β=0,1, nhiễu T=500/1000 lịch
  tuyến tính. Greedy để khớp quy trình chấm của bài (evaluate.py greedy).

Chấm điểm: dùng evaluate.py --predictions <out> --score-here như mọi hệ khác.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

NCS = Path(os.environ.get("NCS_DATA", Path.home() / "ncs-data"))
KTVIC = NCS / "datasets" / "ktvic"

PROMPTS = {
    "detailed": "Mô tả chi tiết bức ảnh này bằng tiếng Việt.",
    "short": (
        "Mô tả bức ảnh này bằng MỘT câu tiếng Việt ngắn gọn, "
        "giống chú thích ảnh. Không liệt kê, không giải thích."
    ),
}
SELF_CORRECT_ASK = (
    "Rà lại mô tả trên so với bức ảnh: xóa mọi chi tiết KHÔNG chắc chắn hoặc "
    "không thấy rõ trong ảnh (vật thể, màu sắc, số lượng, giới tính, hành "
    "động). Không thêm thông tin mới. Chỉ in bản mô tả đã sửa."
)
MAX_NEW = {"short": 60, "detailed": 160}


def load_split(split: str):
    data = json.loads((KTVIC / split).read_text(encoding="utf-8"))
    return {
        str(im.get("id", im.get("image_id"))): im.get("file_name") or im.get("filename")
        for im in data.get("images", [])
    }


def load_model():
    import torch
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    P = 28 * 28
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-VL-7B-Instruct", attn_implementation="sdpa",
        device_map={"": 0}, torch_dtype=torch.bfloat16)
    model.eval()
    processor = AutoProcessor.from_pretrained(
        "Qwen/Qwen2.5-VL-7B-Instruct", min_pixels=256 * P, max_pixels=1024 * P)
    return model, processor


def chat_inputs(processor, image, text_prompt, history=None):
    content = [{"type": "image"}, {"type": "text", "text": text_prompt}]
    messages = [{"role": "user", "content": content}]
    if history:
        messages += history
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)
    return processor(text=[text], images=[image], return_tensors="pt")


def diffusion_noise(image, t_step=500, t_total=1000, seed=42):
    """Nhiễu khuếch tán thuận theo VCD: x_t = sqrt(ᾱ_t)x0 + sqrt(1-ᾱ_t)ε."""
    import numpy as np
    rng = np.random.default_rng(seed)
    x = np.asarray(image, dtype=np.float32) / 255.0
    betas = np.linspace(1e-4, 0.02, t_total, dtype=np.float64)
    abar = float(np.cumprod(1.0 - betas)[t_step - 1])
    noisy = (abar ** 0.5) * x + ((1.0 - abar) ** 0.5) * rng.standard_normal(x.shape)
    from PIL import Image as PILImage
    return PILImage.fromarray(
        (np.clip(noisy, 0.0, 1.0) * 255.0).astype("uint8"))


def gen_greedy(model, inputs, max_new):
    import torch
    with torch.no_grad():
        ids = model.generate(**inputs, max_new_tokens=max_new, do_sample=False)
    return ids[:, inputs["input_ids"].shape[1]:]


def selfcorrect_caption(model, processor, image, mode):
    import torch
    inputs = chat_inputs(processor, image, PROMPTS[mode]).to(model.device)
    draft_ids = gen_greedy(model, inputs, MAX_NEW[mode])
    draft = processor.batch_decode(draft_ids, skip_special_tokens=True)[0].strip()
    history = [
        {"role": "assistant", "content": [{"type": "text", "text": draft}]},
        {"role": "user", "content": [{"type": "text", "text": SELF_CORRECT_ASK}]},
    ]
    inputs2 = chat_inputs(processor, image, PROMPTS[mode], history).to(model.device)
    fixed_ids = gen_greedy(model, inputs2, MAX_NEW[mode])
    fixed = processor.batch_decode(fixed_ids, skip_special_tokens=True)[0].strip()
    return fixed or draft


def vcd_caption(model, processor, image, mode, alpha=1.0, beta=0.1,
                t_step=500, seed=42):
    import torch
    noisy = diffusion_noise(image, t_step=t_step, seed=seed)
    a = chat_inputs(processor, image, PROMPTS[mode]).to(model.device)
    b = chat_inputs(processor, noisy, PROMPTS[mode]).to(model.device)
    eos = model.generation_config.eos_token_id
    eos_set = set(eos if isinstance(eos, (list, tuple)) else [eos])

    # Qwen2.5-VL dùng M-RoPE đa phương thức — feed tay từng token làm lệch
    # position (khói 24/08: output thoái hóa "addCriterion…"). Đường chuẩn:
    # prepare_inputs_for_generation lo cache_position/rope cho từng stream.
    def stream_state(inp):
        return {"input_ids": inp["input_ids"], "attention_mask": inp["attention_mask"],
                "pixel_values": inp.get("pixel_values"),
                "image_grid_thw": inp.get("image_grid_thw"), "past": None}

    def step_logits(st):
        L = st["input_ids"].shape[1]
        kw = dict(attention_mask=st["attention_mask"], use_cache=True)
        if st["past"] is None:
            kw["pixel_values"] = st["pixel_values"]
            kw["image_grid_thw"] = st["image_grid_thw"]
            kw["cache_position"] = torch.arange(L, device=st["input_ids"].device)
            prep = model.prepare_inputs_for_generation(st["input_ids"], **kw)
        else:
            kw["cache_position"] = torch.tensor([L - 1], device=st["input_ids"].device)
            prep = model.prepare_inputs_for_generation(
                st["input_ids"], past_key_values=st["past"], **kw)
        out = model(**prep)
        st["past"] = out.past_key_values
        return out.logits[:, -1, :].float()

    sa, sb = stream_state(a), stream_state(b)
    toks: list[int] = []
    with torch.no_grad():
        la, lb = step_logits(sa), step_logits(sb)
        for _ in range(MAX_NEW[mode]):
            pa = torch.softmax(la, dim=-1)
            keep = pa >= beta * pa.max()
            cd = (1.0 + alpha) * la - alpha * lb
            cd = cd.masked_fill(~keep, float("-inf"))
            nxt = int(cd.argmax(dim=-1))
            if nxt in eos_set:
                break
            toks.append(nxt)
            step = torch.tensor([[nxt]], device=model.device)
            for st in (sa, sb):
                st["input_ids"] = torch.cat([st["input_ids"], step], dim=1)
                st["attention_mask"] = torch.cat(
                    [st["attention_mask"], st["attention_mask"].new_ones((1, 1))], dim=1)
            la, lb = step_logits(sa), step_logits(sb)
    return processor.batch_decode(
        torch.tensor([toks]), skip_special_tokens=True)[0].strip()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--method", required=True, choices=("selfcorrect", "vcd"))
    ap.add_argument("--prompt", default="detailed", choices=("short", "detailed"))
    ap.add_argument("--split", default="test_data.json")
    ap.add_argument("--dataset", default="ktvic", choices=("ktvic", "coco"),
                    help="coco: ảnh từ --coco-images, id từ manifest probe "
                         "(cùng 2.500 ảnh Karpathy của )")
    ap.add_argument("--coco-images", default="/root/coco_images")
    ap.add_argument("--coco-manifest", default=None, help="manifest khác (vd manifest_rest2500.json)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--of", type=int, default=1)
    ap.add_argument("--vcd-alpha", type=float, default=1.0)
    ap.add_argument("--vcd-beta", type=float, default=0.1)
    args = ap.parse_args()

    from PIL import Image
    if args.dataset == "coco":
        man_path = (Path(args.coco_manifest) if args.coco_manifest
                    else Path(__file__).resolve().parents[1].joinpath(
                        "paper/data/coco_probe/manifest.json"))
        man = json.loads(man_path.read_text(encoding="utf-8"))
        items = man.get("images") or man
        names = {str(it["cocoid"]): it["filename"] for it in items}
        img_dir = Path(args.coco_images)
    else:
        names = load_split(args.split)
        img_dir = KTVIC / "images"
    ids = sorted(names, key=lambda x: int(x))[args.shard :: args.of]
    if args.limit:
        ids = ids[: args.limit]
    out_path = Path(args.out)
    done: dict[str, str] = {}
    if out_path.exists():  # kháng lặp
        done = json.loads(out_path.read_text(encoding="utf-8"))
        ids = [i for i in ids if i not in done]
    print(f"{args.method}/{args.prompt}: {len(done)} có sẵn, còn {len(ids)}", flush=True)
    if not ids:
        return 0

    model, processor = load_model()
    fn = selfcorrect_caption if args.method == "selfcorrect" else vcd_caption
    t0 = time.time()
    for n, iid in enumerate(ids, 1):
        img = Image.open(img_dir / names[iid]).convert("RGB")
        if args.method == "vcd":
            cap = fn(model, processor, img, args.prompt,
                     alpha=args.vcd_alpha, beta=args.vcd_beta)
        else:
            cap = fn(model, processor, img, args.prompt)
        done[iid] = cap
        if n % 10 == 0 or n == len(ids):
            out_path.write_text(json.dumps(done, ensure_ascii=False), encoding="utf-8")
            rate = (time.time() - t0) / n
            print(f"  [{n}/{len(ids)}] {rate:.2f}s/ảnh  còn ~{(len(ids)-n)*rate/60:.0f} phút",
                  flush=True)
    out_path.write_text(json.dumps(done, ensure_ascii=False), encoding="utf-8")
    print(f"=== DOICHUNG {args.method} {args.prompt} DONE ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
