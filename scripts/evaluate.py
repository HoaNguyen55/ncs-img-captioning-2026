#!/usr/bin/env python
"""Score captions against KTVIC test and print a table comparable to the paper's.

    # score a file of predictions
    python scripts/evaluate.py --predictions preds.json --name "VSPS-D"

    # generate with a model, then score
    python scripts/evaluate.py --model qwen2.5-vl-7b --name "Qwen zero-shot"
    python scripts/evaluate.py --model qwen2.5-vl-7b --adapter ~/ncs-data/runs/dpo

`--predictions` takes `{image_id: "caption"}` or `{image_id: ["caption"]}`.

**Two things here exist to stop a wrong table being printed.**

*Scale.* `pycocoevalcap` returns CIDEr on its raw scale -- a perfect prediction
scores about 3.4 -- while every published KTVIC number is ×100. GRIT's 136.0 is
1.36 raw. Reporting the raw figure beside theirs would understate us by two
orders of magnitude, and reporting theirs beside our raw one would do the
reverse. Everything is converted once, here, and the column says which scale it
is in.

*Tokenization.* Vietnamese metrics depend on the segmenter, and only
RDRSegmenter reproduces KTVIC's own `segment_caption` (100% against
underthesea's 82.6%, ). The segmenter and its version are printed with the
scores and written into the results file, because a Vietnamese n-gram score
without them cannot be checked by anyone.

**Generation and scoring do not share a process.** RDRSegmenter runs on the JVM
through `jnius`, and starting a JVM in a process that has already initialised
CUDA deadlocks -- observed here as an evaluation stopping dead on
`Loading Word Segmentation model` after captioning all 150 images, while the
identical scoring code ran in 0.3 s in a process that had never touched the GPU.
Predictions are therefore written to disk the moment generation ends, and
scoring re-executes this script with `--predictions`. The GPU work survives any
scoring failure, which it did not before: three runs were lost to this.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

DATA = Path(os.environ.get("NCS_DATA", Path.home() / "ncs-data"))
KTVIC = DATA / "datasets" / "ktvic"
#: Two prompts, because the reference style decides what a metric can reward.
#: KTVIC references are ONE short sentence (~12 words); a detailed caption is
#: 8x longer and CIDEr's length penalty removes it entirely. Which prompt was
#: used therefore has to travel with the number.
PROMPTS = {
    "detailed": "Mô tả chi tiết bức ảnh này bằng tiếng Việt.",
    "short": (
        "Mô tả bức ảnh này bằng MỘT câu tiếng Việt ngắn gọn, "
        "giống chú thích ảnh. Không liệt kê, không giải thích."
    ),
    # baseline instructed-zero-shot: steelman cho câu hỏi "chỉ cần
    # prompt khéo?"; mọi ràng buộc của hệ chính được nêu tường minh trong prompt.
    "instructed": (
        "Mô tả chi tiết bức ảnh này bằng tiếng Việt. CHỈ nêu những gì thấy "
        "rõ ràng trong ảnh, không suy đoán. Nếu không chắc giới tính của "
        "người trong ảnh, dùng từ trung tính \"người\". Không đoán màu sắc "
        "khi không nhìn rõ. Không dùng từ rào đón như \"có vẻ\", \"dường như\"."
    ),
}
PROMPT = PROMPTS["detailed"]

#: pycocoevalcap's raw scale -> the ×100 convention every KTVIC number uses.
PUBLISHED_SCALE = 100.0
SCALED = ("BLEU-1", "BLEU-2", "BLEU-3", "BLEU-4", "METEOR", "ROUGE-L", "CIDEr", "SPICE")


def references(split: str) -> dict[str, list[str]]:
    data = json.loads((KTVIC / split).read_text(encoding="utf-8"))
    refs: dict[str, list[str]] = {}
    for ann in data.get("annotations", []):
        if ann.get("caption"):
            refs.setdefault(str(ann["image_id"]), []).append(ann["caption"])
    return refs


def file_names(split: str) -> dict[str, str]:
    data = json.loads((KTVIC / split).read_text(encoding="utf-8"))
    return {
        str(img.get("id", img.get("image_id"))): img.get("file_name") or img.get("filename")
        for img in data.get("images", [])
    }


def generate(args, image_ids: list[str]) -> dict[str, list[str]]:
    """Caption each image with the model under test."""
    import torch
    from PIL import Image
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    from rescap.vlm.base import dtype_kwarg

    names = file_names(args.split)
    P = 28 * 28
    load_kw = dict(attn_implementation="sdpa", device_map={"": 0},
                   **dtype_kwarg(torch.bfloat16))
    if args.four_bit:
        from transformers import BitsAndBytesConfig

        load_kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-VL-7B-Instruct", **load_kw)
    if args.adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, args.adapter)
        print(f"  adapter: {args.adapter}")
    model.eval()
    processor = AutoProcessor.from_pretrained(
        "Qwen/Qwen2.5-VL-7B-Instruct",
        min_pixels=256 * P, max_pixels=args.max_vision_tokens * P)

    out: dict[str, list[str]] = {}
    started = time.time()
    for n, image_id in enumerate(image_ids, 1):
        image = Image.open(KTVIC / "images" / names[image_id]).convert("RGB")
        messages = [{"role": "user", "content": [
            {"type": "image"}, {"type": "text", "text": args.prompt_text}]}]
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=[image], return_tensors="pt").to(model.device)
        with torch.no_grad():
            ids = model.generate(**inputs, max_new_tokens=args.max_new_tokens,
                                 do_sample=False)
        trimmed = ids[:, inputs.input_ids.shape[1]:]
        out[image_id] = [processor.batch_decode(
            trimmed, skip_special_tokens=True)[0].strip()]
        if n % 25 == 0 or n == len(image_ids):
            rate = (time.time() - started) / n
            print(f"  [{n}/{len(image_ids)}] {rate:.2f}s/ảnh  "
                  f"còn ~{(len(image_ids)-n)*rate/60:.0f} phút", flush=True)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="test_data.json")
    parser.add_argument("--predictions", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--name", default="hệ thống")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument("--max-vision-tokens", type=int, default=512)
    parser.add_argument("--four-bit", action="store_true")
    parser.add_argument(
        "--prompt", default="detailed", choices=sorted(PROMPTS),
        help=("`short` khớp phong cách tham chiếu KTVIC (một câu); "
              "`detailed` là thứ bài báo thật sự muốn sinh ra"),
    )
    parser.add_argument(
        "--score-here", action="store_true",
        help="chấm ngay trong tiến trình này (dùng nội bộ khi đã có --predictions)",
    )
    parser.add_argument("--out", default=None)
    parser.add_argument("--segmenter", default="rdrsegmenter")
    parser.add_argument(
        "--also-syllable", action="store_true",
        help="thêm bảng theo âm tiết (dấu trắng) cho phụ lục",
    )
    args = parser.parse_args()
    args.prompt_text = PROMPTS[args.prompt]

    refs = references(args.split)
    image_ids = sorted(refs)
    if args.limit:
        image_ids = image_ids[: args.limit]
    print(f"{args.split}: {len(image_ids)} ảnh")

    if args.predictions:
        raw = json.loads(Path(args.predictions).read_text(encoding="utf-8"))
        preds = {str(k): (v if isinstance(v, list) else [v]) for k, v in raw.items()}
    elif args.model:
        preds = generate(args, image_ids)
        # Written before anything else can fail. Generation is the expensive
        # half and it must not be repeated because scoring broke.
        pred_path = Path(args.out or DATA / "results" /
                         f"{args.name.replace(' ', '_')}.json").with_suffix(".preds.json")
        pred_path.parent.mkdir(parents=True, exist_ok=True)
        pred_path.write_text(
            json.dumps({i: preds[i][0] for i in image_ids}, ensure_ascii=False, indent=1),
            encoding="utf-8")
        print(f"\n  đã lưu dự đoán -> {pred_path}")

        if not args.score_here:
            # A fresh interpreter, so the JVM starts in a process with no CUDA
            # context. See the module docstring.
            import subprocess

            argv = [sys.executable, __file__, "--predictions", str(pred_path),
                    "--name", args.name, "--split", args.split,
                    "--segmenter", args.segmenter, "--prompt", args.prompt,
                    "--score-here"]
            if args.limit:
                argv += ["--limit", str(args.limit)]
            if args.also_syllable:
                argv += ["--also-syllable"]
            if args.out:
                argv += ["--out", args.out]
            print(f"  chấm điểm ở tiến trình riêng (JVM không dùng chung với CUDA)…\n")
            return subprocess.call(argv)
    else:
        raise SystemExit("cần --predictions hoặc --model")

    missing = [i for i in image_ids if i not in preds]
    if missing:
        # Scoring only the images a model happened to caption would report a
        # number for an easier subset than the one everyone else reports on.
        raise SystemExit(
            f"thiếu dự đoán cho {len(missing)} / {len(image_ids)} ảnh "
            f"(vd {missing[:3]}). Không chấm tập con — số sẽ không so được "
            f"với 558 ảnh của GRIT."
        )

    from rescap.metrics import CaptionMetrics

    gt = {i: refs[i] for i in image_ids}
    rows = []
    modes = [("từ (word-level)", args.segmenter)]
    if args.also_syllable:
        modes.append(("âm tiết (dấu trắng)", "whitespace"))

    for label, segmenter in modes:
        # `tokenize` stays True for both: syllable level is a segmenter that
        # happens to be the identity, not an absence of tokenization. Passing
        # tokenize=False skips the step that turns the aligned structures into
        # plain strings, and the scorers then hang on dicts.
        # METEOR chạy JVM qua pipe: pycocoevalcap ghi TOÀN BỘ dòng rồi mới
        # đọc — 558 caption CHI TIẾT tràn buffer pipe và deadlock (đo được
        # 19/08: chấm short lọt, chấm detailed treo y hệt trên hai máy khác
        # nhau). Caption ngắn vẫn đo METEOR bình thường.
        use_meteor = args.prompt != "detailed"
        scorer = CaptionMetrics(language="vi", use_meteor=use_meteor, tokenize=True,
                                segmenter=segmenter)
        scores = scorer.compute(gt, {i: preds[i] for i in image_ids})
        scaled = {k: (v * PUBLISHED_SCALE if k in SCALED and isinstance(v, float) else v)
                  for k, v in scores.items()}
        # scorer.segmenter, KHÔNG phải biến vòng lặp: khi VnCoreNLP vắng mặt,
        # bộ chấm đổi tên thành "...!whitespace-fallback" và tên đó phải theo
        # điểm số vào JSON — nếu ghi tên được yêu cầu, điểm fallback đội lốt
        # điểm chuẩn và không phép soát nào phía sau bắt được nữa.
        rows.append({"tokenization": label, "segmenter": scorer.segmenter,
                     "scores_scaled_x100": scaled, "scores_raw": scores,
                     "warnings": list(scorer.warnings)})

    print(f"\n{'='*72}\n  {args.name}   ({len(image_ids)} ảnh, thang ×100 như KTVIC công bố)\n{'='*72}")
    header = f"  {'chỉ số':<12}" + "".join(f"{r['tokenization']:>24}" for r in rows)
    print(header + "\n  " + "-" * (len(header) - 2))
    for key in ("BLEU-1", "BLEU-4", "METEOR", "ROUGE-L", "CIDEr"):
        line = f"  {key:<12}"
        for r in rows:
            v = r["scores_scaled_x100"].get(key)
            line += f"{v:>24.1f}" if isinstance(v, float) else f"{'n/a':>24}"
        print(line)
    # CHAIR alongside the n-gram table, because the two disagree about what a
    # long caption is worth and a reader needs to see both at once. It is a
    # RELATIVE measure here: gold comes from the reference captions, so an
    # object no annotator mentioned counts as invented (`rescap/chair.py`).
    from rescap.chair import chair

    chair_result = chair({i: preds[i][0] for i in image_ids}, gt, strict_ids=False)
    print(f"\n  ảo giác vật thể (CHAIR — gold rút từ caption, là CẬN TRÊN):")
    print(f"    CHAIR_s {chair_result.chair_s*100:>5.1f}%   "
          f"CHAIR_i {chair_result.chair_i*100:>5.1f}%   "
          f"{chair_result.mentions_per_caption:.2f} vật thể/caption")
    top = ", ".join(f"{k}({v})" for k, v in chair_result.top_hallucinated.most_common(6))
    print(f"    bịa nhiều nhất: {top}")
    print(f"    ⚠ CHAIR_s tỉ lệ với ĐỘ DÀI caption — chỉ so giữa các hệ "
          f"CÙNG độ dài ")

    print(f"\n  đối chiếu — GRIT (KTVIC Bảng 3, cùng 558 ảnh):")
    print(f"    CIDEr 136.0 · BLEU-4 34.2 · METEOR 28.5 · ROUGE-L 56.1")
    for r in rows:
        for w in r["warnings"]:
            print(f"  ! {w}")

    out = Path(args.out or DATA / "results" / f"{args.name.replace(' ', '_')}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "name": args.name, "split": args.split, "n_images": len(image_ids),
        "prompt_style": args.prompt, "prompt": args.prompt_text,
        "adapter": args.adapter, "model": args.model,
        "scale": "x100 (như số công bố của KTVIC)",
        "results": rows,
        "chair": chair_result.as_dict(),
        "predictions": {i: preds[i][0] for i in image_ids},
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n  đã ghi {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
