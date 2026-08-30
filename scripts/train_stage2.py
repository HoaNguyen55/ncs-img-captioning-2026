#!/usr/bin/env python
"""Stage 2 — distil the verified propositions into the generator.

    python scripts/train_stage2.py --stage sft
    python scripts/train_stage2.py --stage dpo --adapter ~/ncs-data/runs/sft

Two phases, run in order. SFT teaches the shape of a grounded Vietnamese caption
from rung A of the preference data; DPO then teaches the **ordering** between the
three rungs, which is the part SFT cannot express -- a single target caption says
what to write, not what to write *instead of*.

**Configuration is measured, not chosen** (`DECISIONS.md` ). On this card
bf16 LoRA with a real optimizer and DPO's reference pass peaks at 23.82 GB with
256 vision tokens and OOMs at 512; 4-bit reaches 512 tokens at 19.15 GB. So
4-bit is not a concession, it is what buys twice the resolution.

**The reference policy costs no second model.** PEFT's `disable_adapter` turns
the trained model back into its own base, which is exactly the reference DPO
needs. That is also why the DPO phase must start from the SFT adapter rather
than from scratch: the reference should be the model we are actually improving
on.

**What is deliberately absent:** no early stopping on a metric computed from the
same verifier that produced the labels. That loop would optimise agreement with
Vintern rather than agreement with the image, and the number would look good
while meaning less.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

DATA = Path(os.environ.get("NCS_DATA", Path.home() / "ncs-data"))
KTVIC = DATA / "datasets" / "ktvic"
MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
P = 28 * 28
# Chosen from the card, not hard-coded:  measured 512 vision tokens as the
# most a 24 GB card fits at 4-bit;  measured 1,024 at bf16 on 48 GB. 1,280
# also fits there but leaves 3.1 GB, the same thin margin already rejected on the
# 4090 -- real runs vary in sequence length and fragment memory.
def _vision_token_budget() -> tuple[int, bool]:
    """`(vision_tokens, use_4bit)` for the card this process is on."""
    import torch

    gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    if gb >= 44:
        return 1024, False
    return 512, True


MIN_PIXELS = 256 * P


def load_jsonl(path: Path) -> list[dict]:
    """Read a jsonl, and say which file and how many rows.

    The path used to be hard-coded, so a run whose data had been built somewhere
    else read a stale file at the default location and trained on it without
    complaint -- three examples standing in for eighty-six, rc=0, adapter
    saved. Printing the resolved path is what makes that visible.
    """
    if not path.exists():
        raise SystemExit(
            f"không thấy {path}\n"
            "Chạy trước: python scripts/build_dpo_data.py "
            "--in $NCS_DATA/stage1 --out $NCS_DATA/stage2"
        )
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]
    print(f"  đọc {len(rows)} dòng từ {path}")
    return rows


def build_model(args, *, for_training: bool = True):
    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import (
        AutoProcessor,
        BitsAndBytesConfig,
        Qwen2_5_VLForConditionalGeneration,
    )

    from rescap.vlm.base import dtype_kwarg

    tokens, use_4bit = _vision_token_budget()
    if args.vision_tokens:
        tokens = args.vision_tokens
    if args.force_4bit:
        use_4bit = True
    use_8bit = getattr(args, "use_8bit", False)
    if use_8bit:
        # (nhật ký NC): điểm giữa trục lượng tử hóa — LLM.int8 (bitsandbytes), cùng
        # mọi cấu hình còn lại; so 4-bit NF4 (chính) và bf16 ( (nhật ký NC)).
        use_4bit = False
    if getattr(args, "no_4bit", False):
        # (nhật ký NC): đối chứng KHÔNG lượng tử trên 24GB — bf16 LoRA SFT-only
        # (không ref-pass DPO) ước dưới đỉnh 23,82GB đã đo; chạy kiểu fail-fast,
        # OOM thì đó là kết luận phần cứng, không phải lỗi.
        use_4bit = False
    max_pixels = tokens * P
    print(f"  card: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB "
          f"-> {tokens} token thị giác, "
          f"{'4-bit' if use_4bit else ('8-bit' if use_8bit else 'bf16')}")

    load_kw = dict(device_map={"": 0}, attn_implementation="sdpa",
                   **dtype_kwarg(torch.bfloat16))
    if use_4bit:
        load_kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    elif use_8bit:
        load_kw["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(MODEL_ID, **load_kw)
    processor = AutoProcessor.from_pretrained(
        MODEL_ID, min_pixels=MIN_PIXELS, max_pixels=max_pixels
    )

    if not for_training:
        return model, processor

    if use_4bit or use_8bit:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    else:
        model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    if args.adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, args.adapter, is_trainable=True)
        print(f"  tiếp tục từ adapter: {args.adapter}")
        if getattr(args, "stage", "") == "dpo":
            # THE reference policy, done properly (peer review , point 1).
            # `disable_adapter()` strips every adapter and yields the BASE
            # model, but DPO's reference is the SFT policy (Rafailov et al.,
            # 2023) -- anchoring to base loses the KL regularisation around
            # π_SFT. A second, frozen copy of the SFT adapter costs ~80 MB and
            # gives the correct reference: swap to it for the no-grad pass,
            # swap back to train.
            try:
                model.load_adapter(args.adapter, adapter_name="ref_sft")
                model.set_adapter("default")
                args._has_ref_adapter = True
                print("  tham chiếu DPO = π_SFT (adapter đóng băng thứ hai)")
            except Exception as e:
                args._has_ref_adapter = False
                print(f"  ⚠ không nạp được adapter tham chiếu ({type(e).__name__}) — "
                      f"lùi về neo mô hình gốc (Base-anchored), PHẢI ghi rõ trong bài")
    else:
        # Language model only, and deliberately so. Naming the seven projections
        # by bare name also matched them inside the vision tower, where 192 LoRA
        # tensors were created and **never received a gradient**: pixel_values do
        # not require grad, so under gradient checkpointing the visual encoder's
        # backward is skipped entirely. Measured: 392 of 584 trainable tensors
        # had gradients, and every one of the missing 192 was in `visual`.
        #
        # They cost optimizer state and told the paper a false story about what
        # was trained. Adapting the language side is also the right choice on the
        # merits -- the visual features are not what we are changing, what gets
        # said about them is -- but it has to be a choice, not an accident.
        model = get_peft_model(model, LoraConfig(
            r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.05,
            bias="none", task_type="CAUSAL_LM",
            target_modules=(
                r"^(?!.*visual).*\.(q_proj|k_proj|v_proj|o_proj"
                r"|gate_proj|up_proj|down_proj)$"
            ),
        ))
    model.print_trainable_parameters()
    return model, processor


class VLCollator:
    """Batch (image, prompt, response) into what Qwen2.5-VL's forward expects.

    trl 0.19's own dataset preparation reaches for `processing_class.pad_token`,
    which a `Qwen2_5_VLProcessor` does not have -- the tokeniser holds it. Rather
    than shim attributes onto the processor and hope the rest of the path is
    equally close, the collation happens here where it can be read.

    **Prompt tokens are masked to -100.** Training on the prompt as well would
    teach the model to reproduce the instruction, and since every example in this
    set shares one instruction, that is a large fraction of the loss spent on
    text we never want generated.
    """

    def __init__(self, processor):
        self.processor = processor
        tok = processor.tokenizer
        self.pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    def __call__(self, features: list[dict]):
        import torch

        # Rows arrive as plain dicts; the chat structure and the image are built
        # here so the dataset stays JSON-able and uncached.
        features = [
            to_chat(self.processor, f, f.get("_key", "response"))
            if "messages" not in f else f
            for f in features
        ]
        texts, images = [], []
        prompt_lengths = []
        for f in features:
            prompt_only = self.processor.apply_chat_template(
                f["messages"][:1], tokenize=False, add_generation_prompt=True
            )
            full = self.processor.apply_chat_template(f["messages"], tokenize=False)
            texts.append(full)
            images.append(f["images"])
            prompt_lengths.append(
                len(self.processor.tokenizer(prompt_only, add_special_tokens=False)["input_ids"])
            )

        batch = self.processor(
            text=texts, images=images, return_tensors="pt", padding=True
        )
        labels = batch["input_ids"].clone()
        labels[labels == self.pad_id] = -100
        for i, n in enumerate(prompt_lengths):
            labels[i, :n] = -100
        # Image placeholder tokens are inputs, never targets.
        for token_id in self._image_token_ids():
            labels[labels == token_id] = -100
        batch["labels"] = labels
        return batch

    def _image_token_ids(self):
        ids = []
        for name in ("image_token", "vision_start_token", "vision_end_token"):
            token = getattr(self.processor, name, None)
            if token:
                ids.append(self.processor.tokenizer.convert_tokens_to_ids(token))
        return [i for i in ids if isinstance(i, int) and i >= 0]


def image_for(row: dict):
    from PIL import Image

    path = KTVIC / "images" / row["file_name"]
    return Image.open(path).convert("RGB")


def to_chat(processor, row: dict, response_key: str) -> dict:
    """One training example as a chat with the image attached."""
    return {
        "images": [image_for(row)],
        "messages": [
            {"role": "user", "content": [
                {"type": "image"}, {"type": "text", "text": row["prompt"]}]},
            {"role": "assistant", "content": [
                {"type": "text", "text": row[response_key]}]},
        ],
    }


def run_sft(args) -> int:
    from trl import SFTConfig, SFTTrainer
    from datasets import Dataset

    rows = load_jsonl(Path(args.data) / "sft.jsonl")
    if args.limit:
        rows = rows[: args.limit]
    print(f"SFT: {len(rows)} ví dụ")

    model, processor = build_model(args)

    # `Dataset.from_generator` caches by a hash of the generator FUNCTION, and
    # this closure is byte-identical between runs. A run after an earlier
    # `--limit 3` smoke test silently reused that cache and trained on three
    # examples while reporting success -- 1 step, 1,509 tokens, rc=0.
    #
    # `from_list` over plain rows has no such cache, and the images are opened
    # in the collator instead of held in the dataset, so memory does not grow
    # with the training set either.
    dataset = Dataset.from_list([dict(row, _key="response") for row in rows])
    print(f"  tập huấn luyện: {len(dataset)} ví dụ")
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        processing_class=processor.tokenizer,
        data_collator=VLCollator(processor),
        args=SFTConfig(
            output_dir=args.out,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=args.grad_accum,
            num_train_epochs=args.epochs,
            learning_rate=args.lr,
            bf16=True,
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            logging_steps=10,
            save_strategy="epoch",
            report_to=[],
            remove_unused_columns=False,
            dataset_kwargs={"skip_prepare_dataset": True},
            seed=args.seed,
        ),
    )
    trainer.train()
    trainer.save_model(args.out)

    # Training curves are a required figure in the paper (leader's decision,
    # 18/08). The trainer's log_history is dumped verbatim -- plotting reads
    # this file, so the figure can be regenerated without rerunning anything.
    metrics_path = Path(args.out) / "metrics.jsonl"
    with metrics_path.open("w", encoding="utf-8") as fh:
        for entry in trainer.state.log_history:
            fh.write(json.dumps({"stage": "sft", **entry}, ensure_ascii=False) + "\n")
    print(f"\nđã lưu adapter SFT -> {args.out}")
    print(f"số liệu huấn luyện -> {metrics_path}")
    return 0


def sequence_logprob(model, batch, labels):
    """Sum of log p(token) over the response tokens of each sequence."""
    import torch

    out = model(**{k: v for k, v in batch.items() if k != "labels"})
    logits = out.logits[:, :-1, :]
    target = labels[:, 1:]
    mask = target != -100
    safe = target.masked_fill(~mask, 0)
    logp = torch.log_softmax(logits.float(), dim=-1)
    picked = torch.gather(logp, 2, safe.unsqueeze(-1)).squeeze(-1)
    return (picked * mask).sum(dim=-1)


def run_dpo(args) -> int:
    """DPO with a hand-written loop.

    trl 0.19's DPOTrainer drops `image_grid_thw` on the way to the model --
    `rot_pos_emb` receives None and the first forward dies. Its VLM path is not
    finished for Qwen2.5-VL, and trl 1.x cannot be used because it imports a
    symbol the pinned transformers does not export .

    The loss itself is four lines, and writing the loop keeps the image handling
    in the same collator the SFT phase already proved out, so the two phases
    cannot disagree about how a batch is built.

        L = -log sigmoid( beta * [ (pi_c - ref_c) - (pi_r - ref_r) ] )

    The reference is this model with its adapters switched off, so no second copy
    of the weights is loaded.
    """
    import torch
    from datasets import Dataset
    from torch.utils.data import DataLoader

    if not args.adapter:
        raise SystemExit(
            "DPO phải bắt đầu từ adapter SFT: --adapter <đường dẫn>\n"
            "Chính sách tham chiếu nên là mô hình ta đang cải thiện, "
            "không phải mô hình gốc chưa học gì."
        )

    rows = load_jsonl(Path(args.data) / "dpo.jsonl")
    if args.pair_types:
        keep = set(args.pair_types.split(","))
        before = len(rows)
        rows = [r for r in rows if any(r["pair_type"].startswith(k) for k in keep)]
        print(f"  lọc loại cặp {sorted(keep)}: {before} -> {len(rows)}")
    if args.limit:
        rows = rows[: args.limit]

    from collections import Counter
    mix = Counter(r["pair_type"] for r in rows)
    print(f"DPO: {len(rows)} cặp")
    for kind, n in mix.most_common():
        print(f"     {kind:<52} {n:>6}  ({n/len(rows)*100:.1f}%)")

    model, processor = build_model(args)
    collator = VLCollator(processor)

    def pair_collate(features):
        chosen = collator([to_chat(processor, f, "chosen") for f in features])
        rejected = collator([to_chat(processor, f, "rejected") for f in features])
        kinds = [f.get("pair_type", "?") for f in features]
        return chosen, rejected, kinds

    from collections import defaultdict
    per_type = defaultdict(lambda: {"n": 0, "correct": 0})
    loader = DataLoader(
        Dataset.from_list(rows), batch_size=1, shuffle=True,
        collate_fn=pair_collate, generator=torch.Generator().manual_seed(args.seed),
    )

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr)
    device = next(model.parameters()).device
    model.train()

    steps = 0
    metrics_log: list[dict] = []
    running = {"loss": 0.0, "acc": 0.0, "margin": 0.0, "n": 0}
    total_epochs = max(1, int(round(args.epochs)))

    for epoch in range(total_epochs):
        for i, (chosen, rejected, kinds) in enumerate(loader, 1):
            chosen = {k: v.to(device) for k, v in chosen.items()}
            rejected = {k: v.to(device) for k, v in rejected.items()}

            pi_c = sequence_logprob(model, chosen, chosen["labels"])
            pi_r = sequence_logprob(model, rejected, rejected["labels"])
            if getattr(args, "_has_ref_adapter", False):
                # π_SFT reference: frozen copy of the SFT adapter.
                with torch.no_grad():
                    model.set_adapter("ref_sft")
                    ref_c = sequence_logprob(model, chosen, chosen["labels"])
                    ref_r = sequence_logprob(model, rejected, rejected["labels"])
                    model.set_adapter("default")
            else:
                with torch.no_grad(), model.disable_adapter():
                    ref_c = sequence_logprob(model, chosen, chosen["labels"])
                    ref_r = sequence_logprob(model, rejected, rejected["labels"])

            margin = (pi_c - ref_c) - (pi_r - ref_r)
            loss = -torch.nn.functional.logsigmoid(args.beta * margin).mean()
            (loss / args.grad_accum).backward()

            # A DPO loop that silently trains nothing still finishes and still
            # saves an adapter, so the first backward is checked rather than
            # trusted. Gradient checkpointing warns "None of the inputs have
            # requires_grad" for the reference forward, where it is expected --
            # this distinguishes that harmless case from the fatal one.
            if steps == 0 and running["n"] == 0:
                with_grad = [q for q in params if q.grad is not None]
                total_norm = sum(float(q.grad.norm()) for q in with_grad)
                print(f"  kiểm tra gradient: {len(with_grad)}/{len(params)} "
                      f"tham số có gradient, tổng chuẩn {total_norm:.4f}", flush=True)
                if not with_grad or total_norm == 0.0:
                    raise SystemExit(
                        "GRADIENT KHÔNG CHẢY — vòng huấn luyện này không học gì. "
                        "Dừng thay vì lưu ra một adapter trông như đã huấn luyện."
                    )

            # Ordering accuracy PER PAIR TYPE (peer review , point 2):
            # unnormalised log-probs penalise long sequences, and rung A is the
            # longest, so the A>B pairs are where a length bias would show up
            # first. An aggregate number would hide that until the captions
            # came out clipped.
            kind_root = kinds[0].split(":")[0] if kinds else "?"
            per_type[kind_root]["n"] += 1
            per_type[kind_root]["correct"] += int((margin > 0).all().item())

            running["loss"] += loss.item()
            running["acc"] += (margin > 0).float().mean().item()
            running["margin"] += margin.mean().item()
            running["n"] += 1

            if i % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                steps += 1
                metrics_log.append({
                    "stage": "dpo", "epoch": epoch + 1, "step": steps,
                    "loss": loss.item(),
                    "ordering_accuracy": (margin > 0).float().mean().item(),
                    "reward_margin": margin.mean().item(),
                })
                # Short runs are smoke tests; a run that prints nothing tells us
                # nothing about whether it worked.
                if steps % 10 == 0 or len(loader) < 50:
                    n = running["n"]
                    print(f"  epoch {epoch+1} bước {steps:>5}  "
                          f"loss {running['loss']/n:.4f}  "
                          f"đúng thứ tự {running['acc']/n:.3f}  "
                          f"biên {running['margin']/n:+.3f}", flush=True)
                    running = {"loss": 0.0, "acc": 0.0, "margin": 0.0, "n": 0}

    print("\n  đúng-thứ-tự THEO LOẠI CẶP (soi thiên kiến độ dài ở A>B):")
    for kind, st in sorted(per_type.items()):
        rate = st["correct"] / st["n"] if st["n"] else 0.0
        flag = "  ⚠ A>B thấp — kiểm tra caption có bị co ngắn không" \
            if kind.startswith("A>B") and rate < 0.5 and st["n"] >= 10 else ""
        print(f"    {kind:<44} {st['correct']:>4}/{st['n']:<4} = {rate:.3f}{flag}")

    model.save_pretrained(args.out)
    processor.save_pretrained(args.out)
    metrics_path = Path(args.out) / "metrics.jsonl"
    with metrics_path.open("w", encoding="utf-8") as fh:
        for entry in metrics_log:
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
        fh.write(json.dumps({
            "stage": "dpo", "summary_per_pair_type": {
                k: {"n": v["n"], "ordering_accuracy":
                    (v["correct"] / v["n"] if v["n"] else None)}
                for k, v in per_type.items()},
            "reference_policy": ("pi_sft" if getattr(args, "_has_ref_adapter", False)
                                  else "base_anchored"),
        }, ensure_ascii=False) + "\n")
    print(f"\nđã lưu adapter DPO -> {args.out}")
    print(f"số liệu huấn luyện -> {metrics_path}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["sft", "dpo"], required=True)
    parser.add_argument("--out", default=None)
    parser.add_argument("--adapter", default=None, help="adapter để tiếp tục (DPO bắt buộc)")
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--beta", type=float, default=0.1, help="chỉ dùng cho DPO")
    parser.add_argument("--grad-accum", type=int, default=8)   # 
    parser.add_argument("--lora-r", type=int, default=16)      # 
    parser.add_argument("--lora-alpha", type=int, default=32)  # 
    parser.add_argument(
        "--data", default=str(DATA / "stage2"),
        help=("thư mục chứa sft.jsonl / dpo.jsonl. Trước đây cứng đường dẫn, "
              "nên khi build_dpo_data ghi ra chỗ khác thì bộ huấn luyện lặng lẽ "
              "đọc file CŨ ở mặc định và báo thành công"),
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--vision-tokens", type=int, default=0,
                        help="0 = chọn theo dung lượng card ")
    parser.add_argument("--force-4bit", action="store_true")
    parser.add_argument("--no-4bit", dest="no_4bit", action="store_true")
    parser.add_argument("--use-8bit", dest="use_8bit", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--pair-types", default="",
        help=(
            "chỉ dùng những loại cặp này, phân tách bằng dấu phẩy "
            "(vd 'B>C' để chạy ablation hai bậc so với ba bậc)"
        ),
    )
    args = parser.parse_args()

    args.out = args.out or str(DATA / "runs" / args.stage)
    if args.lr is None:
        # DPO moves an already-trained policy and needs the smaller step.
        args.lr = 1e-4 if args.stage == "sft" else 5e-6

    Path(args.out).mkdir(parents=True, exist_ok=True)
    print(f"giai đoạn {args.stage} · batch=1 × tích luỹ {args.grad_accum} "
          f"· lr={args.lr}")

    return run_sft(args) if args.stage == "sft" else run_dpo(args)


if __name__ == "__main__":
    sys.exit(main())
