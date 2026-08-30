"""Qwen2.5-VL adapter — the general multilingual generator.

Role and evidence: `research/MODEL-SELECTION.md` §3. Best open family on
Vietnamese per VMMU, genuinely instruction-tuned (so it can emit the structured
proposition lists doc 08 needs), and 7B fits a 24 GB card at bf16 (~15.2 GB).

Two things this adapter handles that are easy to get wrong:

**Vision-token budget.** Qwen2.5-VL uses dynamic resolution: a large image
becomes a great many vision tokens, and during LoRA SFT that is what blows up
activation memory on a 24 GB card. `max_pixels` caps it, defaulting to the value
`setup_gpu_machine.sh` exports.

**Token probability.** Confidence from generation scores is only meaningful when
the model is not sampling. At `temperature > 0` we return `None` so the caller
falls back to self-consistency rather than trusting a number that means
something different (doc 08 §8.2).
"""

from __future__ import annotations

import os
from typing import Any

from .base import VLM, dtype_kwarg


class QwenVLAdapter(VLM):
    """Qwen2/2.5-VL through `transformers`."""

    supports_logprobs = True

    def __init__(
        self,
        model_id: str = "Qwen/Qwen2.5-VL-7B-Instruct",
        *,
        device: str = "auto",
        dtype: str = "bfloat16",
        max_pixels: int | None = None,
        min_pixels: int | None = None,
        attn_implementation: str | None = None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.name = model_id
        self.model_id = model_id
        self.device = device
        self.dtype = dtype
        # Default matches QWEN_MAX_PIXELS from setup_gpu_machine.sh.
        self.max_pixels = max_pixels or int(
            os.environ.get("QWEN_MAX_PIXELS", 1280 * 28 * 28)
        )
        self.min_pixels = min_pixels or (256 * 28 * 28)
        self.attn_implementation = attn_implementation
        self._model = None
        self._processor = None

    # -- lifecycle ---------------------------------------------------------
    def load(self) -> "QwenVLAdapter":
        if self._model is not None:
            return self

        import torch
        from transformers import AutoProcessor

        try:
            from transformers import Qwen2_5_VLForConditionalGeneration as _Model
        except ImportError:  # older transformers, Qwen2-VL only
            from transformers import Qwen2VLForConditionalGeneration as _Model

        torch_dtype = getattr(torch, self.dtype)

        # flash_attention_2 is a large speedup but is not always installed --
        # setup_gpu_machine.sh treats it as best-effort. Probe rather than
        # assume, so a missing wheel degrades speed instead of failing the run.
        attn = self.attn_implementation
        if attn is None:
            try:
                import flash_attn  # noqa: F401

                attn = "flash_attention_2"
            except Exception:
                attn = "sdpa"

        self._model = _Model.from_pretrained(
            self.model_id,
            **dtype_kwarg(torch_dtype),
            device_map=self.device,
            attn_implementation=attn,
        ).eval()

        self._processor = AutoProcessor.from_pretrained(
            self.model_id, min_pixels=self.min_pixels, max_pixels=self.max_pixels
        )
        self.attn_implementation = attn
        return self

    def unload(self) -> None:
        self._model = None
        self._processor = None
        try:
            import gc

            import torch

            gc.collect()
            torch.cuda.empty_cache()
        except Exception:
            pass

    # -- generation --------------------------------------------------------
    def _generate(
        self,
        image: Any,
        prompt: str,
        *,
        max_new_tokens: int,
        temperature: float,
    ) -> tuple[str, float | None]:
        import torch

        self.load()

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        text = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        try:
            from qwen_vl_utils import process_vision_info

            image_inputs, video_inputs = process_vision_info(messages)
        except ImportError:
            image_inputs, video_inputs = [image], None

        inputs = self._processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self._model.device)

        sampling = temperature > 0
        with torch.no_grad():
            generated = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=sampling,
                temperature=temperature if sampling else None,
                top_p=0.9 if sampling else None,
                output_scores=not sampling,
                return_dict_in_generate=True,
            )

        sequences = generated.sequences
        trimmed = sequences[:, inputs.input_ids.shape[1] :]
        answer = self._processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        # Mean token probability, greedy decoding only. Under sampling this
        # number would not mean what the threshold expects, so return None and
        # let the caller use self-consistency instead.
        probability: float | None = None
        if not sampling and getattr(generated, "scores", None):
            probs = []
            for step, score in enumerate(generated.scores):
                if step >= trimmed.shape[1]:
                    break
                distribution = torch.softmax(score[0].float(), dim=-1)
                probs.append(distribution[trimmed[0, step]].item())
            if probs:
                probability = float(sum(probs) / len(probs))

        return answer.strip(), probability

    def info(self) -> dict[str, Any]:
        return {
            **super().info(),
            "model_id": self.model_id,
            "dtype": self.dtype,
            "max_pixels": self.max_pixels,
            "attn_implementation": self.attn_implementation,
            "loaded": self._model is not None,
        }
