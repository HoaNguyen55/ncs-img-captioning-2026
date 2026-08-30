"""Vintern adapter — the Vietnamese-specialised backbone.

Two roles (`research/MODEL-SELECTION.md` §3):

* **`Vintern-1B-v3_5` as the verifier.** Trained on 3M+ Vietnamese image-QA
  pairs, so answering targeted Vietnamese probes is precisely its training task.
  ~2 GB and ~0.2 s per probe, which is what makes ~15 probes per image
  affordable at 500 images × 5 conditions × 3 seeds.
* **`Vintern-3B-R-beta` as an alternative generator.** Vintern-3B-beta reports
  41.289 on MTVQA Vietnamese. Running it against Qwen2.5-VL is the direct test
  of H5: does a Vietnamese-specialised 4B beat a general multilingual 7B?

Vintern is InternVL-derived, so it does **not** use the `transformers`
generation API the way Qwen does. It exposes a custom `.chat()` and expects
images preprocessed with InternVL's dynamic tiling. That difference is the whole
reason `rescap.vlm` exists.

**Caveat carried into the paper.** Vintern's language model comes from the Qwen
family (Qwen2-0.5B for 1B, Qwen2.5-3B for 3B). Generator and verifier therefore
share a language backbone; what differs is the vision encoder (InternViT-300M)
and the training data. Say "different vision encoder and training data", not
"independent models" (`MODEL-SELECTION.md` §3).
"""

from __future__ import annotations

from typing import Any

from .base import VLM, dtype_kwarg

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _build_transform(input_size: int):
    """InternVL's preprocessing. Must match what the model was trained with."""
    import torchvision.transforms as T
    from torchvision.transforms.functional import InterpolationMode

    return T.Compose(
        [
            T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def _closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_diff = float("inf")
    best = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target = ratio[0] / ratio[1]
        diff = abs(aspect_ratio - target)
        if diff < best_diff:
            best_diff, best = diff, ratio
        elif diff == best_diff and area > 0.5 * image_size**2 * ratio[0] * ratio[1]:
            best = ratio
    return best


def _dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=True):
    """InternVL dynamic tiling: split a high-resolution image into tiles.

    Ported from the model card's reference implementation. Reimplementing it
    rather than importing keeps the adapter working when the remote code layout
    changes, and makes the tile budget explicit — `max_num` is the knob that
    trades detail against VRAM.
    """
    width, height = image.size
    aspect_ratio = width / height

    target_ratios = sorted(
        {
            (i, j)
            for n in range(min_num, max_num + 1)
            for i in range(1, n + 1)
            for j in range(1, n + 1)
            if min_num <= i * j <= max_num
        },
        key=lambda x: x[0] * x[1],
    )

    ratio = _closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size)
    target_width, target_height = image_size * ratio[0], image_size * ratio[1]
    blocks = ratio[0] * ratio[1]

    resized = image.resize((target_width, target_height))
    tiles = []
    for index in range(blocks):
        box = (
            (index % (target_width // image_size)) * image_size,
            (index // (target_width // image_size)) * image_size,
            ((index % (target_width // image_size)) + 1) * image_size,
            ((index // (target_width // image_size)) + 1) * image_size,
        )
        tiles.append(resized.crop(box))

    if use_thumbnail and len(tiles) != 1:
        tiles.append(image.resize((image_size, image_size)))
    return tiles


class VinternAdapter(VLM):
    """Vintern (InternVL-derived) through its `.chat()` API."""

    # InternVL's chat() returns text only, with no scores, so confidence has to
    # come from self-consistency (doc 08 §8.2).
    supports_logprobs = False

    def __init__(
        self,
        model_id: str = "5CD-AI/Vintern-1B-v3_5",
        *,
        device: str = "cuda",
        dtype: str = "bfloat16",
        input_size: int = 448,
        max_tiles: int = 6,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.name = model_id
        self.model_id = model_id
        self.device = device
        self.dtype = dtype
        self.input_size = input_size
        self.max_tiles = max_tiles
        self._model = None
        self._tokenizer = None

    def load(self) -> "VinternAdapter":
        if self._model is not None:
            return self

        import torch
        from transformers import AutoModel, AutoTokenizer

        torch_dtype = getattr(torch, self.dtype)
        self._model = (
            AutoModel.from_pretrained(
                self.model_id,
                **dtype_kwarg(torch_dtype),
                low_cpu_mem_usage=True,
                trust_remote_code=True,
            )
            .eval()
            .to(self.device)
        )
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_id, trust_remote_code=True, use_fast=False
        )
        return self

    def unload(self) -> None:
        self._model = None
        self._tokenizer = None
        try:
            import gc

            import torch

            gc.collect()
            torch.cuda.empty_cache()
        except Exception:
            pass

    def _pixel_values(self, image):
        import torch

        transform = _build_transform(self.input_size)
        tiles = _dynamic_preprocess(
            image, image_size=self.input_size, max_num=self.max_tiles, use_thumbnail=True
        )
        pixel_values = torch.stack([transform(tile) for tile in tiles])
        return pixel_values.to(getattr(torch, self.dtype)).to(self.device)

    def _generate(
        self,
        image: Any,
        prompt: str,
        *,
        max_new_tokens: int,
        temperature: float,
    ) -> tuple[str, float | None]:
        self.load()

        pixel_values = self._pixel_values(image)
        sampling = temperature > 0
        config = {
            "max_new_tokens": max_new_tokens,
            "do_sample": sampling,
            "num_beams": 1,
            "repetition_penalty": 1.05,
        }
        if sampling:
            config["temperature"] = temperature
            config["top_p"] = 0.9

        question = prompt if prompt.startswith("<image>") else f"<image>\n{prompt}"
        response = self._model.chat(self._tokenizer, pixel_values, question, config)

        # No scores available from chat() -- returning None routes the caller to
        # self-consistency rather than inventing a confidence.
        return str(response).strip(), None

    def _generate_many(
        self,
        image: Any,
        prompt: str,
        n: int,
        *,
        max_new_tokens: int,
        temperature: float,
    ) -> list[tuple[str, float | None]]:
        """Draw `n` samples in one batched call rather than `n` sequential ones.

        Verification issues on the order of a hundred probes per image and each
        needs k self-consistency samples, so the sequential loop spends most of
        its time on per-call overhead. InternVL exposes `batch_chat`, which
        takes a list of questions plus the tile count per question, so the k
        samples of one probe go through as a single batch.

        The image is tiled **once** and the tiles repeated across the batch --
        `_pixel_values` is the expensive part and it does not depend on which
        sample we are drawing.

        Falls back to the sequential loop if the remote code has no
        `batch_chat`: a slower run is a much better failure than a crash
        halfway through a 36-hour job.
        """
        self.load()
        if n <= 1 or not hasattr(self._model, "batch_chat"):
            return super()._generate_many(
                image, prompt, n, max_new_tokens=max_new_tokens, temperature=temperature
            )

        import torch

        tiles = self._pixel_values(image)
        n_patches = tiles.shape[0]
        batched = torch.cat([tiles] * n, dim=0)

        sampling = temperature > 0
        config = {
            "max_new_tokens": max_new_tokens,
            "do_sample": sampling,
            "num_beams": 1,
            "repetition_penalty": 1.05,
        }
        if sampling:
            config["temperature"] = temperature
            config["top_p"] = 0.9

        question = prompt if prompt.startswith("<image>") else f"<image>\n{prompt}"
        try:
            responses = self._model.batch_chat(
                self._tokenizer,
                batched,
                [question] * n,
                config,
                num_patches_list=[n_patches] * n,
            )
        except Exception:
            # Any batching problem degrades to the loop rather than killing the
            # run. Recorded by the caller through the samples it gets back.
            return super()._generate_many(
                image, prompt, n, max_new_tokens=max_new_tokens, temperature=temperature
            )

        return [(str(r).strip(), None) for r in responses]

    def info(self) -> dict[str, Any]:
        return {
            **super().info(),
            "model_id": self.model_id,
            "dtype": self.dtype,
            "input_size": self.input_size,
            "max_tiles": self.max_tiles,
            "loaded": self._model is not None,
        }
