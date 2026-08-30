"""Deterministic mock backbone — build and test the pipeline without a GPU.

**Why this exists.** The FAIR'2026 schedule has ten days and the pipeline
(docs 08–11) is five days of that. Waiting for GPU hardware before writing any
pipeline code would spend days on nothing. This backend answers the same three
calls as a real one, deterministically, so the whole propose → verify → select →
realise chain can be assembled and debugged on a laptop, and the GPU is then
spent on experiments rather than on discovering that the plumbing is wrong.

**It is not a model.** It answers from a scripted table keyed by the question,
with a hash-based fallback. Anything it produces is fixture data.

Guard rails, because a mock that silently reaches a results table would be the
worst possible outcome:

* `name` is `"mock"` and appears in every `Answer.model`, so it is recorded in
  provenance (`formulation/07 §5`).
* `is_mock = True` — the pipeline refuses to write a results file when any
  component reports this (see `assert_not_mock`).
"""

from __future__ import annotations

import hashlib
import re
from typing import Any

from .base import VLM, ConfidenceMethod


class MockVLM(VLM):
    """Scripted, deterministic, offline. For wiring tests only."""

    name = "mock"
    is_mock = True
    supports_logprobs = True

    #: question substring -> (answer, confidence)
    DEFAULT_SCRIPT: dict[str, tuple[str, float]] = {
        # entity enumeration
        "liệt kê": ("một người đàn ông, một chiếc xe đạp, một chiếc ô tô", 0.90),
        "mô tả cảnh": ("một người đàn ông đang đạp xe trên đường, phía sau có một chiếc ô tô", 0.88),
        "dễ bị bỏ sót": ("có một chiếc mũ trên đầu người đàn ông", 0.55),
        # attributes
        "màu gì": ("màu đỏ", 0.82),
        "mặc gì": ("áo đỏ", 0.85),
        # actions
        "đang làm gì": ("đang đạp xe", 0.87),
        # yes/no — verification probes
        "có phải là màu đỏ": ("có", 0.86),
        "có phải là màu xanh": ("không", 0.79),
        "có một con chó": ("không", 0.91),
        "đang đạp xe": ("có", 0.89),
        "đang đi làm": ("không xác định được", 0.30),
        "ở phía sau": ("có", 0.74),
        "hai người": ("không", 0.83),
    }

    def __init__(
        self,
        *,
        script: dict[str, tuple[str, float]] | None = None,
        default_answer: str = "không xác định được",
        default_confidence: float = 0.25,
        seed: int = 42,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.script = {**self.DEFAULT_SCRIPT, **(script or {})}
        self.default_answer = default_answer
        self.default_confidence = default_confidence
        self.seed = seed
        self.calls: list[dict[str, Any]] = []  # inspectable in tests

    def _lookup(self, prompt: str) -> tuple[str, float]:
        lowered = prompt.lower()
        # Longest matching key wins, so "có phải là màu xanh" beats "màu".
        best: tuple[str, float] | None = None
        best_len = -1
        for key, value in self.script.items():
            if key in lowered and len(key) > best_len:
                best, best_len = value, len(key)
        if best is not None:
            return best

        # Unscripted question: return the default, but with a *deterministic*
        # jitter so repeated runs are identical and so self-consistency at k>1
        # does not trivially agree with itself.
        digest = hashlib.sha256(f"{self.seed}:{prompt}".encode()).digest()
        jitter = digest[0] / 255.0 * 0.2
        return self.default_answer, round(self.default_confidence + jitter, 3)

    def _generate(
        self,
        image: Any,
        prompt: str,
        *,
        max_new_tokens: int,
        temperature: float,
    ) -> tuple[str, float | None]:
        text, confidence = self._lookup(prompt)

        if temperature > 0:
            # Vary deterministically with the call index so self-consistency
            # exercises its aggregation path instead of seeing k copies.
            index = len(self.calls)
            digest = hashlib.sha256(f"{self.seed}:{prompt}:{index}".encode()).digest()
            if digest[0] % 5 == 0:  # ~20% of draws disagree
                text = self.default_answer

        self.calls.append(
            {"prompt": prompt, "answer": text, "temperature": temperature}
        )
        return text, confidence

    def info(self) -> dict[str, Any]:
        return {**super().info(), "is_mock": True, "scripted_keys": len(self.script)}


def assert_not_mock(*components: object) -> None:
    """Raise if any component is a mock.

    Call this before writing anything into `research/results/`. A fixture number
    reaching a results file is the failure this whole module must not cause.

    >>> assert_not_mock(MockVLM())
    Traceback (most recent call last):
    ...
    RuntimeError: refusing to record results produced by a mock backbone: mock
    """
    offenders = [
        getattr(c, "name", type(c).__name__)
        for c in components
        if getattr(c, "is_mock", False)
    ]
    if offenders:
        raise RuntimeError(
            "refusing to record results produced by a mock backbone: "
            + ", ".join(offenders)
        )


_VN_TOKEN = re.compile(r"[^\W\d_]+", re.UNICODE)


def looks_vietnamese(text: str) -> bool:
    """Cheap check that a backbone answered in Vietnamese, not English.

    Used in adapter smoke tests: a multilingual model asked in Vietnamese
    sometimes replies in English, and that must be caught at setup rather than
    discovered in the results (doc 08 §9).
    """
    vietnamese_marks = set("ăâđêôơưàáảãạằắẳẵặầấẩẫậèéẻẽẹềếểễệ"
                           "ìíỉĩịòóỏõọồốổỗộờớởỡợùúủũụừứửữựỳýỷỹỵ")
    return any(char in vietnamese_marks for char in text.lower())
