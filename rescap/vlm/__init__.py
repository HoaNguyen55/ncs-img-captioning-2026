"""`rescap.vlm` — backbone adapters behind one interface.

**Reusable layer.** No imports from `rescap.pipeline`, and no torch or
transformers at import time — so this package loads on a machine with no GPU
stack, and `MockVLM` works there.

    from rescap.vlm import get_vlm, build_roles

    models = build_roles({"generator": "qwen2.5-vl-7b", "verifier": "vintern-1b"})
    answer = models["generator"].describe(image, "Mô tả cảnh trong ảnh.")
    check  = models["verifier"].probe_yes_no(image, "Người này có mặc áo đỏ không")

Three calls, deliberately few:

| call | used by | returns |
|---|---|---|
| `describe()` | proposition generation (doc 08) | free-form Vietnamese |
| `probe()` | attribute/action/relation checks | short answer + confidence |
| `probe_yes_no()` | verification (doc 09) | three-valued polarity |

`probe_with_negation()` adds the acquiescence guard: a model that affirms both
a claim and its negation is agreeing with the question rather than reading the
image, and that proposition goes to UNCERTAIN (doc 09 §3.3b).

Confidence carries its own provenance. Backbones that expose token
probabilities report `TOKEN_PROB`; those that do not (Vintern's `chat()`) fall
back to self-consistency over k samples. The two are **not comparable**, so the
method is recorded per answer rather than assumed.
"""

from .base import (
    VLM,
    Answer,
    ConfidenceMethod,
    Polarity,
    YesNo,
    batch_probe,
    parse_yes_no,
)
from .mock import MockVLM, assert_not_mock, looks_vietnamese
from .registry import DEFAULT_ROLES, available, build_roles, get_vlm, register

__all__ = [
    "VLM", "Answer", "YesNo", "Polarity", "ConfidenceMethod",
    "parse_yes_no", "batch_probe",
    "MockVLM", "assert_not_mock", "looks_vietnamese",
    "get_vlm", "build_roles", "available", "register", "DEFAULT_ROLES",
]
