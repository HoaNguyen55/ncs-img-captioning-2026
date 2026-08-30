"""Backbone-agnostic interface for vision-language models.

**Reusable layer.** Nothing here imports `rescap.pipeline`, and nothing here
imports torch or transformers at module level — so `rescap.vlm` can be imported
on a machine with no GPU and no deep-learning stack, and the mock backend works
there too.

Why an adapter layer earns its place: Qwen2.5-VL and Vintern have genuinely
different APIs. Qwen uses `Qwen2_5_VLForConditionalGeneration` with a processor
and `process_vision_info`; Vintern is InternVL-derived and exposes a custom
`.chat()` with its own dynamic image tiling. Writing the pipeline against either
one directly would make the backbone a rewrite instead of a config string — and
`MODEL-SELECTION.md` §4 depends on swapping backbones freely.

The three calls the pipeline actually makes:

    describe()   free-form Vietnamese text            -> proposition generation (doc 08)
    probe()      targeted question, short answer      -> attribute/action/relation checks
    probe_yes_no() polarity + confidence              -> verification (doc 09)
"""

from __future__ import annotations

import abc
import time
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Sequence

# `Image` is only needed for typing; importing PIL lazily keeps this module
# light for callers that just want the types.
try:  # pragma: no cover
    from PIL.Image import Image as PILImage
except Exception:  # pragma: no cover
    PILImage = Any  # type: ignore


class Polarity(str, Enum):
    """What a yes/no probe says about the claim it was asked about."""

    AFFIRMS = "affirms"
    REFUTES = "refutes"
    INCONCLUSIVE = "inconclusive"


class ConfidenceMethod(str, Enum):
    """How a confidence number was obtained.

    Recorded per answer because the two are **not comparable**: mixing
    token-probability confidences from one backbone with self-consistency
    confidences from another, and then thresholding both at the same value,
    would silently mean two different things (doc 08 §8.2).
    """

    TOKEN_PROB = "token_prob"
    SELF_CONSISTENCY = "self_consistency"
    NONE = "none"


@dataclass
class Answer:
    """One model response, with everything needed to audit it later."""

    text: str
    confidence: float = 0.0
    method: ConfidenceMethod = ConfidenceMethod.NONE
    samples: list[str] = field(default_factory=list)
    latency_s: float = 0.0
    model: str = ""
    prompt: str = ""

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"Answer({self.text[:48]!r}, conf={self.confidence:.2f}, "
            f"via={self.method.value})"
        )


@dataclass
class YesNo:
    """A yes/no probe result.

    `polarity` is deliberately three-valued. A model that answers neither yes
    nor no is *inconclusive*, which is a different thing from "no" — the same
    distinction the three-way verdict rests on (doc 09 §1).
    """

    polarity: Polarity
    confidence: float
    answer: Answer

    @property
    def affirms(self) -> bool:
        return self.polarity is Polarity.AFFIRMS


# Vietnamese affirmation / negation surface forms.
_YES = ("có", "đúng", "phải", "vâng", "yes", "true")
_NO = ("không", "sai", "chưa", "no", "false")

# Uncertainty markers — checked BEFORE yes/no, and this ordering is essential.
#
# Most of these BEGIN WITH `không` ("không xác định được", "không rõ", "không
# chắc"), so a naive negation check reads "I cannot tell" as "no". That single
# mistake collapses the three-way verdict into a binary one: the proposition
# goes to REJECTED instead of UNCERTAIN, which is precisely the distinction
# doc 09 §1 exists to preserve. Hedges ("có lẽ", "có thể") are the mirror trap:
# they begin with `có` and would otherwise read as "yes".
_UNCERTAIN = (
    "không xác định", "không rõ", "không chắc", "không biết",
    "không thấy rõ", "không nhìn rõ", "không đủ", "khó nói", "khó xác định",
    "có lẽ", "có thể", "dường như", "hình như", "có vẻ",
    "unclear", "uncertain", "cannot tell", "not sure", "maybe",
)


def parse_yes_no(text: str) -> Polarity:
    """Classify a Vietnamese yes/no answer into three values.

    Order matters: uncertainty is tested first, because most Vietnamese
    uncertainty phrases start with the negator `không` and most hedges start
    with the affirmer `có`.

    >>> parse_yes_no("không")
    <Polarity.REFUTES: 'refutes'>
    >>> parse_yes_no("không xác định được")
    <Polarity.INCONCLUSIVE: 'inconclusive'>
    >>> parse_yes_no("có lẽ vậy")
    <Polarity.INCONCLUSIVE: 'inconclusive'>
    """
    lowered = " ".join(text.strip().lower().split())
    if not lowered:
        return Polarity.INCONCLUSIVE

    # 1. "I cannot tell" is neither yes nor no.
    if any(marker in lowered for marker in _UNCERTAIN):
        return Polarity.INCONCLUSIVE

    # 2. The first token decides; in Vietnamese the polarity word leads.
    first = lowered.split()[0].strip(".,!?:;")
    if first in _NO:
        return Polarity.REFUTES
    if first in _YES:
        return Polarity.AFFIRMS

    # 3. Otherwise fall back to whichever polarity appears, negation first —
    # "có một con chó, không phải mèo" is about what is NOT there.
    if any(token in lowered.split() for token in _NO):
        return Polarity.REFUTES
    if any(token in lowered.split() for token in _YES):
        return Polarity.AFFIRMS
    return Polarity.INCONCLUSIVE


class VLM(abc.ABC):
    """The interface every backbone implements.

    Subclasses override `_generate`. Everything else — self-consistency,
    yes/no parsing, timing — is shared, so the backbones cannot drift apart in
    how confidence is computed.
    """

    name: str = "abstract"
    supports_logprobs: bool = False

    def __init__(self, *, max_new_tokens: int = 256, temperature: float = 0.0):
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

    # -- to implement ------------------------------------------------------
    @abc.abstractmethod
    def _generate(
        self,
        image: PILImage,
        prompt: str,
        *,
        max_new_tokens: int,
        temperature: float,
    ) -> tuple[str, float | None]:
        """Return `(text, token_probability_or_None)`."""

    def _generate_many(
        self,
        image: PILImage,
        prompt: str,
        n: int,
        *,
        max_new_tokens: int,
        temperature: float,
    ) -> list[tuple[str, float | None]]:
        """`n` independent samples of the same prompt.

        The default loops, which is correct everywhere and fast nowhere. A
        backbone whose API can draw the samples in one batched call should
        override this: self-consistency needs k samples per probe and
        verification issues on the order of a hundred probes per image, so the
        loop spends most of its time on per-call overhead rather than on
        computation. Overriding is a pure speed change -- the samples are drawn
        from the same distribution either way.
        """
        return [
            self._generate(
                image, prompt, max_new_tokens=max_new_tokens, temperature=temperature
            )
            for _ in range(n)
        ]

    # -- shared ------------------------------------------------------------
    def describe(
        self,
        image: PILImage,
        prompt: str,
        *,
        max_new_tokens: int | None = None,
        temperature: float | None = None,
    ) -> Answer:
        """Free-form generation. Used by proposition generation (doc 08)."""
        start = time.time()
        text, logprob = self._generate(
            image,
            prompt,
            max_new_tokens=max_new_tokens or self.max_new_tokens,
            temperature=self.temperature if temperature is None else temperature,
        )
        return Answer(
            text=text.strip(),
            confidence=logprob if logprob is not None else 0.0,
            method=ConfidenceMethod.TOKEN_PROB if logprob is not None else ConfidenceMethod.NONE,
            latency_s=round(time.time() - start, 3),
            model=self.name,
            prompt=prompt,
        )

    def probe(
        self,
        image: PILImage,
        question: str,
        *,
        k: int = 1,
        max_new_tokens: int = 32,
        temperature: float = 0.0,
        key: Callable[[str], Any] | None = None,
    ) -> Answer:
        """Targeted short-answer probe.

        With `k > 1` the same question is asked `k` times at `temperature > 0`
        and confidence becomes the agreement fraction — the fallback for
        black-box backbones that expose no token probabilities (doc 08 §8.2).

        `key` says what counts as the same answer across samples. It defaults to
        the normalised surface string, which is right for open-ended generation.
        A caller that only cares about one aspect of the answer should pass a
        function extracting it — `probe_yes_no` passes `parse_yes_no`, because
        two differently-worded affirmations are the same answer to a yes/no
        question and scoring them as disagreement makes a decisive model look
        uncertain.
        """
        key = key or (lambda s: s.lower().strip(".,!?:; "))
        start = time.time()

        if k <= 1:
            text, logprob = self._generate(
                image, question, max_new_tokens=max_new_tokens, temperature=temperature
            )
            if logprob is not None:
                method, confidence = ConfidenceMethod.TOKEN_PROB, logprob
            else:
                method, confidence = ConfidenceMethod.NONE, 0.0
            return Answer(
                text=text.strip(),
                confidence=confidence,
                method=method,
                samples=[text.strip()],
                latency_s=round(time.time() - start, 3),
                model=self.name,
                prompt=question,
            )

        # Self-consistency: sampling needs a non-zero temperature, otherwise
        # every draw is identical and the "agreement" is meaningless.
        sample_temperature = max(temperature, 0.7)
        samples = [
            text.strip()
            for text, _ in self._generate_many(
                image, question, k,
                max_new_tokens=max_new_tokens, temperature=sample_temperature,
            )
        ]
        # Agreement is measured on `key(sample)`, not on the raw string,
        # because what counts as "the same answer" depends on what was asked.
        #
        # For a yes/no probe it is the polarity. Counting exact strings made
        # `Có.` and `Có, trong ảnh có một người đàn ông.` two different
        # answers, so five samples that all said yes in five phrasings scored
        # 1/5 agreement. Verification then read that 0.2 as an uncertain model
        # and nothing could reach SUPPORTED: measured over three real images,
        # 106 of 151 propositions failed for exactly this reason and only 1.9%
        # came back SUPPORTED.
        keyed = [key(s) for s in samples]
        most_common, count = Counter(keyed).most_common(1)[0]
        winner = next(s for s, k in zip(samples, keyed) if k == most_common)

        return Answer(
            text=winner,
            confidence=count / k,
            method=ConfidenceMethod.SELF_CONSISTENCY,
            samples=samples,
            latency_s=round(time.time() - start, 3),
            model=self.name,
            prompt=question,
        )

    def probe_yes_no(
        self,
        image: PILImage,
        question: str,
        *,
        k: int = 1,
        temperature: float = 0.0,
    ) -> YesNo:
        """Yes/no probe. The workhorse of verification (doc 09 §3)."""
        prompt = f"{question.rstrip('?')}? Chỉ trả lời 'có' hoặc 'không'."
        answer = self.probe(
            image, prompt, k=k, max_new_tokens=8, temperature=temperature,
            key=parse_yes_no,
        )
        polarity = parse_yes_no(answer.text)
        confidence = answer.confidence if polarity is not Polarity.INCONCLUSIVE else 0.0
        return YesNo(polarity=polarity, confidence=confidence, answer=answer)

    def probe_with_negation(
        self,
        image: PILImage,
        question: str,
        negated_question: str,
        *,
        k: int = 1,
    ) -> tuple[YesNo, YesNo, bool]:
        """Ask a claim and its negation. Returns `(direct, negated, acquiescent)`.

        A model that affirms both a claim and its negation is exhibiting
        acquiescence bias — it is agreeing with the question rather than reading
        the image. This is the cheapest guard against the "fluent but wrong"
        failure mode, and doc 09 §3.3(b) routes such propositions to UNCERTAIN
        rather than SUPPORTED.
        """
        direct = self.probe_yes_no(image, question, k=k)
        negated = self.probe_yes_no(image, negated_question, k=k)
        acquiescent = direct.affirms and negated.affirms
        return direct, negated, acquiescent

    # -- lifecycle ---------------------------------------------------------
    def load(self) -> "VLM":
        """Load weights. Adapters do this lazily so construction stays cheap."""
        return self

    def unload(self) -> None:
        """Free GPU memory. Needed when swapping backbones on one card."""

    def info(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "supports_logprobs": self.supports_logprobs,
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
        }

    def __repr__(self) -> str:  # pragma: no cover
        return f"{type(self).__name__}({self.name!r})"


def dtype_kwarg(torch_dtype: Any) -> dict[str, Any]:
    """The keyword that asks `from_pretrained` for a weight dtype.

    transformers 5.0 renamed `torch_dtype` to `dtype`. Passing the wrong one is
    not a warning -- the unknown keyword falls through `from_pretrained` into
    the model constructor and dies there:

        TypeError: InternVLChatModel.__init__() got an unexpected keyword
                   argument 'dtype'

    We pin 4.51.3 , so `torch_dtype` is the answer today. This function
    exists anyway because the pin is a floor, not a guarantee: a machine that
    resolves differently should degrade to a slower load, never to a crash on
    the first real model.
    """
    import transformers

    major = int(transformers.__version__.split(".", 1)[0])
    return {"dtype": torch_dtype} if major >= 5 else {"torch_dtype": torch_dtype}


def batch_probe(
    model: VLM,
    items: Sequence[tuple[PILImage, str]],
    *,
    k: int = 1,
) -> list[Answer]:
    """Probe a list of (image, question) pairs.

    Sequential by default. Adapters that can genuinely batch should override
    this; a fake batch that loops is worse than an honest loop, because it
    hides where the time goes.
    """
    return [model.probe(image, question, k=k) for image, question in items]
