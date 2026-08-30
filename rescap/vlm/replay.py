"""Replay the probes a Stage 1 run already paid for.

    from rescap.vlm.replay import ReplayVLM
    model = ReplayVLM(record["propositions"])
    results, stats = verify(model, None, record["entities"], props)

Stage 1 stores every probe it issued -- question, answer, polarity, confidence
-- on `evidence.probes`. So a change to a threshold, a ceiling or the decision
rule can be re-scored against the SAME model answers, on a CPU, in seconds,
instead of re-running 3,769 images across the fleet for six hours.

**This is what makes "measure, don't estimate" affordable.** Two rule changes in
a row were justified with an estimate of their effect: the first said a change
would free 852 propositions, the second said 1,369. The truth was zero, and it
took a GPU round trip to find out each time. With replay the question is
answered before anyone is asked to approve it.

**Fidelity is checked, not assumed.** A rule change can alter the QUESTION a
proposition generates, and a question that was never asked has no recorded
answer. Every miss is counted and `fidelity()` reports the fraction matched.
A replay with misses is not a replay of the same run, and the caller is told so
rather than handed a number that looks clean.
"""

from __future__ import annotations

from typing import Any, Sequence

from .base import Answer, ConfidenceMethod, Polarity, VLM


class ReplayVLM(VLM):
    """A `VLM` that answers from a previous run's records instead of a model."""

    name = "replay"
    supports_logprobs = True  # the recorded probabilities are real ones

    def __init__(
        self,
        propositions: Sequence[dict],
        model_name: str | None = None,
        **kwargs: Any,
    ):
        """`model_name` selects whose answers to replay.

        A run with the colour cross-check issues the SAME question to two
        models, so both answers land under one key and the second overwrites
        the first. Filtering by model keeps them apart, and replaying such a run
        needs one instance per model -- the same shape as the original call,
        which passed a verifier and a `colour_verifier`.

        `None` replays every record, which is right for a single-verifier run
        and wrong the moment a second opinion is involved.
        """
        super().__init__(**kwargs)
        self.model_name = model_name
        self._answers: dict[str, dict] = {}
        for proposition in propositions:
            for record in ((proposition.get("evidence") or {}).get("probes") or []):
                if model_name and str(record.get("model", "")) != model_name:
                    continue
                question = self._key(record.get("question_vi", ""))
                if question:
                    self._answers[question] = record
        self.hits = 0
        self.misses = 0
        self.missed_questions: list[str] = []

    @staticmethod
    def _key(question: str) -> str:
        """Normalise so trailing punctuation and spacing do not cause a miss."""
        return " ".join(question.strip().rstrip("?.").split()).lower()

    def _generate(self, image, prompt, *, max_new_tokens, temperature):
        # Free-form generation cannot be replayed -- the captions were produced
        # by a different model than the one being replayed, and inventing text
        # here would put words in a record's mouth.
        raise NotImplementedError(
            "ReplayVLM only replays yes/no probes; it does not generate free text"
        )

    def probe_yes_no(self, image, question: str, *, k: int = 1, temperature: float = 0.0):
        from .base import YesNo

        record = self._answers.get(self._key(question))
        if record is None:
            self.misses += 1
            if len(self.missed_questions) < 20:
                self.missed_questions.append(question)
            return YesNo(
                polarity=Polarity.INCONCLUSIVE,
                confidence=0.0,
                answer=Answer(text="", confidence=0.0, method=ConfidenceMethod.NONE,
                              model=self.name, prompt=question),
            )

        self.hits += 1
        polarity = Polarity(record.get("polarity", "inconclusive"))
        confidence = float(record.get("answer_prob") or 0.0)
        return YesNo(
            polarity=polarity,
            confidence=confidence,
            answer=Answer(
                text=str(record.get("answer", "")),
                confidence=confidence,
                method=ConfidenceMethod.SELF_CONSISTENCY,
                # The samples matter as much as the verdict: `_answer_stability`
                # derives its own signal from them, and returning an Answer
                # without them made the replay score every probe as perfectly
                # stable. That is what left it reproducing only 74.6% of its own
                # source run -- the replay was more decisive than the thing it
                # was replaying.
                samples=list(record.get("samples") or []),
                model=str(record.get("model", "")),
                prompt=question,
            ),
        )

    def fidelity(self) -> dict[str, Any]:
        """How much of this replay came from the record rather than from a gap."""
        total = self.hits + self.misses
        return {
            "hits": self.hits,
            "misses": self.misses,
            "matched": self.hits / total if total else None,
            "missed_examples": self.missed_questions[:5],
        }
