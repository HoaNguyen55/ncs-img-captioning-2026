"""Vocabulary / tokenizer for the from-scratch captioning baselines.

Baseline 1 (CNN+LSTM) and Baseline 2 (Transformer decoder) train their own
word-level vocabulary rather than reusing a pretrained subword tokenizer.  That
is deliberate: the research ladder starts from the classic Show-and-Tell setup
so later gains from subword/pretrained tokenizers are attributable.

The vocabulary is built from the *training split only* -- building it over the
full dataset leaks validation vocabulary into training and inflates scores.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Iterable, Sequence

PAD, START, END, UNK = "<pad>", "<start>", "<end>", "<unk>"

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")


def basic_tokenize(text: str) -> list[str]:
    """Lowercase + strip punctuation.  Matches the classic captioning setup."""
    return _TOKEN_RE.findall(text.lower())


class Vocabulary:
    """Word-level vocabulary with the four standard special tokens.

    >>> vocab = Vocabulary.build(["a dog runs", "a cat sits"], min_freq=1)
    >>> vocab.encode("a dog", max_len=6)
    [1, 4, 5, 2, 0, 0]
    """

    def __init__(self, itos: Sequence[str]):
        self.itos: list[str] = list(itos)
        self.stoi: dict[str, int] = {tok: i for i, tok in enumerate(self.itos)}

    # -- construction ------------------------------------------------------
    @classmethod
    def build(
        cls,
        captions: Iterable[str],
        min_freq: int = 5,
        max_size: int | None = None,
    ) -> "Vocabulary":
        counter: Counter[str] = Counter()
        for caption in captions:
            counter.update(basic_tokenize(caption))

        words = [w for w, c in counter.most_common() if c >= min_freq]
        if max_size is not None:
            words = words[: max(0, max_size - 4)]

        return cls([PAD, START, END, UNK, *words])

    # -- (de)serialisation -------------------------------------------------
    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"itos": self.itos}, ensure_ascii=False, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "Vocabulary":
        return cls(json.loads(Path(path).read_text())["itos"])

    # -- encoding ----------------------------------------------------------
    def __len__(self) -> int:
        return len(self.itos)

    @property
    def pad_idx(self) -> int:
        return self.stoi[PAD]

    @property
    def start_idx(self) -> int:
        return self.stoi[START]

    @property
    def end_idx(self) -> int:
        return self.stoi[END]

    @property
    def unk_idx(self) -> int:
        return self.stoi[UNK]

    def encode(self, caption: str, max_len: int = 20) -> list[int]:
        """<start> tokens... <end>, padded/truncated to exactly `max_len`."""
        tokens = basic_tokenize(caption)[: max_len - 2]
        ids = [self.start_idx]
        ids += [self.stoi.get(t, self.unk_idx) for t in tokens]
        ids.append(self.end_idx)
        ids += [self.pad_idx] * (max_len - len(ids))
        return ids[:max_len]

    def decode(self, ids: Iterable[int], strip_special: bool = True) -> str:
        words = []
        for idx in ids:
            idx = int(idx)
            if idx >= len(self.itos):
                continue
            token = self.itos[idx]
            if strip_special:
                if token == self.itos[self.end_idx]:
                    break
                if token in (PAD, START):
                    continue
            words.append(token)
        return " ".join(words)

    def coverage(self, captions: Iterable[str]) -> dict[str, float]:
        """Fraction of tokens that map to <unk> -- report this in the paper.

        A high UNK rate on validation means `min_freq` is too aggressive and the
        model is being evaluated on words it structurally cannot produce.
        """
        total = unk = 0
        for caption in captions:
            for token in basic_tokenize(caption):
                total += 1
                if token not in self.stoi:
                    unk += 1
        return {
            "tokens": total,
            "unk": unk,
            "unk_rate": (unk / total) if total else 0.0,
            "vocab_size": len(self),
        }
