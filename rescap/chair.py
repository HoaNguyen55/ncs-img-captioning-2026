"""CHAIR for Vietnamese — object hallucination against caption-derived gold.

    CHAIR_i = hallucinated object MENTIONS / all object mentions
    CHAIR_s = captions with >=1 hallucinated object / all captions

The pair is the standard object-hallucination measure (Rohrbach et al., 2018).
Both are needed and they answer different questions: `CHAIR_s` says how often a
caption is spoiled at all, `CHAIR_i` how much of a caption is wrong. A system
that invents one object in every caption and a system that invents ten in a
tenth of them score the same on one of them and very differently on the other.

**Gold comes from the reference captions**, because KTVIC has no instance
annotations. That is the accepted substitute when a corpus ships captions only,
and it has a consequence that must be stated rather than buried: **an object
that is genuinely in the image but that no annotator mentioned counts as a
hallucination.** So CHAIR here is an UPPER BOUND on hallucination, and the
number is only fair between systems measured the same way -- never against a
CHAIR computed from instance annotations.

That bound bites hardest on exactly the systems this project builds: a detailed
caption names more objects, so it has more chances to name a real one the
annotators skipped. Reporting CHAIR without saying so would make verbosity look
like dishonesty.

**Vietnamese specifics.** Matching goes through `svp.matching.canonical`, so
`xe hơi` and `ô tô` are one object rather than a hallucination, and a leading
classifier is stripped -- `chiếc xe máy` and `xe máy` are the same thing.
Hypernyms resolve one way only: a caption saying `người` when the reference says
`phụ nữ` is correct but less specific, while the reverse invents a detail.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Iterable, Sequence

from .svp.matching import canonical
from .vi.lexicon import COLORS, HYPERNYMS, NOUN_CLASSIFIER, XANH_BLUE, XANH_GREEN

#: Words that are not objects even though they parse as nouns. Counting them
#: would put `màu` or `phía` in the object list and make every caption look
#: hallucinated in the same way.
_NOT_OBJECTS = frozenset({
    "màu", "phía", "bên", "cái", "chiếc", "con", "cảnh", "hình", "ảnh",
    "bức", "tấm", "nơi", "chỗ", "lúc", "khi", "việc", "điều", "thứ",
    "người ta", "gì", "đó", "này", "kia",
})

_TOKEN = re.compile(r"[^\W\d_]+", re.UNICODE)

#: Colour phrases that contain a noun. `xanh lá cây` is "green" and was being
#: read as a TREE; `xanh nước biển` would give a sea. Removed before object
#: extraction, longest first so `xanh lá cây` goes before `xanh lá`.
_COLOUR_PHRASES = tuple(sorted(
    (XANH_BLUE | XANH_GREEN | {c for c in COLORS if " " in c}),
    key=len, reverse=True,
))


def _strip_colour_phrases(text: str) -> str:
    lowered = text.lower()
    for phrase in _COLOUR_PHRASES:
        lowered = lowered.replace(phrase, " ")
    return lowered


def _ancestors(noun: str) -> set[str]:
    """`noun` plus everything it is a kind of, following HYPERNYMS upward."""
    seen = {noun}
    current = noun
    while current in HYPERNYMS:
        current = HYPERNYMS[current]
        if current in seen:
            break
        seen.add(current)
    return seen


@dataclass
class ChairResult:
    n_captions: int = 0
    n_mentions: int = 0
    n_hallucinated_mentions: int = 0
    n_captions_with_hallucination: int = 0
    #: Objects named most often that no reference mentions. The error analysis
    #: section needs the list, not just the rate.
    top_hallucinated: Counter = field(default_factory=Counter)
    per_caption: list[dict] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def chair_i(self) -> float | None:
        return self.n_hallucinated_mentions / self.n_mentions if self.n_mentions else None

    @property
    def chair_s(self) -> float | None:
        if not self.n_captions:
            return None
        return self.n_captions_with_hallucination / self.n_captions

    @property
    def mentions_per_caption(self) -> float | None:
        return self.n_mentions / self.n_captions if self.n_captions else None

    def as_dict(self) -> dict:
        return {
            "CHAIR_s": self.chair_s,
            "CHAIR_i": self.chair_i,
            "n_captions": self.n_captions,
            "n_mentions": self.n_mentions,
            "mentions_per_caption": self.mentions_per_caption,
            "n_hallucinated_mentions": self.n_hallucinated_mentions,
            "n_captions_with_hallucination": self.n_captions_with_hallucination,
            "top_hallucinated": dict(self.top_hallucinated.most_common(20)),
            "notes": self.notes,
        }


def objects_in(text: str) -> list[str]:
    """Object head nouns mentioned in a Vietnamese caption.

    Longest lexicon match wins, so `xe máy` is one object rather than `xe`
    followed by a stray syllable -- the compound-truncation failure that once
    turned `bánh lái` into a cake.

    >>> objects_in("một người đàn ông đứng cạnh chiếc xe máy màu đỏ")
    ['đàn ông', 'xe máy']
    >>> objects_in("một người đang đi bộ")
    ['người']
    >>> objects_in("có hai con chó và một cái bàn")
    ['chó', 'bàn']
    >>> objects_in("chiếc áo màu xanh lá cây")   # a colour, not a tree
    ['áo']
    """
    tokens = [t for t in _TOKEN.findall(_strip_colour_phrases(text))]
    found: list[str] = []
    i = 0
    while i < len(tokens):
        hit = None
        for length in (3, 2, 1):
            if i + length > len(tokens):
                continue
            candidate = " ".join(tokens[i : i + length])
            if candidate in NOUN_CLASSIFIER and candidate not in _NOT_OBJECTS:
                hit = (candidate, length)
                break
        if hit:
            found.append(hit[0])
            i += hit[1]
        else:
            i += 1

    # `người đàn ông` matches as `người` then `đàn ông` and counts as two
    # objects when it names one man. Adjacent pairs where the first is a
    # hypernym of the second collapse onto the specific one, which is also the
    # one a reader would say the caption is about.
    collapsed: list[str] = []
    for noun in found:
        if collapsed and collapsed[-1] in _ancestors(noun) - {noun}:
            collapsed[-1] = noun
        else:
            collapsed.append(noun)
    return collapsed


def _gold_objects(references: Iterable[str]) -> set[str]:
    """Every object any annotator named, plus their hypernyms.

    Hypernyms go in the GOLD set, not the prediction: a caption that says
    `người` where the reference says `phụ nữ` is correct but vaguer, and should
    not be penalised. The reverse -- predicting `phụ nữ` from a reference that
    only says `người` -- invents a detail, and is left to count.
    """
    gold: set[str] = set()
    for reference in references:
        for noun in objects_in(reference):
            gold |= _ancestors(canonical(noun) or noun)
    return gold


def chair(
    predictions: dict[str, str],
    references: dict[str, Sequence[str]],
    *,
    strict_ids: bool = True,
) -> ChairResult:
    """CHAIR_s and CHAIR_i over a set of captions.

    `predictions` is `{image_id: caption}`, `references` `{image_id: [ref, …]}`.
    """
    result = ChairResult()
    result.notes.append(
        "gold lấy từ caption tham chiếu (KTVIC không có nhãn instance) — "
        "vật thể CÓ THẬT trong ảnh mà không người gán nhãn nào nhắc tới sẽ bị "
        "tính là ảo giác, nên đây là CẬN TRÊN"
    )

    missing = [i for i in predictions if i not in references]
    if missing and strict_ids:
        raise ValueError(
            f"{len(missing)} ảnh có dự đoán nhưng không có tham chiếu "
            f"(vd {missing[:3]}) — chấm tiếp sẽ ra một con số cho tập khác"
        )

    for image_id, caption in predictions.items():
        refs = references.get(image_id)
        if not refs:
            continue
        gold = _gold_objects(refs)
        # UNIQUE objects, which is what CHAIR counts. Counting every mention
        # let one caption saying `xe` eight times contribute eight
        # hallucinations, so a repetitive system scored worse than a system
        # that invented eight different things.
        mentioned = list(dict.fromkeys(objects_in(caption)))
        # The noun ITSELF against gold, not its ancestors. Gold already holds
        # the hypernyms, which is what lets a vaguer prediction pass; walking up
        # from the prediction as well makes the test symmetric, and then `phụ
        # nữ` matches a reference that said `đàn ông` because both reach
        # `người`. Predicting a woman where the annotators saw a man is exactly
        # the invented detail this metric exists to count.
        hallucinated = [
            noun for noun in mentioned if (canonical(noun) or noun) not in gold
        ]

        result.n_captions += 1
        result.n_mentions += len(mentioned)
        result.n_hallucinated_mentions += len(hallucinated)
        if hallucinated:
            result.n_captions_with_hallucination += 1
            result.top_hallucinated.update(hallucinated)
        result.per_caption.append({
            "image_id": image_id,
            "n_mentions": len(mentioned),
            "hallucinated": hallucinated,
        })

    return result


#: Person nouns that commit to a gender. Vietnamese puts gender in the NOUN,
#: not in a pronoun, so this is a content claim rather than an agreement
#: feature -- there is no equivalent of writing `they` to stay neutral, you
#: have to choose a different noun.
_GENDERED = {
    "đàn ông": "nam", "nam thanh niên": "nam", "chàng trai": "nam",
    "bé trai": "nam", "cậu bé": "nam", "ông": "nam", "anh": "nam",
    "phụ nữ": "nữ", "cô gái": "nữ", "cô gái trẻ": "nữ", "bé gái": "nữ",
    "cô bé": "nữ", "bà": "nữ", "chị": "nữ", "thiếu nữ": "nữ",
}
#: Person nouns that commit to nothing. This is the safe realisation.
_NEUTRAL_PERSON = {
    "người", "người ta", "cặp đôi", "đôi bạn", "bạn trẻ", "thanh niên",
    "trẻ em", "em bé", "đứa trẻ", "học sinh", "công nhân", "nhóm",
}


@dataclass
class GenderReport:
    n_captions: int = 0
    n_with_person: int = 0
    n_gendered: int = 0            # caption commits to a gender
    n_neutral: int = 0             # caption keeps the neutral noun
    n_invented: int = 0            # gendered where every reference stayed neutral
    n_contradicted: int = 0        # gendered the OTHER way from the references
    examples: list[dict] = field(default_factory=list)

    @property
    def invention_rate(self) -> float | None:
        """Of the captions that commit to a gender, how many had no basis."""
        if not self.n_gendered:
            return None
        return (self.n_invented + self.n_contradicted) / self.n_gendered

    def as_dict(self) -> dict:
        return {
            "n_captions": self.n_captions,
            "n_with_person": self.n_with_person,
            "n_gendered": self.n_gendered,
            "n_neutral": self.n_neutral,
            "n_invented": self.n_invented,
            "n_contradicted": self.n_contradicted,
            "gender_invention_rate": self.invention_rate,
            "examples": self.examples[:10],
        }


def _genders_in(text: str) -> set[str]:
    lowered = " " + " ".join(text.lower().split()) + " "
    return {g for noun, g in _GENDERED.items() if f" {noun} " in lowered}


def _mentions_person(text: str) -> bool:
    lowered = " " + " ".join(text.lower().split()) + " "
    return any(f" {n} " in lowered for n in (*_GENDERED, *_NEUTRAL_PERSON))


def gender_invention(
    predictions: dict[str, str],
    references: dict[str, Sequence[str]],
) -> GenderReport:
    """How often the caption commits to a gender the references do not support.

    **This has no English equivalent.** English carries gender in pronouns, so a
    model can stay neutral by writing `they`. Vietnamese carries it in the noun:
    `người` is neutral, `người phụ nữ` is a claim about the image. A model that
    says `phụ nữ` where the annotators wrote `người` has asserted something it
    cannot see, and no pronoun choice can repair it.

    Three outcomes, and the middle one is the point:

    * **neutral** — the caption said `người`. Safe, and the correct default.
    * **invented** — the caption chose a gender, every reference stayed neutral.
    * **contradicted** — the caption chose the OTHER gender from the references.

    Measured on Qwen zero-shot, `phụ nữ` was the single most frequently
    hallucinated noun in the CHAIR run. That is not the model inventing a
    person; it is the model inventing a *gender* for a person that is really
    there.
    """
    report = GenderReport()
    for image_id, caption in predictions.items():
        refs = references.get(image_id)
        if not refs:
            continue
        report.n_captions += 1
        if not _mentions_person(caption):
            continue
        report.n_with_person += 1

        predicted = _genders_in(caption)
        if not predicted:
            report.n_neutral += 1
            continue

        report.n_gendered += 1
        gold = set()
        for reference in refs:
            gold |= _genders_in(reference)

        if not gold:
            report.n_invented += 1
            kind = "bịa giới tính (tham chiếu trung tính)"
        elif not (predicted & gold):
            report.n_contradicted += 1
            kind = "sai giới tính"
        else:
            continue
        if len(report.examples) < 20:
            report.examples.append({
                "image_id": image_id, "kind": kind,
                "predicted": sorted(predicted), "gold": sorted(gold),
                "caption": caption[:120], "reference": refs[0][:120],
            })
    return report


def supported_propositions_per_image(records: Sequence[dict]) -> dict:
    """Mean SUPPORTED propositions per image, from Stage 1 verdict records.

    The detailed mode's coverage number : a caption is only worth its
    length if the extra length carries checked content.
    """
    counts = []
    for record in records:
        n = 0
        for proposition in record.get("propositions") or []:
            status = (proposition.get("verification") or {}).get("status")
            value = str(getattr(status, "value", status)).rsplit(".", 1)[-1].upper()
            n += value == "SUPPORTED"
        counts.append(n)
    if not counts:
        return {"n_images": 0, "mean_supported": None}
    return {
        "n_images": len(counts),
        "mean_supported": sum(counts) / len(counts),
        "median_supported": sorted(counts)[len(counts) // 2],
        "images_with_none": sum(1 for c in counts if c == 0),
    }
