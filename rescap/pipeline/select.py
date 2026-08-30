"""M5 — proposition selection.

    {SUPPORTED, UNCERTAIN, REJECTED}  ──►  [ Selection ]  ──►  P*

Implements `formulation/10-MODULE-SELECTION.md`: the objective of §3 and the
greedy submodular algorithm of §4.1.

    P* = argmax  α Σ f(p)·u(p)  +  β Cov(P)  +  γ Div(P)  −  δ R(P)
         subject to  |P| ≤ B,  Con(P) = true,  |P ∩ P_UNC| ≤ q

**Three things this module refuses to do, and why each refusal is structural
rather than a matter of tuning:**

1. **REJECTED never enters the objective** (§1). Not a large penalty — absent.
   A penalty is a weight, and any weight can be outvoted by enough coverage.

2. **UNCERTAIN is never factual content** (§1, §6). It is admissible only as
   *hedged* content under the quota q, and the admitted ids are returned
   separately so doc 04's PGF/VCF can exclude them. A hedge is not a grounded
   claim; letting `có vẻ như` count would make hedging a free way to raise the
   metric.

3. **Consistency is a hard constraint, not a term** (§2.7). A caption asserting
   both `áo đỏ` and `áo xanh` is broken at any score.

**Reported, not hidden.** Coverage repair (§4.1 line 22), an unavailable
similarity function, a missing salience field and every other degradation are
recorded as `SelectionEvent`s on the result. Their frequency is a property of
the objective that the paper reports (§4.1's closing note), so the code makes
them countable rather than invisible.

No torch, no transformers, no embedding stack: `similarity_fn` is injected, so
this module — like `svp/matching.py` — imports on a CPU box.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

from ..svp.entailment import entailment_pairs, specificity_level
from ..svp.matching import canonical

# Injected so the module stays free of an embedding dependency (§2.6).
# `sim(p, q) -> [0, 1]` over Vietnamese sentence embeddings of SEGMENTED text.
SimilarityFn = Callable[[str, str], float]

# `surprisal(p) -> -log Pr(p | scene type)` from dev-split corpus statistics
# (§2.2), or None when this proposition was never seen in the statistics.
SurprisalFn = Callable[[dict[str, Any]], float | None]


class Verdict:
    """Verification statuses (`formulation/09`). Three-way, never binary."""

    SUPPORTED = "SUPPORTED"
    UNCERTAIN = "UNCERTAIN"
    REJECTED = "REJECTED"


# Float noise below this is not a score difference. Without the rounding the
# §7 tie-break rule would never fire: two propositions that are genuinely tied
# differ at 1e-17 after a different order of additions, and run-to-run
# reproducibility would depend on dict iteration order.
_TIE_PRECISION = 12
_EPS = 1e-12


def _id_order(pid: str) -> tuple[int, str]:
    """Sort key giving P2 < P10. Ids are `P<n>` (schema §Proposition.id)."""
    matched = re.fullmatch(r"P(\d+)", pid or "")
    return (int(matched.group(1)), pid) if matched else (2**31, pid or "")


def _verdict_of(p: dict[str, Any]) -> str | None:
    """The bare verdict string, however `verification.status` was written.

    `pipeline.verify.Verdict` is a `str, Enum`. Its `to_schema()` writes
    `status.value`, but a caller that assigns `VerificationResult.status`
    straight onto the document leaves an enum **member** here, and
    `str(Verdict.UNCERTAIN)` is `'Verdict.UNCERTAIN'`, not `'UNCERTAIN'`.

    That one difference silently turns every UNCERTAIN proposition into
    assertable factual content: `is_uncertain` goes False, so the hedge quota
    never binds, `uncertain_admitted_ids` comes back empty and doc 11 asserts
    `một người đàn ông` where it owed `có vẻ như`. Equality against the enum
    still passes (it *is* a str), so nothing downstream notices. Normalise the
    value once, here, and let everything else compare plain strings.
    """
    raw = (p.get("verification") or {}).get("status")
    raw = getattr(raw, "value", raw)
    return raw if isinstance(raw, str) else None


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SelectionConfig:
    """The `selection:` block of `formulation/07` §8, plus §2's sub-weights.

    Every field is a **prompted** (hand-tuned on the development split)
    hyper-parameter and every one is reported: §3.0 is explicit that weights
    which are never shown are worse than hand-tuned ones that are.

    `lambda_uncertain`, `eta`, `zeta`, `coverage_threshold` and
    `similarity_threshold` are named in doc 10 (§2.1, §2.2, §2.6, §4.1) but
    given no value there. The defaults below are placeholders to be tuned on
    dev and reported with the results — not measurements.
    """

    enabled: bool = True                    # A3 sets this false
    algorithm: str = "greedy_submodular"
    budget: int = 5                         # B — the detail/risk dial (§3.1)
    hedge_quota: int = 1                    # q — small on purpose (§6)

    alpha: float = 1.0                      # factual information
    beta: float = 0.5                       # coverage
    gamma: float = 0.3                      # diversity
    delta: float = 0.4                      # redundancy; 0 disables it (A6)

    lambda_uncertain: float = 0.2           # λ_U ≪ 1 (§2.1)
    eta: float = 0.5                        # salience weight inside u(p) (§2.2)
    zeta: float = 0.3                       # specificity weight inside u(p)

    coverage_threshold: float = 0.6         # θ_cov, triggers repair (§4.1 l.22)
    similarity_threshold: float = 0.75      # ρ cuts below this (§2.6)
    max_coverage_repairs: int = 1           # §4.1 line 22 is a single `if`

    def __post_init__(self) -> None:
        if self.budget < 0 or self.hedge_quota < 0:
            raise ValueError("budget và hedge_quota không được âm")
        if self.delta < 0:
            raise ValueError("delta không được âm — R là hình phạt, không phải phần thưởng")

    @classmethod
    def from_mapping(cls, cfg: dict[str, Any] | None) -> "SelectionConfig":
        """Build from the nested YAML of `formulation/07` §8.

        Accepts `weights: {alpha, beta, gamma, delta}` as well as flat keys, so
        an ablation can override `selection.weights.delta: 0` (A6) or
        `selection.enabled: false` (A3) without a second schema.
        """
        cfg = dict(cfg or {})
        weights = dict(cfg.pop("weights", None) or {})
        merged = {**weights, **cfg}
        known = {f for f in cls.__dataclass_fields__}
        unknown = sorted(set(merged) - known)
        if unknown:
            # Silently ignoring a misspelt key would report a weight that was
            # never applied — the exact failure §3.0 warns about.
            raise ValueError(f"khoá cấu hình selection không hợp lệ: {unknown}")
        return cls(**{k: v for k, v in merged.items() if k in known})

    def reported_weights(self) -> dict[str, float]:
        """Every numeric hyper-parameter, for the `weights` slot of the schema.

        `weights` is the only `additionalProperties: {type: number}` field in
        `SelectionResult`, so integers such as `hedge_quota` and
        `max_coverage_repairs` ride there too. The alternative is a schema
        version bump plus a migration, which `configs/proposition_schema.json`
        requires and which is not warranted to carry two reported integers.

        `max_coverage_repairs` is here because it **changes P\\***: on a set
        where several repairs are feasible, 1 and 2 select different
        propositions. An unreported hyper-parameter that moves the output is
        exactly what §3.0 forbids — two rows of a table would differ with no
        visible reason.
        """
        return {
            "alpha": self.alpha, "beta": self.beta,
            "gamma": self.gamma, "delta": self.delta,
            "lambda_uncertain": self.lambda_uncertain,
            "eta": self.eta, "zeta": self.zeta,
            "coverage_threshold": self.coverage_threshold,
            "similarity_threshold": self.similarity_threshold,
            "hedge_quota": float(self.hedge_quota),
            "max_coverage_repairs": float(self.max_coverage_repairs),
        }


# ---------------------------------------------------------------------------
# Results and logging
# ---------------------------------------------------------------------------
@dataclass
class SelectionEvent:
    """One logged decision. `kind` is a stable key; `message` is for humans."""

    kind: str
    message: str
    detail: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ObjectiveTerms:
    """F(P) broken into its four terms (§3), so a run can be read term by term.

    `coverage` is None — not 0.0 — when there is no salience mass to divide by.
    An objective missing its β term is a different object from one whose
    coverage happens to be zero, and reporting them as the same number would
    make a β-sweep meaningless.
    """

    information: float
    coverage: float | None
    diversity: float
    redundancy: float
    total: float


@dataclass
class SelectionResult:
    """P* plus everything needed to report and audit it (§4.1 line 28)."""

    selected_ids: list[str]
    hedged_ids: list[str]          # UNCERTAIN admitted as hedged content (§6)
    rejected_ids: list[str]
    terms: ObjectiveTerms
    budget: int
    algorithm: str
    weights: dict[str, float]
    coverage_ceiling: float | None = None
    marginal_gains: dict[str, float] = field(default_factory=dict)
    events: list[SelectionEvent] = field(default_factory=list)
    flags: list[str] = field(default_factory=list)

    @property
    def objective_value(self) -> float:
        return self.terms.total

    @property
    def factual_ids(self) -> list[str]:
        """Selected ids that may be asserted, i.e. **excluding hedges**.

        This is the set doc 04's PGF/VCF scores over. `selected_ids` is not:
        it contains the hedged propositions, and counting a hedge as a grounded
        claim is the metric inflation §6 forbids.
        """
        hedged = set(self.hedged_ids)
        return [pid for pid in self.selected_ids if pid not in hedged]

    def is_hedged(self, proposition_id: str) -> bool:
        """Does this proposition need `có vẻ như` / `dường như` at realisation?

        The mark lives here rather than on the proposition because
        `Proposition` is `additionalProperties: false`; doc 11 reads it to set
        `span_role: hedge` and `is_factual: false` on the caption span.
        """
        return proposition_id in self.hedged_ids

    def log(self, kind: str, message: str, **detail: Any) -> None:
        self.events.append(SelectionEvent(kind, message, detail))

    def count(self, kind: str) -> int:
        """How often an event fired — e.g. `count("coverage_repair")` (§4.1)."""
        return sum(1 for event in self.events if event.kind == kind)

    def to_schema(self) -> dict[str, Any]:
        """The `SelectionResult` object of `configs/proposition_schema.json`.

        Only the seven schema properties: the definition is
        `additionalProperties: false`, so events, flags and the term breakdown
        stay on this dataclass and go into the run record instead.

        One policy governs `algorithm` and `budget` alike: **the document states
        what happened, not what was configured.** Under A3 neither ran, and both
        properties are optional, so both are omitted.

        - `algorithm`: `formulation/07` line 31 writes `"none"`, but the schema
          enum is `greedy_submodular | ilp | mmr | graph_dominating_set |
          topk_ranking`, so `"none"` is not writable and inventing one of the
          five would be worse than silence.
        - `budget`: the schema calls it "max propositions allowed into the
          caption". A3 allows all of them, so emitting `budget: 5` beside seven
          selected ids would assert a constraint that never applied — and a
          B-sweep read off these documents would silently gain a flat row.

        The configured values are not lost: they stay on this dataclass and in
        the run's `provenance.config_hash`.
        """
        document: dict[str, Any] = {
            "selected_ids": list(self.selected_ids),
            "objective_value": self.terms.total,
            "weights": dict(self.weights),
            "rejected_ids": list(self.rejected_ids),
            "uncertain_admitted_ids": list(self.hedged_ids),
        }
        if self.algorithm != "none":
            document["algorithm"] = self.algorithm
            document["budget"] = self.budget
        return document


# ---------------------------------------------------------------------------
# The seven criteria (§2)
# ---------------------------------------------------------------------------
def factuality(p: dict[str, Any], *, lambda_uncertain: float) -> float | None:
    """f(p) = S(p|I)·(1 − C(p|I))·λ_verdict  (§2.1).

    Returns **None** when either score is missing, and the caller drops the
    α-term for that proposition rather than substituting a value. Defaulting
    the missing contradiction score to 0.0 would assert "no evidence against
    this" — the inflating direction — and defaulting support to 1.0 would let
    an unscored proposition outrank a verified one. Under Ablation 1
    (`formulation/07` line 22) every status is set to SUPPORTED with no scores,
    so this is the normal path there, and it is logged.
    """
    verification = p.get("verification") or {}
    status = _verdict_of(p)
    if status == Verdict.SUPPORTED:
        weight = 1.0
    elif status == Verdict.UNCERTAIN:
        weight = lambda_uncertain
    else:
        return None  # REJECTED never reaches here; unverified has no f

    support = verification.get("support_score")
    contradiction = verification.get("contradiction_score")
    if support is None or contradiction is None:
        return None
    return float(support) * (1.0 - float(contradiction)) * weight


def informativeness(
    p: dict[str, Any],
    *,
    salience: float,
    eta: float,
    zeta: float,
    surprisal: float | None,
) -> float:
    """u(p) = surprisal + η·sal(subject(p)) + ζ·σ(p)  (§2.2).

    A precomputed `informativeness` field wins, since it is what the schema
    reserves for this value. Otherwise the terms are summed from what exists:
    with no `surprisal_fn` the first term is absent and u is a **lower bound**,
    which the caller flags so no table claims a corpus-estimated u that was
    never computed. `có một người` in a photo of a person stays near-contentless
    either way, because both remaining terms are small for it.
    """
    declared = p.get("informativeness")
    if isinstance(declared, (int, float)) and not isinstance(declared, bool):
        return float(declared)
    value = eta * salience + zeta * specificity_level(p)
    if surprisal is not None:
        value += surprisal
    return value


def coverage(mentioned: set[str], salience: dict[str, float]) -> float | None:
    """Cov(P) = Σ sal(mentioned) / Σ sal(all)  (§2.4).

    None when the denominator is zero: with no entity registry there is nothing
    to cover, and 0.0 would say the opposite — that everything was missed.
    """
    total = sum(salience.values())
    if total <= 0.0:
        return None
    return sum(salience.get(e, 0.0) for e in mentioned) / total


def diversity(types: Sequence[str]) -> float:
    """Div(P) = Shannon entropy of the proposition-type distribution (§2.5).

    In **bits**. The base is part of γ's calibration, so changing it silently
    would rescale one term of the objective against the other three.

    Note for §3.2: entropy over the *empirical* distribution is not monotone —
    adding a second `attribute` to a set of one `attribute` and one `action`
    lowers H — so the (1 − 1/e) guarantee does not follow from this term as
    written. §3.2 already says the property is to be verified empirically
    against an ILP, not asserted; this is one of the reasons it must be.
    """
    if not types:
        return 0.0
    counts: dict[str, int] = {}
    for t in types:
        counts[t] = counts.get(t, 0) + 1
    n = len(types)
    return -sum((c / n) * math.log2(c / n) for c in counts.values())


# ---------------------------------------------------------------------------
# Internal per-proposition state
# ---------------------------------------------------------------------------
@dataclass
class _Candidate:
    prop: dict[str, Any]
    pid: str
    status: str
    factuality: float | None
    informativeness: float
    ptype: str
    entity_ids: frozenset[str]
    order: tuple[int, str]

    @property
    def fu(self) -> float:
        """f(p)·u(p), or 0 when f could not be computed — see `factuality`."""
        return 0.0 if self.factuality is None else self.factuality * self.informativeness

    @property
    def is_uncertain(self) -> bool:
        return self.status == Verdict.UNCERTAIN


def _mentioned_entities(p: dict[str, Any]) -> frozenset[str]:
    """Entity ids a proposition mentions, in either argument slot.

    Coverage counts the object slot too: `ô tô ở phía sau người đàn ông`
    genuinely puts the car in the caption, and ignoring that would send the
    repair after an entity that is already there.
    """
    ids = set()
    for slot in ("subject", "object"):
        argument = p.get(slot)
        if isinstance(argument, dict) and argument.get("entity_id"):
            ids.add(str(argument["entity_id"]))
    return frozenset(ids)


def _similarity_text(p: dict[str, Any]) -> tuple[str, bool]:
    """Text to hand the similarity function, and whether it was segmented.

    §2.6: similarity must run over **segmented** text — unsegmented Vietnamese
    similarity is unreliable because `người đàn ông` looks like three unrelated
    tokens. Falling back to `text_vi` is allowed but flagged, never silent.
    """
    segmented = ((p.get("normalization") or {}).get("word_segmented") or "").strip()
    if segmented:
        return segmented, True
    return canonical(p.get("text_vi")) or str(p.get("id") or ""), False


# ---------------------------------------------------------------------------
# The selector
# ---------------------------------------------------------------------------
class _Selector:
    """Greedy submodular selection (§4.1). One instance per image."""

    def __init__(
        self,
        propositions: Sequence[dict[str, Any]],
        entities: Sequence[dict[str, Any]],
        config: SelectionConfig,
        similarity_fn: SimilarityFn | None,
        surprisal_fn: SurprisalFn | None,
    ) -> None:
        self.config = config
        self.result = SelectionResult(
            selected_ids=[],
            hedged_ids=[],
            rejected_ids=[],
            terms=ObjectiveTerms(0.0, None, 0.0, 0.0, 0.0),
            budget=config.budget,
            algorithm=config.algorithm if config.enabled else "none",
            weights=config.reported_weights(),
        )

        self.salience = self._build_salience(entities)
        self.candidates = self._build_candidates(propositions, surprisal_fn)
        # `_build_contradictions` may drop self-contradicting candidates (§7),
        # so the id index is built after it, never before.
        self.contradictions = self._build_contradictions()
        self.by_id = {c.pid: c for c in self.candidates}
        self.redundancy_pairs = self._build_redundancy(similarity_fn)

        mentionable = set().union(*(c.entity_ids for c in self.candidates)) if self.candidates else set()
        self.result.coverage_ceiling = coverage(mentionable, self.salience)

    # -- setup -------------------------------------------------------------
    def _build_salience(self, entities: Sequence[dict[str, Any]]) -> dict[str, float]:
        """sal(e) per entity. Uniform fallback when the registry carries none.

        A registry with no salience anywhere means nothing is known about
        relative importance, and uniform weights say exactly that; inventing
        per-entity numbers would let coverage repair chase a fabricated main
        subject. An entity missing the field in a registry that otherwise has
        it scores 0 — we have no evidence it is salient — and both cases are
        logged.
        """
        values = {
            str(e["id"]): e["salience"]
            for e in entities
            if e.get("id") is not None and isinstance(e.get("salience"), (int, float))
        }
        if not values and entities:
            self.result.flags.append("salience_unavailable")
            self.result.log(
                "salience_unavailable",
                "không có trường salience — dùng trọng số đều cho mọi đối tượng",
                n_entities=len(entities),
            )
            return {str(e["id"]): 1.0 for e in entities if e.get("id") is not None}

        missing = [str(e["id"]) for e in entities if e.get("id") is not None and str(e["id"]) not in values]
        if missing:
            self.result.log(
                "salience_missing",
                f"{len(missing)} đối tượng không có salience — tính là 0 trong độ phủ",
                entity_ids=missing,
            )
        return {k: float(v) for k, v in values.items()}

    def _build_candidates(
        self,
        propositions: Sequence[dict[str, Any]],
        surprisal_fn: SurprisalFn | None,
    ) -> list[_Candidate]:
        """P_elig (§4.1 lines 1-3). REJECTED is filtered here, never scored."""
        candidates: list[_Candidate] = []
        unverified: list[str] = []
        unscored: list[str] = []
        no_surprisal = 0

        for p in propositions:
            pid = str(p.get("id"))
            status = _verdict_of(p)

            if status == Verdict.REJECTED:
                self.result.rejected_ids.append(pid)
                continue
            if status not in (Verdict.SUPPORTED, Verdict.UNCERTAIN):
                # Admitting an unverified proposition would bypass M4 entirely,
                # the same invariant `generate.assert_clean` protects from the
                # other side. Excluded and counted, never quietly assumed true.
                unverified.append(pid)
                continue

            entity_ids = _mentioned_entities(p)
            subject = p.get("subject") if isinstance(p.get("subject"), dict) else {}
            subject_salience = self.salience.get(str(subject.get("entity_id") or ""), 0.0)

            surprisal = None
            if surprisal_fn is not None:
                surprisal = surprisal_fn(p)
                if surprisal is None:
                    no_surprisal += 1

            f = factuality(p, lambda_uncertain=self.config.lambda_uncertain)
            if f is None:
                unscored.append(pid)

            candidates.append(
                _Candidate(
                    prop=p,
                    pid=pid,
                    status=status,
                    factuality=f,
                    informativeness=informativeness(
                        p,
                        salience=subject_salience,
                        eta=self.config.eta,
                        zeta=self.config.zeta,
                        surprisal=surprisal,
                    ),
                    ptype=str(p.get("type") or "unknown"),
                    entity_ids=entity_ids,
                    order=_id_order(pid),
                )
            )

        if unverified:
            self.result.flags.append("unverified_propositions")
            self.result.log(
                "unverified_propositions",
                f"{len(unverified)} mệnh đề chưa được kiểm chứng — loại khỏi lựa chọn",
                proposition_ids=unverified,
            )
        if unscored:
            self.result.flags.append("factuality_unavailable")
            self.result.log(
                "factuality_unavailable",
                f"{len(unscored)} mệnh đề thiếu điểm kiểm chứng — số hạng α bằng 0, "
                "KHÔNG suy đoán giá trị",
                proposition_ids=unscored,
            )
        if surprisal_fn is None:
            self.result.flags.append("surprisal_unavailable")
            self.result.log(
                "surprisal_unavailable",
                "không có thống kê ngữ liệu — u(p) thiếu số hạng bất ngờ, là cận dưới",
            )
        elif no_surprisal:
            self.result.log(
                "surprisal_missing",
                f"{no_surprisal} mệnh đề không có trong thống kê ngữ liệu",
            )

        candidates.sort(key=lambda c: c.order)
        return candidates

    def _build_contradictions(self) -> dict[str, set[str]]:
        """Symmetric contradiction map (§2.7), plus the §7 cycle repair.

        Symmetrised because `contradicts` is populated per proposition and one
        side may list the pair while the other does not; an asymmetric map
        would make the hard constraint depend on selection order.
        """
        graph: dict[str, set[str]] = {c.pid: set() for c in self.candidates}
        self_contradicting: list[str] = []

        for c in self.candidates:
            for other in c.prop.get("contradicts") or []:
                other = str(other)
                if other == c.pid:
                    self_contradicting.append(c.pid)
                    continue
                if other in graph:
                    graph[c.pid].add(other)
                    graph[other].add(c.pid)

        # §7: a contradiction that survived verification. Only a self-loop makes
        # a proposition individually infeasible — a pairwise or cyclic conflict
        # between distinct propositions is handled by the hard constraint, which
        # simply never puts two of them in P* together.
        if self_contradicting:
            drop = set(self_contradicting)
            lowest = min(
                (c for c in self.candidates if c.pid in drop),
                key=lambda c: ((c.factuality if c.factuality is not None else -1.0), -c.order[0]),
            )
            self.result.log(
                "contradiction_cycle",
                "mệnh đề tự mâu thuẫn — loại bỏ, f thấp nhất được ghi lại",
                proposition_ids=sorted(drop, key=_id_order),
                lowest_factuality_id=lowest.pid,
            )
            self.candidates = [c for c in self.candidates if c.pid not in drop]
            graph = {pid: {o for o in others if o not in drop} for pid, others in graph.items() if pid not in drop}
        return graph

    def _build_redundancy(self, similarity_fn: SimilarityFn | None) -> dict[tuple[str, str], float]:
        """R's pairwise table (§2.6), computed once over UNORDERED pairs.

        `δ = 0` skips the whole table: Ablation 6 must not pay for a term it
        does not use, and it must not call `similarity_fn` either — an ablation
        that still runs the embedding model is not the ablation it claims.

        The entailment pairs are logged **either way**. `entailment_pairs`
        touches no model (only `similarity_fn` does), and §2.6's predicted A6
        effect is "repetitive captions" — which is measured by counting entailed
        pairs left inside P*. Suppressing the log under A6 would delete the
        ablation's own read-out and make its row incomparable with the baseline.

        Ordered pairs would count each conflict twice; §2.6's sum is written
        over `p ≠ q` but the redundancy of a pair is one fact about it.
        """
        entailed = entailment_pairs([c.prop for c in self.candidates])
        for (specific, general), reason in sorted(entailed.items()):
            self.result.log(
                "entailment",
                f"{specific} ⊨ {general} ({reason})",
                specific=specific, general=general, reason=reason,
            )

        if self.config.delta == 0.0:
            self.result.log("redundancy_disabled", "delta = 0 — bỏ hoàn toàn số hạng R (A6)")
            return {}

        # Per PROPOSITION, not per pair: counting inside the O(n²) loop reported
        # 42 unsegmented texts for 7 propositions, and that number is meant to
        # tell a reader how much of the corpus lacked segmentation.
        texts = {c.pid: _similarity_text(c.prop) for c in self.candidates}
        unsegmented = sorted((pid for pid, (_, ok) in texts.items() if not ok), key=_id_order)

        table: dict[tuple[str, str], float] = {}
        for i, a in enumerate(self.candidates):
            for b in self.candidates[i + 1 :]:
                score = 1.0 if ((a.pid, b.pid) in entailed or (b.pid, a.pid) in entailed) else 0.0
                if similarity_fn is not None:
                    similarity = float(similarity_fn(texts[a.pid][0], texts[b.pid][0]))
                    # ρ: overlap below the threshold is not redundancy, it is
                    # two different claims about the same scene.
                    if similarity >= self.config.similarity_threshold:
                        score += similarity
                if score:
                    table[(a.pid, b.pid)] = score

        if similarity_fn is None:
            self.result.flags.append("similarity_unavailable")
            self.result.log(
                "similarity_unavailable",
                "không có hàm tương đồng — R chỉ dựa trên kéo theo, chồng lấp ngữ nghĩa bị bỏ",
            )
        elif unsegmented:
            self.result.flags.append("unsegmented_similarity")
            self.result.log(
                "unsegmented_similarity",
                "so sánh trên văn bản chưa tách từ — độ tương đồng tiếng Việt kém tin cậy",
                n_texts=len(unsegmented),
                proposition_ids=unsegmented,
            )
        return table

    # -- objective ---------------------------------------------------------
    def objective(self, ids: Sequence[str]) -> ObjectiveTerms:
        """F(P) — §3. Recomputed from scratch; |P| ≤ B ≈ 5, so this is cheap.

        Lazy (accelerated) greedy is the §4 optimisation and is deliberately
        not taken: it only pays off at large n, and a stale cached gain is the
        kind of bug that silently changes P* rather than crashing.
        """
        chosen = [self.by_id[pid] for pid in ids]
        information = sum(c.fu for c in chosen)
        cover = coverage(set().union(*(c.entity_ids for c in chosen)) if chosen else set(), self.salience)
        spread = diversity([c.ptype for c in chosen])

        redundancy = 0.0
        if self.config.delta and self.redundancy_pairs:
            for i, a in enumerate(ids):
                for b in ids[i + 1 :]:
                    key = (a, b) if _id_order(a) < _id_order(b) else (b, a)
                    redundancy += self.redundancy_pairs.get(key, 0.0)

        total = (
            self.config.alpha * information
            + self.config.beta * (cover if cover is not None else 0.0)
            + self.config.gamma * spread
            - self.config.delta * redundancy
        )
        return ObjectiveTerms(information, cover, spread, redundancy, total)

    def _coverage_of(self, ids: Sequence[str]) -> float | None:
        """Cov(P) for a candidate set, without building the other three terms."""
        mentioned = set().union(*(self.by_id[p].entity_ids for p in ids)) if ids else set()
        return coverage(mentioned, self.salience)

    def _consistent(self, ids: Sequence[str]) -> bool:
        for i, a in enumerate(ids):
            for b in ids[i + 1 :]:
                if b in self.contradictions.get(a, ()):
                    return False
        return True

    def _admissible(self, candidate: _Candidate, selected: list[str], n_hedge: int) -> bool:
        """Lines 13-14: the hard constraint and the hedge quota."""
        if any(pid in self.contradictions.get(candidate.pid, ()) for pid in selected):
            return False
        if candidate.is_uncertain and n_hedge >= self.config.hedge_quota:
            return False
        return True

    # -- greedy ------------------------------------------------------------
    def run(self) -> SelectionResult:
        selected: list[str] = []
        n_hedge = 0
        current = self.objective(selected)

        while len(selected) < self.config.budget:
            best: _Candidate | None = None
            best_key: tuple[float, float, float] | None = None
            best_gain = 0.0

            # Ascending id, with strict `>` below, so ties resolve to the
            # lowest id without a second sort (§7: deterministic, reproducible).
            for candidate in self.candidates:
                if candidate.pid in selected:
                    continue
                if not self._admissible(candidate, selected, n_hedge):
                    continue
                gain = self.objective(selected + [candidate.pid]).total - current.total
                if gain <= _EPS:
                    # Skipped, not stopped: entropy makes gains non-monotone, so
                    # a later candidate may still gain where this one does not.
                    continue
                key = (
                    round(gain, _TIE_PRECISION),
                    round(candidate.factuality or 0.0, _TIE_PRECISION),
                    round(candidate.informativeness, _TIE_PRECISION),
                )
                if best_key is None or key > best_key:
                    best, best_key, best_gain = candidate, key, gain

            if best is None:
                break  # line 17 — no positive gain

            selected.append(best.pid)
            self.result.marginal_gains[best.pid] = best_gain
            current = self.objective(selected)
            if best.is_uncertain:
                n_hedge += 1

        selected, _ = self._repair_coverage(selected, n_hedge)

        # AFTER the repair, which can fill the last slot when greedy stalled a
        # proposition short of B. Judging the flag on greedy's output alone put
        # `budget_not_binding` on a run whose budget was full — and that flag is
        # exactly what tells a reader whether a flat stretch of a B-sweep is a
        # finding or an artefact (§7).
        if len(selected) < self.config.budget:
            # The reason is recorded because "nothing left to pick" and "nothing
            # left worth picking" are different findings about the objective.
            self.result.flags.append("budget_not_binding")
            self.result.log(
                "budget_not_binding",
                f"chọn {len(selected)}/{self.config.budget} — ngân sách không ràng buộc",
                n_eligible=len(self.candidates),
                reason=(
                    "đã chọn hết mệnh đề đủ điều kiện"
                    if len(selected) == len(self.candidates)
                    else "không còn mệnh đề nào vừa hợp lệ vừa làm tăng mục tiêu"
                ),
            )

        return self.finalise(selected)

    # -- §4.1 line 22 ------------------------------------------------------
    def _repair_coverage(self, selected: list[str], n_hedge: int) -> tuple[list[str], int]:
        """Forced coverage repair, and it is LOGGED.

        Greedy can leave a salient entity unmentioned when several high-u
        propositions cluster on one subject. How often this fires is a
        reportable property of the objective — if it fires constantly, β is
        wrong and the repair is masking it — so it is an event, not a fix.
        """
        for _ in range(self.config.max_coverage_repairs):
            current = self.objective(selected)
            if current.coverage is None or current.coverage >= self.config.coverage_threshold:
                break

            mentioned = set().union(*(self.by_id[p].entity_ids for p in selected)) if selected else set()
            unmentioned = sorted(
                (e for e in self.salience if e not in mentioned and self.salience[e] > 0),
                key=lambda e: (-self.salience[e], e),
            )

            repaired = False
            for entity in unmentioned:
                replacement, victim = self._repair_pair(entity, selected, n_hedge)
                if replacement is None:
                    continue
                after = [p for p in selected if p != victim] + [replacement.pid]
                after.sort(key=_id_order)
                self.result.log(
                    "coverage_repair",
                    f"buộc phủ đối tượng {entity} — thêm {replacement.pid}"
                    + (f", bỏ {victim}" if victim else ""),
                    entity_id=entity,
                    added_id=replacement.pid,
                    removed_id=victim,
                    coverage_before=current.coverage,
                    objective_before=current.total,
                    objective_after=self.objective(after).total,
                )
                if victim is not None and self.by_id[victim].is_uncertain:
                    n_hedge -= 1
                if replacement.is_uncertain:
                    n_hedge += 1
                # Keep the reported gains describing the set that shipped: the
                # evicted proposition is no longer in P*, and the one forced in
                # had no greedy gain recorded at all.
                self.result.marginal_gains.pop(victim, None)
                self.result.marginal_gains[replacement.pid] = (
                    self.objective(after).total
                    - self.objective([p for p in after if p != replacement.pid]).total
                )
                selected = after
                repaired = True
                break

            if not repaired:
                break

        final = self.objective(selected)
        if final.coverage is not None and final.coverage < self.config.coverage_threshold:
            # §7: accept and log. Never fabricate a proposition to fill coverage
            # — an invented claim is a hallucination whatever the metric says.
            self.result.flags.append("coverage_below_threshold")
            self.result.log(
                "coverage_below_threshold",
                f"độ phủ {final.coverage:.3f} < θ={self.config.coverage_threshold} sau sửa chữa "
                "— chấp nhận, KHÔNG bịa mệnh đề",
                coverage=final.coverage,
                coverage_ceiling=self.result.coverage_ceiling,
            )
        return selected, n_hedge

    def _repair_pair(
        self, entity: str, selected: list[str], n_hedge: int
    ) -> tuple[_Candidate | None, str | None]:
        """`(p_e, victim)` for the repair, or `(None, None)` if none is feasible.

        The victim is the **current** leave-one-out loss argmin, recomputed
        here rather than reusing the marginal gain recorded during greedy: with
        an entropy diversity term those gains are stale, since each was
        measured against the set as it stood at that step, not against the
        final set. Reusing them would drop the wrong proposition.

        A swap is only a repair if coverage actually rises. The cheapest victim
        can itself be the sole mention of some other entity, and trading one
        unmentioned entity for another pays the objective cost of a repair for
        no coverage at all — with `max_coverage_repairs > 1` it also cycles,
        swapping the same propositions back and forth and reporting each lap as
        a `coverage_repair`. Strict improvement is the guard.
        """
        pool = [
            c for c in self.candidates
            if entity in c.entity_ids and c.pid not in selected
        ]
        if not pool:
            return None, None
        # Highest-scoring proposition about e (§4.1 line 23), same tie-break.
        pool.sort(key=lambda c: (-round(c.fu, _TIE_PRECISION),
                                 -round(c.factuality or 0.0, _TIE_PRECISION),
                                 c.order))

        room = len(selected) < self.config.budget
        victims: list[str | None]
        if room:
            # Under budget: adding costs nothing, so there is no victim. Still
            # a repair, still logged, with `removed_id: null`.
            victims = [None]
        else:
            base = self.objective(selected).total
            victims = sorted(
                selected,
                key=lambda pid: (
                    round(base - self.objective([q for q in selected if q != pid]).total,
                          _TIE_PRECISION),
                    _id_order(pid),
                ),
            )

        before = self._coverage_of(selected) or 0.0
        for replacement in pool:
            if replacement.is_uncertain and room and n_hedge >= self.config.hedge_quota:
                continue
            for victim in victims:
                after = [p for p in selected if p != victim] + [replacement.pid]
                if not self._consistent(after):
                    continue
                hedges = sum(1 for pid in after if self.by_id[pid].is_uncertain)
                if hedges > self.config.hedge_quota:
                    continue
                if (self._coverage_of(after) or 0.0) <= before + _EPS:
                    continue  # not a repair — see the docstring
                return replacement, victim
        return None, None

    # -- close out ---------------------------------------------------------
    def finalise(self, selected: list[str]) -> SelectionResult:
        selected = sorted(selected, key=_id_order)
        self.result.selected_ids = selected
        self.result.hedged_ids = [pid for pid in selected if self.by_id[pid].is_uncertain]
        self.result.terms = self.objective(selected)

        if self.result.terms.coverage is None:
            self.result.flags.append("coverage_unavailable")
            self.result.log(
                "coverage_unavailable",
                "không có khối lượng salience — bỏ số hạng β khỏi mục tiêu",
            )
        if not any(c.status == Verdict.SUPPORTED for c in self.candidates):
            # §7 row 1. The fallback branch is recorded because the two are
            # different outputs: a minimal hedged caption still says something
            # about the image, an existence-only one barely does, and doc 11
            # picks between them from here.
            # `fallback` names what P* ACTUALLY carries, not what was eligible.
            # An UNCERTAIN proposition that q kept out is not hedged content:
            # under q = 0 (§6, "UNCERTAIN content is entirely dropped") the
            # eligible-set reading told doc 11 to emit a hedged caption from an
            # empty hedge set. The eligible count is reported alongside so the
            # "we had detail but the quota refused it" case stays visible.
            n_uncertain = sum(1 for c in self.candidates if c.is_uncertain)
            hedgeable = bool(self.result.hedged_ids)
            self.result.flags.append("no_supported_content")
            self.result.log(
                "no_supported_content",
                "không có mệnh đề nào được xác nhận — "
                + ("chỉ còn nội dung rào đón" if hedgeable else "chỉ còn chú thích tồn tại"),
                fallback="hedged" if hedgeable else "existence_only",
                n_uncertain_eligible=n_uncertain,
                n_hedged_admitted=len(self.result.hedged_ids),
            )
        _assert_invariants(self.result, self.by_id, self.contradictions, self.config)
        return self.result


def _assert_invariants(
    result: SelectionResult,
    by_id: dict[str, _Candidate],
    contradictions: dict[str, set[str]],
    config: SelectionConfig,
) -> None:
    """§7's closing invariant. **Asserted, not assumed.**

    A hard error rather than a flag: every one of these means the caption
    downstream would state something the pipeline had already ruled out, and a
    warning in a log file does not stop that from being published.
    """
    rejected = set(result.rejected_ids)
    guilty = [pid for pid in result.selected_ids if pid in rejected]
    if guilty:
        # Holds under every ablation: A3 turns the optimisation off, not the
        # admission policy (§1).
        raise RuntimeError(f"P* chứa mệnh đề bị BÁC BỎ: {guilty}")

    conflicts = [
        (a, b)
        for i, a in enumerate(result.selected_ids)
        for b in result.selected_ids[i + 1 :]
        if b in contradictions.get(a, ())
    ]
    if conflicts and config.enabled:
        raise RuntimeError(f"P* chứa cặp mâu thuẫn: {conflicts}")
    if conflicts:
        # A3 keeps everything, so a surviving contradiction is the ablation's
        # own result — Baseline D shipping `áo đỏ` and `áo xanh` together is the
        # cost of removing selection. Raising here would make A3 unrunnable and
        # hide the finding it exists to produce.
        result.flags.append("contradictions_present")
        result.log(
            "contradictions_present",
            f"{len(conflicts)} cặp mâu thuẫn còn trong P* — A3 không áp ràng buộc nhất quán",
            pairs=[list(pair) for pair in conflicts],
        )

    if config.enabled and len(result.selected_ids) > config.budget:
        raise RuntimeError(f"P* vượt ngân sách B={config.budget}: {len(result.selected_ids)}")

    # Read back from the PROPOSITION, not from `_Candidate.status`: recomputing
    # the mark with the same accessor that produced it makes the check
    # tautological, and it is the one check standing between an UNCERTAIN
    # proposition and the assertable set.
    hedged = [
        pid for pid in result.selected_ids
        if _verdict_of(by_id[pid].prop) == Verdict.UNCERTAIN
    ]
    if set(hedged) != set(result.hedged_ids):
        raise RuntimeError("mệnh đề KHÔNG CHẮC CHẮN không được đánh dấu rào đón")
    if config.enabled and len(hedged) > config.hedge_quota:
        raise RuntimeError(f"P* vượt hạn ngạch rào đón q={config.hedge_quota}: {len(hedged)}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def select(
    propositions: Sequence[dict[str, Any]],
    entities: Sequence[dict[str, Any]] = (),
    *,
    config: SelectionConfig | None = None,
    similarity_fn: SimilarityFn | None = None,
    surprisal_fn: SurprisalFn | None = None,
) -> SelectionResult:
    """Choose P* from verified propositions (`formulation/10` §3, §4.1).

    `entities` is the registry M1 minted; `salience` on it drives coverage
    (§2.4) and the η term of u(p) (§2.2). `similarity_fn` and `surprisal_fn`
    are injected so this module needs no embedding stack and no corpus file —
    both are optional, and their absence is recorded on the result rather than
    filled in with a default.

    Raises `ValueError` for an algorithm this module does not implement:
    labelling greedy output `"mmr"` would put a number in a comparator column
    that no comparator produced. MMR, top-k, ILP and the dominating-set
    heuristic are §4's comparators and belong in their own modules.
    """
    config = config or SelectionConfig()

    if config.enabled and config.algorithm != "greedy_submodular":
        raise ValueError(
            f"thuật toán {config.algorithm!r} chưa được cài đặt trong module này "
            "(xem formulation/10 §4 — chỉ greedy_submodular là phương pháp chính)"
        )

    selector = _Selector(propositions, entities, config, similarity_fn, surprisal_fn)
    if config.enabled:
        return selector.run()

    # ------------------------------------------------------------------
    # Ablation 3 — no selection (Baseline D).
    #
    # Two documents disagree on what A3 keeps: doc 10 §4.4 says `P* ← P_sup`,
    # `formulation/07` line 30 says every proposition with status ≠ REJECTED.
    # We follow doc 07 (and the module brief) and keep the UNCERTAIN ones — but
    # as HEDGED content, because "UNCERTAIN is not factual content" is an
    # admission policy (§1), not an artefact of the optimisation being on. So
    # the two readings differ only in how that content is realised, never in
    # what may be asserted: `factual_ids` is P_sup under either.
    #
    # The quota is not enforced here: enforcing q would be a selection
    # decision, and A3 is the condition where selection does not run.
    # ------------------------------------------------------------------
    selected = sorted((c.pid for c in selector.candidates), key=_id_order)
    selector.result.log(
        "selection_disabled",
        "A3 — bỏ qua tối ưu hoá, giữ mọi mệnh đề không bị bác bỏ",
        n_selected=len(selected),
        # `budget` is still reported so the run record shows which B this row
        # would have been compared against — but B did not constrain anything
        # here, and a B-sweep over A3 is therefore a flat line by construction.
        budget_applied=False,
    )
    hedged = [pid for pid in selected if selector.by_id[pid].is_uncertain]
    if len(hedged) > config.hedge_quota:
        selector.result.flags.append("hedge_quota_not_enforced")
        selector.result.log(
            "hedge_quota_not_enforced",
            f"{len(hedged)} mệnh đề rào đón > q={config.hedge_quota} — A3 không áp hạn ngạch",
        )
    return selector.finalise(selected)
