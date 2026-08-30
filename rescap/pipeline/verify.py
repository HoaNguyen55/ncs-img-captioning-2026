"""M4 — three-way proposition verification.

    Image I + p  ──►  V(p) ∈ {SUPPORTED, UNCERTAIN, REJECTED}

Implements `formulation/09-MODULE-VERIFICATION.md`: the five score components
(§2), the ten channels routed by proposition type (§3), and the decision rule
(§4). Thresholds and weights come from `formulation/07 §9`.

**The distinction this module exists to preserve** (formulation/09 §1):

    "the image contradicts this"   ->  REJECTED
    "the image cannot settle this" ->  UNCERTAIN   <- collapsed away by binary
    "not visually determinable"    ->  REJECTED *as a visual fact*, never "sai"

Three properties are load-bearing and are enforced structurally rather than by
convention, because each of them, if it slipped, would turn the system's own
output into a false claim:

1. **The step order in §4 is not a style choice.** Hard constraint, then the
   existence cascade, then the E gate, then the C/S thresholds. `decide()`
   implements exactly that order and nothing else reorders it.
2. **E is a gate, not a term in S.** Averaging sufficiency into support would
   let a confident model outvote the fact that there was nothing to look at.
3. **A ceiling may only ever lower a verdict.** Epistemic caps (INFERENCE,
   unresolved `xanh`, depth relations, unconfirmed subject entities, exact
   counts above five, contradiction cycles) are applied as `min` over verdict
   rank, so no cap can promote a REJECTED.

**What the module deliberately does not do.** It never invents a number. Every
signal that cannot be computed on the machine at hand — no bbox, no numpy, a
probe that timed out — returns `None` with a recorded reason and is excluded
from the aggregate, rather than being scored 0. "Not measured" and "measured
and bad" are different facts and are kept apart, which is the same distinction
as UNCERTAIN vs REJECTED one level up.

No torch / transformers / numpy at module level: this must import on a CPU box.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Iterable, Sequence

from ..svp.matching import canonical, compatible_categories
from ..vi.color import Xanh, colour_term, parse_color
from ..vi.lexicon import COLOR_MODIFIERS, DEPTH_DEPENDENT, EXACT_COUNT_LIMIT
from ..vlm.base import Polarity, VLM, YesNo, parse_yes_no

# ---------------------------------------------------------------------------
# Verdicts
# ---------------------------------------------------------------------------


class Verdict(str, Enum):
    """The three-way verdict. Binary is not representable, on purpose."""

    SUPPORTED = "SUPPORTED"
    UNCERTAIN = "UNCERTAIN"
    REJECTED = "REJECTED"


# Rank, used only to clamp: a ceiling takes the MINIMUM of the two ranks, so it
# can lower a verdict and never raise one. REJECTED is the floor because a
# proposition the image contradicts must not be rescued by an epistemic cap.
_RANK: dict[Verdict, int] = {
    Verdict.REJECTED: 0,
    Verdict.UNCERTAIN: 1,
    Verdict.SUPPORTED: 2,
}


class Mode:
    """`verification.mode` from formulation/07 §9. A1 and A2 are config flags."""

    THREE_WAY = "three_way"
    BINARY = "binary"      # A2 = Baseline C
    NONE = "none"          # A1 = Baseline B


MODES = (Mode.THREE_WAY, Mode.BINARY, Mode.NONE)


# ---------------------------------------------------------------------------
# The hard constraint (formulation/09 §4 line 2, formulation/08 §2)
# ---------------------------------------------------------------------------
# SPECULATION-class inference: not determinable from pixels *in principle*.
# `requires_inference` alone is NOT enough to trigger it — see the note on
# `hard_constraint()`.
SPECULATIVE_INFERENCE_TYPES = frozenset(
    {
        "purpose",
        "intention",
        "causation",
        "identity",
        "profession",
        "emotion",
        "temporal_before_after",
    }
)

INFERENCE_TYPE_VI: dict[str, str] = {
    "purpose": "mục đích",
    "intention": "ý định",
    "causation": "nguyên nhân",
    "identity": "danh tính",
    "profession": "nghề nghiệp",
    "emotion": "cảm xúc",
    "temporal_before_after": "thời điểm trước/sau",
}

INFERENCE_TYPE_EN: dict[str, str] = {
    "purpose": "purpose",
    "intention": "intention",
    "causation": "causation",
    "identity": "identity",
    "profession": "profession",
    "emotion": "emotion",
    "temporal_before_after": "before/after in time",
}

# Proposition type -> the authoritative channel name. The names are the closed
# set `Verification.channel_scores` allows (configs/proposition_schema.json);
# emitting anything else would fail schema validation.
TYPE_CHANNEL: dict[str, str] = {
    "entity": "existence",
    "attribute": "attribute",
    "action": "action",
    "relation": "relation",
    "interaction": "relation",
    "spatial_relation": "spatial",
    "counting": "counting",
    "scene": "scene",
}

# Dependency order (formulation/09 §3.1): existence -> attribute/action ->
# relation/spatial -> scene. Verifying out of this order would make the cascade
# depend on list order instead of on the entity's verdict.
TYPE_STAGE: dict[str, int] = {
    "entity": 0,
    "counting": 1,
    "attribute": 2,
    "action": 2,
    "relation": 3,
    "interaction": 3,
    "spatial_relation": 3,
    "scene": 4,
}

# Spatial relations, normalised to the schema's underscore spelling. lexicon.py
# spells them with spaces and the schema enum with underscores; comparing the
# two forms directly silently fails every membership test.
DEPTH_DEPENDENT_NORM = frozenset(r.replace(" ", "_") for r in DEPTH_DEPENDENT)

# Inverse pairs. This is structural asymmetry (A left-of B excludes B left-of A),
# not a lexical resource, so it lives here rather than in vi/lexicon.py.
SPATIAL_INVERSE: dict[str, str] = {
    "bên_trái": "bên_phải",
    "bên_phải": "bên_trái",
    "trên": "dưới",
    "dưới": "trên",
    "phía_trước": "phía_sau",
    "phía_sau": "phía_trước",
    "trong": "ngoài",
    "ngoài": "trong",
    "gần": "xa",
    "xa": "gần",
}

# Relations that cannot hold in both directions between the same two entities.
ASYMMETRIC_RELATIONS = frozenset(
    {"bên_trái", "bên_phải", "trên", "dưới", "trong", "phía_trước", "phía_sau", "trên_đầu"}
)


# `Entity.gender.value` members that commit to a gender. `khong_xac_dinh` is the
# schema default and commits to nothing, so it needs no licence.
GENDERED_VALUES = frozenset({"nam", "nu"})


def normalise_relation(relation: str | None) -> str:
    """Schema spelling for a spatial relation (`bên trái` -> `bên_trái`)."""
    return " ".join(str(relation or "").strip().lower().split()).replace(" ", "_")


# ---------------------------------------------------------------------------
# Configuration (formulation/07 §9)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Thresholds:
    """θ = {θ_E, θ_S^lo, θ_S^hi, θ_C^lo, θ_C^hi} (formulation/09 §4.2).

    Defaults are the values written in formulation/07 §9. They are **not**
    dev-tuned: §4.2 requires tuning on the development split and reporting the
    set, so the default `decision_rule_version` says UNTUNED and any results
    table carrying that string is declaring it used untuned thresholds.

    Note on formulation/09 §2.1's third row (V=0.8, C=0.7, E=0.9 -> UNCERTAIN):
    under these defaults C=0.7 exceeds θ_C^hi=0.60 and rejects. The illustrative
    numbers there presume a higher θ_C^hi. What makes "supported AND
    contradicted" representable is the *band* θ_C^lo < C < θ_C^hi, not any
    particular constant — the band exists under every admissible θ.
    """

    E: float = 0.35
    S_lo: float = 0.35
    S_hi: float = 0.70
    C_lo: float = 0.20
    C_hi: float = 0.60

    def validate(self, where: str = "thresholds") -> None:
        for name in ("E", "S_lo", "S_hi", "C_lo", "C_hi"):
            value = getattr(self, name)
            if not 0.0 <= float(value) <= 1.0:
                raise ValueError(f"{where}.{name} must be in [0,1], got {value!r}")
        if self.S_lo > self.S_hi:
            raise ValueError(f"{where}: S_lo ({self.S_lo}) > S_hi ({self.S_hi})")
        if self.C_lo > self.C_hi:
            raise ValueError(f"{where}: C_lo ({self.C_lo}) > C_hi ({self.C_hi})")


@dataclass(frozen=True)
class Weights:
    """w_V, w_M, w_G with w_V + w_M + w_G = 1 (formulation/09 §2.1)."""

    V: float = 0.6
    M: float = 0.2
    G: float = 0.2

    def validate(self) -> None:
        if min(self.V, self.M, self.G) < 0:
            raise ValueError(f"weights must be non-negative, got {self}")
        total = self.V + self.M + self.G
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"weights must sum to 1, got {total} from {self}")


@dataclass(frozen=True)
class SignalCalibration:
    """Constants the evidence-sufficiency signals are measured against.

    **All UNTUNED.** They are hyper-parameters exactly like θ (formulation/09
    §4.2) and belong in the dev-split sweep; they are named here rather than
    buried as literals so a reviewer can see what was assumed.
    """

    min_area_ratio: float = 0.005      # below this, the region is too small to judge
    good_area_ratio: float = 0.05      # at or above this, region size is a non-issue
    sharpness_reference: float = 100.0  # Laplacian variance treated as "sharp"
    exposure_ok_low: float = 60.0       # mean luminance band, 0-255
    exposure_ok_high: float = 200.0
    exposure_tolerance: float = 60.0
    clipped_fraction_limit: float = 0.5  # fraction of black/white-clipped pixels
    near_threshold: float = 0.25         # `gần` cut-off, fraction of image diagonal
    geometric_margin: float = 0.03       # dead band on dx/dy, fraction of the frame
    containment_high: float = 0.80       # intersection / area(subject) for `trong`
    containment_low: float = 0.20


#: Samples used to estimate confidence by agreement when the backbone exposes
#: no token probabilities. 5 is the smallest odd k that can distinguish
#: unanimous (5/5) from a clear majority (4/5) from a bare one (3/5); k=3 only
#: separates 3/3 from 2/3 and gives the threshold band almost nothing to grip.
SELF_CONSISTENCY_K = 5


@dataclass
class VerificationConfig:
    """One config object per experimental condition (formulation/07 §9).

    Ablation flags, and what each one is measuring:

    | field | ablation | effect |
    |---|---|---|
    | `mode = "none"` | **A1** = Baseline B | no verification at all |
    | `mode = "binary"` | **A2** = Baseline C | UNCERTAIN collapsed away |
    | `contradiction_detection = False` | **A4** | channel 9(a) and 9(b) off |
    | `spatial_verification = False` | **A7** | channel 6 (geometry) off |
    """

    mode: str = Mode.THREE_WAY
    contradiction_detection: bool = True     # A4
    spatial_verification: bool = True        # A7

    thresholds: Thresholds = field(default_factory=Thresholds)
    weights: Weights = field(default_factory=Weights)
    calibration: SignalCalibration = field(default_factory=SignalCalibration)

    #: Per-type overrides. formulation/09 §4.2 expects these (counting and scene
    #: warrant a higher bar than existence) but supplies no values, so the
    #: default is empty: inventing "tuned" numbers would fabricate a result.
    per_type_thresholds: dict[str, Thresholds] = field(default_factory=dict)

    #: Where UNCERTAIN goes under A2. Binary must pick one of the two failure
    #: modes in formulation/09 §1 — dropped detail or asserted hallucination.
    #: REJECTED (drop) is the conservative default; the choice is reported,
    #: never assumed, because it decides which failure Baseline C exhibits.
    binary_uncertain_to: Verdict = Verdict.REJECTED

    #: Self-consistency samples per probe. With k=1 on a backbone that exposes
    #: no token probabilities (`VLM.supports_logprobs is False`, e.g. Vintern)
    #: every probe confidence is 0.0, V lands at the neutral 0.5 and nothing can
    #: reach SUPPORTED. That is the honest outcome for an uncalibrated verifier;
    #: k > 1 buys agreement-fraction confidence instead (formulation/08 §8.2).
    probe_k: int = 1

    #: Acquiescence (the model affirms a claim AND its negation) must push to
    #: UNCERTAIN, never to REJECTED — the model's failure to discriminate is not
    #: evidence against the claim. Validated to sit strictly inside the band.
    acquiescence_c: float = 0.50

    #: A decidable geometric test that fails is decisive: geometry is the one
    #: channel that cannot hallucinate (formulation/09 §3.2), so it must be able
    #: to reject on its own, which under §4 means C >= θ_C^hi.
    geometric_contradiction_c: float = 1.0

    probe_retries: int = 2
    retry_backoff_s: float = 0.5

    decision_rule_version: str = "v1.0-doc07-UNTUNED"
    verifier: str = ""  # filled from the backbone name at run time

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        if self.mode not in MODES:
            raise ValueError(f"verification.mode must be one of {MODES}, got {self.mode!r}")
        self.thresholds.validate()
        for ptype, thresholds in self.per_type_thresholds.items():
            thresholds.validate(f"per_type_thresholds[{ptype}]")
        self.weights.validate()
        if self.probe_k < 1:
            raise ValueError(f"probe_k must be >= 1, got {self.probe_k}")
        # The acquiescence invariant. If this sits above θ_C^hi the guard turns
        # a model's inability to discriminate into "the image says no", which is
        # exactly the false claim the three-way verdict exists to avoid.
        # Checked against EVERY threshold set that can be selected, not just the
        # global one: `thresholds_for()` swaps in a per-type set, so a per-type
        # θ_C^hi below acquiescence_c would reinstate exactly the collapse this
        # invariant forbids, for that one proposition type, invisibly.
        sets = [("thresholds", self.thresholds)] + [
            (f"per_type_thresholds[{ptype}]", t) for ptype, t in self.per_type_thresholds.items()
        ]
        for where, thresholds in sets:
            lo, hi = thresholds.C_lo, thresholds.C_hi
            if not lo < self.acquiescence_c < hi:
                raise ValueError(
                    f"acquiescence_c ({self.acquiescence_c}) must sit strictly between "
                    f"{where}.C_lo ({lo}) and {where}.C_hi ({hi}) so acquiescence yields "
                    "UNCERTAIN, never REJECTED"
                )

    def thresholds_for(self, proposition_type: str) -> Thresholds:
        return self.per_type_thresholds.get(str(proposition_type), self.thresholds)

    def resolved_decision_rule_version(self) -> str:
        """Version string written into every verdict.

        Carries the aggregation choice and the ablation flags, because two runs
        with the same θ but different flags are not the same rule and a verdict
        whose rule is not reconstructible is unreproducible (formulation/09 §7).
        """
        flags = [f"mode={self.mode}", "E=min-prereq/max-channel"]
        if not self.contradiction_detection:
            flags.append("A4")
        if not self.spatial_verification:
            flags.append("A7")
        if self.per_type_thresholds:
            flags.append("per_type")
        return f"{self.decision_rule_version}+{'+'.join(flags)}"

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> "VerificationConfig":
        """Build from the `verification:` block of a pipeline config.

        **Missing thresholds or weights is a hard error** (formulation/09 §7):
        a config file that forgot the block must not quietly inherit a default
        set, because the resulting numbers would be attributed to thresholds
        that were never chosen. In-code construction is a different case — there
        the defaults are visible at the call site and are tagged UNTUNED.
        """
        for required in ("thresholds", "weights"):
            if required not in config:
                raise KeyError(
                    f"verification config is missing {required!r}; refusing to "
                    "fall back to a default threshold set silently "
                    "(formulation/09 §7)"
                )
        thresholds = Thresholds(**config["thresholds"])
        weights = Weights(**config["weights"])
        per_type = {
            ptype: Thresholds(**values)
            for ptype, values in (config.get("per_type_thresholds") or {}).items()
        }
        return cls(
            mode=config.get("mode", Mode.THREE_WAY),
            contradiction_detection=bool(config.get("contradiction_detection", True)),
            spatial_verification=bool(config.get("spatial_verification", True)),
            thresholds=thresholds,
            weights=weights,
            calibration=SignalCalibration(**(config.get("calibration") or {})),
            per_type_thresholds=per_type,
            binary_uncertain_to=Verdict(config.get("binary_uncertain_to", Verdict.REJECTED)),
            probe_k=int(config.get("probe_k", 1)),
            acquiescence_c=float(config.get("acquiescence_c", 0.50)),
            decision_rule_version=str(config.get("decision_rule_version", "v1.0-doc07-UNTUNED")),
        )


# ---------------------------------------------------------------------------
# Score containers
# ---------------------------------------------------------------------------


@dataclass
class Scores:
    """The five components of formulation/09 §2, plus the aggregate S.

    `None` means ⊥ — *not computed*, which is different from computed-as-zero.
    A ⊥ component is dropped from S and its weight redistributed (§2.1).
    """

    v: float | None = None   # visual support
    m: float | None = None   # semantic consistency with the rest of the set
    g: float | None = None   # geometric consistency, ⊥ for non-spatial types
    c: float = 0.0           # contradiction
    e: float | None = None   # evidence sufficiency
    s: float | None = None   # w_V·V + w_M·M + w_G·G, weights redistributed

    def combine(self, weights: Weights) -> "Scores":
        """Fill `s` from the live components, redistributing dropped weights."""
        live = [(weights.V, self.v), (weights.M, self.m), (weights.G, self.g)]
        live = [(w, x) for w, x in live if x is not None and w > 0]
        total = sum(w for w, _ in live)
        self.s = (sum(w * x for w, x in live) / total) if total > 0 else None
        return self


@dataclass
class EvidenceSignal:
    """One input to E, with what a low value *means* (formulation/09 §3.4).

    `value is None` means the signal could not be measured at all; it is
    excluded from the aggregate rather than counted as zero, and `unavailable`
    records why so a run with no bboxes is distinguishable from a run with tiny
    ones.

    `kind` decides how the signal aggregates:

    * `prerequisite` — a condition for being able to judge *at all* (region
      size, blur, exposure, occlusion, grounding). Any one of these can veto.
    * `channel` — one source of an answer (the VLM probe, the geometric test).
      Channels are **alternatives**, so the best one counts.
    """

    name: str
    value: float | None
    low_means_vi: str = ""
    low_means_en: str = ""
    unavailable: str = ""
    kind: str = "prerequisite"


def verdict_name(proposition: dict) -> str | None:
    """The verdict on a proposition, as an upper-case name, or None.

    **One implementation, because there were six and they disagreed.** An audit
    found `metrics_svp`, `select`, `realize`, and three scripts each parsing this
    themselves, differing on three inputs:

    * an enum — `metrics_svp` returned the string `"Verdict.UNCERTAIN"`, the
      exact defect already fixed in `select` and `realize`, still live in the
      module that computes PGF, VCF and the hallucination rates;
    * a lower-case string — `None` from one, `"uncertain"` from two,
      `"UNCERTAIN"` from two more;
    * a missing verdict — `None` from most, the string `"NONE"` from one, which
      makes an unverified proposition look like it carries a verdict called NONE.

    Accepts an enum, a `str` enum, a plain string in any case, or nothing.

    >>> verdict_name({"verification": {"status": "uncertain"}})
    'UNCERTAIN'
    >>> verdict_name({"verification": {"status": None}}) is None
    True
    >>> verdict_name({}) is None
    True
    """
    status = (proposition.get("verification") or {}).get("status")
    if status is None:
        return None
    # `.value` first: `str(Verdict.UNCERTAIN)` is `'Verdict.UNCERTAIN'` on a
    # str-Enum, which is what made four of the six implementations wrong.
    text = str(getattr(status, "value", status)).strip()
    if not text or text.lower() in ("none", "null"):
        return None
    return text.rsplit(".", 1)[-1].upper()


@dataclass
class Ceiling:
    """The best verdict a proposition is still allowed to receive.

    Applied as `min` over rank after the decision rule, so it can only lower.
    """

    verdict: Verdict
    reason_vi: str
    reason_en: str


@dataclass
class Cascade:
    """An inherited failure (formulation/09 §3.1, doc 04 §3.2)."""

    entity_id: str
    role: str            # "subject" | "object"
    from_proposition: str


@dataclass
class GeometricCheck:
    """Output of the model-free spatial test (formulation/09 §3.2).

    `consistent is None` means the geometry does not settle it — depth
    relations, a missing bbox, a difference inside the margin. That is a real
    answer and it caps the verdict at UNCERTAIN; it is not a failure.
    """

    relation: str
    consistent: bool | None
    iou: float | None = None
    centroid_dx: float | None = None
    centroid_dy: float | None = None
    depth_order: str = "unknown"
    reason_vi: str = ""
    reason_en: str = ""

    @property
    def score(self) -> float | None:
        """G ∈ [0,1] ∪ {⊥}."""
        if self.consistent is None:
            return None
        return 1.0 if self.consistent else 0.0

    def to_schema(self) -> dict[str, Any]:
        """`SpatialRelation.geometric_check`; only schema-allowed keys."""
        out: dict[str, Any] = {"depth_order": self.depth_order, "consistent": self.consistent}
        if self.iou is not None:
            out["iou"] = round(self.iou, 4)
        if self.centroid_dx is not None:
            out["centroid_dx"] = round(self.centroid_dx, 4)
        if self.centroid_dy is not None:
            out["centroid_dy"] = round(self.centroid_dy, 4)
        return out


@dataclass
class VerificationResult:
    """Everything M4 knows about one proposition.

    Wider than the schema's `Verification` object on purpose: schema v1.0.0 sets
    `additionalProperties: false` and has no field for the cascade, for the
    per-component V/M/G breakdown, or for the acquiescence/disagreement flags.
    Those stay here so doc 04 can split primary from inherited hallucinations
    and doc 05 can report the diagnostics; writing them into the JSON document
    needs a schema version bump, not a silent extra key.
    """

    proposition_id: str
    status: Verdict | None
    branch: str
    scores: Scores
    explanation_vi: str
    explanation_en: str
    channel_scores: dict[str, float] = field(default_factory=dict)
    evidence_signals: list[EvidenceSignal] = field(default_factory=list)
    limiting_signal: str | None = None
    ceilings: list[Ceiling] = field(default_factory=list)
    #: How many of the proposition's ceilings have already been clamped in.
    #: The cycle ceiling is added after every S is known, so clamping happens in
    #: two rounds and the second must not re-append the first round's reasons.
    ceilings_applied: int = 0
    cascade_from: list[Cascade] = field(default_factory=list)
    contradicts: list[str] = field(default_factory=list)
    supported_by: list[str] = field(default_factory=list)
    geometric: GeometricCheck | None = None
    probes: list[dict[str, Any]] = field(default_factory=list)
    #: Verdict before the A2 collapse. formulation/05 §7.4's C-vs-D confusion
    #: table is exactly the mapping from this field to `status`.
    three_way_status: Verdict | None = None
    acquiescent: bool = False
    geometry_probe_disagreement: bool = False
    lang_fallback: bool = False
    verifier: str = ""
    decision_rule_version: str = ""
    verified_at: str = ""

    def to_schema(self) -> dict[str, Any]:
        """The `Verification` object, with only the keys the schema allows."""
        out: dict[str, Any] = {
            "status": self.status.value if self.status is not None else None,
            "explanation_vi": self.explanation_vi,
            "explanation_en": self.explanation_en,
            "verifier": self.verifier,
            "decision_rule_version": self.decision_rule_version,
            "verified_at": self.verified_at,
        }
        if self.scores.s is not None:
            out["support_score"] = round(self.scores.s, 4)
        if self.scores.s is not None or self.scores.e is not None or self.scores.c > 0.0:
            # Omitted entirely when nothing was scored (a hard-constraint
            # rejection returns before scoring): a 0.0 there would read as
            # "we looked for counter-evidence and found none".
            out["contradiction_score"] = round(self.scores.c, 4)
        if self.scores.e is not None:
            out["evidence_sufficiency"] = round(self.scores.e, 4)
        if self.channel_scores:
            out["channel_scores"] = {k: round(v, 4) for k, v in self.channel_scores.items()}
        return out


@dataclass
class VerificationStats:
    """Reported alongside the verdicts. Silence about a degraded channel reads
    as a working one (formulation/07 §6)."""

    mode: str = Mode.THREE_WAY
    n_propositions: int = 0
    by_status: dict[str, int] = field(default_factory=dict)
    hard_constraint_rejections: int = 0
    cascaded: int = 0
    gate_uncertain: int = 0
    ceilinged: int = 0
    binary_collapsed: int = 0
    contradiction_pairs: int = 0
    contradiction_cycles: list[list[str]] = field(default_factory=list)
    acquiescence: int = 0
    geometry_probe_disagreements: int = 0
    geometry_decided: int = 0
    #: Backbone calls that returned an answer, retries included. Not the number
    #: of propositions probed — a language retry costs two calls and shows up as
    #: two here, which is what a cost report needs.
    probe_calls: int = 0
    probe_failures: int = 0
    lang_fallbacks: int = 0
    xanh_unresolved: int = 0
    #: Colour propositions put to a second, independent verifier, and how often
    #: the two disagreed. A disagreement rate near zero would mean the second
    #: verifier is buying nothing and the extra probes should be dropped.
    colour_cross_checked: int = 0
    colour_disagreements: int = 0
    entities_without_bbox: int = 0
    total_rejection: bool = False
    verification_disabled: bool = False
    notes: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Geometry — channel 6, the model-free anchor (formulation/09 §3.2)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Box:
    """A bbox in the schema's format: [x, y, w, h] in ABSOLUTE pixels.

    Not corners. Reading it as [x1, y1, x2, y2] flips every geometric verdict
    while still producing plausible-looking numbers, so the conversion happens
    in exactly one place.
    """

    x: float
    y: float
    w: float
    h: float

    @property
    def cx(self) -> float:
        return self.x + self.w / 2.0

    @property
    def cy(self) -> float:
        return self.y + self.h / 2.0

    @property
    def area(self) -> float:
        return max(0.0, self.w) * max(0.0, self.h)

    @property
    def right(self) -> float:
        return self.x + self.w

    @property
    def bottom(self) -> float:
        return self.y + self.h


def as_box(bbox: Any) -> Box | None:
    """Parse an entity bbox, or None when it is absent or malformed."""
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return None
    try:
        x, y, w, h = (float(v) for v in bbox)
    except (TypeError, ValueError):
        return None
    if w <= 0 or h <= 0:
        return None
    return Box(x, y, w, h)


def _intersection_area(a: Box, b: Box) -> float:
    dx = min(a.right, b.right) - max(a.x, b.x)
    dy = min(a.bottom, b.bottom) - max(a.y, b.y)
    return dx * dy if dx > 0 and dy > 0 else 0.0


def iou(a: Box, b: Box) -> float:
    inter = _intersection_area(a, b)
    union = a.area + b.area - inter
    return inter / union if union > 0 else 0.0


def geometric_check(
    relation: str,
    subject: Box | None,
    obj: Box | None,
    *,
    image_size: tuple[float, float] | None = None,
    calibration: SignalCalibration | None = None,
) -> GeometricCheck:
    """Deterministic bbox test for one spatial relation.

    Returns `consistent = None` whenever geometry genuinely cannot settle the
    question — no boxes, a depth relation, a difference inside the margin, or a
    relation with no deterministic test. Guessing in those cases would give the
    one channel that cannot hallucinate a way to hallucinate.

    >>> a, b = Box(0, 0, 10, 10), Box(50, 0, 10, 10)
    >>> geometric_check("bên_trái", a, b, image_size=(100, 100)).consistent
    True
    >>> geometric_check("bên_phải", a, b, image_size=(100, 100)).consistent
    False
    >>> geometric_check("phía_trước", a, b, image_size=(100, 100)).consistent is None
    True
    """
    cal = calibration or SignalCalibration()
    rel = normalise_relation(relation)
    check = GeometricCheck(relation=rel, consistent=None)

    if rel in DEPTH_DEPENDENT_NORM:
        # 2-D boxes carry no depth. This caps the verdict at UNCERTAIN rather
        # than letting the probe assert an ordering nothing can check.
        return GeometricCheck(
            relation=rel,
            consistent=None,
            depth_order="unknown",
            reason_vi="quan hệ theo chiều sâu, không xác định được từ ảnh 2 chiều",
            reason_en="depth relation, undecidable from 2-D boxes",
        )

    if subject is None or obj is None:
        return GeometricCheck(
            relation=rel,
            consistent=None,
            reason_vi="thiếu hộp giới hạn của đối tượng",
            reason_en="missing bounding box",
        )

    width, height = image_size if image_size else (None, None)
    span_x = float(width) if width else max(subject.w, obj.w, abs(subject.cx - obj.cx), 1.0)
    span_y = float(height) if height else max(subject.h, obj.h, abs(subject.cy - obj.cy), 1.0)
    diagonal = math.hypot(span_x, span_y)

    dx = (subject.cx - obj.cx) / span_x
    dy = (subject.cy - obj.cy) / span_y
    check = GeometricCheck(
        relation=rel,
        consistent=None,
        iou=iou(subject, obj),
        centroid_dx=dx,
        centroid_dy=dy,
    )
    margin = cal.geometric_margin
    inter = _intersection_area(subject, obj)
    containment = inter / subject.area if subject.area > 0 else 0.0
    distance = math.hypot(subject.cx - obj.cx, subject.cy - obj.cy) / diagonal

    def decided(flag: bool, vi: str, en: str) -> GeometricCheck:
        check.consistent = flag
        check.reason_vi = vi
        check.reason_en = en
        return check

    def undecided(vi: str, en: str) -> GeometricCheck:
        check.consistent = None
        check.reason_vi = vi
        check.reason_en = en
        return check

    if rel in ("bên_trái", "bên_phải"):
        if abs(dx) < margin:
            return undecided(
                "chênh lệch ngang nhỏ hơn ngưỡng, không kết luận được",
                "horizontal offset inside the margin",
            )
        left = dx < 0
        return decided(
            left if rel == "bên_trái" else not left,
            f"lệch ngang dx={dx:.3f} (khung ảnh)",
            f"horizontal offset dx={dx:.3f} (image frame)",
        )

    if rel in ("trên", "dưới", "trên_đầu"):
        if abs(dy) < margin:
            return undecided(
                "chênh lệch dọc nhỏ hơn ngưỡng, không kết luận được",
                "vertical offset inside the margin",
            )
        # Image y grows downward, so "above" means a SMALLER centroid y.
        above = dy < 0
        if rel == "trên_đầu":
            # Stricter than `trên`: the whole subject must clear the object's top.
            above = subject.bottom <= obj.y
        return decided(
            above if rel in ("trên", "trên_đầu") else not above,
            f"lệch dọc dy={dy:.3f}, tiếp giáp={_adjacent(subject, obj, span_y)}",
            f"vertical offset dy={dy:.3f}, adjacent={_adjacent(subject, obj, span_y)}",
        )

    if rel == "trong":
        if containment >= cal.containment_high and subject.area < obj.area:
            return decided(
                True,
                f"chủ thể nằm trong đối tượng ({containment:.2f} diện tích)",
                f"subject contained ({containment:.2f} of its area)",
            )
        if containment <= cal.containment_low:
            return decided(
                False,
                f"hầu như không chồng lấn ({containment:.2f})",
                f"almost no overlap ({containment:.2f})",
            )
        return undecided(
            f"chồng lấn một phần ({containment:.2f}), không kết luận được",
            f"partial overlap ({containment:.2f})",
        )

    if rel == "ngoài":
        if inter == 0.0:
            return decided(True, "hai hộp không chồng lấn", "boxes do not overlap")
        if containment >= cal.containment_high:
            return decided(False, "chủ thể nằm trong đối tượng", "subject is contained")
        return undecided("chồng lấn một phần", "partial overlap")

    if rel in ("gần", "xa", "bên_cạnh"):
        near_cut, far_cut = cal.near_threshold - margin, cal.near_threshold + margin
        if near_cut < distance < far_cut:
            return undecided(
                f"khoảng cách {distance:.3f} nằm trong dải biên, không kết luận được",
                f"distance {distance:.3f} inside the margin band",
            )
        near = distance <= near_cut
        return decided(
            near if rel in ("gần", "bên_cạnh") else not near,
            f"khoảng cách tâm chuẩn hoá {distance:.3f}",
            f"normalised centroid distance {distance:.3f}",
        )

    if rel == "ở_giữa":
        # `ở_giữa` needs TWO reference objects; Proposition carries one object,
        # so the hull test is not computable from this record. Not a failure —
        # a missing input, and it is reported as such.
        return undecided(
            "cần hai đối tượng tham chiếu, lược đồ chỉ ghi một",
            "needs two reference objects; the record has one",
        )

    return undecided(
        f"chưa có kiểm tra hình học xác định cho quan hệ {rel!r}",
        f"no deterministic geometric test for {rel!r}",
    )


def _adjacent(subject: Box, obj: Box, span_y: float) -> bool:
    """Box adjacency, the contact half of the `trên` test."""
    horizontal_overlap = min(subject.right, obj.right) - max(subject.x, obj.x) > 0
    gap = abs(subject.bottom - obj.y) / span_y if span_y else 1.0
    return bool(horizontal_overlap and gap < 0.05)


# ---------------------------------------------------------------------------
# Channel 10 — evidence sufficiency (formulation/09 §3.4)
# ---------------------------------------------------------------------------


def _ramp(value: float, low: float, high: float) -> float:
    """0 at or below `low`, 1 at or above `high`, linear between."""
    if high <= low:
        return 1.0 if value >= high else 0.0
    return max(0.0, min(1.0, (value - low) / (high - low)))


def _image_size(image: Any) -> tuple[float, float] | None:
    size = getattr(image, "size", None)
    if isinstance(size, (tuple, list)) and len(size) == 2:
        try:
            width, height = float(size[0]), float(size[1])
        except (TypeError, ValueError):
            return None
        if width > 0 and height > 0:
            return width, height
    return None


def _crop(image: Any, box: Box) -> Any | None:
    """Crop for a region probe. None when the object is not a PIL image."""
    try:
        return image.crop((int(box.x), int(box.y), int(box.right), int(box.bottom)))
    except Exception:  # pragma: no cover - depends on the caller's image type
        return None


def _sharpness(region: Any) -> tuple[float | None, str]:
    """Laplacian variance over the region. Returns `(value, why_unavailable)`.

    numpy is imported here rather than at module level: this file must import
    on a box with no scientific stack, and a missing numpy is a *reason the
    signal is unmeasurable*, not a crash and not a zero.
    """
    try:
        import numpy as np  # noqa: PLC0415 - deliberate lazy import
    except Exception:
        return None, "numpy không có sẵn nên không đo được độ nét"
    try:
        grey = np.asarray(region.convert("L"), dtype="float64")
    except Exception:
        return None, "không đọc được điểm ảnh của vùng"
    if grey.ndim != 2 or grey.shape[0] < 3 or grey.shape[1] < 3:
        return None, "vùng ảnh quá nhỏ để tính độ nét"
    laplacian = (
        -4.0 * grey[1:-1, 1:-1]
        + grey[:-2, 1:-1]
        + grey[2:, 1:-1]
        + grey[1:-1, :-2]
        + grey[1:-1, 2:]
    )
    return float(laplacian.var()), ""


def _exposure(region: Any) -> tuple[tuple[float, float] | None, str]:
    """`((mean_luminance, clipped_fraction), "")` from PIL's histogram alone.

    Histogram-based so it needs no numpy: exposure is the signal that catches
    the backlit-umbrella case of formulation/09 §1, and losing it on a machine
    without a scientific stack would lose the paper's own worked example.
    """
    try:
        histogram = region.convert("L").histogram()
    except Exception:
        return None, "không đọc được biểu đồ độ sáng của vùng"
    total = sum(histogram)
    if total <= 0:
        return None, "vùng ảnh rỗng"
    mean = sum(i * n for i, n in enumerate(histogram)) / total
    clipped = (sum(histogram[:8]) + sum(histogram[248:])) / total
    return (mean, clipped), ""


def evidence_signals(
    proposition: dict,
    subject_entity: dict | None,
    image: Any,
    entities: Sequence[dict],
    *,
    probe: YesNo | None,
    probe_failed: bool,
    calibration: SignalCalibration,
    geometric: GeometricCheck | None = None,
    acquiescent: bool = False,
    xanh_unresolved: bool = False,
) -> list[EvidenceSignal]:
    """The inputs to E, each with its own reason string (formulation/09 §3.4)."""
    cal = calibration
    ptype = str(proposition.get("type"))
    signals: list[EvidenceSignal] = []
    box = as_box((subject_entity or {}).get("bbox"))
    size = _image_size(image)

    # -- bbox availability --------------------------------------------------
    if ptype == "scene":
        # A scene claim is about the whole frame; "no bbox" is not a defect.
        signals.append(
            EvidenceSignal("bbox", None, unavailable="mệnh đề bối cảnh không cần hộp giới hạn")
        )
    elif box is not None:
        signals.append(EvidenceSignal("bbox", 1.0))
    elif any(as_box(e.get("bbox")) is not None for e in entities):
        # Other entities in THIS image are grounded and this one is not — that
        # is a measured deficiency of this proposition.
        signals.append(
            EvidenceSignal(
                "bbox",
                0.0,
                "đối tượng không được định vị trong ảnh (không có hộp giới hạn)",
                "the mention is not grounded to a region",
            )
        )
    else:
        # No entity in the document has a bbox: the upstream module did not
        # produce them at all. Scoring 0 here would report a missing pipeline
        # stage as evidence about this image.
        signals.append(
            EvidenceSignal(
                "bbox",
                None,
                unavailable="chưa có hộp giới hạn nào trong tài liệu (M1 chưa cung cấp)",
            )
        )

    # -- region geometry ----------------------------------------------------
    if box is not None and size is not None:
        ratio = box.area / (size[0] * size[1])
        signals.append(
            EvidenceSignal(
                "region_area",
                _ramp(ratio, cal.min_area_ratio, cal.good_area_ratio),
                "vùng ảnh của đối tượng quá nhỏ để đánh giá",
                "the region is too small to judge",
            )
        )
        subject_id = str((subject_entity or {}).get("id", ""))
        others = [
            other
            for other in (
                as_box(e.get("bbox")) for e in entities if str(e.get("id")) != subject_id
            )
            if other is not None
        ]
        if others:
            # Occlusion proxy: how much of this region another entity's box
            # covers. Compared by entity id, not by box value — two entities
            # may legitimately share identical boxes.
            overlap = max(iou(box, other) for other in others)
            signals.append(
                EvidenceSignal(
                    "occlusion",
                    1.0 - overlap,
                    "đối tượng bị vật khác che khuất một phần",
                    "the object is partly occluded",
                )
            )
    elif box is None:
        signals.append(
            EvidenceSignal("region_area", None, unavailable="không có hộp giới hạn")
        )

    # -- pixel statistics ---------------------------------------------------
    region = _crop(image, box) if box is not None else image
    if region is None:
        signals.append(
            EvidenceSignal("sharpness", None, unavailable="không cắt được vùng ảnh")
        )
        signals.append(
            EvidenceSignal("exposure", None, unavailable="không cắt được vùng ảnh")
        )
    else:
        variance, why = _sharpness(region)
        if variance is None:
            signals.append(EvidenceSignal("sharpness", None, unavailable=why))
        else:
            signals.append(
                EvidenceSignal(
                    "sharpness",
                    _ramp(variance, 0.0, cal.sharpness_reference),
                    "vùng ảnh bị mờ",
                    "the region is blurred",
                )
            )
        measured, why = _exposure(region)
        if measured is None:
            signals.append(EvidenceSignal("exposure", None, unavailable=why))
        else:
            mean, clipped = measured
            if mean < cal.exposure_ok_low:
                brightness = _ramp(
                    mean, cal.exposure_ok_low - cal.exposure_tolerance, cal.exposure_ok_low
                )
            elif mean > cal.exposure_ok_high:
                brightness = 1.0 - _ramp(
                    mean, cal.exposure_ok_high, cal.exposure_ok_high + cal.exposure_tolerance
                )
            else:
                brightness = 1.0
            headroom = 1.0 - min(1.0, clipped / cal.clipped_fraction_limit)
            signals.append(
                EvidenceSignal(
                    "exposure",
                    min(brightness, headroom),
                    "vùng ảnh quá tối, quá sáng hoặc bị ngược sáng",
                    "the region is too dark, blown out or backlit",
                )
            )

    # -- channel: the verifier's own answer ---------------------------------
    if probe_failed:
        # formulation/09 §7: a probe that never answered drops sufficiency.
        signals.append(
            EvidenceSignal(
                "probe",
                0.0,
                "câu hỏi kiểm chứng thất bại sau khi thử lại",
                "the verification probe failed after retries",
                kind="channel",
            )
        )
    elif probe is None:
        # No probe was asked at all — `build_questions` could not build one,
        # because the proposition is missing the field its channel interrogates
        # (an `attribute` with no value, a `relation` with no object) or is of a
        # type no channel covers. That is a MEASURED zero, not an unmeasurable
        # signal: every channel in formulation/09 §3 except 6 and 10 is a probe,
        # so "no question" means nothing looked at the image.
        #
        # Recording it as ⊥ excluded it from the aggregate, leaving E resting on
        # the prerequisites alone. A well-lit, sharp, generously-boxed region
        # then scored E = 1.0 for a proposition nobody ever asked about, the
        # gate opened, and S — by then w_M·M over the *other* propositions, with
        # no image term in it at all — could reach θ_S^hi and return SUPPORTED.
        # A verdict of "the image shows this" without a single pixel consulted.
        # Geometry is unaffected: channels aggregate by max, so a decisive
        # channel-6 test still carries E on its own (formulation/09 §3.2).
        signals.append(
            EvidenceSignal(
                "probe",
                0.0,
                "không dựng được câu hỏi kiểm chứng cho mệnh đề này (thiếu trường bắt buộc)",
                "no verification question could be built for this proposition "
                "(a required field is missing)",
                kind="channel",
            )
        )
    elif acquiescent:
        signals.append(
            EvidenceSignal(
                "probe",
                0.0,
                "mô hình khẳng định cả mệnh đề lẫn phủ định của nó (thiên kiến đồng ý)",
                "the model affirmed both the claim and its negation (acquiescence bias)",
                kind="channel",
            )
        )
    elif probe.polarity is Polarity.INCONCLUSIVE:
        signals.append(
            EvidenceSignal(
                "probe",
                0.0,
                "mô hình không đưa ra được câu trả lời xác định",
                "the model could not give a decisive answer",
                kind="channel",
            )
        )
    else:
        # Decisiveness and stability fold into ONE channel value: a model that
        # answers decisively but differently on each resample has not given a
        # usable answer either (formulation/09 §3.4, answer entropy).
        stability = _answer_stability(probe)
        signals.append(
            EvidenceSignal(
                "probe",
                1.0 if stability is None else stability,
                "mô hình trả lời không nhất quán giữa các lần hỏi",
                "the model's answers disagree across resamples",
                kind="channel",
            )
        )

    # -- channel: the model-free geometric test -----------------------------
    # A decisive bbox test IS evidence, whatever the VLM managed to say. Without
    # this the E gate (checked before C) would discard a geometric refutation
    # whenever the probe came back inconclusive, and channel 6 could never
    # decide anything on its own.
    if geometric is not None and geometric.consistent is not None:
        signals.append(EvidenceSignal("geometry", 1.0, kind="channel"))

    if xanh_unresolved:
        signals.append(
            EvidenceSignal(
                "xanh",
                0.0,
                "màu 'xanh' chưa phân giải được thành xanh dương hay xanh lá",
                "bare 'xanh' could not be resolved to blue or green",
            )
        )
    return signals


def _answer_stability(probe: YesNo) -> float | None:
    """1 − normalised entropy over the k resampled answers.

    **Entropy over POLARITY, not over wording.** For a yes/no probe the answer
    is the polarity: `Có.` and `Có, trong ảnh có một người đàn ông.` are the
    same answer given twice, and counting them as two made five agreeing
    samples look maximally unstable.

    This is the same defect that was fixed in `vlm.base.probe`, which computes
    confidence by polarity -- and missed here, because this function
    independently re-derives disagreement from the raw strings. Confidence rose
    and stability stayed pinned near zero, so E stayed low and the three-way
    verdict still collapsed toward UNCERTAIN. Fixing one of two places that
    measure the same thing fixes nothing.

    None at k = 1: with a single sample there is nothing to disagree with, and
    reporting perfect stability would manufacture evidence.
    """
    samples = [
        parse_yes_no(s).value for s in (probe.answer.samples or []) if s.strip()
    ]
    if len(samples) < 2:
        return None
    counts: dict[str, int] = {}
    for sample in samples:
        counts[sample] = counts.get(sample, 0) + 1
    total = len(samples)
    entropy = -sum((n / total) * math.log(n / total) for n in counts.values())
    maximum = math.log(min(len(counts), total)) if len(counts) > 1 else 0.0
    return 1.0 - (entropy / maximum) if maximum > 0 else 1.0


def aggregate_evidence(
    signals: Sequence[EvidenceSignal],
) -> tuple[float | None, EvidenceSignal | None]:
    """E = min(prerequisites, best channel), plus the signal that set it.

    **Not a mean, and not a flat min.** Two different relations are at work:

    * Prerequisites are conjunctive — a sharp, well-exposed view of a
      three-pixel region is still not enough to judge its colour, so any one of
      them can veto. An average would let three easy signals outvote the one
      that actually made the judgement impossible, and E would stop being a
      gate.
    * Channels are disjunctive — they are alternative ways to get an answer. A
      flat min over everything would let an inconclusive VLM probe veto a
      decisive geometric test, which would silently reverse formulation/09
      §3.2's "geometry wins": the E gate fires before the C test, so the
      model-free channel would never get to speak.

    Returning the limiting signal is what lets `explanation_vi` name the
    *cause* rather than restate the verdict (formulation/09 §6).
    """
    measured = [s for s in signals if s.value is not None]
    if not measured:
        return None, None
    effective = [s for s in measured if s.kind != "channel"]
    channels = [s for s in measured if s.kind == "channel"]
    if channels:
        effective.append(max(channels, key=lambda s: s.value))  # type: ignore[arg-type,return-value]
    if not effective:
        return None, None
    weakest = min(effective, key=lambda s: s.value)  # type: ignore[arg-type,return-value]
    return float(weakest.value), weakest  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Probes — channels 1-5, 7, 8 (formulation/09 §3)
# ---------------------------------------------------------------------------
# Verification questions are phrased DIFFERENTLY from the generation prompts in
# pipeline/generate.py. That is mitigation 4 in formulation/09 §8: asking the
# model to re-read its own wording measures agreement with itself, not the
# image.

_VI_MARKS = set(
    "ăâđêôơưàáảãạằắẳẵặầấẩẫậèéẻẽẹềếểễệ"
    "ìíỉĩịòóỏõọồốổỗộờớởỡợùúủũụừứửữựỳýỷỹỵ"
)


def _looks_vietnamese(text: str) -> bool:
    """Cheap language check on a verifier answer.

    Duplicates `vlm.mock.looks_vietnamese` deliberately: that module is a test
    double, and importing a mock into the production path is how fixtures reach
    results tables.
    """
    return any(char in _VI_MARKS for char in text.lower())


# Used as a PREFIX on the language retry. As a suffix it would land between the
# question and the `'có'/'không'` instruction that `VLM.probe_yes_no` appends,
# producing a malformed prompt on exactly the path that already went wrong once.
_STRICTER_VI = "Trả lời bằng tiếng Việt."


def _strip_aspect(text: str) -> str:
    """Drop a leading aspect marker so questions do not read `có đang đang …`."""
    stripped = " ".join(str(text).strip().split())
    for marker in ("đang ", "đã ", "sẽ ", "vừa "):
        if stripped.lower().startswith(marker):
            return stripped[len(marker) :]
    return stripped


def _subject_text(proposition: dict) -> str:
    subject = proposition.get("subject") or {}
    return str(subject.get("text_vi") or subject.get("head_noun_vi") or "đối tượng này")


def _object_text(proposition: dict) -> str:
    obj = proposition.get("object") or {}
    return str(obj.get("text_vi") or obj.get("head_noun_vi") or "")


def build_questions(proposition: dict) -> tuple[str, str] | None:
    """`(claim, negated_claim)` in Vietnamese, by proposition type.

    The negated form is built here rather than by string surgery on the claim,
    because `VLM.probe_yes_no` appends its own `'có'/'không'` instruction and a
    malformed negation would be answered as if it were the claim — turning the
    acquiescence guard into noise.

    >>> build_questions({"type": "entity", "subject": {"text_vi": "một con chó"}})[0]
    'Trong ảnh có một con chó không'
    """
    ptype = str(proposition.get("type"))
    subject = _subject_text(proposition)

    if ptype == "entity":
        return (
            f"Trong ảnh có {subject} không",
            f"Có đúng là trong ảnh KHÔNG có {subject} không",
        )

    if ptype == "attribute":
        attributes = proposition.get("attributes") or []
        value = str(attributes[0].get("value_vi")) if attributes else ""
        if not value:
            return None
        if attributes[0].get("kind") == "màu_sắc":
            # `value_vi` holds the colour, not the phrase: `đen và nâu`. Each
            # renderer supplies its own `màu`, because realize.py needs one too
            # and storing it in the value gave `màu màu đen và nâu`.
            hue = value if value.lower().startswith("màu") else f"màu {value}"
            return (
                f"Trong ảnh, {subject} có {hue} không",
                f"Có đúng là {subject} KHÔNG có {hue} không",
            )
        return (
            f"Trong ảnh, {subject} có {value} không",
            f"Có đúng là {subject} KHÔNG {value} không",
        )

    if ptype == "action":
        action = _strip_aspect((proposition.get("predicate") or {}).get("lemma_vi", ""))
        if not action:
            return None
        return (
            f"Trong ảnh, {subject} có đang {action} không",
            f"Có đúng là {subject} KHÔNG {action} không",
        )

    if ptype in ("relation", "interaction"):
        predicate = _strip_aspect((proposition.get("predicate") or {}).get("lemma_vi", ""))
        obj = _object_text(proposition)
        if not predicate or not obj:
            return None
        return (
            f"Trong ảnh, {subject} có {predicate} {obj} không",
            f"Có đúng là {subject} KHÔNG {predicate} {obj} không",
        )

    if ptype == "spatial_relation":
        relation = normalise_relation(
            (proposition.get("spatial_relation") or {}).get("relation_vi")
        ).replace("_", " ")
        obj = _object_text(proposition)
        if not relation or not obj:
            return None
        return (
            f"Trong ảnh, {subject} có ở {relation} {obj} không",
            f"Có đúng là {subject} KHÔNG ở {relation} {obj} không",
        )

    if ptype == "counting":
        count = proposition.get("count") or {}
        value = count.get("value")
        if value is None:
            return None
        phrase = " ".join(
            part
            for part in (str(value), count.get("classifier"), count.get("entity_category_vi"))
            if part
        )
        return (
            f"Trong ảnh có đúng {phrase} không",
            f"Có đúng là trong ảnh KHÔNG có {phrase} không",
        )

    if ptype == "scene":
        scene = proposition.get("scene") or {}
        place = scene.get("place_vi") or scene.get("activity_vi") or proposition.get("text_vi")
        if not place:
            return None
        return (
            f"Bối cảnh trong ảnh có phải là {place} không",
            f"Có đúng là bối cảnh trong ảnh KHÔNG phải là {place} không",
        )

    return None


@dataclass
class ProbeOutcome:
    direct: YesNo | None = None
    negated: YesNo | None = None
    acquiescent: bool = False
    #: The two colour verifiers gave opposite answers. V is then ⊥: a disputed
    #: channel has produced no usable evidence, the same treatment acquiescence
    #: gets, and for the same reason.
    colour_disputed: bool = False
    failed: bool = False
    lang_fallback: bool = False
    records: list[dict[str, Any]] = field(default_factory=list)


def run_probe(
    model: VLM,
    image: Any,
    question: str,
    negated: str | None,
    config: VerificationConfig,
    stats: VerificationStats,
) -> ProbeOutcome:
    """Ask the claim, and its negation when contradiction detection is on.

    Retries twice with backoff, then gives up and reports failure rather than
    substituting a value (formulation/09 §7). A failed probe scores the channel
    ⊥ and drops E, which lands the proposition in UNCERTAIN — the correct
    outcome for "we did not manage to look".
    """
    outcome = ProbeOutcome()

    def ask(text: str) -> YesNo | None:
        prompt = text
        for attempt in range(config.probe_retries + 1):
            try:
                answer = model.probe_yes_no(image, prompt, k=config.probe_k)
            except Exception:
                if attempt >= config.probe_retries:
                    return None
                time.sleep(config.retry_backoff_s * (attempt + 1))
                continue
            stats.probe_calls += 1
            if _looks_vietnamese(answer.answer.text) or attempt >= config.probe_retries:
                if not _looks_vietnamese(answer.answer.text):
                    # Accepted, but flagged: a verifier answering in English was
                    # not asked the question we think we asked. Counted once per
                    # proposition by the caller, not once per retry.
                    outcome.lang_fallback = True
                return answer
            prompt = f"{_STRICTER_VI} {text}"
        return None

    outcome.direct = ask(question)
    if outcome.direct is None:
        outcome.failed = True
        stats.probe_failures += 1
        return outcome
    outcome.records.append(_probe_record(question, outcome.direct, model.name))

    if config.contradiction_detection and negated:
        # formulation/09 §3.3(b). probe_with_negation is not called directly
        # because it has no retry path and we need the failure handling above.
        outcome.negated = ask(negated)
        if outcome.negated is not None:
            outcome.records.append(_probe_record(negated, outcome.negated, model.name))
            outcome.acquiescent = outcome.direct.affirms and outcome.negated.affirms
            if outcome.acquiescent:
                stats.acquiescence += 1
    return outcome


def _probe_record(question: str, answer: YesNo, model_name: str) -> dict[str, Any]:
    """`Evidence.probes[]` — recorded verbatim so a verdict is auditable."""
    return {
        "question_vi": question,
        "answer": answer.answer.text,
        "answer_prob": round(float(answer.answer.confidence), 4),
        "polarity": answer.polarity.value,
        # Recorded so a run can be re-scored from disk without the GPU. Without
        # them `_answer_stability` sees nothing to disagree with and a replay
        # silently scores every probe as perfectly stable.
        "samples": list(answer.answer.samples or []),
        "model": model_name,
    }


def visual_support(probe: YesNo | None) -> float | None:
    """V from a yes/no probe. ⊥ when the model could not tell.

    An affirmation with no confidence estimate lands at 0.5, not 1.0: a
    backbone that exposes neither token probabilities nor self-consistency has
    told us what it thinks, not how sure it is, and treating that as full
    support would let an uncalibrated model manufacture SUPPORTED verdicts.

    An INCONCLUSIVE answer returns ⊥ rather than 0. Scoring it 0 would feed
    "the model could not tell" into S as "no support", and S ≤ θ_S^lo rejects —
    the exact collapse of UNCERTAIN into REJECTED this module exists to prevent.

    >>> visual_support(None) is None
    True
    """
    if probe is None or probe.polarity is Polarity.INCONCLUSIVE:
        return None
    confidence = max(0.0, min(1.0, float(probe.confidence)))
    return 0.5 + 0.5 * confidence if probe.affirms else 0.5 - 0.5 * confidence


def _probe_verdict(outcome: ProbeOutcome) -> bool | None:
    """What the model actually claimed, or None if it never committed.

    Reads the negated probe when the direct one was inconclusive: a model that
    affirms `KHÔNG ở bên trái` has committed to "no", even though it declined to
    answer the positive form.
    """
    if outcome.direct is not None and outcome.direct.polarity is not Polarity.INCONCLUSIVE:
        return outcome.direct.affirms
    if outcome.negated is not None and outcome.negated.polarity is not Polarity.INCONCLUSIVE:
        return not outcome.negated.affirms
    return None


def negation_contradiction(outcome: ProbeOutcome, config: VerificationConfig) -> tuple[float, str]:
    """C from channel 9(b). Returns `(score, reason_key)`."""
    if not config.contradiction_detection or outcome.negated is None:
        return 0.0, ""
    if outcome.acquiescent:
        # The model agreed with the question, not with the image. That is a
        # failure to discriminate, so it raises C only into the UNCERTAIN band.
        return config.acquiescence_c, "acquiescence"
    if outcome.negated.affirms and outcome.direct is not None and not outcome.direct.affirms:
        # The model affirmed the negation and refused the claim: this is the
        # image speaking against the proposition.
        return max(0.0, min(1.0, float(outcome.negated.confidence))), "negation"
    return 0.0, ""


# ---------------------------------------------------------------------------
# The `xanh` guard (formulation/02 §4.5, formulation/09 §5)
# ---------------------------------------------------------------------------

XANH_QUESTION = (
    "Trong ảnh, {subject} có màu xanh dương (như bầu trời) hay xanh lá (như lá cây)? "
    "Chỉ trả lời 'xanh dương' hoặc 'xanh lá'. Nếu không nhìn rõ, trả lời 'không rõ'."
)


def resolve_xanh(
    model: VLM,
    image: Any,
    proposition: dict,
    attribute: dict,
    config: VerificationConfig,
    stats: VerificationStats,
) -> bool:
    """Try to resolve a bare `xanh` from the image. True when resolved.

    Bare `xanh` is BOTH blue and green in Vietnamese and is never resolved
    silently: an unresolved attribute is capped at UNCERTAIN and recorded as
    `xanh_không_xác_định`, which evaluation scores as under-specification, not
    as a colour hallucination (formulation/02 §4.5 rule 4).

    A forced-choice answer is accepted only when it parses to blue or green;
    anything else leaves the attribute unresolved. `resolution_source` is set to
    `pixel_evidence` because schema v1.0.0's enum has no `vlm_probe` member —
    here it means "resolved from the image by the verifier", not by an
    annotator, and that reading needs a schema note.
    """
    disambiguation = attribute.setdefault(
        "color_disambiguation",
        {"raw": attribute.get("value_vi", ""), "resolved": Xanh.UNRESOLVED.value,
         "resolution_source": "unresolved"},
    )
    question = XANH_QUESTION.format(subject=_subject_text(proposition))
    try:
        answer = model.probe(image, question, k=config.probe_k, max_new_tokens=16)
    except Exception:
        stats.probe_failures += 1
        stats.xanh_unresolved += 1
        return False
    stats.probe_calls += 1
    reading = parse_color(answer.text.strip().strip(".,!?"))
    if reading.xanh_value in (Xanh.BLUE, Xanh.GREEN):
        disambiguation["resolved"] = reading.xanh_value.value
        disambiguation["resolution_source"] = "pixel_evidence"
        return True
    stats.xanh_unresolved += 1
    return False


# ---------------------------------------------------------------------------
# Channel 9(a) — cross-proposition contradiction (formulation/09 §3.3)
# ---------------------------------------------------------------------------


@dataclass
class ContradictionGraph:
    """Which propositions are incompatible, and why.

    Undirected: incompatibility says the pair cannot both be true, not which of
    the two is false. Deciding that is the job of the image, not of this graph.
    """

    edges: dict[str, dict[str, str]] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def add(self, a: str, b: str, reason: str) -> None:
        self.edges.setdefault(a, {})[b] = reason
        self.edges.setdefault(b, {})[a] = reason

    def contradictors(self, pid: str) -> dict[str, str]:
        return self.edges.get(pid, {})

    @property
    def pair_count(self) -> int:
        return sum(len(v) for v in self.edges.values()) // 2

    def cycles(self) -> list[list[str]]:
        """Connected components that contain a cycle (formulation/09 §7).

        For an undirected component, a cycle exists iff edges ≥ nodes. A→B→C→A
        cannot be resolved by dropping one member, so the caller keeps the
        best-supported one and caps the rest at UNCERTAIN.
        """
        seen: set[str] = set()
        found: list[list[str]] = []
        for start in self.edges:
            if start in seen:
                continue
            stack, component = [start], []
            seen.add(start)
            while stack:
                node = stack.pop()
                component.append(node)
                for neighbour in self.edges.get(node, {}):
                    if neighbour not in seen:
                        seen.add(neighbour)
                        stack.append(neighbour)
            edge_count = sum(len(self.edges.get(n, {})) for n in component) // 2
            if edge_count >= len(component) and len(component) >= 3:
                found.append(sorted(component))
        return found


def _subject_id(proposition: dict) -> str | None:
    return (proposition.get("subject") or {}).get("entity_id")


def _object_id(proposition: dict) -> str | None:
    return (proposition.get("object") or {}).get("entity_id")


# Attribute families in which one target can hold exactly one value, so two
# different values are a contradiction. `trạng_thái`, `chất_liệu` and `hoa_văn`
# are deliberately absent: an object can be simultaneously `đang mở` and
# `bị ướt`, and treating every pair of states as incompatible would manufacture
# contradictions out of two compatible facts. These are schema enum members
# (Attribute.kind), not new vocabulary.
SINGLE_VALUED_ATTRIBUTE_KINDS = frozenset(
    {"màu_sắc", "kích_thước", "độ_tuổi", "hình_dạng", "tư_thế"}
)


def _colour_term(value: str) -> str:
    """Thin wrapper over `vi.color.colour_term`, returning "" instead of None.

    The empty string is what this module's callers already test for; the shared
    function returns None, and `metrics_svp` tested for that. Both spellings of
    "no colour here" now come from one place.

    >>> _colour_term("áo xanh dương")
    'xanh dương'
    >>> _colour_term("quần bò")
    ''
    """
    return colour_term(value, canonicalise=canonical) or ""


def _attribute_target(value: str) -> str:
    """What the attribute value is *about*, with the colour term removed.

    `áo đỏ` and `áo xanh dương` share the target `áo` and cannot both hold.
    `áo đỏ` and `quần xanh lá` do not, and flagging them would invent a
    contradiction out of two compatible facts about different garments — the
    single most likely false positive in this channel.

    >>> _attribute_target("áo đỏ"), _attribute_target("áo xanh dương")
    ('áo', 'áo')
    >>> _attribute_target("quần xanh lá")
    'quần'
    """
    text = canonical(value)
    term = _colour_term(text)
    if term:
        text = text.replace(term, " ")
    return " ".join(t for t in text.split() if t not in COLOR_MODIFIERS and t != "màu")


def _colours_conflict(a: str, b: str) -> bool:
    """True only for a genuine colour conflict.

    Bare `xanh` against a resolved value is under-specification, not a
    contradiction: the model said something true but vague, and merging the two
    would misattribute a vagueness problem to hallucination (vi/color.py).

    >>> _colours_conflict("áo đỏ", "áo xanh dương")
    True
    >>> _colours_conflict("áo đỏ", "áo xanh")
    False
    """
    pa, pb = parse_color(_colour_term(a)), parse_color(_colour_term(b))
    if not pa.resolved or not pb.resolved:
        return False
    if pa.is_xanh and pb.is_xanh:
        return pa.xanh_value is not pb.xanh_value
    if pa.is_xanh != pb.is_xanh:
        return True
    return pa.canonical != pb.canonical


def detect_contradictions(
    propositions: Sequence[dict],
    entities: Sequence[dict],
) -> ContradictionGraph:
    """Channel 9(a): cross-proposition contradiction (formulation/09 §3.3a).

    Detects the five listed families. Where a family cannot be decided without
    inventing vocabulary, it under-detects rather than guessing — the same
    policy vi/lexicon.py states for its word lists, and for the same reason: a
    missed contradiction lowers recall, a fabricated one asserts something
    false.
    """
    graph = ContradictionGraph()
    by_id = {str(p.get("id")): p for p in propositions}
    entity_group = {
        str(e.get("id")): str(e.get("coreference_group") or e.get("id")) for e in entities
    }

    def group_of(pid: str) -> str | None:
        entity_id = _subject_id(by_id[pid])
        if entity_id is None:
            return None
        return entity_group.get(entity_id, entity_id)

    ids = [str(p.get("id")) for p in propositions]

    for i, a_id in enumerate(ids):
        a = by_id[a_id]
        a_type = str(a.get("type"))
        for b_id in ids[i + 1 :]:
            b = by_id[b_id]
            b_type = str(b.get("type"))
            same_entity = (
                group_of(a_id) is not None and group_of(a_id) == group_of(b_id)
            )

            # (1) mutually exclusive categories on one entity
            if a_type == b_type == "entity" and same_entity:
                a_noun = (a.get("subject") or {}).get("head_noun_vi") or ""
                b_noun = (b.get("subject") or {}).get("head_noun_vi") or ""
                if a_noun and b_noun and not compatible_categories(a_noun, b_noun):
                    graph.add(a_id, b_id, f"phạm trù loại trừ nhau: {a_noun} / {b_noun}")
                    continue

            # (2) mutually exclusive attribute values in one family
            if a_type == b_type == "attribute" and same_entity:
                if _attributes_conflict(a, b):
                    graph.add(a_id, b_id, "hai giá trị thuộc tính loại trừ nhau")
                    continue

            # (3) count conflicts
            if a_type == b_type == "counting":
                if _counts_conflict(a, b):
                    graph.add(
                        a_id,
                        b_id,
                        f"số lượng mâu thuẫn: {(a.get('count') or {}).get('value')} / "
                        f"{(b.get('count') or {}).get('value')}",
                    )
                    continue

            # (4) asymmetric spatial conflicts
            if a_type == b_type == "spatial_relation":
                reason = _spatial_conflict(a, b)
                if reason:
                    graph.add(a_id, b_id, reason)
                    continue

            # (5) interaction asymmetry, explicit denial only
            if {a_type, b_type} <= {"relation", "interaction"}:
                reason = _interaction_conflict(a, b)
                if reason:
                    graph.add(a_id, b_id, reason)

    # An interaction whose converse is simply absent is NOT contradictory:
    # absence of evidence is not evidence of absence, and generate.py builds
    # entity pairs one way only, so the converse is structurally never proposed.
    # Recorded as a note so the asymmetry is visible without scoring it.
    for pid, p in by_id.items():
        if str(p.get("type")) == "interaction" and _object_id(p) and not graph.contradictors(pid):
            graph.notes.append(f"{pid}: không có mệnh đề nghịch đảo — không tính là mâu thuẫn")
    return graph


def _attributes_conflict(a: dict, b: dict) -> bool:
    for pa in a.get("attributes") or []:
        for pb in b.get("attributes") or []:
            kind = pa.get("kind")
            if kind != pb.get("kind") or kind not in SINGLE_VALUED_ATTRIBUTE_KINDS:
                continue
            va, vb = str(pa.get("value_vi", "")), str(pb.get("value_vi", ""))
            if not va or not vb or canonical(va) == canonical(vb):
                continue
            if kind == "màu_sắc":
                # Colour values carry the garment they describe, so the two must
                # be about the same part before they can conflict.
                if _attribute_target(va) == _attribute_target(vb) and _colours_conflict(va, vb):
                    return True
                continue
            return True
    return False


def _counts_conflict(a: dict, b: dict) -> bool:
    ca, cb = a.get("count") or {}, b.get("count") or {}
    va, vb = ca.get("value"), cb.get("value")
    if va is None or vb is None:
        return False
    if not compatible_categories(
        str(ca.get("entity_category_vi", "")), str(cb.get("entity_category_vi", ""))
    ):
        return False
    tolerance = max(int(ca.get("tolerance", 0) or 0), int(cb.get("tolerance", 0) or 0))
    return abs(int(va) - int(vb)) > tolerance


def _spatial_conflict(a: dict, b: dict) -> str | None:
    ra = normalise_relation((a.get("spatial_relation") or {}).get("relation_vi"))
    rb = normalise_relation((b.get("spatial_relation") or {}).get("relation_vi"))
    if not ra or not rb:
        return None
    fa = (a.get("spatial_relation") or {}).get("frame_of_reference", "image_relative")
    fb = (b.get("spatial_relation") or {}).get("frame_of_reference", "image_relative")
    if fa != fb:
        # `bên trái` viewer-relative and object-relative describe different
        # configurations, so they cannot contradict each other.
        return None
    sa, oa, sb, ob = _subject_id(a), _object_id(a), _subject_id(b), _object_id(b)
    if None in (sa, oa, sb, ob):
        return None
    if sa == sb and oa == ob and SPATIAL_INVERSE.get(ra) == rb:
        return f"quan hệ nghịch đảo trên cùng một cặp: {ra} / {rb}"
    if sa == ob and oa == sb and ra == rb and ra in ASYMMETRIC_RELATIONS:
        return f"quan hệ bất đối xứng theo cả hai chiều: {ra}"
    return None


def _interaction_conflict(a: dict, b: dict) -> str | None:
    """Interaction asymmetry, but only when the converse is EXPLICITLY denied.

    A missing converse is not a contradiction. generate.py enumerates entity
    pairs one way only (`for i, a in … for b in entities[i+1:]`), so the
    converse of an interaction is structurally never proposed; scoring its
    absence would fire on every interaction proposition and would break the
    project's own rule that "cannot determine" is never "no".
    """
    sa, oa = _subject_id(a), _object_id(a)
    sb, ob = _subject_id(b), _object_id(b)
    if sa is None or oa is None or sb is None or ob is None:
        return None
    same_pair = (sa, oa) == (sb, ob)
    converse_pair = (sa, oa) == (ob, sb)
    if not (same_pair or converse_pair):
        return None
    predicate_a = (a.get("predicate") or {}).get("lemma_vi", "")
    predicate_b = (b.get("predicate") or {}).get("lemma_vi", "")
    if not predicate_a or canonical(predicate_a) != canonical(predicate_b):
        return None
    polarity_a = (a.get("predicate") or {}).get("polarity", "positive")
    polarity_b = (b.get("predicate") or {}).get("polarity", "positive")
    if polarity_a != polarity_b:
        return f"một bên phủ định tương tác {predicate_a!r}"
    return None


def semantic_consistency(
    proposition: dict,
    supporting: Sequence[str],
    contradicting: Sequence[str],
) -> float:
    """M — coherence with the rest of the proposition set (channel 9a, support side).

    Laplace-smoothed so that a proposition with no related propositions scores
    0.5. Zero would penalise an isolated true claim for being isolated; one
    would hand it support it never earned.

    >>> semantic_consistency({}, [], [])
    0.5
    """
    n_sup, n_con = len(supporting), len(contradicting)
    return (1.0 + n_sup) / (2.0 + n_sup + n_con)


# ---------------------------------------------------------------------------
# The decision rule (formulation/09 §4)
# ---------------------------------------------------------------------------


@dataclass
class HardConstraint:
    """The epistemic ceiling of formulation/09 §4 line 2. Not a score."""

    inference_type: str

    @property
    def reason_vi(self) -> str:
        kind = INFERENCE_TYPE_VI.get(self.inference_type, self.inference_type)
        # The wording is fixed by formulation/09 §5.1: "không thể xác định trực
        # tiếp từ ảnh", never "sai". Saying the claim is false would make the
        # system's own explanation a false claim.
        return (
            f"Mệnh đề này nói về {kind}, không thể xác định trực tiếp từ ảnh. "
            "Đây là bác bỏ với tư cách một sự kiện thị giác, không phải khẳng định rằng mệnh đề sai."
        )

    @property
    def reason_en(self) -> str:
        kind = INFERENCE_TYPE_EN.get(self.inference_type, self.inference_type)
        return (
            f"This proposition is about {kind}, which cannot be determined directly "
            "from the image. Rejected as a visual fact, not asserted to be false."
        )


def hard_constraint(proposition: dict) -> HardConstraint | None:
    """Step 1 of §4: SPECULATION-class claims never reach scoring.

    **Both conditions must hold.** `requires_inference` alone is not enough:
    pipeline/generate.py tags a gendered noun as INFERENCE with
    `inference_type = "none"`, and firing here on that would reject every
    `một người đàn ông` with "cannot be determined from the image". Those route
    to the UNCERTAIN ceiling instead (formulation/08 §2).

    >>> hard_constraint({"evidence": {"external_knowledge": {
    ...     "requires_inference": True, "inference_type": "purpose"}}}).inference_type
    'purpose'
    >>> hard_constraint({"evidence": {"external_knowledge": {
    ...     "requires_inference": True, "inference_type": "none"}}}) is None
    True
    """
    external = ((proposition.get("evidence") or {}).get("external_knowledge") or {})
    if not external.get("requires_inference"):
        return None
    inference_type = str(external.get("inference_type", "none"))
    if inference_type not in SPECULATIVE_INFERENCE_TYPES:
        return None
    return HardConstraint(inference_type)


def decide(
    scores: Scores,
    thresholds: Thresholds,
    *,
    constraint: HardConstraint | None = None,
    cascade_from: Sequence[Cascade] = (),
) -> tuple[Verdict, str]:
    """The decision rule of formulation/09 §4, in the specified order.

    The order is the rule. Moving the E gate below the C/S tests would let
    "we could not tell" be reported as "it is false"; moving the hard constraint
    below scoring would let a confident model promote a purpose claim to a
    visual fact.

    Returns `(verdict, branch)`; `branch` names which line fired, so the caller
    can attach a cause rather than restating the verdict.

    >>> t = Thresholds()
    >>> decide(Scores(s=0.20, c=0.80, e=0.90), t)      # image shows the opposite
    (<Verdict.REJECTED: 'REJECTED'>, 'contradiction')
    >>> decide(Scores(s=0.20, c=0.10, e=0.20), t)      # we could not tell
    (<Verdict.UNCERTAIN: 'UNCERTAIN'>, 'gate')
    >>> decide(Scores(s=0.80, c=0.50, e=0.90), t)      # strong evidence BOTH ways
    (<Verdict.UNCERTAIN: 'UNCERTAIN'>, 'unclear')
    >>> decide(Scores(s=0.80, c=0.10, e=0.90), t)
    (<Verdict.SUPPORTED: 'SUPPORTED'>, 'support')
    >>> decide(Scores(s=0.99, c=0.0, e=0.99), t, constraint=HardConstraint("emotion"))
    (<Verdict.REJECTED: 'REJECTED'>, 'hard_constraint')
    """
    # 1. HARD CONSTRAINT — before any score is looked at.
    if constraint is not None:
        return Verdict.REJECTED, "hard_constraint"

    # 2. PRECONDITION — existence cascade.
    if cascade_from:
        return Verdict.REJECTED, "cascade"

    # 3. GATE — insufficient evidence cannot yield a confident verdict.
    #    E is None when nothing could be measured, which is the strongest
    #    possible form of "not enough evidence".
    if scores.e is None or scores.e < thresholds.E:
        return Verdict.UNCERTAIN, "gate"

    # 4. DECISION. S is None when every support component was ⊥.
    if scores.c >= thresholds.C_hi:
        return Verdict.REJECTED, "contradiction"
    if scores.s is None:
        return Verdict.UNCERTAIN, "no_support_signal"
    if scores.s >= thresholds.S_hi and scores.c <= thresholds.C_lo:
        return Verdict.SUPPORTED, "support"
    if scores.s <= thresholds.S_lo:
        return Verdict.REJECTED, "no_support"
    return Verdict.UNCERTAIN, "unclear"


def apply_ceilings(verdict: Verdict, ceilings: Sequence[Ceiling]) -> Verdict:
    """Clamp a verdict down to the strictest ceiling. Never up.

    >>> apply_ceilings(Verdict.SUPPORTED, [Ceiling(Verdict.UNCERTAIN, "", "")])
    <Verdict.UNCERTAIN: 'UNCERTAIN'>
    >>> apply_ceilings(Verdict.REJECTED, [Ceiling(Verdict.UNCERTAIN, "", "")])
    <Verdict.REJECTED: 'REJECTED'>
    """
    for ceiling in ceilings:
        if _RANK[ceiling.verdict] < _RANK[verdict]:
            verdict = ceiling.verdict
    return verdict


# ---------------------------------------------------------------------------
# Explanations (formulation/09 §6)
# ---------------------------------------------------------------------------
# Every explanation must name the REASON, not restate the verdict: "không đủ
# bằng chứng" with no cause is useless for error analysis.

_BRANCH_VI: dict[str, str] = {
    "contradiction": "Có bằng chứng phủ định",
    "support": "Được hỗ trợ bởi bằng chứng thị giác",
    "no_support": "Không có bằng chứng hỗ trợ trong ảnh",
    "no_support_signal": "Không thu được tín hiệu hỗ trợ nào",
    "unclear": "Bằng chứng không rõ ràng",
}

_BRANCH_EN: dict[str, str] = {
    "contradiction": "There is evidence against this",
    "support": "Supported by visual evidence",
    "no_support": "No supporting evidence in the image",
    "no_support_signal": "No support signal could be obtained",
    "unclear": "The evidence is inconclusive",
}


def _cause_vi(
    branch: str,
    limiting: EvidenceSignal | None,
    contradiction_reason: str,
    geometric: GeometricCheck | None,
    contradictors: dict[str, str],
) -> tuple[str, str]:
    """The cause clause for a verdict, Vietnamese and English."""
    if branch == "gate":
        if limiting is None:
            return (
                "không đo được bất kỳ tín hiệu bằng chứng nào",
                "no evidence signal could be measured at all",
            )
        return limiting.low_means_vi or limiting.name, limiting.low_means_en or limiting.name

    if branch == "contradiction":
        if contradiction_reason == "geometry" and geometric is not None:
            return (
                f"kiểm tra hình học từ hộp giới hạn bác bỏ quan hệ này ({geometric.reason_vi})",
                f"the model-free geometric test refutes it ({geometric.reason_en})",
            )
        if contradiction_reason == "negation":
            return (
                "mô hình khẳng định mệnh đề phủ định của nó",
                "the model affirmed the negation of the claim",
            )
        if contradiction_reason == "acquiescence":
            return (
                "mô hình khẳng định cả mệnh đề lẫn phủ định của nó",
                "the model affirmed both the claim and its negation",
            )
        if contradictors:
            pairs = ", ".join(f"{pid} ({why})" for pid, why in sorted(contradictors.items()))
            return (f"mâu thuẫn với {pairs}", f"contradicted by {pairs}")
        return "có bằng chứng ngược lại", "there is counter-evidence"

    if branch in ("unclear", "no_support", "no_support_signal"):
        if limiting is not None and limiting.value is not None and limiting.value < 0.6:
            return limiting.low_means_vi or limiting.name, limiting.low_means_en or limiting.name
        if contradictors:
            pairs = ", ".join(sorted(contradictors))
            return (
                f"vừa có hỗ trợ vừa có mâu thuẫn với {pairs}",
                f"both supported and contradicted by {pairs}",
            )
        return (
            "mức hỗ trợ thị giác nằm giữa hai ngưỡng quyết định",
            "visual support falls between the decision thresholds",
        )

    if branch == "support" and geometric is not None and geometric.consistent:
        return (
            f"kiểm tra hình học đồng thuận ({geometric.reason_vi})",
            f"the geometric test agrees ({geometric.reason_en})",
        )
    return "", ""


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def _entity_index(entities: Sequence[dict]) -> dict[str, dict]:
    return {str(e.get("id")): e for e in entities if e.get("id")}


def _evidence(proposition: dict) -> dict[str, Any]:
    """`Evidence`, with the sub-objects M2 does not create.

    pipeline/generate.py writes only `external_knowledge`, so the other channel
    records have to be created before they can be filled.
    """
    evidence = proposition.setdefault("evidence", {})
    evidence.setdefault("probes", [])
    evidence.setdefault("visual_grounding", {})
    evidence.setdefault("geometric", {"checks_passed": [], "checks_failed": []})
    evidence.setdefault("cross_proposition", {"supporting_ids": [], "contradicting_ids": []})
    return evidence


def verify(
    model: VLM,
    image: Any,
    entities: Sequence[dict],
    propositions: Sequence[dict],
    config: VerificationConfig | None = None,
    colour_verifier: VLM | None = None,
) -> tuple[list[VerificationResult], VerificationStats]:
    """Run M4 over one image document.

    Mutates each proposition in place — `verification`, `contradicts` and the
    `evidence.*` channels — and returns the richer results plus statistics.

    `config = None` builds the formulation/07 §9 defaults, whose version string
    says UNTUNED and which are recorded in `stats.notes`. That is not the
    silent fallback formulation/09 §7 forbids: the forbidden case is a config
    file that omitted its threshold block, and `VerificationConfig.from_dict`
    raises on exactly that.

    Four passes, and the split matters:

    1. image-only scores (V, G, E) — order-independent;
    2. the contradiction graph — needs no scores;
    3. verdicts in dependency order (existence → attribute/action →
       relation/spatial → scene) so the cascade and M see settled verdicts;
    4. cycle resolution and ceilings, which only ever lower a verdict.

    Computing C from a contradictor's *verdict* instead of its pass-1 V would
    make a symmetric relation asymmetric: whichever proposition was verified
    second would see a settled opponent and the first would not.
    """
    config = config or VerificationConfig()

    # A backbone with no token probabilities cannot produce a confidence at
    # k=1, so V pins to the neutral 0.5 for every probe and NOTHING can reach
    # SUPPORTED -- the three-way verdict collapses to all-UNCERTAIN. Measured
    # on a real image: 64 propositions, 64 UNCERTAIN, and a caption built from
    # the one proposition selection could still find.
    #
    # `probe_k` is therefore resolved from the backbone rather than left at a
    # default the backbone cannot honour. This is not the silent fallback §7
    # forbids: the substitution is recorded in `stats.notes` and travels with
    # the results. An explicit k > 1 from the caller is always respected.
    blind_confidence = not getattr(model, "supports_logprobs", True)
    auto_k = blind_confidence and config.probe_k <= 1
    if auto_k:
        config = replace(config, probe_k=SELF_CONSISTENCY_K)

    stats = VerificationStats(mode=config.mode, n_propositions=len(propositions))
    if blind_confidence:
        stats.notes.append(
            f"{model.name} không cung cấp xác suất token — độ tin cậy lấy bằng "
            f"tự nhất quán, k={config.probe_k}"
            f"{' (tự chọn)' if auto_k else ' (do cấu hình)'} "
            f"(formulation/08 §8.2). Chi phí: gấp {config.probe_k}× số lượt probe."
        )
    if config.decision_rule_version.endswith("UNTUNED"):
        stats.notes.append(
            "bộ ngưỡng mặc định từ formulation/07 §9 CHƯA hiệu chỉnh trên tập dev"
        )
    # The geometric channel is named only when it actually ran: under A7 it is
    # off, and a results row whose `verifier` still claims "+ geometric" would
    # attribute the ablation's numbers to a channel that never executed.
    verifier = config.verifier or (
        f"{model.name} + geometric" if config.spatial_verification else model.name
    )
    rule_version = config.resolved_decision_rule_version()
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    index = _entity_index(entities)
    stats.entities_without_bbox = sum(
        1 for e in entities if as_box(e.get("bbox")) is None
    )
    if entities and stats.entities_without_bbox == len(entities):
        stats.notes.append(
            "không đối tượng nào có hộp giới hạn: kênh hình học (§3.2) không chạy được"
        )

    # ---- A1: verification disabled ---------------------------------------
    if config.mode == Mode.NONE:
        stats.verification_disabled = True
        stats.notes.append(
            "A1: kiểm chứng bị tắt — status để None, KHÔNG gán SUPPORTED "
            "(gán một phán quyết chưa từng được tính là bịa dữ liệu)"
        )
        results = []
        for proposition in propositions:
            result = VerificationResult(
                proposition_id=str(proposition.get("id")),
                status=None,
                branch="disabled",
                scores=Scores(),
                explanation_vi=(
                    "Kiểm chứng bị tắt trong cấu hình này (A1 / Baseline B); "
                    "mệnh đề chưa được đối chiếu với ảnh."
                ),
                explanation_en=(
                    "Verification is disabled in this configuration (A1 / Baseline B); "
                    "this proposition was never checked against the image."
                ),
                verifier=verifier,
                decision_rule_version=rule_version,
                verified_at=now,
            )
            proposition["verification"] = result.to_schema()
            results.append(result)
            stats.by_status["None"] = stats.by_status.get("None", 0) + 1
        return results, stats

    # ---- pass 1: image-only scores ---------------------------------------
    probe_outcomes: dict[str, ProbeOutcome] = {}
    image_scores: dict[str, Scores] = {}
    geometry: dict[str, GeometricCheck] = {}
    signals: dict[str, list[EvidenceSignal]] = {}
    limiting: dict[str, EvidenceSignal | None] = {}
    constraints: dict[str, HardConstraint | None] = {}
    ceilings: dict[str, list[Ceiling]] = {}
    size = _image_size(image)

    for proposition in propositions:
        pid = str(proposition.get("id"))
        constraints[pid] = hard_constraint(proposition)
        ceilings[pid] = []
        scores = Scores()

        if constraints[pid] is not None:
            # Step 1 returns before scoring; probing a purpose claim would spend
            # budget on a question the image cannot answer in principle.
            image_scores[pid] = scores
            signals[pid], limiting[pid] = [], None
            continue

        ceilings[pid].extend(_epistemic_ceilings(proposition, index))
        xanh_unresolved = _handle_xanh(proposition, model, image, config, stats, ceilings[pid])

        questions = build_questions(proposition)
        outcome = ProbeOutcome()
        if questions is not None:
            outcome = run_probe(model, image, questions[0], questions[1], config, stats)

            # Colour goes to a second, independent verifier. Measured 17/08:
            # Vintern-1B scores 0.727 precision on colour propositions against
            # Qwen2.5-VL's 1.000, and every one of its errors falls on the
            # blue/green pair -- the Vietnamese verifier merges the same two
            # hues this work is about. Under distillation its verdicts become
            # training labels, so a colour error is not one bad caption but the
            # confusion taught at scale on the paper's central axis.
            #
            # Disagreement forces UNCERTAIN rather than picking a winner. We
            # know which model is more accurate on synthetic discs; we do not
            # know it for every real object, and choosing on that basis would
            # be asserting a colour on the strength of one small experiment.
            # UNCERTAIN still reaches the caption through hedging, so the
            # information is kept rather than discarded.
            if colour_verifier is not None and is_colour_proposition(proposition):
                stats.colour_cross_checked += 1
                second = run_probe(
                    colour_verifier, image, questions[0], questions[1], config, stats
                )
                outcome.records.extend(second.records)
                first_p = outcome.direct.polarity if outcome.direct else None
                second_p = second.direct.polarity if second.direct else None
                if first_p is not None and second_p is not None and first_p is not second_p:
                    stats.colour_disagreements += 1
                    outcome.colour_disputed = True
                    ceilings[pid].append(
                        Ceiling(
                            Verdict.UNCERTAIN,
                            f"hai bộ kiểm chứng bất đồng về màu "
                            f"({model.name}: {first_p.value}, "
                            f"{colour_verifier.name}: {second_p.value})",
                            f"colour verifiers disagree ({model.name}: "
                            f"{first_p.value}, {colour_verifier.name}: {second_p.value})",
                        )
                    )
                elif second_p is None:
                    stats.notes.append(
                        f"{pid}: bộ kiểm chứng màu thứ hai không trả lời được — "
                        f"chỉ còn một nguồn cho mệnh đề màu"
                    )
        probe_outcomes[pid] = outcome
        # Acquiescence invalidates the answer rather than capping the verdict.
        # A model that affirms both a claim and its negation has produced no
        # information, so V is ⊥ and the probe channel contributes nothing to E
        # — which sends the proposition to UNCERTAIN through the gate, with the
        # bias named as the cause. A blanket ceiling would have been the cruder
        # tool: it would also have overridden a decisive geometric test, and
        # formulation/09 §3.2 gives the model-free channel priority precisely
        # because it does not depend on the model behaving.
        scores.v = (
            None
            if outcome.acquiescent or outcome.colour_disputed
            else visual_support(outcome.direct)
        )

        # Channel 6 — geometry. A7 removes it entirely, INCLUDING the depth cap:
        # if the cap survived the ablation, the specific failure A7 predicts
        # (spatial claims decided by the VLM alone) would stay invisible.
        if proposition.get("type") == "spatial_relation" and config.spatial_verification:
            check = geometric_check(
                (proposition.get("spatial_relation") or {}).get("relation_vi", ""),
                as_box((index.get(_subject_id(proposition) or "") or {}).get("bbox")),
                as_box((index.get(_object_id(proposition) or "") or {}).get("bbox")),
                image_size=size,
                calibration=config.calibration,
            )
            geometry[pid] = check
            scores.g = check.score
            if check.consistent is None:
                ceilings[pid].append(
                    Ceiling(
                        Verdict.UNCERTAIN,
                        f"hình học không kết luận được: {check.reason_vi}",
                        f"geometry is inconclusive: {check.reason_en}",
                    )
                )
            else:
                stats.geometry_decided += 1

        signals[pid] = evidence_signals(
            proposition,
            index.get(_subject_id(proposition) or ""),
            image,
            entities,
            probe=outcome.direct,
            probe_failed=outcome.failed,
            calibration=config.calibration,
            geometric=geometry.get(pid),
            acquiescent=outcome.acquiescent,
            xanh_unresolved=xanh_unresolved,
        )
        scores.e, limiting[pid] = aggregate_evidence(signals[pid])
        image_scores[pid] = scores

    # ---- pass 2: contradiction graph -------------------------------------
    if config.contradiction_detection:
        graph = detect_contradictions(propositions, entities)
    else:
        # A4 disables channel 9 entirely — both the contradicting side (C) and
        # the supporting side (M). Leaving M at a neutral 0.5 would keep a
        # phantom fifth of the support weight alive; ⊥ redistributes it instead.
        graph = ContradictionGraph()
        stats.notes.append("A4: phát hiện mâu thuẫn bị tắt (kênh 9a và 9b)")
    stats.contradiction_pairs = graph.pair_count
    stats.notes.extend(graph.notes)

    # ---- pass 3: verdicts, in dependency order ---------------------------
    order = sorted(
        range(len(propositions)),
        key=lambda i: (TYPE_STAGE.get(str(propositions[i].get("type")), 5), i),
    )
    results_by_id: dict[str, VerificationResult] = {}
    entity_verdict: dict[str, Verdict] = {}

    for i in order:
        proposition = propositions[i]
        pid = str(proposition.get("id"))
        scores = image_scores[pid]
        outcome = probe_outcomes.get(pid, ProbeOutcome())
        constraint = constraints[pid]
        contradictors = graph.contradictors(pid) if constraint is None else {}
        check = geometry.get(pid)
        supporting: list[str] = []
        contradiction_reason: tuple[float, str] = (0.0, "")
        disagreement = False

        # A hard-constrained proposition is never scored: §4 returns at line 6,
        # before any component is computed. Publishing an S for it would attach a
        # measurement to a claim nothing measured.
        if constraint is None:
            # M and C from channel 9(a).
            if config.contradiction_detection:
                subject = _subject_id(proposition)
                for other in propositions:
                    other_id = str(other.get("id"))
                    if other_id == pid or other_id in contradictors:
                        continue
                    settled = results_by_id.get(other_id)
                    if subject and _subject_id(other) == subject and settled is not None:
                        if settled.status is Verdict.SUPPORTED:
                            supporting.append(other_id)
                # Entailed propositions count as support only once they are
                # themselves SUPPORTED — the same rule applied to c_cross above.
                # An unverified entailment is a claim, not evidence.
                for entailed in proposition.get("entails") or []:
                    settled = results_by_id.get(str(entailed))
                    if str(entailed) != pid and settled is not None:
                        if settled.status is Verdict.SUPPORTED:
                            supporting.append(str(entailed))
                scores.m = semantic_consistency(proposition, supporting, list(contradictors))

            c_cross = 0.0
            for other_id in contradictors:
                # The strength of a contradiction is how well the OTHER
                # proposition is grounded in the image (its pass-1 V), not its
                # generator confidence: generator confidence is a different
                # quantity by schema definition, and using it would import M2's
                # optimism into M4. Using the other's *verdict* would be worse
                # still — whichever of the pair was decided second would see a
                # settled opponent and the first would not, making a symmetric
                # relation asymmetric.
                other_v = image_scores.get(other_id, Scores()).v
                c_cross = max(c_cross, other_v if other_v is not None else 0.5)

            c_negation, negation_reason = negation_contradiction(outcome, config)

            if check is not None and check.consistent is not None:
                # GEOMETRY WINS for decidable relations (formulation/09 §3.2):
                # it is the only channel that cannot hallucinate, so on a
                # spatial claim it *determines* C — 1.0 when it refutes the
                # relation, 0.0 when it confirms it. Both directions matter. If
                # only the refuting direction were honoured, a VLM answering
                # "no" to a geometrically confirmed relation could still reject
                # it through C, and geometry would win only when convenient.
                # The model's disagreement is not discarded: it is counted as
                # the backbone-spatial-reasoning diagnostic §3.2 asks for.
                c_geometry = 0.0 if check.consistent else config.geometric_contradiction_c
                contradiction_reason = (c_geometry, "geometry")
                model_says = _probe_verdict(outcome)
                # Only a decisive model answer can *disagree*. Counting an
                # inconclusive probe as disagreement would inflate the very
                # diagnostic §3.2 wants — "the model could not tell" is not
                # "the model was wrong".
                if model_says is not None and model_says != check.consistent:
                    disagreement = True
                    stats.geometry_probe_disagreements += 1
            else:
                contradiction_reason = max(
                    (
                        (c_negation, negation_reason or "negation"),
                        (c_cross, "cross_proposition"),
                    ),
                    key=lambda pair: pair[0],
                )
            scores.c = max(0.0, min(1.0, contradiction_reason[0]))
            scores.combine(config.weights)

        # Cascade. Extended beyond §4's `subject_entity(p)` to the object as
        # well: a relation to an entity that does not exist is an inherited
        # error, and counting it as a primary relation hallucination is the
        # inflation doc 04 §3.2 exists to prevent.
        cascade: list[Cascade] = []
        roles = (
            ("subject", _subject_id(proposition)),
            ("object", _object_id(proposition)),
        )
        for role, entity_id in roles:
            if not entity_id or proposition.get("type") == "entity":
                continue
            if entity_verdict.get(entity_id) is Verdict.REJECTED:
                cascade.append(Cascade(entity_id, role, _existence_pid(propositions, entity_id)))
            elif entity_verdict.get(entity_id) is Verdict.UNCERTAIN:
                ceilings[pid].append(
                    Ceiling(
                        Verdict.UNCERTAIN,
                        f"sự tồn tại của {entity_id} chưa được xác nhận",
                        f"the existence of {entity_id} is itself unconfirmed",
                    )
                )

        thresholds = config.thresholds_for(str(proposition.get("type")))
        verdict, branch = decide(scores, thresholds, constraint=constraint, cascade_from=cascade)

        result = _build_result(
            proposition,
            verdict,
            branch,
            scores,
            constraint=constraint,
            cascade=cascade,
            ceilings=ceilings[pid],
            limiting=limiting.get(pid),
            contradictors=contradictors,
            supporting=supporting,
            geometric=check,
            outcome=outcome,
            disagreement=disagreement,
            contradiction_reason=contradiction_reason[1],
            signals=signals.get(pid, []),
            verifier=verifier,
            rule_version=rule_version,
            now=now,
        )
        # Ceilings known at this point are applied NOW, not in pass 4: the
        # cascade and M both read verdicts settled earlier in this same loop,
        # and a proposition that will end UNCERTAIN must not look SUPPORTED to
        # the propositions decided after it. Only the cycle ceiling, which needs
        # every S, is deferred.
        _clamp(result, ceilings[pid], stats)
        result.ceilings_applied = len(ceilings[pid])
        results_by_id[pid] = result
        if str(proposition.get("type")) == "entity":
            subject_id = _subject_id(proposition)
            if subject_id:
                entity_verdict[subject_id] = result.status or Verdict.UNCERTAIN

        if branch == "hard_constraint":
            stats.hard_constraint_rejections += 1
        elif branch == "cascade":
            stats.cascaded += 1
        elif branch == "gate":
            stats.gate_uncertain += 1
        if outcome.lang_fallback:
            stats.lang_fallbacks += 1

    # ---- pass 4: cycles, ceilings, output --------------------------------
    _resolve_cycles(graph, results_by_id, ceilings, stats)

    results: list[VerificationResult] = []
    for proposition in propositions:
        pid = str(proposition.get("id"))
        result = results_by_id[pid]
        _finalise(result, ceilings[pid], config, stats)
        _write_back(proposition, result, geometry.get(pid))
        results.append(result)
        key = result.status.value if result.status else "None"
        stats.by_status[key] = stats.by_status.get(key, 0) + 1

    stats.total_rejection = bool(propositions) and all(
        r.status is Verdict.REJECTED for r in results
    )
    if stats.total_rejection:
        stats.notes.append(
            "toàn bộ mệnh đề bị bác bỏ — cần chú thích tối thiểu chỉ gồm đối tượng (§7)"
        )
    return results, stats


def _existence_pid(propositions: Sequence[dict], entity_id: str) -> str:
    for proposition in propositions:
        if proposition.get("type") == "entity" and _subject_id(proposition) == entity_id:
            return str(proposition.get("id"))
    return ""


def _epistemic_ceilings(proposition: dict, index: dict[str, dict]) -> list[Ceiling]:
    """Caps that apply before any score is computed.

    INFERENCE (formulation/08 §2) and a gendered noun on an entity whose gender
    is not determinable (formulation/02 §4.3): in Vietnamese gender sits in the
    noun, so asserting it is a content error, not a pronoun slip.
    """
    out: list[Ceiling] = []
    external = ((proposition.get("evidence") or {}).get("external_knowledge") or {})
    if external.get("requires_inference"):
        out.append(
            Ceiling(
                Verdict.UNCERTAIN,
                "mệnh đề dựa trên suy luận ngoài những gì nhìn thấy",
                "the claim rests on inference beyond what is visible",
            )
        )
    count = proposition.get("count") or {}
    value = count.get("value") if proposition.get("type") == "counting" else None
    if value is not None and int(value) > EXACT_COUNT_LIMIT and count.get("exact", True):
        # Beyond ~5 objects human counts disagree too (formulation/02 §4.2), so
        # an exact count above the limit is not a claim a verifier can confirm;
        # it is realised approximately or hedged instead.
        out.append(
            Ceiling(
                Verdict.UNCERTAIN,
                f"số lượng chính xác lớn hơn {EXACT_COUNT_LIMIT} không kiểm chứng được đáng tin cậy",
                f"an exact count above {EXACT_COUNT_LIMIT} cannot be verified reliably",
            )
        )

    entity = index.get(_subject_id(proposition) or "") or {}
    gender = entity.get("gender") or {}
    # Identity claims only (research log ). An uncertain gender makes `có một
    # người phụ nữ` uncertain; it does not make the colour of her áo dài
    # uncertain, and `realize.py` already declines to write the gendered noun,
    # so nothing about gender is asserted either way.
    if (
        str(proposition.get("type")) in ("entity", "counting")
        and gender.get("value") in GENDERED_VALUES
        and gender.get("evidence") != "clearly_visible"
    ):
        # formulation/02 §4.3: of the three evidence values, "**only the first**
        # [clearly_visible] licenses a gendered noun". Checking only for
        # `not_determinable` was dead code on real input: pipeline/generate.py
        # writes `value = "khong_xac_dinh"` whenever it writes
        # `evidence = "not_determinable"`, and tags every gendered head noun the
        # backbone produced as `inferred_from_clothing` — the case it explicitly
        # leaves for verification to decide. Uncapped, a gender guessed from
        # clothing reaches SUPPORTED, i.e. is asserted as a visual fact, which
        # doc 02 §4.3 scores as a hallucination.
        evidence = str(gender.get("evidence") or "không ghi nhận")
        out.append(
            Ceiling(
                Verdict.UNCERTAIN,
                f"giới tính không quan sát trực tiếp được ({evidence}) nên phải dùng "
                "danh từ trung tính 'người'",
                f"gender is not directly observed ({evidence}); the neutral noun "
                "'người' must be used",
            )
        )
    return out


def is_colour_proposition(proposition: dict) -> bool:
    """True when the claim is about colour, so it needs the second verifier.

    Reads the `kind` tag `pipeline/generate.py` writes from the backbone's own
    attribute label, rather than sniffing the text for colour words: a caption
    mentioning a colour in passing is not a colour *claim*, and re-deriving the
    type here would let the two modules drift apart.

    >>> is_colour_proposition({"type": "attribute",
    ...                        "attributes": [{"kind": "màu_sắc", "value_vi": "màu xanh"}]})
    True
    >>> is_colour_proposition({"type": "entity", "attributes": []})
    False
    """
    if proposition.get("type") != "attribute":
        return False
    return any(
        a.get("kind") == "màu_sắc" for a in (proposition.get("attributes") or [])
    )


def _handle_xanh(
    proposition: dict,
    model: VLM,
    image: Any,
    config: VerificationConfig,
    stats: VerificationStats,
    ceilings: list[Ceiling],
) -> bool:
    """Resolve bare `xanh` if possible; cap the proposition if not."""
    if proposition.get("type") != "attribute":
        return False
    unresolved = False
    for attribute in proposition.get("attributes") or []:
        if attribute.get("kind") != "màu_sắc":
            continue
        # Through `_colour_term`, not `parse_color` directly: pipeline/generate.py
        # writes `value_vi` as the backbone's whole line (`áo màu xanh`), and
        # `parse_color` only recognises a bare colour expression, so it reads
        # `màu xanh` as "not a colour at all" and the guard never fired. A bare
        # `xanh` would then pass through unresolved AND uncapped — silently
        # asserting a colour Vietnamese leaves ambiguous (formulation/02 §4.5).
        raw = str(attribute.get("value_vi", ""))
        reading = parse_color(_colour_term(raw) or raw)
        if reading.xanh_value is not Xanh.UNRESOLVED:
            continue
        if resolve_xanh(model, image, proposition, attribute, config, stats):
            continue
        unresolved = True
        ceilings.append(
            Ceiling(
                Verdict.UNCERTAIN,
                "'xanh' chưa phân giải được thành xanh dương hay xanh lá",
                "bare 'xanh' could not be resolved to blue or green",
            )
        )
    return unresolved


def _resolve_cycles(
    graph: ContradictionGraph,
    results: dict[str, VerificationResult],
    ceilings: dict[str, list[Ceiling]],
    stats: VerificationStats,
) -> None:
    """A⊥B, B⊥C, C⊥A: keep the best-supported member, cap the rest.

    formulation/09 §7 words this as "mark the rest UNCERTAIN", but it is applied
    as a ceiling: a member the image already rejected must stay REJECTED.
    """
    for cycle in graph.cycles():
        stats.contradiction_cycles.append(cycle)
        best = max(cycle, key=lambda pid: (results[pid].scores.s or 0.0) if pid in results else 0.0)
        for pid in cycle:
            if pid == best or pid not in ceilings:
                continue
            ceilings[pid].append(
                Ceiling(
                    Verdict.UNCERTAIN,
                    f"nằm trong một vòng mâu thuẫn ({' ⊥ '.join(cycle)}); "
                    f"chỉ giữ mệnh đề được hỗ trợ mạnh nhất là {best}",
                    f"member of a contradiction cycle ({' vs '.join(cycle)}); "
                    f"only the best-supported member {best} is kept",
                )
            )


def _build_result(
    proposition: dict,
    verdict: Verdict,
    branch: str,
    scores: Scores,
    *,
    constraint: HardConstraint | None,
    cascade: Sequence[Cascade],
    ceilings: Sequence[Ceiling],
    limiting: EvidenceSignal | None,
    contradictors: dict[str, str],
    supporting: Sequence[str],
    geometric: GeometricCheck | None,
    outcome: ProbeOutcome,
    disagreement: bool,
    contradiction_reason: str,
    signals: Sequence[EvidenceSignal],
    verifier: str,
    rule_version: str,
    now: str,
) -> VerificationResult:
    pid = str(proposition.get("id"))
    channel = TYPE_CHANNEL.get(str(proposition.get("type")), "grounding")
    channel_scores: dict[str, float] = {}
    if scores.v is not None:
        channel_scores[channel] = scores.v
    if geometric is not None and geometric.score is not None:
        channel_scores["spatial"] = geometric.score

    if branch == "hard_constraint" and constraint is not None:
        explanation_vi, explanation_en = constraint.reason_vi, constraint.reason_en
    elif branch == "cascade":
        names = ", ".join(f"{c.entity_id} ({c.role})" for c in cascade)
        explanation_vi = (
            f"Không xác nhận được sự tồn tại của {names}, nên mệnh đề này bị bác bỏ "
            "theo dây chuyền — đây là lỗi kế thừa, không phải lỗi sơ cấp."
        )
        explanation_en = (
            f"The existence of {names} was not confirmed, so this proposition is "
            "rejected by cascade — an inherited error, not a primary one."
        )
    elif branch == "gate":
        cause_vi, cause_en = _cause_vi(branch, limiting, contradiction_reason, geometric, contradictors)
        explanation_vi = f"Bằng chứng không đủ để kết luận ({cause_vi})."
        explanation_en = f"Not enough evidence to decide ({cause_en})."
    else:
        cause_vi, cause_en = _cause_vi(branch, limiting, contradiction_reason, geometric, contradictors)
        head_vi = _BRANCH_VI.get(branch, "Bằng chứng không rõ ràng")
        head_en = _BRANCH_EN.get(branch, "The evidence is inconclusive")
        explanation_vi = f"{head_vi}: {cause_vi}." if cause_vi else f"{head_vi}."
        explanation_en = f"{head_en}: {cause_en}." if cause_en else f"{head_en}."

    return VerificationResult(
        proposition_id=pid,
        status=verdict,
        branch=branch,
        scores=scores,
        explanation_vi=explanation_vi,
        explanation_en=explanation_en,
        channel_scores=channel_scores,
        evidence_signals=list(signals),
        limiting_signal=limiting.name if limiting else None,
        ceilings=list(ceilings),
        cascade_from=list(cascade),
        contradicts=sorted(contradictors),
        supported_by=sorted(set(supporting)),
        geometric=geometric,
        probes=list(outcome.records),
        acquiescent=outcome.acquiescent,
        geometry_probe_disagreement=disagreement,
        lang_fallback=outcome.lang_fallback,
        verifier=verifier,
        decision_rule_version=rule_version,
        verified_at=now,
    )


def _clamp(
    result: VerificationResult,
    ceilings: Sequence[Ceiling],
    stats: VerificationStats,
) -> None:
    """Lower a verdict to its strictest ceiling and say why. Never raises it."""
    if result.status is None or not ceilings:
        return
    capped = apply_ceilings(result.status, ceilings)
    if capped is result.status:
        return
    stats.ceilinged += 1
    binding = [c for c in ceilings if _RANK[c.verdict] <= _RANK[capped]]
    reasons = "; ".join(c.reason_vi for c in binding if c.reason_vi)
    reasons_en = "; ".join(c.reason_en for c in binding if c.reason_en)
    result.explanation_vi += f" Mức tối đa có thể đạt là {capped.value} vì {reasons}."
    result.explanation_en += f" Capped at {capped.value} because {reasons_en}."
    result.status = capped


def _finalise(
    result: VerificationResult,
    ceilings: Sequence[Ceiling],
    config: VerificationConfig,
    stats: VerificationStats,
) -> None:
    """Apply the deferred ceilings, then the A2 collapse.

    Order matters: the three-way verdict must be settled before it is
    collapsed, or formulation/05 §7.4's C-vs-D confusion table would be built
    from a value that never existed.
    """
    if result.status is None:
        return
    # `ceilings_applied` indexes into this list, so the result must mirror it
    # exactly — deduplicating here would shift the index and silently re-apply
    # or skip a reason.
    _clamp(result, list(ceilings)[result.ceilings_applied :], stats)
    result.ceilings = list(ceilings)
    result.ceilings_applied = len(ceilings)
    result.three_way_status = result.status
    if config.mode == Mode.BINARY and result.status is Verdict.UNCERTAIN:
        # A2 / Baseline C. The collapse is recorded, not hidden: formulation/09
        # §1's whole point is that this choice costs either detail or accuracy.
        result.status = config.binary_uncertain_to
        stats.binary_collapsed += 1
        result.explanation_vi += (
            " (Chế độ nhị phân A2: trạng thái KHÔNG CHẮC CHẮN bị quy về "
            f"{config.binary_uncertain_to.value}; đây là quy ước của cấu hình, "
            "không phải kết luận từ ảnh.)"
        )
        result.explanation_en += (
            " (Binary mode A2: UNCERTAIN was collapsed to "
            f"{config.binary_uncertain_to.value}; a configuration convention, "
            "not a conclusion from the image.)"
        )


def _write_back(
    proposition: dict,
    result: VerificationResult,
    geometric: GeometricCheck | None,
) -> None:
    """Write the schema-legal parts of the result onto the proposition."""
    evidence = _evidence(proposition)
    # Assigned, not appended: a threshold sweep re-runs M4 over the same cached
    # propositions (formulation/09 §4.2), and appending would silently double
    # the audit record on every pass.
    evidence["probes"] = list(result.probes)
    evidence["geometric"] = {"checks_passed": [], "checks_failed": []}
    cross = evidence["cross_proposition"]
    cross["supporting_ids"] = result.supported_by
    cross["contradicting_ids"] = result.contradicts
    if result.scores.m is not None:
        cross["consistency_score"] = round(2.0 * result.scores.m - 1.0, 4)

    if result.scores.v is not None:
        evidence["visual_grounding"]["grounding_score"] = round(result.scores.v, 4)
        evidence["visual_grounding"]["method"] = "vqa_probe"

    if geometric is not None:
        target = evidence["geometric"]
        label = f"{geometric.relation}: {geometric.reason_en or 'no test'}"
        if geometric.consistent is True:
            target["checks_passed"].append(label)
        elif geometric.consistent is False:
            target["checks_failed"].append(label)
        spatial = proposition.setdefault("spatial_relation", {})
        spatial["geometric_check"] = geometric.to_schema()

    if result.contradicts:
        existing = proposition.setdefault("contradicts", [])
        for pid in result.contradicts:
            if pid not in existing:
                existing.append(pid)

    proposition["verification"] = result.to_schema()


def assert_verified(propositions: Sequence[dict], config: VerificationConfig) -> None:
    """The formulation/09 §7 invariant, checked rather than assumed.

    Every proposition leaves this module with a populated `explanation_vi` and a
    `decision_rule_version`, and — outside A1 — a non-null `status`. A verdict
    without a traceable rule version is unreproducible, and an empty explanation
    is unusable for error analysis, so both are hard errors.
    """
    problems: list[str] = []
    for proposition in propositions:
        pid = str(proposition.get("id"))
        verification = proposition.get("verification") or {}
        if config.mode != Mode.NONE and verification.get("status") is None:
            problems.append(f"{pid}: thiếu status")
        if not verification.get("explanation_vi"):
            problems.append(f"{pid}: thiếu explanation_vi")
        if not verification.get("decision_rule_version"):
            problems.append(f"{pid}: thiếu decision_rule_version")
        # A speculative claim must be explained as *not determinable from the
        # image*, never as false: writing "sai" would make the system's own
        # explanation a false claim (formulation/09 §5.1).
        if config.mode != Mode.NONE and hard_constraint(proposition) is not None and (
            "không thể xác định trực tiếp từ ảnh"
            not in str(verification.get("explanation_vi", ""))
        ):
            problems.append(
                f"{pid}: mệnh đề suy đoán phải được giải thích là 'không thể xác định "
                "trực tiếp từ ảnh', không phải là sai"
            )
    if problems:
        raise RuntimeError("kết quả kiểm chứng không hợp lệ: " + "; ".join(problems))


def verdict_counts(results: Iterable[VerificationResult]) -> dict[str, int]:
    """Verdict histogram, for the doc 05 §7.4 confusion table."""
    counts: dict[str, int] = {}
    for result in results:
        key = result.status.value if result.status else "None"
        counts[key] = counts.get(key, 0) + 1
    return counts


__all__ = [
    "Verdict",
    "Mode",
    "MODES",
    "Thresholds",
    "Weights",
    "SignalCalibration",
    "VerificationConfig",
    "Scores",
    "EvidenceSignal",
    "Ceiling",
    "Cascade",
    "GeometricCheck",
    "VerificationResult",
    "VerificationStats",
    "Box",
    "as_box",
    "iou",
    "geometric_check",
    "evidence_signals",
    "aggregate_evidence",
    "build_questions",
    "run_probe",
    "visual_support",
    "resolve_xanh",
    "ContradictionGraph",
    "detect_contradictions",
    "semantic_consistency",
    "HardConstraint",
    "hard_constraint",
    "decide",
    "apply_ceilings",
    "verify",
    "assert_verified",
    "verdict_counts",
    "normalise_relation",
    "SPECULATIVE_INFERENCE_TYPES",
]
