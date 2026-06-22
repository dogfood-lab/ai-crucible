"""Cross-module contracts for the Epic-4 catalog layer (``catalog`` package).

Everything that crosses a boundary between the three catalog leaves —
:mod:`ai_crucible.catalog.store` (event-sourced persistence), ``.graduation``
(the Lab→Arena→Regression lifecycle rules), and ``.differential`` (the typology
statistics) — is defined here. The leaves implement *against* these types; they
do not redefine them and they do not import each other (DECOMPOSE_BY_SECRETS —
the store takes the lifecycle decisions as INJECTED callables, see
:data:`PromoteFn` / :data:`DemoteFn` / :data:`ClassifyFn`).

Design lock: ``swarm/epic-4/CONTRACT.md`` (grounded in the Epic-4 study-swarm,
``swarm/epic-4/STUDY-SWARM.json``). Citations there; the load-bearing ones recur
in the leaf docstrings.

The catalog log is the SOURCE OF TRUTH (an :class:`ai_crucible.attestation.JsonlEventStore`
of hash-chained RUN/TRANSITION events); tier state is a DERIVED projection rebuilt
by folding the events. Two invariants are structural, not aspirational:

1. **Tier is never mutated in place** — a tier change is an appended TRANSITION
   event; the current tier is the fold of all transitions.
2. **`rule_version` is pinned on every event** — graduation/saturation thresholds
   are policy knobs (:class:`RuleConfig`), so the timeline records the decision made
   *at the time*, never a retroactive re-judgment.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from ai_crucible.types import CatalogTier

__all__ = [
    "CATALOG_SCHEMA_VERSION",
    "Decision",
    "Typology",
    "TransitionReason",
    "RuleConfig",
    "PanelSignal",
    "RunRecord",
    "TransitionRecord",
    "ModelRunSeries",
    "PuzzleAggregate",
    "PromoteVerdict",
    "DemoteVerdict",
    "ClassifyResult",
    "PromoteFn",
    "DemoteFn",
    "ClassifyFn",
    "FrontierFn",
    "stable_hash",
]

#: Forward-incompat guard. The fold refuses to read a record stamped with a HIGHER
#: schema_version than it understands (a downgrade reading newer data), mirroring
#: ``characterize.panel_store``'s version guard. Bump on any payload-shape change.
CATALOG_SCHEMA_VERSION = 1


def stable_hash(*parts: object) -> str:
    """A deterministic short id over ``parts`` — the catalog's id primitive.

    Joins the parts with a NUL separator (a byte that cannot appear in any of our
    string fields) and returns the first 16 hex chars of their SHA-256. Used to
    derive the idempotent ``run_id`` / ``transition_id`` so the same logical event
    always hashes to the same id (PIN_PER_STEP / Q4 idempotency). A list part is
    sorted+joined first so contributing-run sets hash order-independently.
    """
    flat: list[str] = []
    for p in parts:
        if isinstance(p, (list, tuple, set, frozenset)):
            flat.append(",".join(sorted(str(x) for x in p)))
        else:
            flat.append(str(p))
    material = "\x00".join(flat).encode("utf-8")
    return hashlib.sha256(material).hexdigest()[:16]


# --------------------------------------------------------------------------- #
# Enums
# --------------------------------------------------------------------------- #


class Decision(StrEnum):
    """The three-valued Lab→Arena promotion verdict (CONTRACT §A).

    ``DEFER_TO_DESIGNER`` is the abstention-aware keystone: a low-confidence
    cross-family fairness verdict (sub-quorum panel or a thin/escalating margin)
    must abstain and escalate, never fake confidence (Trust-or-Escalate, Jung et
    al. ICLR 2025).
    """

    PROMOTE = "promote"
    HOLD = "hold"
    DEFER_TO_DESIGNER = "defer_to_designer"


class Typology(StrEnum):
    """The differential diagnostic typology (CONTRACT §C, research-grounding §4).

    ``INCONCLUSIVE_UNDERPOWERED`` is a FIRST-CLASS outcome, not a failure: at
    N=20/k=3 the minimum detectable effect is large, so most puzzles legitimately
    land here until more evidence accumulates.
    """

    CLAUDE_SPECIFIC_GAP = "claude_specific_gap"      # Claude underperforms cohort (highest value)
    CLAUDE_STRENGTH = "claude_strength"              # Claude outperforms cohort (anti-regression)
    LLM_GENERAL_GAP = "llm_general_gap"              # everyone fails (frontier gap)
    INCONCLUSIVE_UNDERPOWERED = "inconclusive_underpowered"  # delta-CI cannot resolve at this N


class TransitionReason(StrEnum):
    """Why a tier transition was recorded (CONTRACT §D ``reason_code``)."""

    GRADUATE_WILSON = "graduate_wilson"            # Lab→Arena, all promotion clauses passed
    DEFER_TO_DESIGNER = "defer_to_designer"        # Lab→Lab hold with an escalation receipt
    SATURATED_REGRESSION = "saturated_regression"  # Arena→Regression, e-process crossed
    DEMOTE_NOT_DELETE = "demote_not_delete"        # generic demotion (preserve timeline)
    REPROMOTE_REGRESSION = "repromote_regression"  # Regression→Arena, capability regressed
    MANUAL = "manual"                              # an attested human/Designer override


# --------------------------------------------------------------------------- #
# Rule configuration — the pinned policy knobs (PIN_PER_STEP)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class RuleConfig:
    """The pinned graduation/saturation/differential thresholds.

    These are POLICY KNOBS, not derived constants (CONTRACT cross-cutting lock 3).
    :attr:`rule_version` is a content hash of the knobs; it is stamped on every
    RUN and TRANSITION so the timeline reflects the rule actually in force when a
    decision was made — replaying under changed knobs never silently rewrites the
    historical verdicts.

    Defaults are the study-swarm-grounded values (CONTRACT §A–§C).
    """

    # Graduation (fixed-N Wilson; CONTRACT §A) — Clause-1 bounds live in
    # stats.graduates(); Clause-2 reuses the upper bound here.
    cohort_trivial_upper: float = 0.90    # cohort Wilson-upper above this => trivial => HOLD

    # Saturation (anytime-valid e-process; CONTRACT §B) — DO NOT unify with graduation.
    sat_p0: float = 0.90                  # H0: p <= p0 (not yet saturated)
    sat_p1: float = 0.97                  # H1: p >= p1 (saturated)
    sat_alpha: float = 0.05              # demote when e-value >= 1/alpha (=20)
    repromote_p0: float = 0.75            # re-promote when rate falls through the dead band
    dwell_k: int = 2                      # demotion needs >= this many SEPARATE runs of evidence
    post_trigger_min_k_mult: int = 3      # AND post-trigger attempts >= mult * puzzle.min_k

    # Differential (Newcombe delta-CI; CONTRACT §C).
    mde_floor: float = 0.30               # |delta| must clear this to call a direction
    general_fail_ceiling: float = 0.20    # both Wilson uppers <= this => LLM-general gap
    differential_conf: float = 0.95       # delta-CI confidence
    fdr_q: float = 0.10                   # catalog-level BH-FDR q

    @property
    def e_threshold(self) -> float:
        """The e-value demotion threshold ``1/alpha`` (CONTRACT §B)."""
        return 1.0 / self.sat_alpha

    @property
    def rule_version(self) -> str:
        """Deterministic content hash of the knobs — pinned on every event."""
        return stable_hash(
            self.cohort_trivial_upper, self.sat_p0, self.sat_p1, self.sat_alpha,
            self.repromote_p0, self.dwell_k, self.post_trigger_min_k_mult,
            self.mde_floor, self.general_fail_ceiling, self.differential_conf, self.fdr_q,
        )


# --------------------------------------------------------------------------- #
# Panel signal — the cross-family fairness input graduation consumes
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class PanelSignal:
    """The cross-family fairness signal Clause-3 of graduation reads (CONTRACT §A).

    Decouples ``graduation`` from the concrete :class:`~ai_crucible.scoring.judge_panel.JudgePanel`:
    the integrator distils a run's panel score into this small value object, the store
    aggregates it per puzzle, and ``promote_decision`` reads only these fields.

    ``present`` is the honest default: with NO cross-family panel signal, fairness
    cannot be certified, so graduation must DEFER (Arena REQUIRES cross-family
    validation — research-grounding §1; absence of evidence is not a pass).
    ``fairness`` is the panel's confident verdict (``"fair"`` / ``"unfair"``) or
    ``None`` when it abstained.
    """

    present: bool = False
    meets_quorum: bool = False
    escalate: bool = False
    fairness: str | None = None  # "fair" | "unfair" | None (abstained)

    @property
    def deferred(self) -> bool:
        """True when the fairness verdict is low-confidence → DEFER (CONTRACT §A)."""
        return (not self.present) or (not self.meets_quorum) or self.escalate

    def to_payload(self) -> dict[str, Any]:
        return {
            "present": self.present,
            "meets_quorum": self.meets_quorum,
            "escalate": self.escalate,
            "fairness": self.fairness,
        }

    @classmethod
    def from_payload(cls, d: dict[str, Any] | None) -> PanelSignal:
        if not d:
            return cls()
        return cls(
            present=bool(d.get("present", False)),
            meets_quorum=bool(d.get("meets_quorum", False)),
            escalate=bool(d.get("escalate", False)),
            fairness=d.get("fairness"),
        )


# --------------------------------------------------------------------------- #
# Event records — the durable RUN / TRANSITION payloads (CONTRACT §D)
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class RunRecord:
    """One accumulated diagnostic RUN against a puzzle by one model (CONTRACT §D).

    The unit of the catalog timeline: ``ai-crucible run`` appends one of these per
    cycle. ``puzzle_content_hash`` (over meta.json + oracle + setup_script) lineages
    the puzzle so an edit forks a NEW lineage (DVC content-addressing). ``family`` is
    the cross-family axis the differential + graduation cohort split on (Claude vs
    non-Claude). The saturation e-process treats one RunRecord as one RUN (its unit —
    CONTRACT §B / Liu-2025 within-batch correlation), so ``k``/``n``/``successes`` are
    per-run.
    """

    kind = "run"

    puzzle_id: str
    puzzle_content_hash: str
    model_id: str
    family: str | None
    role: str            # "solver" | "cohort_solver"
    k: int
    n: int
    successes: int
    pass_hat_k: float
    wilson_lower: float
    wilson_upper: float
    arm: str
    started_at: str      # ISO 8601 (injected at the edge; pinned in tests)
    finished_at: str
    nonce: str           # disambiguates two otherwise-identical runs
    # The puzzle's pass^k floor (PuzzleMeta.min_k), recorded so the catalog is
    # SELF-CONTAINED for graduation/saturation — the demote floor (3·min_k) is
    # recoverable from the log alone, without the puzzle files still being present.
    # Not part of run_id (idempotency is unaffected by recording it).
    min_k: int = 10
    schema_version: int = CATALOG_SCHEMA_VERSION
    rule_version: str = ""
    provenance_level: int = 1   # 1 = hash-chain-only (SLSA~L1); 2 = cosign-signed (honest)
    panel: PanelSignal | None = None

    @property
    def run_id(self) -> str:
        """Deterministic idempotent id (CONTRACT §D)."""
        return stable_hash(
            self.puzzle_content_hash, self.model_id, self.started_at, self.nonce
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "run_id": self.run_id,
            "puzzle_id": self.puzzle_id,
            "puzzle_content_hash": self.puzzle_content_hash,
            "model_id": self.model_id,
            "family": self.family,
            "role": self.role,
            "k": self.k,
            "n": self.n,
            "successes": self.successes,
            "pass_hat_k": self.pass_hat_k,
            "wilson_lower": self.wilson_lower,
            "wilson_upper": self.wilson_upper,
            "arm": self.arm,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "nonce": self.nonce,
            "min_k": self.min_k,
            "schema_version": self.schema_version,
            "rule_version": self.rule_version,
            "provenance_level": self.provenance_level,
            "panel": self.panel.to_payload() if self.panel is not None else None,
        }

    @classmethod
    def from_payload(cls, d: dict[str, Any]) -> RunRecord:
        return cls(
            puzzle_id=d["puzzle_id"],
            puzzle_content_hash=d["puzzle_content_hash"],
            model_id=d["model_id"],
            family=d.get("family"),
            role=d.get("role", "solver"),
            k=int(d["k"]),
            n=int(d["n"]),
            successes=int(d["successes"]),
            pass_hat_k=float(d.get("pass_hat_k", 0.0)),
            wilson_lower=float(d.get("wilson_lower", 0.0)),
            wilson_upper=float(d.get("wilson_upper", 1.0)),
            arm=d.get("arm", ""),
            started_at=d.get("started_at", ""),
            finished_at=d.get("finished_at", ""),
            nonce=d.get("nonce", ""),
            min_k=int(d.get("min_k", 10)),
            schema_version=int(d.get("schema_version", CATALOG_SCHEMA_VERSION)),
            rule_version=d.get("rule_version", ""),
            provenance_level=int(d.get("provenance_level", 1)),
            panel=PanelSignal.from_payload(d.get("panel")),
        )


@dataclass(slots=True)
class TransitionRecord:
    """A recorded tier transition (CONTRACT §D). Append-only; never edited.

    ``contributing_run_ids`` are the runs whose accumulated evidence justified the
    transition; together with ``rule_version`` they make ``transition_id``
    idempotent (the same evidence under the same rule fires the transition at most
    once). ``decided_by`` is ``"auto"`` for a rule-derived transition or an attested
    actor for a MANUAL override (e.g. ``"designer:<name>"``).
    """

    kind = "transition"

    puzzle_id: str
    from_tier: CatalogTier
    to_tier: CatalogTier
    reason_code: TransitionReason
    recorded_at: str
    decided_by: str = "auto"
    contributing_run_ids: list[str] = field(default_factory=list)
    evidence: dict[str, Any] = field(default_factory=dict)
    schema_version: int = CATALOG_SCHEMA_VERSION
    rule_version: str = ""

    @property
    def transition_id(self) -> str:
        """Deterministic idempotent id (CONTRACT §D)."""
        return stable_hash(
            self.puzzle_id, str(self.from_tier), str(self.to_tier),
            self.contributing_run_ids, self.rule_version,
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "transition_id": self.transition_id,
            "puzzle_id": self.puzzle_id,
            "from_tier": str(self.from_tier),
            "to_tier": str(self.to_tier),
            "reason_code": str(self.reason_code),
            "recorded_at": self.recorded_at,
            "decided_by": self.decided_by,
            "contributing_run_ids": list(self.contributing_run_ids),
            "evidence": dict(self.evidence),
            "schema_version": self.schema_version,
            "rule_version": self.rule_version,
        }

    @classmethod
    def from_payload(cls, d: dict[str, Any]) -> TransitionRecord:
        return cls(
            puzzle_id=d["puzzle_id"],
            from_tier=CatalogTier(d["from_tier"]),
            to_tier=CatalogTier(d["to_tier"]),
            reason_code=TransitionReason(d["reason_code"]),
            recorded_at=d.get("recorded_at", ""),
            decided_by=d.get("decided_by", "auto"),
            contributing_run_ids=list(d.get("contributing_run_ids", [])),
            evidence=dict(d.get("evidence", {})),
            schema_version=int(d.get("schema_version", CATALOG_SCHEMA_VERSION)),
            rule_version=d.get("rule_version", ""),
        )


# --------------------------------------------------------------------------- #
# Folded aggregates — what the fold hands the injected decision fns
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class ModelRunSeries:
    """The ordered per-(puzzle, model) run series — the saturation unit (CONTRACT §B).

    ``runs`` is the chronological list of ``(successes, n)`` per RUN (NOT per
    attempt): the e-process multiplies one likelihood factor per run, and dwell
    requires ``>= dwell_k`` SEPARATE runs. ``is_frontier`` is resolved by the store
    from the injected :data:`FrontierFn` (a weak local model passing never demotes).
    """

    puzzle_id: str
    model_id: str
    family: str | None
    is_frontier: bool
    min_k: int
    runs: list[tuple[int, int]] = field(default_factory=list)

    @property
    def total_successes(self) -> int:
        return sum(s for s, _ in self.runs)

    @property
    def total_n(self) -> int:
        return sum(n for _, n in self.runs)


@dataclass(slots=True)
class PuzzleAggregate:
    """The folded per-puzzle view graduation + differential consume (CONTRACT §A/§C).

    The cross-family cohort is the union of all NON-generator (here: non-Claude)
    families' runs on the puzzle; ``claude_*`` is the generator's own runs. Both are
    accumulated at the SAME N floor (CONTRACT cross-cutting lock 1). ``panel`` is the
    aggregated fairness signal. ``per_model`` carries the per-(puzzle,model) series
    for saturation.
    """

    puzzle_id: str
    puzzle_content_hash: str
    claude_successes: int
    claude_n: int
    cohort_successes: int
    cohort_n: int
    panel: PanelSignal
    per_model: dict[str, ModelRunSeries] = field(default_factory=dict)
    contributing_run_ids: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Decision verdicts returned by the injected fns
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class PromoteVerdict:
    """The graduation decision + its clause-level evidence (CONTRACT §A)."""

    decision: Decision
    claude_band_ok: bool
    cohort_nontrivial: bool
    fairness_ok: bool
    deferred: bool
    born_typology: Typology | None = None
    evidence: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class DemoteVerdict:
    """The saturation decision for one (puzzle, model) (CONTRACT §B)."""

    demote: bool
    e_value: float
    dwell_runs: int
    post_trigger_n: int
    frontier_ok: bool
    reason: str = ""
    evidence: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ClassifyResult:
    """The differential typology result for one puzzle (CONTRACT §C)."""

    typology: Typology
    delta: float
    delta_ci: tuple[float, float]
    realized_mde: float
    p_value: float
    bh_survived: bool = True
    evidence: dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Injected decision-fn contracts (the store ↔ leaves decoupling seam)
# --------------------------------------------------------------------------- #

#: Lab→Arena promotion rule. ``graduation.promote_decision`` satisfies this; the
#: store calls it during the fold and the store's tests inject a fake.
PromoteFn = Callable[[PuzzleAggregate, RuleConfig], PromoteVerdict]

#: Arena→Regression saturation rule, per (puzzle, model). ``graduation.demote_decision``.
DemoteFn = Callable[[ModelRunSeries, RuleConfig], DemoteVerdict]

#: Differential typology classifier. ``differential.classify_puzzle``.
ClassifyFn = Callable[[PuzzleAggregate, RuleConfig], ClassifyResult]

#: Resolves whether a model id is a current-frontier subject (CONTRACT §B frontier
#: gate). Policy input injected into the fold; a weak local model passing never demotes.
FrontierFn = Callable[[str], bool]
