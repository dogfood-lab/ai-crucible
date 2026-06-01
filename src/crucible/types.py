"""Cross-module contracts for the Crucible kernel.

Everything that crosses a module boundary is defined here. Module bodies
(``kernel``, ``scoring``, ``engagement``, ``observability``, ``attestation``,
...) implement *against* these types; they do not redefine them. This file is
the exclusive-ownership boundary for the Phase-1 build waves.

Citations below reference docs/research-grounding.md (e.g. "§10.2").
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, Field, model_validator

# --------------------------------------------------------------------------- #
# Enums
# --------------------------------------------------------------------------- #


class RoleName(StrEnum):
    """The five model-agnostic role slots (gameplan Phase 1; §10.3).

    ``DESIGNER`` stays Claude (the creative role crucible is built for). Only
    ``SOLVER`` is granted sandbox tools. ``CRITIC`` is interface-reserved but
    default-off (§10.3). Judge/CohortSolver draw from the cross-family panel.
    """

    DESIGNER = "designer"
    SOLVER = "solver"
    CRITIC = "critic"
    JUDGE = "judge"
    COHORT_SOLVER = "cohort_solver"


class FramingArm(StrEnum):
    """Prompt-framing as a first-class *measured arm* (§10.1(f), director 2026-06-01).

    The kernel can run the same puzzle under each arm so crucible characterizes
    its own prompt-effect. ``SELF_REFERENTIAL`` is the default; ``SOCIAL_STANDINGS``
    is the old peer-standings prompt, retained only as a measured arm and rendered
    in Tier-3 chrome, never as the default scored context.
    """

    NEUTRAL = "neutral"
    SELF_REFERENTIAL = "self_referential"
    SOCIAL_STANDINGS = "social_standings"


class TerminatedBy(StrEnum):
    """Why an attempt ended. Set kernel-side, never Solver-self-reported (§10.2)."""

    COMPLETED = "completed"
    BUDGET = "budget"          # tool/step/token budget exhausted (ANDON)
    TIME = "time"              # wall-clock budget exhausted
    HARD_KILL = "hard_kill"    # 3 consecutive identical (tool, args) calls (§8.4)
    ERROR = "error"


class CatalogTier(StrEnum):
    """The three catalog tiers (research-grounding §1)."""

    LAB = "lab"                # live, in-iteration
    ARENA = "arena"            # graduated, active diagnostic (cross-family validated)
    REGRESSION = "regression"  # solved/retired; must-still-pass forever


class GoodhartFlavor(StrEnum):
    """Reward-hacking taxonomy per Skalse et al. 2022 (arXiv:2209.13085), used to
    tag each declared answer-bypass penalty (§8.2)."""

    REGRESSIONAL = "regressional"
    EXTREMAL = "extremal"
    CAUSAL = "causal"
    ADVERSARIAL = "adversarial"


class PuzzleClass(StrEnum):
    """Heterogeneous per-class tool budgets (§8.4, AgentBench ranges)."""

    FILE_INSPECTION = "file_inspection"      # budget ~5
    MULTI_FILE_SEARCH = "multi_file_search"  # budget ~12
    FULL_REPO_TRACE = "full_repo_trace"      # budget ~20
    LONG_HORIZON = "long_horizon"            # budget up to ~30


# --------------------------------------------------------------------------- #
# Runtime value objects (dataclasses)
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class Budget:
    """The displayed budget injected each turn (§8.4, BATS arXiv:2511.17006).

    Externalizing the rationing decision is a Tier-1 *task-relevant* signal — it
    points at task features, so it belongs in the scored context (§10.1(a)).
    """

    tool_call_budget: int
    time_budget_seconds: int
    tool_calls_used: int = 0
    elapsed_seconds: float = 0.0

    @property
    def tool_calls_remaining(self) -> int:
        return max(0, self.tool_call_budget - self.tool_calls_used)

    @property
    def time_remaining(self) -> float:
        return max(0.0, self.time_budget_seconds - self.elapsed_seconds)


@dataclass(slots=True)
class Score:
    """Inspect-shaped score (§10.2). ``value`` is the headline; ``metadata`` carries
    component breakdown (solve/elegance/novelty/penalties), panel votes, etc."""

    value: float | bool | str
    answer: str | None = None
    explanation: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class TraceEvent:
    """One structured event in the per-attempt transcript (Inspect EvalLog shape,
    §10.2). Every model/tool call the kernel makes is appended as an event; large
    blobs are stored as attachments referenced by digest, not inlined."""

    kind: str                      # "model" | "tool" | "score" | "info" | "error"
    role: RoleName | None = None
    payload: dict[str, Any] = field(default_factory=dict)
    attachments: dict[str, str] = field(default_factory=dict)  # name -> sha256
    seq: int = 0


@dataclass(slots=True)
class Chrome:
    """Tier-3 engagement signals — rank, leaderboard, catalog standings, prizes.

    SEALED BOUNDARY INVARIANT (§10.1(d,e)): Chrome is for the human-facing UI and
    the records only. It MUST NEVER be serialized into ``AttemptState.messages``
    or any context window the model solves in. The engagement module enforces this;
    keeping it a separate object makes the boundary structural, not aspirational.
    """

    rank: int | None = None
    cohort_size: int | None = None
    leaderboard: list[dict[str, Any]] = field(default_factory=list)
    catalog_standing: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class AttemptState:
    """The single mutable bus threaded Designer -> Solver -> (Critic) -> Judge.

    The only function permitted to mutate ``output`` or call a model is the
    injected ``generate``/``step`` closure (§10.2), so all model I/O funnels
    through one observable choke point.

    ``messages`` is the *scored context* (Tier 1 + the deployment-plausible Tier-2
    engagement framing selected by ``framing_arm``). ``chrome`` (Tier 3) is held
    separately and never enters ``messages`` (see :class:`Chrome`).
    """

    attempt_id: str
    puzzle_id: str
    model: str
    framing_arm: FramingArm = FramingArm.SELF_REFERENTIAL
    messages: list[dict[str, Any]] = field(default_factory=list)
    output: str | None = None
    budget: Budget | None = None
    events: list[TraceEvent] = field(default_factory=list)
    scores: dict[str, Score] = field(default_factory=dict)  # e.g. {"oracle":..,"panel":..}
    usage: dict[str, Any] = field(default_factory=dict)     # tokens, cost
    wall_time: float = 0.0
    terminated_by: TerminatedBy | None = None
    error: str | None = None
    chrome: Chrome | None = None  # Tier-3; NEVER injected into `messages`
    metadata: dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Puzzle artifact schema (meta.json) — pydantic-validated (§1, §8.8)
# --------------------------------------------------------------------------- #


class Rewards(BaseModel):
    """Reward components. Bounded per Pan/Bhatia/Steinhardt 2022 (§8.3): elegance
    <= 30% of solve, novelty <= 50% (validated only)."""

    solve: float
    elegance_bonus_max: float = 0.0
    novelty_bonus_max: float = 0.0
    canonical_call_count: int = Field(ge=0, default=1)


class Penalty(BaseModel):
    """A declared answer-bypass penalty (§8.2). Per-puzzle, Goodhart-flavor-tagged."""

    name: str
    goodhart_flavor: GoodhartFlavor
    weight: float  # negative
    trigger: str
    description: str = ""


class PuzzleMeta(BaseModel):
    """``meta.json`` — the per-puzzle contract (§1 artifact structure + §8.8).

    The oracle is NOT described here in a Solver-readable way; it lives on the
    grading side (§10.4). This schema is what the kernel loads and the Designer
    reward surface validates against.
    """

    puzzle_id: str
    created_at: str                       # ISO 8601
    source_url: str | None = None         # GitHub issue / paper grounding the gap
    capability_aspect: str                # what the puzzle probes
    puzzle_class: PuzzleClass
    catalog_tier: CatalogTier = CatalogTier.LAB

    point_threshold: float
    time_budget_seconds: int = Field(gt=0)
    tool_call_budget: int = Field(gt=0)
    min_k: int = Field(ge=1, default=10)  # statistical floor (§1); graduation uses pass^k

    rewards: Rewards
    penalties: list[Penalty] = Field(default_factory=list)
    hard_kill_consecutive_identical: int = Field(ge=2, default=3)  # §8.4 WebArena loop
    novelty_validation_panel: str = "cross-family"

    @model_validator(mode="after")
    def _bound_components(self) -> PuzzleMeta:
        """Enforce the §8.3 component bounds so no single axis is an unbounded
        Goodhart magnet."""
        if self.rewards.elegance_bonus_max > 0.30 * self.rewards.solve:
            raise ValueError("elegance_bonus_max exceeds 30% of solve reward (§8.3)")
        if self.rewards.novelty_bonus_max > 0.50 * self.rewards.solve:
            raise ValueError("novelty_bonus_max exceeds 50% of solve reward (§8.3)")
        return self


# --------------------------------------------------------------------------- #
# Role contract (Protocol) — §10.3
# --------------------------------------------------------------------------- #


@runtime_checkable
class Role(Protocol):
    """Uniform role interface. Concrete roles (Designer/Solver/Critic/Judge/
    CohortSolver) implement ``act``. Only Solver is wired to sandbox tools; the
    Critic uses a string-in/string-out boundary with no env access (§10.3).
    """

    name: RoleName

    async def act(self, state: AttemptState) -> AttemptState:
        """Advance the attempt. Must route all model I/O through the kernel's
        injected ``generate`` closure (never call a model client directly), and
        must not read Tier-3 ``chrome``."""
        ...
