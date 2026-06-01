---
title: API reference
description: The public surface — kernel entry points, the small-N scoring functions, the oracle gate, and the core data contracts.
sidebar:
  order: 4
---

This page documents crucible's public Python surface. Signatures are drawn from the source; the prose
explains intent and the contract each call upholds.

## Kernel entry points

`crucible.kernel` is the thin integrator: it wires the leaf modules together into the two entry points
crucible runs on. Both are `async`.

### `run_attempt`

One Solver attempt against one puzzle, graded out-of-band.

```python
async def run_attempt(
    puzzle: LoadedPuzzle | Path,
    model: str,
    *,
    generate: GenerateFn,
    oracle_runner: OracleRunner,
    arm: FramingArm = FramingArm.SELF_REFERENTIAL,
    sandbox: SandboxEnvironment | None = None,
    judges: list[JudgeFn] | None = None,
    enable_critic: bool = False,
    chrome: Chrome | None = None,
    panel_reducer: str = "majority",
    generator_family: str | None = None,
    event_store: object | None = None,
    time_source: Callable[[], float] = time.monotonic,
) -> AttemptState
```

Key parameters:

- **`puzzle`** — a `LoadedPuzzle` or a path to a puzzle directory (loaded with the oracle held
  grading-side).
- **`generate`** — the single model-I/O choke point: `Callable[[AttemptState], Awaitable[str]]`. Every
  model call funnels through this.
- **`oracle_runner`** — the out-of-band grading edge:
  `Callable[[AttemptState, PuzzleMeta], Awaitable[OracleOutcome]]`. Stands in for the separate grading
  host; the kernel never reads the oracle itself.
- **`arm`** — which framing arm to render the scored context under (default `self_referential`).
- **`sandbox`** — the Solver's narrow environment channel; `None` runs a pure-reasoning puzzle with no
  environment.
- **`judges`** — cross-family judges for the panel; `None` skips the panel. When a novelty bonus is
  claimed, the panel is the validation authority.
- **`generator_family`** — the family of `model`, so the panel can structurally exclude same-family
  judges.
- **`time_source`** — the monotonic clock the kernel reads for the live time-budget check; injectable
  so the time enforcement is deterministically testable.

Returns the populated `AttemptState`: `messages` (the scored context, never chrome), `output`, `events`
(the kernel-owned trace), `scores` (at least `"oracle"`; `"panel"` when judged), `terminated_by`,
`budget`, and `wall_time`.

### `run_pass_hat_k`

`k` sibling attempts collected into the native `pass^k` unit.

```python
async def run_pass_hat_k(
    puzzle: LoadedPuzzle | Path,
    model: str,
    k: int,
    **kwargs: object,
) -> PuzzleHistory
```

Each sibling is an independent `run_attempt` with a fresh budget, governor, and trace; `**kwargs` are
forwarded unchanged (so the same `generate` / `oracle_runner` / `arm` / `sandbox` / `judges` apply to
every sibling). The puzzle is loaded once and shared. Raises `ValueError` if `k < 1`.

## Scoring functions

`crucible.scoring.stats` ships the small-N statistics. All functions are pure and deterministic —
the same inputs grade identically tomorrow.

```python
def pass_hat_k(successes: int, n: int, k: int) -> float
```
The probability that *all* `k` i.i.d. attempts succeed: the plug-in estimate `(successes / n) ** k`.
Measures consistency, not best-of-`k`. Raises `ValueError` on out-of-range counts or `k < 1`.

```python
def wilson_interval(successes: int, n: int, conf: float = 0.95) -> tuple[float, float]
```
The Wilson score confidence interval for a binomial proportion — the small-N-admissible interval the
graduation rule is built on. Returns `(lower, upper)` clamped to `[0.0, 1.0]`.

```python
def clopper_pearson(successes: int, n: int, conf: float = 0.95) -> tuple[float, float]
```
The conservative Clopper–Pearson *exact* interval (wider than Wilson), for when uncertainty must not
be understated.

```python
def mcnemar_exact(b: int, c: int) -> float
```
The exact McNemar paired two-sided p-value — the primary significance test for comparing two models on
the same puzzle set. `b` and `c` are the discordant-pair counts; only discordant pairs carry
information. Returns `1.0` when there are none.

```python
def graduates(successes: int, n: int) -> bool
```
The graduation rule, in one call: `True` iff the Wilson 95% interval satisfies
`0.10 ≤ lower ∧ upper ≤ 0.90` — neither trivial nor impossible.

### The oracle gate

`crucible.scoring.oracle` applies the conjunctive hard gate.

```python
@dataclass(slots=True)
class OracleOutcome:
    solved: bool
    solve_quality: float
    no_regression: bool
    tool_calls_used: int
    time_used: float
    triggered_penalties: list[str] = ...
    novelty_claimed: bool = False
    novelty_validated: bool = False

def grade(attempt: AttemptState, puzzle: PuzzleMeta, outcome: OracleOutcome) -> Score
```

`grade` opens the gate only when *all* hard conditions hold (solved-and-no-regression, solve quality
at or above `point_threshold`, no critical-flavor penalty, within tool and time budgets, and any
claimed novelty validated). Within the passing region the net score
(`solve + elegance − penalties`, plus a validated `novelty` bonus) is the tiebreaker; a failing
attempt returns `value = 0.0`. Either way, `Score.metadata` carries `gate_passed`, the component
breakdown, and the `failed_conditions`, so a failure is always legible. The critical
(gate-closing) flavor is exported as `CRITICAL_FLAVOR`.

### The judge panel

`crucible.scoring.judge_panel` is the external-verifier surface.

```python
class JudgePanel:
    def __init__(
        self,
        judges: list[JudgeFn],
        reducer: str = "majority",
        generator_family: str | None = None,
    ) -> None: ...

    def eligible_judges(self) -> list[JudgeFn]: ...
    async def score(self, attempt: AttemptState) -> Score: ...

def reduce_scores(scores: list[Score], method: str) -> Score
def judge_family(judge: JudgeFn) -> str | None
```

`score` runs the eligible judges concurrently (same-generator-family judges excluded) and reduces them
with `reducer` (`"majority"` or `"median"`). The reduced `Score.metadata` records the excluded
families, the eligible count, and the aggregated `novelty_validated` verdict. It raises `ValueError` if
exclusion would leave no eligible judges.

## Core data contracts

`crucible.types` is the exclusive home of every cross-module type. The most load-bearing:

### `AttemptState`

The single mutable bus threaded through every role.

```python
@dataclass(slots=True)
class AttemptState:
    attempt_id: str
    puzzle_id: str
    model: str
    framing_arm: FramingArm = FramingArm.SELF_REFERENTIAL
    messages: list[dict[str, Any]] = ...       # the SCORED context (Tier 1 + Tier 2)
    output: str | None = None
    budget: Budget | None = None
    events: list[TraceEvent] = ...
    scores: dict[str, Score] = ...             # e.g. {"oracle": ..., "panel": ...}
    usage: dict[str, Any] = ...
    wall_time: float = 0.0
    terminated_by: TerminatedBy | None = None
    error: str | None = None
    chrome: Chrome | None = None               # Tier-3; NEVER injected into `messages`
    metadata: dict[str, Any] = ...
```

The invariant: only the injected `generate` closure may mutate `output` or call a model, and `chrome`
never enters `messages`.

### `FramingArm`

```python
class FramingArm(StrEnum):
    NEUTRAL = "neutral"
    SELF_REFERENTIAL = "self_referential"   # default
    SOCIAL_STANDINGS = "social_standings"
```
Prompt framing as a first-class measured arm. `SOCIAL_STANDINGS` is retained only as a measured
variable and is rendered as chrome, never as the default scored context.

### `PuzzleMeta`

The validated `meta.json` contract (a Pydantic model).

```python
class PuzzleMeta(BaseModel):
    puzzle_id: str
    created_at: str
    source_url: str | None = None
    capability_aspect: str
    puzzle_class: PuzzleClass
    catalog_tier: CatalogTier = CatalogTier.LAB
    point_threshold: float
    time_budget_seconds: int = Field(gt=0)
    tool_call_budget: int = Field(gt=0)
    min_k: int = Field(ge=1, default=10)
    rewards: Rewards
    penalties: list[Penalty] = ...
    hard_kill_consecutive_identical: int = Field(ge=2, default=3)
    novelty_validation_panel: str = "cross-family"
```

A model validator enforces the component bounds at load time: `elegance_bonus_max` may not exceed 30%
of the solve reward, and `novelty_bonus_max` may not exceed 50% — so a misconfigured puzzle fails fast
rather than shipping an unbounded gaming magnet. The supporting `Rewards` and `Penalty` models, and
the `PuzzleClass` / `CatalogTier` / `GoodhartFlavor` enums, live alongside it.

### `Role`

The uniform role protocol every participant implements.

```python
@runtime_checkable
class Role(Protocol):
    name: RoleName
    async def act(self, state: AttemptState) -> AttemptState: ...
```

An `act` must route all model I/O through the kernel's injected `generate` closure and must not read
Tier-3 `chrome`. The five concrete roles (`Designer`, `Solver`, `Critic`, `Judge`, `CohortSolver`)
live in `crucible.roles`; their slots are named by the `RoleName` enum.
