"""End-to-end tests for the Crucible kernel (crucible.kernel).

The kernel is the Wave-2 integrator: it composes the Wave-1 leaf modules into
:func:`run_attempt` and :func:`run_pass_hat_k`. These tests drive the *real*
pipeline against the real sample puzzle fixture with injected fakes — a fake
``generate`` (canned model output) and a fake ``oracle_runner`` (canned
:class:`OracleOutcome`) — so the composition is exercised without a model runtime
or a grading host.

Async is driven with ``anyio.run`` (the repo convention; anyio ships with
inspect-ai, no pytest-asyncio).

Invariants proven (happy path + the RED paths the spec requires):
- A full :func:`run_attempt` on the sample fixture produces an AttemptState with
  ``scores["oracle"]``, a populated trace, and ``terminated_by == COMPLETED``.
- A looping fake ``generate`` (3 identical tool calls) trips the kernel-side
  hard-kill ANDON → ``terminated_by == HARD_KILL``.
- A fake ``generate`` exceeding the TOOL budget (distinct calls) → ``terminated_by
  == BUDGET`` (M3); a fake clock past the time budget → ``terminated_by == TIME``
  via the LIVE kernel-side ``check_time`` (H3 / M3).
- A chrome leak (a chrome value present in the scored context) raises
  :class:`SealedBoundaryViolation` BEFORE the Solver runs (RED, §10.1(e)); the
  kernel ALSO re-checks AFTER the Solver and Critic turns, catching an in-place
  message edit a per-turn role guard misses (H1, §10.1(e)).
- ``run_pass_hat_k(..., k=3)`` returns a :class:`PuzzleHistory` with 3 attempts.
- The oracle is NEVER present in ``attempt.messages`` (sealed boundary, §10.4).
"""

from __future__ import annotations

from pathlib import Path

import anyio
import pytest

from crucible.characterize.aggregate import compose_panel
from crucible.characterize.types import (
    JudgeProfile,
    JudgmentRecord,
    RoleSlot,
    SeatDecision,
)
from crucible.engagement import SealedBoundaryViolation, build_chrome
from crucible.kernel import run_attempt, run_pass_hat_k
from crucible.observability import PuzzleHistory
from crucible.puzzle import LoadedPuzzle, load_puzzle
from crucible.scoring.judge_panel import JudgePanel
from crucible.scoring.oracle import OracleOutcome
from crucible.types import (
    AttemptState,
    Chrome,
    FramingArm,
    PuzzleMeta,
    Score,
    TerminatedBy,
)

# --------------------------------------------------------------------------- #
# Fixtures + test doubles
# --------------------------------------------------------------------------- #

SAMPLE_PUZZLE_DIR = Path(__file__).parent / "fixtures" / "puzzles" / "sample"

# The sample fixture stages config.py with MAX_RETRIES = 7; the canonical answer.
_CORRECT_ANSWER = "7"


@pytest.fixture
def sample_dir() -> Path:
    """The on-disk sample puzzle directory (read by load_puzzle)."""
    assert SAMPLE_PUZZLE_DIR.is_dir(), f"sample fixture missing: {SAMPLE_PUZZLE_DIR}"
    return SAMPLE_PUZZLE_DIR


def fake_generate(text: str):
    """A fake model-I/O choke point returning canned ``text``, recording calls.

    This is the simplest Solver: it does no tool calls and returns the answer
    string directly — exactly what a clean one-shot solve looks like.
    """
    state = {"calls": 0}

    async def _gen(_attempt: AttemptState) -> str:
        state["calls"] += 1
        return text

    _gen.calls = state  # type: ignore[attr-defined]
    return _gen


def tool_using_generate(text: str, *, calls: list[tuple[str, dict]]):
    """A fake ``generate`` that drives real tool calls through the kernel governor.

    It reaches the live Solver parked on ``state.metadata`` by the kernel and
    records each ``(tool, args)`` in ``calls`` through ``record_tool_call`` — the
    only legitimate accounting path (§10.2 / §8.4). Returns ``text`` after.
    """
    from crucible.kernel import _SOLVER_HANDLE

    async def _gen(attempt: AttemptState) -> str:
        solver = attempt.metadata[_SOLVER_HANDLE]
        for tool, args in calls:
            await solver.record_tool_call(tool, args)
        return text

    return _gen


def looping_generate():
    """A fake ``generate`` that loops on one identical tool call (pathological).

    Three consecutive identical ``(tool, args)`` calls is the §8.4 hard-kill
    pattern; the third raises ``BudgetExceeded(HARD_KILL)`` inside the governor,
    which :meth:`Solver.act` catches and stamps onto ``terminated_by``.
    """
    from crucible.kernel import _SOLVER_HANDLE

    async def _gen(attempt: AttemptState) -> str:
        solver = attempt.metadata[_SOLVER_HANDLE]
        for _ in range(3):
            await solver.record_tool_call("read_file", {"path": "config.py"})
        return "unreachable"  # pragma: no cover - hard-kill fires before this

    return _gen


def tool_budget_overrun_generate(budget: int):
    """A fake ``generate`` that makes ``budget + 1`` *distinct* tool calls.

    Distinct ``(tool, args)`` avoids the hard-kill loop signal so the kernel halts
    on the tool-call budget specifically — the call that would push usage past
    ``tool_call_budget`` raises ``BudgetExceeded(BUDGET)`` inside the governor,
    which :meth:`Solver.act` catches and stamps onto ``terminated_by`` (§8.4 ANDON).
    """
    from crucible.kernel import _SOLVER_HANDLE

    async def _gen(attempt: AttemptState) -> str:
        solver = attempt.metadata[_SOLVER_HANDLE]
        for i in range(budget + 1):  # one over budget
            await solver.record_tool_call("read_file", {"path": f"file_{i}.py"})
        return "unreachable"  # pragma: no cover - budget fires before this

    return _gen


class FakeClock:
    """A deterministic monotonic clock the kernel can read for time accounting.

    ``run_attempt`` reads wall-clock through an injected ``time_source`` so the
    live time-budget check (§8.4 / H3) is testable without real sleeps. Each call
    returns the next queued tick; the last tick repeats once exhausted so a single
    extra read (the attempt-boundary check) is always defined.
    """

    def __init__(self, ticks: list[float]) -> None:
        self._ticks = ticks
        self._i = 0

    def __call__(self) -> float:
        tick = self._ticks[min(self._i, len(self._ticks) - 1)]
        self._i += 1
        return tick


def time_overrun_generate():
    """A fake ``generate`` that does a single benign tool call then returns.

    Paired with a :class:`FakeClock` whose ticks jump past ``time_budget_seconds``,
    the kernel's per-turn / attempt-boundary :meth:`BudgetGovernor.check_time`
    sees the overrun and raises ``BudgetExceeded(TIME)`` live (H3), which the
    Solver stamps onto ``terminated_by``.
    """
    from crucible.kernel import _SOLVER_HANDLE

    async def _gen(attempt: AttemptState) -> str:
        solver = attempt.metadata[_SOLVER_HANDLE]
        await solver.record_tool_call("read_file", {"path": "config.py"})
        return _CORRECT_ANSWER  # pragma: no cover - time check fires at/after this

    return _gen


def oracle_runner_for(outcome: OracleOutcome):
    """A fake out-of-band ``oracle_runner`` returning a canned ``outcome``.

    Stands in for the separate grading host that reads the sealed oracle against
    the copied-out workdir (§10.4). It receives the halted attempt + the puzzle
    contract and returns the task verdict — the kernel never reads the oracle.
    """

    async def _run(_attempt: AttemptState, _meta: PuzzleMeta) -> OracleOutcome:
        return outcome

    return _run


def clean_solve_outcome(meta: PuzzleMeta) -> OracleOutcome:
    """An OracleOutcome that clears the §8.3 conjunctive gate for ``meta``."""
    return OracleOutcome(
        solved=True,
        solve_quality=meta.point_threshold + 10.0,  # comfortably over threshold
        no_regression=True,
        tool_calls_used=1,
        time_used=1.0,
    )


def make_judge(verdict: bool | float, *, family: str | None = None):
    """A fake cross-family judge callable tagged with a model ``family``."""

    async def _judge(_attempt: AttemptState) -> Score:
        return Score(value=verdict)

    if family is not None:
        _judge.family = family  # type: ignore[attr-defined]
    return _judge


def make_novelty_judge(
    novelty_validated: bool, *, verdict: bool = True, family: str | None = None
):
    """A fake judge that carries a per-judge ``novelty_validated`` vote (§8.7).

    The panel adjudicates novelty by aggregating each judge's ``novelty_validated``
    metadata vote (the panel — not the oracle_runner — is the authority). ``verdict``
    is the judge's primary legitimacy value (the existing panel ``value`` axis)."""

    async def _judge(_attempt: AttemptState) -> Score:
        return Score(value=verdict, metadata={"novelty_validated": novelty_validated})

    if family is not None:
        _judge.family = family  # type: ignore[attr-defined]
    return _judge


def novelty_claimed_outcome(meta: PuzzleMeta, *, runner_validated: bool) -> OracleOutcome:
    """A clean solve that CLAIMS novelty, with the oracle_runner reporting
    ``novelty_validated=runner_validated``.

    The oracle_runner's novelty_validated must be INERT for the gate (the panel is
    the authority, §8.7) — these tests set ``runner_validated`` adversarially to
    prove the kernel does not trust it. Uses ``tool_calls_used == canonical`` so
    elegance is 0 and the novelty bonus is the only optional component (clean
    assertion on the score value)."""
    return OracleOutcome(
        solved=True,
        solve_quality=meta.point_threshold + 10.0,
        no_regression=True,
        tool_calls_used=meta.rewards.canonical_call_count,  # == canonical → elegance 0
        time_used=1.0,
        novelty_claimed=True,
        novelty_validated=runner_validated,
    )


# --------------------------------------------------------------------------- #
# 1. Full run_attempt happy path on the real sample fixture
# --------------------------------------------------------------------------- #


def test_run_attempt_happy_path_on_sample_fixture(sample_dir: Path) -> None:
    """A full attempt on the sample fixture: oracle score, populated trace, COMPLETED."""
    loaded = load_puzzle(sample_dir)
    gen = fake_generate(_CORRECT_ANSWER)

    attempt = anyio.run(
        lambda: run_attempt(
            sample_dir,
            "claude-opus-4-8",
            generate=gen,
            oracle_runner=oracle_runner_for(clean_solve_outcome(loaded.meta)),
        )
    )

    # Terminal status is a clean completion.
    assert attempt.terminated_by is TerminatedBy.COMPLETED
    # The Solver produced output through the choke point exactly once.
    assert attempt.output == _CORRECT_ANSWER
    assert gen.calls["calls"] == 1
    # The oracle score is present and the gate passed (clean solve over threshold).
    assert "oracle" in attempt.scores
    assert attempt.scores["oracle"].metadata["gate_passed"] is True
    # The trace is populated and rendered into an Inspect-shaped eval-log.
    assert attempt.events, "expected a populated trace"
    eval_log = attempt.metadata["eval_log"]
    assert eval_log["samples"][0]["id"] == attempt.attempt_id
    assert eval_log["results"]["terminated_by"] == "completed"
    # The scored context was built (Tier-1/Tier-2), and the budget is live.
    assert attempt.messages
    assert attempt.budget is not None


def test_run_attempt_path_or_loaded_puzzle_equivalent(sample_dir: Path) -> None:
    """run_attempt accepts either a Path or a pre-loaded LoadedPuzzle."""
    loaded = load_puzzle(sample_dir)
    assert isinstance(loaded, LoadedPuzzle)
    outcome = clean_solve_outcome(loaded.meta)

    by_path = anyio.run(
        lambda: run_attempt(
            sample_dir, "m", generate=fake_generate(_CORRECT_ANSWER),
            oracle_runner=oracle_runner_for(outcome),
        )
    )
    by_loaded = anyio.run(
        lambda: run_attempt(
            loaded, "m", generate=fake_generate(_CORRECT_ANSWER),
            oracle_runner=oracle_runner_for(outcome),
        )
    )
    assert by_path.scores["oracle"].metadata["gate_passed"]
    assert by_loaded.scores["oracle"].metadata["gate_passed"]
    assert by_path.puzzle_id == by_loaded.puzzle_id == loaded.meta.puzzle_id


# --------------------------------------------------------------------------- #
# 2. Hard-kill ANDON path
# --------------------------------------------------------------------------- #


def test_run_attempt_hard_kill_on_looping_generate(sample_dir: Path) -> None:
    """A looping generate (3 identical tool calls) → terminated_by == HARD_KILL."""
    loaded = load_puzzle(sample_dir)

    attempt = anyio.run(
        lambda: run_attempt(
            sample_dir,
            "m",
            generate=looping_generate(),
            oracle_runner=oracle_runner_for(clean_solve_outcome(loaded.meta)),
        )
    )

    assert attempt.terminated_by is TerminatedBy.HARD_KILL
    # The Solver halted before producing output.
    assert attempt.output is None
    # The kernel still grades out-of-band even on a halted attempt (the gate then
    # closes because the attempt did not actually solve cleanly under the loop) —
    # but crucially the oracle score is still recorded for the trace.
    assert "oracle" in attempt.scores
    # An error event for the hard kill is in the trace.
    assert any(e.kind == "error" for e in attempt.events)


def test_hard_kill_attempt_is_not_a_solve(sample_dir: Path) -> None:
    """A hard-killed attempt never counts as solved in a PuzzleHistory rollup."""
    loaded = load_puzzle(sample_dir)
    attempt = anyio.run(
        lambda: run_attempt(
            sample_dir, "m", generate=looping_generate(),
            # Even if the oracle outcome claims solved, a non-COMPLETED terminal
            # status must not count as a solve (terminal-world-state, finding 6).
            oracle_runner=oracle_runner_for(clean_solve_outcome(loaded.meta)),
        )
    )
    hist = PuzzleHistory(puzzle_id=loaded.meta.puzzle_id)
    hist.add(attempt)
    assert hist.outcomes == [False]


# --------------------------------------------------------------------------- #
# 2b. Tool-call budget ANDON (M3) + live time-budget ANDON (H3 / M3)
# --------------------------------------------------------------------------- #


def test_run_attempt_tool_budget_overrun_terminates_by_budget(sample_dir: Path) -> None:
    """A generate that exceeds the TOOL budget (distinct calls, no loop) →
    terminated_by == BUDGET (M3, mirrors the hard-kill e2e test)."""
    loaded = load_puzzle(sample_dir)
    budget = loaded.meta.tool_call_budget

    attempt = anyio.run(
        lambda: run_attempt(
            sample_dir,
            "m",
            generate=tool_budget_overrun_generate(budget),
            oracle_runner=oracle_runner_for(clean_solve_outcome(loaded.meta)),
        )
    )

    assert attempt.terminated_by is TerminatedBy.BUDGET
    # Halted before producing output; an error event records the budget breach.
    assert attempt.output is None
    assert any(e.kind == "error" for e in attempt.events)
    # The governor admitted exactly the budget, then refused the over-budget call.
    assert attempt.budget is not None
    assert attempt.budget.tool_calls_used == budget


def test_run_attempt_time_budget_overrun_terminates_by_time(sample_dir: Path) -> None:
    """LIVE time enforcement (H3 / M3): when the injected clock shows elapsed past
    time_budget_seconds, the kernel calls governor.check_time and the attempt ends
    with terminated_by == TIME — not merely flagged post-hoc by the oracle."""
    loaded = load_puzzle(sample_dir)
    budget_s = loaded.meta.time_budget_seconds
    # Clock: start at 0, then jump well past the time budget for the in-turn check.
    clock = FakeClock([0.0, float(budget_s) + 1.0, float(budget_s) + 2.0])

    attempt = anyio.run(
        lambda: run_attempt(
            sample_dir,
            "m",
            generate=time_overrun_generate(),
            oracle_runner=oracle_runner_for(clean_solve_outcome(loaded.meta)),
            time_source=clock,
        )
    )

    assert attempt.terminated_by is TerminatedBy.TIME
    assert any(e.kind == "error" for e in attempt.events)
    # The live elapsed was recorded on the budget by check_time (kernel-side).
    assert attempt.budget is not None
    assert attempt.budget.elapsed_seconds > budget_s


def test_run_attempt_within_time_budget_completes(sample_dir: Path) -> None:
    """A clock that stays under the time budget does not trip the live check —
    the attempt completes normally (guards against a false-positive TIME halt)."""
    loaded = load_puzzle(sample_dir)
    clock = FakeClock([0.0, 1.0, 2.0])  # well under the 600s budget

    attempt = anyio.run(
        lambda: run_attempt(
            sample_dir,
            "m",
            generate=fake_generate(_CORRECT_ANSWER),
            oracle_runner=oracle_runner_for(clean_solve_outcome(loaded.meta)),
            time_source=clock,
        )
    )
    assert attempt.terminated_by is TerminatedBy.COMPLETED


# --------------------------------------------------------------------------- #
# 3. Sealed-boundary RED path — chrome leak halts BEFORE the Solver
# --------------------------------------------------------------------------- #


def test_chrome_leak_raises_before_solver_runs(sample_dir: Path) -> None:
    """A chrome value present in the scored context raises SealedBoundaryViolation
    BEFORE any model call (prove the andon goes RED, §10.1(e))."""
    loaded = load_puzzle(sample_dir)
    # Simulate a bad framing: a puzzle whose prompt text carries a value that also
    # appears in Tier-3 chrome (rank 4242). build_scored_context will place the
    # prompt into messages; assert_no_chrome_leak must then catch the leaked rank.
    leaky_meta = loaded.meta.model_copy()
    leaky = LoadedPuzzle(
        meta=leaky_meta,
        prompt="Report MAX_RETRIES. Note: your leaderboard rank is 4242.",
        setup_script=loaded.setup_script,
        root=loaded.root,
    )
    chrome: Chrome = build_chrome(rank=4242, cohort_size=12)

    # The generate must NEVER be called — assert by raising if it is.
    def exploding_generate():
        async def _gen(_attempt: AttemptState) -> str:  # pragma: no cover - must not run
            raise AssertionError("Solver ran despite a chrome leak — boundary failed")

        return _gen

    with pytest.raises(SealedBoundaryViolation):
        anyio.run(
            lambda: run_attempt(
                leaky,
                "m",
                generate=exploding_generate(),
                oracle_runner=oracle_runner_for(clean_solve_outcome(leaky_meta)),
                chrome=chrome,
            )
        )


def test_clean_chrome_does_not_block(sample_dir: Path) -> None:
    """A populated-but-non-leaking chrome passes the guard and the attempt runs."""
    loaded = load_puzzle(sample_dir)
    # rank value that does NOT appear in the sample prompt → no leak.
    chrome = build_chrome(rank=999999, cohort_size=8)
    attempt = anyio.run(
        lambda: run_attempt(
            sample_dir, "m", generate=fake_generate(_CORRECT_ANSWER),
            oracle_runner=oracle_runner_for(clean_solve_outcome(loaded.meta)),
            chrome=chrome,
        )
    )
    assert attempt.terminated_by is TerminatedBy.COMPLETED
    # Chrome is held on the attempt but never entered the scored context.
    assert attempt.chrome is chrome
    assert not _chrome_in_messages(attempt)


# --------------------------------------------------------------------------- #
# 3b. Sealed-boundary RED — kernel re-checks AFTER the Solver / Critic turns (H1)
# --------------------------------------------------------------------------- #


def _in_place_leaking_generate(token: str, *, return_text: str = _CORRECT_ANSWER):
    """A fake ``generate`` that splices ``token`` into an EXISTING scored message.

    In-place mutation is the leak shape a *per-turn* role guard cannot see (the
    role guard only inspects messages *appended* during its own turn, not edits to
    messages already present). It is exactly why the kernel must re-run
    :func:`assert_no_chrome_leak` over the FULL message list after each
    message-mutating turn (H1, the "family of message-mutating sites"). The kernel
    re-check raises :class:`SealedBoundaryViolation`, which propagates out.
    """

    async def _gen(attempt: AttemptState) -> str:
        # Edit the first scored message in place to inject the chrome token.
        if attempt.messages:
            attempt.messages[0] = dict(attempt.messages[0])
            attempt.messages[0]["content"] = (
                str(attempt.messages[0].get("content", "")) + f" [leak: {token}]"
            )
        return return_text

    return _gen


def _critic_leaking_generate(token: str):
    """A fake ``generate`` shared by Solver+Critic that leaks only on the CRITIC
    turn: the first call (Solver) returns clean; the second call (Critic) splices
    ``token`` into an existing message in place. Isolates the post-Critic re-check.
    """
    state = {"calls": 0}

    async def _gen(attempt: AttemptState) -> str:
        state["calls"] += 1
        if state["calls"] == 1:
            return _CORRECT_ANSWER  # Solver: clean
        # Critic turn: in-place leak the kernel's post-Critic re-check must catch.
        if attempt.messages:
            attempt.messages[0] = dict(attempt.messages[0])
            attempt.messages[0]["content"] = (
                str(attempt.messages[0].get("content", "")) + f" [leak: {token}]"
            )
        return "a critique"

    return _gen


def test_chrome_leak_after_solver_turn_raises(sample_dir: Path) -> None:
    """A leak introduced by the Solver turn via in-place message edit is caught by
    the kernel's POST-Solver re-check → SealedBoundaryViolation (H1, §10.1(e)).

    The per-turn role guard misses an in-place edit (it only sees appended
    messages); the kernel re-check over the full context is what closes the gap."""
    loaded = load_puzzle(sample_dir)
    # A string-rendered chrome value (solver-id) the strong guard tokenizes.
    chrome = build_chrome(
        rank=11, cohort_size=12, leaderboard=[{"solver": "solver-omega"}]
    )
    with pytest.raises(SealedBoundaryViolation):
        anyio.run(
            lambda: run_attempt(
                sample_dir,
                "m",
                generate=_in_place_leaking_generate("solver-omega"),
                oracle_runner=oracle_runner_for(clean_solve_outcome(loaded.meta)),
                chrome=chrome,
            )
        )


def test_chrome_leak_after_critic_turn_raises(sample_dir: Path) -> None:
    """A leak introduced by the CRITIC turn (in-place edit) is caught by the
    kernel's POST-Critic re-check → SealedBoundaryViolation (H1, §10.1(e)).

    The Solver runs clean; the enabled Critic mutates an existing message in place.
    Only the kernel's post-Critic re-check covers this message-mutating site."""
    loaded = load_puzzle(sample_dir)
    chrome = build_chrome(rank=11, cohort_size=12, leaderboard=[{"solver": "solver-omega"}])
    with pytest.raises(SealedBoundaryViolation):
        anyio.run(
            lambda: run_attempt(
                sample_dir,
                "m",
                # Solver returns clean; the Critic generate does the in-place leak.
                generate=_critic_leaking_generate("solver-omega"),
                oracle_runner=oracle_runner_for(clean_solve_outcome(loaded.meta)),
                chrome=chrome,
                enable_critic=True,
            )
        )


# --------------------------------------------------------------------------- #
# 4. pass^k native: k sibling attempts → PuzzleHistory
# --------------------------------------------------------------------------- #


def test_run_pass_hat_k_returns_history_with_k_attempts(sample_dir: Path) -> None:
    """run_pass_hat_k(k=3) returns a PuzzleHistory with exactly 3 sibling attempts."""
    loaded = load_puzzle(sample_dir)

    history = anyio.run(
        lambda: run_pass_hat_k(
            sample_dir,
            "m",
            3,
            generate=fake_generate(_CORRECT_ANSWER),
            oracle_runner=oracle_runner_for(clean_solve_outcome(loaded.meta)),
        )
    )

    assert isinstance(history, PuzzleHistory)
    assert history.n_attempts == 3
    assert history.puzzle_id == loaded.meta.puzzle_id
    # All three were clean solves over threshold → pass^3 == 1.0.
    assert history.outcomes == [True, True, True]
    assert history.pass_hat_k(3) == pytest.approx(1.0)
    # Each sibling is an independent attempt with its own id.
    ids = {a.attempt_id for a in history.attempts}
    assert len(ids) == 3


def test_run_pass_hat_k_rejects_k_below_one(sample_dir: Path) -> None:
    """k < 1 is rejected — pass^k needs at least one trial."""
    loaded = load_puzzle(sample_dir)
    with pytest.raises(ValueError, match="k >= 1"):
        anyio.run(
            lambda: run_pass_hat_k(
                sample_dir, "m", 0,
                generate=fake_generate(_CORRECT_ANSWER),
                oracle_runner=oracle_runner_for(clean_solve_outcome(loaded.meta)),
            )
        )


# --------------------------------------------------------------------------- #
# 5. The oracle is never present in attempt.messages (sealed boundary, §10.4)
# --------------------------------------------------------------------------- #


def test_oracle_never_in_messages(sample_dir: Path) -> None:
    """No oracle/answer artifact ever appears in the Solver-visible scored context."""
    loaded = load_puzzle(sample_dir)
    # An oracle outcome carrying the literal answer; it must never reach messages.
    outcome = OracleOutcome(
        solved=True,
        solve_quality=loaded.meta.point_threshold + 5.0,
        no_regression=True,
        tool_calls_used=1,
        time_used=1.0,
    )
    attempt = anyio.run(
        lambda: run_attempt(
            sample_dir, "m", generate=fake_generate(_CORRECT_ANSWER),
            oracle_runner=oracle_runner_for(outcome),
        )
    )
    # The scored context (input) carries no oracle/answer-key markers, and the
    # rendered eval-log records target=None (oracle stays grading-side).
    blob = repr(attempt.messages).lower()
    for forbidden in ("oracle", "answer_key", "answer key", "solution_a", "gold commit"):
        assert forbidden not in blob, f"oracle marker {forbidden!r} leaked into messages"
    eval_log = attempt.metadata["eval_log"]
    assert eval_log["samples"][0]["target"] is None


# --------------------------------------------------------------------------- #
# Judge panel wiring — EXTERNAL_VERIFIER generator-family exclusion (§10.2)
# --------------------------------------------------------------------------- #


def test_panel_scores_and_excludes_generator_family(sample_dir: Path) -> None:
    """With judges supplied, the panel scores into scores["panel"] and a same-family
    judge is excluded (EXTERNAL_VERIFIER)."""
    loaded = load_puzzle(sample_dir)
    judges = [
        make_judge(True, family="qwen"),
        make_judge(True, family="mistral"),
        make_judge(False, family="claude"),  # same family as generator → excluded
    ]
    attempt = anyio.run(
        lambda: run_attempt(
            sample_dir,
            "claude-opus-4-8",
            generate=fake_generate(_CORRECT_ANSWER),
            oracle_runner=oracle_runner_for(clean_solve_outcome(loaded.meta)),
            judges=judges,
            generator_family="claude",
        )
    )
    assert "panel" in attempt.scores
    panel = attempt.scores["panel"]
    # The claude judge was excluded; only qwen + mistral voted (both True).
    assert panel.metadata["excluded"] == ["claude"]
    assert panel.metadata["eligible_count"] == 2
    assert panel.value is True


def test_no_judges_means_no_panel_score(sample_dir: Path) -> None:
    """Without judges, only the oracle score is present (panel is opt-in)."""
    loaded = load_puzzle(sample_dir)
    attempt = anyio.run(
        lambda: run_attempt(
            sample_dir, "m", generate=fake_generate(_CORRECT_ANSWER),
            oracle_runner=oracle_runner_for(clean_solve_outcome(loaded.meta)),
        )
    )
    assert "oracle" in attempt.scores
    assert "panel" not in attempt.scores


# --------------------------------------------------------------------------- #
# Novelty is PANEL-adjudicated, not oracle_runner-trusted (H2, §8.3/§8.7)
# --------------------------------------------------------------------------- #


def test_e2e_composed_weighted_panel_drives_kernel_gate(sample_dir: Path) -> None:
    """End-to-end arc (§11.4): characterization profiles → compose_panel →
    JudgePanel.from_seated → run_attempt(panel=...). The seated reliability weights
    decide the panel verdict, and the cross-family panel — not the oracle_runner —
    adjudicates the claimed novelty bonus. This locks the whole vertical slice together:
    characterize → compose → score."""
    loaded = load_puzzle(sample_dir)

    def _prof(model_id: str, weight: float) -> JudgeProfile:
        return JudgeProfile(
            model_id=model_id,
            role=RoleSlot.JUDGE,
            n_items=4,
            reliability_weight=weight,
            seat_decision=SeatDecision.SEAT,
            metadata={},
        )

    def _perfect(model_id: str) -> list[JudgmentRecord]:
        # perfect → zero-variance error vector → ρ=0 with every peer → all three seat
        return [
            JudgmentRecord(
                item_id=f"i{j}", model_id=model_id, predicted=1, gold=1,
                correct=True, family=model_id,
            )
            for j in range(4)
        ]

    profiles = {
        "qwen": _prof("qwen", 0.9),
        "mistral": _prof("mistral", 0.1),
        "cohere": _prof("cohere", 0.1),
    }
    records = {m: _perfect(m) for m in profiles}

    seated = compose_panel(profiles, records)
    assert seated.meets_quorum and not seated.escalate
    # qwen first (w=0.9); the two w=0.1 seats tie → ordered by model_id (cohere < mistral)
    assert [s.model_id for s in seated.seats] == ["qwen", "cohere", "mistral"]

    # The reliable qwen rules the attempt ILLEGITIMATE but all three VALIDATE novelty.
    legitimacy = {"qwen": False, "mistral": True, "cohere": True}

    def judge_for(model_id: str):
        async def _j(_a: AttemptState) -> Score:
            return Score(value=legitimacy[model_id], metadata={"novelty_validated": True})

        return _j

    panel = JudgePanel.from_seated(seated, judge_for, generator_family="claude")

    attempt = anyio.run(
        lambda: run_attempt(
            sample_dir,
            "claude-opus-4-8",
            generate=fake_generate(_CORRECT_ANSWER),
            # adversarial: oracle_runner says novelty NOT validated — must be ignored.
            oracle_runner=oracle_runner_for(
                novelty_claimed_outcome(loaded.meta, runner_validated=False)
            ),
            panel=panel,
        )
    )

    panel_score = attempt.scores["panel"]
    # the composed, reliability-weighted panel ran through the kernel:
    assert panel_score.metadata["reducer"] == "weighted"
    assert panel_score.value is False  # 0.9 (False) outweighs 0.1 + 0.1 (True)
    assert panel_score.metadata["margin"] == pytest.approx(0.9 / 1.1)
    # the panel (not the oracle_runner's False) governed the novelty bonus → reached the gate:
    assert panel_score.metadata["novelty_validated"] is True
    oracle = attempt.scores["oracle"]
    assert oracle.metadata["gate_passed"] is True
    assert oracle.metadata["components"]["novelty"] == pytest.approx(
        loaded.meta.rewards.novelty_bonus_max
    )


def test_panel_validated_novelty_reaches_the_gate_value(sample_dir: Path) -> None:
    """When the PANEL validates novelty, the bonus reaches the oracle score value —
    even though the oracle_runner reported novelty_validated=False (H2, §8.7).

    The panel — a cross-family verifier — is the novelty authority; its verdict, not
    the oracle_runner's self-report, drives the gate + bonus."""
    loaded = load_puzzle(sample_dir)
    judges = [
        make_novelty_judge(True, family="qwen"),
        make_novelty_judge(True, family="mistral"),
    ]
    attempt = anyio.run(
        lambda: run_attempt(
            sample_dir,
            "claude-opus-4-8",
            generate=fake_generate(_CORRECT_ANSWER),
            # oracle_runner says NOT validated — must be ignored for novelty.
            oracle_runner=oracle_runner_for(
                novelty_claimed_outcome(loaded.meta, runner_validated=False)
            ),
            judges=judges,
            generator_family="claude",
        )
    )
    oracle = attempt.scores["oracle"]
    assert oracle.metadata["gate_passed"] is True
    # grade() uses outcome.solve_quality as the solve component (not rewards.solve);
    # tool_calls == canonical → elegance 0; so value = solve_quality + novelty bonus.
    solve_quality = loaded.meta.point_threshold + 10.0  # per novelty_claimed_outcome
    expected = solve_quality + loaded.meta.rewards.novelty_bonus_max
    assert oracle.value == pytest.approx(expected)
    assert oracle.metadata["components"]["novelty"] == pytest.approx(
        loaded.meta.rewards.novelty_bonus_max
    )
    # The panel surfaced the novelty verdict it adjudicated.
    assert attempt.scores["panel"].metadata["novelty_validated"] is True


def test_panel_rejected_novelty_blocks_bonus_even_if_runner_said_true(
    sample_dir: Path,
) -> None:
    """LOAD-BEARING (H2, §8.7): the panel REJECTS novelty → no bonus AND the gate
    closes (novelty_unvalidated), EVEN THOUGH the oracle_runner reported
    novelty_validated=True. Proves the panel overrides the oracle_runner."""
    loaded = load_puzzle(sample_dir)
    judges = [
        make_novelty_judge(False, family="qwen"),
        make_novelty_judge(False, family="mistral"),
    ]
    attempt = anyio.run(
        lambda: run_attempt(
            sample_dir,
            "claude-opus-4-8",
            generate=fake_generate(_CORRECT_ANSWER),
            # The trap: oracle_runner CLAIMS the novelty is validated.
            oracle_runner=oracle_runner_for(
                novelty_claimed_outcome(loaded.meta, runner_validated=True)
            ),
            judges=judges,
            generator_family="claude",
        )
    )
    oracle = attempt.scores["oracle"]
    # Panel said no → an unvalidated novelty claim closes the gate (§8.3).
    assert oracle.metadata["gate_passed"] is False
    assert "novelty_unvalidated" in oracle.metadata["failed_conditions"]
    assert oracle.value == 0.0
    assert attempt.scores["panel"].metadata["novelty_validated"] is False


def test_novelty_claim_without_judges_is_not_validated(sample_dir: Path) -> None:
    """A novelty claim with NO panel cannot be validated — the oracle_runner is not
    the authority, so the gate closes even if the runner self-reports validated
    (H2, §8.7). Absent a panel, novelty is never auto-granted."""
    loaded = load_puzzle(sample_dir)
    attempt = anyio.run(
        lambda: run_attempt(
            sample_dir,
            "claude-opus-4-8",
            generate=fake_generate(_CORRECT_ANSWER),
            oracle_runner=oracle_runner_for(
                novelty_claimed_outcome(loaded.meta, runner_validated=True)
            ),
            # no judges supplied
        )
    )
    oracle = attempt.scores["oracle"]
    assert oracle.metadata["gate_passed"] is False
    assert "novelty_unvalidated" in oracle.metadata["failed_conditions"]


# --------------------------------------------------------------------------- #
# Critic opt-in (default OFF, §10.3)
# --------------------------------------------------------------------------- #


def test_critic_default_off_does_not_add_critique(sample_dir: Path) -> None:
    """By default the Critic is OFF and adds no critique message (§10.3)."""
    loaded = load_puzzle(sample_dir)
    attempt = anyio.run(
        lambda: run_attempt(
            sample_dir, "m", generate=fake_generate(_CORRECT_ANSWER),
            oracle_runner=oracle_runner_for(clean_solve_outcome(loaded.meta)),
        )
    )
    assert not any(m.get("role") == "critic" for m in attempt.messages)


def test_enable_critic_adds_a_critique_message(sample_dir: Path) -> None:
    """enable_critic=True invokes the Critic, which appends a critique message."""
    loaded = load_puzzle(sample_dir)
    attempt = anyio.run(
        lambda: run_attempt(
            sample_dir, "m", generate=fake_generate(_CORRECT_ANSWER),
            oracle_runner=oracle_runner_for(clean_solve_outcome(loaded.meta)),
            enable_critic=True,
        )
    )
    assert any(m.get("role") == "critic" for m in attempt.messages)


# --------------------------------------------------------------------------- #
# Sandbox wiring — a real LocalSandbox end-to-end through the kernel adapter
# --------------------------------------------------------------------------- #


def test_run_attempt_with_local_sandbox_reads_file(sample_dir: Path) -> None:
    """The kernel adapts a real LocalSandbox to the Solver's tool surface; the
    generate can read a staged file through the kernel-side governor + sandbox."""
    from crucible.sandbox import LocalSandbox

    loaded = load_puzzle(sample_dir)
    from crucible.kernel import _SOLVER_HANDLE

    seen: dict[str, str] = {}

    async def reading_generate(attempt: AttemptState) -> str:
        solver = attempt.metadata[_SOLVER_HANDLE]
        # Record the tool call (governor accounting) then actually read via the box.
        await solver.record_tool_call("read_file", {"path": "config.py"})
        content = await solver.tools.read_file("config.py")
        seen["config"] = content
        # Parse the value the way a grounded Solver would.
        return content.split("=")[-1].strip()

    async def scenario() -> AttemptState:
        async with LocalSandbox() as box:
            await box.write_file("config.py", "MAX_RETRIES = 7\n")
            return await run_attempt(
                loaded,
                "m",
                generate=reading_generate,
                oracle_runner=oracle_runner_for(clean_solve_outcome(loaded.meta)),
                sandbox=box,
            )

    attempt = anyio.run(scenario)
    assert attempt.terminated_by is TerminatedBy.COMPLETED
    assert seen["config"].strip() == "MAX_RETRIES = 7"
    assert attempt.output == "7"
    # The tool call was accounted kernel-side.
    assert attempt.budget is not None and attempt.budget.tool_calls_used == 1


# --------------------------------------------------------------------------- #
# Durable event store append (§9.5)
# --------------------------------------------------------------------------- #


def test_event_store_receives_eval_log(sample_dir: Path, tmp_path: Path) -> None:
    """When an event_store is supplied, the eval-log is appended for provenance."""
    from crucible.attestation import JsonlEventStore

    loaded = load_puzzle(sample_dir)
    store = JsonlEventStore(tmp_path / "trajectory.jsonl")
    anyio.run(
        lambda: run_attempt(
            sample_dir, "m", generate=fake_generate(_CORRECT_ANSWER),
            oracle_runner=oracle_runner_for(clean_solve_outcome(loaded.meta)),
            event_store=store,
        )
    )
    assert len(store) == 1
    assert store.verify_hash_chain() is True
    stored = store.read_events()[0]
    assert stored["samples"][0]["target"] is None  # oracle stays grading-side


def test_framing_arm_recorded_in_attempt_and_log(sample_dir: Path) -> None:
    """The framing arm is a first-class measured variable on the attempt (§10.1(f))."""
    loaded = load_puzzle(sample_dir)
    attempt = anyio.run(
        lambda: run_attempt(
            sample_dir, "m", generate=fake_generate(_CORRECT_ANSWER),
            oracle_runner=oracle_runner_for(clean_solve_outcome(loaded.meta)),
            arm=FramingArm.NEUTRAL,
        )
    )
    assert attempt.framing_arm is FramingArm.NEUTRAL
    assert attempt.metadata["eval_log"]["eval"]["framing_arm"] == "neutral"


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


def _chrome_in_messages(attempt: AttemptState) -> bool:
    """True if any populated chrome value appears verbatim in the scored context."""
    if attempt.chrome is None:
        return False
    blob = repr(attempt.messages)
    for value in (attempt.chrome.rank, attempt.chrome.cohort_size):
        if value is not None and str(value) in blob:
            return True
    return False
