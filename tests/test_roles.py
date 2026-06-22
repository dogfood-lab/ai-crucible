"""Tests for the concrete roles (ai_crucible.roles).

Async roles are driven with ``anyio.run`` (no pytest-asyncio dependency, and
``anyio`` ships with inspect-ai). The model-I/O choke point is exercised with a
fake ``generate`` (a canned async function), proving roles are testable without a
real model — which is the whole point of the §10.2 injected choke point.

Invariants proven (happy path + RED path where applicable):
- Single ``generate`` choke point: roles call only the injected fn (§10.2).
- Solver is the only role bound to sandbox tools + governor (§10.3).
- Budget enforcement flows through the governor and stamps ``terminated_by``.
- Sealed boundary: a role leaking Tier-3 chrome into ``messages`` goes RED
  (§10.1(e)); a well-behaved role does not.
- Critic is default-OFF and a no-op until explicitly enabled (§10.3).
"""

from __future__ import annotations

import anyio
import pytest

from ai_crucible.budget import BudgetGovernor
from ai_crucible.roles import (
    ChromeAccessError,
    CohortSolver,
    Critic,
    Designer,
    Judge,
    Solver,
)
from ai_crucible.types import (
    AttemptState,
    Budget,
    Chrome,
    Role,
    RoleName,
    TerminatedBy,
)

# --------------------------------------------------------------------------- #
# Test doubles
# --------------------------------------------------------------------------- #


def canned(text: str):
    """A fake `generate` choke point that records call count and returns `text`."""
    state = {"calls": 0}

    async def _gen(_attempt: AttemptState) -> str:
        state["calls"] += 1
        return text

    _gen.calls = state  # type: ignore[attr-defined]
    return _gen


class FakeSandbox:
    """Minimal sandbox tool stub satisfying the SandboxTools protocol."""

    async def exec(self, command: str) -> str:
        return ""

    async def read_file(self, path: str) -> str:
        return "MAX_RETRIES = 7"

    async def write_file(self, path: str, content: str) -> None:
        return None


def _attempt() -> AttemptState:
    return AttemptState(
        attempt_id="att-1",
        puzzle_id="p-1",
        model="m",
        budget=Budget(tool_call_budget=5, time_budget_seconds=600),
    )


def _governor(tool: int = 5) -> BudgetGovernor:
    return BudgetGovernor(Budget(tool_call_budget=tool, time_budget_seconds=600))


# --------------------------------------------------------------------------- #
# Protocol conformance + names
# --------------------------------------------------------------------------- #


def test_all_roles_satisfy_protocol() -> None:
    gen = canned("x")
    roles = [
        Designer(gen),
        Solver(gen, FakeSandbox(), _governor()),
        Critic(gen),
        Judge(gen),
        CohortSolver(gen, FakeSandbox(), _governor()),
    ]
    for r in roles:
        assert isinstance(r, Role)


def test_role_names_match_enum() -> None:
    gen = canned("x")
    assert Designer(gen).name is RoleName.DESIGNER
    assert Solver(gen, FakeSandbox(), _governor()).name is RoleName.SOLVER
    assert Critic(gen).name is RoleName.CRITIC
    assert Judge(gen).name is RoleName.JUDGE
    assert CohortSolver(gen, FakeSandbox(), _governor()).name is RoleName.COHORT_SOLVER


# --------------------------------------------------------------------------- #
# The single model-I/O choke point (§10.2)
# --------------------------------------------------------------------------- #


def test_designer_routes_through_generate() -> None:
    gen = canned("a puzzle candidate")
    state = anyio.run(Designer(gen).act, _attempt())
    assert gen.calls["calls"] == 1
    assert state.metadata["designer_output"] == "a puzzle candidate"
    assert any(e.kind == "model" and e.role is RoleName.DESIGNER for e in state.events)


def test_solver_sets_output_and_completed() -> None:
    gen = canned("7")
    solver = Solver(gen, FakeSandbox(), _governor())
    state = anyio.run(solver.act, _attempt())
    assert state.output == "7"
    assert state.terminated_by is TerminatedBy.COMPLETED
    assert gen.calls["calls"] == 1


def test_judge_records_verdict() -> None:
    gen = canned("legitimate")
    state = anyio.run(Judge(gen).act, _attempt())
    assert state.metadata["judge_verdicts"] == ["legitimate"]
    assert any(e.kind == "score" and e.role is RoleName.JUDGE for e in state.events)


# --------------------------------------------------------------------------- #
# Solver tool boundary + governor (§10.3 / §8.4)
# --------------------------------------------------------------------------- #


def test_solver_records_tool_calls_through_governor() -> None:
    gen = canned("done")
    gov = _governor(tool=3)
    solver = Solver(gen, FakeSandbox(), gov)

    async def scenario() -> None:
        # Solver must record each tool call through the kernel-side governor.
        solver._last_state = _attempt()  # the kernel sets this each turn
        await solver.record_tool_call("read_file", {"path": "a.py"})
        await solver.record_tool_call("read_file", {"path": "b.py"})

    anyio.run(scenario)
    assert gov.budget.tool_calls_used == 2


def test_solver_budget_exhaustion_stamps_terminated_by() -> None:
    """If `generate` overruns the budget via the governor, the Solver stamps
    terminated_by=BUDGET and stops (ANDON at the attempt boundary)."""
    gov = _governor(tool=1)

    async def overrunning_generate(state: AttemptState) -> str:
        # Simulate the kernel's tool loop overrunning the budget mid-turn.
        gov.record_tool_call("exec", {"cmd": "a"})
        gov.record_tool_call("exec", {"cmd": "b"})  # raises BudgetExceeded(BUDGET)
        return "unreachable"

    solver = Solver(overrunning_generate, FakeSandbox(), gov)
    state = anyio.run(solver.act, _attempt())
    assert state.terminated_by is TerminatedBy.BUDGET
    assert state.output is None
    assert any(e.kind == "error" for e in state.events)


def test_solver_hard_kill_stamps_terminated_by() -> None:
    gov = _governor(tool=100)

    async def looping_generate(state: AttemptState) -> str:
        for _ in range(3):
            gov.record_tool_call("exec", {"cmd": "loop"})  # 3rd raises HARD_KILL
        return "unreachable"

    solver = Solver(looping_generate, FakeSandbox(), gov)
    state = anyio.run(solver.act, _attempt())
    assert state.terminated_by is TerminatedBy.HARD_KILL


def test_solver_generic_exception_is_error_terminal() -> None:
    async def boom(state: AttemptState) -> str:
        raise RuntimeError("model exploded")

    solver = Solver(boom, FakeSandbox(), _governor())
    state = anyio.run(solver.act, _attempt())
    assert state.terminated_by is TerminatedBy.ERROR
    assert "model exploded" in (state.error or "")


def test_only_solver_has_tools() -> None:
    """§10.3 tool boundary: Designer/Critic/Judge expose no sandbox tools."""
    gen = canned("x")
    assert not hasattr(Designer(gen), "tools")
    assert not hasattr(Critic(gen), "tools")
    assert not hasattr(Judge(gen), "tools")
    assert hasattr(Solver(gen, FakeSandbox(), _governor()), "tools")


# --------------------------------------------------------------------------- #
# Sealed Tier-3 boundary (§10.1(e), §10.3)
# --------------------------------------------------------------------------- #


def test_chrome_present_but_not_leaked_is_fine() -> None:
    """A role may run with chrome set on the state, as long as it does not pull
    chrome into the scored `messages`."""
    gen = canned("clean answer")
    state = _attempt()
    state.chrome = Chrome(rank=7, cohort_size=12)
    out = anyio.run(Solver(gen, FakeSandbox(), _governor()).act, state)
    assert out.terminated_by is TerminatedBy.COMPLETED
    # messages untouched by the Solver here → boundary intact
    assert out.messages == []


def _leaking_role(content: str):
    """A (mis)behaving role that serializes ``content`` into the scored messages
    under the :class:`_ChromeGuard`. Used to drive the RED prose-leak paths."""

    class LeakyRole:
        name = RoleName.SOLVER

        async def act(self, state: AttemptState) -> AttemptState:
            from ai_crucible.roles import _ChromeGuard

            with _ChromeGuard(state):
                # Violation: pull a Tier-3 chrome VALUE into the scored context as
                # natural prose — the realistic leak shape, not a labeled key.
                state.messages.append({"role": "user", "content": content})
            return state

    return LeakyRole()


def test_prose_rank_leak_into_messages_goes_red() -> None:
    """RED path: a role that writes the rank as PROSE ("ranked 7 of 12 on the
    leaderboard") leaks Tier-3 chrome into the scored context and must be caught.

    This is the leak the OLD ``chrome_rank``-keyed test could not see: there is no
    literal ``"chrome"`` token here, so the previous weak guard passed it. The
    guard now delegates to :func:`ai_crucible.engagement.assert_no_chrome_leak`, whose
    token-based match catches the bare rank value however it is phrased (§10.1(e))."""
    state = _attempt()
    state.chrome = Chrome(rank=7, cohort_size=12)
    with pytest.raises(ChromeAccessError):
        anyio.run(_leaking_role("You are ranked 7 of 12 on the leaderboard.").act, state)


def test_prose_solver_id_leak_into_messages_goes_red() -> None:
    """RED path: a leaderboard solver-id leaked in ``content`` (no literal
    ``"leaderboard"`` word, no ``"chrome"`` token) must be caught.

    The old weak guard's leaderboard branch only fired on the literal word
    ""leaderboard"" appearing in the message; a bare solver-id slipped through. The
    delegated token-based guard catches the actual leaked value (§10.1(e))."""
    state = _attempt()
    state.chrome = Chrome(
        rank=3,
        cohort_size=12,
        leaderboard=[{"solver": "solver-zeta", "score": 99}],
    )
    with pytest.raises(ChromeAccessError):
        anyio.run(
            _leaking_role("The current best solution is from solver-zeta.").act, state
        )


def test_clean_prose_does_not_trip_chrome_guard() -> None:
    """A role that writes only task prose (no chrome value) passes the guard — the
    delegated guard must not be a false-positive machine (it only fires on a real
    populated chrome value, §10.1(e))."""
    state = _attempt()
    state.chrome = Chrome(rank=7, cohort_size=12, leaderboard=[{"solver": "solver-zeta"}])
    role = _leaking_role("Read config.py and report MAX_RETRIES to the user.")
    out = anyio.run(role.act, state)  # must NOT raise
    assert out.messages[-1]["content"].startswith("Read config.py")


def test_chrome_access_error_carries_code_and_hint() -> None:
    """PROVE RED (error-hint-sweep-003): the raised ChromeAccessError message must
    follow the repo's structured-error shape — a stable ``[CHROME_ACCESS]`` code
    prefix and an explicit ``(hint: ...)`` telling the designer what to DO (keep
    rank/leaderboard/standings out of ``messages``; move the value into the Chrome
    object) — matching PuzzleLoadError / ModelMismatchError. The exception TYPE and
    control flow are unchanged; only the message text is enriched."""
    state = _attempt()
    state.chrome = Chrome(rank=7, cohort_size=12)
    with pytest.raises(ChromeAccessError) as exc:
        anyio.run(_leaking_role("You are ranked 7 of 12 on the leaderboard.").act, state)
    msg = str(exc.value)
    assert msg.startswith("[CHROME_ACCESS]"), msg
    assert "(hint:" in msg, msg
    # The actionable remediation names the surface to keep clean / where chrome lives.
    assert "messages" in msg.lower() or "chrome" in msg.lower()


# --------------------------------------------------------------------------- #
# Critic default-off (§10.3)
# --------------------------------------------------------------------------- #


def test_critic_default_off_is_noop() -> None:
    gen = canned("a critique")
    critic = Critic(gen)
    assert critic.enabled is False
    state = anyio.run(critic.act, _attempt())
    # No model call, no message appended — just a skip record.
    assert gen.calls["calls"] == 0
    assert state.messages == []
    assert any(e.kind == "info" and "default-off" in str(e.payload) for e in state.events)


def test_critic_enabled_produces_critique() -> None:
    gen = canned("this skips the grounded read")
    critic = Critic(gen, enabled=True)
    state = anyio.run(critic.act, _attempt())
    assert gen.calls["calls"] == 1
    assert any(m.get("role") == "critic" for m in state.messages)


def test_critic_message_schema_anonymizes_by_default() -> None:
    msg = Critic.message("c")
    assert msg["role"] == "critic"
    assert msg["anonymized"] is True


# --------------------------------------------------------------------------- #
# CohortSolver (pass^k sibling, §10.2)
# --------------------------------------------------------------------------- #


def test_cohort_solver_acts_like_solver_with_own_role() -> None:
    gen = canned("7")
    cohort = CohortSolver(gen, FakeSandbox(), _governor())
    state = anyio.run(cohort.act, _attempt())
    assert state.output == "7"
    assert state.terminated_by is TerminatedBy.COMPLETED
    # Events are re-attributed to COHORT_SOLVER, not SOLVER.
    assert any(e.role is RoleName.COHORT_SOLVER for e in state.events)
    assert not any(e.role is RoleName.SOLVER for e in state.events)
