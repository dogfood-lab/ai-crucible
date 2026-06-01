"""Tests for the kernel-side budget governor (crucible.budget).

Per LAW 4 and the dogfood-swarm "prove the gate goes RED" discipline: where we
assert an enforcement invariant, we also prove the failing case actually raises
with the correct TerminatedBy reason (the hard-kill, budget, and time gates all
fire for real here).
"""

from __future__ import annotations

import pytest

from crucible.budget import BudgetExceeded, BudgetGovernor
from crucible.types import Budget, TerminatedBy


def _budget(tool: int = 5, time_s: int = 600) -> Budget:
    return Budget(tool_call_budget=tool, time_budget_seconds=time_s)


# --------------------------------------------------------------------------- #
# Tool-call budget
# --------------------------------------------------------------------------- #


def test_records_calls_up_to_budget() -> None:
    gov = BudgetGovernor(_budget(tool=3))
    # Distinct args so the hard-kill loop detector never trips.
    gov.record_tool_call("read_file", {"path": "a.py"})
    gov.record_tool_call("read_file", {"path": "b.py"})
    gov.record_tool_call("read_file", {"path": "c.py"})
    assert gov.budget.tool_calls_used == 3
    assert gov.budget.tool_calls_remaining == 0


def test_tool_budget_exhaustion_raises_BUDGET() -> None:
    """The (budget+1)-th distinct call must raise BudgetExceeded(BUDGET)."""
    gov = BudgetGovernor(_budget(tool=2))
    gov.record_tool_call("read_file", {"path": "a.py"})
    gov.record_tool_call("read_file", {"path": "b.py"})
    with pytest.raises(BudgetExceeded) as exc:
        gov.record_tool_call("read_file", {"path": "c.py"})
    assert exc.value.terminated_by is TerminatedBy.BUDGET
    # Authoritative counter did not over-increment past the budget.
    assert gov.budget.tool_calls_used == 2


def test_budget_not_incremented_on_rejected_call() -> None:
    gov = BudgetGovernor(_budget(tool=1))
    gov.record_tool_call("exec", {"cmd": "ls"})
    with pytest.raises(BudgetExceeded):
        gov.record_tool_call("exec", {"cmd": "pwd"})
    assert gov.budget.tool_calls_used == 1


# --------------------------------------------------------------------------- #
# Hard kill — 3 consecutive identical (tool, args) calls (§8.4)
# --------------------------------------------------------------------------- #


def test_three_identical_calls_hard_kill() -> None:
    """Default threshold 3: the 3rd identical call raises HARD_KILL."""
    gov = BudgetGovernor(_budget(tool=100))  # generous budget so BUDGET can't fire first
    gov.record_tool_call("exec", {"cmd": "ls"})
    gov.record_tool_call("exec", {"cmd": "ls"})
    with pytest.raises(BudgetExceeded) as exc:
        gov.record_tool_call("exec", {"cmd": "ls"})
    assert exc.value.terminated_by is TerminatedBy.HARD_KILL


def test_identical_args_different_order_still_loops() -> None:
    """Arg dict ordering must not let a real loop slip past detection."""
    gov = BudgetGovernor(_budget(tool=100))
    gov.record_tool_call("exec", {"a": 1, "b": 2})
    gov.record_tool_call("exec", {"b": 2, "a": 1})
    with pytest.raises(BudgetExceeded) as exc:
        gov.record_tool_call("exec", {"a": 1, "b": 2})
    assert exc.value.terminated_by is TerminatedBy.HARD_KILL


def test_non_consecutive_repeats_do_not_hard_kill() -> None:
    """Only *consecutive* identical calls are a pathological loop; an interleaved
    different call resets the streak."""
    gov = BudgetGovernor(_budget(tool=100))
    gov.record_tool_call("exec", {"cmd": "ls"})
    gov.record_tool_call("exec", {"cmd": "ls"})
    gov.record_tool_call("exec", {"cmd": "pwd"})  # breaks the streak
    gov.record_tool_call("exec", {"cmd": "ls"})
    gov.record_tool_call("exec", {"cmd": "ls"})   # streak only 2 again
    assert gov.budget.tool_calls_used == 5  # no raise


def test_hard_kill_threshold_from_meta(sample_meta) -> None:
    """The threshold comes from PuzzleMeta.hard_kill_consecutive_identical."""
    meta = sample_meta.model_copy(update={"hard_kill_consecutive_identical": 2})
    gov = BudgetGovernor(_budget(tool=100), meta=meta)
    assert gov.hard_kill_threshold == 2
    gov.record_tool_call("exec", {"cmd": "x"})
    with pytest.raises(BudgetExceeded) as exc:
        gov.record_tool_call("exec", {"cmd": "x"})  # 2nd identical → kill at threshold 2
    assert exc.value.terminated_by is TerminatedBy.HARD_KILL


def test_explicit_threshold_override() -> None:
    gov = BudgetGovernor(_budget(tool=100), hard_kill_threshold=4)
    assert gov.hard_kill_threshold == 4
    for _ in range(3):
        gov.record_tool_call("exec", {"cmd": "x"})  # 3 identical, threshold 4 → ok
    with pytest.raises(BudgetExceeded):
        gov.record_tool_call("exec", {"cmd": "x"})  # 4th → kill


def test_threshold_below_two_rejected() -> None:
    with pytest.raises(ValueError, match=">= 2"):
        BudgetGovernor(_budget(), hard_kill_threshold=1)


def test_hard_kill_precedes_budget() -> None:
    """A pathological loop is reported as HARD_KILL even when the budget would
    also be exhausted on the same call (loop is the more specific signal)."""
    gov = BudgetGovernor(_budget(tool=2))
    gov.record_tool_call("exec", {"cmd": "loop"})
    gov.record_tool_call("exec", {"cmd": "loop"})  # used == budget now
    with pytest.raises(BudgetExceeded) as exc:
        gov.record_tool_call("exec", {"cmd": "loop"})  # 3rd identical
    assert exc.value.terminated_by is TerminatedBy.HARD_KILL


# --------------------------------------------------------------------------- #
# Time budget
# --------------------------------------------------------------------------- #


def test_check_time_within_budget() -> None:
    gov = BudgetGovernor(_budget(time_s=600))
    gov.check_time(599.9)  # no raise
    assert gov.budget.elapsed_seconds == pytest.approx(599.9)


def test_time_exhaustion_raises_TIME() -> None:
    gov = BudgetGovernor(_budget(time_s=600))
    with pytest.raises(BudgetExceeded) as exc:
        gov.check_time(600.1)
    assert exc.value.terminated_by is TerminatedBy.TIME


# --------------------------------------------------------------------------- #
# Context-manager usage around the Solver block
# --------------------------------------------------------------------------- #


def test_context_manager_does_not_suppress() -> None:
    """A BudgetExceeded raised inside the `with` must propagate to the attempt
    boundary (the governor never swallows it)."""
    gov = BudgetGovernor(_budget(tool=1))
    with pytest.raises(BudgetExceeded) as exc, gov as g:
        g.record_tool_call("exec", {"cmd": "a"})
        g.record_tool_call("exec", {"cmd": "b"})  # over budget → raises
    assert exc.value.terminated_by is TerminatedBy.BUDGET


def test_exceeded_carries_terminated_by_attribute() -> None:
    err = BudgetExceeded(TerminatedBy.HARD_KILL, "loop")
    assert err.terminated_by is TerminatedBy.HARD_KILL
    assert "loop" in str(err)
