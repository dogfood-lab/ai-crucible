"""Smoke tests for the locked cross-module contracts (ai_crucible.types).

These exist before the build wave so the foundation is provably green and the
§8.3 reward-bound invariant is pinned (a meta-test that the validator actually
fires, per the dogfood-swarm "prove the gate goes RED" discipline).
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from ai_crucible import __version__
from ai_crucible.types import (
    AttemptState,
    Budget,
    Chrome,
    FramingArm,
    PuzzleMeta,
    Rewards,
    Role,
    TerminatedBy,
)


def test_version() -> None:
    assert __version__ == "0.3.0"


def test_default_framing_arm_is_self_referential() -> None:
    """The director-locked default (§10.1(f))."""
    a = AttemptState(attempt_id="a", puzzle_id="p", model="m")
    assert a.framing_arm is FramingArm.SELF_REFERENTIAL


def test_budget_remaining_clamps_at_zero() -> None:
    b = Budget(tool_call_budget=5, time_budget_seconds=60, tool_calls_used=7, elapsed_seconds=90)
    assert b.tool_calls_remaining == 0
    assert b.time_remaining == 0.0


def test_chrome_is_separate_from_messages(fresh_attempt: AttemptState) -> None:
    """Sealed-boundary structure: Tier-3 chrome is a distinct field, never the
    message list (§10.1(e)). The engagement module enforces non-injection; here
    we only assert the structural separation exists."""
    fresh_attempt.chrome = Chrome(rank=7, cohort_size=12)
    assert fresh_attempt.chrome is not None
    assert fresh_attempt.messages == []  # chrome did not leak into the scored context


def test_terminated_by_values() -> None:
    assert TerminatedBy.HARD_KILL.value == "hard_kill"


def test_meta_validates_within_bounds(sample_meta: PuzzleMeta) -> None:
    assert sample_meta.rewards.solve == 80.0


def test_meta_rejects_unbounded_elegance() -> None:
    """§8.3: elegance_bonus_max must be ≤30% of solve — prove the gate goes RED."""
    with pytest.raises(ValidationError):
        PuzzleMeta(
            puzzle_id="bad",
            created_at="2026-06-01T00:00:00Z",
            capability_aspect="x",
            puzzle_class="multi_file_search",
            point_threshold=10,
            time_budget_seconds=60,
            tool_call_budget=5,
            rewards=Rewards(solve=80.0, elegance_bonus_max=40.0),  # 50% > 30% cap
        )


def test_meta_rejects_unbounded_novelty() -> None:
    """§8.3: novelty_bonus_max must be ≤50% of solve — prove the gate goes RED."""
    with pytest.raises(ValidationError):
        PuzzleMeta(
            puzzle_id="bad",
            created_at="2026-06-01T00:00:00Z",
            capability_aspect="x",
            puzzle_class="multi_file_search",
            point_threshold=10,
            time_budget_seconds=60,
            tool_call_budget=5,
            rewards=Rewards(solve=80.0, novelty_bonus_max=60.0),  # 75% > 50% cap
        )


def test_role_is_runtime_checkable() -> None:
    """The uniform Role protocol (§10.3) is runtime-checkable for kernel wiring."""
    class _Stub:
        name = "solver"

        async def act(self, state: AttemptState) -> AttemptState:  # noqa: D401
            return state

    assert isinstance(_Stub(), Role)
