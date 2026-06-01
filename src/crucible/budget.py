"""Kernel-side budget governor (``budget_governor`` module, §8.4 + §10.2).

The governor wraps a :class:`crucible.types.Budget` and is used as a context
manager around the **Solver block only** (§10.2). It is the single authority on
tool-call accounting, wall-clock budget, and pathological-loop detection — it
**never trusts the model to self-report** its usage (§10.2, Vivaria pattern).
When a limit is breached it raises :class:`BudgetExceeded`, carrying the
:class:`crucible.types.TerminatedBy` reason so the kernel can stamp
``AttemptState.terminated_by`` at the attempt boundary.

Three independent signals (§8.4):

- **Tool-call budget** — ``record_tool_call`` increments ``tool_calls_used`` and
  raises ``BudgetExceeded(BUDGET)`` the moment usage would exceed
  ``tool_call_budget`` (an ANDON halt: bad/over-budget output never propagates).
- **Hard kill** — N *consecutive* identical ``(tool, args)`` calls is a
  pathological loop (WebArena pattern, arXiv:2307.13854). Raises
  ``BudgetExceeded(HARD_KILL)``. N defaults to 3, overridable per puzzle via
  ``PuzzleMeta.hard_kill_consecutive_identical``.
- **Time budget** — ``check_time`` raises ``BudgetExceeded(TIME)`` once elapsed
  wall-clock exceeds ``time_budget_seconds``.

This module is the *enforcement floor*; the displayed budget that the Solver
self-rations against (BATS, arXiv:2511.17006) is a Tier-1 task-relevant signal
rendered separately from the live :class:`Budget` value object.
"""

from __future__ import annotations

import json
from types import TracebackType

from crucible.types import Budget, PuzzleMeta, TerminatedBy

__all__ = ["BudgetExceeded", "BudgetGovernor"]

# Floor for the consecutive-identical hard-kill (§8.4). PuzzleMeta enforces
# ``hard_kill_consecutive_identical >= 2``; this mirrors its default of 3.
_DEFAULT_HARD_KILL = 3


class BudgetExceeded(Exception):
    """Raised when any budget dimension is breached.

    Attributes:
        terminated_by: which limit fired — ``BUDGET`` (tool/step), ``TIME``
            (wall-clock), or ``HARD_KILL`` (pathological loop). The kernel copies
            this onto ``AttemptState.terminated_by``.
    """

    def __init__(self, terminated_by: TerminatedBy, message: str = "") -> None:
        self.terminated_by = terminated_by
        super().__init__(message or terminated_by.value)


class BudgetGovernor:
    """Authoritative, kernel-side enforcement of a single attempt's budget.

    Construct from a :class:`Budget`. The hard-kill threshold is taken from a
    :class:`PuzzleMeta` when supplied (``hard_kill_consecutive_identical``),
    otherwise from the explicit ``hard_kill_threshold`` argument, otherwise the
    §8.4 default of 3.

    Usable as a context manager around the Solver block::

        with BudgetGovernor(budget) as gov:
            gov.record_tool_call("read_file", {"path": "a.py"})
            gov.check_time(elapsed)

    A :class:`BudgetExceeded` raised inside the ``with`` block propagates out to
    the attempt boundary unchanged (the governor does not swallow it); the kernel
    catches it there and stamps ``terminated_by``.
    """

    def __init__(
        self,
        budget: Budget,
        *,
        meta: PuzzleMeta | None = None,
        hard_kill_threshold: int | None = None,
    ) -> None:
        self._budget = budget
        if meta is not None:
            self._hard_kill_threshold = meta.hard_kill_consecutive_identical
        elif hard_kill_threshold is not None:
            self._hard_kill_threshold = hard_kill_threshold
        else:
            self._hard_kill_threshold = _DEFAULT_HARD_KILL
        if self._hard_kill_threshold < 2:
            raise ValueError("hard_kill_threshold must be >= 2 (§8.4)")
        # Loop detection state: the canonical form of the last call and how many
        # times in a row it has repeated.
        self._last_signature: str | None = None
        self._consecutive_identical: int = 0

    # -- introspection ------------------------------------------------------- #

    @property
    def budget(self) -> Budget:
        """The live budget value object (authoritative usage counters)."""
        return self._budget

    @property
    def hard_kill_threshold(self) -> int:
        return self._hard_kill_threshold

    # -- enforcement --------------------------------------------------------- #

    def record_tool_call(self, tool: str, args: dict) -> None:
        """Record one Solver tool call. Kernel-side; never model-self-reported.

        Increments ``tool_calls_used`` only for an admitted call. Raises:

        - ``BudgetExceeded(HARD_KILL)`` when this is the Nth *consecutive*
          identical ``(tool, args)`` call (pathological loop, §8.4). Checked
          before the budget increment so a loop is reported as a loop, not as
          mere exhaustion.
        - ``BudgetExceeded(BUDGET)`` when admitting this call would push
          ``tool_calls_used`` past ``tool_call_budget``.
        """
        signature = self._signature(tool, args)

        # 1. Pathological-loop detection (§8.4 hard kill).
        if signature == self._last_signature:
            self._consecutive_identical += 1
        else:
            self._last_signature = signature
            self._consecutive_identical = 1

        if self._consecutive_identical >= self._hard_kill_threshold:
            raise BudgetExceeded(
                TerminatedBy.HARD_KILL,
                f"hard kill: {self._consecutive_identical} consecutive identical "
                f"calls to {tool!r} (threshold {self._hard_kill_threshold}, §8.4)",
            )

        # 2. Tool-call budget (ANDON: refuse the call that would overrun).
        if self._budget.tool_calls_used + 1 > self._budget.tool_call_budget:
            raise BudgetExceeded(
                TerminatedBy.BUDGET,
                f"tool-call budget exhausted: {self._budget.tool_call_budget} "
                "calls used (§8.4)",
            )

        self._budget.tool_calls_used += 1

    def check_time(self, elapsed: float) -> None:
        """Record elapsed wall-clock and raise ``BudgetExceeded(TIME)`` past the
        time budget. Kernel-side; the model never reports its own elapsed time."""
        self._budget.elapsed_seconds = elapsed
        if elapsed > self._budget.time_budget_seconds:
            raise BudgetExceeded(
                TerminatedBy.TIME,
                f"time budget exhausted: {elapsed:.1f}s > "
                f"{self._budget.time_budget_seconds}s (§8.4)",
            )

    # -- context manager ----------------------------------------------------- #

    def __enter__(self) -> BudgetGovernor:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> bool:
        # Never suppress — a BudgetExceeded must reach the attempt boundary so
        # the kernel can stamp terminated_by.
        return False

    # -- internals ----------------------------------------------------------- #

    @staticmethod
    def _signature(tool: str, args: dict) -> str:
        """Canonical, order-independent signature of a ``(tool, args)`` call.

        ``sort_keys`` makes ``{"a":1,"b":2}`` and ``{"b":2,"a":1}`` identical so
        a genuine loop is caught regardless of dict ordering. Non-JSON-able args
        fall back to ``repr`` (still stable within a single process run)."""
        try:
            args_repr = json.dumps(args, sort_keys=True, default=repr)
        except (TypeError, ValueError):
            args_repr = repr(sorted(args.items()))
        return f"{tool}\x00{args_repr}"
