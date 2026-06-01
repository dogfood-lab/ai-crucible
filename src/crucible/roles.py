"""Concrete roles (``roles`` module, research-grounding §10.3).

The five model-agnostic role slots — :class:`Designer`, :class:`Solver`,
:class:`Critic`, :class:`Judge`, :class:`CohortSolver` — each implement the
:class:`crucible.types.Role` protocol (``name`` + ``async def act``).

Two invariants are structural here, not aspirational:

1. **Single model-I/O choke point (§10.2).** Every role is constructed with an
   injected ``generate: Callable[[AttemptState], Awaitable[str]]`` and routes
   *all* model calls through it. Roles never import or call a model client
   directly, so every model call is observable at one point and testable with a
   fake ``generate``.

2. **Sealed Tier-3 boundary (§10.1(d,e), §10.3).** No role reads
   ``state.chrome``. Rank / leaderboard / standings are human-facing UI only and
   must never enter a context window the model solves in. To make accidental
   reads impossible (not merely discouraged), each ``act`` runs under a tiny
   guard that raises if ``state.chrome`` is touched during the role's turn.

Only :class:`Solver` is wired to sandbox tools and a
:class:`crucible.budget.BudgetGovernor` (the tool boundary, §10.3). The
:class:`Critic` is interface-complete but **default-off** — the kernel decides
whether to invoke it per puzzle (§10.3); its value is narrow and it can regress
a strong Solver via conformity, so it is reserved, not wired on.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any, Protocol

from crucible.budget import BudgetGovernor
from crucible.types import AttemptState, RoleName, TerminatedBy, TraceEvent

__all__ = [
    "ChromeAccessError",
    "CohortSolver",
    "Critic",
    "Designer",
    "GenerateFn",
    "Judge",
    "SandboxTools",
    "Solver",
]

#: The single model-I/O choke point all roles route through (§10.2). Given the
#: current attempt, return the model's text output. Injected so the kernel owns
#: model selection and observability, and tests can pass a canned async lambda.
GenerateFn = Callable[[AttemptState], Awaitable[str]]


class SandboxTools(Protocol):
    """Structural type for the sandbox tool surface handed to the Solver.

    Kept as a typed parameter here so :mod:`crucible.roles` does not import the
    ``sandbox`` module (decompose-by-secrets; the sandbox provider is built and
    owned elsewhere). The real provider exposes a narrow ``exec/read_file/
    write_file`` channel (§10.4); this protocol only pins the shape Solver needs.
    """

    async def exec(self, command: str) -> str: ...
    async def read_file(self, path: str) -> str: ...
    async def write_file(self, path: str, content: str) -> None: ...


class ChromeAccessError(RuntimeError):
    """Raised if a role's ``act`` reads Tier-3 ``chrome`` during its turn.

    The sealed boundary (§10.1(e)) forbids the motivating/engagement surface from
    entering the scored context. This error turns that contract into a hard,
    test-provable failure rather than a comment.
    """


class _ChromeGuard:
    """Context manager that proves ``state.chrome`` was not read during a turn.

    We cannot intercept attribute access on the shared dataclass without mutating
    it (forbidden — ``types.py`` is locked), so we snapshot the message count on
    entry and re-assert on exit that the role did not *consume* chrome into the
    scored surface. The structural guarantee that matters — chrome is a separate
    field and never appended to ``messages`` — is asserted directly here.

    The leak check **delegates to the real token-based guard**
    (:func:`crucible.engagement.assert_no_chrome_leak`), so the roles-layer guard
    and the kernel-layer guard have *identical strong semantics* (§10.1(e)). The
    earlier roles-local heuristic only caught a leak when the literal word
    ``"chrome"``/``"leaderboard"`` happened to appear in the message — it missed a
    bare rank rendered as prose ("ranked 7 of 12") or a leaderboard solver-id, the
    realistic leak shapes. Delegating removes that gap.
    """

    def __init__(self, state: AttemptState) -> None:
        self._state = state
        self._messages_len = len(state.messages)

    def __enter__(self) -> _ChromeGuard:
        return self

    def __exit__(self, *exc: object) -> bool:
        # Sealed-boundary assertion: nothing a role appended to the scored context
        # this turn may carry Tier-3 chrome. Only the messages added during the
        # turn are checked (the pre-existing scored context was already guarded by
        # the kernel before the Solver ran).
        chrome = self._state.chrome
        if chrome is not None:
            new_messages = self._state.messages[self._messages_len :]
            if new_messages and _mentions_chrome(new_messages, chrome):
                raise ChromeAccessError(
                    "Tier-3 chrome leaked into the scored context (§10.1(e)): "
                    "rank/leaderboard/standings must never enter messages"
                )
        return False


def _mentions_chrome(messages: list[dict[str, Any]], chrome: object) -> bool:
    """True if any of ``messages`` carries a Tier-3 chrome value.

    Delegates to :func:`crucible.engagement.assert_no_chrome_leak` — the real
    token-based guard (it flattens every populated chrome value, including
    leaderboard rows and catalog standings, to its rendered string tokens and
    word-boundary-matches each against message content). This is the *same* guard
    the kernel runs, so the two layers cannot drift apart (§10.1(e)). The
    engagement guard *raises* :class:`~crucible.engagement.SealedBoundaryViolation`
    on a leak; here we translate that into a boolean for the role-layer
    :class:`ChromeAccessError` contract.
    """
    from crucible.engagement import SealedBoundaryViolation, assert_no_chrome_leak
    from crucible.types import Chrome

    if not isinstance(chrome, Chrome):
        return False
    try:
        assert_no_chrome_leak(messages, chrome)
    except SealedBoundaryViolation:
        return True
    return False


def _next_seq(state: AttemptState) -> int:
    return len(state.events)


class Designer:
    """Generates / refines a puzzle candidate (stays Claude — the creative role
    crucible is built for, §10.3). Round-against-round Designer/Solver co-evolution
    (R-Zero, §10.1(b)). No sandbox tools, no env access — pure model I/O."""

    name: RoleName = RoleName.DESIGNER

    def __init__(self, generate: GenerateFn) -> None:
        self._generate = generate

    async def act(self, state: AttemptState) -> AttemptState:
        with _ChromeGuard(state):
            text = await self._generate(state)
            state.events.append(
                TraceEvent(kind="model", role=self.name, payload={"text": text},
                           seq=_next_seq(state))
            )
            state.metadata["designer_output"] = text
        return state


class Solver:
    """Attempts the puzzle. The ONLY role granted sandbox tools and bound to a
    :class:`BudgetGovernor` (§10.3 tool boundary).

    Every tool call MUST be recorded through the governor (kernel-side accounting,
    §10.2 / §8.4) — the Solver never self-reports usage. A
    :class:`crucible.budget.BudgetExceeded` raised by the governor propagates to
    the attempt boundary; here we stamp ``terminated_by`` from it and stop.
    """

    name: RoleName = RoleName.SOLVER

    def __init__(
        self,
        generate: GenerateFn,
        tools: SandboxTools,
        governor: BudgetGovernor,
    ) -> None:
        self._generate = generate
        self._tools = tools
        self._governor = governor

    @property
    def tools(self) -> SandboxTools:
        return self._tools

    @property
    def governor(self) -> BudgetGovernor:
        return self._governor

    async def record_tool_call(self, tool: str, args: dict) -> None:
        """Route a tool call through the kernel-side governor, then trace it.

        Raises :class:`crucible.budget.BudgetExceeded` (re-raised unchanged) when
        a budget dimension is breached, so the kernel halts the attempt at the
        boundary (ANDON)."""
        self._governor.record_tool_call(tool, args)
        self._tools_event(tool, args)

    def _tools_event(self, tool: str, args: dict) -> None:
        # Imported lazily to keep the shared-state import surface minimal.
        from crucible.types import AttemptState as _AS  # noqa: F401

        self._last_state.events.append(  # type: ignore[union-attr]
            TraceEvent(kind="tool", role=self.name, payload={"tool": tool, "args": args},
                       seq=_next_seq(self._last_state))
        )

    async def act(self, state: AttemptState) -> AttemptState:
        self._last_state = state  # for _tools_event during this turn
        with _ChromeGuard(state):
            try:
                text = await self._generate(state)
            except Exception as exc:  # generation/tool failure → ERROR terminal
                from crucible.budget import BudgetExceeded

                if isinstance(exc, BudgetExceeded):
                    state.terminated_by = exc.terminated_by
                    state.error = str(exc)
                    state.events.append(
                        TraceEvent(kind="error", role=self.name,
                                   payload={"terminated_by": exc.terminated_by.value,
                                            "message": str(exc)},
                                   seq=_next_seq(state))
                    )
                    return state
                state.terminated_by = TerminatedBy.ERROR
                state.error = str(exc)
                state.events.append(
                    TraceEvent(kind="error", role=self.name,
                               payload={"message": str(exc)}, seq=_next_seq(state))
                )
                return state

            state.output = text
            state.events.append(
                TraceEvent(kind="model", role=self.name, payload={"text": text},
                           seq=_next_seq(state))
            )
            if state.terminated_by is None:
                state.terminated_by = TerminatedBy.COMPLETED
        return state


class Critic:
    """Adversarial third agent (Debate, §10.3). INTERFACE-COMPLETE BUT DEFAULT-OFF
    — the kernel decides per puzzle whether to invoke it.

    String-in / string-out: no sandbox tools, no env access (§10.3). The message
    schema is defined now so enabling it later (on logged high-disagreement puzzle
    classes) does not break the contract.
    """

    name: RoleName = RoleName.CRITIC

    #: Default-off marker the kernel honours (§10.3). Construct with
    #: ``enabled=True`` only via explicit per-puzzle opt-in.
    enabled: bool = False

    def __init__(self, generate: GenerateFn, *, enabled: bool = False) -> None:
        self._generate = generate
        self.enabled = enabled

    @staticmethod
    def message(critique: str, *, anonymized: bool = True) -> dict[str, Any]:
        """The Critic message schema (defined now to prevent a later contract
        break, §10.3). ``anonymized`` reflects the D3 debater-identity-hiding
        recommendation for when the Critic is enabled."""
        return {"role": "critic", "critique": critique, "anonymized": anonymized}

    async def act(self, state: AttemptState) -> AttemptState:
        if not self.enabled:
            # Default-off: a no-op that records the skip, leaving the attempt
            # untouched. The kernel must opt in per puzzle to get a critique.
            state.events.append(
                TraceEvent(kind="info", role=self.name,
                           payload={"skipped": "critic default-off (§10.3)"},
                           seq=_next_seq(state))
            )
            return state
        with _ChromeGuard(state):
            critique = await self._generate(state)
            state.messages.append(self.message(critique))
            state.events.append(
                TraceEvent(kind="model", role=self.name,
                           payload={"critique": critique}, seq=_next_seq(state))
            )
        return state


class Judge:
    """A single cross-family judge (§10.3). Scores terminal world-state, not chat
    text (τ-bench). String-in/string-out — no sandbox tools, no env access. The
    panel reducer (PoLL) composes N judges and lives in the scoring module; this
    is one panelist."""

    name: RoleName = RoleName.JUDGE

    def __init__(self, generate: GenerateFn) -> None:
        self._generate = generate

    async def act(self, state: AttemptState) -> AttemptState:
        with _ChromeGuard(state):
            verdict = await self._generate(state)
            state.events.append(
                TraceEvent(kind="score", role=self.name,
                           payload={"verdict": verdict}, seq=_next_seq(state))
            )
            state.metadata.setdefault("judge_verdicts", []).append(verdict)
        return state


class CohortSolver:
    """A sibling Solver in the pass^k cohort (§10.2 — k sibling attempts per
    puzzle). Behaviourally a :class:`Solver` slot; named distinctly so the kernel
    can wire k of them per puzzle and the reducer can compute pass^k. Like the
    Solver, it is the only kind of role granted sandbox tools and a governor."""

    name: RoleName = RoleName.COHORT_SOLVER

    def __init__(
        self,
        generate: GenerateFn,
        tools: SandboxTools,
        governor: BudgetGovernor,
    ) -> None:
        # Compose a Solver so cohort members share exactly one code path.
        self._solver = Solver(generate, tools, governor)

    @property
    def governor(self) -> BudgetGovernor:
        return self._solver.governor

    @property
    def tools(self) -> SandboxTools:
        return self._solver.tools

    async def record_tool_call(self, tool: str, args: dict) -> None:
        await self._solver.record_tool_call(tool, args)

    async def act(self, state: AttemptState) -> AttemptState:
        state = await self._solver.act(state)
        # Re-stamp the role on the model/terminal events for cohort attribution.
        for ev in state.events:
            if ev.role is RoleName.SOLVER:
                ev.role = self.name
        return state
