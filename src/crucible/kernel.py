"""The Crucible kernel — the thin policy layer that composes everything (§10.2).

This is the integrator module (Wave 2). It does **not** reimplement scoring,
sandboxing, framing, budgeting, tracing, or the role machinery — it *wires the
Wave-1 leaf modules together* into the two public entry points crucible runs on:

- :func:`run_attempt` — one Solver attempt against one puzzle, graded out-of-band.
- :func:`run_pass_hat_k` — ``k`` sibling attempts → a :class:`PuzzleHistory`, the
  native pass^k unit (τ-bench, Yao 2024 — §1, swarm-17 finding 6).

The pipeline (research-grounding §10.2, composed top-to-bottom):

1. **Load** the puzzle (:func:`crucible.puzzle.load_puzzle`) if given a path. The
   oracle is never loaded into Solver-visible state — :class:`LoadedPuzzle` has no
   oracle field by construction (§10.4).
2. **Build the scored context** via :func:`crucible.framing.build_scored_context`
   (Tier-1 task + Tier-2 framing arm). Any Tier-3 :class:`Chrome` is built
   *separately* and the kernel calls :func:`crucible.engagement.assert_no_chrome_leak`
   BEFORE the Solver runs — the sealed boundary is a fail-closed andon (§10.1(d,e)).
3. **Run the Solver** (:class:`crucible.roles.Solver`) inside a
   :class:`crucible.budget.BudgetGovernor` and (if provided) the
   :class:`crucible.sandbox.SandboxEnvironment`. The Solver routes all model I/O
   through the injected ``generate`` choke point; the kernel — never the model —
   records every model/tool call as a :class:`TraceEvent` and stamps
   ``terminated_by`` from any :class:`BudgetExceeded` (ANDON).
4. **Grade out-of-band** (§10.4): after the Solver halts, the injected
   ``oracle_runner`` (which represents the separate grading host reading the sealed
   oracle against the copied workdir) yields an :class:`OracleOutcome`; the kernel
   passes it to :func:`crucible.scoring.oracle.grade` → ``attempt.scores["oracle"]``.
   The kernel itself never reads the oracle into the Solver context.
5. **Panel** (optional, EXTERNAL_VERIFIER §10.2): if ``judges`` are supplied, run
   :class:`crucible.scoring.judge_panel.JudgePanel` (generator family excluded) →
   ``attempt.scores["panel"]``. If ``enable_critic`` (default OFF, §10.3), invoke
   the :class:`crucible.roles.Critic`.
6. **Write the trace** (:meth:`TraceWriter.to_eval_log`) and optionally append the
   eval-log to a :class:`crucible.attestation.JsonlEventStore`.

Standards compliance (the six — workflow-standards.md):

- **PIN_PER_STEP — 2:** every dependency is injected (``generate``, ``oracle_runner``,
  ``judges``, ``sandbox``), so a run is a pure function of its pinned inputs +
  puzzle artifact; the framing/oracle/panel sub-steps are themselves replayable
  (each Wave-1 module scored 3 here). Pinning the *model* + image digest is the
  provider's job (the local ``generate``/Docker provider waves).
- **ANDON_AUTHORITY — 3:** the kernel is the andon authority — a budget/time/
  hard-kill breach halts the attempt with the right ``terminated_by`` (via the
  governor), and :func:`assert_no_chrome_leak` halts BEFORE any model call on a
  sealed-boundary violation. Both are proven RED in ``tests/test_kernel.py``.
- **NAMED_COMPENSATORS — 2:** the only irreversible local action the kernel takes
  is creating a sandbox workdir / event-log file; both are owned and torn down by
  their modules (:meth:`LocalSandbox.cleanup`, append-only ``JsonlEventStore``).
  No external/irreversible calls (publish/release/network) happen here.
- **DECOMPOSE_BY_SECRETS — 3:** the kernel composes modules across the
  secret/non-secret split — it can name ``oracle_runner`` (the grading edge) but
  never the oracle artifact, and it asserts the chrome decomposition held at
  runtime. The architecture *is* this principle.
- **UNCERTAINTY_GATED_HUMANS — n/a:** the kernel runs unattended; human checkpoints
  live in the catalog/graduation layer (Phase 4), not in a single attempt.
- **EXTERNAL_VERIFIER — 3:** grading is out-of-band (the injected ``oracle_runner``
  reading a sealed oracle the kernel never loads), and the judge panel is a
  different model family with the generator's reasoning hidden (the kernel passes
  ``generator_family`` so same-family judges are excluded structurally).
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Awaitable, Callable
from pathlib import Path

from crucible.budget import BudgetExceeded, BudgetGovernor
from crucible.engagement import assert_no_chrome_leak
from crucible.framing import build_scored_context
from crucible.observability import PuzzleHistory
from crucible.puzzle import LoadedPuzzle, load_puzzle
from crucible.roles import Critic, GenerateFn, SandboxTools, Solver
from crucible.sandbox import SandboxEnvironment
from crucible.scoring.judge_panel import JudgeFn, JudgePanel
from crucible.scoring.oracle import OracleOutcome, grade
from crucible.trace import TraceWriter
from crucible.types import (
    AttemptState,
    Budget,
    Chrome,
    FramingArm,
    PuzzleMeta,
    Score,
    TerminatedBy,
    TraceEvent,
)

__all__ = [
    "OracleRunner",
    "run_attempt",
    "run_pass_hat_k",
]

#: The out-of-band grading edge (§10.4). Given the halted attempt and the puzzle
#: contract, return the task-specific :class:`OracleOutcome`. This stands in for
#: the SEPARATE grading host that reads the sealed oracle against the copied-out
#: workdir — the kernel never reads the oracle itself, it only *asks* this edge.
#: Injected so tests pass a canned async outcome and Phase-2 wires the real
#: copy-workdir-out + sealed-oracle harness behind the same shape.
OracleRunner = Callable[[AttemptState, PuzzleMeta], Awaitable[OracleOutcome]]

#: Key under which the live :class:`Solver` is parked on ``state.metadata`` for the
#: duration of the Solver turn, so the injected ``generate`` can drive tool calls
#: through the kernel-side governor (``await solver.record_tool_call(...)``) without
#: widening the ``generate`` signature. Removed once the turn ends.
_SOLVER_HANDLE = "_kernel_solver"


def _resolve_puzzle(puzzle: LoadedPuzzle | Path) -> LoadedPuzzle:
    """Accept either a pre-loaded puzzle or a directory path.

    A :class:`Path` is loaded via :func:`crucible.puzzle.load_puzzle` (which reads
    only meta/prompt/setup and never the oracle, §10.4). A :class:`LoadedPuzzle` is
    passed through unchanged.
    """
    if isinstance(puzzle, LoadedPuzzle):
        return puzzle
    return load_puzzle(Path(puzzle))


def _new_attempt(
    loaded: LoadedPuzzle, model: str, arm: FramingArm, *, chrome: Chrome | None
) -> AttemptState:
    """Construct a fresh :class:`AttemptState` with a live displayed budget.

    The budget here is the *authoritative* one the governor mutates (distinct from
    the fresh budget :func:`build_scored_context` renders into the prompt text).
    Tier-3 ``chrome`` is attached to the attempt but, per the sealed boundary, will
    never be serialized into ``messages``.
    """
    meta = loaded.meta
    return AttemptState(
        attempt_id=f"att-{uuid.uuid4().hex[:12]}",
        puzzle_id=meta.puzzle_id,
        model=model,
        framing_arm=arm,
        budget=Budget(
            tool_call_budget=meta.tool_call_budget,
            time_budget_seconds=meta.time_budget_seconds,
        ),
        chrome=chrome,
    )


def _wrap_generate(generate: GenerateFn, solver: Solver) -> GenerateFn:
    """Park the live Solver on the attempt so ``generate`` can record tool calls.

    The role contract keeps ``generate`` as ``(AttemptState) -> Awaitable[str]`` so
    every role shares one choke point. To let a Solver-side ``generate`` route tool
    calls through the kernel-side governor (the only legitimate accounting path,
    §10.2 / §8.4) without changing that signature, the kernel stashes the live
    :class:`Solver` under ``state.metadata[_SOLVER_HANDLE]`` for the turn. The
    injected ``generate`` may then ``await state.metadata[_SOLVER_HANDLE].record_tool_call(...)``
    — and a budget/hard-kill breach raised there propagates into
    :meth:`Solver.act`, which stamps ``terminated_by`` (ANDON at the boundary).
    """

    async def _wrapped(state: AttemptState) -> str:
        state.metadata[_SOLVER_HANDLE] = solver
        return await generate(state)

    return _wrapped


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
) -> AttemptState:
    """Run one Solver attempt end-to-end and return the populated attempt.

    Pipeline (§10.2; see module docstring for the cited rationale):

    1. Load the puzzle if a :class:`Path` (oracle stays grading-side, §10.4).
    2. Build the scored context (:func:`build_scored_context`) for ``arm`` and set
       ``attempt.messages``; assert the Tier-3 ``chrome`` did not leak into that
       context BEFORE the Solver runs (:func:`assert_no_chrome_leak`, fail-closed).
    3. Run the :class:`Solver` inside a :class:`BudgetGovernor` (and ``sandbox`` if
       given), recording every model/tool call as a :class:`TraceEvent`. A
       :class:`BudgetExceeded` stamps ``terminated_by``.
    4. Grade out-of-band: ``oracle_runner`` → :class:`OracleOutcome` →
       :func:`crucible.scoring.oracle.grade` → ``attempt.scores["oracle"]``.
    5. If ``judges`` given: :class:`JudgePanel` (generator family excluded) →
       ``attempt.scores["panel"]``. If ``enable_critic``: invoke the Critic.
    6. Write the Inspect-shaped eval-log and optionally append it to ``event_store``.

    Args:
        puzzle: a loaded puzzle or the path to a puzzle directory.
        model: the model id under test (recorded on the attempt; ``generate`` owns
            the actual model call).
        generate: the single model-I/O choke point (§10.2). May drive tool calls
            via ``await state.metadata[_SOLVER_HANDLE].record_tool_call(...)``.
        oracle_runner: the out-of-band grading edge (§10.4) → :class:`OracleOutcome`.
        arm: which framing arm to render the scored context under (default
            self-referential; §10.1(f)).
        sandbox: the Solver's narrow env channel (§10.4); ``None`` runs with no env
            (a pure-reasoning puzzle / a fake-tool test).
        judges: cross-family judges for the panel; ``None`` skips the panel.
        enable_critic: opt the default-OFF Critic in for this attempt (§10.3).
        chrome: Tier-3 chrome held on the attempt for the human UI; guarded out of
            the scored context.
        panel_reducer: ``"majority"`` (default) or ``"median"`` for the panel.
        generator_family: the model family of ``model`` so the panel can exclude
            same-family judges (EXTERNAL_VERIFIER, §10.2).
        event_store: optional :class:`crucible.attestation.JsonlEventStore`; when
            given, the rendered eval-log is appended for durable provenance (§9.5).

    Returns:
        The populated :class:`AttemptState` — ``messages`` (scored context, never
        chrome), ``output``, ``events`` (the kernel-owned trace), ``scores`` (at
        least ``"oracle"``; ``"panel"`` when judged), ``terminated_by``, ``budget``,
        and ``wall_time``.
    """
    loaded = _resolve_puzzle(puzzle)
    meta = loaded.meta
    attempt = _new_attempt(loaded, model, arm, chrome=chrome)
    writer = TraceWriter()

    # -- 2. Scored context + sealed-boundary andon (BEFORE any model call). ---- #
    attempt.messages = build_scored_context(meta, loaded.prompt, _prior_scores(loaded), arm)
    # Fail-closed: if chrome leaked into the scored context, halt now — motivation
    # must never share a context window with measurement (§10.1(d,e)). This raises
    # SealedBoundaryViolation, never reaching the Solver.
    if attempt.chrome is not None:
        assert_no_chrome_leak(attempt.messages, attempt.chrome)

    # -- 3. Solver inside the governor (+ sandbox), kernel-side trace + ANDON. - #
    governor = BudgetGovernor(attempt.budget, meta=meta)
    # _wrap_generate needs the constructed Solver (so the injected generate can
    # drive tool calls through it), but Solver needs a generate at construction —
    # resolve the chicken/egg by constructing with a sentinel, then assigning the
    # real wrapped generate onto the Solver's choke point.
    solver = Solver(_sentinel_generate(), _sandbox_tools(sandbox), governor)
    solver._generate = _wrap_generate(generate, solver)  # type: ignore[attr-defined]

    start = time.monotonic()
    try:
        attempt = await solver.act(attempt)
    finally:
        attempt.wall_time = time.monotonic() - start
        attempt.metadata.pop(_SOLVER_HANDLE, None)  # don't leak the handle downstream

    # Mirror the role-recorded events into the kernel-owned writer (the writer owns
    # canonical seq numbering + attachment spilling, §10.2 finding 5). The Solver
    # appended TraceEvents during its turn; replay them through the writer so the
    # eval-log is the single audit-ready transcript. ``mirrored`` tracks how many of
    # ``attempt.events`` are already in the writer, independent of the kernel's own
    # direct score-event appends below.
    mirrored = _mirror_events(writer, attempt, since=0)

    # -- 4. Out-of-band oracle grading (§10.4). ------------------------------- #
    outcome = await oracle_runner(attempt, meta)
    attempt.scores["oracle"] = grade(attempt, meta, outcome)
    writer.append(
        TraceEvent(kind="score", payload={"scorer": "oracle",
                                          "value": attempt.scores["oracle"].value})
    )

    # -- 5. Cross-family judge panel (EXTERNAL_VERIFIER, §10.2) + opt-in Critic. #
    if judges:
        panel = JudgePanel(judges, reducer=panel_reducer, generator_family=generator_family)
        attempt.scores["panel"] = await panel.score(attempt)
        writer.append(
            TraceEvent(kind="score", payload={"scorer": "panel",
                                              "value": attempt.scores["panel"].value})
        )
        # Propagate panel-validated novelty onto the oracle score so the rollups
        # (observability._is_novel) see a single validated flag, not a self-claim.
        _propagate_novelty(attempt)

    if enable_critic:
        critic = Critic(_wrap_generate(generate, solver), enabled=True)
        attempt = await critic.act(attempt)
        mirrored = _mirror_events(writer, attempt, since=mirrored)

    # -- 6. Render the Inspect-shaped eval-log + optional durable append (§9.5). #
    eval_log = writer.to_eval_log(attempt)
    attempt.metadata["eval_log"] = eval_log
    if event_store is not None:
        # JsonlEventStore.append (or any object exposing append(dict) -> str).
        event_store.append(eval_log)  # type: ignore[attr-defined]

    return attempt


async def run_pass_hat_k(
    puzzle: LoadedPuzzle | Path,
    model: str,
    k: int,
    **kwargs: object,
) -> PuzzleHistory:
    """Run ``k`` sibling attempts and collect them into a :class:`PuzzleHistory`.

    pass^k — "all k independent trials succeeded" — is the reliability metric the
    literature settles on (τ-bench, Yao 2024 — §1); crucible records it **natively**
    as k sibling attempts under one puzzle history (swarm-17 finding 6), not k
    samples in one log. Each attempt is an independent :func:`run_attempt` call with
    a fresh budget/governor/trace; ``**kwargs`` are forwarded unchanged (so the same
    ``generate`` / ``oracle_runner`` / ``arm`` / ``sandbox`` / ``judges`` apply to
    every sibling).

    The puzzle is loaded once and the :class:`LoadedPuzzle` reused across siblings
    (the oracle is never in it, §10.4), so a directory ``path`` is read from disk a
    single time.

    Args:
        puzzle: a loaded puzzle or path; loaded once and shared across siblings.
        model: the model id under test.
        k: the number of i.i.d. sibling attempts (the consistency depth).
        **kwargs: forwarded to each :func:`run_attempt` (must include ``generate``
            and ``oracle_runner``).

    Returns:
        A :class:`PuzzleHistory` holding the ``k`` attempts; query
        :meth:`PuzzleHistory.pass_hat_k` / :meth:`PuzzleHistory.wilson` for the
        reliability views.

    Raises:
        ValueError: if ``k < 1`` (pass^k needs at least one trial).
    """
    if k < 1:
        raise ValueError(f"run_pass_hat_k needs k >= 1, got {k}")

    loaded = _resolve_puzzle(puzzle)
    history = PuzzleHistory(puzzle_id=loaded.meta.puzzle_id)
    for _ in range(k):
        attempt = await run_attempt(loaded, model, **kwargs)  # type: ignore[arg-type]
        history.add(attempt)
    return history


# --------------------------------------------------------------------------- #
# Internal helpers
# --------------------------------------------------------------------------- #


def _prior_scores(loaded: LoadedPuzzle) -> list[Score] | None:
    """The Solver's own prior scores on this puzzle class for the self-referential
    personal-best ledger (§10.1(b,c)).

    Phase 1 has no persisted per-model history surface yet, so this returns
    ``None`` (the self-referential arm degrades cleanly to NEUTRAL when there is no
    record to beat — see :func:`crucible.framing._personal_best_line`). The seam is
    here so a later wave can inject a model's prior-best without touching the
    pipeline. A caller that already has priors can build the context itself; the
    kernel keeps the default path framing-pure.
    """
    return None


def _sandbox_tools(sandbox: SandboxEnvironment | None) -> SandboxTools:
    """Adapt the optional :class:`SandboxEnvironment` to the Solver's
    :class:`~crucible.roles.SandboxTools` shape.

    The Solver protocol wants ``exec(command: str)/read_file/write_file``; the
    sandbox provider exposes ``exec(cmd: list[str], timeout: float)/read_file/
    write_file`` (§10.4). This thin adapter bridges the two so the kernel does not
    reimplement either. When no sandbox is supplied (a pure-reasoning puzzle or a
    fake-tool test), a no-op tools object is returned — the Solver still routes any
    tool call through the kernel-side governor for accounting; it just has no env.
    """
    if sandbox is None:
        return _NoEnvTools()
    return _SandboxAdapter(sandbox)


def _mirror_events(writer: TraceWriter, attempt: AttemptState, *, since: int) -> int:
    """Replay role-appended :class:`TraceEvent`s into the kernel-owned writer.

    The roles append events onto ``attempt.events`` during their turn; the writer
    owns canonical sequence numbering + large-blob attachment spilling (§10.2
    finding 5). Mirroring keeps the eval-log the single audit-ready transcript while
    leaving the live ``attempt.events`` intact for callers that read it directly.

    ``since`` is the index into ``attempt.events`` to start mirroring from, so a
    second call (after the Critic turn) appends only the newly-added role events —
    independent of any direct score-event appends the kernel made on the writer in
    between. Returns the new high-water mark (``len(attempt.events)``) to thread
    into the next call.
    """
    for event in attempt.events[since:]:
        # Re-wrap so the writer assigns its own seq; large payload text is spilled
        # to an attachment when it crosses the threshold.
        writer.append(
            TraceEvent(
                kind=event.kind,
                role=event.role,
                payload=dict(event.payload),
                attachments=dict(event.attachments),
            )
        )
    return len(attempt.events)


def _propagate_novelty(attempt: AttemptState) -> None:
    """Stamp panel-validated novelty onto the oracle score metadata.

    Novelty is panel-adjudicated, never Solver-self-asserted (§8.7). When the panel
    validated a novel path, the rollup layer (:func:`crucible.observability._is_novel`)
    looks for a ``novelty_validated`` truthy marker on the oracle/panel score. The
    panel score already carries its own metadata; here we copy the validated flag
    onto the oracle score so a single source records it for the leaderboard.
    """
    panel = attempt.scores.get("panel")
    oracle = attempt.scores.get("oracle")
    if panel is None or oracle is None:
        return
    if panel.metadata.get("novelty_validated"):
        oracle.metadata["novelty_validated"] = True


# -- Solver-construction shims (keep the public generate signature stable) ---- #


def _sentinel_generate() -> GenerateFn:
    """A placeholder ``generate`` replaced immediately after Solver construction.

    :func:`_wrap_generate` needs a reference to the constructed Solver, but the
    Solver needs a ``generate`` at construction time — a chicken/egg the kernel
    resolves by constructing with this sentinel then assigning the real wrapped
    generate onto ``solver._generate``. The sentinel is never called.
    """

    async def _never(_state: AttemptState) -> str:  # pragma: no cover - never invoked
        raise RuntimeError("sentinel generate must be replaced before use")

    return _never


class _NoEnvTools:
    """A :class:`~crucible.roles.SandboxTools` with no environment.

    Used when :func:`run_attempt` is called without a ``sandbox``. Tool calls still
    flow through the kernel-side governor for accounting (the Solver records them);
    these methods are the inert backing for a puzzle that needs no env or a test
    that drives tool calls purely for budget/loop accounting.
    """

    async def exec(self, command: str) -> str:
        return ""

    async def read_file(self, path: str) -> str:
        return ""

    async def write_file(self, path: str, content: str) -> None:
        return None


class _SandboxAdapter:
    """Adapt a :class:`crucible.sandbox.SandboxEnvironment` to
    :class:`~crucible.roles.SandboxTools`.

    Bridges the Solver's string-command tool shape to the sandbox provider's
    argv+timeout channel (§10.4) without either module importing the other — the
    kernel owns the seam. ``exec`` splits the command into an argv list (the
    provider runs **no shell**) and applies the attempt's time budget as the
    per-call timeout.
    """

    def __init__(self, sandbox: SandboxEnvironment, *, default_timeout: float = 60.0) -> None:
        self._sandbox = sandbox
        self._timeout = default_timeout

    async def exec(self, command: str) -> str:
        import shlex

        argv = shlex.split(command)
        if not argv:
            return ""
        result = await self._sandbox.exec(argv, timeout=self._timeout)
        if result.timed_out:
            # Surface the wall-clock breach as a budget signal at the call site.
            raise BudgetExceeded(
                TerminatedBy.TIME,
                f"sandbox exec timed out after {self._timeout}s (§8.4)",
            )
        return result.stdout

    async def read_file(self, path: str) -> str:
        return await self._sandbox.read_file(path)

    async def write_file(self, path: str, content: str) -> None:
        await self._sandbox.write_file(path, content)
