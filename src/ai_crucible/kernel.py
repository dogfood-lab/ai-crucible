"""The AI Crucible kernel — the thin policy layer that composes everything (§10.2).

This is the integrator module (Wave 2). It does **not** reimplement scoring,
sandboxing, framing, budgeting, tracing, or the role machinery — it *wires the
Wave-1 leaf modules together* into the two public entry points ai_crucible runs on:

- :func:`run_attempt` — one Solver attempt against one puzzle, graded out-of-band.
- :func:`run_pass_hat_k` — ``k`` sibling attempts → a :class:`PuzzleHistory`, the
  native pass^k unit (τ-bench, Yao 2024 — §1, swarm-17 finding 6).

The pipeline (research-grounding §10.2, composed top-to-bottom):

1. **Load** the puzzle (:func:`ai_crucible.puzzle.load_puzzle`) if given a path. The
   oracle is never loaded into Solver-visible state — :class:`LoadedPuzzle` has no
   oracle field by construction (§10.4).
2. **Build the scored context** via :func:`ai_crucible.framing.build_scored_context`
   (Tier-1 task + Tier-2 framing arm). Any Tier-3 :class:`Chrome` is built
   *separately* and the kernel calls :func:`ai_crucible.engagement.assert_no_chrome_leak`
   BEFORE the Solver runs — the sealed boundary is a fail-closed andon (§10.1(d,e)).
3. **Run the Solver** (:class:`ai_crucible.roles.Solver`) inside a
   :class:`ai_crucible.budget.BudgetGovernor` and (if provided) the
   :class:`ai_crucible.sandbox.SandboxEnvironment`. The Solver routes all model I/O
   through the injected ``generate`` choke point; the kernel — never the model —
   records every model/tool call as a :class:`TraceEvent` and stamps
   ``terminated_by`` from any :class:`BudgetExceeded` (ANDON). The wall-clock
   budget is enforced LIVE (§8.4): the kernel reads ``time_source`` per Solver turn
   *and* at the attempt boundary, calling :meth:`BudgetGovernor.check_time`, so a
   time overrun halts with ``terminated_by == TIME`` rather than only being flagged
   post-hoc by the oracle.
4. **Grade out-of-band** (§10.4) with **panel-adjudicated novelty** (§8.7): after
   the Solver halts, the injected ``oracle_runner`` (the separate grading host
   reading the sealed oracle against the copied workdir) yields an
   :class:`OracleOutcome` (authoritative for solve / regression / penalties /
   budgets). When ``novelty`` is *claimed*, the cross-family panel rules FIRST and
   ITS verdict — not the oracle_runner's self-report — is fed into
   ``outcome.novelty_validated`` before :func:`ai_crucible.scoring.oracle.grade` →
   ``attempt.scores["oracle"]``. The panel is the novelty authority; absent a panel
   an unvalidatable claim never validates. The kernel never reads the oracle itself.
5. **Panel score** (optional, EXTERNAL_VERIFIER §10.2): if ``judges`` are supplied,
   the :class:`ai_crucible.scoring.judge_panel.JudgePanel` (generator family excluded)
   score computed in step 4 is recorded as ``attempt.scores["panel"]`` (it carries
   the aggregated ``novelty_validated`` verdict). If ``enable_critic`` (default OFF,
   §10.3), invoke the :class:`ai_crucible.roles.Critic`.
6. **Write the trace** (:meth:`TraceWriter.to_eval_log`) and optionally append the
   eval-log to a :class:`ai_crucible.attestation.JsonlEventStore`.

The post-Solver tail (steps 4–6: oracle grade, panel score, eval-log render/persist)
is wrapped in a single degrade envelope (kernel-runtime-001). The grading host (§10.4)
and the judge panel (§10.2 / §8.6) are the dependencies most likely to fail at runtime;
a raw exception there would discard the Solver's completed (expensive) work with no
trace and no diagnosable cause. Instead any failure stamps ``terminated_by = ERROR`` and
a structured ``attempt.error`` NAMING the failing stage ('oracle grading' / 'panel
scoring' / 'eval-log persist'), appends an error :class:`TraceEvent`, STILL renders +
best-effort persists the eval-log so the Solver transcript survives, and returns the
populated (degraded) attempt. A sealed-boundary andon
(:class:`~ai_crucible.engagement.SealedBoundaryViolation` /
:class:`~ai_crucible.roles.ChromeAccessError`) is the one thing NOT degraded — a
contaminated attempt must HALT, never be returned as a graded result (§10.1(e)).

Standards compliance (the six — workflow-standards.md):

- **PIN_PER_STEP — 2:** every dependency is injected (``generate``, ``oracle_runner``,
  ``judges``, ``sandbox``), so a run is a pure function of its pinned inputs +
  puzzle artifact; the framing/oracle/panel sub-steps are themselves replayable
  (each Wave-1 module scored 3 here). Pinning the *model* + image digest is the
  provider's job (the local ``generate``/Docker provider waves).
- **ANDON_AUTHORITY — 3:** the kernel is the andon authority — a tool-budget,
  LIVE wall-clock (``check_time`` per turn + at the boundary, §8.4 / H3), or
  hard-kill breach halts the attempt with the right ``terminated_by`` (via the
  governor), and :func:`assert_no_chrome_leak` halts BEFORE the Solver AND re-runs
  AFTER the Solver/Critic turns on a sealed-boundary violation. The post-Solver
  grading tail degrades a dead grading host / zero-survivor panel into a traced,
  stage-named ``terminated_by == ERROR`` attempt rather than discarding the run
  (kernel-runtime-001) — but a sealed-boundary andon is never degraded (it still
  halts). All are proven in ``tests/test_kernel.py`` (BUDGET, TIME, HARD_KILL,
  post-turn chrome leak, and the oracle/panel grading-failure degrade paths).
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
  ``generator_family`` so same-family judges are excluded structurally). Novelty —
  the one capability claim the Solver could otherwise self-certify — is adjudicated
  by that cross-family panel BEFORE the gate; the oracle_runner's self-reported
  ``novelty_validated`` is never trusted for the gate (H2, §8.7).
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Awaitable, Callable, Iterator
from contextlib import contextmanager
from pathlib import Path

from ai_crucible.budget import BudgetExceeded, BudgetGovernor
from ai_crucible.engagement import SealedBoundaryViolation, assert_no_chrome_leak
from ai_crucible.framing import build_scored_context
from ai_crucible.observability import PuzzleHistory
from ai_crucible.puzzle import LoadedPuzzle, load_puzzle
from ai_crucible.roles import ChromeAccessError, Critic, GenerateFn, SandboxTools, Solver
from ai_crucible.sandbox import SandboxEnvironment
from ai_crucible.scoring.judge_panel import JudgeFn, JudgePanel
from ai_crucible.scoring.oracle import OracleOutcome, grade
from ai_crucible.trace import TraceWriter
from ai_crucible.types import (
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

    A :class:`Path` is loaded via :func:`ai_crucible.puzzle.load_puzzle` (which reads
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


def _wrap_generate(
    generate: GenerateFn,
    solver: Solver,
    *,
    governor: BudgetGovernor | None = None,
    time_source: Callable[[], float] | None = None,
    start: float = 0.0,
) -> GenerateFn:
    """Park the live Solver on the attempt so ``generate`` can record tool calls.

    The role contract keeps ``generate`` as ``(AttemptState) -> Awaitable[str]`` so
    every role shares one choke point. To let a Solver-side ``generate`` route tool
    calls through the kernel-side governor (the only legitimate accounting path,
    §10.2 / §8.4) without changing that signature, the kernel stashes the live
    :class:`Solver` under ``state.metadata[_SOLVER_HANDLE]`` for the turn. The
    injected ``generate`` may then ``await state.metadata[_SOLVER_HANDLE].record_tool_call(...)``
    — and a budget/hard-kill breach raised there propagates into
    :meth:`Solver.act`, which stamps ``terminated_by`` (ANDON at the boundary).

    **Live time budget (H3, §8.4).** When ``governor`` + ``time_source`` are given,
    the wrapper calls :meth:`BudgetGovernor.check_time` *per turn* (before handing
    off to the model) with ``time_source() - start``. A wall-clock overrun therefore
    raises ``BudgetExceeded(TIME)`` from *inside* :meth:`Solver.act` — caught there
    and stamped onto ``terminated_by`` — so a long-running attempt is halted LIVE,
    not merely flagged after the fact by the oracle. The check is kernel-side; the
    model never reports its own elapsed time (Vivaria, §10.2). The Critic turn wires
    its own ``generate`` without these, so only the Solver block is time-governed
    (the budget is scoped to the Solver block, §10.2).
    """

    async def _wrapped(state: AttemptState) -> str:
        state.metadata[_SOLVER_HANDLE] = solver
        if governor is not None and time_source is not None:
            governor.check_time(time_source() - start)
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
    panel: JudgePanel | None = None,
    enable_critic: bool = False,
    chrome: Chrome | None = None,
    panel_reducer: str = "majority",
    generator_family: str | None = None,
    event_store: object | None = None,
    time_source: Callable[[], float] = time.monotonic,
) -> AttemptState:
    """Run one Solver attempt end-to-end and return the populated attempt.

    Pipeline (§10.2; see module docstring for the cited rationale):

    1. Load the puzzle if a :class:`Path` (oracle stays grading-side, §10.4).
    2. Build the scored context (:func:`build_scored_context`) for ``arm`` and set
       ``attempt.messages``; assert the Tier-3 ``chrome`` did not leak into that
       context BEFORE the Solver runs (:func:`assert_no_chrome_leak`, fail-closed).
    3. Run the :class:`Solver` inside a :class:`BudgetGovernor` (and ``sandbox`` if
       given), recording every model/tool call as a :class:`TraceEvent`. A
       :class:`BudgetExceeded` (tool / LIVE wall-clock / hard-kill) stamps
       ``terminated_by``; re-assert the chrome boundary after the Solver turn.
    4. Grade out-of-band: ``oracle_runner`` → :class:`OracleOutcome`; when novelty is
       claimed, the :class:`JudgePanel` rules first and its verdict overrides
       ``outcome.novelty_validated`` (§8.7) → :func:`ai_crucible.scoring.oracle.grade`
       → ``attempt.scores["oracle"]``.
    5. Record the panel score (generator family excluded) → ``attempt.scores["panel"]``.
       If ``enable_critic``: invoke the Critic and re-assert the chrome boundary.
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
        judges: cross-family judges for the panel; ``None`` skips the panel. When a
            novelty bonus is claimed, the panel is the validation authority (§8.7) —
            each judge may carry a ``novelty_validated`` vote in its score metadata,
            aggregated into the panel verdict that drives the oracle gate. Ignored when
            ``panel`` is supplied.
        panel: a pre-built :class:`JudgePanel` to use as-is (e.g.
            :meth:`JudgePanel.from_seated` — the composed, reliability-weighted,
            cross-family panel from characterization, §11.4). Takes precedence over
            ``judges``/``panel_reducer``/``generator_family`` (the supplied panel carries
            its own reducer + exclusion). ``None`` (default) builds a panel from ``judges``.
        enable_critic: opt the default-OFF Critic in for this attempt (§10.3).
        chrome: Tier-3 chrome held on the attempt for the human UI; guarded out of
            the scored context.
        panel_reducer: ``"majority"`` (default) or ``"median"`` for the panel.
        generator_family: the model family of ``model`` so the panel can exclude
            same-family judges (EXTERNAL_VERIFIER, §10.2).
        event_store: optional :class:`ai_crucible.attestation.JsonlEventStore`; when
            given, the rendered eval-log is appended for durable provenance (§9.5).
        time_source: monotonic clock the kernel reads for the live time-budget
            check (§8.4 / H3); defaults to :func:`time.monotonic`. Injected so the
            time enforcement is deterministically testable (a fake clock) without
            real sleeps. Read once at attempt start, then per Solver turn and once
            at the attempt boundary — a reading past ``time_budget_seconds`` halts
            the attempt with ``terminated_by == TIME``.

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
    # Capture the wall-clock start BEFORE wiring generate so the per-turn live time
    # check (§8.4 / H3) measures from the true attempt start.
    start = time_source()
    solver._generate = _wrap_generate(  # type: ignore[attr-defined]
        generate, solver, governor=governor, time_source=time_source, start=start
    )

    try:
        attempt = await solver.act(attempt)
        # Attempt-boundary time check (§8.4 / H3): even if every turn read under
        # budget, the *total* elapsed may have overrun. Re-check at the boundary and
        # stamp TIME over a clean COMPLETED so a wall-clock overrun never escapes as
        # a success. A budget exception already stamped by the Solver is left as-is.
        if attempt.terminated_by in (None, TerminatedBy.COMPLETED):
            try:
                governor.check_time(time_source() - start)
            except BudgetExceeded as exc:
                attempt.terminated_by = exc.terminated_by
                attempt.error = str(exc)
                attempt.events.append(
                    TraceEvent(
                        kind="error",
                        payload={"terminated_by": exc.terminated_by.value,
                                 "message": str(exc)},
                        seq=len(attempt.events),
                    )
                )
    finally:
        attempt.wall_time = time_source() - start
        attempt.metadata.pop(_SOLVER_HANDLE, None)  # don't leak the handle downstream

    # Sealed-boundary re-check AFTER the Solver turn (H1, §10.1(e)). The pre-Solver
    # guard covers the *initial* scored context, but the Solver turn is a
    # message-mutating site — it (or its injected generate) may add or edit messages
    # — so the andon must re-fire over the full context. This catches leaks the
    # per-turn role guard misses (e.g. an in-place edit of an existing message,
    # which is outside the role guard's appended-slice view).
    _assert_no_chrome_leak_if_present(attempt)

    # Mirror the role-recorded events into the kernel-owned writer (the writer owns
    # canonical seq numbering + attachment spilling, §10.2 finding 5). The Solver
    # appended TraceEvents during its turn; replay them through the writer so the
    # eval-log is the single audit-ready transcript. ``mirrored`` tracks how many of
    # ``attempt.events`` are already in the writer, independent of the kernel's own
    # direct score-event appends below.
    mirrored = _mirror_events(writer, attempt, since=0)

    # -- 4–6. Post-Solver grading / panel / eval-log, GRACEFULLY DEGRADED
    # (kernel-runtime-001). The Solver's turn is done and its transcript is captured;
    # the remaining steps each depend on a runtime edge most likely to fail (the
    # SEPARATE grading host per §10.4, the cross-family judge panel per §10.2/§8.6).
    # A raw exception here would discard an expensive completed run with no trace and
    # no diagnosable cause. Instead we wrap the whole block: on ANY failure we stamp
    # terminated_by = ERROR, set a structured attempt.error NAMING the failing stage
    # ('oracle grading' / 'panel scoring' / 'eval-log persist'), append an error
    # TraceEvent, STILL render + best-effort persist the eval_log so the Solver
    # transcript survives, and RETURN the populated (degraded) attempt rather than
    # re-raising (SHARED GRADING-SEAM CONTRACT, Agent A). A flaky single judge is
    # degraded one level down by JudgePanel.score (a partial panel, no raise); only a
    # zero-survivor panel or a dead grading host reaches this catch.
    try:
        await _grade_and_panel(
            attempt,
            meta,
            writer,
            oracle_runner=oracle_runner,
            judges=judges,
            panel=panel,
            panel_reducer=panel_reducer,
            generator_family=generator_family,
            enable_critic=enable_critic,
            generate=generate,
            solver=solver,
            mirrored=mirrored,
        )
    except (SealedBoundaryViolation, ChromeAccessError):
        # A sealed-boundary andon is NOT a degradable grading failure — a contaminated
        # attempt must HALT, never be returned as a degraded result (§10.1(e)). It
        # propagates past the degrade path unchanged (it already rendered no eval_log,
        # by design — a leaked attempt is never gradeable/persistable).
        raise
    except Exception as exc:  # noqa: BLE001 — degrade, don't discard the Solver's work.
        _degrade_into_error(attempt, writer, exc)

    # -- Render the Inspect-shaped eval-log + best-effort durable append (§9.5). --
    # Always runs — on the happy path AND after a degraded grading failure — so the
    # Solver transcript is rendered (and persisted if a store is given) even when
    # grading died. A persist failure is itself degraded into the ERROR stage rather
    # than masking the (possibly already-stamped) grading cause.
    eval_log = writer.to_eval_log(attempt)
    attempt.metadata["eval_log"] = eval_log
    if event_store is not None:
        try:
            # JsonlEventStore.append (or any object exposing append(dict) -> str).
            event_store.append(eval_log)  # type: ignore[attr-defined]
        except Exception as exc:  # noqa: BLE001 — persist is best-effort; trace + return.
            if attempt.terminated_by not in (TerminatedBy.ERROR,):
                _degrade_into_error(attempt, writer, exc, stage="eval-log persist")
            # Re-render so the eval_log on the attempt reflects the persist-failure
            # error stamp (the in-memory record stays consistent even though the
            # durable append did not land).
            attempt.metadata["eval_log"] = writer.to_eval_log(attempt)

    return attempt


async def run_pass_hat_k(
    puzzle: LoadedPuzzle | Path,
    model: str,
    k: int,
    **kwargs: object,
) -> PuzzleHistory:
    """Run ``k`` sibling attempts and collect them into a :class:`PuzzleHistory`.

    pass^k — "all k independent trials succeeded" — is the reliability metric the
    literature settles on (τ-bench, Yao 2024 — §1); ai_crucible records it **natively**
    as k sibling attempts under one puzzle history (swarm-17 finding 6), not k
    samples in one log. Each attempt is an independent :func:`run_attempt` call with
    a fresh budget/governor/trace; ``**kwargs`` are forwarded unchanged (so the same
    ``generate`` / ``oracle_runner`` / ``arm`` / ``sandbox`` / ``judges`` apply to
    every sibling).

    The puzzle is loaded once and the :class:`LoadedPuzzle` reused across siblings
    (the oracle is never in it, §10.4), so a directory ``path`` is read from disk a
    single time.

    **Per-sibling isolation (kernel-runtime-001).** A grading failure in one sibling
    (dead grading host, zero-survivor panel) does NOT abort the cohort: :func:`run_attempt`
    degrades that sibling into a returned ``terminated_by == ERROR`` attempt rather than
    raising, so it is recorded as a non-solve (a non-COMPLETED attempt never counts as a
    solve, finding 6) and the remaining siblings still run and are counted. A
    sealed-boundary violation is the exception — it propagates (a contaminated framing
    breaks the whole cohort), by design.

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
    record to beat — see :func:`ai_crucible.framing._personal_best_line`). The seam is
    here so a later wave can inject a model's prior-best without touching the
    pipeline. A caller that already has priors can build the context itself; the
    kernel keeps the default path framing-pure.
    """
    return None


def _sandbox_tools(sandbox: SandboxEnvironment | None) -> SandboxTools:
    """Adapt the optional :class:`SandboxEnvironment` to the Solver's
    :class:`~ai_crucible.roles.SandboxTools` shape.

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


async def _grade_and_panel(
    attempt: AttemptState,
    meta: PuzzleMeta,
    writer: TraceWriter,
    *,
    oracle_runner: OracleRunner,
    judges: list[JudgeFn] | None,
    panel: JudgePanel | None,
    panel_reducer: str,
    generator_family: str | None,
    enable_critic: bool,
    generate: GenerateFn,
    solver: Solver,
    mirrored: int,
) -> None:
    """Steps 4–6 of the pipeline (oracle grade + panel score + opt-in Critic), with
    each failing stage NAMED so :func:`_degrade_into_error` can label the cause.

    Order matters (H2, §8.7): the cross-family panel is the novelty authority, so it
    must rule BEFORE the gate is computed and ITS verdict — never the oracle_runner's
    self-report — decides whether the novelty bonus applies. We therefore: (a) get the
    task verdict from the oracle_runner (solve / regression / penalties / budgets stay
    authoritative from it); (b) run the panel; (c) OVERRIDE outcome.novelty_validated
    with the panel verdict; (d) grade. Each ``await`` on a runtime edge is wrapped so a
    failure re-raises as a :class:`_GradingStageError` naming the stage; the caller's
    single try/except degrades that into a traced, returned attempt. The opt-in Critic
    turn (and its post-turn sealed-boundary re-check + event mirroring) runs here too,
    so the whole post-Solver tail shares one degrade envelope. Mutates ``attempt`` and
    ``writer`` in place (returns nothing).
    """
    # -- 4. Out-of-band oracle grading (§10.4) with PANEL-adjudicated novelty. - #
    with _stage("oracle grading"):
        outcome = await oracle_runner(attempt, meta)

    panel_score: Score | None = None
    # A pre-composed panel (e.g. JudgePanel.from_seated — characterization → scoring,
    # §11.4) is used as-is, carrying its OWN reducer + generator_family; otherwise build
    # one from the injected judges. ``panel`` takes precedence over ``judges`` when both
    # are given.
    active_panel = panel
    if active_panel is None and judges:
        active_panel = JudgePanel(judges, reducer=panel_reducer, generator_family=generator_family)
    if active_panel is not None:
        with _stage("panel scoring"):
            panel_score = await active_panel.score(attempt)

    # The oracle_runner is NOT trusted for novelty validation. With a panel, the
    # panel's aggregated verdict governs; without a panel, an unvalidatable claim is
    # never validated (you don't get to assert your own bonus, §8.3/§8.7). Solve and
    # all other gate dimensions remain as the oracle_runner reported them.
    if outcome.novelty_claimed:
        outcome.novelty_validated = bool(
            panel_score.metadata.get("novelty_validated")
        ) if panel_score is not None else False

    attempt.scores["oracle"] = grade(attempt, meta, outcome)
    writer.append(
        TraceEvent(kind="score", payload={"scorer": "oracle",
                                          "value": attempt.scores["oracle"].value})
    )

    # -- 5. Record the panel score (EXTERNAL_VERIFIER, §10.2) + opt-in Critic. -- #
    if panel_score is not None:
        attempt.scores["panel"] = panel_score
        writer.append(
            TraceEvent(kind="score", payload={"scorer": "panel",
                                              "value": panel_score.value})
        )

    if enable_critic:
        with _stage("critic"):
            critic = Critic(_wrap_generate(generate, solver), enabled=True)
            attempt = await critic.act(attempt)
        # Sealed-boundary re-check AFTER the Critic turn (H1, §10.1(e)). The Critic
        # appends/edits messages (string-in/string-out), so it is another
        # message-mutating site the andon must cover with the same strong guard. This
        # andon is INTENTIONALLY outside the degrade path: a contaminated attempt must
        # halt (it is never gradeable), not be returned as a degraded result.
        _assert_no_chrome_leak_if_present(attempt)
        # Mirror the Critic's newly-appended role events into the kernel-owned writer
        # (the caller renders the eval-log from the writer right after this returns).
        _mirror_events(writer, attempt, since=mirrored)


class _GradingStageError(RuntimeError):
    """Internal: wraps a post-Solver stage failure with the stage name attached, so
    :func:`_degrade_into_error` can build a structured, stage-naming attempt.error
    (kernel-runtime-001). Never escapes :func:`run_attempt`."""

    def __init__(self, stage: str, cause: BaseException) -> None:
        super().__init__(str(cause))
        self.stage = stage
        self.__cause__ = cause


@contextmanager
def _stage(name: str) -> Iterator[None]:
    """Tag any exception raised inside the block with the failing pipeline ``name``.

    A sealed-boundary andon is NOT a degradable grading failure — a contaminated
    attempt must HALT, never be returned as a degraded result — so a
    :class:`SealedBoundaryViolation` or the Critic-turn role :class:`ChromeAccessError`
    (and a deliberately-raised :class:`_GradingStageError`) propagate unwrapped. Every
    other exception becomes a stage-named :class:`_GradingStageError` so the caller's
    single catch can name 'oracle grading' vs 'panel scoring' vs 'critic' in the
    structured error.
    """
    try:
        yield
    except (SealedBoundaryViolation, ChromeAccessError, _GradingStageError):
        raise
    except Exception as exc:  # noqa: BLE001 — re-raise tagged with the stage.
        raise _GradingStageError(name, exc) from exc


def _degrade_into_error(
    attempt: AttemptState,
    writer: TraceWriter,
    exc: BaseException,
    *,
    stage: str | None = None,
) -> None:
    """Stamp a post-Solver failure onto the attempt as a traced, structured ERROR.

    The Solver's completed work is preserved; only the grading/persist tail failed.
    Sets ``terminated_by = ERROR`` and a structured ``attempt.error`` (Ship-Gate-B
    code/message/hint shape) NAMING the failing stage, and appends an error
    :class:`TraceEvent` to the kernel-owned writer so the eval-log records WHY the
    attempt degraded (kernel-runtime-001). Does NOT re-raise — the caller returns the
    degraded attempt.
    """
    failing_stage = stage or getattr(exc, "stage", None) or "grading"
    message = (
        f"[GRADING_FAILED] post-Solver {failing_stage} raised "
        f"{type(exc).__name__}: {exc} "
        f"(hint: the Solver transcript is preserved in this attempt's eval_log; "
        f"the {failing_stage} edge — the out-of-band grading host (§10.4) or the "
        f"cross-family judge panel (§10.2) — failed at runtime and was degraded "
        f"rather than discarding the run; inspect/retry that edge)"
    )
    attempt.terminated_by = TerminatedBy.ERROR
    attempt.error = message
    payload = {
        "terminated_by": TerminatedBy.ERROR.value,
        "stage": failing_stage,
        "message": message,
    }
    # Append to BOTH the live attempt trace (so a caller reading attempt.events sees
    # the failure, mirroring the budget-breach convention at the Solver boundary) and
    # the kernel-owned writer (so the rendered eval-log records WHY it degraded).
    attempt.events.append(TraceEvent(kind="error", payload=dict(payload), seq=len(attempt.events)))
    writer.append(TraceEvent(kind="error", payload=dict(payload)))


def _assert_no_chrome_leak_if_present(attempt: AttemptState) -> None:
    """Re-run the sealed-boundary andon over the FULL scored context (H1, §10.1(e)).

    A no-op when the attempt carries no Tier-3 chrome (nothing can leak). Otherwise
    it re-runs :func:`ai_crucible.engagement.assert_no_chrome_leak` over every message
    so the kernel — the andon authority — asserts the boundary held after each
    message-mutating turn (Solver, Critic). This is broader than any per-turn role
    guard, which only inspects the messages *appended* during its own turn; an
    in-place edit of an existing message, or any path that bypasses a role guard,
    is still caught here. Raises :class:`~ai_crucible.engagement.SealedBoundaryViolation`
    on a leak (never swallowed — a contaminated attempt must halt, not be graded).
    """
    if attempt.chrome is not None:
        assert_no_chrome_leak(attempt.messages, attempt.chrome)


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
    """A :class:`~ai_crucible.roles.SandboxTools` with no environment.

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
    """Adapt a :class:`ai_crucible.sandbox.SandboxEnvironment` to
    :class:`~ai_crucible.roles.SandboxTools`.

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
