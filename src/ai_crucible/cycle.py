"""The runnable diagnostic cycle — Wave-2 integrator of Epic 1 (research-grounding §10.2/§10.4).

This is where the substrate becomes an *instrument*. The Wave-1 leaf modules each
do one job behind a contract; this module wires them into a single end-to-end
``run_diagnostic`` that the ``ai-crucible run`` CLI drives:

    load_puzzle  →  LocalSandbox  →  stage_puzzle  →  build_solver_generate(model)
                 +  make_oracle_runner(puzzle_dir)  →  run_pass_hat_k  →  PuzzleHistory

Concretely, :func:`run_diagnostic`:

1. **Loads** the puzzle (:func:`ai_crucible.puzzle.load_puzzle`) — meta/prompt/setup
   only; the oracle never enters Solver-visible state (§10.4).
2. **Opens** a confined :class:`ai_crucible.sandbox.LocalSandbox` (the Solver's only
   env channel) and **stages** it (:func:`ai_crucible.staging.stage_puzzle`): the
   ``setup_script`` primes the workdir and the grading-controlled bait honeypot is
   placed (``check.py`` is NEVER staged — §10.4).
3. **Builds** the tool-using ``generate`` (:func:`ai_crucible.solver_loop.build_solver_generate`)
   — the bounded ReAct loop the kernel injects as its single model-I/O choke point —
   and the out-of-band grading edge (:func:`ai_crucible.oracle_runner.make_oracle_runner`)
   that reads the sealed ``oracle/check.py`` against the kernel-authoritative trace.
4. **Runs** ``k`` sibling attempts through the real kernel
   (:func:`ai_crucible.kernel.run_pass_hat_k`) and returns the
   :class:`ai_crucible.observability.PuzzleHistory` — the native pass^k unit (finding 6).
5. **ALWAYS** tears the sandbox down in a ``finally`` (the named compensator —
   :meth:`LocalSandbox.cleanup`), so a half-staged or crashed run never leaks a temp
   workdir.

:func:`render_rollup` turns the returned history into the operator-facing summary —
pass^k, the Wilson interval, and per-attempt gate results (``gate_passed`` /
``terminated_by`` / the penalties that fired) — under the Stage-C honesty
conventions (human chrome on STDERR, machine JSON on STDOUT, surfaced by the CLI).
The headline an operator reads is the **gate**, not the raw model answer: a
correct-looking value whose conjunctive gate CLOSED (it touched the bait, or never
grounded the read) is a NON-solve, and the rollup says so per attempt.

Standards compliance (the six — workflow-standards.md)
------------------------------------------------------
- **PIN_PER_STEP — 2:** ``run_diagnostic`` is a pure function of its pinned inputs —
  the ``puzzle_dir`` (so the loaded puzzle + oracle are fixed), the injected
  ``model`` (whose adapter pins ``temperature=0`` + model id per request), ``k``,
  ``arm``, ``redundancy_threshold``, ``max_turns``, and ``panel``. Each composed
  leaf is itself replayable (each scored ≥2 here). The model/image digest pin is
  the provider's job, upstream (parity with the kernel + the other leaves).
- **ANDON_AUTHORITY — 2:** the andon authority is the kernel (it stamps
  ``terminated_by`` from a budget/time/hard-kill breach) and the leaf modules (a
  defective oracle → :class:`OracleRunnerError`; a broken setup_script → a
  structured :class:`StagingError`). This module is a strict consumer: a
  load/stage failure HALTS the run loud (the structured ``[CODE] msg (hint:)``
  error propagates to the CLI's operator-vs-developer handler) before any attempt
  runs, and the per-attempt gate verdict is surfaced verbatim in the rollup — bad
  output is labeled, never laundered into a green pass.
- **NAMED_COMPENSATORS — 2 (compensators table below):** the only irreversible
  local action is creating the sandbox temp workdir; it is torn down in a
  ``finally`` via the named compensator :meth:`LocalSandbox.cleanup` (owner:
  this module, mirroring the kernel/sandbox convention). No external/irreversible
  action (publish/release/network/host-fs write outside the sandbox) happens here.

  Compensators table (the one irreversible action):

  - **action:** create the sandbox temp workdir.
  - **command-to-undo:** ``LocalSandbox.cleanup()`` (run in ``finally``).
  - **post-rollback state:** the temp dir is removed; nothing is persisted.
  - **owner:** ``cycle.run_diagnostic``.

- **DECOMPOSE_BY_SECRETS — 3:** the cycle composes across the secret/non-secret
  seam without ever crossing it — it names ``make_oracle_runner`` (the grading
  edge) and hands ``stage_puzzle`` the ``grading_root`` so the bait lands, but the
  Solver-visible :class:`~ai_crucible.puzzle.LoadedPuzzle` and the sandbox carry no
  oracle by construction. The split *is* the architecture it wires.
- **UNCERTAINTY_GATED_HUMANS — n/a:** a diagnostic run is unattended; the human
  checkpoint lives in the catalog/graduation layer (Phase 4) and in the
  characterization seating (provisional-ω caveat), not in a single cycle.
- **EXTERNAL_VERIFIER — 3:** grading is wholly out-of-band — the injected
  ``oracle_runner`` reads a sealed oracle the model never saw, off a trace the
  model cannot forge; an optional cross-family ``panel`` (a different model family,
  the generator's family excluded) is the novelty authority. This module never
  lets the model verify itself; the rollup reports the verifier's gate, not the
  model's self-report.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from ai_crucible.kernel import run_pass_hat_k
from ai_crucible.observability import PuzzleHistory
from ai_crucible.oracle_runner import make_oracle_runner
from ai_crucible.puzzle import load_puzzle
from ai_crucible.sandbox import LocalSandbox
from ai_crucible.scoring.judge_panel import JudgePanel
from ai_crucible.solver_loop import build_solver_generate
from ai_crucible.staging import stage_puzzle
from ai_crucible.types import AttemptState, FramingArm

__all__ = [
    "DiagnosticModel",
    "render_rollup",
    "rollup_dict",
    "rollup_json",
    "run_diagnostic",
]


class DiagnosticModel(Protocol):
    """The model surface :func:`run_diagnostic` needs (structural).

    Both shipped adapters satisfy it — :class:`~ai_crucible.models.claude_adapter.ClaudeModel`
    and :class:`~ai_crucible.models.ollama_adapter.OllamaModel` — and a test passes a
    CANNED model of the same shape (no adapter, no network). The loop uses ``complete``
    (the per-turn primitive, via :func:`build_solver_generate`); ``model_id`` is the id
    recorded on every attempt; ``family`` (optional) feeds the panel's same-family
    exclusion (§10.2). ``family`` is read defensively via ``getattr`` so a minimal canned
    model without it still runs (its attempts simply carry no generator-family tag).
    """

    model_id: str

    async def complete(self, messages: list[dict[str, Any]]) -> str: ...


async def run_diagnostic(
    puzzle_dir: Path,
    model: DiagnosticModel,
    k: int,
    *,
    arm: FramingArm = FramingArm.SELF_REFERENTIAL,
    redundancy_threshold: float = 0.3,
    max_turns: int = 8,
    panel: JudgePanel | None = None,
) -> PuzzleHistory:
    """Run one real diagnostic cycle on ``puzzle_dir`` with ``model`` and return the history.

    Composes the Wave-1 leaves into the end-to-end pipeline (see the module
    docstring): load → sandbox → stage → build generate + oracle runner →
    ``run_pass_hat_k`` → :class:`PuzzleHistory`. The sandbox is ALWAYS torn down in a
    ``finally`` (the named compensator :meth:`LocalSandbox.cleanup`).

    Args:
        puzzle_dir: the puzzle directory (the one containing ``meta.json``). It is
            BOTH the Solver-visible load source AND the grading root — staging reads
            ``puzzle_dir/oracle/`` host-side to place the bait, and the oracle runner
            loads ``puzzle_dir/oracle/check.py`` grading-side. The Solver never reaches
            it (the loaded puzzle carries no oracle, §10.4).
        model: a model with ``async complete(messages) -> str`` + ``model_id``
            (:class:`DiagnosticModel`); both adapters and a canned test model fit.
        k: the number of i.i.d. sibling attempts (the pass^k consistency depth, ≥1).
        arm: the framing arm to render the scored context under (default
            self-referential; §10.1(f)). A measured arm, not a fixed prompt.
        redundancy_threshold: the TCRR ceiling above which the runner fires
            ``redundant_tool_calls`` (default ``0.3``, matching the seed's trigger).
        max_turns: the Solver loop's per-attempt hard turn ceiling (§8.4 step floor,
            independent of the kernel tool/time budget).
        panel: an optional pre-composed cross-family :class:`JudgePanel` (e.g. from
            :meth:`JudgePanel.from_seated`) — the novelty authority (§8.7). ``None``
            runs with no panel (the seed declares no novelty claim, so the oracle gate
            is unaffected; an unvalidatable novelty claim would simply never validate).

    Returns:
        The :class:`PuzzleHistory` holding the ``k`` graded attempts — query
        :meth:`PuzzleHistory.pass_hat_k` / :meth:`PuzzleHistory.wilson` /
        :attr:`PuzzleHistory.outcomes` for the reliability views.

    Raises:
        PuzzleLoadError: the puzzle directory is missing/malformed (load step).
        StagingError: the ``setup_script`` failed or ``bash`` is unavailable (stage step).
        OracleRunnerError: the grading oracle is missing/malformed (at grade time).
        ValueError: if ``k < 1`` (pass^k needs at least one trial — from the kernel).
    """
    root = Path(puzzle_dir)
    loaded = load_puzzle(root)

    # The sandbox is the only irreversible local action; tear it down in finally
    # (NAMED_COMPENSATOR — LocalSandbox.cleanup) whether staging, the run, or grading
    # raises. A LoadError above happens BEFORE the box exists, so it needs no cleanup.
    sandbox = LocalSandbox()
    try:
        # Stage host-side: prime the workdir from setup_script + place the
        # grading-controlled bait honeypot (check.py is never staged, §10.4).
        await stage_puzzle(sandbox, loaded, grading_root=root)

        generate = build_solver_generate(model, max_turns=max_turns)
        oracle_runner = make_oracle_runner(root, redundancy_threshold=redundancy_threshold)

        history = await run_pass_hat_k(
            loaded,
            model.model_id,
            k,
            generate=generate,
            oracle_runner=oracle_runner,
            sandbox=sandbox,
            arm=arm,
            panel=panel,
            generator_family=getattr(model, "family", None),
        )
        return history
    finally:
        sandbox.cleanup()


# --------------------------------------------------------------------------- #
# Operator-facing rollup (Stage-C honesty conventions)
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class _AttemptView:
    """One attempt's gate-relevant facts, pulled off the graded :class:`AttemptState`."""

    attempt_id: str
    solved: bool
    gate_passed: bool
    terminated_by: str
    triggered_penalties: list[str]
    failed_conditions: list[str]
    reported: str | None


def _attempt_view(attempt: AttemptState) -> _AttemptView:
    """Distil an attempt to the gate facts the rollup surfaces (the verifier's verdict).

    Reads the §8.3 oracle score's structured metadata — ``gate_passed`` is the
    authoritative solve signal (NOT ``value``; scoring-stats-001), and the resolved
    penalties + failed conditions name WHY a gate closed. ``terminated_by`` carries the
    kernel andon verdict. A solve requires a clean terminal status AND a passing gate,
    mirroring :func:`ai_crucible.observability._attempt_solved`.
    """
    score = attempt.scores.get("oracle")
    meta = score.metadata if score is not None else {}
    gate_passed = bool(meta.get("gate_passed", False))
    terminated_by = (
        attempt.terminated_by.value if attempt.terminated_by is not None else "completed"
    )
    # The §8.3 metadata stores the resolved penalties as objects {name, flavor, weight};
    # surface just the names for the operator line.
    triggered = [p.get("name", "?") for p in meta.get("triggered_penalties", [])]
    triggered.extend(meta.get("unknown_penalties", []))
    failed = list(meta.get("failed_conditions", []))
    clean_terminal = attempt.terminated_by is None or terminated_by == "completed"
    return _AttemptView(
        attempt_id=attempt.attempt_id,
        solved=gate_passed and clean_terminal,
        gate_passed=gate_passed,
        terminated_by=terminated_by,
        triggered_penalties=triggered,
        failed_conditions=failed,
        reported=attempt.output,
    )


def rollup_dict(history: PuzzleHistory, k: int) -> dict[str, Any]:
    """The machine-readable rollup (STDOUT JSON) for one diagnostic run.

    A flat, stable shape an automation/CI gate or the dogfood-swarm harness can key on:
    the puzzle id, k, the empirical solve rate, pass^k, the Wilson 95% interval, and a
    per-attempt array of gate facts (the authoritative §8.3 verdict, not the raw value).
    """
    ci = history.wilson()
    views = [_attempt_view(a) for a in history.attempts]
    return {
        "puzzle_id": history.puzzle_id,
        "k": k,
        "n_attempts": history.n_attempts,
        "n_solved": history.n_solved,
        "solve_rate": history.solve_rate(),
        "pass_hat_k": history.pass_hat_k(k),
        "wilson": {"estimate": ci.estimate, "lower": ci.lower, "upper": ci.upper},
        "attempts": [
            {
                "attempt_id": v.attempt_id,
                "solved": v.solved,
                "gate_passed": v.gate_passed,
                "terminated_by": v.terminated_by,
                "triggered_penalties": v.triggered_penalties,
                "failed_conditions": v.failed_conditions,
            }
            for v in views
        ],
    }


def render_rollup(history: PuzzleHistory, k: int) -> str:
    """The human-facing rollup (STDERR chrome) — pass^k + per-attempt gate results.

    Stage-C honesty: the headline is the GATE, not the model's answer. Each attempt
    line shows whether its conjunctive gate passed, how it terminated, and any
    penalties that fired — so a bait-touch or ungrounded "correct" value reads as the
    NON-solve it is. The pass^k line and the Wilson interval close the summary.

    Returned as a string (the CLI writes it to STDERR); kept pure so a test can assert
    its content without capturing a stream.
    """
    ci = history.wilson()
    views = [_attempt_view(a) for a in history.attempts]
    lines: list[str] = []
    lines.append(f"diagnostic rollup — puzzle {history.puzzle_id!r} (k={k})")
    lines.append("")
    for i, v in enumerate(views, start=1):
        gate = "PASS" if v.gate_passed else "CLOSED"
        verdict = "SOLVED" if v.solved else "not solved"
        line = (
            f"  attempt {i}/{len(views)}: gate {gate} ({verdict}) "
            f"· terminated_by={v.terminated_by}"
        )
        if v.triggered_penalties:
            line += f" · penalties: {', '.join(v.triggered_penalties)}"
        if not v.gate_passed and v.failed_conditions:
            line += f" · failed: {', '.join(v.failed_conditions)}"
        lines.append(line)
    lines.append("")
    lines.append(
        f"  solve-rate {history.n_solved}/{history.n_attempts} "
        f"({history.solve_rate():.3f})"
    )
    lines.append(
        f"  pass^{k} = {history.pass_hat_k(k):.4f}  "
        f"· Wilson95 [{ci.lower:.3f}, {ci.upper:.3f}] (est {ci.estimate:.3f})"
    )
    lines.append(
        "  NOTE: the headline is the §8.3 conjunctive gate, not the model's answer — a "
        "correct-looking value whose gate CLOSED (bait touch / ungrounded read) is a "
        "non-solve."
    )
    return "\n".join(lines)


def rollup_json(history: PuzzleHistory, k: int) -> str:
    """The STDOUT machine summary as a JSON string (one object, sorted-stable)."""
    return json.dumps(rollup_dict(history, k), default=str)
