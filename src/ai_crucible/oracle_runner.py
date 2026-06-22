"""The out-of-band grading host — Leaf A of Epic 1 (research-grounding §10.4).

This module is the concrete :data:`ai_crucible.kernel.OracleRunner` the kernel
injects at step 4 of the pipeline: given a halted :class:`AttemptState` and the
:class:`PuzzleMeta` contract, it produces the task-specific
:class:`~ai_crucible.scoring.oracle.OracleOutcome` that the §8.3 conjunctive gate
(:func:`ai_crucible.scoring.oracle.grade`) then judges.

The grading host is the ONE side allowed to read ``oracle/`` (§10.4): the Solver's
``load_puzzle`` refuses any oracle artifact, but the runner dynamically loads the
puzzle's sealed ``oracle/check.py`` and calls its ``grade(reported, transcript)``
contract. That keeps penalty detection **trace-driven, not filesystem-scanning**
(Wave-0 lock): the runner reconstructs ``transcript['read_paths']`` from the
kernel-authoritative tool-call trace (``attempt.events`` with ``kind=="tool"``),
so a Solver cannot forge what it touched (§10.2, Vivaria).

Two responsibilities live HERE, not in ``check.py``:

1. **Transcript construction.** Every ``read_file``/``write_file`` records a
   ``path``; every ``exec`` records a ``command`` string. The runner pulls the
   file-path the tool touched: ``args["path"]`` for the file tools, and for an
   ``exec`` of a known file-reading command (``grep``/``cat``/``head``/``tail``/
   ``less``/``open``/``sed``/``awk``) it tokenizes the command and keeps the
   file-path-looking tokens. ``check.py`` grades from that normalized
   ``read_paths`` set; the bait-touch and grounded-read facts fall out of it.

2. **TCRR — the redundant-tool-call ratio (§8.4).** The seed *declares* a
   ``redundant_tool_calls`` penalty but its ``check.py`` deliberately does NOT
   compute it — looping detection is a kernel-trace concern, so the runner owns
   it. TCRR is the fraction of tool calls that are either an exact-duplicate
   ``(tool, args)`` within a 3-turn window OR the >2nd call to the same function
   on the same path. When ``TCRR > redundancy_threshold`` the runner adds
   ``"redundant_tool_calls"`` to the triggered set (deduplicated against any name
   ``check.py`` itself returns).

Standards compliance (the six — workflow-standards.md):

- **PIN_PER_STEP — 2:** the runner is a pure function of its pinned inputs — the
  ``puzzle_root`` (so the loaded ``check.py`` is fixed) + ``redundancy_threshold``
  bound at factory time, and the ``(attempt, meta)`` it grades. No hidden state,
  no clock, no network: the same attempt + the same oracle always grade
  identically, so a graded outcome is byte-for-byte replayable. (Pinning the
  *model* that produced the attempt is the provider's job, upstream.)
- **ANDON_AUTHORITY — 3:** a missing/malformed ``oracle/check.py`` raises a
  structured :class:`OracleRunnerError` (the ``[CODE] msg (hint:)`` house shape)
  rather than letting an ``AttributeError``/``ImportError`` escape as a raw stack
  — the grading host halts loud on a defective oracle instead of silently
  emitting a corrupt outcome. Proven in ``tests/test_oracle_runner.py`` (missing
  ``check.py`` + a ``check.py`` with no ``grade``).
- **NAMED_COMPENSATORS — n/a (skip):** the runner makes no irreversible action —
  it reads the trace and the oracle, performs no FS writes, no network, no
  external call. The dynamically-imported oracle module is registered under a
  unique throwaway name and not relied on after the call; nothing to undo.
- **DECOMPOSE_BY_SECRETS — 3:** this module IS the secret/non-secret seam — it is
  the one place that names ``oracle/check.py`` (the grading edge) while the
  Solver-visible :class:`~ai_crucible.puzzle.LoadedPuzzle` cannot. It loads the
  oracle from the grading-side ``puzzle_root`` and hands the kernel only the
  resulting verdict; the oracle artifact never travels back toward the Solver.
- **UNCERTAINTY_GATED_HUMANS — n/a:** the runner grades unattended; human
  checkpoints live in the catalog/graduation layer, not in a single grade.
- **EXTERNAL_VERIFIER — 3:** this is the external verifier — grading is wholly
  out-of-band from the model under test, reading a sealed oracle the generator
  never saw. The runner trusts the kernel trace (which the model cannot forge),
  not the model's self-report; novelty is left ``False`` here so the cross-family
  panel — never this runner — is the novelty authority (§8.7).
"""

from __future__ import annotations

import importlib.util
import shlex
import sys
import uuid
from collections.abc import Awaitable, Callable
from pathlib import Path

from ai_crucible.scoring.oracle import OracleOutcome
from ai_crucible.types import AttemptState, PuzzleMeta

__all__ = ["OracleRunnerError", "make_oracle_runner"]

#: The penalty name TCRR contributes (the seed declares it; ``check.py`` does not
#: compute it — the runner does, from the kernel trace, §8.4).
REDUNDANT_PENALTY = "redundant_tool_calls"

#: Window (in tool-call turns) within which an exact-duplicate ``(tool, args)`` is
#: counted as redundant (§8.4 WebArena loop signal).
_DUP_WINDOW = 3

#: ``exec`` commands whose positional file-path arguments are a *read* of those
#: paths (so an ``exec(grep ... config/limits.py)`` grounds exactly like a
#: ``read_file``). Tokenized from ``args["command"]`` (or an ``args`` argv list).
_READING_COMMANDS = frozenset({"grep", "cat", "head", "tail", "less", "open", "sed", "awk"})


class OracleRunnerError(Exception):
    """Raised when the puzzle's grading oracle is missing or malformed.

    Carries a stable, structured message (Ship-Gate-B ``[CODE] message (hint:)``
    shape, mirroring :class:`ai_crucible.puzzle.PuzzleLoadError`) so the kernel can
    surface a redacted, actionable error instead of a raw ``ImportError`` /
    ``AttributeError`` stack from the dynamically-loaded ``oracle/check.py``.
    """


def _fail(code: str, message: str, hint: str) -> OracleRunnerError:
    """Build a structured :class:`OracleRunnerError` (code/message/hint)."""
    return OracleRunnerError(f"[{code}] {message} (hint: {hint})")


def _looks_like_path(token: str) -> bool:
    """Heuristic: is ``token`` a file-path argument rather than a flag/pattern?

    A path token is non-empty, is not an option flag (``-n``, ``--recursive``),
    and looks path-shaped — it contains a ``/`` or a ``.`` (an extension or a
    relative segment). This keeps the grep *pattern* (``UPLOAD_MAX_ATTEMPTS``) out
    of ``read_paths`` while keeping the file operand (``config/limits.py``) in. It
    is deliberately lenient: over-including a non-path token only ever adds an
    unmatched entry to ``read_paths``; ``check.py`` keys on suffix/substring
    matches, so a spurious token cannot forge a grounded read or a bait touch.
    """
    if not token or token.startswith("-"):
        return False
    return "/" in token or "." in token


def _exec_read_paths(args: dict) -> list[str]:
    """File-path operands touched by one ``exec`` tool call.

    ``args["command"]`` is the canonical string form (per the solver-loop line
    protocol, Leaf C); an argv-list ``args["command"]``/``args["argv"]`` is also
    tolerated. Only commands in :data:`_READING_COMMANDS` contribute paths — an
    arbitrary ``exec`` (e.g. ``python -c ...``) is a tool call for budget/TCRR
    accounting but does not, by itself, count as *reading* a source file.
    """
    raw = args.get("command", args.get("argv", ""))
    if isinstance(raw, list):
        tokens = [str(t) for t in raw]
    else:
        try:
            tokens = shlex.split(str(raw))
        except ValueError:
            # An unbalanced-quote command still counts as a tool call elsewhere;
            # for path extraction we fall back to a whitespace split.
            tokens = str(raw).split()
    if not tokens:
        return []
    if tokens[0] not in _READING_COMMANDS:
        return []
    return [t for t in tokens[1:] if _looks_like_path(t)]


def _tool_calls(attempt: AttemptState) -> list[tuple[str, dict]]:
    """The ordered ``(tool, args)`` tool calls from the kernel trace.

    Only ``kind=="tool"`` events count (model/score/info/error events are skipped),
    so ``tool_calls_used`` and TCRR are computed off the authoritative tool trace
    the kernel recorded — not Solver self-report (§10.2).
    """
    calls: list[tuple[str, dict]] = []
    for event in attempt.events:
        if event.kind != "tool":
            continue
        payload = event.payload
        tool = str(payload.get("tool", ""))
        args = payload.get("args", {})
        calls.append((tool, dict(args) if isinstance(args, dict) else {}))
    return calls


def _read_paths(calls: list[tuple[str, dict]]) -> list[str]:
    """Every source path the tool calls touched (read_file/write_file path +
    file-path operands of a reading ``exec``)."""
    paths: list[str] = []
    for tool, args in calls:
        if tool in ("read_file", "write_file"):
            path = args.get("path")
            if path:
                paths.append(str(path))
        elif tool == "exec":
            paths.extend(_exec_read_paths(args))
    return paths


def _arg_path(tool: str, args: dict) -> str | None:
    """The single path a call is 'about', for the >2-calls-same-fn+path TCRR rule.

    File tools key on ``path``; an ``exec`` keys on its first file-path operand (so
    re-greping the same file thrice is caught). Calls with no recoverable path do
    not participate in the same-path rule (they may still trip the exact-dup rule).
    """
    if tool in ("read_file", "write_file"):
        path = args.get("path")
        return str(path) if path else None
    if tool == "exec":
        operands = _exec_read_paths(args)
        return operands[0] if operands else None
    return None


def _compute_tcrr(calls: list[tuple[str, dict]]) -> float:
    """The redundant-tool-call ratio (§8.4).

    A call is redundant if EITHER:

    * it is an exact-duplicate ``(tool, args)`` of a call within the prior
      :data:`_DUP_WINDOW` turns (a tight loop), OR
    * it is the >2nd call to the same function on the same path (re-fetching the
      same file repeatedly — the looping-on-diagnosis signal).

    Returns the fraction of redundant calls over total calls (0.0 when no calls,
    so an empty trace can never trip the threshold).
    """
    n = len(calls)
    if n == 0:
        return 0.0

    redundant = [False] * n
    fingerprints = [(tool, _stable_args(args)) for tool, args in calls]
    fn_path_counts: dict[tuple[str, str], int] = {}

    for i, (tool, args) in enumerate(calls):
        # Rule 1: exact-duplicate (tool, args) within the trailing window.
        window_start = max(0, i - _DUP_WINDOW)
        if fingerprints[i] in fingerprints[window_start:i]:
            redundant[i] = True

        # Rule 2: >2 calls to the same function on the same path.
        path = _arg_path(tool, args)
        if path is not None:
            key = (tool, path)
            fn_path_counts[key] = fn_path_counts.get(key, 0) + 1
            if fn_path_counts[key] > 2:
                redundant[i] = True

    return sum(redundant) / n


def _stable_args(args: dict) -> tuple[tuple[str, str], ...]:
    """A hashable, order-stable fingerprint of a tool call's args for exact-dup
    comparison (dicts are unhashable; values are stringified for a total order)."""
    return tuple(sorted((str(k), str(v)) for k, v in args.items()))


def _load_grade(puzzle_root: Path) -> Callable[..., object]:
    """Dynamically load ``puzzle_root/oracle/check.py`` and return its ``grade``.

    Uses :func:`importlib.util.spec_from_file_location` with a unique module name
    (so repeated loads of sibling puzzles never collide in ``sys.modules``). A
    missing file, an import-time failure, or a missing/non-callable ``grade`` all
    raise a structured :class:`OracleRunnerError` (ANDON: a defective oracle halts
    loud, it never silently grades).
    """
    check_path = puzzle_root / "oracle" / "check.py"
    if not check_path.is_file():
        raise _fail(
            "ORACLE_CHECK_MISSING",
            f"no grading oracle at {check_path}",
            "every puzzle must ship oracle/check.py exposing grade(reported, transcript); "
            "it lives grading-side only (§10.4)",
        )

    module_name = f"_ai_crucible_oracle_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(module_name, check_path)
    if spec is None or spec.loader is None:
        raise _fail(
            "ORACLE_CHECK_UNLOADABLE",
            f"could not build an import spec for {check_path}",
            "ensure oracle/check.py is a readable, importable Python module",
        )
    module = importlib.util.module_from_spec(spec)
    # Register BEFORE exec so the oracle's own ``@dataclass``-decorated GradeResult
    # can resolve its ``__module__`` during class creation (dataclass machinery
    # looks the defining module up in ``sys.modules``); otherwise exec_module raises
    # a confusing ``AttributeError: 'NoneType' object has no attribute '__dict__'``.
    # The unique name (uuid) means sibling-puzzle oracles never collide. Deregister
    # in a finally so a throwaway grading module never lingers in ``sys.modules``.
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as exc:  # noqa: BLE001 — surface as a structured oracle error.
        raise _fail(
            "ORACLE_CHECK_IMPORT_FAILED",
            f"oracle/check.py raised at import: {type(exc).__name__}: {exc}",
            "fix the grading oracle so it imports cleanly",
        ) from exc
    finally:
        # Class creation completed during exec_module (or it raised); the returned
        # ``grade`` keeps the module object alive via its globals, so we don't need
        # the throwaway entry parked in sys.modules afterward — clear it either way.
        sys.modules.pop(module_name, None)

    grade = getattr(module, "grade", None)
    if not callable(grade):
        raise _fail(
            "ORACLE_GRADE_MISSING",
            f"oracle/check.py at {check_path} exposes no callable grade()",
            "define grade(reported: str, transcript: dict) -> GradeResult in oracle/check.py",
        )
    return grade


def make_oracle_runner(
    puzzle_root: Path,
    *,
    redundancy_threshold: float = 0.3,
) -> Callable[[AttemptState, PuzzleMeta], Awaitable[OracleOutcome]]:
    """Build the out-of-band grading edge for the puzzle at ``puzzle_root`` (§10.4).

    Returns an async ``(attempt, meta) -> OracleOutcome`` matching the kernel's
    :data:`ai_crucible.kernel.OracleRunner` type alias. The returned runner:

    1. Reconstructs ``transcript`` from ``attempt.events`` (``kind=="tool"``):
       ``read_paths`` (read_file/write_file ``path`` + file operands of a reading
       ``exec``), ``tool_calls_used`` (count of tool events), and ``time_used`` (=
       ``attempt.wall_time``).
    2. Computes TCRR and, when ``TCRR > redundancy_threshold``, marks the
       ``redundant_tool_calls`` penalty (deduplicated against ``check.py``'s own
       fired penalties).
    3. Dynamically loads ``puzzle_root/oracle/check.py`` and calls
       ``grade(reported=attempt.output or "", transcript=transcript)``.
    4. Maps the ``GradeResult`` → :class:`OracleOutcome`: ``solved`` from the
       grade; ``solve_quality = meta.rewards.solve`` when solved else ``0.0``;
       ``no_regression=True`` (the seed has no regression dimension — a future
       puzzle can add one); ``tool_calls_used``/``time_used`` from the kernel
       trace; ``triggered_penalties`` = the grade's fired names + the redundancy
       name when TCRR trips; ``novelty_claimed=False`` (the cross-family panel,
       not this runner, is the novelty authority — §8.7).

    Args:
        puzzle_root: the GRADING-side puzzle directory (the one whose ``oracle/``
            sibling holds ``check.py``). Bound at factory time so the loaded oracle
            is pinned for every grade this runner performs.
        redundancy_threshold: the TCRR ceiling above which ``redundant_tool_calls``
            fires (default ``0.3``, matching the seed's declared trigger).

    Returns:
        An async runner satisfying :data:`ai_crucible.kernel.OracleRunner`.

    Raises:
        OracleRunnerError: at *call* time, if ``oracle/check.py`` is missing,
            fails to import, or exposes no callable ``grade`` (a structured
            ``[CODE] message (hint:)`` error, never a raw stack).
    """
    root = Path(puzzle_root)

    async def _run(attempt: AttemptState, meta: PuzzleMeta) -> OracleOutcome:
        calls = _tool_calls(attempt)
        read_paths = _read_paths(calls)
        tool_calls_used = len(calls)
        time_used = attempt.wall_time

        transcript = {
            "read_paths": read_paths,
            "tool_calls_used": tool_calls_used,
            "time_used": time_used,
        }

        tcrr = _compute_tcrr(calls)
        redundant_fired = tcrr > redundancy_threshold

        grade = _load_grade(root)
        result = grade(reported=attempt.output or "", transcript=transcript)

        # GradeResult is a grading-side dataclass loaded dynamically; read its
        # fields by attribute (it is not importable as a typed symbol here).
        solved = bool(getattr(result, "solved", False))
        fired = list(getattr(result, "penalties_fired", []))

        triggered = list(fired)
        if redundant_fired and REDUNDANT_PENALTY not in triggered:
            triggered.append(REDUNDANT_PENALTY)

        return OracleOutcome(
            solved=solved,
            solve_quality=float(meta.rewards.solve) if solved else 0.0,
            no_regression=True,
            tool_calls_used=tool_calls_used,
            time_used=time_used,
            triggered_penalties=triggered,
            novelty_claimed=False,
        )

    return _run
