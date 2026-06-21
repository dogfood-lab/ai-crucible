"""ai-crucible CLI — the unified entry point.

Backs both ``python -m ai_crucible`` and the ``ai-crucible`` console script /
PyInstaller binary that the npm launcher (``@dogfood-lab/ai-crucible``) distributes.
It is a thin dispatcher: the real work lives in the subcommand modules (today, the
judge-admission characterization in :mod:`ai_crucible.characterize.run`), and their
own argparse handles flags — this layer only routes ``argv[0]`` and forwards the rest
verbatim, so ``ai-crucible characterize --k 3`` is exactly ``python -m
ai_crucible.characterize.run --k 3``.
"""

from __future__ import annotations

import contextlib
import re
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # Type-only imports — kept out of the runtime path so ``--help``/``--version`` stay
    # instant and free of the heavy kernel/model stack (the lazy imports inside the
    # ``run`` handler do the real loading, under main()'s structured-error guard).
    from ai_crucible.cycle import DiagnosticModel
    from ai_crucible.scoring.judge_panel import JudgeFn, JudgePanel

# A structured Ship-Gate-B error string: ``[CODE] message (hint: ...)`` — the shape every
# loader/adapter in the repo emits via its ``_fail(code, message, hint)`` helper. We match on
# the rendered string (not the exception class) so the dispatcher stays free of the heavy
# scientific stack those classes live in, and so ANY future module using the same house shape
# is rendered cleanly without a code change here.
_STRUCTURED_ERROR_RE = re.compile(r"^\[[A-Z0-9_]+\] .+ \(hint: .+\)$", re.DOTALL)


def _ensure_utf8_streams() -> None:
    """Make stdout/stderr UTF-8 so the operator-facing banner + run caveat (which carry
    ``ω`` / ``κ`` and other non-ASCII) don't crash on a legacy console.

    On a stock Windows console (cp1252) ``sys.stdout.write`` of a non-ASCII char raises
    ``UnicodeEncodeError`` — so ``ai-crucible --help`` (the banner has ``ω``) would crash
    before printing anything. Best-effort + guarded: a stream without ``reconfigure`` (a
    redirected pipe / a non-``TextIOWrapper``) or one already UTF-8 is left untouched, and
    a reconfigure failure never takes down the CLI. Called once at the top of :func:`main`.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        enc = (getattr(stream, "encoding", "") or "").lower().replace("-", "")
        if reconfigure is not None and enc != "utf8":
            # Best-effort: a reconfigure failure (locked stream, exotic wrapper) must never
            # take down the CLI — the worst case is the legacy-console encoding we started with.
            with contextlib.suppress(Exception):
                reconfigure(encoding="utf-8")


def _version() -> str:
    try:
        return version("ai-crucible")
    except PackageNotFoundError:  # running from a source tree without an install
        return "0.0.0+local"


def _usage() -> str:
    """The usage banner. The version is read from package metadata — the SAME single source
    as ``--version`` (models-cli-004), so the banner never drifts from the installed version
    on the next bump (the old banner hardcoded ``v0.2.0`` while ``--version`` read metadata).
    """
    return f"""\
ai-crucible — a diagnostic measurement instrument (research preview, v{_version()}).

Seats a cross-family panel of local LLM judges under a sealed measurement boundary and
scores attempts against a hidden oracle. NOTE: the judge panel's alt-test ω is still a
circular model-jury bootstrap until a human-labeling round runs; seats are provisional.

usage: ai-crucible <command> [options]

commands:
  characterize   run the judge-admission characterization on the local model panel
                 (needs Ollama + the local panel; forwards all flags — see
                 `ai-crucible characterize --help`)
  run            run one diagnostic cycle: a Solver attempts a puzzle in the sandbox,
                 graded out-of-band against the sealed oracle, emitting the pass^k /
                 Wilson rollup. usage:
                   ai-crucible run <puzzle-dir> --model <id>[@family]
                       [--k N] [--arm neutral|self_referential|social_standings]
                       [--panel <path>]
                 The model adapter is chosen by the optional @family tag: no tag or
                 @claude → Claude (Anthropic API; needs ANTHROPIC_API_KEY); any other
                 @family → an Ollama local model of that family (needs Ollama).
                 Human rollup chrome → STDERR; machine JSON summary → STDOUT.

options:
  --debug, -v    on error, print the full Python traceback (developer mode) instead of the
                 one-line structured error
  -V, --version  print the installed version and exit
  -h, --help     show this message and exit

exit codes:
  0  success (also: --help, --version; `run` completed a cycle and emitted a rollup)
  2  usage error — unknown command, or `run` invoked without a puzzle-dir / --model
  1  (characterize) ran but collected zero judgments — every model failed/unreachable;
     stderr carries a structured [CHARACTERIZE_NO_JUDGMENTS] {{code,message,hint}} JSON. CI
     gates should treat 1 (degraded/empty result) distinctly from 2 (bad invocation).
  1  (run) the puzzle failed to load or stage (a structured [CODE] msg (hint:) error on
     stderr) — distinct from 2 (a bad invocation that never reached the cycle).
"""


def _dispatch(command: str, rest: list[str]) -> int:
    """Route ``command`` to its subcommand. Raises the subcommand's exceptions unwrapped —
    :func:`main` owns the operator-vs-developer error contract around this call."""
    if command == "characterize":
        # Lazy import: keep `--version`/`--help` instant and free of the heavy
        # scientific/inspect-ai stack the characterization run pulls in. The import is
        # INSIDE main()'s guard so a packaging fault here (e.g. a missing scientific dep)
        # is rendered as a clean structured error, not a raw ModuleNotFoundError traceback.
        from ai_crucible.characterize.run import main as characterize_main

        return characterize_main(rest)

    if command == "run":
        return _run_diagnostic_command(rest)

    sys.stderr.write(f"ai-crucible: unknown command {command!r}\n\n{_usage()}")
    return 2


def _build_model(model_spec: str) -> DiagnosticModel:
    """Construct a model adapter from a ``<id>[@family]`` spec (the ``--model`` value).

    The optional ``@family`` tag (split on the LAST ``@``, since a model id may itself
    contain ``@``-free ``:`` tags like ``mistral-small:24b``) chooses the adapter and
    feeds the panel's same-family exclusion (§10.2):

    * no ``@family`` tag, or ``@claude`` → :class:`~ai_crucible.models.claude_adapter.ClaudeModel`
      (the default Designer/Solver, Anthropic API — reads ``ANTHROPIC_API_KEY`` at call time);
    * any other ``@family`` → :class:`~ai_crucible.models.ollama_adapter.OllamaModel` of that
      family (a local model served by Ollama).

    Kept a module-level seam (not inlined) so the ``run`` tests inject a CANNED model via
    ``monkeypatch.setattr(cli, "_build_model", ...)`` and never construct a real adapter or
    hit a network/API. Imports the adapters lazily so ``--help``/``--version`` stay free of
    the model stack and a packaging fault is rendered as a structured error by ``main()``.
    """
    if "@" in model_spec:
        model_id, _, family = model_spec.rpartition("@")
    else:
        model_id, family = model_spec, ""

    fam = family.strip().lower()
    if fam and fam != "claude":
        from ai_crucible.models.ollama_adapter import OllamaModel

        return OllamaModel(model_id=model_id, family=fam)

    from ai_crucible.models.claude_adapter import ClaudeModel

    return ClaudeModel(model_id)


def _run_diagnostic_command(rest: list[str]) -> int:
    """Handle ``ai-crucible run <puzzle-dir> --model <id>[@family] [--k N] [--arm ...]
    [--panel <path>]``.

    Parses the args, builds the model adapter (:func:`_build_model`), runs one
    :func:`ai_crucible.cycle.run_diagnostic` cycle via ``asyncio.run``, writes the human
    rollup chrome to STDERR and the machine JSON summary to STDOUT, and returns 0 on a
    completed run. A bad invocation (missing puzzle-dir / ``--model``, unknown ``--arm``)
    returns 2 with a usage message; a load/stage failure raises a structured
    ``[CODE] msg (hint:)`` error that ``main()``'s top-level handler renders as one line
    (exit 1). Heavy imports (the kernel/cycle stack) are lazy so they live inside ``main``'s
    guard and a packaging fault renders cleanly.
    """
    import argparse
    import asyncio

    from ai_crucible.types import FramingArm

    parser = argparse.ArgumentParser(
        prog="ai-crucible run",
        description="Run one diagnostic cycle against a puzzle and emit the pass^k rollup.",
        add_help=True,
    )
    parser.add_argument("puzzle_dir", help="the puzzle directory (the one containing meta.json)")
    parser.add_argument(
        "--model",
        required=True,
        metavar="<id>[@family]",
        help="model spec; @family chooses the adapter (none/@claude → Claude, else Ollama)",
    )
    parser.add_argument("--k", type=int, default=1, help="sibling attempts for pass^k (>=1)")
    parser.add_argument(
        "--arm",
        choices=[a.value for a in FramingArm],
        default=FramingArm.SELF_REFERENTIAL.value,
        help="framing arm for the scored context (default: self_referential)",
    )
    parser.add_argument(
        "--panel",
        type=Path,
        default=None,
        help="optional composed seated-panel artifact (panel.json) for cross-family novelty",
    )
    # argparse exits(2) on a parse error and prints usage to stderr — exactly the
    # bad-invocation contract (exit 2, distinct from a load/stage failure's exit 1). It
    # signals that exit by RAISING SystemExit; main()'s top-level handler only catches
    # Exception (not the BaseException SystemExit), so we translate it to a RETURN code here
    # — keeping `run` a plain `main(...) -> int` like every other subcommand (and -h → 0).
    try:
        args = parser.parse_args(rest)
    except SystemExit as exc:
        return int(exc.code) if isinstance(exc.code, int) else 2

    if args.k < 1:
        sys.stderr.write("ai-crucible run: --k must be >= 1\n")
        return 2

    # Lazy: keep the heavy kernel/cycle stack out of the --help/--version path.
    from ai_crucible.cycle import render_rollup, rollup_json, run_diagnostic

    panel = _load_panel(args.panel) if args.panel is not None else None
    model = _build_model(args.model)
    arm = FramingArm(args.arm)

    history = asyncio.run(
        run_diagnostic(Path(args.puzzle_dir), model, args.k, arm=arm, panel=panel)
    )

    # Stage-C honesty: human chrome → STDERR, machine JSON summary → STDOUT.
    sys.stderr.write(render_rollup(history, args.k) + "\n")
    sys.stdout.write(rollup_json(history, args.k) + "\n")
    return 0


def _load_panel(path: Path) -> JudgePanel:
    """Load a composed seated-panel artifact into a :class:`JudgePanel` for ``run --panel``.

    Reads the committed panel artifact (:func:`ai_crucible.characterize.panel_store.load_panel`)
    and seats it via :meth:`JudgePanel.from_seated`, instantiating each seated judge as an
    Ollama judge of its recorded family (the local cross-family panel, §11.4). A missing /
    malformed artifact raises the panel store's structured ``[CODE] msg (hint:)`` error,
    rendered as one line by ``main()`` (exit 1). Lazy imports keep the model stack off the
    no-panel path.
    """
    from ai_crucible.characterize.panel_store import load_panel
    from ai_crucible.models.ollama_adapter import OllamaModel
    from ai_crucible.scoring.judge_panel import JudgePanel

    seated = load_panel(path)

    def judge_for(model_id: str) -> JudgeFn:
        # The seat carries the family; instantiate a local Ollama judge for it. The
        # family is re-bound by from_seated from the seat record, so a placeholder here
        # is fine — the panel reads the seat's family for exclusion.
        return OllamaModel(model_id=model_id, family="").as_judge()

    return JudgePanel.from_seated(seated, judge_for)


def main(argv: list[str] | None = None) -> int:
    """Dispatch ``argv`` to a subcommand. Returns a process exit code.

    Owns the operator-facing error contract (cli-operator-001 / error-hint-sweep-001): an
    exception escaping the dispatch is rendered as a SINGLE structured stderr line, never a raw
    multi-frame traceback. If the exception already carries the repo's ``[CODE] msg (hint: ...)``
    house shape (CalibrationLoadError, OllamaUnreachableError, PuzzleLoadError, …) that one line
    is written verbatim; otherwise it is wrapped in a generic ``[CLI_UNEXPECTED] …`` line that
    points to ``--debug``. The full traceback is opt-IN via ``--debug``/``-v`` (developer mode) —
    mirroring the kernel, where SealedBoundaryViolation/ChromeAccessError propagate unwrapped for
    the same operator-vs-developer reason. KeyboardInterrupt is a clean abort (exit 130).
    """
    _ensure_utf8_streams()  # banner/caveat carry non-ASCII (ω/κ) — survive a cp1252 console.
    argv = list(sys.argv[1:] if argv is None else argv)

    # Top-level --debug/-v is consumed HERE (before the rest is forwarded) so a subcommand's own
    # parser never sees it. It only changes how an error is rendered, not what runs.
    debug = False
    filtered: list[str] = []
    for tok in argv:
        if tok in ("--debug", "-v"):
            debug = True
        else:
            filtered.append(tok)
    argv = filtered

    if not argv or argv[0] in ("-h", "--help"):
        sys.stdout.write(_usage())
        return 0
    if argv[0] in ("-V", "--version"):
        sys.stdout.write(f"ai-crucible {_version()}\n")
        return 0

    command, rest = argv[0], argv[1:]
    try:
        return _dispatch(command, rest)
    except KeyboardInterrupt:
        # Ctrl-C is a deliberate operator abort, not a crash. 128 + SIGINT(2) = 130.
        sys.stderr.write("\nai-crucible: interrupted\n")
        return 130
    except Exception as exc:  # noqa: BLE001 — the operator-facing top-level handler.
        if debug:
            raise  # developer mode: let the interpreter print the full traceback.
        msg = str(exc)
        if _STRUCTURED_ERROR_RE.match(msg):
            # Already in the house shape — emit the one authored line, no stack chrome.
            sys.stderr.write(msg + "\n")
        else:
            # An unexpected/unstructured fault (packaging, import, programming bug): wrap it in
            # the house shape so the operator gets a code + an actionable next step.
            sys.stderr.write(
                f"[CLI_UNEXPECTED] {msg} (hint: re-run with --debug for the full traceback)\n"
            )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
