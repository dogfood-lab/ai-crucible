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

options:
  --debug, -v    on error, print the full Python traceback (developer mode) instead of the
                 one-line structured error
  -V, --version  print the installed version and exit
  -h, --help     show this message and exit

exit codes:
  0  success (also: --help, --version)
  2  usage error — unknown command
  1  (characterize) ran but collected zero judgments — every model failed/unreachable;
     stderr carries a structured [CHARACTERIZE_NO_JUDGMENTS] {{code,message,hint}} JSON. CI
     gates should treat 1 (degraded/empty result) distinctly from 2 (bad invocation).
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

    sys.stderr.write(f"ai-crucible: unknown command {command!r}\n\n{_usage()}")
    return 2


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
