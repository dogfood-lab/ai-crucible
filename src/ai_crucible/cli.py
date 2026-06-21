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

import sys
from importlib.metadata import PackageNotFoundError, version


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
  -V, --version  print the installed version and exit
  -h, --help     show this message and exit
"""


def main(argv: list[str] | None = None) -> int:
    """Dispatch ``argv`` to a subcommand. Returns a process exit code."""
    argv = list(sys.argv[1:] if argv is None else argv)

    if not argv or argv[0] in ("-h", "--help"):
        sys.stdout.write(_usage())
        return 0
    if argv[0] in ("-V", "--version"):
        sys.stdout.write(f"ai-crucible {_version()}\n")
        return 0

    command, rest = argv[0], argv[1:]
    if command == "characterize":
        # Lazy import: keep `--version`/`--help` instant and free of the heavy
        # scientific/inspect-ai stack the characterization run pulls in.
        from ai_crucible.characterize.run import main as characterize_main

        return characterize_main(rest)

    sys.stderr.write(f"ai-crucible: unknown command {command!r}\n\n{_usage()}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
