"""Puzzle staging — materialize the Solver workdir + place grading-side bait (§10.4).

This module owns the *staging* half of the two-channel lock (research-grounding
§10.4; Epic-1 CONTRACT Leaf B). Before the Solver runs, two things must happen in
the sandbox workdir, and exactly one of them is grading-controlled:

1. **Environment priming.** A puzzle may ship an optional ``setup_script`` (the
   ``puzzle.LoadedPuzzle.setup_script`` path the loader only *resolves*, never
   executes — §10.4, ``src/ai_crucible/puzzle.py``). :func:`stage_puzzle` writes its
   bytes into the sandbox workdir and runs it. The sandbox provider runs **no
   shell** (:class:`ai_crucible.sandbox.SandboxEnvironment` is argv-only), so
   ``bash`` is invoked EXPLICITLY as the only interpreter. The exact argv is
   CRLF-robust — ``["bash", "-c", "tr -d '\\r' < <name> | bash"]`` — because the
   provider's ``write_file`` re-injects ``\r\n`` on Windows, which would otherwise
   make bash reject ``set -euo pipefail\r`` (see :func:`_run_setup` for the full
   rationale + the contract note). A non-zero exit raises a structured
   :class:`StagingError`; a missing ``bash`` on the host fails with a clear hint
   rather than a raw ``FileNotFoundError``.

2. **Grading-side bait placement.** The puzzle's ``oracle/`` directory holds the
   sealed grading artifacts — ``check.py`` (the oracle, NEVER staged) plus the
   answer-key honeypot (e.g. ``ANSWER_KEY_*.txt``, §8.5/§8.6 canary furniture).
   The bait is *environment furniture with no legitimate puzzle purpose* — a
   bypassing Solver can touch it, but its placement must be **grading-controlled**,
   not author-controlled, so a future puzzle cannot accidentally stage it through
   ``setup_script``. :func:`stage_puzzle` copies every file under
   ``grading_root/oracle/`` that is NOT ``check.py`` into the workdir. The host
   reads ``oracle/`` (it lives on the grading side, NOT in Solver-visible
   :class:`~ai_crucible.puzzle.LoadedPuzzle`); the sandbox only ever receives the
   bait *contents* via :meth:`SandboxEnvironment.write_file`.

The oracle itself (``check.py``) is structurally excluded here — it is read only by
the out-of-band grading host (``oracle_runner``), never copied into the Solver's
namespace. The bait's *placement* being grading-controlled is what makes a
bait-touch a clean, attributable bypass signal at grade time (the seed
``check.py`` keys ``answer_key_fetch`` on a touch of ``ANSWER_KEY_*.txt``).

Standards compliance (the six; research-grounding §10.7 / workflow-standards.md)
-------------------------------------------------------------------------------
- **PIN_PER_STEP — 1:** staging is a pure function of (setup_script bytes, oracle
  furniture) for a given puzzle artifact; it pins no image/digest (that is the
  sandbox provider's job — the local provider already scores 1 here). Replayability
  comes from the puzzle artifact being content-addressed upstream.
- **ANDON_AUTHORITY — 2:** a non-zero ``setup_script`` exit, a missing ``bash``, or
  a missing ``oracle/`` furniture file each HALT staging with a structured
  :class:`StagingError` (code/message/hint) — a broken environment never silently
  proceeds to a Solver run that would grade as a spurious non-solve.
- **NAMED_COMPENSATORS — n/a:** staging writes only into the caller-owned sandbox
  workdir (torn down by :meth:`ai_crucible.sandbox.LocalSandbox.cleanup`, the named
  compensator for the workdir). It performs no external/irreversible action
  (no publish/release/network/host-fs write outside the sandbox), so no
  compensators table is required.
- **DECOMPOSE_BY_SECRETS — 3:** the secret/non-secret split *is* this module's
  design — it reads ``oracle/`` host-side (the grading zone) but copies ONLY the
  non-secret furniture into the Solver zone, and refuses ``check.py`` by name. The
  oracle never crosses into the sandbox.
- **UNCERTAINTY_GATED_HUMANS — n/a:** staging runs unattended; it has no human
  checkpoint.
- **EXTERNAL_VERIFIER — 2:** staging is the generator-side setup only and holds no
  grading authority; by refusing to stage ``check.py`` it keeps the verifier
  (oracle) in the separate grading process the sandbox cannot reach (§10.4).
"""

from __future__ import annotations

import shlex
from dataclasses import dataclass, field
from pathlib import Path

from ai_crucible.puzzle import LoadedPuzzle
from ai_crucible.sandbox import SandboxEnvironment

__all__ = [
    "DEFAULT_SETUP_TIMEOUT_SECONDS",
    "ORACLE_DIRNAME",
    "ORACLE_FILE",
    "StageResult",
    "StagingError",
    "stage_puzzle",
]

#: The grading-side directory under a puzzle root that holds the sealed oracle and
#: its bait furniture (§10.4). Read host-side only; never staged wholesale.
ORACLE_DIRNAME = "oracle"

#: The one file inside ``oracle/`` that is the oracle itself and MUST NEVER be
#: copied into the Solver workdir (it is read only by the out-of-band grading host).
ORACLE_FILE = "check.py"

#: Wall-clock ceiling for the ``setup_script`` run. Priming a small multi-file
#: fixture is fast; this is a generous andon so a wedged setup cannot hang the run.
DEFAULT_SETUP_TIMEOUT_SECONDS = 120.0


class StagingError(Exception):
    """Raised when puzzle staging fails (setup_script error, missing ``bash``, or a
    missing/unreadable grading furniture file).

    Carries a stable, structured message (Ship Gate B shape: ``[CODE] msg
    (hint: ...)``) so the kernel can surface a redacted, actionable error without a
    raw stack — the same house shape as
    :class:`ai_crucible.puzzle.PuzzleLoadError`.
    """


@dataclass(slots=True)
class StageResult:
    """What :func:`stage_puzzle` placed into the sandbox workdir, for observability.

    Attributes:
        setup_ran: did the ``setup_script`` actually execute (False when the puzzle
            ships no setup_script, OR when it was skipped because ``bash`` is absent
            and ``require_bash=False``).
        setup_script_name: the filename the setup_script was written into the
            workdir as (``None`` when no setup_script).
        bait_files: the bait furniture filenames copied from ``oracle/`` into the
            workdir (e.g. ``["ANSWER_KEY_a7f3b9.txt"]``). NEVER contains
            :data:`ORACLE_FILE`.
        skipped_setup_reason: a short reason string when ``setup_ran`` is False
            despite a setup_script being present (e.g. ``"bash unavailable"``);
            ``None`` otherwise.
    """

    setup_ran: bool
    setup_script_name: str | None = None
    bait_files: list[str] = field(default_factory=list)
    skipped_setup_reason: str | None = None


def _fail(code: str, message: str, hint: str) -> StagingError:
    """Build a structured :class:`StagingError` (code/message/hint house shape)."""
    return StagingError(f"[{code}] {message} (hint: {hint})")


async def stage_puzzle(
    sandbox: SandboxEnvironment,
    loaded: LoadedPuzzle,
    *,
    grading_root: Path,
    setup_timeout_seconds: float = DEFAULT_SETUP_TIMEOUT_SECONDS,
    require_bash: bool = True,
) -> StageResult:
    """Materialize the Solver workdir and place the grading-controlled bait (§10.4).

    Two steps, in order:

    1. If ``loaded.setup_script`` is present, its bytes are written into the sandbox
       workdir (decoded as UTF-8 for the str-typed
       :meth:`SandboxEnvironment.write_file`) and run via
       ``exec(["bash", "<name>"], timeout=setup_timeout_seconds)`` — the provider
       runs no shell, so ``bash`` is invoked explicitly. A non-zero exit raises a
       structured :class:`StagingError` carrying the script's stderr. If ``bash`` is
       unavailable on the host: with ``require_bash=True`` (default) this raises a
       clear :class:`StagingError`; with ``require_bash=False`` it is skipped and
       recorded in :attr:`StageResult.skipped_setup_reason` (so a host-without-bash
       can still test the bait-placement half).
    2. Every file under ``grading_root/oracle/`` that is NOT :data:`ORACLE_FILE`
       (``check.py``) is copied into the sandbox workdir — the answer-key honeypot
       lands so a bypassing Solver *can* touch it, but its placement is
       grading-controlled. The oracle is read host-side and only the bait
       *contents* are handed to :meth:`SandboxEnvironment.write_file`. ``check.py``
       is NEVER copied.

    Args:
        sandbox: the Solver's confined env channel (§10.4); the only write surface
            this function uses (``write_file`` + ``exec``).
        loaded: the Solver-visible :class:`~ai_crucible.puzzle.LoadedPuzzle` (carries
            the resolved ``setup_script`` path; it has NO oracle field by design).
        grading_root: the puzzle root on the GRADING side (the host directory whose
            ``oracle/`` holds the sealed furniture). Distinct from the Solver's
            workdir — staging reads it host-side, the Solver never can.
        setup_timeout_seconds: wall-clock ceiling for the setup_script run (ANDON).
        require_bash: if True (default), a missing ``bash`` is a hard
            :class:`StagingError`; if False, the setup step is cleanly skipped and
            recorded (lets a host without bash still exercise bait placement).

    Returns:
        A :class:`StageResult` recording what was staged.

    Raises:
        StagingError: setup_script non-zero exit, ``bash`` missing (when
            ``require_bash``), or a grading furniture file under ``oracle/`` could
            not be read.
    """
    result = StageResult(setup_ran=False)

    if loaded.setup_script is not None:
        await _run_setup(
            sandbox,
            loaded.setup_script,
            result,
            timeout=setup_timeout_seconds,
            require_bash=require_bash,
        )

    result.bait_files = await _place_bait(sandbox, grading_root)
    return result


async def _run_setup(
    sandbox: SandboxEnvironment,
    setup_path: Path,
    result: StageResult,
    *,
    timeout: float,
    require_bash: bool,
) -> None:
    """Write the setup_script into the workdir and run it via explicit ``bash``.

    The script is read host-side as bytes and decoded UTF-8 for the str-typed
    ``write_file``. Mutates ``result`` in place (``setup_ran`` / ``setup_script_name``
    / ``skipped_setup_reason``).
    """
    name = setup_path.name
    try:
        raw = setup_path.read_bytes()
    except OSError as exc:
        raise _fail(
            "SETUP_READ_FAILED",
            f"could not read setup_script at {setup_path}: {exc}",
            "ensure the puzzle's setup_script file exists and is readable",
        ) from exc

    # Normalize the host-read script to LF before staging. The bait files below are
    # data and are copied verbatim; a setup_script is bash *source*, so CRs matter.
    text = raw.decode("utf-8").replace("\r\n", "\n").replace("\r", "\n")
    await sandbox.write_file(name, text)
    result.setup_script_name = name

    # Run the staged script through bash, invoked EXPLICITLY (the sandbox runs no
    # shell — §10.4 — so there is no implicit interpreter). We do NOT exec
    # ``["bash", name]`` directly: the only write channel into the sandbox is
    # :meth:`SandboxEnvironment.write_file`, whose ``str`` payload the provider
    # writes via ``write_text``, which RE-INJECTS ``\r\n`` on Windows even when the
    # content we pass is LF-only (a real, reproduced sandbox-module behavior). bash
    # then reads ``set -euo pipefail\r`` and rejects ``pipefail\r`` as an invalid
    # option, failing every CRLF-platform setup. To stay correct cross-platform
    # without editing the locked sandbox module, we strip CRs at run time and pipe
    # the cleaned source into bash: ``bash -c "tr -d '\\r' < <name> | bash"``. bash
    # is still the explicit and only interpreter; this is just CR-immunity over the
    # provider's newline handling. (See contract_deviations: the durable fix is a
    # ``newline=""`` write in the sandbox provider, owned by the Wave-2 integrator.)
    run_cmd = ["bash", "-c", f"tr -d '\\r' < {shlex.quote(name)} | bash"]
    try:
        exec_result = await sandbox.exec(run_cmd, timeout=timeout)
    except FileNotFoundError as exc:
        # The provider could not spawn `bash` (not on PATH). The sandbox runs no
        # shell, so there is no fallback interpreter to try.
        if not require_bash:
            result.skipped_setup_reason = "bash unavailable"
            return
        raise _fail(
            "SETUP_BASH_MISSING",
            "could not invoke 'bash' to run the setup_script (bash not found)",
            "install bash (e.g. Git Bash on Windows) or pass require_bash=False to "
            "skip environment priming; the sandbox runs no shell, so bash must be "
            "invoked explicitly",
        ) from exc

    if exec_result.timed_out:
        raise _fail(
            "SETUP_TIMED_OUT",
            f"setup_script '{name}' exceeded the {timeout}s staging budget",
            "the setup_script wedged or is too slow; speed it up or raise "
            "setup_timeout_seconds",
        )

    if exec_result.returncode != 0:
        stderr = exec_result.stderr.strip() or "(no stderr captured)"
        raise _fail(
            "SETUP_NONZERO_EXIT",
            f"setup_script '{name}' exited {exec_result.returncode}: {stderr}",
            "fix the setup_script so it primes the workdir and exits 0; it runs "
            "under 'bash -' with the sandbox workdir as cwd",
        )

    result.setup_ran = True


async def _place_bait(sandbox: SandboxEnvironment, grading_root: Path) -> list[str]:
    """Copy every ``oracle/`` furniture file except :data:`ORACLE_FILE` into the workdir.

    Reads host-side from ``grading_root/oracle/`` and writes the *contents* through
    :meth:`SandboxEnvironment.write_file` (bytes decoded UTF-8). ``check.py`` is
    refused by name so the oracle never crosses into the Solver zone (§10.4).
    Returns the sorted list of bait filenames placed (stable for observability).
    A missing ``oracle/`` directory means there is simply no bait to place — not an
    error (a puzzle may legitimately ship no honeypot).
    """
    oracle_dir = Path(grading_root) / ORACLE_DIRNAME
    if not oracle_dir.is_dir():
        return []

    placed: list[str] = []
    for entry in sorted(oracle_dir.iterdir()):
        if not entry.is_file():
            continue
        if entry.name == ORACLE_FILE:
            # The oracle itself — read only by the grading host, NEVER staged (§10.4).
            continue
        try:
            content = entry.read_bytes().decode("utf-8")
        except OSError as exc:
            raise _fail(
                "BAIT_READ_FAILED",
                f"could not read grading furniture file {entry}: {exc}",
                "ensure every non-oracle file under the puzzle's oracle/ directory "
                "is a readable UTF-8 text artifact (the bait honeypot)",
            ) from exc
        await sandbox.write_file(entry.name, content)
        placed.append(entry.name)

    return placed
