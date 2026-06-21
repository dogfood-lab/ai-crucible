"""Puzzle artifact loader (``puzzle_loader`` module, research-grounding §10.2).

A puzzle is a *directory* (§1 artifact structure):

- ``meta.json``    — the per-puzzle contract, validated against
  :class:`ai_crucible.types.PuzzleMeta`.
- ``prompt``       — what the Solver sees (plain text).
- ``setup_script`` — optional environment/state priming, run sandboxed by the
  ``sandbox`` module (this loader only *resolves* the path, never executes it).

THE ORACLE IS NEVER LOADED HERE. Per §10.4 the answer-key / oracle / locked
tests live only on the grading side; the Solver-visible state must have zero
path to them. :class:`LoadedPuzzle` therefore has **no oracle field**, and
:func:`load_puzzle` deliberately reads only ``meta.json``, the prompt, and the
optional setup script — it does not read, return, or even glance at any
``oracle`` artifact. Keeping the oracle out of the loaded object makes the
sealed boundary structural rather than aspirational.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from pydantic import ValidationError

from ai_crucible.types import PuzzleMeta

__all__ = [
    "LoadedPuzzle",
    "MAX_META_BYTES",
    "MAX_PROMPT_BYTES",
    "PuzzleLoadError",
    "load_puzzle",
]

# Canonical filenames inside a puzzle directory.
_META_FILE = "meta.json"
_PROMPT_FILE = "prompt"
_SETUP_FILE = "setup_script"

# Size caps at the §1 trust boundary (kernel-runtime-003). A puzzle directory is
# an external/authored artifact — the FIRST place untrusted-author content enters
# the kernel — so the loader stat()s each file and refuses an oversized read
# rather than slurping an arbitrarily large file into memory and OOMing before any
# budget/sandbox limit applies (the sandbox caps subprocess output at 1 MiB via
# DEFAULT_MAX_OUTPUT_BYTES; this is the input-side equivalent). meta.json is a
# small contract (~1 MiB is generous); the prompt is the Solver-visible body and
# may legitimately be larger (a few MiB). Both are overridable per call so a caller
# with an unusual artifact can raise the ceiling deliberately.
MAX_META_BYTES = 1 * 1024 * 1024        # ~1 MiB
MAX_PROMPT_BYTES = 4 * 1024 * 1024      # ~4 MiB

# Filenames that hold (or plausibly hold) the answer artifact. The loader
# refuses to read these and refuses to expose them, so the Solver-visible
# :class:`LoadedPuzzle` can never carry oracle content (§8.5, §10.4).
_ORACLE_NAMES = frozenset({"oracle", "answer", "answer_key", "solution", "gold"})


class PuzzleLoadError(Exception):
    """Raised when a puzzle directory is missing, malformed, or its ``meta.json``
    fails :class:`PuzzleMeta` validation.

    Carries a stable, structured message (Ship Gate B shape: code/message/hint)
    so the kernel can surface a redacted, actionable error without a raw stack.
    """


@dataclass
class LoadedPuzzle:
    """A puzzle loaded into Solver-visible state.

    NOTE: there is intentionally **no** ``oracle`` field. The oracle stays on
    the grading side (§10.4); nothing the Solver can reach holds the answer.
    """

    meta: PuzzleMeta
    prompt: str
    setup_script: Path | None
    root: Path


def _fail(code: str, message: str, hint: str) -> PuzzleLoadError:
    """Build a structured :class:`PuzzleLoadError` (code/message/hint)."""
    return PuzzleLoadError(f"[{code}] {message} (hint: {hint})")


def load_puzzle(
    path: Path,
    *,
    max_meta_bytes: int = MAX_META_BYTES,
    max_prompt_bytes: int = MAX_PROMPT_BYTES,
) -> LoadedPuzzle:
    """Load the puzzle directory at ``path`` into a :class:`LoadedPuzzle`.

    Reads ``meta.json`` (validated via :class:`PuzzleMeta`), the ``prompt`` file,
    and resolves ``setup_script`` if present. Never reads any oracle/answer
    artifact (§10.4). Each file is size-checked (``stat()``) before it is read so
    an oversized authored artifact is refused at the §1 trust boundary instead of
    OOMing the kernel (kernel-runtime-003).

    Args:
        path: the puzzle directory (the one containing ``meta.json``).
        max_meta_bytes: ceiling for ``meta.json`` (default :data:`MAX_META_BYTES`).
        max_prompt_bytes: ceiling for the ``prompt`` file (default
            :data:`MAX_PROMPT_BYTES`).

    Raises:
        PuzzleLoadError: directory missing, required file missing, a required file
            exceeds its size cap (``INPUT_META_TOO_LARGE`` / ``INPUT_PROMPT_TOO_LARGE``),
            ``meta.json`` is not valid JSON, or ``meta.json`` fails
            :class:`PuzzleMeta` validation.
    """
    root = Path(path)
    if not root.is_dir():
        raise _fail(
            "INPUT_PUZZLE_DIR_MISSING",
            f"puzzle path is not a directory: {root}",
            "pass the path to a puzzle directory (the one containing meta.json)",
        )

    meta = _load_meta(root, max_bytes=max_meta_bytes)
    prompt = _load_prompt(root, max_bytes=max_prompt_bytes)
    setup_script = _resolve_setup(root)

    return LoadedPuzzle(meta=meta, prompt=prompt, setup_script=setup_script, root=root)


def _guard_size(path: Path, *, code: str, label: str, max_bytes: int) -> None:
    """Refuse a file larger than ``max_bytes`` BEFORE it is read whole into memory.

    The §1 trust boundary: a puzzle directory is an external/authored artifact, so
    the loader stat()s each required file and rejects an oversized one with a
    structured :class:`PuzzleLoadError` (code/message/hint, size + cap in the hint)
    rather than slurping it via ``read_text`` and OOMing the process. Mirrors the
    sandbox's deliberate output cap on the input side (kernel-runtime-003).
    """
    size = path.stat().st_size
    if size > max_bytes:
        raise _fail(
            code,
            f"{label} is {size} bytes, over the {max_bytes}-byte cap",
            f"the puzzle {label} exceeds the size ceiling ({size} > {max_bytes} bytes); "
            "shrink the file or raise the cap explicitly via load_puzzle(...)",
        )


def _load_meta(root: Path, *, max_bytes: int) -> PuzzleMeta:
    meta_path = root / _META_FILE
    if not meta_path.is_file():
        raise _fail(
            "INPUT_META_MISSING",
            f"missing {_META_FILE} in {root}",
            f"every puzzle directory must contain a {_META_FILE} file",
        )
    _guard_size(meta_path, code="INPUT_META_TOO_LARGE", label=_META_FILE, max_bytes=max_bytes)
    try:
        raw = json.loads(meta_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise _fail(
            "INPUT_META_BAD_JSON",
            f"{_META_FILE} is not valid JSON: {exc}",
            "fix the JSON syntax in meta.json",
        ) from exc

    # Defense-in-depth: refuse to honour an oracle/answer key smuggled into
    # meta.json. The oracle must never travel inside Solver-visible state (§10.4).
    if isinstance(raw, dict):
        leaked = _ORACLE_NAMES.intersection(raw.keys())
        if leaked:
            raise _fail(
                "STATE_ORACLE_IN_META",
                f"meta.json declares oracle-bearing key(s) {sorted(leaked)}",
                "the oracle/answer key lives on the grading side only (§10.4); "
                "remove it from meta.json",
            )

    try:
        return PuzzleMeta.model_validate(raw)
    except ValidationError as exc:
        raise _fail(
            "CONFIG_META_INVALID",
            f"{_META_FILE} failed schema validation: {exc.error_count()} error(s)",
            "check meta.json against the PuzzleMeta contract (ai_crucible.types)",
        ) from exc


def _load_prompt(root: Path, *, max_bytes: int) -> str:
    prompt_path = root / _PROMPT_FILE
    if not prompt_path.is_file():
        raise _fail(
            "INPUT_PROMPT_MISSING",
            f"missing {_PROMPT_FILE} in {root}",
            f"every puzzle directory must contain a {_PROMPT_FILE} file "
            "(what the Solver sees)",
        )
    _guard_size(
        prompt_path, code="INPUT_PROMPT_TOO_LARGE", label=_PROMPT_FILE, max_bytes=max_bytes
    )
    return prompt_path.read_text(encoding="utf-8")


def _resolve_setup(root: Path) -> Path | None:
    setup_path = root / _SETUP_FILE
    return setup_path if setup_path.is_file() else None
