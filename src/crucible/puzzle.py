"""Puzzle artifact loader (``puzzle_loader`` module, research-grounding §10.2).

A puzzle is a *directory* (§1 artifact structure):

- ``meta.json``    — the per-puzzle contract, validated against
  :class:`crucible.types.PuzzleMeta`.
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

from crucible.types import PuzzleMeta

__all__ = ["LoadedPuzzle", "PuzzleLoadError", "load_puzzle"]

# Canonical filenames inside a puzzle directory.
_META_FILE = "meta.json"
_PROMPT_FILE = "prompt"
_SETUP_FILE = "setup_script"

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


def load_puzzle(path: Path) -> LoadedPuzzle:
    """Load the puzzle directory at ``path`` into a :class:`LoadedPuzzle`.

    Reads ``meta.json`` (validated via :class:`PuzzleMeta`), the ``prompt`` file,
    and resolves ``setup_script`` if present. Never reads any oracle/answer
    artifact (§10.4).

    Raises:
        PuzzleLoadError: directory missing, required file missing, ``meta.json``
            is not valid JSON, or ``meta.json`` fails :class:`PuzzleMeta`
            validation.
    """
    root = Path(path)
    if not root.is_dir():
        raise _fail(
            "INPUT_PUZZLE_DIR_MISSING",
            f"puzzle path is not a directory: {root}",
            "pass the path to a puzzle directory (the one containing meta.json)",
        )

    meta = _load_meta(root)
    prompt = _load_prompt(root)
    setup_script = _resolve_setup(root)

    return LoadedPuzzle(meta=meta, prompt=prompt, setup_script=setup_script, root=root)


def _load_meta(root: Path) -> PuzzleMeta:
    meta_path = root / _META_FILE
    if not meta_path.is_file():
        raise _fail(
            "INPUT_META_MISSING",
            f"missing {_META_FILE} in {root}",
            f"every puzzle directory must contain a {_META_FILE} file",
        )
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
            "check meta.json against the PuzzleMeta contract (crucible.types)",
        ) from exc


def _load_prompt(root: Path) -> str:
    prompt_path = root / _PROMPT_FILE
    if not prompt_path.is_file():
        raise _fail(
            "INPUT_PROMPT_MISSING",
            f"missing {_PROMPT_FILE} in {root}",
            f"every puzzle directory must contain a {_PROMPT_FILE} file "
            "(what the Solver sees)",
        )
    return prompt_path.read_text(encoding="utf-8")


def _resolve_setup(root: Path) -> Path | None:
    setup_path = root / _SETUP_FILE
    return setup_path if setup_path.is_file() else None
