"""Calibration item loader (research-grounding §11.3).

Reads calibration items from a single JSON file *or* a directory of ``*.json``
files, validates each record into a :class:`ai_crucible.calibration.types.CalibrationItem`,
and raises a structured :class:`CalibrationLoadError` (Ship-Gate-B shape:
``[CODE] message (hint: ...)``) on bad data.

``CalibrationItem`` is a plain ``@dataclass`` (no Pydantic), so validation is
performed here: required fields present, correct types, ``category`` resolves to
a known :class:`CalibrationCategory`, and the known-groups expectation
(``expected_pass``) is a ``dict[str, bool]``.

THE GOLD IS GRADING-SIDE. ``CalibrationItem.gold`` is the correct verdict and is
NEVER shown to the model under characterization (the sealed-boundary principle,
§8.5/§10.4 — mirrored from the puzzle loader). The loader reads ``gold`` because
it lives *with the grader*; callers that feed an item to a model must send only
``prompt`` (never ``gold``).

A SMALL bundled starter bank ships under ``items/`` as fixtures/examples that
exercise this loader + the known-groups check and give an authoring template.
The full ~40-60-item set (§11.3) is a later director decision (§11.7).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ai_crucible.calibration.types import CalibrationCategory, CalibrationItem

__all__ = ["CalibrationLoadError", "load_default", "load_items"]

# Directory holding the bundled starter bank (sibling of this module).
_ITEMS_DIR = Path(__file__).parent / "items"

# Resource bounds at the calibration trust boundary (an operator-supplied dir is
# external/authored content — §11.3). Without a ceiling, a directory with tens of
# thousands of files, or a single multi-gigabyte JSON, is slurped whole via
# read_text and can OOM the process before any validation runs. These caps make
# the loader fail with a structured error instead. They are generous (the real
# bank is ~14 files of a few KB each; the discriminating pair-set ~80-150) so they
# never trip on legitimate input, and overridable for a deliberately large bank.
_MAX_FILE_BYTES = 16 * 1024 * 1024  # 16 MiB per JSON file
_MAX_DIR_FILES = 10_000  # *.json files in a single operator directory

# Required (no-default) fields on CalibrationItem that every record must supply.
_REQUIRED_STR_FIELDS = ("id", "construct", "confound_controlled", "prompt")

# All keys a record may carry (anything else is a typo / smuggled field).
_ALLOWED_KEYS = frozenset(
    {
        "id",
        "category",
        "construct",
        "confound_controlled",
        "prompt",
        "gold",
        "difficulty",
        "discrimination",
        "expected_pass",
        "metadata",
    }
)


class CalibrationLoadError(Exception):
    """Raised when a calibration source is missing, malformed, or a record fails
    :class:`CalibrationItem` validation.

    Carries a stable, structured message (Ship Gate B shape: code/message/hint)
    so callers can surface a redacted, actionable error without a raw stack.
    """


def _fail(code: str, message: str, hint: str) -> CalibrationLoadError:
    """Build a structured :class:`CalibrationLoadError` (code/message/hint)."""
    return CalibrationLoadError(f"[{code}] {message} (hint: {hint})")


def load_items(path: Path) -> list[CalibrationItem]:
    """Load calibration items from ``path``.

    ``path`` may be:

    - a single ``*.json`` file holding either one item object or a JSON array of
      item objects;
    - a directory, in which case every ``*.json`` file directly inside it is read
      (sorted by name for deterministic ordering) and concatenated.

    Each record is validated into a :class:`CalibrationItem`.

    Raises:
        CalibrationLoadError: path missing; a directory exceeds the file-count
            cap or a file exceeds the per-file size cap (resource bounds at the
            trust boundary, §11.3); a file is not valid JSON; JSON is not an
            object or list-of-objects; a record is missing a required field, has a
            wrong field type, names an unknown ``category``, or carries an
            unexpected key; or duplicate item ids are found across the source.
    """
    root = Path(path)
    if root.is_dir():
        records = _read_dir(root)
    elif root.is_file():
        records = _read_file(root)
    else:
        raise _fail(
            "INPUT_PATH_MISSING",
            f"calibration path does not exist: {root}",
            "pass a path to a .json file or a directory of .json files",
        )

    items = [_build_item(rec, source=src) for rec, src in records]
    _reject_duplicate_ids(items)
    return items


def load_default() -> list[CalibrationItem]:
    """Load the bundled starter item bank from ``items/``.

    The starter bank is a SMALL set of fixtures/examples (~2-3 per category) that
    exercise the loader + known-groups check and serve as an authoring template.
    The full ~40-60-item set (§11.3) is a later director decision (§11.7).

    Raises:
        CalibrationLoadError: the bundled bank is missing or any record is
            malformed (an installation/packaging fault).
    """
    if not _ITEMS_DIR.is_dir():
        raise _fail(
            "STATE_BUNDLE_MISSING",
            f"bundled starter bank not found at {_ITEMS_DIR}",
            "reinstall ai_crucible; the calibration/items/ directory is missing",
        )
    return load_items(_ITEMS_DIR)


# --------------------------------------------------------------------------- #
# Reading
# --------------------------------------------------------------------------- #


def _read_dir(root: Path) -> list[tuple[dict[str, Any], Path]]:
    files = sorted(root.glob("*.json"))
    if not files:
        raise _fail(
            "INPUT_DIR_EMPTY",
            f"no .json files found in directory {root}",
            "put one or more *.json item files in the directory",
        )
    # Resource bound: refuse an unreasonably large operator directory before
    # reading any of it (§11.3 trust boundary). A clean rejection beats an OOM.
    if len(files) > _MAX_DIR_FILES:
        raise _fail(
            "INPUT_DIR_TOO_MANY_FILES",
            f"calibration directory {root} has {len(files)} .json files, over the "
            f"{_MAX_DIR_FILES} cap",
            "split the bank or raise _MAX_DIR_FILES deliberately; the cap guards "
            "against reading an unbounded operator directory into memory",
        )
    out: list[tuple[dict[str, Any], Path]] = []
    for file in files:
        out.extend(_read_file(file))
    return out


def _read_file(file: Path) -> list[tuple[dict[str, Any], Path]]:
    # Resource bound: stat the file and refuse an oversized one before slurping
    # it into memory via read_text (§11.3 trust boundary; mirrors the sandbox's
    # output cap). A multi-GiB JSON would otherwise OOM the loader.
    try:
        size = file.stat().st_size
    except OSError as exc:
        raise _fail(
            "INPUT_FILE_UNREADABLE",
            f"cannot stat calibration file {file.name}: {exc}",
            f"ensure {file.name} exists and is readable",
        ) from exc
    if size > _MAX_FILE_BYTES:
        raise _fail(
            "INPUT_FILE_TOO_LARGE",
            f"calibration file {file.name} is {size} bytes, over the "
            f"{_MAX_FILE_BYTES}-byte cap",
            "a calibration JSON file should be small (items, not corpora); split "
            "the file or raise _MAX_FILE_BYTES deliberately",
        )
    try:
        raw = json.loads(file.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise _fail(
            "INPUT_BAD_JSON",
            f"{file.name} is not valid JSON: {exc}",
            f"fix the JSON syntax in {file.name}",
        ) from exc

    if isinstance(raw, dict):
        return [(raw, file)]
    if isinstance(raw, list):
        out: list[tuple[dict[str, Any], Path]] = []
        for idx, entry in enumerate(raw):
            if not isinstance(entry, dict):
                raise _fail(
                    "INPUT_RECORD_NOT_OBJECT",
                    f"{file.name}[{idx}] is {type(entry).__name__}, not a JSON object",
                    "each calibration item must be a JSON object",
                )
            out.append((entry, file))
        return out

    raise _fail(
        "INPUT_JSON_SHAPE",
        f"{file.name} top-level JSON is {type(raw).__name__}, not object or array",
        "a calibration file is one item object or an array of item objects",
    )


# --------------------------------------------------------------------------- #
# Validation -> CalibrationItem
# --------------------------------------------------------------------------- #


def _build_item(rec: dict[str, Any], *, source: Path) -> CalibrationItem:
    where = f"{source.name} (id={rec.get('id', '?')!r})"

    unknown = set(rec.keys()) - _ALLOWED_KEYS
    if unknown:
        raise _fail(
            "CONFIG_UNKNOWN_KEY",
            f"{where} has unknown key(s) {sorted(unknown)}",
            f"allowed keys are {sorted(_ALLOWED_KEYS)}",
        )

    for fname in _REQUIRED_STR_FIELDS:
        if fname not in rec:
            raise _fail(
                "INPUT_FIELD_MISSING",
                f"{where} is missing required field {fname!r}",
                f"every item must declare {list(_REQUIRED_STR_FIELDS)} + 'category' + 'gold'",
            )
        if not isinstance(rec[fname], str) or not rec[fname]:
            raise _fail(
                "CONFIG_FIELD_TYPE",
                f"{where} field {fname!r} must be a non-empty string",
                f"set {fname!r} to a non-empty string",
            )

    if "gold" not in rec:
        raise _fail(
            "INPUT_FIELD_MISSING",
            f"{where} is missing required field 'gold'",
            "declare the grading-side correct verdict/answer in 'gold'",
        )

    category = _coerce_category(rec.get("category"), where=where)
    expected_pass = _coerce_expected_pass(rec.get("expected_pass", {}), where=where)
    difficulty = _coerce_opt_float(rec.get("difficulty"), field="difficulty", where=where)
    discrimination = _coerce_opt_float(
        rec.get("discrimination"), field="discrimination", where=where
    )
    metadata = rec.get("metadata", {})
    if not isinstance(metadata, dict):
        raise _fail(
            "CONFIG_FIELD_TYPE",
            f"{where} field 'metadata' must be an object",
            "set 'metadata' to a JSON object (or omit it)",
        )

    return CalibrationItem(
        id=rec["id"],
        category=category,
        construct=rec["construct"],
        confound_controlled=rec["confound_controlled"],
        prompt=rec["prompt"],
        gold=rec["gold"],
        difficulty=difficulty,
        discrimination=discrimination,
        expected_pass=expected_pass,
        metadata=dict(metadata),
    )


def _coerce_category(value: Any, *, where: str) -> CalibrationCategory:
    if value is None:
        raise _fail(
            "INPUT_FIELD_MISSING",
            f"{where} is missing required field 'category'",
            f"set 'category' to one of {[c.value for c in CalibrationCategory]}",
        )
    if not isinstance(value, str):
        raise _fail(
            "CONFIG_FIELD_TYPE",
            f"{where} field 'category' must be a string",
            f"use one of {[c.value for c in CalibrationCategory]}",
        )
    try:
        return CalibrationCategory(value)
    except ValueError as exc:
        raise _fail(
            "CONFIG_CATEGORY_UNKNOWN",
            f"{where} has unknown category {value!r}",
            f"category must be one of {[c.value for c in CalibrationCategory]}",
        ) from exc


def _coerce_expected_pass(value: Any, *, where: str) -> dict[str, bool]:
    if not isinstance(value, dict):
        raise _fail(
            "CONFIG_FIELD_TYPE",
            f"{where} field 'expected_pass' must be an object",
            "map ability-tier -> bool, e.g. {\"strong\": true, \"weak\": false}",
        )
    out: dict[str, bool] = {}
    for tier, expected in value.items():
        if not isinstance(tier, str) or not tier:
            raise _fail(
                "CONFIG_EXPECTED_PASS_TIER",
                f"{where} 'expected_pass' has a non-string/empty tier key",
                "tier keys must be non-empty strings (e.g. 'strong', 'weak')",
            )
        if not isinstance(expected, bool):
            raise _fail(
                "CONFIG_EXPECTED_PASS_VALUE",
                f"{where} 'expected_pass[{tier!r}]' must be a bool",
                "each tier maps to true/false",
            )
        out[tier] = expected
    return out


def _coerce_opt_float(value: Any, *, field: str, where: str) -> float | None:
    if value is None:
        return None
    # bool is a subclass of int; reject it explicitly to avoid silent True->1.0.
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise _fail(
            "CONFIG_FIELD_TYPE",
            f"{where} field {field!r} must be a number or null",
            f"set {field!r} to a float (IRT parameter) or omit it",
        )
    return float(value)


def _reject_duplicate_ids(items: list[CalibrationItem]) -> None:
    seen: set[str] = set()
    dupes: list[str] = []
    for item in items:
        if item.id in seen:
            dupes.append(item.id)
        seen.add(item.id)
    if dupes:
        raise _fail(
            "CONFIG_DUPLICATE_ID",
            f"duplicate calibration item id(s): {sorted(set(dupes))}",
            "every calibration item id must be unique across the source",
        )
