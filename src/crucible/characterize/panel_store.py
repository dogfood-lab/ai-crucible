"""Seated-panel artifact store — persist/load the composed panel as a pinned config.

:func:`~crucible.characterize.aggregate.compose_panel` produces the seated panel from a
characterization run; this module makes it **durable**: a small, reviewable JSON artifact
the scorer loads instead of re-deriving the panel every time. The artifact is the
instrument's PIN_PER_STEP receipt — *which* judges sit, with *what* reliability weights,
under *which* ρ-threshold and quorum, and *why* (the composition notes + the dropped
redundancies). It is **generated, not hand-edited** (regenerate it from a run); commit it
so the panel crucible scores with is version-controlled and auditable.

Errors carry the Ship-Gate-B structured shape (``[CODE] message (hint: ...)``) — mirrors
:class:`crucible.calibration.loader.CalibrationLoadError` — so a bad artifact surfaces an
actionable error, never a raw stack.

Standards (the six): **PIN_PER_STEP — 3** — the artifact pins the exact seated set +
weights + ``threshold``/``min_judges`` so a scoring run replays the same panel without
re-running characterization; **NAMED_COMPENSATORS — n/a** — :func:`save_panel` overwrites
one file (the compensator is git: the artifact is committed, so a bad write is reverted by
version control); EXTERNAL_VERIFIER/ANDON live with :func:`compose_panel` that produced it.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from crucible.characterize.aggregate import SeatedJudge, SeatedPanel

__all__ = ["SCHEMA_VERSION", "PanelArtifactError", "save_panel", "load_panel", "panel_to_dict"]

#: Bump when the artifact shape changes incompatibly; :func:`load_panel` refuses a newer
#: major than it understands (forward-incompatible) rather than silently mis-reading.
SCHEMA_VERSION = 1

_SEAT_KEYS = frozenset({"model_id", "reliability_weight", "family", "review_flag"})


class PanelArtifactError(Exception):
    """Raised when a panel artifact is missing, malformed, or schema-incompatible.

    Carries a stable, structured message (Ship Gate B shape: code/message/hint) so a
    caller can surface a redacted, actionable error without a raw stack.
    """


def _fail(code: str, message: str, hint: str) -> PanelArtifactError:
    return PanelArtifactError(f"[{code}] {message} (hint: {hint})")


def panel_to_dict(panel: SeatedPanel) -> dict[str, Any]:
    """Serialize a :class:`SeatedPanel` to a plain dict with the schema version stamped."""
    return {"schema_version": SCHEMA_VERSION, **asdict(panel)}


def save_panel(panel: SeatedPanel, path: Path) -> None:
    """Write ``panel`` to ``path`` as pretty JSON (the committed instrument artifact).

    Overwrites ``path``. The parent directory must exist. The written shape is
    :func:`panel_to_dict` — round-trips through :func:`load_panel`.
    """
    path = Path(path)
    path.write_text(json.dumps(panel_to_dict(panel), indent=2) + "\n", encoding="utf-8")


def _coerce_seat(raw: Any, *, index: int) -> SeatedJudge:
    if not isinstance(raw, dict):
        raise _fail(
            "PANEL_SEAT_NOT_OBJECT",
            f"seats[{index}] is {type(raw).__name__}, not an object",
            "each seat must be a JSON object",
        )
    unknown = set(raw) - _SEAT_KEYS
    if unknown:
        raise _fail(
            "PANEL_SEAT_UNKNOWN_KEY",
            f"seats[{index}] has unknown key(s) {sorted(unknown)}",
            f"allowed seat keys are {sorted(_SEAT_KEYS)}",
        )
    if not isinstance(raw.get("model_id"), str) or not raw["model_id"]:
        raise _fail(
            "PANEL_SEAT_FIELD",
            f"seats[{index}] needs a non-empty string 'model_id'",
            "set 'model_id' to the seated model's id",
        )
    weight = raw.get("reliability_weight")
    if isinstance(weight, bool) or not isinstance(weight, (int, float)):
        raise _fail(
            "PANEL_SEAT_FIELD",
            f"seats[{index}] needs a numeric 'reliability_weight'",
            "set 'reliability_weight' to the judge's weight (a number)",
        )
    return SeatedJudge(
        model_id=raw["model_id"],
        reliability_weight=float(weight),
        family=raw.get("family"),
        review_flag=bool(raw.get("review_flag", False)),
    )


def panel_from_dict(data: Any) -> SeatedPanel:
    """Reconstruct a :class:`SeatedPanel` from a parsed artifact dict (validated)."""
    if not isinstance(data, dict):
        raise _fail(
            "PANEL_JSON_SHAPE",
            f"artifact top-level JSON is {type(data).__name__}, not an object",
            "a panel artifact is a single JSON object",
        )
    version = data.get("schema_version")
    if version is None:
        raise _fail(
            "PANEL_NO_VERSION",
            "artifact is missing 'schema_version'",
            f"this build writes/reads schema_version {SCHEMA_VERSION}",
        )
    if not isinstance(version, int) or version > SCHEMA_VERSION:
        raise _fail(
            "PANEL_VERSION_UNSUPPORTED",
            f"artifact schema_version {version!r} is newer than supported ({SCHEMA_VERSION})",
            "regenerate the artifact with this build, or upgrade crucible",
        )
    seats_raw = data.get("seats")
    if not isinstance(seats_raw, list):
        raise _fail(
            "PANEL_SEATS_SHAPE",
            f"'seats' is {type(seats_raw).__name__}, not a list",
            "seats must be a JSON array of seat objects (possibly empty)",
        )
    seats = [_coerce_seat(s, index=i) for i, s in enumerate(seats_raw)]
    try:
        return SeatedPanel(
            seats=seats,
            submodular=bool(data["submodular"]),
            meets_quorum=bool(data["meets_quorum"]),
            escalate=bool(data["escalate"]),
            not_seated=list(data.get("not_seated", [])),
            dropped_redundant=list(data.get("dropped_redundant", [])),
            threshold=float(data["threshold"]),
            min_judges=int(data["min_judges"]),
            notes=list(data.get("notes", [])),
        )
    except KeyError as exc:
        raise _fail(
            "PANEL_FIELD_MISSING",
            f"artifact is missing required field {exc.args[0]!r}",
            "regenerate the artifact from a characterization run (compose_panel)",
        ) from exc


def load_panel(path: Path) -> SeatedPanel:
    """Read + validate a panel artifact from ``path`` → :class:`SeatedPanel`.

    Raises:
        PanelArtifactError: the path is missing, the file is not valid JSON, the schema
            version is unsupported, or a field is missing/mistyped.
    """
    path = Path(path)
    if not path.is_file():
        raise _fail(
            "PANEL_PATH_MISSING",
            f"panel artifact does not exist: {path}",
            "pass a path to a panel.json written by save_panel / run.py --write-panel",
        )
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise _fail(
            "PANEL_BAD_JSON",
            f"{path.name} is not valid JSON: {exc}",
            f"fix the JSON syntax in {path.name} or regenerate it",
        ) from exc
    return panel_from_dict(data)
