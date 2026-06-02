"""Unit tests for the seated-panel artifact store (save/load + validation)."""

from __future__ import annotations

import json

import pytest

from crucible.characterize.aggregate import SeatedJudge, SeatedPanel
from crucible.characterize.panel_store import (
    SCHEMA_VERSION,
    PanelArtifactError,
    load_panel,
    panel_from_dict,
    save_panel,
)


def _panel() -> SeatedPanel:
    return SeatedPanel(
        seats=[
            SeatedJudge("qwen3.6:27b", 1.0, "qwen", True),
            SeatedJudge("gemma4:31b", 1.0, "gemma", True),
            SeatedJudge("granite4.1:30b", 0.978, "granite", True),
        ],
        submodular=True,
        meets_quorum=True,
        escalate=False,
        not_seated=["aya-expanse:32b"],
        dropped_redundant=[],
        threshold=0.25,
        min_judges=3,
        notes=["3 seated from 3 admitted"],
    )


def _valid_dict(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "seats": [{"model_id": "m", "reliability_weight": 1.0, "family": "f"}],
        "submodular": True,
        "meets_quorum": True,
        "escalate": False,
        "not_seated": [],
        "dropped_redundant": [],
        "threshold": 0.25,
        "min_judges": 3,
        "notes": [],
    }
    base.update(overrides)
    return base


def test_round_trip(tmp_path) -> None:
    path = tmp_path / "panel.json"
    save_panel(_panel(), path)
    loaded = load_panel(path)
    assert [s.model_id for s in loaded.seats] == ["qwen3.6:27b", "gemma4:31b", "granite4.1:30b"]
    assert loaded.weights == {"qwen3.6:27b": 1.0, "gemma4:31b": 1.0, "granite4.1:30b": 0.978}
    assert loaded.seats[0].family == "qwen" and loaded.seats[0].review_flag is True
    assert loaded.submodular and loaded.meets_quorum and not loaded.escalate
    assert loaded.not_seated == ["aya-expanse:32b"]
    assert loaded.threshold == 0.25 and loaded.min_judges == 3


def test_save_stamps_schema_version(tmp_path) -> None:
    path = tmp_path / "panel.json"
    save_panel(_panel(), path)
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["schema_version"] == SCHEMA_VERSION
    assert isinstance(data["seats"], list) and len(data["seats"]) == 3


def test_empty_panel_round_trip(tmp_path) -> None:
    """A sub-quorum (escalate) panel persists too — the honest artifact records it."""
    panel = SeatedPanel(
        seats=[],
        submodular=True,
        meets_quorum=False,
        escalate=True,
        not_seated=["x"],
        dropped_redundant=[],
        threshold=0.25,
        min_judges=3,
        notes=["quorum 0/3 → escalate"],
    )
    path = tmp_path / "panel.json"
    save_panel(panel, path)
    loaded = load_panel(path)
    assert loaded.seats == [] and loaded.escalate is True and not loaded.meets_quorum


def test_load_missing_path(tmp_path) -> None:
    with pytest.raises(PanelArtifactError, match="PANEL_PATH_MISSING"):
        load_panel(tmp_path / "nope.json")


def test_load_bad_json(tmp_path) -> None:
    path = tmp_path / "bad.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(PanelArtifactError, match="PANEL_BAD_JSON"):
        load_panel(path)


def test_missing_schema_version() -> None:
    bad = _valid_dict()
    del bad["schema_version"]
    with pytest.raises(PanelArtifactError, match="PANEL_NO_VERSION"):
        panel_from_dict(bad)


def test_future_schema_version() -> None:
    with pytest.raises(PanelArtifactError, match="PANEL_VERSION_UNSUPPORTED"):
        panel_from_dict(_valid_dict(schema_version=SCHEMA_VERSION + 1))


def test_seat_unknown_key() -> None:
    bad = _valid_dict(
        seats=[{"model_id": "m", "reliability_weight": 1.0, "bogus": 1}]
    )
    with pytest.raises(PanelArtifactError, match="PANEL_SEAT_UNKNOWN_KEY"):
        panel_from_dict(bad)


def test_seat_bad_weight_rejects_string_and_bool() -> None:
    with pytest.raises(PanelArtifactError, match="PANEL_SEAT_FIELD"):
        panel_from_dict(_valid_dict(seats=[{"model_id": "m", "reliability_weight": "high"}]))
    # bool is an int subclass — must be rejected so True never becomes 1.0
    with pytest.raises(PanelArtifactError, match="PANEL_SEAT_FIELD"):
        panel_from_dict(_valid_dict(seats=[{"model_id": "m", "reliability_weight": True}]))


def test_seat_missing_model_id() -> None:
    with pytest.raises(PanelArtifactError, match="PANEL_SEAT_FIELD"):
        panel_from_dict(_valid_dict(seats=[{"reliability_weight": 1.0}]))


def test_missing_required_field() -> None:
    bad = _valid_dict()
    del bad["threshold"]
    with pytest.raises(PanelArtifactError, match="PANEL_FIELD_MISSING"):
        panel_from_dict(bad)


def test_non_object_top_level() -> None:
    with pytest.raises(PanelArtifactError, match="PANEL_JSON_SHAPE"):
        panel_from_dict([1, 2, 3])
