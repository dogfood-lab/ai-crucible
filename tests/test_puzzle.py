"""Tests for the puzzle loader (crucible.puzzle).

Discipline (dogfood-swarm): for every invariant we assert the happy path AND
prove the failing path goes RED. The load-bearing invariant here is the sealed
oracle boundary (§10.4) — the loader must never read or expose an answer
artifact — so we test that directly with a planted oracle file.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from crucible.puzzle import LoadedPuzzle, PuzzleLoadError, load_puzzle
from crucible.types import PuzzleClass, PuzzleMeta

FIXTURE = Path(__file__).parent / "fixtures" / "puzzles" / "sample"


# --------------------------------------------------------------------------- #
# Happy path
# --------------------------------------------------------------------------- #


def test_loads_sample_fixture() -> None:
    loaded = load_puzzle(FIXTURE)
    assert isinstance(loaded, LoadedPuzzle)
    assert isinstance(loaded.meta, PuzzleMeta)
    assert loaded.meta.puzzle_id == "sample-fixture-0001"
    assert loaded.meta.puzzle_class is PuzzleClass.FILE_INSPECTION
    assert "MAX_RETRIES" in loaded.prompt
    assert loaded.root == FIXTURE


def test_resolves_setup_script_when_present() -> None:
    loaded = load_puzzle(FIXTURE)
    assert loaded.setup_script is not None
    assert loaded.setup_script.name == "setup_script"
    assert loaded.setup_script.is_file()


def test_setup_script_is_none_when_absent(tmp_path: Path) -> None:
    _write_min_puzzle(tmp_path, with_setup=False)
    loaded = load_puzzle(tmp_path)
    assert loaded.setup_script is None


# --------------------------------------------------------------------------- #
# Sealed oracle boundary (§10.4) — the load-bearing invariant
# --------------------------------------------------------------------------- #


def test_loaded_puzzle_has_no_oracle_field() -> None:
    """Structural proof: there is no attribute that could carry the answer."""
    loaded = load_puzzle(FIXTURE)
    for forbidden in ("oracle", "answer", "answer_key", "solution", "gold"):
        assert not hasattr(loaded, forbidden), f"LoadedPuzzle exposes {forbidden!r}"


def test_oracle_file_is_never_read(tmp_path: Path) -> None:
    """Plant an oracle file in the puzzle dir. The loader must ignore it entirely
    — its contents must not appear anywhere in the returned object."""
    _write_min_puzzle(tmp_path, with_setup=False)
    secret = "THE_ANSWER_IS_42_DO_NOT_LEAK"
    (tmp_path / "oracle").write_text(secret, encoding="utf-8")
    loaded = load_puzzle(tmp_path)
    blob = repr(loaded) + loaded.prompt + (loaded.meta.model_dump_json())
    assert secret not in blob


def test_oracle_key_in_meta_is_rejected(tmp_path: Path) -> None:
    """Defense-in-depth: an oracle smuggled into meta.json goes RED, it does not
    silently load into Solver-visible state (§10.4)."""
    meta = _min_meta_dict()
    meta["oracle"] = {"expected": 7}
    (tmp_path / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    (tmp_path / "prompt").write_text("p", encoding="utf-8")
    with pytest.raises(PuzzleLoadError, match="ORACLE_IN_META"):
        load_puzzle(tmp_path)


# --------------------------------------------------------------------------- #
# Error paths — prove each gate goes RED
# --------------------------------------------------------------------------- #


def test_missing_directory_raises(tmp_path: Path) -> None:
    with pytest.raises(PuzzleLoadError, match="PUZZLE_DIR_MISSING"):
        load_puzzle(tmp_path / "does-not-exist")


def test_missing_meta_raises(tmp_path: Path) -> None:
    (tmp_path / "prompt").write_text("p", encoding="utf-8")
    with pytest.raises(PuzzleLoadError, match="META_MISSING"):
        load_puzzle(tmp_path)


def test_missing_prompt_raises(tmp_path: Path) -> None:
    (tmp_path / "meta.json").write_text(json.dumps(_min_meta_dict()), encoding="utf-8")
    with pytest.raises(PuzzleLoadError, match="PROMPT_MISSING"):
        load_puzzle(tmp_path)


def test_bad_json_raises(tmp_path: Path) -> None:
    (tmp_path / "meta.json").write_text("{not valid json", encoding="utf-8")
    (tmp_path / "prompt").write_text("p", encoding="utf-8")
    with pytest.raises(PuzzleLoadError, match="BAD_JSON"):
        load_puzzle(tmp_path)


def test_schema_violation_raises(tmp_path: Path) -> None:
    """meta.json violating the §8.3 bound (elegance > 30% of solve) goes RED via
    PuzzleMeta validation surfaced as PuzzleLoadError."""
    meta = _min_meta_dict()
    meta["rewards"]["elegance_bonus_max"] = 999.0  # >> 30% of solve
    (tmp_path / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    (tmp_path / "prompt").write_text("p", encoding="utf-8")
    with pytest.raises(PuzzleLoadError, match="META_INVALID"):
        load_puzzle(tmp_path)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _min_meta_dict() -> dict:
    return {
        "puzzle_id": "tmp-0001",
        "created_at": "2026-06-01T00:00:00Z",
        "capability_aspect": "x",
        "puzzle_class": "file_inspection",
        "point_threshold": 50.0,
        "time_budget_seconds": 60,
        "tool_call_budget": 5,
        "rewards": {"solve": 100.0, "elegance_bonus_max": 30.0, "novelty_bonus_max": 50.0},
    }


def _write_min_puzzle(root: Path, *, with_setup: bool) -> None:
    (root / "meta.json").write_text(json.dumps(_min_meta_dict()), encoding="utf-8")
    (root / "prompt").write_text("read the file and report the value", encoding="utf-8")
    if with_setup:
        (root / "setup_script").write_text("#!/usr/bin/env bash\n", encoding="utf-8")
