"""Tests for the puzzle loader (ai_crucible.puzzle).

Discipline (dogfood-swarm): for every invariant we assert the happy path AND
prove the failing path goes RED. The load-bearing invariant here is the sealed
oracle boundary (§10.4) — the loader must never read or expose an answer
artifact — so we test that directly with a planted oracle file.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ai_crucible.puzzle import LoadedPuzzle, PuzzleLoadError, load_puzzle
from ai_crucible.types import PuzzleClass, PuzzleMeta

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
# Size guard at the §1 trust boundary (kernel-runtime-003) — prove RED
# --------------------------------------------------------------------------- #


def test_oversized_prompt_is_rejected(tmp_path: Path) -> None:
    """An over-cap prompt file is refused with a structured PuzzleLoadError BEFORE
    it is slurped whole into memory (kernel-runtime-003, §1 trust boundary).

    The puzzle directory is an external/authored artifact; an unbounded read.text
    of a multi-gigabyte prompt would OOM the kernel before any budget/sandbox cap
    applies. The loader stats the file first and rejects an oversized prompt with
    INPUT_PROMPT_TOO_LARGE (size + cap in the hint), mirroring the sandbox's own
    1 MiB output cap. Caps are overridable so the test plants a tiny cap rather
    than a real multi-MiB file."""
    from ai_crucible.puzzle import MAX_PROMPT_BYTES

    _write_min_puzzle(tmp_path, with_setup=False)
    # Replace the prompt with one that exceeds a deliberately tiny cap.
    (tmp_path / "prompt").write_text("x" * 4096, encoding="utf-8")
    with pytest.raises(PuzzleLoadError, match="INPUT_PROMPT_TOO_LARGE"):
        load_puzzle(tmp_path, max_prompt_bytes=1024)
    # And the default cap is a sane positive ceiling (a few MiB).
    assert MAX_PROMPT_BYTES >= 1_000_000


def test_oversized_meta_is_rejected(tmp_path: Path) -> None:
    """An over-cap meta.json is refused with INPUT_META_TOO_LARGE before json.loads
    reads the whole file (kernel-runtime-003). The meta cap is tighter (~1 MiB) than
    the prompt cap; this plants a tiny override to prove the gate without a real
    1 MiB file."""
    from ai_crucible.puzzle import MAX_META_BYTES

    meta = _min_meta_dict()
    # Pad the JSON with a large ignored-by-validation field to exceed the tiny cap.
    meta["_pad"] = "y" * 4096
    (tmp_path / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    (tmp_path / "prompt").write_text("p", encoding="utf-8")
    with pytest.raises(PuzzleLoadError, match="INPUT_META_TOO_LARGE"):
        load_puzzle(tmp_path, max_meta_bytes=1024)
    assert MAX_META_BYTES >= 1_000_000


def test_prompt_at_cap_loads(tmp_path: Path) -> None:
    """A prompt exactly at the cap is admitted (the guard rejects *over* the cap,
    not at it) — guards against an off-by-one false rejection."""
    _write_min_puzzle(tmp_path, with_setup=False)
    (tmp_path / "prompt").write_text("z" * 1024, encoding="utf-8")
    loaded = load_puzzle(tmp_path, max_prompt_bytes=1024)
    assert len(loaded.prompt) == 1024


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
