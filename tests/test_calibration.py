"""Tests for the calibration loader + known-groups acceptance check (§11.3).

Discipline (dogfood-swarm): for every invariant, assert the happy path AND prove
the failing path goes RED. The load-bearing invariants here are:

- the loader round-trips the bundled starter bank into valid ``CalibrationItem``s
  and rejects malformed / under-specified records with a structured error;
- the known-groups laws (§11.3) PASS a clean outcome set and FAIL — with the
  *correct* violation — when a trivial item fails, an impossible item passes, or a
  diagnostic item is non-monotone with ability.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import ai_crucible.calibration.loader as loader_mod
from ai_crucible.calibration.irt import (
    IRTError,
    fit_irt_bayesian,
    point_biserial,
    prune_items,
)
from ai_crucible.calibration.known_groups import (
    KnownGroupsResult,
    check_known_groups,
)
from ai_crucible.calibration.loader import (
    CalibrationLoadError,
    load_default,
    load_items,
)
from ai_crucible.calibration.types import CalibrationCategory, CalibrationItem

_CALIB_DIR = Path(__file__).resolve().parents[1] / "src" / "ai_crucible" / "calibration"
ITEMS_DIR = _CALIB_DIR / "items"
ADMISSION_PAIRS = _CALIB_DIR / "admission_pairs.json"


# --------------------------------------------------------------------------- #
# Loader — happy path / round-trip
# --------------------------------------------------------------------------- #


def test_load_default_returns_starter_bank() -> None:
    items = load_default()
    assert all(isinstance(it, CalibrationItem) for it in items)
    # ~2-3 per category across the 5 categories.
    assert len(items) == 14
    ids = {it.id for it in items}
    assert len(ids) == len(items), "starter-bank ids must be unique"


def test_starter_bank_covers_all_five_categories() -> None:
    items = load_default()
    seen = {it.category for it in items}
    assert seen == set(CalibrationCategory), "starter bank must exercise every category"


def test_load_default_matches_load_items_on_dir() -> None:
    assert {it.id for it in load_default()} == {it.id for it in load_items(ITEMS_DIR)}


def test_load_single_file_round_trips(tmp_path: Path) -> None:
    src = tmp_path / "one.json"
    src.write_text(json.dumps(_min_record()), encoding="utf-8")
    items = load_items(src)
    assert len(items) == 1
    item = items[0]
    assert item.id == "x-0001"
    assert item.category is CalibrationCategory.KNOWN_TRIVIAL
    assert item.gold == "4"
    assert item.expected_pass == {"weak": True, "strong": True}


def test_load_array_file_round_trips(tmp_path: Path) -> None:
    a, b = _min_record("a-1"), _min_record("b-2")
    (tmp_path / "arr.json").write_text(json.dumps([a, b]), encoding="utf-8")
    items = load_items(tmp_path / "arr.json")
    assert {it.id for it in items} == {"a-1", "b-2"}


def test_load_dir_is_sorted_and_concatenated(tmp_path: Path) -> None:
    (tmp_path / "b.json").write_text(json.dumps(_min_record("from-b")), encoding="utf-8")
    (tmp_path / "a.json").write_text(json.dumps(_min_record("from-a")), encoding="utf-8")
    items = load_items(tmp_path)
    # sorted by filename: a.json before b.json
    assert [it.id for it in items] == ["from-a", "from-b"]


def test_optional_irt_fields_round_trip(tmp_path: Path) -> None:
    rec = _min_record()
    rec["difficulty"] = 0.5
    rec["discrimination"] = 1.6
    (tmp_path / "i.json").write_text(json.dumps(rec), encoding="utf-8")
    item = load_items(tmp_path / "i.json")[0]
    assert item.difficulty == 0.5
    assert item.discrimination == 1.6


# --------------------------------------------------------------------------- #
# Loader — error paths (prove each gate goes RED)
# --------------------------------------------------------------------------- #


def test_missing_path_raises(tmp_path: Path) -> None:
    with pytest.raises(CalibrationLoadError, match="PATH_MISSING"):
        load_items(tmp_path / "nope.json")


def test_malformed_json_raises(tmp_path: Path) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text("{not valid json", encoding="utf-8")
    with pytest.raises(CalibrationLoadError, match="BAD_JSON"):
        load_items(bad)


def test_empty_dir_raises(tmp_path: Path) -> None:
    with pytest.raises(CalibrationLoadError, match="DIR_EMPTY"):
        load_items(tmp_path)


def test_non_object_record_in_array_raises(tmp_path: Path) -> None:
    (tmp_path / "x.json").write_text(json.dumps([_min_record(), 5]), encoding="utf-8")
    with pytest.raises(CalibrationLoadError, match="RECORD_NOT_OBJECT"):
        load_items(tmp_path / "x.json")


def test_top_level_scalar_raises(tmp_path: Path) -> None:
    (tmp_path / "x.json").write_text(json.dumps(42), encoding="utf-8")
    with pytest.raises(CalibrationLoadError, match="JSON_SHAPE"):
        load_items(tmp_path / "x.json")


def test_missing_required_field_raises(tmp_path: Path) -> None:
    rec = _min_record()
    del rec["prompt"]
    (tmp_path / "x.json").write_text(json.dumps(rec), encoding="utf-8")
    with pytest.raises(CalibrationLoadError, match="FIELD_MISSING"):
        load_items(tmp_path / "x.json")


def test_missing_gold_raises(tmp_path: Path) -> None:
    rec = _min_record()
    del rec["gold"]
    (tmp_path / "x.json").write_text(json.dumps(rec), encoding="utf-8")
    with pytest.raises(CalibrationLoadError, match="FIELD_MISSING"):
        load_items(tmp_path / "x.json")


def test_unknown_category_raises(tmp_path: Path) -> None:
    rec = _min_record()
    rec["category"] = "totally_made_up"
    (tmp_path / "x.json").write_text(json.dumps(rec), encoding="utf-8")
    with pytest.raises(CalibrationLoadError, match="CATEGORY_UNKNOWN"):
        load_items(tmp_path / "x.json")


def test_unknown_key_raises(tmp_path: Path) -> None:
    rec = _min_record()
    rec["oops"] = 1
    (tmp_path / "x.json").write_text(json.dumps(rec), encoding="utf-8")
    with pytest.raises(CalibrationLoadError, match="UNKNOWN_KEY"):
        load_items(tmp_path / "x.json")


def test_bad_expected_pass_value_raises(tmp_path: Path) -> None:
    rec = _min_record()
    rec["expected_pass"] = {"weak": "yes"}  # not a bool
    (tmp_path / "x.json").write_text(json.dumps(rec), encoding="utf-8")
    with pytest.raises(CalibrationLoadError, match="EXPECTED_PASS_VALUE"):
        load_items(tmp_path / "x.json")


def test_difficulty_bool_rejected(tmp_path: Path) -> None:
    rec = _min_record()
    rec["difficulty"] = True  # bool must not be silently coerced to 1.0
    (tmp_path / "x.json").write_text(json.dumps(rec), encoding="utf-8")
    with pytest.raises(CalibrationLoadError, match="FIELD_TYPE"):
        load_items(tmp_path / "x.json")


def test_duplicate_ids_across_dir_raises(tmp_path: Path) -> None:
    (tmp_path / "a.json").write_text(json.dumps(_min_record("dup")), encoding="utf-8")
    (tmp_path / "b.json").write_text(json.dumps(_min_record("dup")), encoding="utf-8")
    with pytest.raises(CalibrationLoadError, match="DUPLICATE_ID"):
        load_items(tmp_path)


# --------------------------------------------------------------------------- #
# Loader — resource bounds at the trust boundary  [finding 005]
# --------------------------------------------------------------------------- #


def test_oversized_calibration_file_is_bounded(tmp_path: Path, monkeypatch) -> None:
    # [instrument-future-deps-005] An operator-supplied file is external content;
    # without a cap it is slurped whole via read_text and can OOM. SIMULATE an
    # oversized file (shrink the cap so the test stays cheap) and assert the
    # bounded structured rejection BEFORE the read, not a crash.
    monkeypatch.setattr(loader_mod, "_MAX_FILE_BYTES", 64)
    big = tmp_path / "big.json"
    # A valid-shaped record whose serialisation exceeds the (shrunk) cap.
    rec = _min_record()
    rec["construct"] = "x" * 200  # push the file well past 64 bytes
    big.write_text(json.dumps(rec), encoding="utf-8")
    assert big.stat().st_size > 64

    with pytest.raises(CalibrationLoadError, match="FILE_TOO_LARGE"):
        load_items(big)


def test_over_count_calibration_dir_is_bounded(tmp_path: Path, monkeypatch) -> None:
    # [instrument-future-deps-005] An operator directory with an unbounded file
    # count must be rejected before reading any of it. SIMULATE an over-count dir
    # (shrink the cap) and assert the bounded structured rejection.
    monkeypatch.setattr(loader_mod, "_MAX_DIR_FILES", 2)
    for i in range(3):  # 3 > the shrunk cap of 2
        (tmp_path / f"item-{i}.json").write_text(
            json.dumps(_min_record(f"id-{i}")), encoding="utf-8"
        )

    with pytest.raises(CalibrationLoadError, match="DIR_TOO_MANY_FILES"):
        load_items(tmp_path)


def test_within_bounds_dir_still_loads(tmp_path: Path) -> None:
    # Regression guard: a normal small directory loads cleanly under the real
    # (generous) caps — the bound must not trip on legitimate input.
    (tmp_path / "a.json").write_text(json.dumps(_min_record("ok-1")), encoding="utf-8")
    (tmp_path / "b.json").write_text(json.dumps(_min_record("ok-2")), encoding="utf-8")
    items = load_items(tmp_path)
    assert {it.id for it in items} == {"ok-1", "ok-2"}


# --------------------------------------------------------------------------- #
# Known-groups — clean PASS
# --------------------------------------------------------------------------- #


def test_known_groups_passes_clean_outcomes() -> None:
    items = load_default()
    outcomes = _clean_outcomes(items)
    result = check_known_groups(items, outcomes)
    assert isinstance(result, KnownGroupsResult)
    assert result.passed, result.violations
    assert result.violations == []
    # every category got exercised
    for cat in CalibrationCategory:
        assert result.by_category[cat.value]["checked"] >= 1


# --------------------------------------------------------------------------- #
# Known-groups — each law goes RED with the right violation
# --------------------------------------------------------------------------- #


def test_known_trivial_failure_is_violation() -> None:
    items = load_default()
    outcomes = _clean_outcomes(items)
    target = _first(items, CalibrationCategory.KNOWN_TRIVIAL)
    # make a trivial item fail on one tier -> instrument fault
    outcomes[target.id]["weak"] = False

    result = check_known_groups(items, outcomes)
    assert not result.passed
    assert any(
        "known_trivial" in v and target.id in v and "FAILED" in v
        for v in result.violations
    ), result.violations
    assert result.by_category["known_trivial"]["violations"] == 1


def test_known_impossible_pass_is_violation() -> None:
    items = load_default()
    outcomes = _clean_outcomes(items)
    target = _first(items, CalibrationCategory.KNOWN_IMPOSSIBLE)
    # make an impossible item pass -> leakage/gaming
    outcomes[target.id]["strong"] = True

    result = check_known_groups(items, outcomes)
    assert not result.passed
    assert any(
        "known_impossible" in v and target.id in v and "PASSED" in v
        for v in result.violations
    ), result.violations


def test_known_diagnostic_non_monotone_is_violation() -> None:
    items = load_default()
    outcomes = _clean_outcomes(items)
    target = _first(items, CalibrationCategory.KNOWN_DIAGNOSTIC)
    # weak passes, strong fails -> non-monotone (inversion)
    outcomes[target.id]["weak"] = True
    outcomes[target.id]["strong"] = False

    result = check_known_groups(items, outcomes)
    assert not result.passed
    assert any(
        "known_diagnostic" in v and target.id in v and "non-monotone" in v
        for v in result.violations
    ), result.violations


def test_missing_outcomes_is_violation() -> None:
    items = load_default()
    outcomes = _clean_outcomes(items)
    target = items[0]
    del outcomes[target.id]  # never exercised

    result = check_known_groups(items, outcomes)
    assert not result.passed
    assert any(target.id in v and "no recorded outcomes" in v for v in result.violations)


def test_empty_tier_map_is_violation() -> None:
    items = load_default()
    outcomes = _clean_outcomes(items)
    target = items[0]
    outcomes[target.id] = {}

    result = check_known_groups(items, outcomes)
    assert not result.passed
    assert any(target.id in v and "empty tier map" in v for v in result.violations)


# --------------------------------------------------------------------------- #
# Known-groups — anchor softness + monotone direction sanity
# --------------------------------------------------------------------------- #


def test_difficulty_anchor_inversion_is_soft_by_default() -> None:
    """A non-monotone anchor is a NOTE by default (§11.3 soft), not a hard fail."""
    items = load_default()
    outcomes = _clean_outcomes(items)
    target = _first(items, CalibrationCategory.DIFFICULTY_ANCHOR)
    outcomes[target.id]["weak"] = True
    outcomes[target.id]["strong"] = False

    soft = check_known_groups(items, outcomes)
    assert soft.passed, "anchor inversion must not hard-fail by default"
    assert any("SOFT" in n for n in soft.by_category["difficulty_anchor"]["notes"])

    strict = check_known_groups(items, outcomes, strict_anchors=True)
    assert not strict.passed
    assert any(
        "difficulty_anchor" in v and target.id in v and "non-monotone" in v
        for v in strict.violations
    )


def test_diagnostic_monotone_equal_pass_is_ok() -> None:
    """Stronger tier passing the SAME items as weaker (both pass) is monotone-ok."""
    items = load_default()
    outcomes = _clean_outcomes(items)
    target = _first(items, CalibrationCategory.KNOWN_DIAGNOSTIC)
    outcomes[target.id] = {"weak": True, "strong": True}  # no inversion
    result = check_known_groups(items, outcomes)
    assert result.passed, result.violations


def test_explicit_tier_order_is_honored() -> None:
    """Custom tier names work when an explicit weakest->strongest order is given."""
    item = CalibrationItem(
        id="diag-custom",
        category=CalibrationCategory.KNOWN_DIAGNOSTIC,
        construct="c",
        confound_controlled="cc",
        prompt="p",
        gold="g",
    )
    outcomes = {"diag-custom": {"novice": True, "expert": False}}  # inversion
    result = check_known_groups([item], outcomes, tier_order=["novice", "expert"])
    assert not result.passed
    assert any("non-monotone" in v for v in result.violations)


def test_test_retest_is_noted_not_failed() -> None:
    items = load_default()
    outcomes = _clean_outcomes(items)
    result = check_known_groups(items, outcomes)
    notes = result.by_category["test_retest"]["notes"]
    assert any("measured elsewhere" in n for n in notes)


# --------------------------------------------------------------------------- #
# admission_pairs.json — the discriminating PAIR-SET (§12, Q1) loads + validates
# --------------------------------------------------------------------------- #


def test_admission_pairs_load_via_loader() -> None:
    """The pair-set round-trips through the existing loader into CalibrationItems."""
    items = load_items(ADMISSION_PAIRS)
    assert all(isinstance(it, CalibrationItem) for it in items)
    # §12: plausible-vs-subtly-wrong pairs (replaces the saturating 20-item set). The
    # Fork-A expansion (Phase-2 swarm, 2026-06) grew the base 51 by 42 NON-arithmetic
    # discriminators across 6 new construct domains — the set saturated at the top for
    # strong judges (IRT kept only 15/51), so more medium/hard items with a real tail.
    assert 80 <= len(items) <= 150
    assert len({it.id for it in items}) == len(items), "pair ids must be unique"


def test_admission_pairs_gold_is_a_or_b() -> None:
    """Every pair's gold is the correct CHOICE — 'A' or 'B' (JudgeBench pair form)."""
    items = load_items(ADMISSION_PAIRS)
    assert {it.gold for it in items} == {"A", "B"}
    # both sides are used as the correct one (so position can't be learned as skill).
    golds = [it.gold for it in items]
    assert golds.count("A") >= 1 and golds.count("B") >= 1


def test_admission_pairs_categories_are_diagnostic_or_anchor() -> None:
    """§12: items live in known_diagnostic / difficulty_anchor."""
    items = load_items(ADMISSION_PAIRS)
    cats = {it.category for it in items}
    assert cats <= {
        CalibrationCategory.KNOWN_DIAGNOSTIC,
        CalibrationCategory.DIFFICULTY_ANCHOR,
    }
    # the discriminating core is diagnostic; anchors give the easy/hard tails.
    assert CalibrationCategory.KNOWN_DIAGNOSTIC in cats


def test_admission_pairs_present_two_candidates_and_ask_for_a_letter() -> None:
    """Each prompt shows BOTH candidates and asks for exactly one letter (A/B)."""
    items = load_items(ADMISSION_PAIRS)
    for it in items:
        assert "Candidate A:" in it.prompt, it.id
        assert "Candidate B:" in it.prompt, it.id
        assert "A or B" in it.prompt, it.id


def test_admission_pairs_span_a_difficulty_range() -> None:
    """§12: target a SPREAD (PSN-IRT), not a pile of traps — near-obvious to subtle."""
    items = load_items(ADMISSION_PAIRS)
    diffs = [it.difficulty for it in items if it.difficulty is not None]
    assert len(diffs) >= len(items) - 1, "items should carry a difficulty hint"
    assert min(diffs) <= 0.2, "needs a near-obvious tail"
    assert max(diffs) >= 0.7, "needs a subtle tail"
    # centred in the discriminating band (~0.5–0.65 per §12), not bunched at an extreme.
    assert 0.4 <= (sum(diffs) / len(diffs)) <= 0.65


def test_admission_pairs_known_groups_clean() -> None:
    """The pair-set passes the known-groups laws on a monotone outcome set (instrument
    valid): strong passes the discriminating items, weak passes only the easy ones."""
    items = load_items(ADMISSION_PAIRS)
    outcomes = {
        it.id: {
            "weak": (it.difficulty is not None and it.difficulty < 0.4),
            "strong": True,
        }
        for it in items
    }
    result = check_known_groups(items, outcomes)
    assert result.passed, result.violations


# --------------------------------------------------------------------------- #
# IRT discrimination screen (§12) — point_biserial math
# --------------------------------------------------------------------------- #


def test_point_biserial_perfect_discrimination() -> None:
    """Right-on-item perfectly ordered with ability → r_pb near +1."""
    # weak two fail, strong two pass; totals increasing with ability.
    r = point_biserial([0, 0, 1, 1], [1.0, 2.0, 3.0, 4.0])
    assert r == pytest.approx(0.894, abs=1e-3)
    assert r > 0.8


def test_point_biserial_anti_discrimination_is_negative() -> None:
    """An item the WEAK models pass and strong fail → negative discrimination."""
    r = point_biserial([1, 1, 0, 0], [1.0, 2.0, 3.0, 4.0])
    assert r < -0.8


def test_point_biserial_no_item_variance_is_zero() -> None:
    """All-correct (or all-wrong) item: discrimination undefined → reported 0.0."""
    assert point_biserial([1, 1, 1, 1], [1.0, 2.0, 3.0, 4.0]) == 0.0
    assert point_biserial([0, 0, 0, 0], [1.0, 2.0, 3.0, 4.0]) == 0.0


def test_point_biserial_no_total_variance_is_zero() -> None:
    """Flat ability (all totals equal): nothing to correlate against → 0.0."""
    assert point_biserial([0, 1, 0, 1], [2.0, 2.0, 2.0, 2.0]) == 0.0


def test_point_biserial_length_mismatch_raises() -> None:
    with pytest.raises(IRTError, match="LENGTH_MISMATCH"):
        point_biserial([1, 0, 1], [1.0, 2.0])


def test_point_biserial_single_respondent_is_zero() -> None:
    """Fewer than two respondents can't define a correlation → 0.0 (not a crash)."""
    assert point_biserial([1], [3.0]) == 0.0


# --------------------------------------------------------------------------- #
# IRT discrimination screen (§12) — prune_items drops saturated + flat, keeps disc
# --------------------------------------------------------------------------- #


def _ladder_matrix() -> dict[str, dict[str, bool]]:
    """A 6-model response matrix with a clean latent-ability ladder.

    Five Guttman-style ``rung*`` items establish a real ability gradient (m1 weakest →
    m6 strongest) so the corrected item-total is a valid ability proxy. Three probe items
    exercise each prune branch:

    * ``sat``  — every model correct → zero verdict variance (SATURATED) → DROP.
    * ``flat`` — alternating right/wrong, uncorrelated with ability (r_pb<0.1) → DROP.
    * ``disc`` — only the strong half pass → tracks ability (high r_pb) → KEEP.
    """
    models = ["m1", "m2", "m3", "m4", "m5", "m6"]
    columns = {
        "rung1": [True, True, True, True, True, False],
        "rung2": [False, True, True, True, True, True],
        "rung3": [False, False, True, True, True, True],
        "rung4": [False, False, False, True, True, True],
        "rung5": [False, False, False, False, True, True],
        "sat": [True, True, True, True, True, True],
        "flat": [True, False, True, False, True, False],
        "disc": [False, False, False, True, True, True],
    }
    return {m: {item: col[i] for item, col in columns.items()} for i, m in enumerate(models)}


def test_prune_drops_saturated_item() -> None:
    """A saturated (all-correct) item has zero variance and is dropped (§12 failure)."""
    kept, dropped = prune_items(_ladder_matrix())
    assert "sat" in dropped
    assert "sat" not in kept


def test_prune_drops_zero_discrimination_item() -> None:
    """An item uncorrelated with ability (point-biserial below floor) is dropped."""
    kept, dropped = prune_items(_ladder_matrix())
    assert "flat" in dropped
    assert "flat" not in kept


def test_prune_keeps_discriminating_item() -> None:
    """An item that tracks ability (high point-biserial) is kept."""
    kept, dropped = prune_items(_ladder_matrix())
    assert "disc" in kept
    assert "disc" not in dropped


def test_prune_partition_is_complete_and_disjoint() -> None:
    """kept ∪ dropped = all items, and the two are disjoint."""
    matrix = _ladder_matrix()
    kept, dropped = prune_items(matrix)
    all_items = set(next(iter(matrix.values())))
    assert set(kept) | set(dropped) == all_items
    assert set(kept).isdisjoint(dropped)


def test_prune_min_variance_threshold_prunes_near_saturated() -> None:
    """Raising min_variance prunes a barely-split item (1/6 wrong, var≈0.139)."""
    matrix = _ladder_matrix()
    # rung1 is wrong for exactly one model → low but non-zero variance.
    lenient_kept, _ = prune_items(matrix, min_variance=0.0, min_point_biserial=-1.0)
    strict_kept, strict_dropped = prune_items(
        matrix, min_variance=0.15, min_point_biserial=-1.0
    )
    assert "rung1" in lenient_kept
    assert "rung1" in strict_dropped
    assert "rung1" not in strict_kept


def test_prune_point_biserial_threshold_is_respected() -> None:
    """A high min_point_biserial drops a weakly-discriminating but-variable item."""
    matrix = _ladder_matrix()
    # With an impossibly high r_pb floor, even 'disc' fails the discrimination bar.
    kept, dropped = prune_items(matrix, min_variance=0.0, min_point_biserial=0.99)
    assert "disc" in dropped


def test_prune_empty_matrix_raises() -> None:
    with pytest.raises(IRTError, match="EMPTY_MATRIX"):
        prune_items({})


def test_prune_ragged_matrix_raises() -> None:
    """Models scored on different item sets is a structural error, not a silent skip."""
    with pytest.raises(IRTError, match="RAGGED_MATRIX"):
        prune_items({"a": {"i1": True, "i2": False}, "b": {"i1": True, "i3": False}})


def test_prune_items_no_items_raises() -> None:
    with pytest.raises(IRTError, match="EMPTY_ITEMS"):
        prune_items({"a": {}, "b": {}})


# --------------------------------------------------------------------------- #
# IRT — Bayesian path is a lazy optional (py-irt); raises a clear NotImplementedError
# --------------------------------------------------------------------------- #


def test_fit_irt_bayesian_raises_not_implemented() -> None:
    """py-irt is not a hard dep; the Bayesian path documents itself via NotImplementedError
    (§12 secondary screen; the model-free prune is primary)."""
    with pytest.raises(NotImplementedError, match="py-irt"):
        fit_irt_bayesian(_ladder_matrix())


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _min_record(item_id: str = "x-0001") -> dict:
    return {
        "id": item_id,
        "category": "known_trivial",
        "construct": "harness wiring",
        "confound_controlled": "difficulty",
        "prompt": "What is 2 + 2?",
        "gold": "4",
        "expected_pass": {"weak": True, "strong": True},
    }


def _first(items: list[CalibrationItem], category: CalibrationCategory) -> CalibrationItem:
    for it in items:
        if it.category is category:
            return it
    raise AssertionError(f"no item with category {category}")


def _clean_outcomes(items: list[CalibrationItem]) -> dict[str, dict[str, bool]]:
    """Build an outcome set that satisfies every known-groups law.

    - trivial: all tiers pass
    - impossible: no tier passes
    - diagnostic/anchor: monotone (weak fails, strong passes; medium passes if used)
    - test_retest: anything (skipped) — use all-pass for realism
    """
    out: dict[str, dict[str, bool]] = {}
    for it in items:
        if it.category is CalibrationCategory.KNOWN_TRIVIAL:
            out[it.id] = {"weak": True, "medium": True, "strong": True}
        elif it.category is CalibrationCategory.KNOWN_IMPOSSIBLE:
            out[it.id] = {"weak": False, "medium": False, "strong": False}
        elif it.category in (
            CalibrationCategory.KNOWN_DIAGNOSTIC,
            CalibrationCategory.DIFFICULTY_ANCHOR,
        ):
            out[it.id] = {"weak": False, "medium": True, "strong": True}
        else:  # TEST_RETEST
            out[it.id] = {"weak": True, "strong": True}
    return out
