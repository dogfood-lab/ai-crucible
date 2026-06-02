"""Unit tests for the characterization runner's pure helpers (no live model needed)."""

from __future__ import annotations

from crucible.calibration.types import CalibrationCategory, CalibrationItem
from crucible.characterize.run import (
    _jury,
    _parse_models,
    _to_num,
    irt_prune_report,
    known_groups_report,
    panel_correlation_report,
    parse_choice,
    parse_verdict,
)
from crucible.characterize.types import JudgmentRecord


def _rec(item_id: str, model_id: str, *, correct: bool, gold: int = 1) -> JudgmentRecord:
    """A graded record with predicted/gold/correct coherent (pred==gold iff correct)."""
    predicted = gold if correct else (1 - gold)
    return JudgmentRecord(
        item_id=item_id, model_id=model_id, predicted=predicted, gold=gold, correct=correct
    )


def test_parse_verdict() -> None:
    assert parse_verdict("PASS") == "PASS"
    assert parse_verdict("I think this is FAIL.") == "FAIL"
    assert parse_verdict("pass") == "PASS"
    assert parse_verdict("no verdict here") is None
    assert parse_verdict("") is None


def test_parse_models_handles_colon_in_model_id() -> None:
    # Regression: a model_id like "qwen3.6:27b" contains a ':'; split on the LAST '@'.
    specs = _parse_models(["qwen3.6:27b@qwen", "mistral-small:24b@mistral"])
    assert specs == [
        ("qwen3.6:27b", "qwen", None),
        ("mistral-small:24b", "mistral", None),
    ]


def test_parse_models_no_family() -> None:
    assert _parse_models(["llama3"]) == [("llama3", "unknown", None)]


def test_known_groups_report() -> None:
    items = [
        CalibrationItem(
            id="t1",
            category=CalibrationCategory.KNOWN_TRIVIAL,
            construct="x",
            confound_controlled="y",
            prompt="p",
            gold=1,
        )
    ]
    miss = {"m": [JudgmentRecord(item_id="t1", model_id="m", predicted=0, gold=1, correct=False)]}
    rep = known_groups_report(items, miss)
    assert rep["passed"] is False and rep["violations"]

    hit = {"m": [JudgmentRecord(item_id="t1", model_id="m", predicted=1, gold=1, correct=True)]}
    assert known_groups_report(items, hit)["passed"] is True


def test_parse_choice_ab_and_verdict_spaces() -> None:
    # A/B space (gold ∈ {A, B}): parse a standalone letter, case-sensitive.
    assert parse_choice("A", "A") == "A"
    assert parse_choice("The correct answer is B.", "B") == "B"
    assert parse_choice("Both look fine", "A") is None
    # a lowercase article must NOT be read as a choice
    assert parse_choice("a thing happened", "A") is None
    # PASS/FAIL space (any other gold) defers to the verdict parser.
    assert parse_choice("PASS", "PASS") == "PASS"
    assert parse_choice("the verdict is fail", "FAIL") == "FAIL"


def test_to_num_mapping() -> None:
    assert _to_num("PASS") == 1
    assert _to_num("A") == 1
    assert _to_num("FAIL") == 0
    assert _to_num("B") == 0
    assert _to_num(None) == 0


def test_known_groups_via_expected_pass_weak() -> None:
    # A difficulty_anchor item (no KNOWN_TRIVIAL category) is still a trivial anchor
    # when the weak tier is expected to pass — the pairs set's convention.
    items = [
        CalibrationItem(
            id="easy",
            category=CalibrationCategory.DIFFICULTY_ANCHOR,
            construct="x",
            confound_controlled="y",
            prompt="p",
            gold="A",
            expected_pass={"strong": True, "weak": True},
        )
    ]
    miss = {"m": [_rec("easy", "m", correct=False, gold=1)]}
    rep = known_groups_report(items, miss)
    assert rep["trivial_item_count"] == 1
    assert rep["passed"] is False and rep["violations"]


def test_panel_correlation_report_shape_fix() -> None:
    # Regression: the old glue passed dict[str, dict[str, int]] to
    # pairwise_error_correlation (which wants {judge: [JudgmentRecord]}), raising
    # "'str' object has no attribute 'predicted'". Pass records straight through now.
    records = {
        "mA": [_rec("i1", "mA", correct=True), _rec("i2", "mA", correct=True),
               _rec("i3", "mA", correct=False)],
        "mB": [_rec("i1", "mB", correct=True), _rec("i2", "mB", correct=False),
               _rec("i3", "mB", correct=False)],
    }
    rep = panel_correlation_report(records)
    assert "error" not in rep
    assert "pairwise_error_correlation" in rep
    assert "mA|mB" in rep["pairwise_error_correlation"]
    assert isinstance(rep["submodular"], bool)


def test_panel_correlation_single_judge_is_vacuous() -> None:
    rep = panel_correlation_report({"only": [_rec("i1", "only", correct=True)]})
    assert rep["submodular"] is True


def test_irt_prune_drops_saturated_item() -> None:
    # i_sat: every model correct → zero variance → dropped (the saturating-set failure).
    # The discriminating items give an ability spread so the screen has signal.
    records = {
        "mA": [_rec("i_sat", "mA", correct=True), _rec("d1", "mA", correct=True),
               _rec("d2", "mA", correct=True), _rec("d3", "mA", correct=True)],
        "mB": [_rec("i_sat", "mB", correct=True), _rec("d1", "mB", correct=True),
               _rec("d2", "mB", correct=False), _rec("d3", "mB", correct=False)],
        "mC": [_rec("i_sat", "mC", correct=True), _rec("d1", "mC", correct=False),
               _rec("d2", "mC", correct=False), _rec("d3", "mC", correct=False)],
    }
    rep = irt_prune_report(records)
    assert "error" not in rep
    assert "i_sat" in rep["dropped"]
    assert rep["n_items"] == 4


def test_jury_needs_two_peers() -> None:
    recs_a = [_rec("i1", "a", correct=True)]
    recs_b = [_rec("i1", "b", correct=True)]
    recs_c = [_rec("i1", "c", correct=True)]
    # <2 peers → None (alt-test leave-one-out needs ≥2 annotators).
    assert _jury({"a": recs_a, "b": recs_b}, "a", recs_a) is None
    # ≥2 peers → reserved "judge" key + the peers.
    jury = _jury({"a": recs_a, "b": recs_b, "c": recs_c}, "a", recs_a)
    assert jury is not None
    assert jury["judge"] is recs_a
    assert set(jury) == {"judge", "b", "c"}
