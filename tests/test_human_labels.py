"""Fork C — human alt-test harness tests (research-grounding §12.1).

Covers the four pieces that retire the circular model-jury ω:

* :func:`crucible.characterize.metrics.alt_test` — the audit-ready ω (ε + paired t-test +
  Benjamini-Yekutieli FDR against human annotators) and :func:`krippendorff_alpha` (sparse IAA);
* :func:`crucible.scoring.stats.conformal_coverage_interval` — the honest small-N coverage spread;
* :mod:`crucible.characterize.human_labels` — loading/validating ``human_labels.json``;
* the end-to-end human-grounded :func:`build_profile` path.

These are synthetic-label tests (the kickoff's Fork-C verify criterion: "ω computed against
humans on synthetic-human-label tests"). No model or GPU is touched.
"""

from __future__ import annotations

import json

import pytest

from crucible.calibration.types import CalibrationCategory, CalibrationItem
from crucible.characterize.human_labels import (
    HumanLabelError,
    build_records_per_annotator,
    load_human_labels,
)
from crucible.characterize.metrics import alt_test, krippendorff_alpha
from crucible.characterize.profile import build_profile
from crucible.characterize.types import JudgmentRecord, RoleSlot, SeatDecision
from crucible.scoring.stats import conformal_coverage_interval


def _rec(item: str, pred: int, ann: str, gold: int | None = None) -> JudgmentRecord:
    return JudgmentRecord(item_id=item, model_id=ann, predicted=pred,
                          gold=pred if gold is None else gold)


def _items(n: int) -> list[CalibrationItem]:
    """``n`` A/B calibration items, gold alternating A/B (id = ``i0``..)."""
    return [
        CalibrationItem(
            id=f"i{i}",
            category=CalibrationCategory.KNOWN_DIAGNOSTIC,
            construct="t",
            confound_controlled="t",
            prompt="p",
            gold="A" if i % 2 == 0 else "B",
        )
        for i in range(n)
    ]


# --------------------------------------------------------------------------- #
# krippendorff_alpha
# --------------------------------------------------------------------------- #


def test_krippendorff_perfect_agreement_is_one() -> None:
    rpa = {a: [_rec(f"i{i}", i % 2, a) for i in range(6)] for a in ("h1", "h2", "h3")}
    assert krippendorff_alpha(rpa) == pytest.approx(1.0)


def test_krippendorff_hand_case_is_zero() -> None:
    """2 items × 2 coders: one item agreed, one split → α = 0 (hand-computed)."""
    rpa = {
        "h1": [_rec("i1", 1, "h1"), _rec("i2", 1, "h1")],
        "h2": [_rec("i1", 1, "h2"), _rec("i2", 0, "h2")],
    }
    assert krippendorff_alpha(rpa) == pytest.approx(0.0)


def test_krippendorff_excludes_judge_key() -> None:
    """The reserved 'judge' key is NOT a human and must not enter the IAA."""
    rpa = {
        "judge": [_rec(f"i{i}", 1, "judge") for i in range(4)],  # would skew it
        "h1": [_rec(f"i{i}", i % 2, "h1") for i in range(4)],
        "h2": [_rec(f"i{i}", i % 2, "h2") for i in range(4)],
    }
    assert krippendorff_alpha(rpa) == pytest.approx(1.0)  # the two humans agree perfectly


def test_krippendorff_single_value_degenerate_is_one() -> None:
    rpa = {a: [_rec("i0", 1, a)] for a in ("h1", "h2", "h3")}  # 1 item, all same
    assert krippendorff_alpha(rpa) == 1.0


# --------------------------------------------------------------------------- #
# alt_test (the audit-ready ω)
# --------------------------------------------------------------------------- #


def _unanimous_humans(n: int, names=("h1", "h2", "h3", "h4")) -> dict:
    gold = [i % 2 for i in range(n)]
    return {a: [_rec(f"i{i}", gold[i], a) for i in range(n)] for a in names}


def test_alt_test_seats_judge_matching_human_consensus() -> None:
    humans = _unanimous_humans(40)
    judge = [_rec(f"i{i}", i % 2, "judge") for i in range(40)]
    omega = alt_test({"judge": judge, **humans}, epsilon=0.2)
    assert omega == pytest.approx(1.0)


def test_alt_test_rejects_always_wrong_judge() -> None:
    humans = _unanimous_humans(40)
    judge = [_rec(f"i{i}", 1 - (i % 2), "judge") for i in range(40)]
    omega = alt_test({"judge": judge, **humans}, epsilon=0.2)
    assert omega == 0.0


def test_alt_test_epsilon_gives_benefit_of_doubt_on_ties() -> None:
    """A judge that exactly ties the humans wins every fold once ε > 0 (no-worse-than)."""
    humans = _unanimous_humans(40)
    judge = [_rec(f"i{i}", i % 2, "judge") for i in range(40)]  # identical to humans
    assert alt_test({"judge": judge, **humans}, epsilon=0.1) == pytest.approx(1.0)


def test_alt_test_requires_three_humans() -> None:
    humans = _unanimous_humans(30, names=("h1", "h2"))
    judge = [_rec(f"i{i}", i % 2, "judge") for i in range(30)]
    with pytest.raises(ValueError, match="3 human annotators"):
        alt_test({"judge": judge, **humans}, epsilon=0.2)


def test_alt_test_requires_judge_key_and_valid_epsilon() -> None:
    humans = _unanimous_humans(30)
    with pytest.raises(ValueError, match="'judge' entry"):
        alt_test(humans, epsilon=0.2)
    judge = [_rec(f"i{i}", i % 2, "judge") for i in range(30)]
    with pytest.raises(ValueError, match="epsilon"):
        alt_test({"judge": judge, **humans}, epsilon=1.5)


def test_alt_test_exclude_items_drops_them_from_folds() -> None:
    humans = _unanimous_humans(40)
    judge = [_rec(f"i{i}", i % 2, "judge") for i in range(40)]
    # excluding a couple items still yields a valid ω (fewer per-fold points).
    omega = alt_test({"judge": judge, **humans}, epsilon=0.2, exclude_items={"i0", "i1"})
    assert 0.0 <= omega <= 1.0


# --------------------------------------------------------------------------- #
# conformal_coverage_interval
# --------------------------------------------------------------------------- #


def test_conformal_interval_tightens_with_n() -> None:
    lo_small, hi_small, _ = conformal_coverage_interval(30, 0.1)
    lo_big, hi_big, _ = conformal_coverage_interval(1000, 0.1)
    assert (hi_small - lo_small) > (hi_big - lo_big)  # small N is wider
    assert lo_small < 0.9 < hi_small  # nominal 90% sits inside the realized band


def test_conformal_mean_in_known_band() -> None:
    """Mean realized coverage ∈ [1−α, 1−α + 1/(n+1)] (Angelopoulos-Bates §3.2)."""
    n, alpha = 50, 0.1
    _, _, mean = conformal_coverage_interval(n, alpha)
    assert (1 - alpha) <= mean <= (1 - alpha) + 1.0 / (n + 1) + 1e-9


def test_conformal_degenerate_when_alpha_too_small_for_n() -> None:
    assert conformal_coverage_interval(5, 0.01) == (1.0, 1.0, 1.0)  # l = floor(6*0.01)=0


def test_conformal_validates_args() -> None:
    for bad in (lambda: conformal_coverage_interval(0, 0.1),
                lambda: conformal_coverage_interval(10, 0.0),
                lambda: conformal_coverage_interval(10, 1.0),
                lambda: conformal_coverage_interval(10, 0.1, conf=1.0)):
        with pytest.raises(ValueError):
            bad()


# --------------------------------------------------------------------------- #
# load_human_labels
# --------------------------------------------------------------------------- #


def _write(tmp_path, obj) -> object:
    p = tmp_path / "human_labels.json"
    p.write_text(json.dumps(obj), encoding="utf-8")
    return p


def _valid_payload(n: int = 32, tier: str = "expert") -> dict:
    annotators = {a: {"tier": tier} for a in ("h1", "h2", "h3")}
    labels = {
        f"i{i}": {"h1": ("A" if i % 2 == 0 else "B"),
                  "h2": ("A" if i % 2 == 0 else "B"),
                  "h3": ("A" if i % 2 == 0 else "B")}
        for i in range(n)
    }
    return {"schema_version": 1, "annotators": annotators, "labels": labels}


def test_load_human_labels_happy_path(tmp_path) -> None:
    hl = load_human_labels(_write(tmp_path, _valid_payload(32)), _items(32))
    assert hl.n_annotators == 3
    assert hl.n_items == 32
    assert hl.epsilon == pytest.approx(0.2)         # all-expert → ε 0.2
    assert hl.iaa_alpha == pytest.approx(1.0)        # unanimous humans
    assert hl.disputed == []
    assert set(hl.records_per_annotator) == {"h1", "h2", "h3"}


def test_load_human_labels_min_tier_epsilon(tmp_path) -> None:
    payload = _valid_payload(32)
    payload["annotators"]["h3"]["tier"] = "crowd"  # mixed → most conservative ε
    hl = load_human_labels(_write(tmp_path, payload), _items(32))
    assert hl.epsilon == pytest.approx(0.1)


def test_load_human_labels_flags_disputed_and_unsure(tmp_path) -> None:
    payload = _valid_payload(32)
    # 4th annotator, and make i0 a 2-2 split (disputed); 'unsure' on i1 is dropped.
    payload["annotators"]["h4"] = {"tier": "expert"}
    payload["labels"]["i0"] = {"h1": "A", "h2": "A", "h3": "B", "h4": "B"}
    payload["labels"]["i1"] = {"h1": "unsure", "h2": "B", "h3": "B", "h4": "B"}
    hl = load_human_labels(_write(tmp_path, payload), _items(32))
    assert "i0" in hl.disputed
    # h1's 'unsure' on i1 produced no record for that (annotator, item).
    assert all(r.item_id != "i1" for r in hl.records_per_annotator["h1"])


def test_load_human_labels_low_iaa_clamps_epsilon(tmp_path) -> None:
    # Three experts that disagree heavily → low α → ε clamped to 0.10 + a note.
    annotators = {a: {"tier": "expert"} for a in ("h1", "h2", "h3")}
    labels = {
        f"i{i}": {"h1": "A", "h2": "B", "h3": ("A" if i % 2 else "B")}
        for i in range(32)
    }
    src = _write(tmp_path, {"annotators": annotators, "labels": labels})
    hl = load_human_labels(src, _items(32))
    assert hl.epsilon == pytest.approx(0.1)
    assert any("clamped" in n for n in hl.notes)


def test_load_human_labels_too_few_items_note(tmp_path) -> None:
    hl = load_human_labels(_write(tmp_path, _valid_payload(10)), _items(10))
    assert any("under-powered" in n or "< 30" in n or "30" in n for n in hl.notes)


def test_load_human_labels_errors(tmp_path) -> None:
    with pytest.raises(HumanLabelError, match="INPUT_PATH_MISSING"):
        load_human_labels(tmp_path / "nope.json", _items(5))

    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    with pytest.raises(HumanLabelError, match="INPUT_BAD_JSON"):
        load_human_labels(bad, _items(5))

    # unknown item id
    p = _valid_payload(3)
    p["labels"]["i999"] = {"h1": "A", "h2": "A", "h3": "A"}
    with pytest.raises(HumanLabelError, match="INPUT_UNKNOWN_ITEM"):
        load_human_labels(_write(tmp_path, p), _items(3))

    # unknown annotator
    p = _valid_payload(3)
    p["labels"]["i0"]["ghost"] = "A"
    with pytest.raises(HumanLabelError, match="UNKNOWN_ANNOTATOR"):
        load_human_labels(_write(tmp_path, p), _items(3))

    # bad tier
    p = _valid_payload(3)
    p["annotators"]["h1"]["tier"] = "wizard"
    with pytest.raises(HumanLabelError, match="UNKNOWN_TIER"):
        load_human_labels(_write(tmp_path, p), _items(3))

    # too few annotators with usable labels
    p = {"annotators": {"h1": {"tier": "expert"}, "h2": {"tier": "expert"}},
         "labels": {f"i{i}": {"h1": "A", "h2": "A"} for i in range(30)}}
    with pytest.raises(HumanLabelError, match="TOO_FEW_ANNOTATORS"):
        load_human_labels(_write(tmp_path, p), _items(30))


# --------------------------------------------------------------------------- #
# end-to-end: the human-grounded build_profile path
# --------------------------------------------------------------------------- #


def test_build_records_per_annotator_adds_judge_key(tmp_path) -> None:
    hl = load_human_labels(_write(tmp_path, _valid_payload(32)), _items(32))
    judge = [_rec(f"i{i}", i % 2, "judge") for i in range(32)]
    rpa = build_records_per_annotator(hl, judge)
    assert "judge" in rpa and set(rpa) == {"judge", "h1", "h2", "h3"}


def test_human_grounded_profile_computes_omega_against_humans(tmp_path) -> None:
    """A judge that matches the human consensus seats with ω=1.0 from the REAL alt-test."""
    hl = load_human_labels(_write(tmp_path, _valid_payload(40)), _items(40))
    # judge matches the human consensus: even items → "A" (_to_num=1), odd → "B" (=0).
    def consensus(i: int) -> int:
        return 1 if i % 2 == 0 else 0
    judge = [_rec(f"i{i}", consensus(i), "judge", gold=consensus(i)) for i in range(40)]
    profile = build_profile(
        "cand", RoleSlot.JUDGE, judge,
        records_per_annotator=build_records_per_annotator(hl, judge),
        human_grounded=True,
        alt_test_epsilon=hl.epsilon,
        alt_test_exclude=set(hl.disputed),
        human_human_kappa=hl.iaa_alpha,
    )
    assert profile.alt_test_omega == pytest.approx(1.0)
    assert profile.seat_decision in (SeatDecision.SEAT, SeatDecision.SCREEN)
    assert any("HUMAN-grounded" in n for n in profile.notes)


# --------------------------------------------------------------------------- #
# re-audit regressions (fork-c-stats MEDIUM/LOW, fork-c-integration INFO)
# --------------------------------------------------------------------------- #


def test_alt_test_raises_when_no_comparable_folds() -> None:
    """A judge sharing NO items with the humans must RAISE, not silently score ω=0.0."""
    humans = _unanimous_humans(30)  # items i0..i29
    judge = [_rec(f"j{i}", i % 2, "judge") for i in range(30)]  # disjoint ids
    with pytest.raises(ValueError, match="comparable"):
        alt_test({"judge": judge, **humans}, epsilon=0.2)


def test_alt_test_skips_degenerate_single_item_fold() -> None:
    """A fold sharing <2 items is skipped, not fed a phantom p=1.0 that would depress ω."""
    base = _unanimous_humans(30, names=("h1", "h2", "h3"))
    base["h_thin"] = [_rec("i0", 0, "h_thin")]  # only one label → its fold is degenerate
    judge = [_rec(f"i{i}", i % 2, "judge") for i in range(30)]  # matches consensus
    # Without the skip, h_thin's phantom fold inflates m+c(m) and ω < 1; with it, ω == 1.
    assert alt_test({"judge": judge, **base}, epsilon=0.2) == pytest.approx(1.0)


def test_conformal_floor_robust_to_float_representation() -> None:
    """n=89, α=0.7: (n+1)·α is exactly 63 but float64 gives 62.999…; floor must be 63."""
    _, _, mean = conformal_coverage_interval(89, 0.7)
    assert mean == pytest.approx(27 / 90)  # l=63 → a=27,b=63 → mean 0.3 (was 28/90 with int())


def test_load_human_labels_rejects_oov_verdict(tmp_path) -> None:
    """A typo'd verdict ('MAYBE') is a config error, never silently coerced to a side."""
    p = _valid_payload(30)
    p["labels"]["i0"]["h1"] = "MAYBE"
    with pytest.raises(HumanLabelError, match="BAD_VERDICT"):
        load_human_labels(_write(tmp_path, p), _items(30))
