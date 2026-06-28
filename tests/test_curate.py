"""Tests for the harder-set curation pipeline (study-swarm Phase A — calibration.curate + CLI).

Covers the two steps the design specifies: the AFLite/§12 discrimination screen (keep what strong
judges DISAGREE on, drop the saturated) and the ambiguity gate (a hard item must keep a DEFENSIBLE
key — verifier disagreement = ambiguous, verifiers-agree-but-differ-from-gold = mislabeled), plus
the offline `ai-crucible calibration curate` subcommand.
"""

from __future__ import annotations

import json

from ai_crucible import cli
from ai_crucible.calibration.curate import (
    AmbiguityStatus,
    ambiguity_gate,
    curate,
    select_discriminators,
)
from ai_crucible.calibration.types import CalibrationCategory, CalibrationItem

# A synthetic panel grade-matrix: `sat` is saturated (every model correct -> zero variance), and
# d1/d2/d3 discriminate (the panel splits, tracking mA>mB>mC ability) — the same shape the IRT
# prune test uses, so the discrimination screen keeps d1/d2/d3 and drops `sat`.
_GM = {
    "mA": {"sat": True, "d1": True, "d2": True, "d3": True},
    "mB": {"sat": True, "d1": True, "d2": False, "d3": False},
    "mC": {"sat": True, "d1": False, "d2": False, "d3": False},
}

# The same panel, but mC is MISSING `d3` — a per-item salvage drop (a cloud-run timeout). The
# persisted grade-matrix is then RAGGED, the shape the live-ρ demo hit: a single timeout crashed
# the whole curation (`IRT_RAGGED_MATRIX`). The screen must degrade to the shared {sat,d1,d2}
# subset and REPORT `d3` as ragged-dropped, mirroring the run's own irt_prune degradation.
_GM_RAGGED = {
    "mA": {"sat": True, "d1": True, "d2": True, "d3": True},
    "mB": {"sat": True, "d1": True, "d2": False, "d3": False},
    "mC": {"sat": True, "d1": False, "d2": False},  # `d3` missing — per-item salvage drop
}


def _item(iid: str, gold: str = "A") -> CalibrationItem:
    return CalibrationItem(
        id=iid,
        category=CalibrationCategory.KNOWN_DIAGNOSTIC,
        construct="c",
        confound_controlled="x",
        prompt="p",
        gold=gold,
    )


# --------------------------------------------------------------------------- #
# Ambiguity gate — the defensible-key check (GPQA difficulty-vs-ambiguity)
# --------------------------------------------------------------------------- #


def test_ambiguity_gate_defensible_when_verifiers_agree_with_gold() -> None:
    v = ambiguity_gate(_item("x", gold="A"), ["A", "a", " A "])  # case/space-insensitive
    assert v.status is AmbiguityStatus.DEFENSIBLE


def test_ambiguity_gate_ambiguous_on_verifier_disagreement() -> None:
    """Verifiers split → the disagreement is the ITEM's (no defensible key), not a judge's
    difficulty → omit. This is the crown-jewel constraint: hard must not mean ambiguous."""
    v = ambiguity_gate(_item("x", gold="A"), ["A", "B"])
    assert v.status is AmbiguityStatus.AMBIGUOUS


def test_ambiguity_gate_mislabeled_when_verifiers_agree_against_gold() -> None:
    v = ambiguity_gate(_item("x", gold="A"), ["B", "B", "B"])
    assert v.status is AmbiguityStatus.MISLABELED


def test_ambiguity_gate_not_verified_under_two_choices() -> None:
    v = ambiguity_gate(_item("x", gold="A"), ["A"])
    assert v.status is AmbiguityStatus.NOT_VERIFIED


# --------------------------------------------------------------------------- #
# Discrimination selection — AFLite/§12 (drop saturated, keep discriminators)
# --------------------------------------------------------------------------- #


def test_select_discriminators_drops_saturated_keeps_discriminating() -> None:
    kept, saturated, low_disc, ragged = select_discriminators(_GM)
    assert "sat" in saturated            # every model agreed → no discrimination
    assert "sat" not in kept
    assert set(kept) == {"d1", "d2", "d3"}
    assert ragged == []                  # a clean (non-ragged) matrix drops nothing for raggedness
    # the partition is complete + disjoint
    assert len(kept) + len(saturated) + len(low_disc) == 4
    assert not (set(kept) & set(saturated))


def test_select_discriminators_degrades_ragged_matrix() -> None:
    """A per-item salvage drop (a cloud-run timeout) leaves one model missing an item — a RAGGED
    grade-matrix. The forward screen must degrade to the shared item subset and REPORT the ragged
    drop, not crash with IRT_RAGGED_MATRIX (the live-ρ demo, 2026-06-28, caught the crash; the run's
    own irt_prune_report already degrades the same way)."""
    kept, saturated, low_disc, ragged = select_discriminators(_GM_RAGGED)
    assert ragged == ["d3"]                                 # surfaced, not silently dropped
    assert "sat" in saturated                               # screened on the shared {sat,d1,d2}
    assert "d3" not in kept and "d3" not in saturated and "d3" not in low_disc
    assert set(kept) <= {"d1", "d2"}


def test_shared_item_matrix_restricts_and_reports_ragged() -> None:
    from ai_crucible.calibration import irt

    restricted, ragged = irt.shared_item_matrix(_GM_RAGGED)
    assert ragged == ["d3"]
    assert all(set(row) == {"sat", "d1", "d2"} for row in restricted.values())
    assert irt.shared_item_matrix({}) == ({}, [])           # empty → no crash


# --------------------------------------------------------------------------- #
# curate() — the full Phase-A pipeline
# --------------------------------------------------------------------------- #


def test_curate_pipeline_filters_then_gates() -> None:
    items = [_item("sat"), _item("d1", "A"), _item("d2", "A"), _item("d3", "A")]
    verdicts = {
        "d1": ["A", "A"],   # verifiers agree with gold  → defensible → kept
        "d2": ["A", "B"],   # verifiers disagree         → ambiguous  → dropped
        "d3": ["B", "B"],   # verifiers agree != gold    → mislabeled → dropped
    }
    res = curate(items, _GM, ambiguity_verdicts=verdicts)
    assert "sat" in res.dropped_saturated and "sat" not in res.kept
    assert "d1" in res.kept
    assert "d2" in res.dropped_ambiguous and "d2" not in res.kept
    assert "d3" in res.dropped_mislabeled and "d3" not in res.kept


def test_curate_without_verdicts_keeps_discriminators_flagged_unverified() -> None:
    """No verifier verdicts → the gate is NOT silently skipped: survivors are kept but reported
    `not_verified` (the absence of the ambiguity check is surfaced, not hidden)."""
    items = [_item("sat"), _item("d1"), _item("d2"), _item("d3")]
    res = curate(items, _GM)
    assert "sat" in res.dropped_saturated
    assert set(res.not_verified) == {"d1", "d2", "d3"}
    assert set(res.kept) == {"d1", "d2", "d3"}
    assert not res.dropped_ambiguous and not res.dropped_mislabeled


def test_curate_reports_ragged_drops_instead_of_crashing() -> None:
    """A ragged grade-matrix (one model timed out on an item) degrades to the shared subset and
    surfaces the ragged drop on the result — it does not crash the whole curation."""
    items = [_item("sat"), _item("d1"), _item("d2"), _item("d3")]
    res = curate(items, _GM_RAGGED)
    assert res.dropped_ragged == ["d3"]
    assert "d3" not in res.kept
    assert "sat" in res.dropped_saturated
    assert set(res.kept) == {"d1", "d2"}


# --------------------------------------------------------------------------- #
# CLI — `ai-crucible calibration curate` (offline, no model)
# --------------------------------------------------------------------------- #


def _write_pool(tmp_path):
    """A 4-item pool in the admission_pairs.json shape (a flat list keyed by id)."""
    pool = [
        {"id": iid, "category": "known_diagnostic", "construct": "c",
         "confound_controlled": "x", "prompt": "p", "gold": "A",
         "difficulty": 0.5, "expected_pass": {}, "metadata": {}}
        for iid in ("sat", "d1", "d2", "d3")
    ]
    p = tmp_path / "pool.json"
    p.write_text(json.dumps(pool), encoding="utf-8")
    return p


def test_cli_calibration_curate_writes_discriminating_subset(tmp_path, capsys) -> None:
    pool = _write_pool(tmp_path)
    report = tmp_path / "report.json"
    report.write_text(json.dumps({"grade_matrix": _GM}), encoding="utf-8")
    out = tmp_path / "curated.json"

    rc = cli.main(
        ["calibration", "curate", "--from-run", str(report),
         "--items", str(pool), "--out", str(out)]
    )
    assert rc == 0
    data = json.loads(capsys.readouterr().out)           # machine JSON on STDOUT
    assert data["ok"] is True
    assert data["n_kept"] == 3 and set(data["kept"]) == {"d1", "d2", "d3"}
    assert data["n_dropped_saturated"] == 1 and data["dropped_saturated"] == ["sat"]
    assert data["ambiguity_gate"] == "not_run_offline"
    # the written subset is the raw pool filtered to the kept ids (re-usable as --items)
    subset = json.loads(out.read_text(encoding="utf-8"))
    assert {it["id"] for it in subset} == {"d1", "d2", "d3"}


def test_cli_calibration_curate_structured_error_on_report_without_grade_matrix(
    tmp_path, capsys
) -> None:
    pool = _write_pool(tmp_path)
    report = tmp_path / "old_report.json"
    report.write_text(json.dumps({"n_items": 4}), encoding="utf-8")  # predates grade_matrix
    rc = cli.main(["calibration", "curate", "--from-run", str(report), "--items", str(pool)])
    assert rc == 1
    err = capsys.readouterr().err
    assert "[CALIBRATION_NO_GRADE_MATRIX]" in err
    assert "Traceback" not in err  # structured one-liner, not a raw stack


def test_cli_calibration_requires_subcommand(capsys) -> None:
    assert cli.main(["calibration"]) == 2
    assert "subcommand is required" in capsys.readouterr().err
