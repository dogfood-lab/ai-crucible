"""Unit tests for the characterization runner's pure helpers (no live model needed)."""

from __future__ import annotations

from ai_crucible.calibration.types import CalibrationCategory, CalibrationItem
from ai_crucible.characterize.run import (
    _grade_matrix,
    _jury,
    _parse_models,
    _to_num,
    irt_prune_report,
    known_groups_report,
    main,
    panel_composition_report,
    panel_correlation_report,
    parse_choice,
    parse_verdict,
)
from ai_crucible.characterize.types import JudgeProfile, JudgmentRecord, RoleSlot, SeatDecision


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


def test_parse_models_untagged_yields_none_not_unknown() -> None:
    """models-cli-001 (SHARED FAMILY CONTRACT): an untagged ``--models`` entry must stamp
    family=None, NOT the literal string "unknown".

    Per the cross-family seal, an untagged/unknown family is represented as Python None so
    judge_panel's None-handling applies (a None-family judge is admitted as cross-family on
    OPERATOR responsibility, never silently kept under a colliding "unknown" tag). The
    literal "unknown" is a genuine string value that collides with every other "unknown",
    so a mixed tagged/untagged SAME-family judge could survive exclusion against a generator
    also tagged "unknown". None never equals a concrete family, so the residual is honest.
    """
    assert _parse_models(["llama3"]) == [("llama3", None, None)]
    # mixed tagged/untagged in one call: the tagged one keeps its family, the untagged → None
    assert _parse_models(["qwen3.6:27b@qwen", "llama3"]) == [
        ("qwen3.6:27b", "qwen", None),
        ("llama3", None, None),
    ]
    # the colliding literal must not appear anywhere in the parsed families.
    fams = [fam for _mid, fam, _q in _parse_models(["a", "b", "c"])]
    assert "unknown" not in fams
    assert fams == [None, None, None]


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


def test_panel_composition_report_wiring() -> None:
    # Glue: profiles + records → asdict(SeatedPanel) in the report (seats as plain dicts).
    def _prof(m: str, w: float, dec: SeatDecision = SeatDecision.SEAT) -> JudgeProfile:
        return JudgeProfile(
            model_id=m, role=RoleSlot.JUDGE, n_items=4,
            reliability_weight=w, seat_decision=dec, metadata={},
        )

    profiles = {
        "a": _prof("a", 1.0), "b": _prof("b", 0.9), "c": _prof("c", 0.8),
        "z": _prof("z", 0.0, SeatDecision.REJECT),
    }
    records = {m: [_rec(f"i{j}", m, correct=True) for j in range(4)] for m in profiles}
    rep = panel_composition_report(profiles, records)
    assert "error" not in rep
    assert [s["model_id"] for s in rep["seats"]] == ["a", "b", "c"]
    assert rep["meets_quorum"] is True and rep["escalate"] is False
    assert rep["not_seated"] == ["z"]


# --------------------------------------------------------------------------- #
# calibration-instrument-003 — _grade_matrix tie-break is conservative (<0.5)
# --------------------------------------------------------------------------- #


def test_grade_matrix_even_split_tie_resolves_to_incorrect() -> None:
    """calibration-instrument-003: an exact 50/50 tie on an even number of reruns must
    resolve DETERMINISTICALLY to ``incorrect`` (conservative toward non-solve), NOT round up
    to ``correct``.

    The majority-collapsed boolean grid feeds prune_items (the IRT calibration screen).
    Rounding ties UP toward 'pass' was an optimistic tie-break that could classify a
    knife-edge item as all-pass (zero variance → dropped) or push it across the
    point-biserial floor, systematically biasing which calibration items survive. The
    instrument's item bank is the foundation of every later measurement, so the tie-break
    must be explicit and conservative: a fraction strictly GREATER than 0.5 is the only
    'correct' verdict; an exact tie is 'incorrect'.
    """
    # k=2, one pass one fail → fraction 0.5 → must collapse to False (incorrect).
    records = {
        "m": [
            _rec("tie", "m", correct=True),
            _rec("tie", "m", correct=False),
        ]
    }
    matrix = _grade_matrix(records)
    assert matrix["m"]["tie"] is False
    # Sanity: a clear majority pass (>0.5) is still correct; a clear majority fail is False.
    clear = {
        "m": [
            _rec("pass", "m", correct=True),
            _rec("pass", "m", correct=True),
            _rec("pass", "m", correct=False),  # 2/3 > 0.5 → correct
            _rec("fail", "m", correct=True),
            _rec("fail", "m", correct=False),
            _rec("fail", "m", correct=False),  # 1/3 < 0.5 → incorrect
        ]
    }
    cm = _grade_matrix(clear)
    assert cm["m"]["pass"] is True
    assert cm["m"]["fail"] is False


# --------------------------------------------------------------------------- #
# models-cli-003 — main() exits non-zero when zero judgments are collected
# --------------------------------------------------------------------------- #


def test_main_exits_nonzero_when_every_model_fails(tmp_path, monkeypatch, capsys) -> None:
    """models-cli-003: ``ai-crucible characterize`` must exit NON-ZERO (with a structured
    code/message/hint) when run_panel collects zero judgments — Ollama down / wrong host /
    every model tag unservable. A measurement instrument that measured NOTHING is a failure,
    not a green pass; an automation/CI gate that shells out and checks the exit code must
    not treat empty profiles as success (silent data loss presented as green)."""

    async def _empty_panel(*_a, **_k):
        # both profiles and all_records empty — the total-failure path.
        return {}, {}

    monkeypatch.setattr("ai_crucible.characterize.run.run_panel", _empty_panel)
    out_path = tmp_path / "report.json"
    code = main(["--out", str(out_path)])
    assert code != 0
    err = capsys.readouterr().err
    # a structured error shape (code/message/hint) — not a raw stack.
    assert "code" in err and "message" in err and "hint" in err


def test_main_exits_zero_on_a_real_profile(tmp_path, monkeypatch) -> None:
    """Contrast: when at least one profile is produced, main() returns 0 (the happy path
    is unchanged by the total-failure guard)."""
    prof = JudgeProfile(
        model_id="m", role=RoleSlot.JUDGE, n_items=4,
        reliability_weight=1.0, seat_decision=SeatDecision.SEAT, metadata={},
    )
    records = {"m": [_rec(f"i{j}", "m", correct=True) for j in range(4)]}

    async def _one_panel(*_a, **_k):
        return {"m": prof}, records

    monkeypatch.setattr("ai_crucible.characterize.run.run_panel", _one_panel)
    out_path = tmp_path / "report.json"
    assert main(["--out", str(out_path)]) == 0


# --------------------------------------------------------------------------- #
# characterize-002 — default (model-jury) report carries a machine-readable
#                    provisional caveat on the κ baseline
# --------------------------------------------------------------------------- #


def test_default_report_marks_kappa_baseline_provisional(tmp_path, monkeypatch) -> None:
    """characterize-002: the default (no --human-labels) path applies the κ one-sided floor
    + κ-z gate against a hardcoded human_human_kappa=0.80 with no humans in the loop. The
    emitted report must carry a MACHINE-READABLE flag (provisional=True /
    alt_test_reference='model-jury-bootstrap' + a kappa_baseline.provisional flag) so a
    consumer can SEE the κ seat/reject gate is provisional — without fabricating human
    labels. The human-grounded path must NOT carry the provisional flag."""
    prof = JudgeProfile(
        model_id="m", role=RoleSlot.JUDGE, n_items=4,
        reliability_weight=1.0, seat_decision=SeatDecision.SEAT, metadata={},
    )
    records = {"m": [_rec(f"i{j}", "m", correct=True) for j in range(4)]}

    async def _one_panel(*_a, **_k):
        return {"m": prof}, records

    monkeypatch.setattr("ai_crucible.characterize.run.run_panel", _one_panel)
    out_path = tmp_path / "report.json"
    assert main(["--out", str(out_path)]) == 0
    import json

    report = json.loads(out_path.read_text(encoding="utf-8"))
    assert report["alt_test_reference"] == "model-jury-bootstrap"
    kb = report.get("kappa_baseline")
    assert kb is not None
    assert kb["provisional"] is True
    assert kb["value"] == 0.80
    # the source string must make the fabrication explicit (not human-estimated).
    assert "human" in kb["source"].lower()
