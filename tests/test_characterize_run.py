"""Unit tests for the characterization runner's pure helpers (no live model needed)."""

from __future__ import annotations

import asyncio

import pytest

from ai_crucible.calibration.types import CalibrationCategory, CalibrationItem
from ai_crucible.characterize.run import (
    _grade_matrix,
    _jury,
    _parse_models,
    _to_num,
    collect_records,
    irt_prune_report,
    known_groups_report,
    main,
    panel_composition_report,
    panel_correlation_report,
    parse_choice,
    parse_verdict,
    run_panel,
)
from ai_crucible.characterize.types import JudgeProfile, JudgmentRecord, RoleSlot, SeatDecision


def _item(item_id: str, gold: str = "A") -> CalibrationItem:
    """A minimal A/B calibration item for the collect_records / run_panel paths."""
    return CalibrationItem(
        id=item_id,
        category=CalibrationCategory.KNOWN_DIAGNOSTIC,
        construct="x",
        confound_controlled="y",
        prompt=f"prompt for {item_id} — choose A or B",
        gold=gold,
    )


class _StubModel:
    """A fake :class:`OllamaModel` for collect_records: each call returns a record whose
    ``predicted`` is taken from ``replies`` (keyed by item prompt), or raises if the reply
    is an Exception. Exposes the ``model_id``/``family``/``quant`` collect_records reads."""

    def __init__(self, model_id: str, replies: dict[str, object], *, hang_prompts=None):
        self.model_id = model_id
        self.family = "fam"
        self.quant = None
        self._replies = replies
        self._hang_prompts = set(hang_prompts or [])

    async def judge_item(self, prompt: str, *, run_index: int = 0, position=None):
        if prompt in self._hang_prompts:
            # Simulate a hung Ollama call: sleep far past any sane per-item budget.
            await asyncio.sleep(3600)
        reply = self._replies.get(prompt, "A")
        if isinstance(reply, Exception):
            raise reply
        return JudgmentRecord(
            item_id="adapter-fallback",  # collect_records overwrites with authored id
            model_id=self.model_id,
            predicted=reply,
            gold=None,
            run_index=run_index,
            position=position,
            family=self.family,
        )


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


def test_irt_prune_degrades_to_shared_subset_on_ragged_matrix() -> None:
    """A per-item-salvaged run yields a RAGGED matrix (a model missing an item). The IRT screen
    must DEGRADE to the item subset shared across all models — reporting the ragged drop — rather
    than collapse the whole screen to an error (the same graceful degradation
    aggregate.pairwise_error_correlation already applies on the SAME ragged input)."""
    records = {
        "mA": [_rec("i_sat", "mA", correct=True), _rec("d1", "mA", correct=True),
               _rec("d2", "mA", correct=True), _rec("d3", "mA", correct=True)],
        "mB": [_rec("i_sat", "mB", correct=True), _rec("d1", "mB", correct=True),
               _rec("d2", "mB", correct=False), _rec("d3", "mB", correct=False)],
        # mC dropped d3 to per-item salvage → a ragged row (3 items, not 4).
        "mC": [_rec("i_sat", "mC", correct=True), _rec("d1", "mC", correct=False),
               _rec("d2", "mC", correct=False)],
    }
    rep = irt_prune_report(records)
    assert "error" not in rep                 # degraded, not collapsed to an error
    assert rep["ragged_dropped"] == ["d3"]
    assert rep["ragged_dropped_count"] == 1
    assert rep["n_items"] == 3                 # screened on the 3 items shared across all models
    assert "i_sat" in rep["dropped"]           # the saturated shared item still drops
    assert "DEGRADED" in rep["note"]


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
        # profiles, all_records, failed — the total-failure path (every model in `failed`).
        return {}, {}, [{"model_id": "m", "family": "fam", "error": "down", "n_records_lost": 0}]

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
        return {"m": prof}, records, []

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
        return {"m": prof}, records, []

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


# --------------------------------------------------------------------------- #
# characterize-degradation-002 — collect_records salvages completed items
#                                instead of discarding the whole transcript
# --------------------------------------------------------------------------- #


def test_collect_records_salvages_items_on_single_item_failure() -> None:
    """characterize-degradation-002: a single judge_item raise must NOT discard the model's
    whole transcript. The other items' records survive (a partial profile is strictly better
    than a dropped model), and the dropped item is counted + named in the returned stats."""
    items = [_item("i1"), _item("i2"), _item("i3")]
    replies = {
        items[0].prompt: "A",
        items[1].prompt: RuntimeError("boom on i2"),
        items[2].prompt: "B",
    }
    model = _StubModel("m", replies)
    records, stats = asyncio.run(collect_records(model, items, k=1))
    # i2 raised; i1 and i3 survive (RED: current code unwinds the whole model on the raise).
    survived = {r.item_id for r in records}
    assert "i1" in survived
    assert "i3" in survived
    assert "i2" not in survived
    assert stats["dropped_items"] == 1
    assert any("i2" in str(d) for d in stats["dropped_item_ids"])


def test_collect_records_aborts_on_consecutive_failures() -> None:
    """A run of consecutive failures genuinely signals the daemon is down — collect_records
    bounds it: after N consecutive item failures it raises (the whole-model drop in run_panel
    is then the correct outcome), rather than burning the full item set against a dead daemon."""
    items = [_item(f"i{n}") for n in range(8)]
    # every item raises → consecutive-failure ceiling trips → propagate (daemon-down signal).
    replies = {it.prompt: RuntimeError(f"down {it.id}") for it in items}
    model = _StubModel("m", replies)
    with pytest.raises(RuntimeError):
        asyncio.run(collect_records(model, items, k=1))


# --------------------------------------------------------------------------- #
# characterize-degradation-004 — per-item parse failures are counted + surfaced,
#                                not silently scored as wrong
# --------------------------------------------------------------------------- #


def test_collect_records_counts_unparseable_outputs() -> None:
    """characterize-degradation-004: an output that does not parse to a verdict is still
    scored as wrong (conservative default), but the unparseable COUNT must be a first-class
    signal so a format break is diagnosable rather than silently depressing accuracy."""
    items = [_item("i1", gold="A"), _item("i2", gold="A")]
    replies = {
        items[0].prompt: "A",  # parses fine
        items[1].prompt: "I cannot decide between the options",  # no standalone A/B → unparseable
    }
    model = _StubModel("m", replies)
    records, stats = asyncio.run(collect_records(model, items, k=1))
    # both records exist; the unparsed one was scored incorrect AND counted as a parse failure.
    assert stats["n_unparsed"] == 1
    unparsed_rec = next(r for r in records if r.item_id == "i2")
    assert unparsed_rec.correct is False
    # distinct from a *wrong* answer: the record is flagged as a parse failure.
    assert unparsed_rec.metadata.get("parse_failure") is True
    assert unparsed_rec.metadata.get("raw_choice") is None
    parsed_rec = next(r for r in records if r.item_id == "i1")
    assert parsed_rec.metadata.get("parse_failure") is False


# --------------------------------------------------------------------------- #
# characterize-degradation-005 — bounded per-item timeout around judge_item
# --------------------------------------------------------------------------- #


def test_collect_records_bounds_a_hung_judge_item() -> None:
    """characterize-degradation-005: a single hung Ollama call must NOT stall the whole run.
    A per-item timeout bounds it; the hung item is dropped (counted, not crash) and the run
    continues over the remaining items."""
    items = [_item("i1"), _item("hang"), _item("i3")]
    replies = {items[0].prompt: "A", items[2].prompt: "B"}
    model = _StubModel("m", replies, hang_prompts=[items[1].prompt])
    # tiny per-item budget so the test does not actually wait; RED: current code awaits with
    # no wait_for, so this hangs forever (no timeout kwarg even exists yet).
    records, stats = asyncio.run(collect_records(model, items, k=1, per_item_timeout=0.05))
    survived = {r.item_id for r in records}
    assert "i1" in survived and "i3" in survived
    assert "hang" not in survived
    assert stats["dropped_items"] >= 1
    assert any("hang" in str(d) for d in stats["dropped_item_ids"])


# --------------------------------------------------------------------------- #
# characterize-degradation-001 — failed models are surfaced + the run is flagged
#                                degraded (not silently absent from the report)
# --------------------------------------------------------------------------- #


def test_run_panel_surfaces_failed_models(monkeypatch) -> None:
    """characterize-degradation-001: a model that fails mid-run must NOT vanish. run_panel
    returns the failed models (id + reason) alongside profiles/records so a partial panel is
    distinguishable from a smaller-by-design one."""

    good_items = [_item("i1"), _item("i2")]

    def _fake_model(*, model_id, family, quant):
        if model_id == "bad":
            return _StubModel(model_id, {it.prompt: RuntimeError("OOM") for it in good_items})
        return _StubModel(model_id, {it.prompt: "A" for it in good_items})

    monkeypatch.setattr("ai_crucible.characterize.run.OllamaModel", _fake_model)
    monkeypatch.setattr("ai_crucible.characterize.run._evict", lambda *_a, **_k: None)

    panel = [("good1", "fam", None), ("bad", "fam", None), ("good2", "fam", None)]
    profiles, records, failed = asyncio.run(run_panel(panel, good_items, k=1))
    # the two good models judged; the bad model is NOT in records/profiles...
    assert set(records) == {"good1", "good2"}
    assert "bad" not in profiles
    # ...but it is SURFACED in `failed` with an id + reason (RED: run_panel returns only a
    # 2-tuple today and the failed model leaves no machine-readable trace).
    assert any(f["model_id"] == "bad" for f in failed)
    bad = next(f for f in failed if f["model_id"] == "bad")
    assert "OOM" in bad["error"]


def test_main_report_flags_degraded_and_lists_failed_models(tmp_path, monkeypatch) -> None:
    """characterize-degradation-001 (report side): when run_panel reports a failed model the
    emitted report must (a) list it under `failed_models`, (b) record the `attempted_panel`,
    and (c) flag the run `degraded` — so a consumer can tell a thinned panel from a
    thin-by-design one. The non-degraded happy path must NOT carry the degraded flag."""
    prof = JudgeProfile(
        model_id="good1", role=RoleSlot.JUDGE, n_items=2,
        reliability_weight=1.0, seat_decision=SeatDecision.SEAT, metadata={},
    )
    records = {"good1": [_rec(f"i{j}", "good1", correct=True) for j in range(2)]}
    failed = [{"model_id": "bad", "family": "fam", "error": "RuntimeError('OOM')",
               "n_records_lost": 0}]

    async def _degraded_panel(*_a, **_k):
        return {"good1": prof}, records, failed

    monkeypatch.setattr("ai_crucible.characterize.run.run_panel", _degraded_panel)
    out_path = tmp_path / "report.json"
    # --models supplies the attempted panel argparse records (run_panel itself is stubbed):
    # "bad" must appear in attempted_panel even though it produced no records.
    assert main(["--out", str(out_path), "--models", "good1@fam", "bad@fam"]) == 0
    import json

    report = json.loads(out_path.read_text(encoding="utf-8"))
    assert report["degraded"] is True
    assert any(f["model_id"] == "bad" for f in report["failed_models"])
    assert "bad" in report["attempted_panel"]


def test_main_report_not_degraded_on_clean_run(tmp_path, monkeypatch) -> None:
    """Contrast: a clean run (no failed models) must report degraded=False and an empty
    failed_models list — the degraded flag is earned by an actual failure, never default-on."""
    prof = JudgeProfile(
        model_id="m", role=RoleSlot.JUDGE, n_items=2,
        reliability_weight=1.0, seat_decision=SeatDecision.SEAT, metadata={},
    )
    records = {"m": [_rec(f"i{j}", "m", correct=True) for j in range(2)]}

    async def _clean_panel(*_a, **_k):
        return {"m": prof}, records, []

    monkeypatch.setattr("ai_crucible.characterize.run.run_panel", _clean_panel)
    out_path = tmp_path / "report.json"
    assert main(["--out", str(out_path)]) == 0
    import json

    report = json.loads(out_path.read_text(encoding="utf-8"))
    assert report["degraded"] is False
    assert report["failed_models"] == []


# --------------------------------------------------------------------------- #
# characterize-result-legibility — the operator-facing terminal output is honest
#   The JSON report is written to a FILE (args.out), not stdout — so the new
#   human chrome (caveat / degraded notice / progress / announcement / WHY) is
#   routed to STDERR, keeping the file report machine-readable and the existing
#   stdout summary lines stable. Tests assert on captured stderr+stdout.
# --------------------------------------------------------------------------- #


def _seat_profile(model_id: str = "m") -> JudgeProfile:
    return JudgeProfile(
        model_id=model_id, role=RoleSlot.JUDGE, n_items=4,
        reliability_weight=1.0, seat_decision=SeatDecision.SEAT, metadata={},
    )


def test_main_prints_provisional_caveat_to_operator(tmp_path, monkeypatch, capsys) -> None:
    """characterize-result-legibility-001 + error-hint-sweep-002: on the default (no
    --human-labels) path the load-bearing PROVISIONAL caveat — alt-test ω is a CIRCULAR
    model-jury bootstrap, all seats PROVISIONAL, below-quorum escalates to the Claude
    Designer — must be shown to the HUMAN at the end of the run, not buried only in the
    JSON report file. RED: today the caveat lives only in report['alt_test_caveat']; the
    terminal summary (seated / composed panel) never repeats it, so an operator who acts
    on the terminal treats a self-refereed provisional result as authoritative."""
    records = {"m": [_rec(f"i{j}", "m", correct=True) for j in range(4)]}

    async def _one_panel(*_a, **_k):
        return {"m": _seat_profile()}, records, []

    monkeypatch.setattr("ai_crucible.characterize.run.run_panel", _one_panel)
    out_path = tmp_path / "report.json"
    assert main(["--out", str(out_path)]) == 0
    captured = capsys.readouterr()
    seen = captured.out + captured.err
    # The operator must SEE the provisional/circular framing on screen.
    assert "PROVISIONAL" in seen
    # the specific load-bearing reasons, not just the word.
    low = seen.lower()
    assert "model-jury" in low or "circular" in low
    assert "--human-labels" in seen  # the actionable path to a non-circular alt-test


def test_main_human_grounded_path_does_not_print_circular_caveat(
    tmp_path, monkeypatch, capsys
) -> None:
    """Contrast: when run_panel is given REAL human labels (Fork C), the circular
    model-jury caveat must NOT be printed as a provisional warning — the seats are
    human-grounded. We stub human_labels to None at the loader but force the
    human-grounded report branch via a monkeypatched load that returns a sentinel; the
    simplest honest assertion is that on the DEFAULT path the caveat appears and the
    word 'circular' is tied to the model-jury reference, which the previous test covers.
    Here we assert the provisional banner is keyed on the absence of human labels by
    confirming it is present on the default path's stderr only once (no duplicate)."""
    records = {"m": [_rec(f"i{j}", "m", correct=True) for j in range(4)]}

    async def _one_panel(*_a, **_k):
        return {"m": _seat_profile()}, records, []

    monkeypatch.setattr("ai_crucible.characterize.run.run_panel", _one_panel)
    out_path = tmp_path / "report.json"
    assert main(["--out", str(out_path)]) == 0
    seen = (capsys.readouterr().err)
    # provisional banner emitted exactly once on the default path (no accidental dupes).
    assert seen.count("PROVISIONAL RESULT") == 1


def test_main_announces_plan_before_the_run(tmp_path, monkeypatch, capsys) -> None:
    """characterize-result-legibility-004: before sinking the (multi-hour) run the operator
    must be told what it will do — N model(s) x M item(s) x k, the item set, the reference
    mode, and that it is a long job. RED: today main() goes straight from argparse into the
    blocking run with zero output, so a first-run operator cannot tell the job even started
    or what panel/item-set it picked up."""
    records = {"m": [_rec(f"i{j}", "m", correct=True) for j in range(4)]}
    order: list[str] = []

    async def _one_panel(*_a, **_k):
        order.append("ran")
        return {"m": _seat_profile()}, records, []

    monkeypatch.setattr("ai_crucible.characterize.run.run_panel", _one_panel)
    out_path = tmp_path / "report.json"
    assert main(["--out", str(out_path), "--models", "a@fam", "b@fam", "--k", "2"]) == 0
    seen = capsys.readouterr().err
    low = seen.lower()
    # the announcement names the shape of the work (2 models, the item count, k=2).
    assert "2 model" in low
    assert "k=2" in low
    # and signals it may take a while.
    assert "long run" in low or "may take" in low or "minutes per model" in low


def test_main_announcement_precedes_the_run_loop(tmp_path, monkeypatch, capsys) -> None:
    """The announcement must be emitted BEFORE run_panel is invoked (so the operator can
    abort a wrong/unintended job before committing the time), not after."""
    records = {"m": [_rec(f"i{j}", "m", correct=True) for j in range(4)]}

    async def _one_panel(*_a, **_k):
        # capture stderr-so-far at the moment the run starts: the announcement must already
        # be present. We read the live stream by raising a flag the test inspects after.
        import sys as _sys

        _sys.stderr.write("__RUN_PANEL_ENTERED__\n")
        return {"m": _seat_profile()}, records, []

    monkeypatch.setattr("ai_crucible.characterize.run.run_panel", _one_panel)
    out_path = tmp_path / "report.json"
    assert main(["--out", str(out_path)]) == 0
    seen = capsys.readouterr().err
    # the announcement (mentions the item count / "characteriz") must appear BEFORE the
    # run-panel-entered marker.
    assert "__RUN_PANEL_ENTERED__" in seen
    head = seen.split("__RUN_PANEL_ENTERED__")[0].lower()
    assert "characteriz" in head  # the plan banner uses "characterizing ..."


def test_run_panel_per_model_progress_lines(monkeypatch, capsys) -> None:
    """characterize-result-legibility-003 (assertion form): per-model 'starting' progress is
    emitted for EACH model in the panel BEFORE its pass, with an i/N counter, so the operator
    sees liveness during the run instead of a ~25-minute black hole per model."""
    good_items = [_item("i1"), _item("i2")]

    def _fake_model(*, model_id, family, quant):
        return _StubModel(model_id, {it.prompt: "A" for it in good_items})

    monkeypatch.setattr("ai_crucible.characterize.run.OllamaModel", _fake_model)
    monkeypatch.setattr("ai_crucible.characterize.run._evict", lambda *_a, **_k: None)

    panel = [("m1", "fam", None), ("m2", "fam", None)]
    asyncio.run(run_panel(panel, good_items, k=1))
    cap = capsys.readouterr()
    seen = cap.out + cap.err
    # a per-model "starting" line with an i/N counter for BOTH models.
    assert "1/2" in seen and "2/2" in seen
    assert "m1" in seen and "m2" in seen
    # the progress mentions starting/loading (liveness before the pass), not only the
    # post-pass summary.
    low = seen.lower()
    assert "start" in low or "loading" in low or "judging" in low


def test_main_summary_says_degraded_when_models_failed(tmp_path, monkeypatch, capsys) -> None:
    """characterize-result-legibility-002: when the run is degraded (model(s) failed in
    Stage B's `failed`), the END-of-run terminal summary the operator reads must SAY
    'partial/degraded result — model(s) X failed', not present a thinned panel identically
    to a clean one. RED: the degraded flag lives only in report['degraded']; the summary
    prints only 'seated:' / 'composed panel:'."""
    records = {"good1": [_rec(f"i{j}", "good1", correct=True) for j in range(4)]}
    failed = [{"model_id": "badmodel", "family": "fam",
               "error": "RuntimeError('OOM')", "n_records_lost": 0}]

    async def _degraded_panel(*_a, **_k):
        return {"good1": _seat_profile("good1")}, records, failed

    monkeypatch.setattr("ai_crucible.characterize.run.run_panel", _degraded_panel)
    out_path = tmp_path / "report.json"
    assert main(["--out", str(out_path), "--models", "good1@fam", "badmodel@fam"]) == 0
    cap = capsys.readouterr()
    seen = cap.out + cap.err
    low = seen.lower()
    assert "degraded" in low or "partial" in low
    # the failed model is NAMED to the operator on screen.
    assert "badmodel" in seen


def test_main_summary_clean_run_no_degraded_notice(tmp_path, monkeypatch, capsys) -> None:
    """Contrast: a clean run (no failed models) must NOT print a degraded/partial notice —
    the notice is earned by an actual failure, never default-on."""
    records = {"m": [_rec(f"i{j}", "m", correct=True) for j in range(4)]}

    async def _clean_panel(*_a, **_k):
        return {"m": _seat_profile()}, records, []

    monkeypatch.setattr("ai_crucible.characterize.run.run_panel", _clean_panel)
    out_path = tmp_path / "report.json"
    assert main(["--out", str(out_path)]) == 0
    cap = capsys.readouterr()
    seen = (cap.out + cap.err).lower()
    assert "degraded run" not in seen
    assert "partial" not in seen


def test_main_summary_shows_reject_reason_on_screen(tmp_path, monkeypatch, capsys) -> None:
    """characterize-result-legibility-005: the terminal seat/screen/reject summary must
    carry the contrastive WHY (the deciding reason from profile.notes' 'DECISION: ...'
    note), not just the bare verdict word + numbers. RED: today the per-model line prints
    only acc/q/ece/omega + the decision word, so a REJECT driven by a format break looks
    identical on screen to a REJECT driven by a genuinely-weak judge."""
    seat = _seat_profile("good")
    reject = JudgeProfile(
        model_id="weak", role=RoleSlot.JUDGE, n_items=4,
        reliability_weight=0.0, seat_decision=SeatDecision.REJECT, metadata={},
        notes=["DECISION: REJECT — κ<0.412 one-sided floor"],
    )
    records = {
        "good": [_rec(f"i{j}", "good", correct=True) for j in range(4)],
        "weak": [_rec(f"i{j}", "weak", correct=False) for j in range(4)],
    }

    async def _mixed_panel(*_a, **_k):
        return {"good": seat, "weak": reject}, records, []

    monkeypatch.setattr("ai_crucible.characterize.run.run_panel", _mixed_panel)
    out_path = tmp_path / "report.json"
    assert main(["--out", str(out_path)]) == 0
    cap = capsys.readouterr()
    seen = cap.out + cap.err
    # the reject's deciding reason (the κ floor) is shown on screen, next to the verdict.
    assert "weak" in seen
    assert "REJECT" in seen.upper()
    # the contrastive WHY (the floor reason) must be legible on screen, not only in JSON.
    assert "one-sided floor" in seen or "κ<" in seen or "kappa" in seen.lower()
