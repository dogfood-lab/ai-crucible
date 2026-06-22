"""Tests for :mod:`ai_crucible.catalog.store` — event-sourced persistence + the fold.

These tests exercise the store in ISOLATION (DECOMPOSE_BY_SECRETS): the lifecycle
decisions (promote / demote / classify / frontier / min-k) are injected as FAKE
callables, so nothing here imports ``graduation`` or ``differential``. They prove:

* append + read roundtrip for RUN and TRANSITION events,
* idempotency — a second ``record_run`` / ``record_transition`` on the same
  deterministic id is a no-op,
* the hash chain verifies (delegation to ``verify_hash_chain``),
* ``current_tiers`` folds transitions correctly (last-per-puzzle wins; LAB default),
* ``aggregate`` splits Claude vs cohort, builds ``per_model``, and picks the latest
  present panel,
* ``derive_tiers`` emits the right transition for each tier given a fake decision,
  AND is idempotent (a second call with the same evidence appends nothing),
* the schema-future guard raises a structured ``[CATALOG_SCHEMA_FUTURE]`` error.

Design lock: ``swarm/epic-4/CONTRACT.md`` §D + cross-cutting locks.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ai_crucible.catalog.store import CATALOG_STORE_FILENAME, CatalogStore, CatalogStoreError
from ai_crucible.catalog.types import (
    Decision,
    DemoteVerdict,
    ModelRunSeries,
    PanelSignal,
    PromoteVerdict,
    PuzzleAggregate,
    RuleConfig,
    RunRecord,
    TransitionReason,
    TransitionRecord,
    Typology,
)
from ai_crucible.types import CatalogTier

# --------------------------------------------------------------------------- #
# Builders — small helpers that pin every determinism input
# --------------------------------------------------------------------------- #


def _run(
    *,
    puzzle_id: str = "p1",
    content_hash: str = "h1",
    model_id: str = "claude-opus",
    family: str | None = "claude",
    role: str = "solver",
    k: int = 3,
    n: int = 20,
    successes: int = 10,
    started_at: str = "2026-06-21T00:00:00Z",
    finished_at: str = "2026-06-21T00:01:00Z",
    nonce: str = "n0",
    panel: PanelSignal | None = None,
    rule_version: str = "rv0",
) -> RunRecord:
    return RunRecord(
        puzzle_id=puzzle_id,
        puzzle_content_hash=content_hash,
        model_id=model_id,
        family=family,
        role=role,
        k=k,
        n=n,
        successes=successes,
        pass_hat_k=successes / n if n else 0.0,
        wilson_lower=0.2,
        wilson_upper=0.7,
        arm="baseline",
        started_at=started_at,
        finished_at=finished_at,
        nonce=nonce,
        rule_version=rule_version,
        panel=panel,
    )


def _transition(
    *,
    puzzle_id: str = "p1",
    from_tier: CatalogTier = CatalogTier.LAB,
    to_tier: CatalogTier = CatalogTier.ARENA,
    reason: TransitionReason = TransitionReason.GRADUATE_WILSON,
    recorded_at: str = "2026-06-21T01:00:00Z",
    run_ids: list[str] | None = None,
    rule_version: str = "rv0",
) -> TransitionRecord:
    return TransitionRecord(
        puzzle_id=puzzle_id,
        from_tier=from_tier,
        to_tier=to_tier,
        reason_code=reason,
        recorded_at=recorded_at,
        contributing_run_ids=run_ids or [],
        rule_version=rule_version,
    )


def _store(tmp_path: Path) -> CatalogStore:
    return CatalogStore(tmp_path / CATALOG_STORE_FILENAME)


# --------------------------------------------------------------------------- #
# Append + read roundtrip
# --------------------------------------------------------------------------- #


def test_record_and_read_run_roundtrip(tmp_path: Path) -> None:
    store = _store(tmp_path)
    run = _run(successes=12)
    h = store.record_run(run)
    assert isinstance(h, str) and len(h) == 64  # chain hash

    runs = store.read_runs()
    assert len(runs) == 1
    assert runs[0].run_id == run.run_id
    assert runs[0].successes == 12
    assert runs[0].puzzle_id == "p1"


def test_record_and_read_transition_roundtrip(tmp_path: Path) -> None:
    store = _store(tmp_path)
    t = _transition()
    h = store.record_transition(t)
    assert isinstance(h, str) and len(h) == 64

    transitions = store.read_transitions()
    assert len(transitions) == 1
    assert transitions[0].transition_id == t.transition_id
    assert transitions[0].to_tier is CatalogTier.ARENA


def test_runs_and_transitions_share_one_log_but_filter_by_kind(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.record_run(_run())
    store.record_transition(_transition())
    assert len(store.read_runs()) == 1
    assert len(store.read_transitions()) == 1


# --------------------------------------------------------------------------- #
# Idempotency (Q4 / PIN_PER_STEP)
# --------------------------------------------------------------------------- #


def test_record_run_is_idempotent_on_run_id(tmp_path: Path) -> None:
    store = _store(tmp_path)
    run = _run()
    first = store.record_run(run)
    second = store.record_run(run)  # same deterministic run_id
    assert first is not None
    assert second is None  # skipped
    assert len(store.read_runs()) == 1


def test_record_transition_is_idempotent_on_transition_id(tmp_path: Path) -> None:
    store = _store(tmp_path)
    t = _transition()
    assert store.record_transition(t) is not None
    assert store.record_transition(t) is None
    assert len(store.read_transitions()) == 1


def test_read_runs_dedups_first_wins(tmp_path: Path) -> None:
    # Two physically appended runs with the SAME id (e.g. a crash mid-append that
    # bypassed the idempotency guard): the fold keeps the first.
    store = _store(tmp_path)
    run = _run(successes=5)
    store._store.append(run.to_payload())  # raw append, bypass guard
    store._store.append(run.to_payload())
    runs = store.read_runs()
    assert len(runs) == 1
    assert runs[0].successes == 5


# --------------------------------------------------------------------------- #
# Hash chain
# --------------------------------------------------------------------------- #


def test_verify_delegates_to_hash_chain(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.record_run(_run())
    store.record_transition(_transition())
    assert store.verify() is True


def test_verify_empty_log_is_true(tmp_path: Path) -> None:
    store = _store(tmp_path)
    assert store.verify() is True


# --------------------------------------------------------------------------- #
# current_tiers fold
# --------------------------------------------------------------------------- #


def test_current_tiers_defaults_lab_for_runs_without_transition(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.record_run(_run(puzzle_id="p1"))
    tiers = store.current_tiers()
    assert tiers["p1"] is CatalogTier.LAB


def test_current_tiers_last_transition_per_puzzle_wins(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.record_run(_run(puzzle_id="p1"))
    # LAB -> ARENA, then ARENA -> REGRESSION; last wins.
    store.record_transition(
        _transition(
            puzzle_id="p1",
            from_tier=CatalogTier.LAB,
            to_tier=CatalogTier.ARENA,
            recorded_at="2026-06-21T01:00:00Z",
            run_ids=["a"],
        )
    )
    store.record_transition(
        _transition(
            puzzle_id="p1",
            from_tier=CatalogTier.ARENA,
            to_tier=CatalogTier.REGRESSION,
            reason=TransitionReason.SATURATED_REGRESSION,
            recorded_at="2026-06-21T02:00:00Z",
            run_ids=["b"],
        )
    )
    tiers = store.current_tiers()
    assert tiers["p1"] is CatalogTier.REGRESSION


def test_current_tiers_defer_receipt_does_not_change_tier(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.record_run(_run(puzzle_id="p1"))
    # A LAB->LAB defer receipt is the last transition but keeps the tier at LAB.
    store.record_transition(
        _transition(
            puzzle_id="p1",
            from_tier=CatalogTier.LAB,
            to_tier=CatalogTier.LAB,
            reason=TransitionReason.DEFER_TO_DESIGNER,
            run_ids=["x"],
        )
    )
    assert store.current_tiers()["p1"] is CatalogTier.LAB


# --------------------------------------------------------------------------- #
# aggregate
# --------------------------------------------------------------------------- #


def _is_frontier(model_id: str) -> bool:
    return "frontier" in model_id or model_id == "claude-opus"


def _min_k_for(_puzzle_id: str) -> int:
    return 3


def test_aggregate_splits_claude_vs_cohort_case_insensitive(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.record_run(_run(model_id="claude-opus", family="Claude", successes=8, nonce="a"))
    store.record_run(_run(model_id="gpt-frontier", family="openai", successes=15, nonce="b"))
    store.record_run(_run(model_id="llama-x", family="meta", successes=3, nonce="c"))

    aggs = store.aggregate(
        generator_family="claude", is_frontier=_is_frontier, min_k_for=_min_k_for
    )
    agg = aggs["p1"]
    # Claude arm: only the Claude-family run.
    assert agg.claude_successes == 8
    assert agg.claude_n == 20
    # Cohort: union of the two non-claude families.
    assert agg.cohort_successes == 18
    assert agg.cohort_n == 40


def test_aggregate_builds_per_model_series_in_log_order(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.record_run(_run(model_id="gpt-frontier", family="openai", successes=4, nonce="a"))
    store.record_run(_run(model_id="gpt-frontier", family="openai", successes=9, nonce="b"))

    aggs = store.aggregate(
        generator_family="claude", is_frontier=_is_frontier, min_k_for=_min_k_for
    )
    series = aggs["p1"].per_model["gpt-frontier"]
    assert isinstance(series, ModelRunSeries)
    assert series.runs == [(4, 20), (9, 20)]  # log order preserved
    assert series.is_frontier is True
    assert series.min_k == 3
    assert series.family == "openai"


def test_aggregate_picks_latest_present_panel_by_started_at(tmp_path: Path) -> None:
    store = _store(tmp_path)
    early_panel = PanelSignal(present=True, meets_quorum=True, fairness="fair")
    late_panel = PanelSignal(present=True, meets_quorum=False, escalate=True)
    store.record_run(
        _run(started_at="2026-06-21T00:00:00Z", nonce="a", panel=early_panel)
    )
    store.record_run(
        _run(started_at="2026-06-21T05:00:00Z", nonce="b", panel=late_panel)
    )
    # A later-by-clock run carrying NO panel must not clobber the present one.
    store.record_run(
        _run(started_at="2026-06-21T09:00:00Z", nonce="c", panel=None)
    )

    agg = store.aggregate(
        generator_family="claude", is_frontier=_is_frontier, min_k_for=_min_k_for
    )["p1"]
    assert agg.panel.present is True
    assert agg.panel.escalate is True  # the latest PRESENT panel


def test_aggregate_panel_absent_when_no_run_carried_one(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.record_run(_run(panel=None))
    agg = store.aggregate(
        generator_family="claude", is_frontier=_is_frontier, min_k_for=_min_k_for
    )["p1"]
    assert agg.panel.present is False


def test_aggregate_collects_contributing_run_ids_and_latest_hash(tmp_path: Path) -> None:
    store = _store(tmp_path)
    r1 = _run(content_hash="h1", nonce="a")
    r2 = _run(content_hash="h2", nonce="b")
    store.record_run(r1)
    store.record_run(r2)
    agg = store.aggregate(
        generator_family="claude", is_frontier=_is_frontier, min_k_for=_min_k_for
    )["p1"]
    assert set(agg.contributing_run_ids) == {r1.run_id, r2.run_id}
    assert agg.puzzle_content_hash == "h2"  # latest run's hash


# --------------------------------------------------------------------------- #
# derive_tiers — the projection
# --------------------------------------------------------------------------- #

_RULE = RuleConfig()


def _classify_fn(_agg: PuzzleAggregate, _rc: RuleConfig):
    from ai_crucible.catalog.types import ClassifyResult

    return ClassifyResult(
        typology=Typology.CLAUDE_SPECIFIC_GAP,
        delta=-0.4,
        delta_ci=(-0.6, -0.2),
        realized_mde=0.3,
        p_value=0.01,
    )


def _promote_yes(_agg: PuzzleAggregate, _rc: RuleConfig) -> PromoteVerdict:
    return PromoteVerdict(
        decision=Decision.PROMOTE,
        claude_band_ok=True,
        cohort_nontrivial=True,
        fairness_ok=True,
        deferred=False,
        evidence={"claude_band_ok": True},
    )


def _promote_defer(_agg: PuzzleAggregate, _rc: RuleConfig) -> PromoteVerdict:
    return PromoteVerdict(
        decision=Decision.DEFER_TO_DESIGNER,
        claude_band_ok=True,
        cohort_nontrivial=True,
        fairness_ok=False,
        deferred=True,
        evidence={"deferred": True},
    )


def _promote_hold(_agg: PuzzleAggregate, _rc: RuleConfig) -> PromoteVerdict:
    return PromoteVerdict(
        decision=Decision.HOLD,
        claude_band_ok=False,
        cohort_nontrivial=True,
        fairness_ok=True,
        deferred=False,
    )


def _demote_yes(series: ModelRunSeries, _rc: RuleConfig) -> DemoteVerdict:
    return DemoteVerdict(
        demote=True,
        e_value=42.0,
        dwell_runs=2,
        post_trigger_n=20,
        frontier_ok=True,
        reason="saturated",
        evidence={"e_value": 42.0},
    )


def _demote_no(series: ModelRunSeries, _rc: RuleConfig) -> DemoteVerdict:
    return DemoteVerdict(
        demote=False, e_value=1.0, dwell_runs=0, post_trigger_n=0, frontier_ok=True
    )


def test_derive_tiers_lab_promote_emits_graduate_with_born_typology(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.record_run(_run(puzzle_id="p1", model_id="claude-opus"))

    emitted = store.derive_tiers(
        promote_fn=_promote_yes,
        demote_fn=_demote_no,
        classify_fn=_classify_fn,
        rule_config=_RULE,
        is_frontier=_is_frontier,
        min_k_for=_min_k_for,
        now="2026-06-21T02:00:00Z",
    )
    assert len(emitted) == 1
    t = emitted[0]
    assert t.from_tier is CatalogTier.LAB
    assert t.to_tier is CatalogTier.ARENA
    assert t.reason_code is TransitionReason.GRADUATE_WILSON
    assert t.evidence["born_typology"] == str(Typology.CLAUDE_SPECIFIC_GAP)
    assert t.rule_version == _RULE.rule_version
    assert t.recorded_at == "2026-06-21T02:00:00Z"
    # And the projection now reflects ARENA.
    assert store.current_tiers()["p1"] is CatalogTier.ARENA


def test_derive_tiers_lab_defer_emits_receipt_without_changing_tier(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.record_run(_run(puzzle_id="p1"))
    emitted = store.derive_tiers(
        promote_fn=_promote_defer,
        demote_fn=_demote_no,
        classify_fn=_classify_fn,
        rule_config=_RULE,
        is_frontier=_is_frontier,
        min_k_for=_min_k_for,
        now="2026-06-21T02:00:00Z",
    )
    assert len(emitted) == 1
    t = emitted[0]
    assert t.from_tier is CatalogTier.LAB
    assert t.to_tier is CatalogTier.LAB
    assert t.reason_code is TransitionReason.DEFER_TO_DESIGNER
    assert store.current_tiers()["p1"] is CatalogTier.LAB


def test_derive_tiers_lab_hold_emits_nothing(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.record_run(_run(puzzle_id="p1"))
    emitted = store.derive_tiers(
        promote_fn=_promote_hold,
        demote_fn=_demote_no,
        classify_fn=_classify_fn,
        rule_config=_RULE,
        is_frontier=_is_frontier,
        min_k_for=_min_k_for,
        now="2026-06-21T02:00:00Z",
    )
    assert emitted == []
    assert store.current_tiers()["p1"] is CatalogTier.LAB


def test_derive_tiers_arena_saturated_demotes_to_regression(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.record_run(_run(puzzle_id="p1", model_id="gpt-frontier", family="openai"))
    # Seed the puzzle into ARENA.
    store.record_transition(
        _transition(puzzle_id="p1", from_tier=CatalogTier.LAB, to_tier=CatalogTier.ARENA)
    )

    emitted = store.derive_tiers(
        promote_fn=_promote_hold,
        demote_fn=_demote_yes,
        classify_fn=_classify_fn,
        rule_config=_RULE,
        is_frontier=_is_frontier,
        min_k_for=_min_k_for,
        now="2026-06-21T03:00:00Z",
    )
    assert len(emitted) == 1
    t = emitted[0]
    assert t.from_tier is CatalogTier.ARENA
    assert t.to_tier is CatalogTier.REGRESSION
    assert t.reason_code is TransitionReason.SATURATED_REGRESSION
    assert t.evidence["model_id"] == "gpt-frontier"
    assert t.evidence["e_value"] == 42.0
    assert store.current_tiers()["p1"] is CatalogTier.REGRESSION


def test_derive_tiers_arena_no_demote_stays(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.record_run(_run(puzzle_id="p1", model_id="gpt-frontier", family="openai"))
    store.record_transition(
        _transition(puzzle_id="p1", from_tier=CatalogTier.LAB, to_tier=CatalogTier.ARENA)
    )
    emitted = store.derive_tiers(
        promote_fn=_promote_hold,
        demote_fn=_demote_no,
        classify_fn=_classify_fn,
        rule_config=_RULE,
        is_frontier=_is_frontier,
        min_k_for=_min_k_for,
        now="2026-06-21T03:00:00Z",
    )
    assert emitted == []
    assert store.current_tiers()["p1"] is CatalogTier.ARENA


def test_derive_tiers_regression_repromote_when_fn_supplied(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.record_run(_run(puzzle_id="p1", model_id="gpt-frontier", family="openai"))
    store.record_transition(
        _transition(puzzle_id="p1", from_tier=CatalogTier.LAB, to_tier=CatalogTier.ARENA)
    )
    store.record_transition(
        _transition(
            puzzle_id="p1",
            from_tier=CatalogTier.ARENA,
            to_tier=CatalogTier.REGRESSION,
            reason=TransitionReason.SATURATED_REGRESSION,
            recorded_at="2026-06-21T02:30:00Z",
            run_ids=["seed"],
        )
    )
    emitted = store.derive_tiers(
        promote_fn=_promote_hold,
        demote_fn=_demote_no,
        classify_fn=_classify_fn,
        rule_config=_RULE,
        is_frontier=_is_frontier,
        min_k_for=_min_k_for,
        now="2026-06-21T04:00:00Z",
        repromote_fn=_demote_yes,  # symmetric "fire" verdict
    )
    assert len(emitted) == 1
    t = emitted[0]
    assert t.from_tier is CatalogTier.REGRESSION
    assert t.to_tier is CatalogTier.ARENA
    assert t.reason_code is TransitionReason.REPROMOTE_REGRESSION
    assert store.current_tiers()["p1"] is CatalogTier.ARENA


def test_derive_tiers_regression_no_repromote_fn_is_noop(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.record_run(_run(puzzle_id="p1", model_id="gpt-frontier", family="openai"))
    store.record_transition(
        _transition(puzzle_id="p1", from_tier=CatalogTier.LAB, to_tier=CatalogTier.ARENA)
    )
    store.record_transition(
        _transition(
            puzzle_id="p1",
            from_tier=CatalogTier.ARENA,
            to_tier=CatalogTier.REGRESSION,
            reason=TransitionReason.SATURATED_REGRESSION,
            recorded_at="2026-06-21T02:30:00Z",
            run_ids=["seed"],
        )
    )
    emitted = store.derive_tiers(
        promote_fn=_promote_hold,
        demote_fn=_demote_no,
        classify_fn=_classify_fn,
        rule_config=_RULE,
        is_frontier=_is_frontier,
        min_k_for=_min_k_for,
        now="2026-06-21T04:00:00Z",
        repromote_fn=None,
    )
    assert emitted == []
    assert store.current_tiers()["p1"] is CatalogTier.REGRESSION


def test_derive_tiers_is_idempotent_on_second_call(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.record_run(_run(puzzle_id="p1", model_id="claude-opus"))
    first = store.derive_tiers(
        promote_fn=_promote_yes,
        demote_fn=_demote_no,
        classify_fn=_classify_fn,
        rule_config=_RULE,
        is_frontier=_is_frontier,
        min_k_for=_min_k_for,
        now="2026-06-21T02:00:00Z",
    )
    assert len(first) == 1
    # Puzzle is now ARENA; with the same evidence + no demote, nothing new appends.
    second = store.derive_tiers(
        promote_fn=_promote_yes,
        demote_fn=_demote_no,
        classify_fn=_classify_fn,
        rule_config=_RULE,
        is_frontier=_is_frontier,
        min_k_for=_min_k_for,
        now="2026-06-21T02:05:00Z",
    )
    assert second == []
    assert len(store.read_transitions()) == 1


def test_derive_tiers_only_frontier_models_can_demote(tmp_path: Path) -> None:
    # A weak local model passing must never demote (frontier gate is the store's
    # filter before it even calls demote_fn).
    store = _store(tmp_path)
    store.record_run(_run(puzzle_id="p1", model_id="llama-weak", family="meta"))
    store.record_transition(
        _transition(puzzle_id="p1", from_tier=CatalogTier.LAB, to_tier=CatalogTier.ARENA)
    )
    emitted = store.derive_tiers(
        promote_fn=_promote_hold,
        demote_fn=_demote_yes,  # would demote if called
        classify_fn=_classify_fn,
        rule_config=_RULE,
        is_frontier=_is_frontier,  # llama-weak is NOT frontier
        min_k_for=_min_k_for,
        now="2026-06-21T03:00:00Z",
    )
    assert emitted == []
    assert store.current_tiers()["p1"] is CatalogTier.ARENA


# --------------------------------------------------------------------------- #
# catalog_summary
# --------------------------------------------------------------------------- #


def test_catalog_summary_counts_and_defer_fraction(tmp_path: Path) -> None:
    store = _store(tmp_path)
    # p1: LAB with a defer receipt (its latest grad attempt deferred).
    store.record_run(_run(puzzle_id="p1"))
    store.record_transition(
        _transition(
            puzzle_id="p1",
            from_tier=CatalogTier.LAB,
            to_tier=CatalogTier.LAB,
            reason=TransitionReason.DEFER_TO_DESIGNER,
        )
    )
    # p2: LAB, no defer.
    store.record_run(_run(puzzle_id="p2", nonce="z"))
    # p3: ARENA.
    store.record_run(_run(puzzle_id="p3", nonce="y"))
    store.record_transition(
        _transition(puzzle_id="p3", from_tier=CatalogTier.LAB, to_tier=CatalogTier.ARENA)
    )

    summary = store.catalog_summary()
    assert summary["tiers"]["lab"] == 2
    assert summary["tiers"]["arena"] == 1
    assert summary["tiers"].get("regression", 0) == 0
    # 1 of 2 LAB puzzles last-deferred.
    assert summary["defer_fraction"] == pytest.approx(0.5)


def test_catalog_summary_defer_fraction_zero_when_no_lab(tmp_path: Path) -> None:
    store = _store(tmp_path)
    assert store.catalog_summary()["defer_fraction"] == 0.0


# --------------------------------------------------------------------------- #
# Schema-future guard (ANDON)
# --------------------------------------------------------------------------- #


def test_read_runs_raises_on_future_schema(tmp_path: Path) -> None:
    store = _store(tmp_path)
    run = _run()
    payload = run.to_payload()
    payload["schema_version"] = 999  # a downgrade reading newer data
    store._store.append(payload)
    with pytest.raises(CatalogStoreError) as exc:
        store.read_runs()
    assert "CATALOG_SCHEMA_FUTURE" in str(exc.value)


def test_read_transitions_raises_on_future_schema(tmp_path: Path) -> None:
    store = _store(tmp_path)
    t = _transition()
    payload = t.to_payload()
    payload["schema_version"] = 999
    store._store.append(payload)
    with pytest.raises(CatalogStoreError) as exc:
        store.read_transitions()
    assert "CATALOG_SCHEMA_FUTURE" in str(exc.value)
