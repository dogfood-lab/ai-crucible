"""Tests for the catalog graduation leaf (CONTRACT §A graduation + §B saturation).

These are PURE, deterministic tests: every aggregate / run-series is built by hand
(no store, no network, no clock), so the same inputs grade identically tomorrow
(PIN_PER_STEP). They pin the three load-bearing behaviors of the lifecycle rules:

* **§A graduation is THREE-VALUED** — ``promote_decision`` returns each of
  ``PROMOTE`` / ``HOLD`` / ``DEFER_TO_DESIGNER`` for crafted aggregates, with the
  decision precedence (a CONFIDENT negative HOLDs; clauses-1&2-pass-but-low-confidence
  cross-family signal DEFERs).
* **§B saturation is ANYTIME-VALID** — ``demote_decision`` accumulates the e-process
  per RUN (not per attempt), so one lucky 20/20 cannot demote: dwell across
  ``>= dwell_k`` separate runs is EMERGENT from the e-process, and the frontier gate
  blocks a weak local model from ever demoting.
* **§B re-promote** — ``repromote_decision`` fires when a Regression puzzle's rate has
  fallen through the 0.75–0.90 dead band on a frontier subject.
"""

from __future__ import annotations

import math

import pytest

from ai_crucible.catalog.graduation import (
    DEFAULT_FRONTIER_IDS,
    demote_decision,
    make_frontier_fn,
    promote_decision,
    repromote_decision,
)
from ai_crucible.catalog.types import (
    Decision,
    ModelRunSeries,
    PanelSignal,
    PuzzleAggregate,
    RuleConfig,
)

RULE = RuleConfig()


# --------------------------------------------------------------------------- #
# Builders — craft the folded aggregates / series by hand (deterministic)
# --------------------------------------------------------------------------- #


def _panel(
    *,
    present: bool = True,
    quorum: bool = True,
    escalate: bool = False,
    fairness: str | None = "fair",
) -> PanelSignal:
    return PanelSignal(present=present, meets_quorum=quorum, escalate=escalate, fairness=fairness)


def _agg(
    *,
    claude_s: int = 6,
    claude_n: int = 20,
    cohort_s: int = 6,
    cohort_n: int = 20,
    panel: PanelSignal | None = None,
) -> PuzzleAggregate:
    return PuzzleAggregate(
        puzzle_id="pz",
        puzzle_content_hash="hash",
        claude_successes=claude_s,
        claude_n=claude_n,
        cohort_successes=cohort_s,
        cohort_n=cohort_n,
        panel=panel if panel is not None else _panel(),
    )


def _series(
    runs: list[tuple[int, int]],
    *,
    model_id: str = "claude-opus-4-8",
    is_frontier: bool = True,
    min_k: int = 20,
) -> ModelRunSeries:
    return ModelRunSeries(
        puzzle_id="pz",
        model_id=model_id,
        family="claude" if "claude" in model_id else "other",
        is_frontier=is_frontier,
        min_k=min_k,
        runs=runs,
    )


# --------------------------------------------------------------------------- #
# §A promote_decision — three-valued
# --------------------------------------------------------------------------- #


def test_promote_mid_claude_nontrivial_cohort_fair_panel_promotes() -> None:
    """Mid-rate Claude + non-trivial cohort + present/quorum/fair panel → PROMOTE."""
    verdict = promote_decision(_agg(), RULE)
    assert verdict.decision is Decision.PROMOTE
    assert verdict.claude_band_ok is True
    assert verdict.cohort_nontrivial is True
    assert verdict.fairness_ok is True
    assert verdict.deferred is False
    # PROMOTE never pre-stamps the born typology — the store does it via classify_fn.
    assert verdict.born_typology is None
    # Evidence carries the wilson bounds + panel snapshot for the audit log.
    assert "claude_wilson" in verdict.evidence
    assert "cohort_wilson" in verdict.evidence
    assert "panel" in verdict.evidence


def test_promote_trivial_claude_holds() -> None:
    """A trivially-easy Claude rate (20/20) fails Clause 1 → HOLD (confident)."""
    verdict = promote_decision(_agg(claude_s=20, claude_n=20), RULE)
    assert verdict.decision is Decision.HOLD
    assert verdict.claude_band_ok is False


def test_promote_impossible_claude_holds() -> None:
    """An impossible Claude rate (0/20) also fails Clause 1 → HOLD."""
    verdict = promote_decision(_agg(claude_s=0, claude_n=20), RULE)
    assert verdict.decision is Decision.HOLD
    assert verdict.claude_band_ok is False


def test_promote_no_panel_defers() -> None:
    """Mid Claude + valid cohort but NO panel signal (present=False) → DEFER.

    Absence of a cross-family fairness verdict is low-confidence, not a pass
    (Trust-or-Escalate). Clauses 1 & 2 pass; the deferred posture wins.
    """
    verdict = promote_decision(_agg(panel=_panel(present=False, quorum=False, fairness=None)), RULE)
    assert verdict.decision is Decision.DEFER_TO_DESIGNER
    assert verdict.claude_band_ok is True
    assert verdict.cohort_nontrivial is True
    assert verdict.deferred is True


def test_promote_quorum_but_fairness_abstained_defers() -> None:
    """Panel present + quorum + not-escalating but ABSTAINS on fairness (fairness=None)
    → DEFER, never PROMOTE (re-audit fix).

    A panel that reached quorum and is not escalating but issues NO confident fairness
    verdict has NOT certified the puzzle fair. PROMOTE requires fairness_ok (a confident
    "fair"); an abstention is low-confidence on the fairness axis, so it escalates to the
    Designer. This is the exact edge the CONTRACT §A "PROMOTE iff ... fairness_ok" demands
    and that the original precedence prose let slip through to PROMOTE.
    """
    verdict = promote_decision(
        _agg(panel=_panel(present=True, quorum=True, escalate=False, fairness=None)), RULE
    )
    assert verdict.decision is Decision.DEFER_TO_DESIGNER
    assert verdict.fairness_ok is False
    assert verdict.claude_band_ok is True
    assert verdict.cohort_nontrivial is True


def test_promote_cohort_trivially_aces_holds() -> None:
    """Mid Claude but cohort trivially aces (20/20) → HOLD (confident negative)."""
    verdict = promote_decision(_agg(cohort_s=20, cohort_n=20), RULE)
    assert verdict.decision is Decision.HOLD
    assert verdict.cohort_nontrivial is False


def test_promote_confident_unfair_panel_holds() -> None:
    """Mid Claude + present/quorum/non-escalating but UNFAIR panel → HOLD.

    A CONFIDENT unfair verdict is a confident negative (precedes the DEFER check).
    """
    verdict = promote_decision(
        _agg(panel=_panel(present=True, quorum=True, escalate=False, fairness="unfair")), RULE
    )
    assert verdict.decision is Decision.HOLD
    assert verdict.fairness_ok is False
    assert verdict.deferred is False


def test_promote_escalating_panel_defers() -> None:
    """A panel that escalates (present + quorum but escalate=True) is low-confidence → DEFER."""
    verdict = promote_decision(
        _agg(panel=_panel(present=True, quorum=True, escalate=True, fairness="fair")), RULE
    )
    assert verdict.decision is Decision.DEFER_TO_DESIGNER
    assert verdict.deferred is True


def test_promote_no_cohort_evidence_defers() -> None:
    """Mid Claude + fair panel but ZERO cohort runs → DEFER (no cross-family solve evidence)."""
    verdict = promote_decision(_agg(cohort_s=0, cohort_n=0), RULE)
    assert verdict.decision is Decision.DEFER_TO_DESIGNER
    assert verdict.deferred is True


def test_promote_confident_negative_precedes_defer() -> None:
    """A confident-unfair panel HOLDs even when cohort evidence is also missing.

    Precedence: confident negative (Clause-3 unfair) beats the DEFER posture.
    """
    verdict = promote_decision(
        _agg(cohort_s=20, cohort_n=20, panel=_panel(present=False, quorum=False, fairness=None)),
        RULE,
    )
    # cohort 20/20 is a confident negative (clause 2) → HOLD, not DEFER.
    assert verdict.decision is Decision.HOLD


# --------------------------------------------------------------------------- #
# §B demote_decision — anytime-valid saturation
# --------------------------------------------------------------------------- #


def test_demote_two_clean_frontier_runs_demotes() -> None:
    """Two clean 20/20 frontier runs cross the e-process AND clear dwell → demote True.

    ``min_k=10`` so the post-trigger floor (3*min_k=30) is met by total_n=40; the
    per-puzzle floor is policy and the spec test isolates the e-process + dwell +
    frontier conjunction.
    """
    verdict = demote_decision(_series([(20, 20), (20, 20)], min_k=10), RULE)
    assert verdict.demote is True
    assert verdict.frontier_ok is True
    assert verdict.dwell_runs == 2
    assert verdict.post_trigger_n == 40
    assert verdict.e_value >= RULE.e_threshold


def test_demote_post_trigger_floor_blocks_at_high_min_k() -> None:
    """Same two 20/20 runs but a HIGH min_k floor (3*20=60 > 40) → blocked despite saturation."""
    verdict = demote_decision(_series([(20, 20), (20, 20)], min_k=20), RULE)
    assert verdict.e_value >= RULE.e_threshold  # e-process DID cross
    assert verdict.dwell_runs == 2               # dwell DID clear
    assert verdict.post_trigger_n == 40
    assert verdict.demote is False               # ...but post-trigger floor blocks (40 < 60)


def test_demote_one_run_blocked_by_dwell() -> None:
    """ONE lucky 20/20 run does NOT demote — dwell needs >= dwell_k separate runs."""
    verdict = demote_decision(_series([(20, 20)]), RULE)
    assert verdict.demote is False
    assert verdict.dwell_runs == 1


def test_demote_non_frontier_blocked_by_gate() -> None:
    """A non-frontier model with two 20/20 runs never demotes (frontier gate)."""
    verdict = demote_decision(_series([(20, 20), (20, 20)], is_frontier=False), RULE)
    assert verdict.demote is False
    assert verdict.frontier_ok is False


def test_demote_low_rate_does_not_saturate() -> None:
    """Low-rate runs (5/20) accumulate NO saturation evidence → demote False."""
    verdict = demote_decision(_series([(5, 20), (5, 20)]), RULE)
    assert verdict.demote is False
    assert verdict.e_value < RULE.e_threshold


def test_demote_post_trigger_n_floor_blocks() -> None:
    """High rate + enough runs but too few total attempts → blocked by post_trigger_n floor.

    Two 5/5 runs cross the e-process and clear dwell_k=2, but total_n=10 < 3*min_k=60.
    """
    verdict = demote_decision(_series([(5, 5), (5, 5), (5, 5), (5, 5)], min_k=20), RULE)
    assert verdict.post_trigger_n == 20
    # 4 runs of 5/5 still < 3*20 = 60 attempts → blocked.
    assert verdict.demote is False


def test_demote_e_value_matches_manual_eprocess() -> None:
    """Cross-check e_value against a hand-summed bernoulli_log_eprocess (RUN unit)."""
    from ai_crucible.scoring.stats import bernoulli_log_eprocess

    runs = [(20, 20), (20, 20)]
    log_e = sum(bernoulli_log_eprocess(s, n, p0=RULE.sat_p0, p1=RULE.sat_p1) for s, n in runs)
    verdict = demote_decision(_series(runs), RULE)
    assert verdict.e_value == pytest.approx(math.exp(log_e))


def test_demote_overflow_guarded_to_inf() -> None:
    """A massive run series pushes log_e past the overflow guard → e_value is inf, demote True.

    Each clean 20/20 run adds ~1.5 to log_e; ~500 runs exceeds the 700 guard, where
    ``math.exp`` would overflow float64 — the impl returns ``+inf`` instead of raising.
    """
    verdict = demote_decision(_series([(20, 20)] * 600, min_k=1), RULE)
    assert math.isinf(verdict.e_value)
    assert verdict.demote is True


# --------------------------------------------------------------------------- #
# §B repromote_decision — Regression→Arena
# --------------------------------------------------------------------------- #


def test_repromote_sustained_low_rate_frontier_fires() -> None:
    """A frontier model with sustained LOW rate in Regression → re-promote fires (demote True)."""
    # Rate dropped to ~0/20 across several runs: the failure e-process crosses.
    verdict = repromote_decision(_series([(1, 20), (1, 20), (0, 20)]), RULE)
    assert verdict.demote is True
    assert verdict.frontier_ok is True
    assert verdict.e_value >= RULE.e_threshold


def test_repromote_one_run_blocked_by_dwell() -> None:
    """One low run does not re-promote — dwell guard (>= dwell_k runs)."""
    verdict = repromote_decision(_series([(0, 20)]), RULE)
    assert verdict.demote is False
    assert verdict.dwell_runs == 1


def test_repromote_non_frontier_blocked() -> None:
    """A non-frontier subject never re-promotes (frontier gate, symmetric with demote)."""
    verdict = repromote_decision(_series([(0, 20), (0, 20)], is_frontier=False), RULE)
    assert verdict.demote is False
    assert verdict.frontier_ok is False


def test_repromote_still_saturated_does_not_fire() -> None:
    """A model still acing the puzzle (20/20) does NOT re-promote (rate is high)."""
    verdict = repromote_decision(_series([(20, 20), (20, 20)]), RULE)
    assert verdict.demote is False
    assert verdict.e_value < RULE.e_threshold


# --------------------------------------------------------------------------- #
# Frontier gate helpers
# --------------------------------------------------------------------------- #


def test_default_frontier_ids_contains_known_subjects() -> None:
    known = ("claude", "qwen3-coder", "deepseek", "glm", "kimi", "gpt-oss", "minimax", "nemotron")
    for sub in known:
        assert sub in DEFAULT_FRONTIER_IDS


def test_make_frontier_fn_substring_case_insensitive() -> None:
    fn = make_frontier_fn()
    assert fn("qwen3-coder:480b") is True          # substring + suffix
    assert fn("Claude-Opus-4-8") is True            # case-insensitive
    assert fn("GPT-OSS-120B") is True
    assert fn("llama-3.1-8b") is False              # not a frontier subject


def test_make_frontier_fn_extra_extends_set() -> None:
    fn = make_frontier_fn(extra=["llama-frontier"])
    assert fn("my-llama-frontier-test") is True
    assert fn("qwen3-coder:480b") is True           # base set still applies


def test_make_frontier_fn_returns_builtin_bool() -> None:
    fn = make_frontier_fn()
    assert fn("claude") is True
    assert fn("nope") is False
