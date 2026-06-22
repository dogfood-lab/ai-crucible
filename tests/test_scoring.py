"""Tests for the AI Crucible scoring layer (stats / oracle / judge panel).

Covers the load-bearing behaviors the research grounding pins:

* ``pass_hat_k`` is the ``p**k`` consistency estimator (τ-bench §1).
* Wilson / Clopper-Pearson stay sane at small N and at the 0/n and n/n
  boundaries (§1).
* ``graduates`` rules out *trivial* (20/20) AND *impossible* (0/20) in one rule,
  and admits a mid-rate puzzle (~5/20) — all three proven (§1).
* The §8.3 conjunctive gate goes **RED** when a critical-flavor penalty is
  triggered even on a successful solve — proven, per the dogfood "prove the gate
  goes RED" discipline.
* The judge panel excludes generator-family judges (EXTERNAL_VERIFIER, §10.2)
  and aggregates via majority / median reducers.

Async panel methods are driven with :func:`asyncio.run` so the suite needs no
``pytest-asyncio`` plugin (it is not a declared dependency).
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable

import pytest

from ai_crucible.characterize.aggregate import SeatedJudge, SeatedPanel
from ai_crucible.scoring import (
    CRITICAL_FLAVOR,
    JudgePanel,
    OracleOutcome,
    benjamini_hochberg,
    bernoulli_log_eprocess,
    clopper_pearson,
    fisher_difference_pvalue,
    grade,
    graduates,
    judge_family,
    mcnemar_exact,
    newcombe_wilson_diff,
    pass_hat_k,
    reduce_scores,
    weighted_judge,
    wilson_interval,
)
from ai_crucible.types import (
    AttemptState,
    GoodhartFlavor,
    Penalty,
    PuzzleClass,
    PuzzleMeta,
    Rewards,
    Score,
)

# --------------------------------------------------------------------------- #
# Local fixtures / helpers
# --------------------------------------------------------------------------- #


@pytest.fixture
def scoring_meta() -> PuzzleMeta:
    """A puzzle with a critical (adversarial) and a minor (regressional) penalty,
    within the §8.3 component bounds. canonical_call_count=8 so elegance is
    exercisable."""
    return PuzzleMeta(
        puzzle_id="score-p1",
        created_at="2026-06-01T00:00:00Z",
        capability_aspect="retrieval-grounding",
        puzzle_class=PuzzleClass.MULTI_FILE_SEARCH,
        point_threshold=50.0,
        time_budget_seconds=600,
        tool_call_budget=12,
        rewards=Rewards(
            solve=80.0,
            elegance_bonus_max=24.0,  # 30% of 80
            novelty_bonus_max=40.0,  # 50% of 80
            canonical_call_count=8,
        ),
        penalties=[
            Penalty(
                name="answer_key_fetch",
                goodhart_flavor=GoodhartFlavor.ADVERSARIAL,
                weight=-150.0,
                trigger="reads the sealed oracle",
            ),
            Penalty(
                name="redundant_tool_calls",
                goodhart_flavor=GoodhartFlavor.REGRESSIONAL,
                weight=-10.0,
                trigger="TCRR > 0.3",
            ),
        ],
    )


@pytest.fixture
def scoring_attempt() -> AttemptState:
    return AttemptState(attempt_id="score-a1", puzzle_id="score-p1", model="claude-opus-4-8")


def _clean_outcome(**overrides: object) -> OracleOutcome:
    """A passing outcome (solved, no regression, under budget, beats canonical)
    with per-test overrides."""
    base: dict[str, object] = {
        "solved": True,
        "solve_quality": 80.0,
        "no_regression": True,
        "tool_calls_used": 4,
        "time_used": 100.0,
    }
    base.update(overrides)
    return OracleOutcome(**base)  # type: ignore[arg-type]


JudgeFn = Callable[[AttemptState], Awaitable[Score]]


def make_judge(family: str | None, value: object) -> JudgeFn:
    """An injected judge returning a fixed Score, tagged with a model family."""

    async def _judge(_attempt: AttemptState) -> Score:
        return Score(value=value)  # type: ignore[arg-type]

    _judge.family = family  # type: ignore[attr-defined]
    return _judge


def make_raising_judge(family: str | None, exc: Exception) -> JudgeFn:
    """An injected judge that RAISES — simulates a dead/flaky model runtime
    (scoring-numerics-001, the SHARED GRADING-SEAM CONTRACT degradation probe)."""

    async def _judge(_attempt: AttemptState) -> Score:
        raise exc

    _judge.family = family  # type: ignore[attr-defined]
    _judge.model_id = f"raiser:{family}"  # type: ignore[attr-defined]
    return _judge


# --------------------------------------------------------------------------- #
# stats — pass_hat_k
# --------------------------------------------------------------------------- #


def test_pass_hat_k_is_p_to_the_k() -> None:
    """pass^k from empirical rate p=successes/n is p**k (τ-bench §1)."""
    assert pass_hat_k(8, 10, 3) == pytest.approx(0.8**3)
    assert pass_hat_k(8, 10, 3) == pytest.approx(0.512)


def test_pass_hat_k_boundaries() -> None:
    assert pass_hat_k(10, 10, 5) == pytest.approx(1.0)  # perfect -> 1.0
    assert pass_hat_k(0, 10, 3) == pytest.approx(0.0)  # never -> 0.0
    assert pass_hat_k(5, 10, 1) == pytest.approx(0.5)  # k=1 -> the rate itself


def test_pass_hat_k_decays_with_k() -> None:
    """Consistency decays exponentially in k (the whole point of pass^k)."""
    p3 = pass_hat_k(7, 10, 3)
    p5 = pass_hat_k(7, 10, 5)
    assert p5 < p3 < 1.0


def test_pass_hat_k_rejects_bad_inputs() -> None:
    with pytest.raises(ValueError):
        pass_hat_k(5, 0, 3)  # n must be > 0
    with pytest.raises(ValueError):
        pass_hat_k(11, 10, 3)  # successes > n
    with pytest.raises(ValueError):
        pass_hat_k(-1, 10, 3)  # successes < 0
    with pytest.raises(ValueError):
        pass_hat_k(5, 10, 0)  # k must be >= 1


# --------------------------------------------------------------------------- #
# stats — Wilson / Clopper-Pearson
# --------------------------------------------------------------------------- #


def test_wilson_interval_is_sane() -> None:
    lo, hi = wilson_interval(5, 20)
    assert 0.0 <= lo < 5 / 20 < hi <= 1.0  # point estimate inside the interval


def test_wilson_interval_stays_in_unit_range_at_boundaries() -> None:
    """Wilson (unlike Wald) never escapes [0, 1] at the 0/n and n/n edges (§1)."""
    lo0, hi0 = wilson_interval(0, 20)
    assert lo0 == 0.0  # clamped; degenerate point estimate
    assert 0.0 < hi0 < 1.0
    lo1, hi1 = wilson_interval(20, 20)
    assert hi1 == 1.0
    assert 0.0 < lo1 < 1.0


def test_wilson_tighter_with_more_data() -> None:
    """More attempts at the same rate -> narrower interval."""
    lo_small, hi_small = wilson_interval(5, 20)
    lo_big, hi_big = wilson_interval(50, 200)
    assert (hi_big - lo_big) < (hi_small - lo_small)


def test_wilson_rejects_bad_conf() -> None:
    with pytest.raises(ValueError):
        wilson_interval(5, 20, conf=0.0)
    with pytest.raises(ValueError):
        wilson_interval(5, 20, conf=1.0)
    with pytest.raises(ValueError):
        wilson_interval(5, 20, conf=1.5)


def test_clopper_pearson_boundaries() -> None:
    lo0, hi0 = clopper_pearson(0, 20)
    assert lo0 == 0.0 and 0.0 < hi0 < 1.0
    lo1, hi1 = clopper_pearson(20, 20)
    assert hi1 == 1.0 and 0.0 < lo1 < 1.0


def test_clopper_pearson_is_wider_than_wilson() -> None:
    """The 'exact' interval is conservative -> at least as wide as Wilson (§1)."""
    w_lo, w_hi = wilson_interval(5, 20)
    cp_lo, cp_hi = clopper_pearson(5, 20)
    assert (cp_hi - cp_lo) >= (w_hi - w_lo)
    assert cp_lo <= w_lo and cp_hi >= w_hi


def test_clopper_pearson_rejects_bad_inputs() -> None:
    with pytest.raises(ValueError):
        clopper_pearson(5, 0)
    with pytest.raises(ValueError):
        clopper_pearson(21, 20)


# --------------------------------------------------------------------------- #
# stats — graduates (§1 graduation rule: rules out trivial AND impossible)
# --------------------------------------------------------------------------- #


def test_graduates_false_for_trivial_20_of_20() -> None:
    """A puzzle nobody fails is trivial -> must NOT graduate (Wilson-upper > 0.90)."""
    assert graduates(20, 20) is False


def test_graduates_false_for_impossible_0_of_20() -> None:
    """A puzzle nobody solves is impossible -> must NOT graduate (Wilson-lower < 0.10)."""
    assert graduates(0, 20) is False


def test_graduates_true_for_mid_rate_5_of_20() -> None:
    """A ~25% solve-rate puzzle clears both Wilson bounds -> graduates (§1)."""
    assert graduates(5, 20) is True


def test_graduates_matches_wilson_bounds_directly() -> None:
    """graduates() is exactly the 0.10<=lower AND upper<=0.90 rule on Wilson."""
    for successes in range(21):
        lo, hi = wilson_interval(successes, 20)
        expected = lo >= 0.10 and hi <= 0.90
        assert graduates(successes, 20) is expected


def test_graduates_rejects_bad_inputs() -> None:
    with pytest.raises(ValueError):
        graduates(5, 0)


# --------------------------------------------------------------------------- #
# stats — McNemar exact
# --------------------------------------------------------------------------- #


def test_mcnemar_no_discordant_pairs_is_p_one() -> None:
    """No disagreement -> no evidence of a difference -> p = 1.0 (§9.3)."""
    assert mcnemar_exact(0, 0) == 1.0


def test_mcnemar_symmetric_is_p_one() -> None:
    """A perfectly even discordant split is maximally non-significant."""
    assert mcnemar_exact(5, 5) == pytest.approx(1.0)


def test_mcnemar_strong_asymmetry_is_significant() -> None:
    """All discordant pairs favor one system -> small p (10 vs 0 -> 2*0.5**10)."""
    p = mcnemar_exact(10, 0)
    assert p < 0.05
    assert p == pytest.approx(2 * (0.5**10))


def test_mcnemar_is_symmetric_in_arguments() -> None:
    """Two-sided p is invariant to which system is 'b' vs 'c'."""
    assert mcnemar_exact(8, 2) == pytest.approx(mcnemar_exact(2, 8))


def test_mcnemar_rejects_negative_counts() -> None:
    with pytest.raises(ValueError):
        mcnemar_exact(-1, 3)


# --------------------------------------------------------------------------- #
# oracle — the §8.3 conjunctive hard gate
# --------------------------------------------------------------------------- #


def test_gate_passes_on_clean_solve(
    scoring_attempt: AttemptState, scoring_meta: PuzzleMeta
) -> None:
    score = grade(scoring_attempt, scoring_meta, _clean_outcome())
    assert score.metadata["gate_passed"] is True
    assert score.metadata["failed_conditions"] == []
    # net = solve(80) + elegance(24 * (8-4)/8 = 12) + novelty(0) - penalties(0)
    assert score.value == pytest.approx(92.0)
    assert score.metadata["components"]["solve"] == pytest.approx(80.0)
    assert score.metadata["components"]["elegance"] == pytest.approx(12.0)


def test_gate_goes_red_on_critical_penalty_even_when_solved(
    scoring_attempt: AttemptState, scoring_meta: PuzzleMeta
) -> None:
    """LOAD-BEARING: a critical-flavor (ADVERSARIAL) penalty closes the gate even
    on a fully successful solve (§8.2/§8.3). Prove the gate goes RED."""
    outcome = _clean_outcome(triggered_penalties=["answer_key_fetch"])
    score = grade(scoring_attempt, scoring_meta, outcome)
    assert score.metadata["gate_passed"] is False
    assert "critical_penalty" in score.metadata["failed_conditions"]
    assert score.value == 0.0  # net is meaningless once a hard condition fails
    assert score.metadata["has_critical_penalty"] is True


def test_minor_penalty_alone_does_not_close_gate(
    scoring_attempt: AttemptState, scoring_meta: PuzzleMeta
) -> None:
    """A SMALL non-critical penalty subtracts from net but does NOT close the gate,
    because it does not drag the penalty-adjusted solve below point_threshold
    (80 − 10 = 70 ≥ 50). A non-critical penalty closes the gate only when it crosses
    that floor (Finding A — see test_causal_penalty_below_floor_closes_gate); a
    critical/ADVERSARIAL penalty closes it unconditionally (§8.2)."""
    outcome = _clean_outcome(triggered_penalties=["redundant_tool_calls"])
    score = grade(scoring_attempt, scoring_meta, outcome)
    assert score.metadata["gate_passed"] is True
    # net = 80 + 12 (elegance) + 0 - 10 = 82
    assert score.value == pytest.approx(82.0)
    assert score.metadata["components"]["penalties"] == pytest.approx(-10.0)
    # penalty-adjusted floor (Finding A) stays above threshold -> no close.
    assert score.metadata["components"]["penalty_adjusted_solve"] == pytest.approx(70.0)
    assert "penalty_adjusted_below_threshold" not in score.metadata["failed_conditions"]


def _meta_with_causal_penalty(weight: float) -> PuzzleMeta:
    """A puzzle declaring a CAUSAL skip_grounded_read penalty of ``weight`` (the
    seed-sulzbach shape) for the penalty-adjusted-floor tests (Finding A)."""
    return PuzzleMeta(
        puzzle_id="score-causal",
        created_at="2026-06-01T00:00:00Z",
        capability_aspect="retrieval-grounding",
        puzzle_class=PuzzleClass.MULTI_FILE_SEARCH,
        point_threshold=50.0,
        time_budget_seconds=600,
        tool_call_budget=12,
        rewards=Rewards(
            solve=80.0, elegance_bonus_max=24.0, novelty_bonus_max=40.0, canonical_call_count=8
        ),
        penalties=[
            Penalty(
                name="skip_grounded_read",
                goodhart_flavor=GoodhartFlavor.CAUSAL,
                weight=weight,
                trigger="reports a value without the grounded read",
            )
        ],
    )


def test_causal_penalty_below_floor_closes_gate(scoring_attempt: AttemptState) -> None:
    """LOAD-BEARING (Finding A): a causal penalty (skip_grounded_read, −60) that drags
    the penalty-adjusted solve (80 − 60 = 20) below point_threshold (50) CLOSES the
    gate even on a SOLVED, correct attempt — a fabricated-but-correct answer is a
    non-solve, not a discounted solve. Prove the gate goes RED."""
    meta = _meta_with_causal_penalty(-60.0)
    outcome = _clean_outcome(triggered_penalties=["skip_grounded_read"])
    score = grade(scoring_attempt, meta, outcome)
    assert outcome.solved is True  # the reported value WAS correct
    assert score.metadata["gate_passed"] is False
    assert "penalty_adjusted_below_threshold" in score.metadata["failed_conditions"]
    # it is the magnitude floor, not the critical-flavor veto, that closed it.
    assert "critical_penalty" not in score.metadata["failed_conditions"]
    assert score.metadata["components"]["penalty_adjusted_solve"] == pytest.approx(20.0)
    assert score.value == 0.0


def test_bonus_does_not_rescue_penalty_adjusted_floor(scoring_attempt: AttemptState) -> None:
    """NON-COMPENSATORY (Finding A): elegance/novelty bonuses are NOT in the floor, so
    they cannot lift a process-failed answer back over the gate. The Solver beats
    canonical (earns elegance) AND claims+validates novelty — net is high (positive) —
    but the gate stays CLOSED because the bonus-free floor solve(80) + causal(−60) = 20
    < 50. This is the whole point of gating on the floor, not on net."""
    meta = _meta_with_causal_penalty(-60.0)
    outcome = _clean_outcome(
        triggered_penalties=["skip_grounded_read"],
        tool_calls_used=1,  # << canonical 8 -> large elegance bonus
        novelty_claimed=True,
        novelty_validated=True,
    )
    score = grade(scoring_attempt, meta, outcome)
    assert score.metadata["components"]["elegance"] > 0.0  # a bonus WAS earned
    assert score.metadata["components"]["net"] > meta.point_threshold  # net alone would pass
    assert score.metadata["gate_passed"] is False  # ...but the floor closes the gate
    assert "penalty_adjusted_below_threshold" in score.metadata["failed_conditions"]


def test_causal_penalty_above_floor_keeps_gate_open(scoring_attempt: AttemptState) -> None:
    """Magnitude-keyed + per-puzzle (Finding A): a non-critical penalty that does NOT
    drag the penalty-adjusted solve below threshold leaves the gate OPEN
    (80 − 20 = 60 ≥ 50). Small process slips dock the tiebreaker, not the gate."""
    meta = _meta_with_causal_penalty(-20.0)
    outcome = _clean_outcome(triggered_penalties=["skip_grounded_read"])
    score = grade(scoring_attempt, meta, outcome)
    assert score.metadata["gate_passed"] is True
    assert "penalty_adjusted_below_threshold" not in score.metadata["failed_conditions"]
    assert score.metadata["components"]["penalty_adjusted_solve"] == pytest.approx(60.0)


def test_gate_closes_when_not_solved(
    scoring_attempt: AttemptState, scoring_meta: PuzzleMeta
) -> None:
    score = grade(scoring_attempt, scoring_meta, _clean_outcome(solved=False))
    assert score.metadata["gate_passed"] is False
    assert "not_solved" in score.metadata["failed_conditions"]
    assert score.value == 0.0


def test_gate_closes_on_regression(
    scoring_attempt: AttemptState, scoring_meta: PuzzleMeta
) -> None:
    """solved-AND-no-regression: a regression closes the gate (§10.2)."""
    score = grade(scoring_attempt, scoring_meta, _clean_outcome(no_regression=False))
    assert score.metadata["gate_passed"] is False
    assert "regression" in score.metadata["failed_conditions"]


def test_gate_closes_below_point_threshold(
    scoring_attempt: AttemptState, scoring_meta: PuzzleMeta
) -> None:
    score = grade(scoring_attempt, scoring_meta, _clean_outcome(solve_quality=49.9))
    assert score.metadata["gate_passed"] is False
    assert "below_point_threshold" in score.metadata["failed_conditions"]


def test_point_threshold_is_inclusive(
    scoring_attempt: AttemptState, scoring_meta: PuzzleMeta
) -> None:
    """quality == threshold passes (>= boundary)."""
    score = grade(scoring_attempt, scoring_meta, _clean_outcome(solve_quality=50.0))
    assert "below_point_threshold" not in score.metadata["failed_conditions"]


def test_gate_closes_over_tool_budget(
    scoring_attempt: AttemptState, scoring_meta: PuzzleMeta
) -> None:
    score = grade(scoring_attempt, scoring_meta, _clean_outcome(tool_calls_used=13))
    assert score.metadata["gate_passed"] is False
    assert "over_tool_budget" in score.metadata["failed_conditions"]


def test_tool_budget_is_inclusive(
    scoring_attempt: AttemptState, scoring_meta: PuzzleMeta
) -> None:
    """used == budget passes (<= boundary)."""
    score = grade(scoring_attempt, scoring_meta, _clean_outcome(tool_calls_used=12))
    assert "over_tool_budget" not in score.metadata["failed_conditions"]


def test_gate_closes_over_time_budget(
    scoring_attempt: AttemptState, scoring_meta: PuzzleMeta
) -> None:
    score = grade(scoring_attempt, scoring_meta, _clean_outcome(time_used=600.1))
    assert score.metadata["gate_passed"] is False
    assert "over_time_budget" in score.metadata["failed_conditions"]


def test_unvalidated_novelty_claim_closes_gate(
    scoring_attempt: AttemptState, scoring_meta: PuzzleMeta
) -> None:
    """Claiming novelty you can't get the panel to validate closes the gate —
    you don't get to assert your own bonus (§8.3/§8.7)."""
    outcome = _clean_outcome(novelty_claimed=True, novelty_validated=False)
    score = grade(scoring_attempt, scoring_meta, outcome)
    assert score.metadata["gate_passed"] is False
    assert "novelty_unvalidated" in score.metadata["failed_conditions"]


def test_validated_novelty_adds_bonus(
    scoring_attempt: AttemptState, scoring_meta: PuzzleMeta
) -> None:
    """Validated novelty adds novelty_bonus_max; used==canonical -> no elegance."""
    outcome = _clean_outcome(
        tool_calls_used=8,  # == canonical -> elegance 0
        novelty_claimed=True,
        novelty_validated=True,
    )
    score = grade(scoring_attempt, scoring_meta, outcome)
    assert score.metadata["gate_passed"] is True
    # net = 80 + 0 (elegance) + 40 (novelty) - 0 = 120
    assert score.value == pytest.approx(120.0)
    assert score.metadata["components"]["novelty"] == pytest.approx(40.0)


def test_unclaimed_novelty_never_adds_bonus(
    scoring_attempt: AttemptState, scoring_meta: PuzzleMeta
) -> None:
    """novelty_validated without a claim must not silently add a bonus."""
    outcome = _clean_outcome(novelty_claimed=False, novelty_validated=True)
    score = grade(scoring_attempt, scoring_meta, outcome)
    assert score.metadata["components"]["novelty"] == 0.0


def test_elegance_is_ratio_capped(
    scoring_attempt: AttemptState, scoring_meta: PuzzleMeta
) -> None:
    """Elegance scales as (canonical - used)/canonical * cap, capped at the max,
    and is zero when the Solver did not beat canonical (§8.4 ratio form)."""
    # Used 1 of canonical 8 -> tightness 7/8 -> 24 * 0.875 = 21.0 (< cap 24)
    s_tight = grade(scoring_attempt, scoring_meta, _clean_outcome(tool_calls_used=1))
    assert s_tight.metadata["components"]["elegance"] == pytest.approx(24.0 * 7 / 8)
    # Used MORE than canonical -> no elegance bonus
    s_over = grade(scoring_attempt, scoring_meta, _clean_outcome(tool_calls_used=10))
    assert s_over.metadata["components"]["elegance"] == 0.0


def test_multiple_failed_conditions_all_reported(
    scoring_attempt: AttemptState, scoring_meta: PuzzleMeta
) -> None:
    """The gate reports EVERY violated condition, not just the first (legible
    failure, not a bare zero)."""
    outcome = _clean_outcome(
        solved=False,
        no_regression=False,
        solve_quality=10.0,
        tool_calls_used=99,
        time_used=9999.0,
        triggered_penalties=["answer_key_fetch"],
    )
    score = grade(scoring_attempt, scoring_meta, outcome)
    failed = set(score.metadata["failed_conditions"])
    assert {
        "not_solved",
        "regression",
        "below_point_threshold",
        "critical_penalty",
        "over_tool_budget",
        "over_time_budget",
    } <= failed


def test_unknown_penalty_name_is_surfaced_not_scored(
    scoring_attempt: AttemptState, scoring_meta: PuzzleMeta
) -> None:
    """A triggered penalty not declared on the puzzle can't be scored (no weight),
    but must be visible in metadata so the misconfig isn't silently swallowed."""
    outcome = _clean_outcome(triggered_penalties=["not_a_declared_penalty"])
    score = grade(scoring_attempt, scoring_meta, outcome)
    assert score.metadata["gate_passed"] is True  # unknown != critical
    assert score.metadata["components"]["penalties"] == 0
    assert "not_a_declared_penalty" in score.metadata["unknown_penalties"]


def test_critical_flavor_constant_is_adversarial() -> None:
    """The §8.2 critical flavor is ADVERSARIAL (answer-key fetch / verifier tamper)."""
    assert CRITICAL_FLAVOR is GoodhartFlavor.ADVERSARIAL


def test_grade_stamps_novelty_validated_true(
    scoring_attempt: AttemptState, scoring_meta: PuzzleMeta
) -> None:
    """scoring-stats-003 (RED→GREEN): grade() must self-describe its novelty
    verdict via ``metadata['novelty_validated']`` so observability._is_novel can
    read it off the oracle score. Previously grade() recorded novelty only as
    ``components['novelty']`` (a numeric bonus) and never the boolean key, so the
    leaderboard novelty-rate read 0 off any attempt scored without a panel."""
    outcome = _clean_outcome(novelty_claimed=True, novelty_validated=True)
    score = grade(scoring_attempt, scoring_meta, outcome)
    assert score.metadata["novelty_validated"] is True


def test_grade_novelty_validated_false_without_claim(
    scoring_attempt: AttemptState, scoring_meta: PuzzleMeta
) -> None:
    """scoring-stats-003: the key is ``claimed AND validated`` — an unclaimed
    (or unvalidated) novelty stamps False, matching the bonus-eligibility rule."""
    # validated but not claimed -> no novelty
    s1 = grade(
        scoring_attempt,
        scoring_meta,
        _clean_outcome(novelty_claimed=False, novelty_validated=True),
    )
    assert s1.metadata["novelty_validated"] is False
    # claimed but not validated -> no novelty (and the gate also closes)
    s2 = grade(
        scoring_attempt,
        scoring_meta,
        _clean_outcome(novelty_claimed=True, novelty_validated=False),
    )
    assert s2.metadata["novelty_validated"] is False


# --------------------------------------------------------------------------- #
# judge_panel — EXTERNAL_VERIFIER exclusion + reducers
# --------------------------------------------------------------------------- #


def test_judge_family_reads_attribute_else_none() -> None:
    tagged = make_judge("qwen", True)
    assert judge_family(tagged) == "qwen"

    async def untagged(_a: AttemptState) -> Score:
        return Score(value=True)

    assert judge_family(untagged) is None


def test_panel_excludes_generator_family(
    scoring_attempt: AttemptState,
) -> None:
    """EXTERNAL_VERIFIER (§10.2): a judge sharing the generator's family is
    dropped before aggregation; the two cross-family judges decide."""
    panel = JudgePanel(
        judges=[
            make_judge("claude", True),  # excluded (same family as generator)
            make_judge("qwen", True),
            make_judge("mistral", False),
        ],
        reducer="majority",
        generator_family="claude",
    )
    result = asyncio.run(panel.score(scoring_attempt))
    assert result.metadata["excluded"] == ["claude"]
    assert result.metadata["eligible_count"] == 2
    assert result.metadata["votes"] == [True, False]  # only qwen + mistral voted
    assert result.value is True  # majority of [True, False] -> first-seen modal


def test_panel_keeps_all_when_no_generator_family(
    scoring_attempt: AttemptState,
) -> None:
    """generator_family=None disables exclusion (nothing can be proven to share)."""
    panel = JudgePanel(
        judges=[make_judge("claude", True), make_judge("qwen", True)],
        reducer="majority",
        generator_family=None,
    )
    result = asyncio.run(panel.score(scoring_attempt))
    assert result.metadata["eligible_count"] == 2
    assert result.metadata["excluded"] == []


def test_panel_keeps_untagged_judges(scoring_attempt: AttemptState) -> None:
    """An untagged judge (family None) is never excluded even with a generator
    family set — exclusion only drops judges PROVEN to share the family."""
    panel = JudgePanel(
        judges=[make_judge(None, True), make_judge("claude", True)],
        generator_family="claude",
    )
    result = asyncio.run(panel.score(scoring_attempt))
    # claude dropped, untagged kept
    assert result.metadata["eligible_count"] == 1


def test_panel_raises_when_all_judges_excluded(
    scoring_attempt: AttemptState,
) -> None:
    """If exclusion empties the panel, there is no cross-family verdict to give —
    that is a configuration error, raised loudly (§3 PoLL needs ≥1 external)."""
    panel = JudgePanel(
        judges=[make_judge("claude", True), make_judge("claude", False)],
        generator_family="claude",
    )
    with pytest.raises(ValueError, match="no eligible judges"):
        asyncio.run(panel.score(scoring_attempt))


def test_panel_excludes_case_variant_same_family(
    scoring_attempt: AttemptState,
) -> None:
    """scoring-stats-002 (RED→GREEN): EXTERNAL_VERIFIER exclusion must normalize
    the family label (casefold + strip) on BOTH sides. A generator tagged
    'claude' MUST drop a judge tagged 'Claude' (or '  CLAUDE ') — a casing/
    whitespace drift in the family vocabulary previously let a same-family judge
    vote on its own family's output, the exact self-preference bias §10.2 exists
    to structurally prevent."""
    panel = JudgePanel(
        judges=[
            make_judge("Claude", True),   # same family, capitalized — MUST drop
            make_judge("  CLAUDE ", True),  # same family, padded+upper — MUST drop
            make_judge("qwen", False),
        ],
        reducer="majority",
        generator_family="claude",
    )
    result = asyncio.run(panel.score(scoring_attempt))
    assert result.metadata["eligible_count"] == 1  # only qwen survives
    assert result.metadata["votes"] == [False]


def test_panel_none_family_never_equals_concrete(
    scoring_attempt: AttemptState,
) -> None:
    """scoring-stats-002 (contract): a None-family (untagged) judge never equals
    any concrete generator family — it is admitted as cross-family on operator
    responsibility, and recorded in ``untagged_judges_seated`` so the operator
    can SEE it was seated on trust (it cannot be PROVEN cross-family)."""
    untagged = make_judge(None, True)
    untagged.model_id = "mystery-model:32b"  # type: ignore[attr-defined]
    panel = JudgePanel(
        judges=[untagged, make_judge("claude", True)],
        reducer="majority",
        generator_family="claude",
    )
    result = asyncio.run(panel.score(scoring_attempt))
    # claude dropped, untagged admitted as cross-family-on-trust
    assert result.metadata["eligible_count"] == 1
    assert "mystery-model:32b" in result.metadata["untagged_judges_seated"]


def test_panel_strict_cross_family_excludes_untagged(
    scoring_attempt: AttemptState,
) -> None:
    """scoring-stats-002 (strict flag): the optional ``strict_cross_family`` flag
    (default False, preserving panel-emptying semantics) excludes None-family
    judges that cannot be PROVEN cross-family. Here it empties the panel, routing
    through the existing empty-panel ValueError."""
    untagged = make_judge(None, True)
    panel = JudgePanel(
        judges=[untagged, make_judge("claude", True)],
        reducer="majority",
        generator_family="claude",
        strict_cross_family=True,
    )
    with pytest.raises(ValueError, match="no eligible judges"):
        asyncio.run(panel.score(scoring_attempt))


def test_panel_strict_cross_family_default_keeps_untagged(
    scoring_attempt: AttemptState,
) -> None:
    """scoring-stats-002 (default preserved): with the flag OFF (default), an
    untagged judge is still admitted — the existing panel-emptying semantics for
    the all-same-family case are unchanged."""
    untagged = make_judge(None, True)
    panel = JudgePanel(
        judges=[untagged, make_judge("claude", True)],
        reducer="majority",
        generator_family="claude",
    )
    result = asyncio.run(panel.score(scoring_attempt))
    assert result.metadata["eligible_count"] == 1


def test_panel_median_reducer(scoring_attempt: AttemptState) -> None:
    """Numeric panel uses median — robust to a single outlier judge."""
    panel = JudgePanel(
        judges=[
            make_judge("qwen", 0.6),
            make_judge("mistral", 0.9),
            make_judge("cohere", 0.3),
        ],
        reducer="median",
        generator_family="claude",
    )
    result = asyncio.run(panel.score(scoring_attempt))
    assert result.value == pytest.approx(0.6)  # median of {0.3, 0.6, 0.9}


def test_panel_runs_judges_with_the_attempt(scoring_attempt: AttemptState) -> None:
    """The injected judge actually receives the AttemptState (the choke point)."""
    seen: list[str] = []

    async def recording_judge(attempt: AttemptState) -> Score:
        seen.append(attempt.attempt_id)
        return Score(value=True)

    recording_judge.family = "qwen"  # type: ignore[attr-defined]
    panel = JudgePanel(judges=[recording_judge], generator_family="claude")
    asyncio.run(panel.score(scoring_attempt))
    assert seen == ["score-a1"]


# --------------------------------------------------------------------------- #
# judge_panel — reduce_scores directly
# --------------------------------------------------------------------------- #


def test_reduce_majority_bool() -> None:
    out = reduce_scores(
        [Score(value=True), Score(value=True), Score(value=False)], "majority"
    )
    assert out.value is True
    assert out.metadata["agreement"] == pytest.approx(2 / 3)


def test_reduce_majority_discrete_strings() -> None:
    out = reduce_scores(
        [Score(value="legit"), Score(value="legit"), Score(value="circumvention")],
        "majority",
    )
    assert out.value == "legit"


def test_reduce_median_numeric() -> None:
    out = reduce_scores(
        [Score(value=1.0), Score(value=2.0), Score(value=3.0)], "median"
    )
    assert out.value == pytest.approx(2.0)


def test_reduce_median_even_count_averages_middle_two() -> None:
    out = reduce_scores([Score(value=1.0), Score(value=3.0)], "median")
    assert out.value == pytest.approx(2.0)


def test_reduce_empty_raises() -> None:
    with pytest.raises(ValueError, match="empty"):
        reduce_scores([], "majority")


def test_reduce_unknown_method_raises() -> None:
    with pytest.raises(ValueError, match="unknown reducer"):
        reduce_scores([Score(value=True)], "mean")


def test_reduce_median_on_non_numeric_raises() -> None:
    with pytest.raises(ValueError, match="numeric"):
        reduce_scores([Score(value="legit"), Score(value="nope")], "median")


# --------------------------------------------------------------------------- #
# judge_panel — reliability-weighted reducer (CARE, §11.4) + the seated bridge
# --------------------------------------------------------------------------- #


def test_reduce_weighted_reliable_dissent_beats_unreliable_majority() -> None:
    """CARE (§11.4): a reliable dissent (w=0.9 False) out-weighs an unreliable agreeing
    majority (two w=0.1 Trues) — unlike plain head-count majority."""
    out = reduce_scores(
        [
            Score(value=True, metadata={"judge_weight": 0.1}),
            Score(value=True, metadata={"judge_weight": 0.1}),
            Score(value=False, metadata={"judge_weight": 0.9}),
        ],
        "weighted",
    )
    assert out.value is False
    assert out.metadata["margin"] == pytest.approx(0.9 / 1.1)
    assert out.metadata["escalate"] is False
    assert out.metadata["weights"] == [0.1, 0.1, 0.9]
    assert out.metadata["reducer"] == "weighted"


def test_reduce_weighted_defaults_to_equal_weight() -> None:
    """No judge_weight in metadata → every judge weighs 1.0 (≡ majority by weight)."""
    out = reduce_scores(
        [Score(value=True), Score(value=True), Score(value=False)], "weighted"
    )
    assert out.value is True
    assert out.metadata["weights"] == [1.0, 1.0, 1.0]
    assert out.metadata["margin"] == pytest.approx(2 / 3)


def test_reduce_weighted_escalates_on_thin_margin() -> None:
    """A weighted dead-even split flags escalate (route to the Designer, §11.1)."""
    out = reduce_scores(
        [
            Score(value="pass", metadata={"judge_weight": 0.5}),
            Score(value="fail", metadata={"judge_weight": 0.5}),
        ],
        "weighted",
    )
    assert out.metadata["margin"] == pytest.approx(0.5)
    assert out.metadata["escalate"] is True


def test_reduce_weighted_all_zero_weights_raises() -> None:
    with pytest.raises(ValueError, match="weighted reducer"):
        reduce_scores([Score(value=True, metadata={"judge_weight": 0.0})], "weighted")


def test_weighted_judge_stamps_weight_and_preserves_family(
    scoring_attempt: AttemptState,
) -> None:
    wj = weighted_judge(make_judge("qwen", True), 0.7)
    assert judge_family(wj) == "qwen"
    score = asyncio.run(wj(scoring_attempt))
    assert score.metadata["judge_weight"] == 0.7
    assert score.value is True


def test_weighted_judge_family_override(scoring_attempt: AttemptState) -> None:
    wj = weighted_judge(make_judge(None, True), 0.5, family="mistral")
    assert judge_family(wj) == "mistral"


def test_weighted_judge_rejects_negative_weight() -> None:
    with pytest.raises(ValueError, match="weight must be >= 0"):
        weighted_judge(make_judge("qwen", True), -0.1)


def _seated_for(model_id: str, weight: float, family: str) -> SeatedJudge:
    return SeatedJudge(model_id=model_id, reliability_weight=weight, family=family)


def test_panel_from_seated_weights_drive_verdict(
    scoring_attempt: AttemptState,
) -> None:
    """JudgePanel.from_seated wires each seat's reliability weight into the weighted
    reducer: a reliable dissenter beats two unreliable agreers (§11.4)."""
    seated = SeatedPanel(
        seats=[
            _seated_for("m_qwen", 0.9, "qwen"),
            _seated_for("m_mistral", 0.1, "mistral"),
            _seated_for("m_cohere", 0.1, "cohere"),
        ],
        submodular=True,
        meets_quorum=True,
        escalate=False,
        not_seated=[],
        dropped_redundant=[],
        threshold=0.25,
        min_judges=3,
        notes=[],
    )
    votes = {"m_qwen": False, "m_mistral": True, "m_cohere": True}

    def judge_for(model_id: str):
        async def _j(_a: AttemptState) -> Score:
            return Score(value=votes[model_id])

        return _j

    panel = JudgePanel.from_seated(seated, judge_for, generator_family="claude")
    result = asyncio.run(panel.score(scoring_attempt))
    assert result.value is False  # 0.9 (False) > 0.1 + 0.1 (True)
    assert result.metadata["reducer"] == "weighted"
    assert result.metadata["margin"] == pytest.approx(0.9 / 1.1)
    assert result.metadata["escalate"] is False


def test_panel_from_seated_excludes_generator_family(
    scoring_attempt: AttemptState,
) -> None:
    """from_seated tags each judge with its seat family, so EXTERNAL_VERIFIER exclusion
    still drops a seat sharing the generator's family (§10.2)."""
    seated = SeatedPanel(
        seats=[_seated_for("m_qwen", 1.0, "qwen"), _seated_for("m_claude", 1.0, "claude")],
        submodular=True,
        meets_quorum=True,
        escalate=False,
        not_seated=[],
        dropped_redundant=[],
        threshold=0.25,
        min_judges=3,
        notes=[],
    )

    def judge_for(_model_id: str):
        async def _j(_a: AttemptState) -> Score:
            return Score(value=True)

        return _j

    panel = JudgePanel.from_seated(seated, judge_for, generator_family="claude")
    result = asyncio.run(panel.score(scoring_attempt))
    assert result.metadata["excluded"] == ["claude"]
    assert result.metadata["eligible_count"] == 1


# --------------------------------------------------------------------------- #
# judge_panel — per-judge novelty metadata is AGGREGATED, not dropped (H2, §8.7)
# --------------------------------------------------------------------------- #


def test_reduce_aggregates_novelty_validated_majority_true() -> None:
    """reduce_scores must aggregate each judge's novelty_validated vote (majority)
    onto the panel score — it previously DROPPED per-judge metadata (H2, §8.7)."""
    out = reduce_scores(
        [
            Score(value=True, metadata={"novelty_validated": True}),
            Score(value=True, metadata={"novelty_validated": True}),
            Score(value=True, metadata={"novelty_validated": False}),
        ],
        "majority",
    )
    # 2 of 3 judges validated novelty → panel verdict True.
    assert out.metadata["novelty_validated"] is True
    # The per-judge votes are preserved for auditability.
    assert out.metadata["novelty_votes"] == [True, True, False]


def test_reduce_aggregates_novelty_validated_majority_false() -> None:
    """Majority of judges reject novelty → panel novelty_validated is False."""
    out = reduce_scores(
        [
            Score(value=True, metadata={"novelty_validated": False}),
            Score(value=True, metadata={"novelty_validated": False}),
            Score(value=True, metadata={"novelty_validated": True}),
        ],
        "majority",
    )
    assert out.metadata["novelty_validated"] is False
    assert out.metadata["novelty_votes"] == [False, False, True]


def test_reduce_novelty_absent_when_no_judge_votes() -> None:
    """When no judge carries a novelty vote, the panel records novelty_validated
    False (a missing vote is not a validation — conservative, §8.7)."""
    out = reduce_scores([Score(value=True), Score(value=False)], "majority")
    assert out.metadata["novelty_validated"] is False
    # Judges with no vote contribute no novelty_votes entries.
    assert out.metadata["novelty_votes"] == []


def test_reduce_novelty_missing_vote_counts_as_not_validated() -> None:
    """A judge that abstains (no novelty_validated key) does not count toward
    validation; only explicit True votes can carry the majority (§8.7)."""
    out = reduce_scores(
        [
            Score(value=True, metadata={"novelty_validated": True}),
            Score(value=True),  # abstains
        ],
        "majority",
    )
    # 1 explicit True vote of 1 cast vote → majority True among voters.
    assert out.metadata["novelty_votes"] == [True]
    assert out.metadata["novelty_validated"] is True


def test_panel_surfaces_novelty_validated_verdict(
    scoring_attempt: AttemptState,
) -> None:
    """The JudgePanel surfaces an aggregated novelty_validated verdict in its score
    metadata (the kernel feeds this into the oracle gate, H2/§8.7)."""

    def nov_judge(family: str, vote: bool) -> JudgeFn:
        async def _j(_a: AttemptState) -> Score:
            return Score(value=True, metadata={"novelty_validated": vote})

        _j.family = family  # type: ignore[attr-defined]
        return _j

    panel = JudgePanel(
        judges=[nov_judge("qwen", True), nov_judge("mistral", True)],
        reducer="majority",
        generator_family="claude",
    )
    result = asyncio.run(panel.score(scoring_attempt))
    assert result.metadata["novelty_validated"] is True


# --------------------------------------------------------------------------- #
# judge_panel — DEGRADE on a flaky judge (scoring-numerics-001, SHARED CONTRACT)
# --------------------------------------------------------------------------- #


def test_panel_degrades_when_one_judge_raises(
    scoring_attempt: AttemptState,
) -> None:
    """scoring-numerics-001 (RED→GREEN, SHARED GRADING-SEAM CONTRACT): a single
    raising judge (a dead/flaky model runtime) must NOT abort the whole panel.
    ``JudgePanel.score`` rules on the SURVIVING judges and records the errored
    judge (id + error) in ``panel.metadata['judges_errored']`` — a partial panel
    returns a Score, never a bare exception, as long as ≥1 judge survives."""
    panel = JudgePanel(
        judges=[
            make_raising_judge("qwen", RuntimeError("ollama daemon down")),
            make_judge("mistral", True),
            make_judge("cohere", True),
        ],
        reducer="majority",
        generator_family="claude",
    )
    result = asyncio.run(panel.score(scoring_attempt))
    # Survivors decided: majority of [True, True] -> True (qwen never voted).
    assert result.value is True
    assert result.metadata["votes"] == [True, True]
    # The errored judge is recorded for auditability (id + error), not silenced.
    errored = result.metadata["judges_errored"]
    assert len(errored) == 1
    assert errored[0]["judge"] == "raiser:qwen"
    assert "ollama daemon down" in errored[0]["error"]
    # eligible_count reflects the judges that were RUN; survivors that produced a
    # score are the panel.
    assert result.metadata["eligible_count"] == 3


def test_panel_empty_when_all_judges_raise_floor_raises(
    scoring_attempt: AttemptState,
) -> None:
    """scoring-numerics-001 (the floor): if EVERY judge raises, zero scores
    survive — there is no verdict to give, so the existing empty-panel ValueError
    is the floor (SHARED CONTRACT: raise ONLY when zero judges survived)."""
    panel = JudgePanel(
        judges=[
            make_raising_judge("qwen", RuntimeError("down")),
            make_raising_judge("mistral", RuntimeError("also down")),
        ],
        reducer="majority",
        generator_family="claude",
    )
    with pytest.raises(ValueError, match="no .*judges|empty"):
        asyncio.run(panel.score(scoring_attempt))


def test_panel_median_degrades_over_survivors(
    scoring_attempt: AttemptState,
) -> None:
    """A numeric panel with one raising judge still reduces over the surviving
    finite votes (the median is taken over survivors, not poisoned to a crash)."""
    panel = JudgePanel(
        judges=[
            make_judge("qwen", 0.3),
            make_raising_judge("mistral", RuntimeError("timeout")),
            make_judge("cohere", 0.9),
        ],
        reducer="median",
        generator_family="claude",
    )
    result = asyncio.run(panel.score(scoring_attempt))
    # median of the two survivors {0.3, 0.9} -> 0.6
    assert result.value == pytest.approx(0.6)
    assert len(result.metadata["judges_errored"]) == 1


# --------------------------------------------------------------------------- #
# judge_panel — median reducer must not return an order-dependent NaN
# (scoring-numerics-002)
# --------------------------------------------------------------------------- #


def test_reduce_median_drops_non_finite_judge_value() -> None:
    """scoring-numerics-002 (RED→GREEN): a judge returning NaN must NOT silently
    poison the panel median (``statistics.median`` over a list containing NaN
    yields an order-dependent NaN). The reducer drops the non-finite vote and
    reduces over the finite peers, flagging the drop in metadata."""
    out = reduce_scores(
        [
            Score(value=0.4),
            Score(value=float("nan")),
            Score(value=0.8),
        ],
        "median",
    )
    import math

    assert math.isfinite(out.value)  # type: ignore[arg-type]
    # median over the finite survivors {0.4, 0.8} -> 0.6
    assert out.value == pytest.approx(0.6)
    # the dropped non-finite votes are flagged, not silently swallowed
    assert out.metadata["non_finite_dropped"] == 1


def test_reduce_median_inf_is_dropped() -> None:
    """An infinite judge value is non-finite too — dropped before the median."""
    out = reduce_scores(
        [Score(value=0.5), Score(value=float("inf")), Score(value=0.7)],
        "median",
    )
    import math

    assert math.isfinite(out.value)  # type: ignore[arg-type]
    assert out.value == pytest.approx(0.6)
    assert out.metadata["non_finite_dropped"] == 1


def test_reduce_median_all_non_finite_raises() -> None:
    """If EVERY judge value is non-finite there is no finite estimate to report —
    fail with a clear reason rather than returning a silent NaN."""
    with pytest.raises(ValueError, match="non-finite|finite"):
        reduce_scores([Score(value=float("nan")), Score(value=float("inf"))], "median")


# --------------------------------------------------------------------------- #
# oracle — grade() must reject a non-finite solve_quality / time_used
# (scoring-numerics-003, eval-integrity-adjacent)
# --------------------------------------------------------------------------- #


def test_grade_rejects_nan_solve_quality(
    scoring_attempt: AttemptState, scoring_meta: PuzzleMeta
) -> None:
    """scoring-numerics-003 (RED→GREEN): a NaN ``solve_quality`` currently passes
    every threshold comparison (``nan < 50`` is False) and emits a non-finite
    ``Score.value`` with NO failed condition. The gate must close with a
    ``non_finite_input`` condition and value 0.0 — a NaN never opens the gate."""
    import math

    outcome = _clean_outcome(solve_quality=float("nan"))
    score = grade(scoring_attempt, scoring_meta, outcome)
    assert score.metadata["gate_passed"] is False
    assert "non_finite_input" in score.metadata["failed_conditions"]
    assert score.value == 0.0
    assert math.isfinite(score.value)  # type: ignore[arg-type]


def test_grade_rejects_inf_time_used(
    scoring_attempt: AttemptState, scoring_meta: PuzzleMeta
) -> None:
    """scoring-numerics-003: an infinite ``time_used`` is non-finite too — the gate
    closes with ``non_finite_input`` (and ``over_time_budget`` may also fire; the
    point is the gate is CLOSED and the value is a finite 0.0)."""
    outcome = _clean_outcome(time_used=float("inf"))
    score = grade(scoring_attempt, scoring_meta, outcome)
    assert score.metadata["gate_passed"] is False
    assert "non_finite_input" in score.metadata["failed_conditions"]
    assert score.value == 0.0


def test_grade_finite_inputs_have_no_non_finite_condition(
    scoring_attempt: AttemptState, scoring_meta: PuzzleMeta
) -> None:
    """A normal (finite) clean solve must NOT spuriously carry non_finite_input."""
    score = grade(scoring_attempt, scoring_meta, _clean_outcome())
    assert "non_finite_input" not in score.metadata["failed_conditions"]


# --------------------------------------------------------------------------- #
# Epic-4 catalog primitives: Newcombe diff CI / Fisher p / BH-FDR / e-process
# --------------------------------------------------------------------------- #


def test_newcombe_diff_zero_when_rates_equal() -> None:
    """Equal proportions → delta 0 and a CI that straddles 0 (no difference)."""
    lower, upper, delta = newcombe_wilson_diff(10, 20, 10, 20)
    assert delta == 0.0
    assert lower < 0.0 < upper


def test_newcombe_diff_excludes_zero_for_strong_gap() -> None:
    """A strong gap (18/20 vs 2/20) yields a positive CI that excludes 0."""
    lower, upper, delta = newcombe_wilson_diff(18, 20, 2, 20)
    assert delta == pytest.approx(0.8)
    assert lower > 0.0  # the difference is real — CI excludes 0


def test_newcombe_diff_sign_flips_with_argument_order() -> None:
    """Swapping the two systems negates the delta and mirrors the interval."""
    lo1, up1, d1 = newcombe_wilson_diff(18, 20, 2, 20)
    lo2, up2, d2 = newcombe_wilson_diff(2, 20, 18, 20)
    assert d2 == pytest.approx(-d1)
    assert lo2 == pytest.approx(-up1)
    assert up2 == pytest.approx(-lo1)


def test_newcombe_diff_stays_in_unit_diff_range() -> None:
    """At the 20/20-vs-0/20 extreme the bounds stay within [-1, 1]."""
    lower, upper, delta = newcombe_wilson_diff(20, 20, 0, 20)
    assert delta == 1.0
    assert -1.0 <= lower <= upper <= 1.0


def test_newcombe_diff_rejects_bad_inputs() -> None:
    with pytest.raises(ValueError):
        newcombe_wilson_diff(5, 0, 1, 10)  # n1 = 0
    with pytest.raises(ValueError):
        newcombe_wilson_diff(11, 10, 1, 10)  # s1 > n1
    with pytest.raises(ValueError):
        newcombe_wilson_diff(5, 10, 1, 10, conf=1.5)  # bad conf


def test_fisher_difference_pvalue_near_one_for_equal() -> None:
    """Identical proportions → no evidence of a difference → p ≈ 1."""
    assert fisher_difference_pvalue(10, 20, 10, 20) == pytest.approx(1.0, abs=1e-9)


def test_fisher_difference_pvalue_small_for_strong_gap() -> None:
    """A strong gap is significant at conventional levels."""
    assert fisher_difference_pvalue(18, 20, 2, 20) < 0.001


def test_fisher_difference_pvalue_rejects_bad_inputs() -> None:
    with pytest.raises(ValueError):
        fisher_difference_pvalue(1, 0, 1, 10)
    with pytest.raises(ValueError):
        fisher_difference_pvalue(11, 10, 1, 10)


def test_benjamini_hochberg_empty_is_empty() -> None:
    assert benjamini_hochberg([], q=0.10) == []


def test_benjamini_hochberg_all_tiny_all_survive() -> None:
    survived = benjamini_hochberg([0.0001, 0.0002, 0.0003], q=0.10)
    assert survived == [True, True, True]


def test_benjamini_hochberg_none_survive_when_all_large() -> None:
    survived = benjamini_hochberg([0.9, 0.8, 0.95], q=0.10)
    assert survived == [False, False, False]


def test_benjamini_hochberg_step_up_rejects_below_threshold() -> None:
    """Classic step-up: with m=4, q=0.10 the sorted p's [0.005,0.02,0.03,0.5]
    have cutoffs 0.025,0.05,0.075,0.10 → the three small p's pass, 0.5 fails, and
    the result preserves INPUT order."""
    survived = benjamini_hochberg([0.5, 0.005, 0.03, 0.02], q=0.10)
    assert survived == [False, True, True, True]


def test_benjamini_hochberg_is_less_conservative_than_bonferroni() -> None:
    """A p above Bonferroni (q/m) but under its BH step still survives.

    m=3, q=0.30 → Bonferroni cut q/m=0.10; BH cuts are 0.10, 0.20, 0.30. The
    0.15 p-value is above Bonferroni's 0.10 but under its rank-2 BH cut (0.20),
    so BH rejects it where Bonferroni would not.
    """
    survived = benjamini_hochberg([0.05, 0.15, 0.9], q=0.30)
    assert survived[0] is True and survived[1] is True and survived[2] is False


def test_benjamini_hochberg_rejects_bad_inputs() -> None:
    with pytest.raises(ValueError):
        benjamini_hochberg([0.1], q=0.0)
    with pytest.raises(ValueError):
        benjamini_hochberg([1.2], q=0.10)


def test_eprocess_positive_for_high_success_run() -> None:
    """A near-saturated run (19/20) accumulates positive log-evidence for H1."""
    assert bernoulli_log_eprocess(19, 20, p0=0.90, p1=0.97) > 0.0


def test_eprocess_negative_for_low_success_run() -> None:
    """A low-success run is evidence AGAINST saturation (negative log-e)."""
    assert bernoulli_log_eprocess(5, 20, p0=0.90, p1=0.97) < 0.0


def test_eprocess_accumulates_across_runs_to_threshold() -> None:
    """Two clean 20/20 runs cross the demotion threshold ln(1/0.05)=ln(20);
    a single 20/20 run does not (dwell is emergent — CONTRACT §B)."""
    import math

    one = bernoulli_log_eprocess(20, 20, p0=0.90, p1=0.97)
    two = one + bernoulli_log_eprocess(20, 20, p0=0.90, p1=0.97)
    assert one < math.log(20.0)   # one lucky run cannot demote
    assert two >= math.log(20.0)  # sustained evidence does


def test_eprocess_expectation_is_one_at_h0_boundary() -> None:
    """e-value validity: E[e_run] = 1 under H0 at the boundary p=p0 (it is a test
    martingale). Sum over all outcomes of P(x|p0)·exp(log_e(x)) == 1."""
    import math

    from scipy.stats import binom

    n, p0, p1 = 8, 0.90, 0.97
    expectation = sum(
        binom.pmf(x, n, p0) * math.exp(bernoulli_log_eprocess(x, n, p0=p0, p1=p1))
        for x in range(n + 1)
    )
    assert expectation == pytest.approx(1.0, abs=1e-9)


def test_eprocess_rejects_bad_inputs() -> None:
    with pytest.raises(ValueError):
        bernoulli_log_eprocess(5, 0, p0=0.9, p1=0.97)
    with pytest.raises(ValueError):
        bernoulli_log_eprocess(5, 10, p0=0.0, p1=0.97)
    with pytest.raises(ValueError):
        bernoulli_log_eprocess(5, 10, p0=0.9, p1=1.0)
