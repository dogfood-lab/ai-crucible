"""Tests for the judge-profiling harness (research-grounding §11.1 + §11.4).

Covers the load-bearing behaviors the Phase-2 admission test pins, with **synthetic
:class:`~ai_crucible.characterize.types.JudgmentRecord`s** and — per the dogfood "prove the
gate goes RED" discipline — explicit RED proofs:

* the six §11.1 metrics compute the right numbers (accuracy math, a perfect-agreement
  set → r≈1 / κ high, ECE on a mis-calibrated set, consistency dropping when
  ``run_index`` verdicts flip — proven RED);
* the alt-test ω **seats** at ≥0.5 and **screens** below 0.5 — both proven, holding the
  judge's own records constant so ω is the only thing that moved;
* :func:`build_profile` yields **SEAT** for a strong synthetic judge and **REJECT** for a
  random one — both proven;
* the §11.4 ρ<0.25 submodularity gate **fails** when two judges' errors correlate (proven
  RED) and passes when they don't;
* :func:`minority_veto` escalates on a single credible bypass flag (and the
  reliability-weighted vote lets a reliable dissent beat an unreliable majority).

Every metric is pure + deterministic, so these tests need no seeds for the library
itself; where a fixture uses randomness it is seeded for reproducibility.
"""

from __future__ import annotations

import math

import pytest

from ai_crucible.characterize.aggregate import (
    SUBMODULARITY_THRESHOLD,
    SeatedPanel,
    VerdictLike,
    compose_panel,
    minority_veto,
    pairwise_error_correlation,
    passes_submodularity,
    redundant_pairs,
    reliability_weighted_vote,
    verdict_histogram,
)
from ai_crucible.characterize.metrics import (
    agreement,
    alt_test_omega,
    apply_temperature,
    consistency,
    difficulty_weighted_accuracy,
    effective_sample_size,
    expected_calibration_error,
    family_pref_delta,
    fit_temperature,
    kappa_zscore,
    objective_accuracy,
    position_bias,
    quality_score,
    temperature_scaled_ece,
    temperature_scaled_ece_cv,
    verbosity_bias,
)
from ai_crucible.characterize.profile import (
    SeatGates,
    _accuracy_wilson,
    build_profile,
    perturbation_audit,
)
from ai_crucible.characterize.types import (
    JudgeProfile,
    JudgmentRecord,
    RoleSlot,
    SeatDecision,
)
from ai_crucible.scoring.stats import wilson_interval

# --------------------------------------------------------------------------- #
# Synthetic-record builders
# --------------------------------------------------------------------------- #


def rec(
    item: str,
    predicted: object,
    gold: object,
    *,
    model: str = "cand",
    confidence: float | None = None,
    run_index: int = 0,
    position: int | None = None,
    family: str | None = None,
    **metadata: object,
) -> JudgmentRecord:
    """Terse :class:`JudgmentRecord` factory for the tests."""
    return JudgmentRecord(
        item_id=item,
        model_id=model,
        predicted=predicted,
        gold=gold,
        confidence=confidence,
        run_index=run_index,
        position=position,
        family=family,
        metadata=dict(metadata),
    )


def _gold(n: int, mod: int = 5) -> list[int]:
    """A repeatable gold rating sequence ``1..mod``."""
    return [(i % mod) + 1 for i in range(n)]


def strong_judge_records(
    n: int = 60, *, off_by_one_every: int = 6
) -> list[JudgmentRecord]:
    """A seat-worthy judge: high accuracy, ratings track gold, κ near the human baseline.

    Deviates by a single rating step on ``1/off_by_one_every`` of the items (and over
    three ``run_index`` passes, *consistently* — so test-retest is perfect). The small,
    bounded disagreement keeps Pearson r high while pulling κ down to a human-like level
    (so |κ z| < 1 against an 0.80 baseline) — i.e. it is *not* the unrealistically
    perfect judge the two-gate is designed to flag.
    """
    gold = _gold(n)

    def predicted(i: int) -> int:
        g = gold[i]
        return min(5, g + 1) if i % off_by_one_every == 0 else g

    return [
        rec(
            f"i{i}",
            predicted(i),
            gold[i],
            confidence=0.75 if predicted(i) == gold[i] else 0.45,
            run_index=run,
        )
        for run in range(3)
        for i in range(n)
    ]


def super_consistent_records(n: int = 60) -> list[JudgmentRecord]:
    """A super-consistent judge: PERFECT agreement (κ=1.0, z>1) over ``n`` items, 3 passes.

    This is exactly the case the OLD two-sided gate inverted — it screened this judge as
    "supra-human". Under the §12 one-sided gate (Han 2025 Tier-1B) it SEATS with a review
    flag. Three identical ``run_index`` passes make test-retest consistency perfect too.
    """
    gold = _gold(n)
    return [
        rec(f"i{i}", gold[i], gold[i], run_index=run)
        for run in range(3)
        for i in range(n)
    ]


def below_floor_kappa_records(n: int = 60) -> list[JudgmentRecord]:
    """A judge that correlates on the scale (high Pearson r) but whose categorical κ sits
    BELOW the one-sided human floor — it should REJECT on the κ floor (not merely screen).

    Off-by-one on ~half the items keeps r high (~0.94, monotone) while dragging κ to ~0.5,
    which is below the ``0.80 − 1.96·SE`` one-sided floor — the §12 hard agreement gate.
    """
    gold = _gold(n)
    return [
        rec(f"i{i}", (gold[i] if i % 2 == 0 else min(5, gold[i] + 1)), gold[i])
        for i in range(n)
    ]


def annotators(
    judge_label, human_label, n: int = 60
) -> dict[str, list[JudgmentRecord]]:
    """Build an alt-test ``{judge, h1, h2, h3}`` record set from two label functions."""
    gold = _gold(n)

    def mk(ann: str, fn) -> list[JudgmentRecord]:
        return [rec(f"i{i}", fn(i), gold[i], model=ann) for i in range(n)]

    return {
        "judge": mk("judge", judge_label),
        "h1": mk("h1", human_label),
        "h2": mk("h2", human_label),
        "h3": mk("h3", human_label),
    }


# --------------------------------------------------------------------------- #
# §11.1 #1 — objective accuracy
# --------------------------------------------------------------------------- #


def test_objective_accuracy_math() -> None:
    """Accuracy is exactly the fraction predicted == gold."""
    records = [rec("i0", 1, 1), rec("i1", 2, 2), rec("i2", 9, 3), rec("i3", 4, 4)]
    assert objective_accuracy(records) == 0.75  # 3 of 4


def test_objective_accuracy_uses_correct_flag_when_present() -> None:
    """When the scorer filled ``correct``, it is the source of truth (not predicted==gold)."""
    # predicted == gold would say "right", but the scorer marked it wrong.
    r = JudgmentRecord(item_id="i0", model_id="m", predicted=1, gold=1, correct=False)
    assert objective_accuracy([r]) == 0.0


def test_objective_accuracy_empty_raises() -> None:
    with pytest.raises(ValueError, match="at least one JudgmentRecord"):
        objective_accuracy([])


# --------------------------------------------------------------------------- #
# §11.1 #2 — agreement (Pearson r + Cohen's κ) and the κ z-score two-gate
# --------------------------------------------------------------------------- #


def test_perfect_agreement_gives_r_one_kappa_high() -> None:
    """A perfect-agreement set → r ≈ 1.0 and κ ≈ 1.0."""
    records = [rec(f"i{i}", v, v) for i, v in enumerate([1, 2, 3, 4, 5] * 4)]
    r, kappa = agreement(records)
    assert r == pytest.approx(1.0)
    assert kappa == pytest.approx(1.0)


def test_constant_predictor_has_zero_r() -> None:
    """A judge that gives every item the same score has no linear agreement (r = 0.0),
    which correctly fails the r ≥ 0.80 gate."""
    records = [rec(f"i{i}", 3, g) for i, g in enumerate([1, 2, 3, 4, 5])]
    r, _ = agreement(records)
    assert r == 0.0


def test_agreement_single_identical_category_kappa_one() -> None:
    """One shared category (0/0 chance variance) → κ maps to 1.0, not NaN."""
    _, kappa = agreement([rec("i0", 1, 1)])
    assert kappa == 1.0


def test_agreement_non_numeric_raises() -> None:
    with pytest.raises(ValueError, match="numeric predicted/gold"):
        agreement([rec("i0", "yes", "no")])


def test_kappa_zscore_human_like_band() -> None:
    """The signed κ z-score: a judge whose κ sits near the human baseline has |z| < 1;
    a κ far above the baseline has z > 1 ('super-consistent').

    Note the §12 reversal lives in the *profile* layer, not here: this metric just reports
    the signed z. Under §12 a z > 1 judge is Tier-1B (Han 2025) — it SEATS with a review
    flag (see ``test_build_profile_seats_super_consistent_judge_with_review_flag``); the
    old two-sided ``|z| < 1`` screen was the inversion that has been removed."""
    n = 60
    # κ = 0.82 vs baseline 0.80 over 60 items → small |z| (human-like).
    z_close = kappa_zscore(0.82, 0.80, n)
    assert abs(z_close) < 1.0
    # κ = 1.0 (agrees with gold far more than humans agree with each other) → z > 1
    # (super-consistent → review flag in the gate, NOT a screen).
    z_far = kappa_zscore(1.0, 0.80, n)
    assert z_far > 1.0


def test_kappa_zscore_degenerate_baseline_returns_zero() -> None:
    """A baseline at 0 or 1 leaves no scale → z is 0.0 (cannot flag as non-human)."""
    assert kappa_zscore(0.9, 1.0, 50) == 0.0
    assert kappa_zscore(0.1, 0.0, 50) == 0.0


def test_kappa_zscore_bad_args_raise() -> None:
    with pytest.raises(ValueError, match="n must be >= 1"):
        kappa_zscore(0.8, 0.8, 0)
    with pytest.raises(ValueError, match=r"human_human_kappa must be in"):
        kappa_zscore(0.8, 1.5, 50)


# --------------------------------------------------------------------------- #
# §11.1 #3 — alt-test substitution ω
# --------------------------------------------------------------------------- #


def test_alt_test_omega_seats_when_judge_matches_humans() -> None:
    """When the judge agrees with the held-out human at least as well as the other
    humans do (here: everyone shares the same labels), ω = 1.0 → seat-eligible."""
    gold = _gold(30)
    rpa = annotators(lambda i: gold[i], lambda i: gold[i], n=30)
    assert alt_test_omega(rpa) == 1.0


def test_alt_test_omega_screens_when_judge_breaks_consensus() -> None:
    """When the humans share a consensus the judge does not replicate, the judge loses
    every leave-one-out fold → ω = 0.0 < 0.5 → screen-only."""
    gold = _gold(30)
    soft = {i for i in range(30) if i % 3 == 0}
    judge = lambda i: (min(5, gold[i] + 1) if i in soft else gold[i])  # noqa: E731
    human = lambda i: gold[i]  # noqa: E731
    rpa = annotators(judge, human, n=30)
    omega = alt_test_omega(rpa)
    assert omega < 0.5


def test_alt_test_omega_requires_judge_and_two_humans() -> None:
    gold = _gold(5)
    one_human = {
        "judge": [rec("i0", gold[0], gold[0])],
        "h1": [rec("i0", gold[0], gold[0])],
    }
    with pytest.raises(ValueError, match=">= 2 human annotators"):
        alt_test_omega(one_human)
    with pytest.raises(ValueError, match="'judge' entry"):
        alt_test_omega({"h1": [], "h2": []})


# --------------------------------------------------------------------------- #
# §11.1 #4 — consistency (test-retest) — prove it drops RED on a flip
# --------------------------------------------------------------------------- #


def test_consistency_perfect_when_no_flips() -> None:
    """Repeated passes with identical verdicts → consistency 1.0."""
    records = [
        rec("i0", 1, 1, run_index=0),
        rec("i0", 1, 1, run_index=1),
        rec("i1", 2, 2, run_index=0),
        rec("i1", 2, 2, run_index=1),
    ]
    assert consistency(records) == 1.0


def test_consistency_drops_when_run_index_verdict_flips_RED() -> None:
    """RED proof: when one of two repeated items flips its verdict across ``run_index``,
    consistency drops to 0.5 (Haldar & Hockenmaier 2025 intra-rater unreliability)."""
    records = [
        rec("i0", 1, 1, run_index=0),
        rec("i0", 1, 1, run_index=1),  # stable
        rec("i1", 1, 1, run_index=0),
        rec("i1", 0, 1, run_index=1),  # FLIP
    ]
    assert consistency(records) == 0.5


def test_consistency_not_measured_without_repeats_is_one() -> None:
    """Items seen once carry no test-retest signal → consistency defaults to 1.0
    (the profile layer records 'not measured')."""
    assert consistency([rec("i0", 1, 1), rec("i1", 2, 2)]) == 1.0


# --------------------------------------------------------------------------- #
# §11.1 #5 — ECE on a mis-calibrated set
# --------------------------------------------------------------------------- #


def test_ece_zero_when_perfectly_calibrated() -> None:
    """Confidence 0.5 with 50% accuracy in that bin → ECE 0.0."""
    records = [rec(f"i{i}", v, 1, confidence=0.5) for i, v in enumerate([1, 0] * 10)]
    assert expected_calibration_error(records) == pytest.approx(0.0)


def test_ece_high_for_overconfident_wrong_judge() -> None:
    """RED-ish proof: a judge that is 95% confident but always wrong has ECE ≈ 0.95."""
    records = [rec(f"i{i}", 0, 1, confidence=0.95) for i in range(20)]
    assert expected_calibration_error(records) == pytest.approx(0.95, abs=1e-9)


def test_ece_returns_none_without_confidence() -> None:
    """§12: ECE returns ``None`` ('calibration not measured') — NOT an error — when no
    record carries a confidence. The verbalized number is untrustworthy (Xiong 2024) and
    the logprob/vote-fraction channel may simply not be wired this run, so a missing
    confidence signal must not fail a judge (the gate treats None as a neutral)."""
    assert expected_calibration_error([rec("i0", 1, 1)]) is None
    # A mix where *some* records carry a confidence still computes (the None-confidence
    # records are ignored, the rest scored).
    mixed = [rec("i0", 1, 1, confidence=0.5), rec("i1", 0, 1)]
    assert expected_calibration_error(mixed) is not None


def test_ece_rejects_out_of_range_confidence() -> None:
    with pytest.raises(ValueError, match=r"confidence must be in"):
        expected_calibration_error([rec("i0", 1, 1, confidence=1.5)])


# --------------------------------------------------------------------------- #
# §11.1 #6 — bias panel (position / verbosity / family-pref)
# --------------------------------------------------------------------------- #


def test_position_bias_flip_rate() -> None:
    """Half the items flip their verdict between position 0 and 1 → bias 0.5."""
    records = [
        rec("i0", "A", "A", position=0),
        rec("i0", "B", "A", position=1),  # flips with the swap
        rec("i1", "A", "A", position=0),
        rec("i1", "A", "A", position=1),  # stable
    ]
    assert position_bias(records) == 0.5


def test_position_bias_zero_without_swaps() -> None:
    """No item seen in ≥2 positions → 0.0 (not flagged; 'not measured')."""
    assert position_bias([rec("i0", "A", "A", position=0)]) == 0.0


def test_verbosity_bias_detects_length_driven_error() -> None:
    """When the judge's error grows with answer length, |r| is high (verbose-biased)."""
    # error == +1 scales with length; length and error perfectly correlated.
    records = [
        rec(f"i{i}", g + 1, g, answer_len=10 * (i + 1))
        for i, g in enumerate([1, 2, 3, 4, 5])
    ]
    # All errors are +1 here → zero variance in error → no measurable association → 0.0.
    assert verbosity_bias(records) == 0.0
    # Now make error track length: longer answers over-rated more.
    biased = [
        rec("a", 2, 1, answer_len=10),
        rec("b", 2, 1, answer_len=12),
        rec("c", 4, 1, answer_len=80),
        rec("d", 5, 1, answer_len=100),
    ]
    assert verbosity_bias(biased) > 0.8


def test_family_pref_delta_positive_when_kin_favored() -> None:
    """Judge is accurate on its own family's outputs and wrong on others → Δ = +1.0."""
    records = [
        rec(f"s{i}", 1, 1, judged_family="qwen", judge_family="qwen") for i in range(5)
    ] + [
        rec(f"d{i}", 0, 1, judged_family="llama", judge_family="qwen") for i in range(5)
    ]
    assert family_pref_delta(records) == pytest.approx(1.0)


def test_family_pref_delta_zero_without_both_groups() -> None:
    """Only same-family records → no contrast → 0.0."""
    records = [rec("s0", 1, 1, judged_family="qwen", judge_family="qwen")]
    assert family_pref_delta(records) == 0.0


def test_family_pref_delta_normalizes_family_casing() -> None:
    """Casing/whitespace drift between the two family-tag sites must NOT split a
    same-family pair into the different-family bucket (sibling of scoring-stats-002).

    The judged-output family and the judge's own family are stamped by independent
    sites; ``"Qwen"`` vs ``"qwen"`` is the same family. A raw ``judged == own``
    compare would route the (correct) same-family records into ``diff``, empty the
    ``same`` group, and return 0.0 — silently MASKING kin-bias the cross-family
    panel exists to detect. With casefold/strip normalization the contrast is real
    (Δ = +1.0). RED against the raw comparison, GREEN after.
    """
    records = [
        rec(f"s{i}", 1, 1, judged_family="Qwen", judge_family="qwen") for i in range(5)
    ] + [
        rec(f"d{i}", 0, 1, judged_family="llama", judge_family="qwen") for i in range(5)
    ]
    assert family_pref_delta(records) == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# build_profile — SEAT for a strong judge, REJECT for a random one (prove both)
# --------------------------------------------------------------------------- #


def test_build_profile_seats_a_strong_judge() -> None:
    """A strong synthetic judge (margin over chance, r ≥ 0.80, κ human-like, consistent,
    low bias, ω ≥ 0.5) is SEATed with a meaningful reliability weight."""
    records = strong_judge_records()
    gold = _gold(60)
    rpa = annotators(lambda i: gold[i], lambda i: gold[i])  # ω = 1.0
    profile = build_profile(
        "qwen3.6:32b",
        RoleSlot.JUDGE,
        records,
        human_human_kappa=0.80,
        records_per_annotator=rpa,
        quant="Q5_K_M",
    )
    assert isinstance(profile, JudgeProfile)
    assert profile.seat_decision is SeatDecision.SEAT
    assert profile.objective_accuracy is not None and profile.objective_accuracy > 0.8
    assert profile.agreement_r is not None and profile.agreement_r >= 0.80
    assert abs(profile.kappa_z) < 1.0  # human-like
    assert profile.alt_test_omega == 1.0
    assert profile.reliability_weight is not None and profile.reliability_weight > 0.3
    assert profile.quant == "Q5_K_M"
    # provenance: thresholds stamped for replayability (PIN_PER_STEP)
    assert profile.metadata["thresholds"]["agreement_r"] == 0.80
    assert any("SEAT" in note for note in profile.notes)


def test_build_profile_rejects_a_random_judge() -> None:
    """RED proof: a judge whose predictions are independent of gold fails the hard gates
    (no accuracy margin and/or r < 0.80) → REJECT with zero weight."""
    import random

    rng = random.Random(20)
    gold = _gold(60)
    records = [
        rec(f"i{i}", rng.randint(1, 5), gold[i], confidence=0.9) for i in range(60)
    ]
    profile = build_profile("random-7b", RoleSlot.JUDGE, records, human_human_kappa=0.80)
    assert profile.seat_decision is SeatDecision.REJECT
    assert profile.reliability_weight == 0.0
    assert any("REJECT" in note for note in profile.notes)


def test_build_profile_screens_when_alt_test_fails_only() -> None:
    """Holding the (seat-worthy) judge records constant, swapping the annotator set so
    ω < 0.5 flips SEAT → SCREEN — proving the alt-test gate is what moved the decision
    (Calderon et al. 2025: ω < 0.5 ⇒ substitution worse than a human ⇒ screen-only)."""
    records = strong_judge_records()
    gold = _gold(60)
    soft = {i for i in range(60) if i % 3 == 0}
    judge_alt = lambda i: (min(5, gold[i] + 1) if i in soft else gold[i])  # noqa: E731
    rpa_screen = annotators(judge_alt, lambda i: gold[i])  # humans share gold; judge breaks it

    seated = build_profile(
        "q", RoleSlot.JUDGE, records, human_human_kappa=0.80,
        records_per_annotator=annotators(lambda i: gold[i], lambda i: gold[i]),
    )
    screened = build_profile(
        "q", RoleSlot.JUDGE, records, human_human_kappa=0.80,
        records_per_annotator=rpa_screen,
    )
    assert seated.seat_decision is SeatDecision.SEAT
    assert screened.seat_decision is SeatDecision.SCREEN
    assert screened.alt_test_omega is not None and screened.alt_test_omega < 0.5
    # SCREEN down-weights (verdicts escalate, Jung 2024) below the SEAT weight.
    assert screened.reliability_weight < seated.reliability_weight
    assert any("ω<0.50" in note for note in screened.notes)


def test_build_profile_bias_penalizes_quality_score() -> None:
    """§12: bias is folded into the continuous quality score (a multiplicative penalty),
    NOT a discrete SCREEN gate. A position-decided judge has its score dragged down by the
    bias penalty; holding everything else fixed, a high-bias judge scores strictly below
    its low-bias twin and the selective CI decision moves accordingly (here: below the
    floor → REJECT, proving the penalty bites)."""
    gold = _gold(40)
    biased: list[JudgmentRecord] = []
    clean: list[JudgmentRecord] = []
    for i in range(40):
        g = gold[i]
        p = min(5, g + 1) if i % 6 == 0 else g
        # Biased judge: half the items flip their verdict with the position swap.
        biased.append(rec(f"i{i}", p, g, position=0, run_index=0))
        flip = min(5, g + 1) if i % 2 == 0 else p
        biased.append(rec(f"i{i}", flip, g, position=1, run_index=0))
        # Clean twin: identical verdicts, position-invariant (no bias).
        clean.append(rec(f"i{i}", p, g, position=0, run_index=0))
        clean.append(rec(f"i{i}", p, g, position=1, run_index=0))

    biased_p = build_profile("biased", RoleSlot.JUDGE, biased, human_human_kappa=0.80)
    clean_p = build_profile("clean", RoleSlot.JUDGE, clean, human_human_kappa=0.80)

    assert biased_p.position_bias is not None and biased_p.position_bias >= 0.25
    assert clean_p.position_bias == 0.0
    # The bias penalty strictly lowers the continuous score (§12 quality_score).
    assert (
        biased_p.metadata["quality_score"] < clean_p.metadata["quality_score"]
    )
    # And it moves the selective decision: the biased judge no longer seats.
    assert biased_p.seat_decision is not SeatDecision.SEAT
    assert any("bias" in note for note in biased_p.notes)


def test_bias_panel_reports_not_measured_when_no_trials_staged() -> None:
    """characterize-001: in the default run (no bias trials staged) the bias panel must
    report "not measured" — NOT "measured 0.000 → PASS".

    collect_records never sets ``position``/``answer_len``/``judged_family``, so the three
    bias metrics are definitionally 0.0 and the bias gate is structurally inert. Unlike
    consistency and ECE — which emit explicit "not measured" branches — the bias panel
    formatted the three zeros as a passing measured gate, so a reviewer reads "this judge has
    no position/verbosity/self-preference bias" when the instrument never staged a single
    bias trial. The fix is honesty-only (do NOT fabricate trials): when no bias signal is
    present, the note must say "not measured" and the JudgeProfile bias fields must be None
    so a consumer cannot read 0.0 as "unbiased". The kin-/position-bias path (with trials)
    must still report a measured number."""
    # A seat-worthy judge with NO bias trials staged (no position/answer_len/family tags).
    records = strong_judge_records()
    profile = build_profile("no-bias-trials", RoleSlot.JUDGE, records, human_human_kappa=0.80)

    # The note must say bias was not measured — not present a passing measured gate.
    assert any(
        "bias not measured" in note.lower() for note in profile.notes
    ), profile.notes
    assert not any(
        "→ max 0.000" in note or "-> max 0.000" in note for note in profile.notes
    ), "inert bias must not be formatted as a measured 0.000 PASS"
    # The JudgeProfile bias fields are None (not 0.0) so a report consumer cannot misread
    # 0.0 as "measured, unbiased".
    assert profile.position_bias is None
    assert profile.verbosity_bias is None
    assert profile.family_pref_delta is None

    # Contrast: when position-swap trials ARE staged, the bias panel reports a measured
    # number (a float), not "not measured".
    gold = _gold(40)
    staged: list[JudgmentRecord] = []
    for i in range(40):
        g = gold[i]
        p = min(5, g + 1) if i % 6 == 0 else g
        staged.append(rec(f"i{i}", p, g, position=0, run_index=0))
        flip = min(5, g + 1) if i % 2 == 0 else p
        staged.append(rec(f"i{i}", flip, g, position=1, run_index=0))
    measured = build_profile("biased", RoleSlot.JUDGE, staged, human_human_kappa=0.80)
    assert measured.position_bias is not None  # a real measured value
    assert not any("bias not measured" in note.lower() for note in measured.notes)


def test_build_profile_empty_records_raises() -> None:
    with pytest.raises(ValueError, match="at least one JudgmentRecord"):
        build_profile("x", RoleSlot.JUDGE, [])


def test_seat_gates_are_overridable_and_stamped() -> None:
    """Custom thresholds are honored and recorded in metadata (provenance)."""
    records = strong_judge_records()
    strict = SeatGates(agreement_r=0.999)  # impossibly strict r gate
    profile = build_profile("q", RoleSlot.JUDGE, records, gates=strict)
    # r is high (~0.97) but below 0.999 → REJECT on the hard r gate.
    assert profile.seat_decision is SeatDecision.REJECT
    assert profile.metadata["thresholds"]["agreement_r"] == 0.999


# --------------------------------------------------------------------------- #
# §12 — the INVERTED-gate fix: super-consistent SEATs; one-sided κ floor;
#       continuous quality score; selective CI decision; perturbation audit
# --------------------------------------------------------------------------- #


def test_build_profile_seats_super_consistent_judge_with_review_flag() -> None:
    """THE §12 fix (flips the now-wrong test). A super-consistent judge — κ=1.0, z>1, i.e.
    it agrees with gold MORE than humans agree with each other — is **SEATED** with a
    ``review_flag``, NOT screened.

    Han et al. 2025 ("Judge's Verdict", arXiv:2510.09738) classifies such judges as
    **Tier-1B: valid, top-ranked, kept** — its own four highest-κ models are z>1. The old
    two-sided ``|z| < 1`` gate screened exactly these (the first characterization run
    wrongly screened the three κ=1.0 models); §12 reverses it to a one-sided floor. The
    review flag is for *later* human review only (IF κ≈1.0 co-occurs with high human
    disagreement) — it does not change the seat decision."""
    records = super_consistent_records()
    profile = build_profile("qwen3.6:27b", RoleSlot.JUDGE, records, human_human_kappa=0.80)
    # The decisive assertion: κ=1.0 now SEATS (it used to SCREEN).
    assert profile.seat_decision is SeatDecision.SEAT
    assert profile.kappa_z > 1.0  # super-consistent, above the human baseline
    # Seated WITH a review flag (a flag for later human review, not a downgrade).
    assert profile.metadata["review_flag"] is True
    assert "review_reason" in profile.metadata
    assert profile.reliability_weight is not None and profile.reliability_weight > 0.5
    # The notes must say it seated *because* it's Tier-1B, not despite it.
    assert any("review_flag SET" in note for note in profile.notes)
    assert any(
        "SEAT" in note and "Tier-1B" in note for note in profile.notes
    )


def test_human_like_judge_seats_without_review_flag() -> None:
    """Contrast: a judge whose κ sits *within* the human band (|z| < 1) also SEATS, but
    with NO review flag — only the super-consistent upper tail is flagged."""
    records = strong_judge_records()
    gold = _gold(60)
    rpa = annotators(lambda i: gold[i], lambda i: gold[i])
    profile = build_profile(
        "human-like", RoleSlot.JUDGE, records, human_human_kappa=0.80,
        records_per_annotator=rpa,
    )
    assert profile.seat_decision is SeatDecision.SEAT
    assert abs(profile.kappa_z) < 1.0
    assert profile.metadata["review_flag"] is False
    assert "review_reason" not in profile.metadata


def test_build_profile_rejects_below_floor_kappa() -> None:
    """§12 one-sided floor as a HARD gate: a judge that correlates on the rating scale
    (high Pearson r) but whose categorical κ sits below ``baseline − margin`` is REJECTed
    on the κ floor — high r alone does not save it (the floor is the §11.1 #2 second gate,
    now one-sided rather than two-sided)."""
    records = below_floor_kappa_records()
    profile = build_profile("weakcat", RoleSlot.JUDGE, records, human_human_kappa=0.80)
    assert profile.agreement_r is not None and profile.agreement_r >= 0.80  # r passes
    assert profile.metadata["kappa"] < profile.metadata["kappa_one_sided_floor"]
    assert profile.seat_decision is SeatDecision.REJECT
    assert any("one-sided floor" in note for note in profile.notes)


def test_one_sided_floor_does_not_screen_the_upper_tail() -> None:
    """The directional property of the fix: raising κ from human-like to super-consistent
    NEVER worsens the decision. A perfect judge does at least as well as a human-like one
    (both SEAT) — proving the gate is one-sided (a ceiling would have flipped the perfect
    judge to SCREEN, which was the bug)."""
    human_like = build_profile(
        "hl", RoleSlot.JUDGE, strong_judge_records(), human_human_kappa=0.80
    )
    super_c = build_profile(
        "sc", RoleSlot.JUDGE, super_consistent_records(), human_human_kappa=0.80
    )
    assert human_like.seat_decision is SeatDecision.SEAT
    assert super_c.seat_decision is SeatDecision.SEAT
    # Higher κ ⇒ at least as high a quality score (monotone, never penalized for being
    # "too good").
    assert (
        super_c.metadata["quality_score"] >= human_like.metadata["quality_score"]
    )


# --- §12 Q4: difficulty-normalized continuous quality score ------------------ #


def test_difficulty_weighted_accuracy_unweighted_without_difficulty() -> None:
    """With no ``metadata['difficulty']`` it degrades to plain accuracy (the trivial-anchor
    path must not penalize the judge)."""
    records = [rec("i0", 1, 1), rec("i1", 2, 2), rec("i2", 9, 3), rec("i3", 4, 4)]
    assert difficulty_weighted_accuracy(records) == pytest.approx(0.75)
    assert difficulty_weighted_accuracy(records) == objective_accuracy(records)


def test_difficulty_weighted_accuracy_weights_hard_items() -> None:
    """Getting the HARD item right is worth more than getting an easy one right — and
    vice-versa — so the weighted accuracy differs from raw accuracy on a saturating-style
    set (§12: raw accuracy is distorted on easy-only sets)."""
    # Right on the easy item (w=1), wrong on the hard item (w=9): raw acc 0.5, but the
    # hard miss dominates the weighted score → well below 0.5.
    hard_miss = [
        rec("easy", 1, 1, difficulty=1.0),
        rec("hard", 9, 2, difficulty=9.0),
    ]
    assert objective_accuracy(hard_miss) == pytest.approx(0.5)
    assert difficulty_weighted_accuracy(hard_miss) == pytest.approx(0.1)  # 1/(1+9)
    # Mirror: right on the hard item, wrong on the easy one → weighted score well above.
    hard_hit = [
        rec("easy", 9, 1, difficulty=1.0),
        rec("hard", 2, 2, difficulty=9.0),
    ]
    assert difficulty_weighted_accuracy(hard_hit) == pytest.approx(0.9)


def test_effective_sample_size_uniform_equals_n_items() -> None:
    """No difficulty signal ⇒ uniform weights ⇒ n_eff == n_items (the correction is a no-op
    on the common, unweighted trivial-anchor path)."""
    records = [rec(f"i{i}", 1, 1) for i in range(8)]
    assert effective_sample_size(records) == pytest.approx(8.0)
    # Reruns of the same item are NOT independent trials — n_eff counts distinct items.
    with_reruns = [rec("i0", 1, 1, run_index=k) for k in range(3)] + [rec("i1", 1, 1)]
    assert effective_sample_size(with_reruns) == pytest.approx(2.0)


def test_effective_sample_size_drops_under_heterogeneous_weights() -> None:
    """Kish design effect: 5 heavy items (w=9) + 5 light (w=1) ⇒ n_eff = (Σw)²/Σw² =
    50²/410 ≈ 6.1, well below the 10 nominal items — the concentration that must widen the CI."""
    records = [rec(f"hard{i}", 1, 1, difficulty=9.0) for i in range(5)]
    records += [rec(f"easy{i}", 0, 1, difficulty=1.0) for i in range(5)]
    assert effective_sample_size(records) == pytest.approx(2500 / 410)


def test_weighted_accuracy_wilson_uses_effective_n_not_fabricated_count() -> None:
    """Eval-integrity regression (characterize HIGH): a judge answering at CHANCE on raw items
    (5/10) that happens to get the few HEAVY items right has difficulty-weighted accuracy 0.9 —
    but with only n_eff≈6.1 effective items the Wilson-95 lower bound is < 0.5, so it earns NO
    confident margin over chance. The pre-fix code fed the FABRICATED count round(0.9·10)=9 to
    Wilson(9,10) (lower≈0.60 > 0.5) and would have admitted it on the accuracy gate.
    """
    # 5 heavy items answered correctly (predicted==gold), 5 light items wrong (predicted=0≠gold).
    correct_hard = [rec(f"hard{i}", (i % 5) + 1, (i % 5) + 1, difficulty=9.0) for i in range(5)]
    wrong_easy = [rec(f"easy{i}", 0, 9, difficulty=1.0) for i in range(5)]
    records = correct_hard + wrong_easy
    assert objective_accuracy(records) == pytest.approx(0.5)
    assert difficulty_weighted_accuracy(records) == pytest.approx(0.9)

    # The FIX: Wilson at the effective N ⇒ lower bound below the 0.5 chance floor.
    acc_lo_fixed, _ = _accuracy_wilson(0.9, effective_sample_size(records))
    assert acc_lo_fixed < 0.50
    # The BUG it replaces: the fabricated binomial count round(0.9·10)=9 at the full n_items.
    acc_lo_buggy, _ = wilson_interval(9, 10, conf=0.95)
    assert acc_lo_buggy > 0.50

    # Integration: build_profile records the CORRECTED lower bound → no accuracy margin.
    prof = build_profile("coinflip-hardhit", RoleSlot.JUDGE, records, human_human_kappa=0.80)
    assert prof.metadata["accuracy_wilson_lower"] < 0.50
    assert prof.seat_decision is SeatDecision.REJECT


def test_quality_score_is_monotonic_in_accuracy() -> None:
    """§12 Q4: the continuous quality score is non-decreasing in accuracy (the property the
    selective-CI decision relies on — a higher accuracy bound ⇒ a higher score bound)."""
    accs = [i / 20 for i in range(21)]  # 0.0 .. 1.0
    scores = [
        quality_score(
            accuracy=a, agreement_r=0.9, consistency=1.0, ece=None, max_bias=0.0
        )
        for a in accs
    ]
    assert scores == sorted(scores)  # monotone non-decreasing
    assert scores[0] >= 0.0 and scores[-1] <= 1.0  # bounded


def test_quality_score_penalties_lower_the_score() -> None:
    """ECE (past the soft ceiling) and bias each strictly lower the score; ECE=None applies
    no calibration penalty (§12)."""
    base = quality_score(
        accuracy=0.9, agreement_r=0.9, consistency=1.0, ece=None, max_bias=0.0
    )
    with_bias = quality_score(
        accuracy=0.9, agreement_r=0.9, consistency=1.0, ece=None, max_bias=0.4
    )
    with_ece = quality_score(
        accuracy=0.9, agreement_r=0.9, consistency=1.0, ece=0.6, max_bias=0.0
    )
    assert with_bias < base
    assert with_ece < base
    # ECE below the soft ceiling applies no penalty (calibration is soft, §11.1).
    calibrated = quality_score(
        accuracy=0.9, agreement_r=0.9, consistency=1.0, ece=0.05, max_bias=0.0
    )
    assert calibrated == pytest.approx(base)


# --- §12: selective (CI-based) seat / screen / reject ------------------------ #


def test_selective_decision_screens_on_straddling_ci() -> None:
    """§12 selective decision: when the quality-score CI STRADDLES the floor (genuine
    uncertainty, not failure), the judge SCREENS rather than being forced to a pass/fail.

    A short, mid-rate set gives a wide accuracy CI; choosing a floor that lands inside the
    score CI exercises the straddle → SCREEN (escalate, Jung 2024)."""
    # 12 items, ~75% correct → wide Wilson CI on accuracy → wide score CI.
    gold = _gold(12)
    records = [
        rec(f"i{i}", (gold[i] if i % 4 != 0 else min(5, gold[i] + 1)), gold[i])
        for i in range(12)
    ]
    point = build_profile("mid", RoleSlot.JUDGE, records, human_human_kappa=0.80)
    lo, hi = point.metadata["score_ci"]
    # Place the floor strictly inside the score CI → the CI straddles it.
    straddle_floor = (lo + hi) / 2
    gates = SeatGates(quality_floor=straddle_floor)
    profile = build_profile("mid", RoleSlot.JUDGE, records, human_human_kappa=0.80, gates=gates)
    assert profile.seat_decision is SeatDecision.SCREEN
    s_lo, s_hi = profile.metadata["score_ci"]
    assert s_lo < straddle_floor <= s_hi  # the defining straddle condition
    assert any("straddles" in note for note in profile.notes)


def test_selective_decision_seat_screen_reject_by_floor() -> None:
    """Sweeping only the floor across the same judge's fixed score CI walks the decision
    SEAT → SCREEN → REJECT, proving the selective rule is the CI-vs-floor comparison."""
    records = strong_judge_records()
    base = build_profile("q", RoleSlot.JUDGE, records, human_human_kappa=0.80)
    lo, hi = base.metadata["score_ci"]

    seat = build_profile(
        "q", RoleSlot.JUDGE, records, human_human_kappa=0.80,
        gates=SeatGates(quality_floor=max(0.01, lo - 0.05)),  # floor below CI → SEAT
    )
    screen = build_profile(
        "q", RoleSlot.JUDGE, records, human_human_kappa=0.80,
        gates=SeatGates(quality_floor=(lo + hi) / 2),         # floor inside CI → SCREEN
    )
    reject = build_profile(
        "q", RoleSlot.JUDGE, records, human_human_kappa=0.80,
        gates=SeatGates(quality_floor=min(1.0, hi + 0.05)),   # floor above CI → REJECT
    )
    assert seat.seat_decision is SeatDecision.SEAT
    assert screen.seat_decision is SeatDecision.SCREEN
    assert reject.seat_decision is SeatDecision.REJECT


# --- §12 / §8.3: perturbation audit (Alzahrani lens on the admission gate) --- #


def test_perturbation_audit_reports_flip_rate_stable_decision() -> None:
    """A rock-solid seat (perfect judge) does not flip under ±1 SE threshold jitter →
    flip_rate 0.0 over a full ±SE sweep of every threshold (§12 / Alzahrani 2024)."""
    records = super_consistent_records()
    audit = perturbation_audit(records, human_human_kappa=0.80)
    assert audit["baseline_decision"] == "seat"
    assert audit["flip_rate"] == 0.0
    assert audit["n_perturbations"] > 0
    assert audit["flips"] == []


def test_perturbation_audit_detects_fragile_decision() -> None:
    """RED-ish proof: when a threshold is parked right at the decision boundary, jitter
    flips the decision → a non-zero flip_rate that names the fragile knob (the andon
    signal §12 wants — a brittle admission threshold is measuring noise, not the judge)."""
    records = strong_judge_records()
    base = build_profile("q", RoleSlot.JUDGE, records, human_human_kappa=0.80)
    lo, _hi = base.metadata["score_ci"]
    # Park the quality floor exactly on the CI lower bound: a downward nudge seats, an
    # upward nudge screens → the decision is fragile to floor jitter.
    knife_edge = SeatGates(quality_floor=lo)
    audit = perturbation_audit(
        records, human_human_kappa=0.80, gates=knife_edge
    )
    assert audit["flip_rate"] > 0.0
    assert any(f["threshold"] == "quality_floor" for f in audit["flips"])


def test_perturbation_audit_empty_records_raises() -> None:
    with pytest.raises(ValueError, match="at least one JudgmentRecord"):
        perturbation_audit([])


# --------------------------------------------------------------------------- #
# §11.4 — submodularity gate (prove RED), reliability-weighted vote, minority veto
# --------------------------------------------------------------------------- #


def _judge_with_errors(wrong_items: set[int], n: int = 30) -> list[JudgmentRecord]:
    """A judge that is wrong (by +1) exactly on ``wrong_items`` (else matches gold)."""
    gold = _gold(n)
    return [
        rec(f"i{i}", (min(5, gold[i] + 1) if i in wrong_items else gold[i]), gold[i])
        for i in range(n)
    ]


def test_pairwise_error_correlation_requires_two_judges() -> None:
    with pytest.raises(ValueError, match=">= 2 judges"):
        pairwise_error_correlation({"only": _judge_with_errors({1})})


def test_submodularity_gate_fails_when_errors_correlate_RED() -> None:
    """RED proof: two judges that make nearly the *same* errors correlate well above
    0.25 → the panel FAILS the §11.4 submodularity gate ('two judges are really one')."""
    a = _judge_with_errors({1, 2, 3, 4, 5, 6, 7, 8})
    b = _judge_with_errors({1, 2, 3, 4, 5, 6, 7, 9})  # 7/8 shared error items
    corr = pairwise_error_correlation({"A": a, "B": b})
    rho = corr[("A", "B")]
    assert rho > SUBMODULARITY_THRESHOLD
    assert passes_submodularity(corr) is False
    assert redundant_pairs(corr) == [("A", "B")]


def test_submodularity_gate_passes_when_errors_independent() -> None:
    """Disjoint error sets → low correlation → the gate passes."""
    a = _judge_with_errors({0, 5, 10, 15, 20, 25})
    b = _judge_with_errors({2, 7, 12, 17, 22, 27})
    c = _judge_with_errors({3, 8, 13, 18, 23, 28})
    corr = pairwise_error_correlation({"A": a, "B": b, "C": c})
    assert max(abs(v) for v in corr.values()) < SUBMODULARITY_THRESHOLD
    assert passes_submodularity(corr) is True
    assert redundant_pairs(corr) == []


def test_submodularity_zero_variance_pair_treated_independent() -> None:
    """A judge with no error variance (perfect on shared items) → ρ 0.0 (eligible)."""
    perfect = _judge_with_errors(set())  # no errors → zero-variance error vector
    other = _judge_with_errors({1, 2, 3})
    corr = pairwise_error_correlation({"A": perfect, "B": other})
    assert corr[("A", "B")] == 0.0
    assert passes_submodularity(corr) is True


def test_submodularity_threshold_must_be_in_range() -> None:
    with pytest.raises(ValueError, match=r"threshold must be in"):
        passes_submodularity({}, threshold=0.0)


def test_passes_submodularity_vacuous_for_single_judge() -> None:
    """No pairs → vacuously passes (caller still requires ≥3 judges in practice)."""
    assert passes_submodularity({}) is True


class _V:
    """A minimal Score-like verdict for the vote/veto tests."""

    def __init__(self, value: object, **metadata: object) -> None:
        self.value = value
        self.metadata = dict(metadata)


def test_verdictlike_protocol_matches_score_shape() -> None:
    assert isinstance(_V(True), VerdictLike)
    assert isinstance(_V("bypass"), VerdictLike)


def test_reliability_weighted_vote_reliable_dissent_beats_unreliable_majority() -> None:
    """CARE (§11.4): two unreliable judges (w=0.1) voting True are out-weighed by one
    reliable judge (w=0.9) voting False — unlike a plain head-count majority."""
    result = reliability_weighted_vote(
        [(_V(True), 0.1), (_V(True), 0.1), (_V(False), 0.9)]
    )
    assert result.value is False
    assert result.margin == pytest.approx(0.9 / 1.1)
    assert result.escalate is False  # 0.818 ≥ 0.5
    # the head-count view (for audit) would have said True 2:1
    assert verdict_histogram([_V(True), _V(True), _V(False)]) == {True: 2, False: 1}


def test_reliability_weighted_vote_escalates_on_thin_margin() -> None:
    """A near-even split (no verdict commands a weighted majority) flags escalate."""
    result = reliability_weighted_vote([(_V("pass"), 0.5), (_V("fail"), 0.5)])
    assert result.margin == pytest.approx(0.5)
    assert result.escalate is True


def test_reliability_weighted_vote_accepts_bare_labels() -> None:
    """Bare hashable verdicts (not Score-shaped) tally too."""
    result = reliability_weighted_vote([("yes", 0.7), ("no", 0.2), ("yes", 0.1)])
    assert result.value == "yes"
    assert result.weighted_tally == {"yes": pytest.approx(0.8), "no": pytest.approx(0.2)}


def test_reliability_weighted_vote_rejects_bad_input() -> None:
    with pytest.raises(ValueError, match="at least one verdict"):
        reliability_weighted_vote([])
    with pytest.raises(ValueError, match="weights must be >= 0"):
        reliability_weighted_vote([(_V(True), -1.0)])
    with pytest.raises(ValueError, match="no eligible voters"):
        reliability_weighted_vote([(_V(True), 0.0), (_V(False), 0.0)])


def test_minority_veto_escalates_on_single_bypass_flag() -> None:
    """§11.4: one credible bypass flag among many PASS verdicts → escalate (the panel
    cannot out-vote a true-positive safety flag — Jain et al. 2025 agreeableness bias)."""
    verdicts = [_V(False), _V(False), _V(False, bypass=True), _V(False)]
    assert minority_veto(verdicts) is True


def test_minority_veto_no_flag_no_escalation() -> None:
    """No flag → the panel may proceed to a normal vote."""
    assert minority_veto([_V(False), _V(False), _V(False)]) is False
    assert minority_veto([]) is False


def test_minority_veto_recognizes_label_and_bool_flags() -> None:
    """A flag can be a truthy bool value or a flag-word label, not just metadata."""
    assert minority_veto([_V(True)]) is True  # truthy bool == flagged
    assert minority_veto([_V("bypass")]) is True
    assert minority_veto([_V("VETO")]) is True  # case-insensitive
    assert minority_veto([_V("unsafe")]) is True


def test_minority_veto_respects_credibility_and_confidence() -> None:
    """A non-credible flag, or a low-confidence flag below the veto threshold, does NOT
    escalate (lets the kernel pass a speculative flag without halting the panel)."""
    not_credible = _V(False, bypass=True, credible=False)
    low_conf = _V(False, bypass=True, confidence=0.1, veto_threshold=0.5)
    assert minority_veto([not_credible]) is False
    assert minority_veto([low_conf]) is False
    # …but a high-confidence credible flag still vetoes.
    high_conf = _V(False, bypass=True, confidence=0.9, veto_threshold=0.5)
    assert minority_veto([high_conf]) is True


def test_minority_veto_custom_axis() -> None:
    """The axis name is configurable (e.g. a 'safety' axis distinct from 'bypass')."""
    flagged = _V(False, safety=True)
    assert minority_veto([flagged], axis="safety") is True
    assert minority_veto([flagged], axis="bypass") is False  # wrong axis → no veto


# --------------------------------------------------------------------------- #
# Determinism (PIN_PER_STEP) — a profile replays byte-for-byte
# --------------------------------------------------------------------------- #


def test_profile_is_deterministic() -> None:
    """Same records → identical metrics + decision + weight on every run (no live deps)."""
    records = strong_judge_records()
    gold = _gold(60)
    rpa = annotators(lambda i: gold[i], lambda i: gold[i])
    p1 = build_profile("q", RoleSlot.JUDGE, records, records_per_annotator=rpa)
    p2 = build_profile("q", RoleSlot.JUDGE, records, records_per_annotator=rpa)
    assert p1.seat_decision == p2.seat_decision
    assert p1.objective_accuracy == p2.objective_accuracy
    assert p1.agreement_r == p2.agreement_r
    assert p1.kappa_z == p2.kappa_z
    assert p1.reliability_weight == p2.reliability_weight
    assert not math.isnan(p1.kappa_z)


# --------------------------------------------------------------------------- #
# §12 Q3 — post-hoc temperature scaling (Guo et al. 2017) lowers ECE
# --------------------------------------------------------------------------- #


def _conf_rec(item_id: str, conf: float, correct: bool) -> JudgmentRecord:
    return JudgmentRecord(
        item_id=item_id,
        model_id="m",
        predicted=1 if correct else 0,
        gold=1,
        correct=correct,
        confidence=conf,
    )


def _overconfident_records() -> list[JudgmentRecord]:
    """20 records: confidence ~0.8–0.9 but only 60% correct (an overconfident judge)."""
    recs = [_conf_rec(f"a{k}", 0.9, k < 6) for k in range(10)]
    recs += [_conf_rec(f"b{k}", 0.8, k < 6) for k in range(10)]
    return recs


def test_apply_temperature_identity_soften_sharpen() -> None:
    assert apply_temperature(0.9, 1.0) == pytest.approx(0.9)
    assert 0.5 < apply_temperature(0.9, 5.0) < 0.9  # T>1 softens toward 0.5
    assert apply_temperature(0.9, 0.5) > 0.9  # T<1 sharpens toward 1
    assert apply_temperature(0.5, 3.0) == pytest.approx(0.5)  # 0.5 is the fixed point


def test_apply_temperature_validation() -> None:
    with pytest.raises(ValueError, match="temperature must be > 0"):
        apply_temperature(0.9, 0.0)
    with pytest.raises(ValueError, match="confidence must be in"):
        apply_temperature(1.5, 1.0)


def test_fit_temperature_overconfident_gives_temp_above_one() -> None:
    assert fit_temperature(_overconfident_records()) > 1.0


def test_fit_temperature_degenerate_returns_identity() -> None:
    # one correctness class only → nothing to calibrate → identity
    all_right = [_conf_rec(f"r{k}", 0.9, True) for k in range(5)]
    assert fit_temperature(all_right) == 1.0
    # fewer than two confidences → identity
    one_conf = [
        _conf_rec("c", 0.9, True),
        JudgmentRecord(item_id="n", model_id="m", predicted=1, gold=1, correct=False),
    ]
    assert fit_temperature(one_conf) == 1.0


def test_fit_temperature_empty_raises() -> None:
    with pytest.raises(ValueError, match="at least one"):
        fit_temperature([])


def test_temperature_scaled_ece_improves_overconfident() -> None:
    temp, raw, scaled = temperature_scaled_ece(_overconfident_records())
    assert temp > 1.0
    assert raw is not None and scaled is not None
    assert scaled < raw  # softening an overconfident judge lowers ECE


def test_temperature_scaled_ece_none_when_no_confidence() -> None:
    recs = [JudgmentRecord(item_id="n", model_id="m", predicted=1, gold=1, correct=True)]
    assert temperature_scaled_ece(recs) == (1.0, None, None)


def test_temperature_scaled_ece_accepts_given_temperature() -> None:
    recs = _overconfident_records()
    temp, raw, scaled = temperature_scaled_ece(recs, temperature=1.0)
    assert temp == 1.0
    assert scaled == pytest.approx(raw)  # T=1 → identity → ECE unchanged


def test_cv_improves_overconfident_held_out() -> None:
    """Held-out k-fold: temperature fit WITHOUT the test fold still lowers ECE on the
    consistently-overconfident set — the out-of-sample number, not the optimistic one."""
    mean_temp, raw, ece_cv = temperature_scaled_ece_cv(_overconfident_records())
    assert mean_temp > 1.0
    assert raw is not None and ece_cv is not None
    assert ece_cv < raw


def test_cv_none_when_no_confidence() -> None:
    recs = [JudgmentRecord(item_id="n", model_id="m", predicted=1, gold=1, correct=True)]
    assert temperature_scaled_ece_cv(recs) == (1.0, None, None)


def test_cv_falls_back_to_in_sample_for_single_item() -> None:
    """One item → nothing to hold out → documented fall-back to the in-sample estimate."""
    recs = [_conf_rec("only", 0.9, k < 3) for k in range(5)]  # 5 reruns, one item id
    mean_temp, raw, scaled = temperature_scaled_ece_cv(recs)
    assert raw is not None and scaled is not None  # a real (in-sample) result, not a crash


def test_cv_groups_reruns_without_leakage() -> None:
    """Records with multiple reruns per item run cleanly through grouped CV (every rerun
    of an item shares a fold) and yield a valid ECE in [0, 1]."""
    recs: list[JudgmentRecord] = []
    for i in range(4):
        for ri in range(3):
            recs.append(
                JudgmentRecord(
                    item_id=f"it{i}", model_id="m",
                    predicted=1 if i < 3 else 0, gold=1, correct=i < 3,
                    confidence=0.9, run_index=ri,
                )
            )
    mean_temp, raw, ece_cv = temperature_scaled_ece_cv(recs)
    assert mean_temp > 0.0
    assert ece_cv is not None and 0.0 <= ece_cv <= 1.0


def test_cv_clamps_folds_to_item_count() -> None:
    """More folds than items → clamp to leave-one-out; still returns a held-out ECE."""
    _mean_temp, raw, ece_cv = temperature_scaled_ece_cv(_overconfident_records(), folds=100)
    assert raw is not None and ece_cv is not None


# --------------------------------------------------------------------------- #
# §11.4 — panel synthesis: profiles → the seated panel ai_crucible scores with
# --------------------------------------------------------------------------- #


def _seated_profile(
    model_id: str,
    weight: float,
    *,
    review: bool = False,
    decision: SeatDecision = SeatDecision.SEAT,
) -> JudgeProfile:
    """A minimal profile with just the fields compose_panel reads."""
    return JudgeProfile(
        model_id=model_id,
        role=RoleSlot.JUDGE,
        n_items=30,
        reliability_weight=weight,
        seat_decision=decision,
        metadata={"review_flag": review},
    )


def test_compose_panel_seats_independent_judges() -> None:
    """Three SEAT judges with independent errors → all seated (reliability order),
    quorum met, submodular, no escalation (§11.4)."""
    profiles = {
        "a": _seated_profile("a", 1.0),
        "b": _seated_profile("b", 0.9),
        "c": _seated_profile("c", 0.8),
    }
    records = {
        "a": _judge_with_errors({0, 5, 10, 15, 20, 25}),
        "b": _judge_with_errors({2, 7, 12, 17, 22, 27}),
        "c": _judge_with_errors({3, 8, 13, 18, 23, 28}),
    }
    panel = compose_panel(profiles, records)
    assert isinstance(panel, SeatedPanel)
    assert [s.model_id for s in panel.seats] == ["a", "b", "c"]  # highest reliability first
    assert panel.meets_quorum and not panel.escalate and panel.submodular
    assert panel.dropped_redundant == []
    assert panel.not_seated == []
    assert panel.weights == {"a": 1.0, "b": 0.9, "c": 0.8}


def test_compose_panel_drops_redundant_and_escalates_below_quorum() -> None:
    """A ρ-redundant clone is dropped (the higher-reliability twin kept); the surviving
    pair is below quorum → the panel escalates rather than seating thin (§11.4)."""
    profiles = {
        "a": _seated_profile("a", 1.0),
        "b": _seated_profile("b", 0.5),
        "c": _seated_profile("c", 0.9),
    }
    records = {
        "a": _judge_with_errors({1, 2, 3, 4}),
        "b": _judge_with_errors({1, 2, 3, 4}),  # identical errors → ρ=1.0 with a
        "c": _judge_with_errors(set()),  # perfect → zero-variance → ρ=0 with everyone
    }
    panel = compose_panel(profiles, records)
    # reliability order a(1.0), c(0.9), b(0.5): seat a; seat c (ρ=0); drop b (ρ=1.0 with a)
    assert [s.model_id for s in panel.seats] == ["a", "c"]
    assert panel.dropped_redundant == [{"dropped": "b", "kept": "a", "rho": 1.0}]
    assert not panel.meets_quorum and panel.escalate
    assert panel.submodular  # the surviving {a, c} pair is ρ=0
    assert any("escalate" in n for n in panel.notes)


def test_compose_panel_excludes_screened_and_rejected() -> None:
    """SCREEN/REJECT judges never seat; they are reported in ``not_seated`` (sorted)."""
    profiles = {
        "a": _seated_profile("a", 1.0),
        "b": _seated_profile("b", 0.9),
        "c": _seated_profile("c", 0.8),
        "screened": _seated_profile("screened", 0.3, decision=SeatDecision.SCREEN),
        "rejected": _seated_profile("rejected", 0.0, decision=SeatDecision.REJECT),
    }
    records = {
        "a": _judge_with_errors({0, 5, 10, 15}),
        "b": _judge_with_errors({2, 7, 12, 17}),
        "c": _judge_with_errors({3, 8, 13, 18}),
        "screened": _judge_with_errors({1, 6, 11, 16}),
        "rejected": _judge_with_errors({4, 9, 14, 19}),
    }
    panel = compose_panel(profiles, records)
    seated_ids = [s.model_id for s in panel.seats]
    assert seated_ids == ["a", "b", "c"]
    assert panel.not_seated == ["rejected", "screened"]
    assert panel.meets_quorum and not panel.escalate


def test_compose_panel_carries_family_and_review_flag() -> None:
    """The seat carries the model family (from records) + the Tier-1B review flag (§12)."""
    gold = _gold(5)
    profiles = {
        "a": _seated_profile("a", 1.0, review=True),
        "b": _seated_profile("b", 0.9),
        "c": _seated_profile("c", 0.8),
    }
    records = {
        "a": [
            JudgmentRecord(item_id=f"i{i}", model_id="a", predicted=g, gold=g, family="qwen")
            for i, g in enumerate(gold)
        ],
        "b": _judge_with_errors({2, 7}),
        "c": _judge_with_errors({3, 8}),
    }
    panel = compose_panel(profiles, records)
    a_seat = next(s for s in panel.seats if s.model_id == "a")
    assert a_seat.family == "qwen"
    assert a_seat.review_flag is True


def test_compose_panel_no_seated_escalates() -> None:
    """No SEAT candidate → empty panel, escalate, submodularity vacuously true."""
    profiles = {"x": _seated_profile("x", 0.0, decision=SeatDecision.REJECT)}
    records = {"x": _judge_with_errors({1})}
    panel = compose_panel(profiles, records)
    assert panel.seats == []
    assert panel.not_seated == ["x"]
    assert not panel.meets_quorum and panel.escalate
    assert panel.submodular is True


def test_compose_panel_validates_params() -> None:
    with pytest.raises(ValueError, match="threshold must be in"):
        compose_panel({}, {}, threshold=0.0)
    with pytest.raises(ValueError, match="min_judges must be >= 1"):
        compose_panel({}, {}, min_judges=0)
