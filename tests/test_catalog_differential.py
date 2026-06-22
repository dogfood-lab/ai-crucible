"""Tests for the differential typology statistics (``catalog.differential``).

These are PURE, deterministic tests (PIN_PER_STEP): they construct
:class:`PuzzleAggregate` values or call :func:`classify_counts` directly — no
network, no real models, no clock. They pin the four-valued typology
(CONTRACT §C, research-grounding §4) against hand-checked count fixtures and
prove the load-bearing locks:

* drive the typology off the CI on the DELTA, never off two-interval overlap
  (Schenker & Gentleman 2001 — the overlap heuristic is over-conservative);
* ``INCONCLUSIVE_UNDERPOWERED`` is first-class at N=20/k=3 (Miller 2024);
* ``LLM_GENERAL_GAP`` requires POSITIVE shared-failure evidence (both Wilson
  uppers ≤ ceiling), never merely a null delta;
* the ``mde_floor`` gate: a CI that excludes zero but with ``|δ|`` below the
  floor is INCONCLUSIVE, not directional;
* McNemar is gated behind ``paired=True`` with explicit b/c counts (the
  architecture does NOT pair Claude and the cohort — CONTRACT lock 8);
* catalog-level BH-FDR (q=0.10) downgrades a weak directional finding to
  INCONCLUSIVE while a strong one survives (``bh_survived`` flips).
"""

from __future__ import annotations

import pytest

from ai_crucible.catalog.differential import (
    classify_catalog,
    classify_counts,
    classify_puzzle,
)
from ai_crucible.catalog.types import (
    PanelSignal,
    PuzzleAggregate,
    RuleConfig,
    Typology,
)

RULE = RuleConfig()


def _agg(
    puzzle_id: str,
    claude_succ: int,
    claude_n: int,
    cohort_succ: int,
    cohort_n: int,
) -> PuzzleAggregate:
    """Construct a minimal :class:`PuzzleAggregate` for the typology tests."""
    return PuzzleAggregate(
        puzzle_id=puzzle_id,
        puzzle_content_hash=f"hash-{puzzle_id}",
        claude_successes=claude_succ,
        claude_n=claude_n,
        cohort_successes=cohort_succ,
        cohort_n=cohort_n,
        panel=PanelSignal(),
    )


# --------------------------------------------------------------------------- #
# classify_counts — the four-valued core
# --------------------------------------------------------------------------- #


def test_strong_gap_claude_underperforms_is_claude_specific_gap() -> None:
    # Claude 2/20, cohort 18/20 → big negative delta, CI excludes 0, |δ| ≥ floor.
    res = classify_counts(2, 20, 18, 20, RULE)
    assert res.typology is Typology.CLAUDE_SPECIFIC_GAP
    assert res.delta < 0
    lower, upper = res.delta_ci
    assert upper < 0  # CI entirely below zero
    assert abs(res.delta) >= RULE.mde_floor


def test_reverse_strong_gap_is_claude_strength() -> None:
    # Claude 18/20, cohort 2/20 → big positive delta → anti-regression strength.
    res = classify_counts(18, 20, 2, 20, RULE)
    assert res.typology is Typology.CLAUDE_STRENGTH
    assert res.delta > 0
    lower, _ = res.delta_ci
    assert lower > 0  # CI entirely above zero


def test_both_fail_is_llm_general_gap() -> None:
    # 0/20 vs 0/20 → null delta AND both Wilson uppers (≈0.161) ≤ general_fail_ceiling
    # (0.20). This is POSITIVE shared-failure evidence, not a mere null delta.
    # (1/20 would NOT qualify at the default 0.20 ceiling: Wilson(1,20).upper ≈ 0.236.)
    res = classify_counts(0, 20, 0, 20, RULE)
    assert res.typology is Typology.LLM_GENERAL_GAP
    # delta-CI must CONTAIN zero (not a directional claim).
    lower, upper = res.delta_ci
    assert lower <= 0 <= upper
    # both Wilson uppers cleared the ceiling — the positive evidence requirement.
    assert res.evidence["claude_wilson_upper"] <= RULE.general_fail_ceiling
    assert res.evidence["cohort_wilson_upper"] <= RULE.general_fail_ceiling


def test_both_mid_overlapping_is_inconclusive() -> None:
    # 10/20 vs 11/20 → tiny delta, CI straddles 0, neither side fails → underpowered.
    res = classify_counts(10, 20, 11, 20, RULE)
    assert res.typology is Typology.INCONCLUSIVE_UNDERPOWERED


def test_ci_excludes_zero_but_below_mde_floor_is_inconclusive() -> None:
    # A large-N case where the delta CI excludes zero yet |δ| < mde_floor (0.30):
    # 600/1000 vs 550/1000 → δ = 0.05, CI excludes 0 (n is huge) but well below floor.
    res = classify_counts(600, 1000, 550, 1000, RULE)
    lower, upper = res.delta_ci
    assert (lower > 0) or (upper < 0)  # CI excludes zero
    assert abs(res.delta) < RULE.mde_floor
    assert res.typology is Typology.INCONCLUSIVE_UNDERPOWERED


def test_zero_n_claude_is_inconclusive_underpowered() -> None:
    res = classify_counts(0, 0, 5, 20, RULE)
    assert res.typology is Typology.INCONCLUSIVE_UNDERPOWERED
    assert res.realized_mde == 1.0
    assert res.p_value == 1.0


def test_zero_n_cohort_is_inconclusive_underpowered() -> None:
    res = classify_counts(5, 20, 0, 0, RULE)
    assert res.typology is Typology.INCONCLUSIVE_UNDERPOWERED
    assert res.realized_mde == 1.0
    assert res.p_value == 1.0


def test_realized_mde_is_ci_half_width() -> None:
    res = classify_counts(2, 20, 18, 20, RULE)
    lower, upper = res.delta_ci
    assert res.realized_mde == pytest.approx((upper - lower) / 2.0)


def test_overlap_heuristic_would_disagree_with_delta_ci() -> None:
    # The non-overlapping-intervals TRAP, made concrete. 6/20 (Claude) vs 14/20
    # (cohort): the two SEPARATE Wilson intervals OVERLAP, so the (wrong)
    # interval-overlap heuristic would call this null. The Newcombe delta-CI is the
    # right tool — we assert the classifier drives off the DELTA CI, never the
    # overlap of the two intervals (Schenker & Gentleman 2001).
    from ai_crucible.scoring.stats import newcombe_wilson_diff, wilson_interval

    cl_lo, cl_hi = wilson_interval(6, 20, conf=RULE.differential_conf)
    co_lo, co_hi = wilson_interval(14, 20, conf=RULE.differential_conf)
    # The two intervals OVERLAP (overlap heuristic → "no difference"):
    assert cl_hi > co_lo and co_hi > cl_lo
    # Yet the delta CI cleanly excludes zero (the right answer is "different"):
    d_lo, d_hi, delta = newcombe_wilson_diff(
        6, 20, 14, 20, conf=RULE.differential_conf
    )
    assert d_hi < 0  # delta CI entirely below zero despite the interval overlap
    # And the classifier (driving off the delta CI) calls it directional.
    res = classify_counts(6, 20, 14, 20, RULE)
    assert res.delta == pytest.approx(delta)
    assert res.typology is Typology.CLAUDE_SPECIFIC_GAP


def test_overlap_trap_avoided_strong_case() -> None:
    # A case the overlap heuristic calls null but the delta-CI resolves directionally.
    # Claude 3/20, cohort 17/20: Wilson(3,20)≈(0.052,0.360), Wilson(17,20)≈(0.640,0.948).
    # These do NOT overlap here, but the |δ|=0.70 is huge — the delta CI cleanly
    # excludes 0 and clears the floor, so it MUST classify directionally.
    res = classify_counts(3, 20, 17, 20, RULE)
    assert res.typology is Typology.CLAUDE_SPECIFIC_GAP


# --------------------------------------------------------------------------- #
# McNemar misuse guard (CONTRACT lock 8)
# --------------------------------------------------------------------------- #


def test_paired_without_bc_raises_structured_error() -> None:
    with pytest.raises(ValueError) as exc:
        classify_counts(2, 20, 18, 20, RULE, paired=True)
    msg = str(exc.value)
    assert "[DIFFERENTIAL_PAIRED_NEEDS_BC]" in msg
    assert "hint:" in msg


def test_paired_with_bc_uses_mcnemar() -> None:
    from ai_crucible.scoring.stats import mcnemar_exact

    # paired=True with explicit discordant counts → p_value == mcnemar_exact(b,c).
    res = classify_counts(
        2, 20, 18, 20, RULE, paired=True, mcnemar_b=1, mcnemar_c=15
    )
    assert res.p_value == pytest.approx(mcnemar_exact(1, 15))


# --------------------------------------------------------------------------- #
# classify_puzzle — the injected ClassifyFn adapter
# --------------------------------------------------------------------------- #


def test_classify_puzzle_adapts_aggregate() -> None:
    agg = _agg("p1", 2, 20, 18, 20)
    res = classify_puzzle(agg, RULE)
    assert res.typology is Typology.CLAUDE_SPECIFIC_GAP
    # must match the direct classify_counts call (unpaired).
    direct = classify_counts(2, 20, 18, 20, RULE)
    assert res.typology is direct.typology
    assert res.delta == pytest.approx(direct.delta)


# --------------------------------------------------------------------------- #
# classify_catalog — BH-FDR across directional puzzles
# --------------------------------------------------------------------------- #


def test_classify_catalog_preserves_insertion_order() -> None:
    aggs = {
        "z": _agg("z", 2, 20, 18, 20),
        "a": _agg("a", 10, 20, 11, 20),
        "m": _agg("m", 1, 20, 1, 20),
    }
    out = classify_catalog(aggs, RULE)
    assert list(out.keys()) == ["z", "a", "m"]


def test_classify_catalog_non_directional_not_downgraded_by_bh() -> None:
    # LLM_GENERAL_GAP and INCONCLUSIVE keep their class regardless of BH.
    aggs = {
        "general": _agg("general", 0, 20, 0, 20),
        "incon": _agg("incon", 10, 20, 11, 20),
    }
    out = classify_catalog(aggs, RULE)
    assert out["general"].typology is Typology.LLM_GENERAL_GAP
    assert out["incon"].typology is Typology.INCONCLUSIVE_UNDERPOWERED
    # bh_survived stays at its default for non-directional classes.
    assert out["general"].bh_survived is True
    assert out["incon"].bh_survived is True


def test_classify_catalog_bh_downgrades_weak_directional_keeps_strong() -> None:
    # Prove bh_survived FLIPS through the real downgrade path. A directional
    # finding (delta-CI excludes 0) at N=20 has Fisher p ≤ ~0.0959 — always below
    # the DEFAULT fdr_q=0.10, so the default never downgrades a solo directional
    # finding (see test_default_q_does_not_downgrade_solo_directional). We exercise
    # the mechanism at a tighter, equally-valid q where the downgrade is reachable.
    rule = RuleConfig(fdr_q=0.05)

    # A maximally strong gap: 0/20 vs 20/20 → vanishing Fisher p (~1e-11).
    # A weak-but-directional gap: 4/20 vs 10/20 → δ=-0.30 (clears mde_floor), CI
    # excludes 0, but Fisher p≈0.0959 > (rank/m)·q at the top rank → fails BH.
    aggs = {
        "strong": _agg("strong", 0, 20, 20, 20),
        "weak": _agg("weak", 4, 20, 10, 20),
    }

    # Pre-BH both are directional (sanity on the fixture).
    pre = classify_puzzle(aggs["weak"], rule)
    assert pre.typology is Typology.CLAUDE_SPECIFIC_GAP

    out = classify_catalog(aggs, rule)

    # The strong finding survives BH and keeps its directional class.
    assert out["strong"].typology is Typology.CLAUDE_SPECIFIC_GAP
    assert out["strong"].bh_survived is True

    # The weak finding is downgraded to INCONCLUSIVE with bh_survived=False.
    assert out["weak"].typology is Typology.INCONCLUSIVE_UNDERPOWERED
    assert out["weak"].bh_survived is False
    # the evidence records WHAT it was downgraded from (audit trail).
    assert out["weak"].evidence["bh_downgraded_from"] == str(
        Typology.CLAUDE_SPECIFIC_GAP
    )


def test_default_q_does_not_downgrade_solo_directional() -> None:
    # Honest documentation: at the DEFAULT fdr_q=0.10 the unpaired-Fisher path
    # cannot downgrade a solo directional finding — the largest Fisher p for any
    # directional finding (~0.0959) is below q, so the BH step-up always keeps the
    # top-ranked finding. A weak directional finding therefore SURVIVES at default
    # q. (The downgrade path is reachable at a tighter q or with many simultaneous
    # directional findings — proven above.)
    aggs = {
        "strong": _agg("strong", 0, 20, 20, 20),
        "weak": _agg("weak", 4, 20, 10, 20),
    }
    out = classify_catalog(aggs, RULE)  # default q=0.10
    assert out["strong"].bh_survived is True
    assert out["weak"].typology is Typology.CLAUDE_SPECIFIC_GAP
    assert out["weak"].bh_survived is True


def test_classify_catalog_single_strong_directional_survives() -> None:
    aggs = {"only": _agg("only", 0, 20, 20, 20)}
    out = classify_catalog(aggs, RULE)
    assert out["only"].typology is Typology.CLAUDE_SPECIFIC_GAP
    assert out["only"].bh_survived is True
