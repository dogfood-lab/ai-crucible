"""Differential diagnostic typology — the catalog's per-puzzle statistics (CONTRACT §C).

The differential answers the instrument's highest-value question: *for this
puzzle, does Claude's solve rate differ from the cross-family cohort's, and how?*
It produces the four-valued :class:`~ai_crucible.catalog.types.Typology`
(``CLAUDE_SPECIFIC_GAP`` / ``CLAUDE_STRENGTH`` / ``LLM_GENERAL_GAP`` /
``INCONCLUSIVE_UNDERPOWERED``), stamped both at graduation (the Arena entry's
*born typology*) and at catalog-summary time (CONTRACT §A/§C: one measurement,
two uses).

The single load-bearing statistical decision (CONTRACT §C, research-grounding §4):

    **Drive the typology from the confidence interval on the DELTA**
    ``δ = claude_rate − cohort_rate`` — NOT by comparing two separate intervals.

Claude and the cross-family cohort run INDEPENDENT attempt sets, so this is an
UNPAIRED two-proportion comparison. The primary estimator is the Newcombe-1998
hybrid-score CI (:func:`ai_crucible.scoring.stats.newcombe_wilson_diff`), built by
combining the two single-proportion Wilson intervals — it "performs well
irrespective of sample size" (Newcombe 1998, Statistics in Medicine 17(8):873-890).

.. warning::

    **Do NOT classify by checking whether two separate Wilson intervals overlap.**
    The interval-overlap heuristic is statistically WRONG here: it is
    over-conservative and loses power — non-overlapping implies a difference, but
    OVERLAPPING does NOT imply no difference, so the heuristic fails to reject when
    it should (Schenker & Gentleman 2001, The American Statistician 55(3):182-186,
    doi:10.1198/000313001317097960). Always test the DIFFERENCE — drive off the
    delta CI. Every function below obeys this; ``classify_counts`` never inspects
    the two single-proportion intervals to decide a *direction*.

    **McNemar is the WRONG tool for the default path.** :func:`mcnemar_exact` is a
    PAIRED test; the architecture does not produce trial-level pairing between
    Claude and the CohortSolver (they run disjoint attempt sets). McNemar is gated
    strictly behind ``paired=True`` requiring explicit ``b``/``c`` discordant counts
    from a single shared attempt log (CONTRACT cross-cutting lock 8). The default —
    and the only path the catalog wires — is UNPAIRED Fisher.

At N=20/k=3 the minimum detectable effect is large (MDE ≈ 1/√N, Miller 2024,
arXiv:2411.00640), so most puzzles legitimately land in
``INCONCLUSIVE_UNDERPOWERED``. That is a FIRST-CLASS outcome, not a bug — the
typology never forces a noisy 3-way call it cannot support.

Standards compliance (the six — workflow-standards.md)
------------------------------------------------------
- **PIN_PER_STEP — 3:** every function is a PURE deterministic function of its
  integer counts + the pinned :class:`RuleConfig` (whose ``rule_version`` content-
  hashes the thresholds). No clock, no RNG, no network, no I/O — the same counts
  classify identically tomorrow. The thresholds the classifier reads
  (``mde_floor`` / ``general_fail_ceiling`` / ``differential_conf`` / ``fdr_q``)
  are policy knobs carried on the rule, so a fold replays byte-for-byte.
- **ANDON_AUTHORITY — 2:** a misuse HALTS loud — ``paired=True`` without explicit
  discordant counts raises the structured ``[DIFFERENTIAL_PAIRED_NEEDS_BC]`` error
  rather than silently running the wrong (unpaired) test; out-of-range counts
  propagate :class:`ValueError` from the stats primitives. A wrong typology never
  silently propagates downstream.
- **NAMED_COMPENSATORS — n/a (skip: no irreversible action):** every function is
  pure and side-effect-free — no durable write, no network call, no external
  state to undo. There is nothing to compensate (the store leaf, not the
  differential, owns the durable-write compensators table).
- **DECOMPOSE_BY_SECRETS — 3:** this leaf knows ONLY the typology statistics. It
  imports nothing from ``catalog.store`` or ``catalog.graduation`` (they never
  import each other — CONTRACT build law 2); it consumes the contract types + the
  stable ``scoring.stats`` primitives. The grading secret/oracle never enters here
  — it classifies aggregated OUTCOME counts, not answers.
- **UNCERTAINTY_GATED_HUMANS — 3:** ``INCONCLUSIVE_UNDERPOWERED`` is the
  uncertainty-aware abstention — when the delta CI cannot resolve a direction at
  this N (or BH-FDR rejects the finding), the classifier abstains instead of
  forcing a confident label. The catalog surfaces these for human attention rather
  than manufacturing a noisy directional claim.
- **EXTERNAL_VERIFIER — 3:** the cohort rate is the CROSS-FAMILY solve anchor —
  the delta is measured against non-Claude families, so Claude never self-certifies
  its own typology. The cohort runs at the SAME N floor as Claude (CONTRACT
  cross-cutting lock 1) so the Newcombe CI is not silently degraded.
"""

from __future__ import annotations

from ai_crucible.catalog.types import (
    ClassifyResult,
    PuzzleAggregate,
    RuleConfig,
    Typology,
)
from ai_crucible.scoring.stats import (
    benjamini_hochberg,
    fisher_difference_pvalue,
    mcnemar_exact,
    newcombe_wilson_diff,
    wilson_interval,
)

__all__ = [
    "classify_counts",
    "classify_puzzle",
    "classify_catalog",
]


class DifferentialError(ValueError):
    """Raised on a misuse of the differential typology API (CONTRACT lock 8).

    Subclasses :class:`ValueError` so a caller catching the broad numeric-domain
    error (the shape every stats primitive raises) also catches this, while the
    ``[CODE] message (hint: ...)`` house shape keeps the message legible.
    """


def _fail(code: str, message: str, hint: str) -> DifferentialError:
    """Build a structured :class:`DifferentialError` (code/message/hint)."""
    return DifferentialError(f"[{code}] {message} (hint: {hint})")


def classify_counts(
    claude_succ: int,
    claude_n: int,
    cohort_succ: int,
    cohort_n: int,
    rule: RuleConfig,
    *,
    paired: bool = False,
    mcnemar_b: int | None = None,
    mcnemar_c: int | None = None,
) -> ClassifyResult:
    """Classify one puzzle's Claude-vs-cohort outcome into a four-valued typology.

    The core of CONTRACT §C. Drives the typology off the confidence interval on
    ``δ = claude_rate − cohort_rate`` (Newcombe-1998 hybrid-score CI), NEVER off
    the overlap of the two single-proportion intervals (see the module-level
    warning — interval-overlap is over-conservative, Schenker & Gentleman 2001).

    Decision procedure:

    1. **Underpowered guard.** If either arm has ``n == 0`` there is no evidence —
       return ``INCONCLUSIVE_UNDERPOWERED`` with ``delta=0``, ``delta_ci=(-1, 1)``,
       ``realized_mde=1.0``, ``p_value=1.0``.
    2. **Delta CI.** ``(lower, upper, delta) =
       newcombe_wilson_diff(claude_succ, claude_n, cohort_succ, cohort_n,
       conf=rule.differential_conf)``.
    3. **Significance handle.** UNPAIRED (default): two-sided Fisher exact p-value
       (:func:`fisher_difference_pvalue`) — the small-N-admissible test the
       catalog-level BH-FDR pass consumes. PAIRED (``paired=True``, opt-in only):
       :func:`mcnemar_exact` on the explicit discordant counts ``mcnemar_b`` /
       ``mcnemar_c`` (both REQUIRED, else a structured
       ``[DIFFERENTIAL_PAIRED_NEEDS_BC]`` error — McNemar needs trial-level pairing
       the architecture does not produce; CONTRACT lock 8).
    4. **Resolvability.** ``realized_mde = (upper − lower) / 2`` (the CI half-width
       — the smallest effect resolvable at this N). ``excludes_zero = lower > 0 or
       upper < 0``. A finding is ``directional`` iff it excludes zero AND
       ``|δ| ≥ rule.mde_floor`` (a CI that excludes zero but with a delta below the
       floor is a real-but-trivial difference → still INCONCLUSIVE).
    5. **Classify.**

       * ``directional and upper < 0`` → ``CLAUDE_SPECIFIC_GAP`` (Claude
         underperforms the cohort; ``δ = claude − cohort < 0`` — the highest-value
         finding).
       * ``directional and lower > 0`` → ``CLAUDE_STRENGTH`` (anti-regression).
       * ``not excludes_zero`` AND both Wilson uppers ≤ ``rule.general_fail_ceiling``
         → ``LLM_GENERAL_GAP``. This requires POSITIVE shared-failure evidence
         (both arms demonstrably fail), never merely a null delta (CONTRACT §C).
       * otherwise → ``INCONCLUSIVE_UNDERPOWERED``.

    ``bh_survived`` defaults ``True`` here; :func:`classify_catalog` flips it to
    ``False`` (and downgrades the class) for a directional finding that does not
    survive the catalog-level BH-FDR step-up.

    Args:
        claude_succ, claude_n: Claude's successes / attempts (``0 <= s <= n``).
        cohort_succ, cohort_n: the cross-family cohort's successes / attempts.
        rule: the pinned :class:`RuleConfig` (thresholds + ``differential_conf``).
        paired: opt-in to the PAIRED McNemar path (default ``False`` — the catalog
            never sets this; the architecture is unpaired).
        mcnemar_b, mcnemar_c: REQUIRED when ``paired=True`` — the discordant counts
            from a single shared attempt log (``b`` = Claude-right/cohort-wrong,
            ``c`` = Claude-wrong/cohort-right).

    Returns:
        A :class:`ClassifyResult` with the typology, the signed ``delta`` and its
        CI, the ``realized_mde``, the ``p_value``, and an ``evidence`` dict of the
        per-arm rates + Wilson uppers + the resolvability flags.

    Raises:
        DifferentialError: ``[DIFFERENTIAL_PAIRED_NEEDS_BC]`` if ``paired=True`` but
            either discordant count is ``None``.
        ValueError: if any non-empty count pair is out of range (propagated from
            the stats primitives).
    """
    # --- 1. underpowered guard: no evidence on at least one arm -------------- #
    if claude_n == 0 or cohort_n == 0:
        return ClassifyResult(
            typology=Typology.INCONCLUSIVE_UNDERPOWERED,
            delta=0.0,
            delta_ci=(-1.0, 1.0),
            realized_mde=1.0,
            p_value=1.0,
            bh_survived=True,
            evidence={
                "reason": "empty_arm",
                "claude_n": claude_n,
                "cohort_n": cohort_n,
            },
        )

    # --- 2. delta CI (Newcombe hybrid-score) -------------------------------- #
    lower, upper, delta = newcombe_wilson_diff(
        claude_succ, claude_n, cohort_succ, cohort_n, conf=rule.differential_conf
    )

    # --- 3. significance handle --------------------------------------------- #
    if paired:
        # McNemar is gated behind explicit discordant counts: the catalog never
        # sets paired=True (Claude and the cohort run disjoint attempt sets), so
        # there is no trial-level pairing to derive b/c from. Demand them
        # explicitly rather than fabricate a pairing (CONTRACT lock 8).
        if mcnemar_b is None or mcnemar_c is None:
            raise _fail(
                "DIFFERENTIAL_PAIRED_NEEDS_BC",
                "paired=True requires explicit McNemar discordant counts "
                f"mcnemar_b/mcnemar_c (got b={mcnemar_b}, c={mcnemar_c})",
                "the architecture does not pair Claude and the cohort; pass "
                "paired=False for the default UNPAIRED Fisher test, or supply b/c "
                "from a single SHARED attempt log",
            )
        p_value = mcnemar_exact(mcnemar_b, mcnemar_c)
    else:
        # UNPAIRED default: Fisher exact on the 2x2 table — the right small-N test
        # for two independent proportions (NOT McNemar).
        p_value = fisher_difference_pvalue(
            claude_succ, claude_n, cohort_succ, cohort_n
        )

    # --- 4. resolvability ---------------------------------------------------- #
    realized_mde = (upper - lower) / 2.0
    excludes_zero = (lower > 0.0) or (upper < 0.0)
    directional = excludes_zero and abs(delta) >= rule.mde_floor

    # Per-arm Wilson uppers — used ONLY for the positive shared-failure evidence
    # of LLM_GENERAL_GAP, never to decide a DIRECTION via interval overlap.
    claude_rate = claude_succ / claude_n
    cohort_rate = cohort_succ / cohort_n
    _, claude_wilson_upper = wilson_interval(
        claude_succ, claude_n, conf=rule.differential_conf
    )
    _, cohort_wilson_upper = wilson_interval(
        cohort_succ, cohort_n, conf=rule.differential_conf
    )

    # --- 5. classify --------------------------------------------------------- #
    if directional and upper < 0.0:
        typology = Typology.CLAUDE_SPECIFIC_GAP
    elif directional and lower > 0.0:
        typology = Typology.CLAUDE_STRENGTH
    elif (
        not excludes_zero
        and claude_wilson_upper <= rule.general_fail_ceiling
        and cohort_wilson_upper <= rule.general_fail_ceiling
    ):
        # POSITIVE shared-failure evidence: both arms demonstrably fail (both
        # Wilson uppers below the ceiling) AND the delta CI contains zero. Never a
        # mere null delta (CONTRACT §C).
        typology = Typology.LLM_GENERAL_GAP
    else:
        typology = Typology.INCONCLUSIVE_UNDERPOWERED

    return ClassifyResult(
        typology=typology,
        delta=delta,
        delta_ci=(lower, upper),
        realized_mde=realized_mde,
        p_value=p_value,
        bh_survived=True,
        evidence={
            "claude_rate": claude_rate,
            "cohort_rate": cohort_rate,
            "claude_wilson_upper": claude_wilson_upper,
            "cohort_wilson_upper": cohort_wilson_upper,
            "excludes_zero": excludes_zero,
            "directional": directional,
            "paired": paired,
        },
    )


def classify_puzzle(agg: PuzzleAggregate, rule: RuleConfig) -> ClassifyResult:
    """The :data:`~ai_crucible.catalog.types.ClassifyFn` the store injects (CONTRACT §C).

    Adapts a folded :class:`PuzzleAggregate` into the UNPAIRED
    :func:`classify_counts` call — the only path the catalog wires (the
    architecture never pairs Claude and the cohort, so ``paired`` stays ``False``;
    CONTRACT cross-cutting lock 8). The store calls this during the fold to stamp a
    puzzle's typology; its own tests inject a fake ``ClassifyFn`` instead.

    Args:
        agg: the per-puzzle aggregate (``claude_successes`` / ``claude_n`` vs
            ``cohort_successes`` / ``cohort_n``, both at the same N floor).
        rule: the pinned :class:`RuleConfig`.

    Returns:
        The single-puzzle :class:`ClassifyResult` (before catalog-level BH-FDR;
        :func:`classify_catalog` applies that across the batch).
    """
    return classify_counts(
        agg.claude_successes,
        agg.claude_n,
        agg.cohort_successes,
        agg.cohort_n,
        rule,
        paired=False,
    )


def classify_catalog(
    aggs: dict[str, PuzzleAggregate], rule: RuleConfig
) -> dict[str, ClassifyResult]:
    """Classify a whole catalog with catalog-level BH-FDR control (CONTRACT §C).

    Across many puzzles the differential runs many comparisons, so a per-puzzle
    "significant" call needs multiple-comparison control or the catalog fills with
    false directional labels. This:

    1. Classifies each puzzle independently via :func:`classify_puzzle`.
    2. Applies Benjamini-Hochberg FDR control
       (:func:`ai_crucible.scoring.stats.benjamini_hochberg` at ``rule.fdr_q``)
       across ONLY the DIRECTIONAL puzzles (``CLAUDE_SPECIFIC_GAP`` /
       ``CLAUDE_STRENGTH``), using their delta ``p_value``\\s. A directional puzzle
       whose p-value does NOT survive the BH step-up is DOWNGRADED to
       ``INCONCLUSIVE_UNDERPOWERED`` with ``bh_survived=False`` (its solo delta-CI
       excluded zero, but the finding does not hold up under the multiple-
       comparison correction).
    3. Leaves ``LLM_GENERAL_GAP`` and ``INCONCLUSIVE_UNDERPOWERED`` puzzles
       untouched by BH — their class is NOT a directional delta claim, so the FDR
       correction on directional deltas does not apply to them (their
       ``bh_survived`` keeps its ``True`` default).

    The dict's INSERTION ORDER is preserved (the catalog timeline order).

    Args:
        aggs: ``{puzzle_id: PuzzleAggregate}`` — the folded per-puzzle aggregates.
        rule: the pinned :class:`RuleConfig` (supplies ``fdr_q`` + the thresholds).

    Returns:
        ``{puzzle_id: ClassifyResult}`` in the same insertion order, with
        directional non-survivors downgraded and ``bh_survived`` set per puzzle.
    """
    # Pass 1: per-puzzle classification (insertion order preserved by dict).
    results: dict[str, ClassifyResult] = {
        pid: classify_puzzle(agg, rule) for pid, agg in aggs.items()
    }

    # Pass 2: collect the DIRECTIONAL findings (the only ones BH applies to).
    directional_keys = [
        pid
        for pid, res in results.items()
        if res.typology in (Typology.CLAUDE_SPECIFIC_GAP, Typology.CLAUDE_STRENGTH)
    ]
    if not directional_keys:
        return results

    pvalues = [results[pid].p_value for pid in directional_keys]
    survived = benjamini_hochberg(pvalues, q=rule.fdr_q)

    # Pass 3: downgrade directional non-survivors; stamp bh_survived on survivors.
    for pid, ok in zip(directional_keys, survived, strict=True):
        res = results[pid]
        if ok:
            res.bh_survived = True
            res.evidence["bh_survived"] = True
        else:
            res.bh_survived = False
            res.evidence["bh_survived"] = False
            res.evidence["bh_downgraded_from"] = str(res.typology)
            res.typology = Typology.INCONCLUSIVE_UNDERPOWERED

    return results
