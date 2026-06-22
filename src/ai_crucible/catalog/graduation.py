"""The Lab→Arena→Regression lifecycle RULES (CONTRACT §A graduation + §B saturation).

This leaf is PURE policy: it owns the *decisions* — when a Lab puzzle graduates to
Arena, when an Arena puzzle saturates into Regression, and when a Regression puzzle
re-promotes — but knows nothing about how events are stored or folded. The store
(:mod:`ai_crucible.catalog.store`) injects these functions as :data:`PromoteFn` /
:data:`DemoteFn` and calls them during its fold; this module never imports the store
or the differential leaf (DECOMPOSE_BY_SECRETS — see CONTRACT "Module layout").

The two estimators here are DELIBERATELY DIFFERENT and must NOT be unified
(CONTRACT cross-cutting lock 2):

* **Graduation (§A)** is a ONE-SHOT decision at the pre-chosen N=20 floor → it uses
  FIXED-N Wilson (``scoring.stats.graduates`` + ``scoring.stats.wilson_interval``).
* **Saturation / re-promotion (§B)** PEEKS after every run → it uses an anytime-valid
  e-process (``scoring.stats.bernoulli_log_eprocess``), valid at every stopping time.

A maintainer who "simplifies" both to ``wilson_interval()`` silently breaks
saturation's anytime-validity; each function carries a loud comment to that effect.

Standards compliance (the six — workflow-standards.md)
------------------------------------------------------
- **PIN_PER_STEP — 3:** every decision is a PURE function of its pinned inputs
  (the folded :class:`PuzzleAggregate` / :class:`ModelRunSeries` + the pinned
  :class:`RuleConfig` whose ``rule_version`` content-hashes the knobs). No clock, no
  ``random``, no network — the same aggregate grades identically tomorrow, and the
  targeted tests pin exact verdicts and cross-check ``e_value`` against a hand-summed
  e-process. The thresholds (``sat_p0``/``sat_p1``/``e_threshold``/``dwell_k``/…) are
  read from the injected ``RuleConfig``, never hard-coded here.
- **ANDON_AUTHORITY — 2:** malformed counts HALT loud — the structured
  ``[CODE] message (hint: ...)`` error (:class:`GraduationRuleError`) fires before any
  decision is returned (a cohort with ``successes > n`` cannot silently produce a
  bogus PROMOTE). The underlying ``scoring.stats`` estimators also raise ``ValueError``
  on out-of-range counts, which this leaf surfaces, never swallows.
- **NAMED_COMPENSATORS — n/a (no skip needed — this leaf performs NO irreversible
  action):** these are pure decision functions. They WRITE nothing — no JSONL append,
  no network, no host-fs mutation. The compensator-bearing actions (event append,
  lock acquisition, release/publish) all live in the store + the coordinator, with the
  named-undo table in CONTRACT §D / the store leaf. There is nothing here to undo.
- **DECOMPOSE_BY_SECRETS — 3:** the lifecycle SECRET (the graduation/saturation
  thresholds + precedence) is isolated here; the store knows event mechanics, the
  differential knows the typology stats, and none import each other. The grading
  oracle never enters this leaf — it decides on OUTCOMES (success counts + a distilled
  panel signal), never on answers.
- **UNCERTAINTY_GATED_HUMANS — 3:** the ``DEFER_TO_DESIGNER`` branch IS the
  uncertainty-gated human checkpoint — it gates on the panel's low-confidence posture
  (sub-quorum / escalate / no cross-family solve evidence), NOT on a step count, and
  it abstains rather than fake a confident graduation (Trust-or-Escalate, Jung et al.
  ICLR 2025). The honesty caveat from CONTRACT §A is honored: the DEFER threshold is a
  HEURISTIC posture, not a provable human-agreement guarantee.
- **EXTERNAL_VERIFIER — 3:** graduation's Clause-3 consumes a CROSS-FAMILY panel
  signal (the generator's family is excluded upstream) and Clause-2 consumes the
  cross-family cohort solve rate — Claude never self-certifies its own graduation.
  The frontier gate keeps demotion anchored to capable external subjects.

Design lock: ``swarm/epic-4/CONTRACT.md`` §A (graduation) + §B (saturation).
"""

from __future__ import annotations

import math
from collections.abc import Iterable

from ai_crucible.catalog.types import (
    Decision,
    DemoteVerdict,
    ModelRunSeries,
    PromoteVerdict,
    PuzzleAggregate,
    RuleConfig,
)
from ai_crucible.scoring.stats import bernoulli_log_eprocess, graduates, wilson_interval

__all__ = [
    "GraduationRuleError",
    "DEFAULT_FRONTIER_IDS",
    "promote_decision",
    "demote_decision",
    "repromote_decision",
    "make_frontier_fn",
]

#: Above this accumulated log-e the exponentiation would overflow ``float64``
#: (``math.exp(709.78…)`` is the last finite value); past it we treat the e-value as
#: ``+inf`` rather than raising ``OverflowError``. The demotion threshold is far below
#: this, so an inf e-value is unambiguously "crossed" (CONTRACT §B).
_LOG_E_OVERFLOW_GUARD = 700.0


class GraduationRuleError(Exception):
    """A graduation/saturation rule was handed structurally invalid evidence.

    Raised with the house structured shape ``[CODE] message (hint: ...)`` (matching
    ``puzzle._fail`` / ``attestation._fail``) so a malformed aggregate HALTS loud
    (ANDON_AUTHORITY) instead of silently producing a bogus verdict.
    """


def _fail(code: str, message: str, hint: str) -> GraduationRuleError:
    """Build a structured :class:`GraduationRuleError` (code/message/hint)."""
    return GraduationRuleError(f"[{code}] {message} (hint: {hint})")


# --------------------------------------------------------------------------- #
# §A — Graduation (Lab→Arena), THREE-VALUED, abstention-aware. FIXED-N Wilson.
# --------------------------------------------------------------------------- #


def promote_decision(agg: PuzzleAggregate, rule: RuleConfig) -> PromoteVerdict:
    """The three-valued, abstention-aware Lab→Arena promotion rule (CONTRACT §A).

    Gates on three clauses that catch orthogonal failure modes, then resolves a
    PROMOTE / HOLD / DEFER_TO_DESIGNER verdict by precedence (a CONFIDENT negative
    HOLDs; clauses 1 & 2 passing but a LOW-CONFIDENCE cross-family signal DEFERs):

    1. **Claude band (Clause 1, unchanged):**
       ``claude_band_ok = claude_n > 0 and stats.graduates(claude_successes, claude_n)``
       — the fixed-N Wilson band rules out trivial AND impossible on Claude's own rate.
    2. **Cohort non-trivial (Clause 2, NEW):**
       ``cohort_nontrivial = cohort_n > 0 and wilson_upper(cohort) <= cohort_trivial_upper``
       — if the cross-family cohort trivially aces the puzzle it is easy for everyone,
       a mode solve-rate-on-Claude-alone is blind to (BBH live-prune, Suzgun 2022).
    3. **Judge-fairness (Clause 3, NEW keystone):**
       ``fairness_ok = panel.present and panel.meets_quorum and not panel.escalate and
       panel.fairness == "fair"`` — the cross-family fairness verdict (SWE-bench
       Verified screening; Trust-or-Escalate abstention, Jung et al. ICLR 2025).

    ``deferred = panel.deferred OR cohort_n == 0`` — a low-confidence panel OR the
    absence of cross-family solve evidence is itself low confidence.

    **Decision precedence (CONTRACT §A):**
      1. ``not claude_band_ok`` → HOLD (confident: Claude's own rate is trivial/impossible).
      2. ``cohort_n > 0 and not cohort_nontrivial`` → HOLD (confident: cohort aces it).
      3. confident UNFAIR panel (present + quorum + not escalate + fairness=="unfair")
         → HOLD (confident negative).
      4. ``deferred`` → DEFER_TO_DESIGNER (low-confidence cross-family signal).
      5. else → PROMOTE.

    ``born_typology`` is left ``None`` — the store stamps it via the injected
    ``classify_fn`` at graduation (one measurement, two uses; CONTRACT §A/§C). The
    ``evidence`` dict carries the Claude/cohort Wilson bounds + a panel snapshot for the
    TRANSITION record's audit trail.

    *(USE FIXED-N WILSON here — graduation is a ONE-SHOT decision at the pre-chosen
    N=20 floor. Do NOT swap in the saturation e-process / confidence sequence; that
    would silently break the locked graduation-vs-saturation distinction — CONTRACT
    lock 2.)*

    Raises:
        GraduationRuleError: if the aggregate carries structurally invalid counts
            (negative, or ``successes > n``) — a malformed aggregate HALTS loud
            (ANDON_AUTHORITY) rather than producing a bogus verdict.
    """
    _validate_agg(agg)

    # Clause 1 — Claude band (FIXED-N Wilson; CONTRACT lock 2 — do NOT use the e-process).
    claude_band_ok = agg.claude_n > 0 and graduates(agg.claude_successes, agg.claude_n)

    # Clause 2 — cohort non-trivial (FIXED-N Wilson upper bound).
    cohort_lower = cohort_upper = None
    cohort_nontrivial = False
    if agg.cohort_n > 0:
        cohort_lower, cohort_upper = wilson_interval(agg.cohort_successes, agg.cohort_n)
        cohort_nontrivial = cohort_upper <= rule.cohort_trivial_upper

    # Clause 3 — judge fairness (cross-family panel; EXTERNAL_VERIFIER).
    fairness_ok = (
        agg.panel.present
        and agg.panel.meets_quorum
        and not agg.panel.escalate
        and agg.panel.fairness == "fair"
    )

    # A CONFIDENT unfair verdict (quorum reached, not escalating) is a confident
    # negative — distinct from the low-confidence "deferred" posture.
    confident_unfair = (
        agg.panel.present
        and agg.panel.meets_quorum
        and not agg.panel.escalate
        and agg.panel.fairness == "unfair"
    )

    # Low-confidence cross-family signal: the panel deferred OR there is NO cross-family
    # solve evidence at all (an empty cohort cannot certify Clause 2).
    deferred = agg.panel.deferred or (agg.cohort_n == 0)

    # Claude bounds (always computable when claude_n > 0; for evidence/audit).
    if agg.claude_n > 0:
        claude_lower, claude_upper = wilson_interval(agg.claude_successes, agg.claude_n)
    else:
        claude_lower = claude_upper = None

    # ---- Decision precedence (CONTRACT §A) ----------------------------------
    # PROMOTE requires a CONFIDENT "fair" verdict (fairness_ok) AND cross-family solve
    # evidence — never a bare "not deferred". An abstained fairness verdict (panel present
    # + quorum + not-escalate but fairness is None) is itself low-confidence on fairness,
    # so it DEFERs: the CONTRACT §A "iff" is PROMOTE ⟺ claude_band_ok ∧ cohort_nontrivial
    # ∧ fairness_ok ∧ not deferred. Anything that is neither a CONFIDENT negative (HOLD)
    # nor a confident positive (PROMOTE) escalates to the Designer (DEFER) — the honest
    # default while ω is on ice and no fairness panel issues a confident verdict.
    if not claude_band_ok:
        decision = Decision.HOLD  # confident: Claude's own rate is trivial/impossible
    elif agg.cohort_n > 0 and not cohort_nontrivial:
        decision = Decision.HOLD  # confident: cohort trivially aces it
    elif confident_unfair:
        decision = Decision.HOLD  # confident: panel ruled the puzzle unfair/broken
    elif fairness_ok and agg.cohort_n > 0 and not deferred:
        # all clauses confidently PASS: fair panel + non-trivial cross-family cohort.
        decision = Decision.PROMOTE
    else:
        # clauses 1 & 2 pass but the cross-family signal is low-confidence (sub-quorum,
        # escalating, fairness-abstained, or no cohort solve evidence) → escalate.
        decision = Decision.DEFER_TO_DESIGNER

    return PromoteVerdict(
        decision=decision,
        claude_band_ok=claude_band_ok,
        cohort_nontrivial=cohort_nontrivial,
        fairness_ok=fairness_ok,
        deferred=deferred,
        born_typology=None,  # stamped by the store via classify_fn (CONTRACT §A/§C)
        evidence={
            "claude_wilson": [claude_lower, claude_upper],
            "cohort_wilson": [cohort_lower, cohort_upper],
            "claude_successes": agg.claude_successes,
            "claude_n": agg.claude_n,
            "cohort_successes": agg.cohort_successes,
            "cohort_n": agg.cohort_n,
            "cohort_trivial_upper": rule.cohort_trivial_upper,
            "confident_unfair": confident_unfair,
            "panel": agg.panel.to_payload(),
        },
    )


# --------------------------------------------------------------------------- #
# §B — Saturation (Arena→Regression). ANYTIME-VALID e-process, NOT fixed-N.
# --------------------------------------------------------------------------- #


def demote_decision(series: ModelRunSeries, rule: RuleConfig) -> DemoteVerdict:
    """The anytime-valid SATURATION rule, per (puzzle, model) (CONTRACT §B).

    DEMOTE Arena→Regression for ``(puzzle, model)`` iff ALL of:

    1. **High bar (anytime-valid):** the e-process for ``H0: p <= sat_p0`` vs
       ``H1: p = sat_p1`` crosses ``e_threshold`` (= ``1/sat_alpha`` = 20). We
       accumulate ``log_e = sum over series.runs of
       bernoulli_log_eprocess(s, n, p0=sat_p0, p1=sat_p1)`` and exponentiate.
    2. **Dwell:** ``len(series.runs) >= dwell_k`` AND
       ``series.total_n >= post_trigger_min_k_mult * series.min_k``. Dwell is EMERGENT
       from the e-process (a single lucky 20/20 cannot cross at dwell_k=2 separate runs).
    3. **Frontier gate:** ``series.is_frontier`` — a weak local model passing never
       demotes; everyone-fails stays in Arena (the maximally diagnostic frontier gap).

    The e-process unit is the RUN, not the attempt (CONTRACT lock 6 / Liu-2025's 71%
    within-batch identical-failure correlation): exactly ONE
    ``bernoulli_log_eprocess`` call per RunRecord, accumulated as a product of
    per-run e-values (sum of logs).

    *(anytime-valid e-process — do NOT swap in fixed-N ``wilson_lower`` (CONTRACT lock
    2). The catalog peeks after every run; acting on a fixed-N interval at a
    data-dependent stopping time inflates the false-demotion rate. The confidence
    SEQUENCE is valid at EVERY sample count — Howard/Ramdas/McAuliffe/Sekhon 2021.)*

    Returns:
        :class:`DemoteVerdict` with ``demote``, the accumulated ``e_value`` (``+inf``
        if ``log_e`` overflowed the guard), ``dwell_runs``, ``post_trigger_n``,
        ``frontier_ok``, a human ``reason``, and the ``evidence`` dict.

    Raises:
        GraduationRuleError: if any run carries structurally invalid counts.
    """
    e_value, dwell_runs, post_trigger_n = _saturation_e_value(
        series, p0=rule.sat_p0, p1=rule.sat_p1
    )

    e_crossed = e_value >= rule.e_threshold
    dwell_ok = dwell_runs >= rule.dwell_k
    post_trigger_ok = post_trigger_n >= rule.post_trigger_min_k_mult * series.min_k
    frontier_ok = series.is_frontier

    demote = e_crossed and dwell_ok and post_trigger_ok and frontier_ok

    reason = _demote_reason(
        demote,
        e_crossed=e_crossed,
        dwell_ok=dwell_ok,
        post_trigger_ok=post_trigger_ok,
        frontier_ok=frontier_ok,
    )

    return DemoteVerdict(
        demote=demote,
        e_value=e_value,
        dwell_runs=dwell_runs,
        post_trigger_n=post_trigger_n,
        frontier_ok=frontier_ok,
        reason=reason,
        evidence={
            "e_threshold": rule.e_threshold,
            "sat_p0": rule.sat_p0,
            "sat_p1": rule.sat_p1,
            "e_crossed": e_crossed,
            "dwell_ok": dwell_ok,
            "dwell_k": rule.dwell_k,
            "post_trigger_ok": post_trigger_ok,
            "post_trigger_floor": rule.post_trigger_min_k_mult * series.min_k,
            "min_k": series.min_k,
            "runs": [list(r) for r in series.runs],
            "model_id": series.model_id,
        },
    )


def repromote_decision(series: ModelRunSeries, rule: RuleConfig) -> DemoteVerdict:
    """The REGRESSION→ARENA re-promotion rule (CONTRACT §B re-promote).

    A genuine new-version capability regression must be able to re-promote a puzzle
    that had saturated: when the success rate FALLS back through the dead band, the
    puzzle becomes diagnostic again. We test ``H0: p >= repromote_p0`` by accumulating
    the e-process on the FAILURES of each run::

        log_e = sum over series.runs of
            bernoulli_log_eprocess(n - s, n, p0=1 - repromote_p0, p1=0.5)

    i.e. the alternative is "the success rate has dropped to ~0.5" (failure rate has
    RISEN to ~0.5), so the FAILURE-side e-process accumulates evidence that ``p`` is
    no longer at/above ``repromote_p0``. Re-promotion fires (``demote=True`` means
    "re-promote fires", reusing :class:`DemoteVerdict`) iff:

    * ``e_value >= e_threshold`` (the failure e-process crossed), AND
    * ``len(series.runs) >= dwell_k`` (dwell across separate runs), AND
    * ``series.is_frontier`` (symmetric with demotion's frontier gate).

    **The 0.75–0.90 dead band** (``repromote_p0`` = 0.75 vs ``sat_p0`` = 0.90) is the
    Schmitt-trigger guard band that KILLS demote/re-promote oscillation (CONTRACT §B):
    a puzzle demotes only above 0.90-ish and re-promotes only below 0.75-ish, so a
    rate hovering in between flips neither direction. Demote-not-delete: the timeline
    records every transition; nothing is erased.

    *(anytime-valid e-process — do NOT swap in fixed-N ``wilson_lower`` (CONTRACT lock
    2). Same confidence-sequence justification as :func:`demote_decision`, applied to
    the failure counts for the symmetric drop-detection.)*

    Returns:
        :class:`DemoteVerdict` with ``demote`` reused as "re-promote fires".

    Raises:
        GraduationRuleError: if any run carries structurally invalid counts.
    """
    # H0: p >= repromote_p0  ->  failures-H0 boundary is (1 - repromote_p0); H1: p=0.5
    # (failure rate has risen to ~0.5). We feed FAILURES (n - s) into the same
    # e-process primitive — the symmetric drop-detection (CONTRACT §B re-promote).
    fail_p0 = 1.0 - rule.repromote_p0
    e_value, dwell_runs, _post = _failure_e_value(series, p0=fail_p0, p1=0.5)

    e_crossed = e_value >= rule.e_threshold
    dwell_ok = dwell_runs >= rule.dwell_k
    frontier_ok = series.is_frontier

    repromote = e_crossed and dwell_ok and frontier_ok

    if repromote:
        reason = "REPROMOTE: rate fell through the 0.75-0.90 dead band (failure e-process crossed)"
    elif not frontier_ok:
        reason = "no re-promote: non-frontier subject (frontier gate)"
    elif not dwell_ok:
        reason = f"no re-promote: dwell not met ({dwell_runs} < {rule.dwell_k} runs)"
    elif not e_crossed:
        reason = "no re-promote: failure e-process below threshold (rate still in/above band)"
    else:
        reason = "no re-promote"

    post_trigger_n = series.total_n
    return DemoteVerdict(
        demote=repromote,
        e_value=e_value,
        dwell_runs=dwell_runs,
        post_trigger_n=post_trigger_n,
        frontier_ok=frontier_ok,
        reason=reason,
        evidence={
            "e_threshold": rule.e_threshold,
            "repromote_p0": rule.repromote_p0,
            "fail_p0": fail_p0,
            "dead_band": [rule.repromote_p0, rule.sat_p0],  # 0.75-0.90 guard band
            "e_crossed": e_crossed,
            "dwell_ok": dwell_ok,
            "dwell_k": rule.dwell_k,
            "runs": [list(r) for r in series.runs],
            "model_id": series.model_id,
        },
    )


# --------------------------------------------------------------------------- #
# Frontier gate helpers (POLICY — the integrator picks the set; CONTRACT §B)
# --------------------------------------------------------------------------- #

#: The known current-frontier subject substrings. POLICY, not a hard rule: the
#: integrator chooses the set it injects, and :func:`make_frontier_fn` accepts
#: ``extra`` to extend it. A model id is "frontier" iff one of these substrings
#: appears in it case-insensitively (so ``"qwen3-coder:480b"`` matches ``"qwen3-coder"``).
#: The frontier gate keeps demotion anchored to capable cross-family subjects — a
#: weak local model passing never demotes (CONTRACT §B frontier gate).
DEFAULT_FRONTIER_IDS: frozenset[str] = frozenset(
    {
        "claude",
        "qwen3-coder",
        "deepseek",
        "glm",
        "kimi",
        "gpt-oss",
        "minimax",
        "nemotron",
    }
)


def make_frontier_fn(extra: Iterable[str] = ()):
    """Build the injected :data:`~ai_crucible.catalog.types.FrontierFn` predicate.

    Returns a predicate ``(model_id: str) -> bool`` that matches ``model_id`` against
    the case-insensitive substring set ``DEFAULT_FRONTIER_IDS | set(extra)`` — so a
    versioned id like ``"qwen3-coder:480b"`` is recognised as the frontier subject
    ``"qwen3-coder"``. The integrator picks the set (this is POLICY input to the fold,
    CONTRACT §B); ``extra`` lets a deployment admit additional capable subjects without
    editing this leaf.

    Args:
        extra: additional frontier substrings to admit beyond the defaults.

    Returns:
        A pure predicate returning a built-in ``bool`` (never ``numpy.bool_``).
    """
    subjects = frozenset(s.lower() for s in (*DEFAULT_FRONTIER_IDS, *extra))

    def is_frontier(model_id: str) -> bool:
        mid = model_id.lower()
        return bool(any(sub in mid for sub in subjects))

    return is_frontier


# --------------------------------------------------------------------------- #
# Internal helpers — pure, deterministic
# --------------------------------------------------------------------------- #


def _validate_agg(agg: PuzzleAggregate) -> None:
    """HALT loud on structurally invalid aggregate counts (ANDON_AUTHORITY)."""
    for label, s, n in (
        ("claude", agg.claude_successes, agg.claude_n),
        ("cohort", agg.cohort_successes, agg.cohort_n),
    ):
        if n < 0:
            raise _fail(
                "GRAD_BAD_N",
                f"{label}_n must be non-negative, got {n}",
                "the store folds successes/n from RUN events; a negative n is a corrupt fold",
            )
        if s < 0 or s > n:
            raise _fail(
                "GRAD_BAD_SUCCESSES",
                f"{label}_successes must be in [0, {label}_n={n}], got {s}",
                "successes cannot exceed attempts; check the RUN aggregation in the store",
            )


def _exp_guarded(log_e: float) -> float:
    """Exponentiate an accumulated log-e, returning ``+inf`` past the overflow guard.

    A long saturated series pushes ``log_e`` past ``math.exp``'s finite range; we
    return ``+inf`` (unambiguously "crossed", since the threshold is far below) rather
    than raising ``OverflowError`` (CONTRACT §B "guard overflow").
    """
    if log_e >= _LOG_E_OVERFLOW_GUARD:
        return math.inf
    return math.exp(log_e)


def _saturation_e_value(
    series: ModelRunSeries, *, p0: float, p1: float
) -> tuple[float, int, int]:
    """Accumulate the SUCCESS-side e-process across runs (the saturation direction).

    Returns ``(e_value, dwell_runs, post_trigger_n)``. ONE
    ``bernoulli_log_eprocess`` call per RUN (the e-process unit is the run, not the
    attempt — CONTRACT lock 6).
    """
    log_e = 0.0
    for successes, n in series.runs:
        _validate_run(series, successes, n)
        # anytime-valid e-process — do NOT swap in fixed-N wilson_lower (CONTRACT lock 2).
        log_e += bernoulli_log_eprocess(successes, n, p0=p0, p1=p1)
    return (_exp_guarded(log_e), len(series.runs), series.total_n)


def _failure_e_value(
    series: ModelRunSeries, *, p0: float, p1: float
) -> tuple[float, int, int]:
    """Accumulate the FAILURE-side e-process across runs (the re-promote direction).

    Feeds ``(n - successes, n)`` into the e-process so it accumulates evidence the
    failure rate has RISEN (success rate has dropped). ONE call per RUN.
    """
    log_e = 0.0
    for successes, n in series.runs:
        _validate_run(series, successes, n)
        failures = n - successes
        # anytime-valid e-process — do NOT swap in fixed-N wilson_lower (CONTRACT lock 2).
        log_e += bernoulli_log_eprocess(failures, n, p0=p0, p1=p1)
    return (_exp_guarded(log_e), len(series.runs), series.total_n)


def _validate_run(series: ModelRunSeries, successes: int, n: int) -> None:
    """HALT loud on a structurally invalid run within a series (ANDON_AUTHORITY)."""
    where = f"model={series.model_id}, puzzle={series.puzzle_id}"
    if n <= 0:
        raise _fail(
            "SAT_BAD_N",
            f"run n must be positive ({where}), got {n}",
            "each RUN must record >= 1 attempt; an empty run is a corrupt series fold",
        )
    if successes < 0 or successes > n:
        raise _fail(
            "SAT_BAD_SUCCESSES",
            f"run successes must be in [0, n={n}] ({where}), got {successes}",
            "successes cannot exceed attempts; check the per-run aggregation in the store",
        )


def _demote_reason(
    demote: bool,
    *,
    e_crossed: bool,
    dwell_ok: bool,
    post_trigger_ok: bool,
    frontier_ok: bool,
) -> str:
    """A human-legible reason for the demotion verdict (recorded in the TRANSITION)."""
    if demote:
        return "SATURATED: e-process crossed, dwell + post-trigger-n + frontier all satisfied"
    if not frontier_ok:
        return "no demote: non-frontier subject (frontier gate — everyone-fails stays in Arena)"
    if not e_crossed:
        return "no demote: saturation e-process below threshold (rate not yet near-perfect)"
    if not dwell_ok:
        return "no demote: dwell not met (one lucky run cannot demote)"
    if not post_trigger_ok:
        return "no demote: post-trigger attempts below floor (need >= mult * min_k)"
    return "no demote"
