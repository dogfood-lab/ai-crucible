"""The seat-or-screen admission test — turn metrics into a :class:`SeatDecision`.

:func:`build_profile` runs the §11.1 metrics (:mod:`crucible.characterize.metrics`) over
a model's :class:`~crucible.characterize.types.JudgmentRecord`s and applies the **seat
gates** to produce a :class:`~crucible.characterize.types.JudgeProfile` with a
SEAT / SCREEN / REJECT decision, human-legible ``notes`` explaining *why*, and a derived
``reliability_weight`` for the panel aggregator (:mod:`crucible.characterize.aggregate`).

**§12 calibration redesign (2026-06-01 — the authoritative gate; supersedes §11.1).**
The first real characterization run (6 local models × 20 items × k=3) self-diagnosed two
defects, both corrected here:

1. **The κ gate was INVERTED (THE fix).** The old gate screened a judge whose κ sat
   *above* the human baseline (``|z| ≥ 1``). Han et al. 2025 ("Judge's Verdict",
   arXiv:2510.09738) keeps such **super-consistent** judges as **Tier-1B: valid,
   top-ranked, seated** — its own four highest-κ models are z>1. crucible's two-sided
   screen was a misreading of its own source. The gate is now a **ONE-SIDED FLOOR**: a
   judge passes the agreement gate when ``κ ≥ human_baseline − margin`` (the margin floors
   at the lower CI of the human baseline). A ``z > 1`` ("super-consistent") judge
   **passes and is seated**; its only consequence is a ``review_flag`` (stored in
   ``notes`` + ``metadata["review_flag"]``) — a flag for *later* human review IF a near-
   perfect κ co-occurs with high human disagreement on the same items (Richie, Grover &
   Tsui 2022, ACL 2022.bionlp-1.26: a well-specified model can legitimately exceed IAA),
   **never a downgrade**. Net: a κ=1.0 judge now SEATS (with a review flag), not screens.

2. **The brittle 7-threshold binary AND is replaced** by a **difficulty-normalized
   continuous quality score** (:func:`~crucible.characterize.metrics.quality_score` over
   :func:`~crucible.characterize.metrics.difficulty_weighted_accuracy`) and a **selective,
   CI-based** decision (Traub 2024, "selective classification", arXiv:2407.01032 — score
   across the operating curve, not at one cut; Sarmah 2024, arXiv:2412.12148 — derive the
   cutoff from data, not by fiat): **seat** if the score's CI lower bound clears the
   quality floor, **screen** if the CI straddles the floor (uncertainty, not failure),
   **reject** if the CI is entirely below. A ship-time :func:`perturbation_audit` jitters
   every threshold ±1 SE and reports the decision-flip rate (the §8.3 Alzahrani
   "When Benchmarks are Targets" lens applied to the admission gate; Eiras 2025,
   arXiv:2503.04474 — style shifts a judge's FNR up to 0.24, so threshold stability must
   be measured).

The gate ladder (§12-corrected, in decreasing severity):

* **REJECT** — the judge is not usable for this role at all. Hard gates (any one fails):
    - objective (difficulty-weighted) accuracy has **no real margin over chance** — the
      Wilson 95% lower bound is ≤ 0.5 (the §1 small-N-admissible test reused from
      ``crucible.scoring.stats``), **or**
    - hard agreement failure: Pearson ``r < 0.80`` (the §11.1 #2 first gate — a judge
      whose ratings don't even correlate with gold cannot be reliability-weighted), **or**
    - the **one-sided κ floor**: ``κ < human_baseline − margin`` (agreement structure
      below the human level), **or**
    - all hard gates pass but the **quality-score CI is entirely below the floor**.
* **SCREEN** — usable only as a *cheap pre-filter whose verdicts are always escalated,
  never final* (Jung, Brahman & Choi 2024, "Trust or Escalate", arXiv:2407.18370). A
  model lands here when the hard gates pass but the **quality-score CI straddles the
  floor** (genuine uncertainty), or alt-test ``ω < 0.5`` (substitution not acceptable —
  Calderon et al. 2025, arXiv:2501.10970).
* **SEAT** — full panel member (reliability-weighted): hard gates pass **and** the
  quality-score CI lower bound clears the floor **and** ``ω ≥ 0.5`` (when measured). A
  ``review_flag`` does **not** change the decision — a super-consistent judge seats.

``reliability_weight`` (consumed by CARE-style aggregation, §11.4) is derived only for
SEAT/SCREEN models (REJECT → ``0.0``). It is the continuous quality score itself,
floored slightly above 0 for a seated/screened judge and halved for SCREEN (their
verdicts escalate, so they barely move the aggregate). This unifies "how good is the
judge" (the score) with "how much does it count" (the weight) — §12's continuous frame.

**Standards compliance (the six)** — the binding section for this domain:

- **PIN_PER_STEP — 3.** Every metric is a pure deterministic reduction over the records;
  given a pinned ``(model, quant, prompt, fixture-hash)`` the resulting profile is
  byte-for-byte reproducible. Thresholds are frozen ``SeatGates`` constants stamped into
  ``metadata["thresholds"]`` (provenance), and the gate *decision* is factored into a
  pure :func:`_decide_from_metrics` so :func:`perturbation_audit` replays it exactly with
  jittered thresholds.
- **ANDON_AUTHORITY — 2.** A single failed hard gate halts promotion (REJECT stops the
  model joining the panel); :func:`perturbation_audit` is itself an andon instrument — a
  high decision-flip rate signals the gate is measuring threshold noise, not the judge,
  and should block the admission decision until the set/thresholds are firmed.
- **NAMED_COMPENSATORS — n/a (skip).** ``build_profile``/``perturbation_audit`` perform
  **no irreversible tool call** — they are pure functions returning a dataclass / dict
  (no publish/push/release/DB write). Per the workflow-standards skip rule, a
  compensators table is not required for a pure in-memory computation; the calling
  harness owns any persistence + its undo.
- **DECOMPOSE_BY_SECRETS — 3.** Gold labels live *in the records the grader supplies*,
  not in the profiled model; the profile module never sees a model runtime. Metrics,
  gate logic, and aggregation are three files split by what changes together (§10.6).
- **UNCERTAINTY_GATED_HUMANS — 3.** The §12 selective decision *is* the uncertainty gate:
  a straddling CI → SCREEN (escalate, Jung 2024) rather than a forced pass/fail; the
  ``review_flag`` routes the genuinely ambiguous super-consistent case to a human without
  penalizing it; ``notes`` frame each outcome contrastively.
- **EXTERNAL_VERIFIER — 3.** The whole point of the panel these profiles compose is that
  no model verifies its own family's output; ``family_pref_delta`` *measures* the bias
  this guards against, and ``aggregate.passes_submodularity`` refuses two judges whose
  *errors* correlate. The §12 reversal was itself caught by the citation-verification
  pass — EXTERNAL_VERIFIER applied to the research.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from crucible.characterize import metrics as M
from crucible.characterize.types import (
    JudgeProfile,
    JudgmentRecord,
    RoleSlot,
    SeatDecision,
)
from crucible.scoring.stats import wilson_interval

__all__ = ["SeatGates", "build_profile", "perturbation_audit"]


@dataclass(frozen=True, slots=True)
class SeatGates:
    """The §12-corrected seat-gate thresholds — frozen so a profile run is replayable.

    Defaults encode the research-grounding numbers; a caller may tighten/loosen them per
    role and the chosen values are stamped into ``JudgeProfile.metadata["thresholds"]``
    for provenance. The §12 redesign repurposes one field and adds three:

    * ``kappa_z_review`` (was the two-sided screen ``kappa_z_abs``) is now the
      **review-flag** trigger only — ``z > kappa_z_review`` flags a super-consistent
      judge for *later* human review; it never screens or rejects.
    * ``kappa_floor_margin`` is the one-sided agreement floor margin; ``None`` derives it
      from the human baseline's own 95% sampling SE (the "lower CI of the baseline").
    * ``quality_floor`` is the continuous-score floor the selective CI decision clears.
    * ``w_accuracy``/``w_agreement``/``w_consistency`` are the score blend weights.
    """

    #: Wilson 95% lower bound on (difficulty-weighted) accuracy must exceed this — a real
    #: margin over chance, not a point estimate (§11.1 #1, §1 small-N admissibility).
    accuracy_floor: float = 0.50
    #: Target point accuracy worth noting (JudgeBench ≥0.60 — §11.1 #1). Not a hard gate;
    #: the *margin* (Wilson lower > accuracy_floor) is the gate.
    accuracy_target: float = 0.60
    #: Pearson r hard gate (§11.1 #2 first gate).
    agreement_r: float = 0.80
    #: §12 ONE-SIDED κ floor margin: a judge passes when ``κ ≥ human_baseline − margin``.
    #: ``None`` ⇒ derive the margin from the baseline's 95% sampling SE (lower CI of the
    #: human baseline). A fixed value (e.g. 0.10) overrides that derivation.
    kappa_floor_margin: float | None = None
    #: §12 review-flag trigger (NOT a screen): ``z > kappa_z_review`` ⇒ super-consistent;
    #: flag for later human review only. Replaces the old two-sided ``|z|<1`` screen.
    kappa_z_review: float = 1.0
    #: §12 continuous quality-score floor — the selective CI decision seats above it,
    #: screens across it, rejects below it.
    quality_floor: float = 0.55
    #: alt-test ω seat threshold (§11.1 #3).
    alt_test_omega: float = 0.50
    #: test-retest consistency floor (§11.1 #4): 1 − flip-rate must be ≥ this.
    consistency_floor: float = 0.80
    #: each bias metric (position / verbosity / |family-pref|) must be below this.
    bias_ceiling: float = 0.25
    #: ECE above this is "overconfident" — soft signal, down-weights, never rejects.
    ece_soft_ceiling: float = 0.15
    #: quality-score blend weights (should sum to 1.0; lead with difficulty-weighted acc).
    w_accuracy: float = 0.50
    w_agreement: float = 0.30
    w_consistency: float = 0.20


@dataclass(frozen=True, slots=True)
class _Metrics:
    """The measured scalars for one judge — the pure input to the gate decision.

    Factoring the metrics out of the gate logic lets :func:`perturbation_audit` re-run the
    *decision* under jittered thresholds without re-measuring anything (the metrics are a
    function of the records, the thresholds are not).
    """

    accuracy: float          # difficulty-weighted accuracy (§12)
    acc_lower: float         # Wilson 95% lower bound on accuracy
    r: float                 # Pearson r vs gold/human
    kappa: float
    kappa_z: float           # signed z vs the human baseline
    omega: float | None      # alt-test ω (None = not measured)
    consistency: float
    cons_measured: bool
    ece: float | None        # None = not measured (§12)
    max_bias: float
    n_items: int


def _kappa_floor(human_human_kappa: float, n_items: int, gates: SeatGates) -> float:
    """The §12 one-sided agreement floor: ``human_baseline − margin``.

    The floor sits at the **lower CI of the human baseline** (§12): a judge whose κ is no
    worse than a human annotator's lower-confidence agreement passes. The margin is
    ``gates.kappa_floor_margin`` when set, else the baseline's own 95% sampling SE
    ``1.96·sqrt(p0(1−p0)/n_items)`` (the same large-sample SE form
    :func:`~crucible.characterize.metrics.kappa_zscore` uses — a transparent, defensible
    SE rather than a black box). A degenerate baseline (0 or 1, no sampling variance)
    yields a zero-width margin, so the floor is the baseline itself.
    """
    if gates.kappa_floor_margin is not None:
        margin = gates.kappa_floor_margin
    else:
        p0 = human_human_kappa
        var = p0 * (1.0 - p0) / max(1, n_items)
        margin = 1.96 * (var**0.5) if var > 0.0 else 0.0
    return human_human_kappa - margin


def _score_with_accuracy(acc: float, m: _Metrics, gates: SeatGates) -> float:
    """The continuous quality score at a *given* accuracy (the others held at ``m``).

    Used both for the point score (``acc = m.accuracy``) and to push the accuracy CI
    bounds through the monotone combiner for the selective decision.
    """
    return M.quality_score(
        accuracy=acc,
        agreement_r=m.r,
        consistency=m.consistency,
        ece=m.ece,
        max_bias=m.max_bias,
        ece_soft_ceiling=gates.ece_soft_ceiling,
        w_accuracy=gates.w_accuracy,
        w_agreement=gates.w_agreement,
        w_consistency=gates.w_consistency,
    )


def _score_ci(m: _Metrics, gates: SeatGates) -> tuple[float, float, float]:
    """``(score, score_lo, score_hi)`` — the §12 selective-decision interval.

    The dominant source of sampling uncertainty in the score is the accuracy term, so we
    build a Wilson 95% interval on the (difficulty-weighted) accuracy over ``n_items`` and
    push its bounds through the **monotone** :func:`~crucible.characterize.metrics.quality_score`
    (the other components are deterministic given the records). Because the combiner is
    non-decreasing in accuracy, the accuracy CI maps to a score CI ordered the same way.
    This is the small-N-admissible interval (Wilson, not Wald — §1) the rest of crucible
    uses, reused rather than re-derived.
    """
    successes = round(m.accuracy * m.n_items)
    acc_lo, acc_hi = wilson_interval(successes, max(1, m.n_items), conf=0.95)
    score = _score_with_accuracy(m.accuracy, m, gates)
    score_lo = _score_with_accuracy(acc_lo, m, gates)
    score_hi = _score_with_accuracy(acc_hi, m, gates)
    return score, score_lo, score_hi


def _decide_from_metrics(
    m: _Metrics, human_human_kappa: float, gates: SeatGates
) -> tuple[SeatDecision, bool, float]:
    """The pure §12 gate decision — ``(decision, review_flag, quality_score)``.

    No metrics are recomputed here; this is a deterministic function of the measured
    scalars + the thresholds, which is exactly what :func:`perturbation_audit` needs to
    replay under jitter. The decision implements the §12 ladder:

    1. **Hard gates** (any fail ⇒ REJECT): accuracy margin (Wilson lower > floor); the
       Pearson ``r`` floor; the **one-sided κ floor** (``κ ≥ baseline − margin``).
    2. **Selective CI decision** among the hard-gate survivors: SEAT if ``score_lo ≥
       quality_floor``; SCREEN if the CI straddles the floor; REJECT if ``score_hi <
       quality_floor``. An alt-test ``ω < 0.5`` caps a would-be SEAT at SCREEN
       (substitution not acceptable — Calderon 2025).
    3. **review_flag** (independent of the decision): ``z > kappa_z_review`` ⇒
       super-consistent ⇒ flag for later human review (Han 2025 Tier-1B), never a
       downgrade.
    """
    accuracy_ok = m.acc_lower > gates.accuracy_floor
    r_ok = m.r >= gates.agreement_r
    kappa_floor = _kappa_floor(human_human_kappa, m.n_items, gates)
    kappa_floor_ok = m.kappa >= kappa_floor

    review_flag = m.kappa_z > gates.kappa_z_review

    if not (accuracy_ok and r_ok and kappa_floor_ok):
        score = _score_with_accuracy(m.accuracy, m, gates)
        return SeatDecision.REJECT, review_flag, score

    score, score_lo, score_hi = _score_ci(m, gates)
    omega_ok = m.omega is None or m.omega >= gates.alt_test_omega

    if score_hi < gates.quality_floor:
        decision = SeatDecision.REJECT
    elif score_lo >= gates.quality_floor and omega_ok:
        decision = SeatDecision.SEAT
    else:
        # CI straddles the floor (uncertainty, not failure) OR ω<0.5 caps a seat → SCREEN.
        decision = SeatDecision.SCREEN
    return decision, review_flag, score


def _reliability_weight(quality: float, decision: SeatDecision) -> float:
    """Derive the CARE-style reliability weight (§11.4) from the §12 quality score.

    §12 unifies "how good is the judge" with "how much it counts": the weight *is* the
    continuous quality score (already monotone in agreement + consistency + calibration
    and decreasing in bias — :func:`~crucible.characterize.metrics.quality_score`). REJECT
    → 0.0 (never votes). SCREEN is halved — its verdicts are meant to escalate (Jung
    2024), so it should barely move the aggregate. A seated/screened judge is floored
    slightly above 0 so a barely-passing judge still casts a (tiny) vote (CARE, Zhao et
    al. 2026, arXiv:2603.00039 — reliability-weighting beats majority vote).
    """
    if decision is SeatDecision.REJECT:
        return 0.0
    weight = quality
    if decision is SeatDecision.SCREEN:
        weight *= 0.5
    return float(max(1e-3, min(1.0, weight)))


def _measure(
    records: list[JudgmentRecord],
    *,
    human_human_kappa: float,
    records_per_annotator: dict[str, list[JudgmentRecord]] | None,
    human_grounded: bool = False,
    alt_test_epsilon: float | None = None,
    alt_test_exclude: set[str] | None = None,
) -> _Metrics:
    """Run the §11.1 metrics over ``records`` → the pure :class:`_Metrics` scalars (§12).

    When ``human_grounded`` is True the alt-test ω is the AUDIT-READY procedure
    (:func:`metrics.alt_test` — per-tier ε + paired t-test + BY-FDR against HUMAN
    annotators, Fork C §12.1); otherwise it is the circular model-jury PROXY
    (:func:`metrics.alt_test_omega`). The seat threshold (ω ≥ 0.5) is unchanged — only
    the estimator differs.
    """
    n_items = len({r.item_id for r in records})

    accuracy = M.difficulty_weighted_accuracy(records)
    # Wilson lower on the difficulty-weighted accuracy (successes recovered as acc × n).
    successes = round(accuracy * n_items)
    acc_lower, _ = wilson_interval(successes, max(1, n_items), conf=0.95)

    r, kappa = M.agreement(records)
    kappa_z = M.kappa_zscore(kappa, human_human_kappa, n_items)

    omega: float | None = None
    if records_per_annotator is not None:
        if human_grounded:
            omega = M.alt_test(
                records_per_annotator,
                epsilon=alt_test_epsilon if alt_test_epsilon is not None else 0.1,
                exclude_items=alt_test_exclude,
            )
        else:
            omega = M.alt_test_omega(records_per_annotator)

    cons = M.consistency(records)
    item_counts: dict[str, int] = {}
    for x in records:
        item_counts[x.item_id] = item_counts.get(x.item_id, 0) + 1
    cons_measured = any(c >= 2 for c in item_counts.values())

    ece = M.expected_calibration_error(records)  # §12: None when no confidences

    pos_b = M.position_bias(records)
    verb_b = M.verbosity_bias(records)
    fam_d = M.family_pref_delta(records)
    max_bias = max(pos_b, verb_b, abs(fam_d))

    return _Metrics(
        accuracy=accuracy,
        acc_lower=acc_lower,
        r=r,
        kappa=kappa,
        kappa_z=kappa_z,
        omega=omega,
        consistency=cons,
        cons_measured=cons_measured,
        ece=ece,
        max_bias=max_bias,
        n_items=n_items,
    )


def build_profile(
    model_id: str,
    role: RoleSlot,
    records: list[JudgmentRecord],
    *,
    human_human_kappa: float = 0.80,
    records_per_annotator: dict[str, list[JudgmentRecord]] | None = None,
    human_grounded: bool = False,
    alt_test_epsilon: float | None = None,
    alt_test_exclude: set[str] | None = None,
    quant: str | None = None,
    gates: SeatGates | None = None,
) -> JudgeProfile:
    """Run the §11.1 admission metrics under the §12 gate and return a :class:`JudgeProfile`.

    Computes the metrics over ``records``, applies the **§12-corrected** gate (one-sided κ
    floor + difficulty-normalized continuous score + selective CI decision), and derives
    the seat decision + ``review_flag`` + reliability weight. ``notes`` records every gate
    outcome in contrastive form (UNCERTAINTY_GATED_HUMANS) so a human reading the profile
    sees exactly which gate moved the decision — and, for a super-consistent judge, sees
    that it **seated with a review flag** rather than being screened.

    Args:
        model_id: the model being profiled (e.g. ``"qwen3.6:32b"``).
        role: the :class:`~crucible.characterize.types.RoleSlot` being tested for
            (Designer stays Claude and is never profiled here).
        records: the model's judgments on the calibration set (non-empty). Should include
            repeated ``run_index`` passes (consistency), position-swap trials (position
            bias), family/length tags (bias panel), and ``metadata["difficulty"]`` (the
            §12 difficulty weighting) where those are to be exercised; gates whose data is
            absent are recorded as "not measured" and do **not** fail the model.
        human_human_kappa: the human–human baseline κ for the one-sided floor + the review
            z-score (§12). Defaults to 0.80; §12 recommends estimating μ_human/σ_human
            from real data rather than hardcoding it.
        records_per_annotator: optional ``{annotator_id: [...]}`` including a ``"judge"``
            entry, used to compute the alt-test ω (§11.1 #3). When ``None``, ω is not
            measured and does not block a seat (with a note).
        quant: the quantization level of the profiled model (recorded; §11.2 measures
            reliability across quant rather than assuming a floor).
        gates: override thresholds (defaults to :class:`SeatGates`).

    Returns:
        A fully-populated :class:`~crucible.characterize.types.JudgeProfile`. The §12
        additions live in ``metadata["quality_score"]``, ``metadata["review_flag"]`` (and
        ``metadata["review_reason"]`` when flagged), and ``metadata["score_ci"]``.

    Raises:
        ValueError: if ``records`` is empty (propagated from the metrics).
    """
    if not records:
        raise ValueError("build_profile requires at least one JudgmentRecord")
    g = gates or SeatGates()
    n = len(records)
    m = _measure(
        records,
        human_human_kappa=human_human_kappa,
        records_per_annotator=records_per_annotator,
        human_grounded=human_grounded,
        alt_test_epsilon=alt_test_epsilon,
        alt_test_exclude=alt_test_exclude,
    )
    notes: list[str] = []

    # --- §11.1 #1 objective accuracy, difficulty-normalized (§12) ---
    accuracy_ok = m.acc_lower > g.accuracy_floor
    dw = " (difficulty-weighted)" if any(
        r.metadata.get("difficulty") is not None for r in records
    ) else ""
    if accuracy_ok:
        tgt = " (≥ target)" if m.accuracy >= g.accuracy_target else " (below 0.60 target)"
        notes.append(
            f"accuracy{dw} {m.accuracy:.3f}{tgt}; Wilson-95 lower {m.acc_lower:.3f} > "
            f"{g.accuracy_floor:.2f} → real margin over chance ✓"
        )
    else:
        notes.append(
            f"accuracy{dw} {m.accuracy:.3f}; Wilson-95 lower {m.acc_lower:.3f} ≤ "
            f"{g.accuracy_floor:.2f} → NO margin over chance ✗ (REJECT)"
        )

    # --- §11.1 #2 / §12 agreement: Pearson r gate + ONE-SIDED κ floor ---
    r_ok = m.r >= g.agreement_r
    kappa_floor = _kappa_floor(human_human_kappa, m.n_items, g)
    kappa_floor_ok = m.kappa >= kappa_floor
    super_consistent = m.kappa_z > g.kappa_z_review
    kz_label = (
        "SUPER-consistent → review-flag (Tier-1B, SEATED)"
        if super_consistent
        else "within band"
    )
    notes.append(
        f"agreement r={m.r:.3f} ({'≥' if r_ok else '<'} {g.agreement_r:.2f}); "
        f"κ={m.kappa:.3f} ({'≥' if kappa_floor_ok else '<'} one-sided floor "
        f"{kappa_floor:.3f}); z vs human {m.kappa_z:+.2f} ({kz_label})"
    )

    # --- §12 review flag (NOT a downgrade) ---
    if super_consistent:
        notes.append(
            f"review_flag SET — κ={m.kappa:.3f} sits z={m.kappa_z:+.2f} above the human "
            f"baseline (Han 2025 Tier-1B: super-consistent judges are TOP-ranked and "
            f"SEATED). Flagged for later human review only IF κ≈1.0 co-occurs with high "
            f"human disagreement on the same items — NOT screened."
        )

    # --- §11.1 #3 alt-test substitution ω (optional data) ---
    if m.omega is not None:
        omega_ok = m.omega >= g.alt_test_omega
        if human_grounded:
            eps = alt_test_epsilon if alt_test_epsilon is not None else 0.1
            src = f"HUMAN-grounded (ε={eps:.2f}, paired-t + BY-FDR; Calderon 2025)"
        else:
            src = "model-jury PROXY (CIRCULAR; not a substitution guarantee)"
        notes.append(
            f"alt-test ω={m.omega:.3f} [{src}] ({'≥' if omega_ok else '<'} "
            f"{g.alt_test_omega:.2f} → {'seat-eligible' if omega_ok else 'SCREEN-only'})"
        )
    else:
        notes.append("alt-test ω not measured (no per-annotator records supplied)")

    # --- §11.1 #4 consistency (test-retest) ---
    if m.cons_measured:
        cons_ok = m.consistency >= g.consistency_floor
        notes.append(
            f"consistency {m.consistency:.3f} ({'≥' if cons_ok else '<'} "
            f"{g.consistency_floor:.2f} test-retest floor)"
        )
    else:
        notes.append("consistency not measured (no repeated run_index passes)")

    # --- §11.1 #5 calibration (ECE) — soft signal, may be None (§12) ---
    if m.ece is not None:
        overconf = m.ece > g.ece_soft_ceiling
        notes.append(
            f"ECE {m.ece:.3f} ({'over' if overconf else 'within'} "
            f"{g.ece_soft_ceiling:.2f} soft ceiling → "
            f"{'down-weight' if overconf else 'ok'})"
        )
    else:
        notes.append("ECE not measured (no confidences) — no calibration penalty (§12)")

    # --- §11.1 #6 bias panel (position / verbosity / family-pref) ---
    pos_b = M.position_bias(records)
    verb_b = M.verbosity_bias(records)
    fam_d = M.family_pref_delta(records)
    bias_ok = m.max_bias < g.bias_ceiling
    notes.append(
        f"bias: position {pos_b:.3f}, verbosity {verb_b:.3f}, family-pref Δ {fam_d:+.3f}"
        f" → max {m.max_bias:.3f} ({'<' if bias_ok else '≥'} {g.bias_ceiling:.2f} ceiling)"
    )

    # --- §12 continuous quality score + selective CI decision ---
    decision, review_flag, quality = _decide_from_metrics(m, human_human_kappa, g)
    _score, score_lo, score_hi = _score_ci(m, g)
    notes.append(
        f"quality_score {quality:.3f} (CI [{score_lo:.3f}, {score_hi:.3f}]) vs floor "
        f"{g.quality_floor:.2f}"
    )

    if decision is SeatDecision.REJECT:
        if not accuracy_ok:
            reason = "no accuracy margin"
        elif not r_ok:
            reason = f"r<{g.agreement_r:.2f}"
        elif not kappa_floor_ok:
            reason = f"κ<{kappa_floor:.3f} one-sided floor"
        else:
            reason = f"score CI entirely below {g.quality_floor:.2f} floor"
        notes.append(f"DECISION: REJECT — {reason}")
    elif decision is SeatDecision.SCREEN:
        if m.omega is not None and m.omega < g.alt_test_omega:
            why = (
                f"alt-test ω<{g.alt_test_omega:.2f} (substitution worse than a human, "
                "Calderon 2025)"
            )
        else:
            why = (
                f"quality-score CI straddles the {g.quality_floor:.2f} floor "
                "(uncertainty, not failure → escalate, Jung 2024)"
            )
        notes.append(f"DECISION: SCREEN — {why}")
    else:
        seat_note = "DECISION: SEAT — hard gates pass and quality-score CI clears the floor"
        if review_flag:
            seat_note += " (review_flag set: super-consistent, Tier-1B — SEATED, not screened)"
        notes.append(seat_note)

    weight = _reliability_weight(quality, decision)

    metadata: dict[str, object] = {
        "human_human_kappa": human_human_kappa,
        "accuracy_wilson_lower": m.acc_lower,
        "difficulty_weighted_accuracy": m.accuracy,
        "kappa": m.kappa,
        "kappa_one_sided_floor": kappa_floor,
        "quality_score": quality,
        "score_ci": (score_lo, score_hi),
        "review_flag": review_flag,
        "thresholds": {
            "accuracy_floor": g.accuracy_floor,
            "accuracy_target": g.accuracy_target,
            "agreement_r": g.agreement_r,
            "kappa_floor_margin": g.kappa_floor_margin,
            "kappa_z_review": g.kappa_z_review,
            "quality_floor": g.quality_floor,
            "alt_test_omega": g.alt_test_omega,
            "consistency_floor": g.consistency_floor,
            "bias_ceiling": g.bias_ceiling,
            "ece_soft_ceiling": g.ece_soft_ceiling,
        },
    }
    if review_flag:
        metadata["review_reason"] = (
            "super-consistent (κ z above the human baseline; Han 2025 Tier-1B) — "
            "human review only IF κ≈1.0 co-occurs with high human disagreement"
        )

    return JudgeProfile(
        model_id=model_id,
        role=role,
        n_items=n,
        quant=quant,
        objective_accuracy=m.accuracy,
        agreement_r=m.r,
        kappa_z=m.kappa_z,
        alt_test_omega=m.omega,
        consistency=m.consistency if m.cons_measured else None,
        ece=m.ece,
        position_bias=pos_b,
        verbosity_bias=verb_b,
        family_pref_delta=fam_d,
        reliability_weight=weight,
        seat_decision=decision,
        notes=notes,
        metadata=metadata,
    )


def perturbation_audit(
    records: list[JudgmentRecord],
    *,
    human_human_kappa: float = 0.80,
    records_per_annotator: dict[str, list[JudgmentRecord]] | None = None,
    gates: SeatGates | None = None,
    jitter: float = 1.0,
) -> dict[str, object]:
    """Jitter each admission threshold ±1 SE and report the decision-flip rate — §12 / §8.3.

    Alzahrani et al. 2024 ("When Benchmarks are Targets", arXiv:2402.01781) show single
    weight perturbations flip benchmark rankings 63% of the time; §12 applies that lens to
    the *admission gate*. A robust seat/screen/reject decision should not change when a
    threshold wobbles within its own measurement noise. This audit:

    1. measures the judge once (the metrics are a function of the records, not the
       thresholds — so they are not recomputed under jitter — PIN_PER_STEP);
    2. for each scalar threshold, derives a small ``± jitter·SE`` step and re-runs the
       **pure** :func:`_decide_from_metrics` at the bumped value, holding everything else
       fixed (one-at-a-time, the Alzahrani single-perturbation design);
    3. reports the fraction of perturbed runs whose decision differs from the baseline.

    The per-threshold SE is the natural scale for each knob: thresholds that sit on the
    accuracy/score scale use the Wilson half-width of the accuracy CI over ``n_items``
    (the same small-N interval the gate itself uses); the κ floor margin uses the human
    baseline's κ SE; the rest use a small fixed delta. ``jitter`` scales the step (1.0 =
    ±1 SE).

    Args:
        records: the judge's calibration records (non-empty).
        human_human_kappa: the human baseline κ (as in :func:`build_profile`).
        records_per_annotator: optional alt-test annotator records (as in
            :func:`build_profile`).
        gates: the baseline thresholds to perturb (defaults to :class:`SeatGates`).
        jitter: SE multiplier for the perturbation step (default 1.0 = ±1 SE).

    Returns:
        A dict with:

        * ``"baseline_decision"`` — the unperturbed :class:`SeatDecision` value (str);
        * ``"flip_rate"`` — fraction of perturbed runs that changed the decision, in
          ``[0, 1]``;
        * ``"n_perturbations"`` — how many perturbed runs were evaluated;
        * ``"flips"`` — a list of ``{threshold, direction, decision}`` for the runs that
          flipped (legible "which knob is fragile" report — the andon signal).

    Raises:
        ValueError: if ``records`` is empty.
    """
    if not records:
        raise ValueError("perturbation_audit requires at least one JudgmentRecord")
    g = gates or SeatGates()
    m = _measure(
        records,
        human_human_kappa=human_human_kappa,
        records_per_annotator=records_per_annotator,
    )
    baseline, _flag, _q = _decide_from_metrics(m, human_human_kappa, g)

    # Per-threshold SE-sized step. Accuracy/score-scale knobs use the Wilson half-width of
    # the accuracy CI; the κ-floor margin uses the baseline κ SE; the rest a small delta.
    successes = round(m.accuracy * m.n_items)
    acc_lo, acc_hi = wilson_interval(successes, max(1, m.n_items), conf=0.95)
    acc_se = max(1e-3, (acc_hi - acc_lo) / 2.0)
    p0 = human_human_kappa
    kappa_se = max(1e-3, (p0 * (1.0 - p0) / max(1, m.n_items)) ** 0.5)
    base_margin = (
        g.kappa_floor_margin
        if g.kappa_floor_margin is not None
        else 1.96 * kappa_se
    )

    # (threshold name, a function applying a signed step to a copy of the gates).
    steps: dict[str, float] = {
        "accuracy_floor": acc_se,
        "agreement_r": acc_se,
        "kappa_floor_margin": kappa_se,
        "quality_floor": acc_se,
        "alt_test_omega": acc_se,
        "consistency_floor": acc_se,
        "bias_ceiling": acc_se,
        "ece_soft_ceiling": acc_se,
    }

    def _bumped(name: str, signed: float) -> SeatGates:
        if name == "kappa_floor_margin":
            return replace(g, kappa_floor_margin=base_margin + signed)
        return replace(g, **{name: getattr(g, name) + signed})

    flips: list[dict[str, object]] = []
    n_perturbations = 0
    for name, se in steps.items():
        for direction, sign in (("+", +1.0), ("-", -1.0)):
            bumped = _bumped(name, sign * jitter * se)
            decision, _rf, _qq = _decide_from_metrics(m, human_human_kappa, bumped)
            n_perturbations += 1
            if decision is not baseline:
                flips.append(
                    {"threshold": name, "direction": direction, "decision": decision.value}
                )

    flip_rate = len(flips) / n_perturbations if n_perturbations else 0.0
    return {
        "baseline_decision": baseline.value,
        "flip_rate": flip_rate,
        "n_perturbations": n_perturbations,
        "flips": flips,
    }
