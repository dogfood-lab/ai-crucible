"""The judge-admission metrics (research-grounding §11.1, recalibrated by §12).

Each function is a **pure, deterministic** reduction over a ``list[JudgmentRecord]``
— same records grade identically tomorrow (the §1 "live external dependencies are
forbidden" discipline that ``crucible.scoring.stats`` already follows). These are the
measurement primitives the §11.1 *seat-or-screen* admission test is built on; the
gate logic that turns these numbers into a :class:`~crucible.characterize.types.SeatDecision`
lives in :mod:`crucible.characterize.profile`, and the panel-level aggregation in
:mod:`crucible.characterize.aggregate`.

**§12 calibration redesign (2026-06-01, after the first characterization run).** The
first run self-diagnosed two defects this module + the gate layer now fix:

* The κ two-gate was **INVERTED** — it screened super-consistent judges (κ above the
  human baseline, z>1) when Han et al. 2025 ("Judge's Verdict", arXiv:2510.09738)
  keeps them as **Tier-1B: valid, top-ranked, seated**. The gate becomes a *one-sided
  floor* (see :mod:`crucible.characterize.profile`); :func:`kappa_zscore` still reports
  the signed z (the profile layer reads ``z>1`` only as a *review flag*, never a
  downgrade).
* The brittle 7-threshold binary AND is replaced by a **difficulty-normalized
  continuous quality score** (:func:`difficulty_weighted_accuracy` + :func:`quality_score`)
  with a **selective, CI-based** seat/screen/reject decision and a **perturbation audit**
  (jitter thresholds ±1 SE, report the decision-flip rate — the §8.3 Alzahrani lens on
  the admission gate). These three live in :mod:`crucible.characterize.profile`; the
  difficulty-weighting + score combiner that feed them are the pure reductions here.

The six (each tied to the paper that grounds it):

1. **Objective accuracy** — fraction ``predicted == gold`` on ground-truth-labeled
   items. Many strong judges sit barely above 0.5, so the gate (in ``profile``)
   requires a *real margin*, not just >0.5 (Tan et al. 2024, "JudgeBench",
   arXiv:2410.12784).
2. **Agreement (two-gate)** — Pearson ``r`` vs gold/human **and** Cohen's κ, plus a
   κ *z-score* against the human–human baseline (|z|<1 = "human-like"; size is not
   decisive — a 4B can correlate yet fail consistency) (Han et al. 2025, "Judge's
   Verdict", arXiv:2510.09738).
3. **Substitution (alt-test)** — leave-one-annotator-out winning rate ω; seat only
   if ω ≥ 0.5, else screen-only (Calderon, Reichart & Dror 2025, "alt-test",
   arXiv:2501.10970).
4. **Consistency / test-retest** — 1 − verdict-flip-rate across repeated ``run_index``
   passes; LLM judges have low intra-rater reliability (Haldar & Hockenmaier 2025,
   "Rating Roulette", arXiv:2510.27106).
5. **Calibration (ECE)** — expected calibration error from confidence vs correctness;
   down-weight overconfident judges (quantization worsens calibration — Proskurina et
   al. 2024, arXiv:2405.00632). Per §12, :func:`expected_calibration_error` returns
   ``None`` (not an error) when no record carries a confidence — "not measured" is a
   first-class outcome the continuous score handles, since the calibration set may have
   no logprob/vote-fraction confidence wired yet.
6. **Bias panel** — position-swap flip-rate, verbosity bias, and a same-vs-different-
   family agreement delta (self/family preference, §1/§3).

Two §12-added reductions feed the continuous gate:

* **Difficulty-weighted accuracy** (:func:`difficulty_weighted_accuracy`) — weights each
  item by ``metadata["difficulty"]`` when present (else unweighted), the model-free
  stand-in for IRT θ-normalization on a ~6-respondent panel where classical 2PL MLE is
  infeasible (ATLAS, Peiyu Li et al. 2025, arXiv:2511.04689; sample-size, Schroeders &
  Gnambs 2025). Raw accuracy is distorted on easy-only sets — crucible's exact failure.
* **Continuous quality score** (:func:`quality_score`) — a calibrated 0–1 combiner of
  difficulty-weighted accuracy + agreement + consistency, minus ECE and bias penalties
  (§12 Q4). Monotone in every "good" component, so the profile layer can push an
  accuracy confidence interval through it to get a score CI for the selective decision
  (Traub 2024, arXiv:2407.01032 — score across the operating curve, not at one cut).

Implementation reuses scipy (Pearson) and statsmodels (Cohen's κ via
``inter_rater.cohens_kappa``); it deliberately does **not** edit ``crucible.scoring``
— it imports nothing mutable from it. Where a metric is structurally a proportion we
let the *profile* layer reach for Wilson bounds from ``crucible.scoring.stats`` rather
than duplicating them here (keeps each metric a single number; the gate layer adds the
uncertainty).

Standards compliance (the six): see :mod:`crucible.characterize.profile` for the full
section — these primitives carry **PIN_PER_STEP** (pure + deterministic → a profile run
is byte-for-byte replayable) and feed **EXTERNAL_VERIFIER** (the panel that consumes
them is cross-family).
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import replace

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.stats import norm, pearsonr
from statsmodels.stats.inter_rater import cohens_kappa

from crucible.characterize.types import JudgmentRecord

_TEMP_EPS = 1e-6  # clamp confidences off {0, 1} so the logit is finite

__all__ = [
    "objective_accuracy",
    "difficulty_weighted_accuracy",
    "agreement",
    "kappa_zscore",
    "human_like_kappa",
    "alt_test_omega",
    "consistency",
    "expected_calibration_error",
    "position_bias",
    "verbosity_bias",
    "family_pref_delta",
    "quality_score",
]


def _require_records(records: list[JudgmentRecord], what: str) -> None:
    """Shared guard — raise a legible ``ValueError`` (not a bare assertion).

    The metrics are the input layer of a measurement instrument; an empty record
    list is a caller error, not a 0.0 to silently propagate into a seat decision.
    """
    if not records:
        raise ValueError(f"{what} requires at least one JudgmentRecord, got none")


def _is_correct(rec: JudgmentRecord) -> bool:
    """Whether a record's prediction matched gold.

    Prefers the scorer-filled ``correct`` flag (the contract says the scorer fills
    it); falls back to ``predicted == gold`` when it is ``None`` so the metrics work
    on raw records too. This keeps the source of truth single while staying robust to
    un-scored input.
    """
    if rec.correct is not None:
        return bool(rec.correct)
    return rec.predicted == rec.gold


def objective_accuracy(records: list[JudgmentRecord]) -> float:
    """Fraction of items the judge got right (``predicted == gold``) — §11.1 #1.

    JudgeBench-style ground-truth accuracy (Tan et al. 2024, arXiv:2410.12784). The
    *seat gate* (in :mod:`~crucible.characterize.profile`) requires a real margin over
    0.5, since strong judges often sit barely above chance.

    Args:
        records: the judge's records on ground-truth-labeled items (non-empty).

    Returns:
        Accuracy in ``[0.0, 1.0]``.

    Raises:
        ValueError: if ``records`` is empty.
    """
    _require_records(records, "objective_accuracy")
    hits = sum(1 for r in records if _is_correct(r))
    return hits / len(records)


def difficulty_weighted_accuracy(records: list[JudgmentRecord]) -> float:
    """Difficulty-normalized accuracy — §12 Q4 (the saturating-set fix).

    Raw accuracy is distorted on easy-only sets: the first characterization run had
    three models score 1.00 on a set that *saturated*, giving the gate no discrimination
    at the top. §12's fix is to score difficulty-normalized accuracy — IRT θ adjusts for
    item difficulty (ATLAS, Peiyu Li et al. 2025, arXiv:2511.04689), but with ~6
    respondents classical 2PL MLE is infeasible (needs ~200–500 — Schroeders & Gnambs
    2025, doi:10.1177/25152459251314798), so we use the **model-free** stand-in: weight
    each item by its declared difficulty.

    Each record's difficulty is read from ``metadata["difficulty"]`` (a number, higher =
    harder). Getting a *hard* item right is worth more than getting an easy one right, so
    the weighted accuracy is ``Σ w_i·correct_i / Σ w_i`` over items. When **no** record
    carries a difficulty the function degrades gracefully to plain
    :func:`objective_accuracy` (unweighted) — "not annotated" is the common path for the
    trivial-anchor items and must not penalize the judge. A non-positive or non-numeric
    difficulty is treated as "unweighted for that item" (weight 1.0) rather than raising,
    so a partially-annotated set still scores.

    Where an item is judged multiple times (test-retest ``run_index`` passes) each record
    contributes (the repeats just reinforce the per-item weight) — the result stays in
    ``[0, 1]`` because it is a convex combination of 0/1 correctness indicators.

    Args:
        records: the judge's records on labeled items (non-empty); difficulty optional
            in ``metadata["difficulty"]``.

    Returns:
        Difficulty-weighted accuracy in ``[0.0, 1.0]`` (== :func:`objective_accuracy`
        when no item carries a usable difficulty).

    Raises:
        ValueError: if ``records`` is empty.
    """
    _require_records(records, "difficulty_weighted_accuracy")
    num = 0.0
    den = 0.0
    any_weighted = False
    for r in records:
        raw = r.metadata.get("difficulty")
        weight = 1.0
        if raw is not None:
            try:
                w = float(raw)
            except (TypeError, ValueError):
                w = 0.0
            if w > 0.0:
                weight = w
                any_weighted = True
        num += weight * (1.0 if _is_correct(r) else 0.0)
        den += weight
    if not any_weighted:
        # No difficulty signal anywhere → plain accuracy (the convex sum collapses to it
        # with all-equal weights, but compute it directly so the contract is obvious).
        return objective_accuracy(records)
    return num / den


def _numeric_pairs(records: list[JudgmentRecord]) -> tuple[list[float], list[float]]:
    """Extract ``(predicted, gold)`` as parallel float lists for correlation.

    ``predicted``/``gold`` are ``Any`` in the contract; agreement is only meaningful
    when they are numeric (a rating/score). Booleans are accepted (cast to 0/1). A
    value that cannot be coerced to float raises a legible error rather than poisoning
    the correlation with a silent NaN.
    """
    pred: list[float] = []
    gold: list[float] = []
    for r in records:
        try:
            pred.append(float(r.predicted))
            gold.append(float(r.gold))
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "agreement requires numeric predicted/gold values "
                f"(item {r.item_id!r} had predicted={r.predicted!r}, gold={r.gold!r})"
            ) from exc
    return pred, gold


def agreement(records: list[JudgmentRecord]) -> tuple[float, float]:
    """Agreement with gold/human as ``(Pearson r, Cohen's κ)`` — §11.1 #2.

    The two-gate (Han et al. 2025, arXiv:2510.09738): ``r`` measures linear
    agreement on the rating scale; κ measures categorical agreement corrected for
    chance. The *profile* layer applies the gate ``r ≥ 0.80`` **and** ``|κ z| < 1``
    (the κ z-score vs the human–human baseline — see :func:`kappa_zscore`).

    Pearson is computed via ``scipy.stats.pearsonr``; κ via
    ``statsmodels.stats.inter_rater.cohens_kappa`` on the predicted×gold contingency
    table built with ``to_table`` (categorical agreement, so the raw category labels —
    not floats — are used).

    Degenerate cases handled explicitly:

    * **Single item** — Pearson is undefined for n<2; we return ``r = 0.0`` (no
      evidence of correlation from one point), and κ over the 1×1 table.
    * **Zero variance** in predicted or gold (e.g. the judge gave every item the same
      score) — Pearson is undefined (would be NaN); we return ``r = 0.0``. A constant
      predictor has *no* linear agreement, and 0.0 fails the ``r ≥ 0.80`` gate, which
      is the correct outcome (a judge that says "5" to everything is useless).

    Args:
        records: the judge's records on labeled items (non-empty).

    Returns:
        ``(pearson_r, cohens_kappa)`` — both built-in floats. κ is in ``[-1, 1]``;
        r is in ``[-1, 1]`` (0.0 in the degenerate cases above).

    Raises:
        ValueError: if ``records`` is empty or values are non-numeric.
    """
    _require_records(records, "agreement")
    pred, gold = _numeric_pairs(records)

    # --- Pearson r (linear agreement on the rating scale) ---
    if len(pred) < 2 or np.std(pred) == 0.0 or np.std(gold) == 0.0:
        # n<2 or a constant column -> Pearson is undefined; "no linear agreement".
        r = 0.0
    else:
        r = float(pearsonr(pred, gold).statistic)

    # --- Cohen's κ (chance-corrected categorical agreement) ---
    # Use the *raw* category labels (predicted/gold as given) so κ measures agreement
    # on the discrete verdict, not on a float coercion. ``_cohens_kappa_from_pairs``
    # builds the square contingency table cohens_kappa needs over the union of labels.
    cats = [(r_.predicted, r_.gold) for r_ in records]
    kappa = _cohens_kappa_from_pairs(cats)
    return (r, kappa)


def _cohens_kappa_from_pairs(pairs: list[tuple[object, object]]) -> float:
    """Cohen's κ on a list of ``(rater_a, rater_b)`` categorical pairs.

    Builds a square contingency table over the *union* of labels seen in either
    column (statsmodels' ``cohens_kappa`` requires a square matrix), then defers the
    arithmetic to ``statsmodels.stats.inter_rater.cohens_kappa``. Returns κ as a
    built-in float.

    Degenerate case: when both raters used exactly one (identical) label for every
    item, agreement is perfect *and* there is no chance variance, so κ is undefined
    (0/0). statsmodels returns NaN there; we map that to ``1.0`` — perfect agreement
    on a single category is perfect agreement, which is the intuitive reading and the
    outcome the seat gate should reward.
    """
    labels = sorted({lbl for pair in pairs for lbl in pair}, key=repr)
    index = {lbl: i for i, lbl in enumerate(labels)}
    k = len(labels)
    if k <= 1:
        # Single shared category: agreement is perfect but chance variance is 0, so κ is
        # the indeterminate 0/0 (statsmodels would emit a divide-by-zero RuntimeWarning
        # and return NaN). Short-circuit to 1.0 — trivial perfect agreement reads as
        # perfect agreement — without invoking the NaN path at all.
        return 1.0
    matrix = np.zeros((k, k), dtype=float)
    for a, b in pairs:
        matrix[index[a], index[b]] += 1.0

    res = cohens_kappa(matrix)
    kappa = float(res.kappa)
    if np.isnan(kappa):
        # Defensive: any residual 0/0 (e.g. all mass on one cell of a larger table) is
        # still perfect agreement.
        return 1.0
    return kappa


def kappa_zscore(kappa: float, human_human_kappa: float, n: int) -> float:
    """The §11.1 κ two-gate: how far the judge's κ sits from the human baseline.

    The "Judge's Verdict" two-gate (Han et al. 2025, arXiv:2510.09738) does not ask
    "is κ high"; it asks "is the judge's agreement *indistinguishable from a human
    annotator's*". A judge that agrees with gold **more** than humans agree with each
    other is suspiciously un-human (often overfit / sycophantic), just as one that
    agrees far less is too weak. So we standardize the judge's κ against the
    human–human κ and read |z|<1 as "human-like".

    z is computed using the large-sample standard error of κ under the null of the
    human baseline, ``SE = sqrt(p0 (1 - p0) / n)`` with ``p0`` the human–human
    observed-agreement proxy ``human_human_kappa`` (Fleiss/Cohen large-sample
    approximation). This is the same admissibility spirit as ``scoring.stats`` using
    Wilson over Wald at small N — a transparent, defensible SE rather than a black box.

    Args:
        kappa: the judge's Cohen's κ vs gold (from :func:`agreement`).
        human_human_kappa: the human–human baseline κ for this task (the anchor).
        n: number of items κ was computed over (n ≥ 1).

    Returns:
        The signed z-score ``(kappa - human_human_kappa) / SE``. ``|z| < 1`` passes
        the human-like gate. Returns ``0.0`` when the baseline leaves no variance
        (``human_human_kappa`` at 0 or 1) — there is no scale to be far on, so the
        judge cannot be flagged as non-human by this gate.

    Raises:
        ValueError: if ``n < 1`` or ``human_human_kappa`` is outside ``[-1, 1]``.
    """
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    if not -1.0 <= human_human_kappa <= 1.0:
        raise ValueError(
            f"human_human_kappa must be in [-1, 1], got {human_human_kappa}"
        )
    # SE of a proportion-like statistic; p0 in (0,1) required for a finite SE.
    p0 = human_human_kappa
    var = p0 * (1.0 - p0) / n
    if var <= 0.0:
        return 0.0
    se = var**0.5
    return float((kappa - human_human_kappa) / se)


def human_like_kappa(kappa: float, human_human_kappa: float, n: int) -> bool:
    """Convenience wrapper: ``|kappa_zscore| < 1`` (the §11.1 "human-like" verdict)."""
    return abs(kappa_zscore(kappa, human_human_kappa, n)) < 1.0


def alt_test_omega(records_per_annotator: dict[str, list[JudgmentRecord]]) -> float:
    """Leave-one-annotator-out winning rate ω — §11.1 #3 (the alt-test).

    The alt-test (Calderon, Reichart & Dror 2025, arXiv:2501.10970) asks the
    *substitution* question directly: if we swapped a human annotator for the
    candidate judge, would the panel be at least as good? For each human annotator we
    leave out, we check whether the judge agrees with the *held-out* human's labels at
    least as well as the *remaining* humans do. ω is the fraction of leave-one-out
    folds the judge "wins" (ties count as wins — substitution is acceptable when the
    judge is *no worse*). Seat iff ``ω ≥ 0.5``; otherwise screen-only.

    Concretely, per held-out annotator ``h`` (with item-aligned labels):

    * judge score = mean agreement(judge_label, h_label) over shared items;
    * human score = mean over the *other* annotators of mean agreement(other_label,
      h_label) over shared items;
    * the fold is a *win* for the judge iff ``judge_score >= human_score``.

    Agreement here is exact-match on the verdict (the labels are categorical
    verdicts). The judge's labels are read from a reserved ``"judge"`` key in
    ``records_per_annotator``; every other key is a human annotator.

    Args:
        records_per_annotator: ``{annotator_id: [JudgmentRecord, ...]}``. Must contain
            a ``"judge"`` entry (the candidate) and **at least two** human annotators
            (you cannot run leave-one-out with fewer than two humans — there would be
            no "remaining humans" to compare against).

    Returns:
        ω in ``[0.0, 1.0]`` — the fraction of leave-one-human-out folds the judge wins.

    Raises:
        ValueError: if the ``"judge"`` entry is missing or there are < 2 human
            annotators.
    """
    if "judge" not in records_per_annotator:
        raise ValueError("alt_test_omega requires a 'judge' entry (the candidate)")
    humans = {k: v for k, v in records_per_annotator.items() if k != "judge"}
    if len(humans) < 2:
        raise ValueError(
            "alt_test_omega requires >= 2 human annotators for leave-one-out, "
            f"got {len(humans)}"
        )

    judge_by_item = {r.item_id: r.predicted for r in records_per_annotator["judge"]}
    human_by_item = {
        h: {r.item_id: r.predicted for r in recs} for h, recs in humans.items()
    }

    def _mean_match(
        a: dict[str, object], b: dict[str, object]
    ) -> float | None:
        shared = a.keys() & b.keys()
        if not shared:
            return None
        return sum(1.0 for i in shared if a[i] == b[i]) / len(shared)

    wins = 0
    folds = 0
    for held_out in humans:
        held_labels = human_by_item[held_out]
        judge_score = _mean_match(judge_by_item, held_labels)
        if judge_score is None:
            continue  # judge shares no items with this annotator; skip the fold
        others = [human_by_item[h] for h in humans if h != held_out]
        other_scores = [
            s for o in others if (s := _mean_match(o, held_labels)) is not None
        ]
        if not other_scores:
            continue
        human_score = sum(other_scores) / len(other_scores)
        folds += 1
        if judge_score >= human_score:  # tie -> win (no worse than a human)
            wins += 1

    if folds == 0:
        raise ValueError(
            "alt_test_omega found no comparable leave-one-out folds "
            "(the judge shares no items with the human annotators)"
        )
    return wins / folds


def consistency(records: list[JudgmentRecord]) -> float:
    """Test-retest stability ``1 − verdict-flip-rate`` across ``run_index`` — §11.1 #4.

    LLM judges have low intra-rater reliability (Haldar & Hockenmaier 2025,
    "Rating Roulette", arXiv:2510.27106): the same fixture scored 3–5× can yield
    different verdicts. We group records by ``item_id``, and an item *flips* iff its
    ``predicted`` verdict is not identical across all of its ``run_index`` passes.
    Consistency is ``1 − (flipped items / items with ≥2 passes)``.

    Items seen only once carry no test-retest information and are excluded from the
    denominator (they can neither flip nor prove stability). If *no* item was scored
    more than once, consistency is undefined and we return ``1.0`` with the
    understanding that the *profile* layer records "consistency not measured" — but the
    common path (the §11.3 test-retest duplicates) always supplies repeats.

    Args:
        records: the judge's records, including repeated ``run_index`` passes per item
            (non-empty).

    Returns:
        Consistency in ``[0.0, 1.0]`` — 1.0 means every repeated item gave an
        identical verdict every time.

    Raises:
        ValueError: if ``records`` is empty.
    """
    _require_records(records, "consistency")
    by_item: dict[str, list[object]] = defaultdict(list)
    for r in records:
        by_item[r.item_id].append(r.predicted)

    repeated = {i: v for i, v in by_item.items() if len(v) >= 2}
    if not repeated:
        return 1.0  # no test-retest signal; profile layer notes "not measured"
    flipped = sum(1 for v in repeated.values() if len({repr(x) for x in v}) > 1)
    return 1.0 - flipped / len(repeated)


def expected_calibration_error(
    records: list[JudgmentRecord], *, n_bins: int = 10
) -> float | None:
    """Expected Calibration Error from confidence vs correctness — §11.1 #5 / §12.

    ECE (the standard binned estimator — Naeini et al. 2015 / Guo et al. 2017,
    grounded for judges by Proskurina et al. 2024, arXiv:2405.00632, which shows
    quantization worsens calibration). Records are bucketed into ``n_bins`` equal-width
    confidence bins over ``[0, 1]``; per bin we take ``|mean_confidence −
    mean_accuracy|`` and weight by the bin's share of records:

        ECE = Σ_b (|B_b| / N) · |acc(B_b) − conf(B_b)|

    A perfectly calibrated judge (confidence == probability-correct in every bin) has
    ECE 0.0; an overconfident judge (high confidence, low accuracy) has ECE → 1.0. The
    *profile* layer down-weights high-ECE judges rather than rejecting them outright
    (calibration is a soft signal, §11.1).

    **§12 contract change.** When **no** record carries a confidence this returns
    ``None`` ("calibration not measured") instead of raising — the §12 finding is that
    the verbalized number is untrustworthy (Xiong et al. 2024, arXiv:2306.13063) and the
    real signal (verdict-token logprob / panel-vote fraction; Ollama exposes
    ``logprobs`` since v0.12.11) may not be wired for a given calibration run. A missing
    calibration signal must not fail a judge, so the gate treats ``None`` as a neutral
    (no down-weight). An out-of-range confidence is still a caller error and raises.

    Args:
        records: the judge's records; those with ``confidence is None`` are ignored.
        n_bins: number of equal-width confidence bins (default 10; must be ≥ 1).

    Returns:
        ECE in ``[0.0, 1.0]``, or ``None`` when no record carries a confidence.

    Raises:
        ValueError: if ``records`` is empty, ``n_bins < 1``, or a confidence is outside
            ``[0, 1]``.
    """
    _require_records(records, "expected_calibration_error")
    if n_bins < 1:
        raise ValueError(f"n_bins must be >= 1, got {n_bins}")

    conf: list[float] = []
    corr: list[float] = []
    for r in records:
        if r.confidence is None:
            continue
        if not 0.0 <= r.confidence <= 1.0:
            raise ValueError(
                f"confidence must be in [0, 1], got {r.confidence} for item {r.item_id!r}"
            )
        conf.append(float(r.confidence))
        corr.append(1.0 if _is_correct(r) else 0.0)

    if not conf:
        # §12: "calibration not measured" is a first-class outcome, not an error — the
        # confidence channel (logprob / vote-fraction) may simply not be wired this run.
        return None

    confidence = np.asarray(conf)
    correct = np.asarray(corr)
    n = confidence.size
    # Equal-width bin edges over [0, 1]; np.clip pulls a confidence of exactly 1.0
    # into the top bin rather than a phantom (n_bins)-th bin.
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_idx = np.clip(np.digitize(confidence, edges[1:-1], right=False), 0, n_bins - 1)

    ece = 0.0
    for b in range(n_bins):
        mask = bin_idx == b
        count = int(mask.sum())
        if count == 0:
            continue
        gap = abs(correct[mask].mean() - confidence[mask].mean())
        ece += (count / n) * gap
    return float(ece)


def apply_temperature(confidence: float, temperature: float) -> float:
    """Rescale a confidence by a post-hoc temperature ``T`` — §12 Q3 (Guo et al. 2017).

    Temperature scaling operates on the **logit**: ``p' = σ(logit(p) / T)``. ``T > 1``
    *softens* an overconfident probability toward 0.5; ``T < 1`` sharpens; ``T == 1`` is
    the identity. The confidence is clamped off ``{0, 1}`` so the logit is finite.

    Args:
        confidence: a probability in ``[0, 1]`` (e.g. a verdict-token logprob → exp).
        temperature: the scaling temperature ``T`` (> 0).

    Returns:
        The recalibrated probability in ``(0, 1)``.

    Raises:
        ValueError: if ``temperature <= 0`` or ``confidence`` is outside ``[0, 1]``.
    """
    if temperature <= 0.0:
        raise ValueError(f"temperature must be > 0, got {temperature}")
    if not 0.0 <= confidence <= 1.0:
        raise ValueError(f"confidence must be in [0, 1], got {confidence}")
    p = min(max(confidence, _TEMP_EPS), 1.0 - _TEMP_EPS)
    z = math.log(p / (1.0 - p)) / temperature
    return 1.0 / (1.0 + math.exp(-z))


def fit_temperature(records: list[JudgmentRecord], *, max_temp: float = 100.0) -> float:
    """Fit the single post-hoc temperature ``T`` that best calibrates confidence — §12 Q3.

    Temperature scaling (Guo et al. 2017, "On Calibration of Modern Neural Networks",
    arXiv:1706.04599) fits one scalar ``T`` minimizing the negative log-likelihood of
    correctness under ``σ(logit(confidence) / T)``. §12 Q3 prescribes it as the post-hoc
    calibration step on the verdict-token logprob confidence (NOT a verbalized number —
    Xiong et al. 2024). It cannot change *which* verdict a judge picked (monotonic in the
    confidence), only how well the confidence *magnitude* tracks correctness — so it
    lowers ECE without touching accuracy/agreement.

    The fit is **in-sample** unless the caller splits the records; a held-out split is the
    honest next step (see :func:`temperature_scaled_ece`'s note + the run report caveat).

    Args:
        records: the judge's records; those with ``confidence is None`` are ignored.
        max_temp: the upper bound on the search (default 100; the lower bound is 0.05).

    Returns:
        The fitted ``T``. Returns ``1.0`` (the identity — no scaling) when calibration
        cannot be fit: fewer than two confidences, or only one correctness class present
        (all-right or all-wrong gives a degenerate, overfit ``T``).

    Raises:
        ValueError: if ``records`` is empty or a confidence is outside ``[0, 1]``.
    """
    _require_records(records, "fit_temperature")
    conf: list[float] = []
    corr: list[float] = []
    for r in records:
        if r.confidence is None:
            continue
        if not 0.0 <= r.confidence <= 1.0:
            raise ValueError(
                f"confidence must be in [0, 1], got {r.confidence} for item {r.item_id!r}"
            )
        conf.append(float(r.confidence))
        corr.append(1.0 if _is_correct(r) else 0.0)

    if len(conf) < 2:
        return 1.0
    y = np.asarray(corr)
    if y.min() == y.max():
        # one correctness class only — NLL is minimized by T→0 (a degenerate sharpen);
        # there is nothing to calibrate against, so return the identity.
        return 1.0

    p = np.clip(np.asarray(conf), _TEMP_EPS, 1.0 - _TEMP_EPS)
    z = np.log(p / (1.0 - p))

    def _nll(temp: float) -> float:
        scaled = 1.0 / (1.0 + np.exp(-z / temp))
        scaled = np.clip(scaled, _TEMP_EPS, 1.0 - _TEMP_EPS)
        return float(-np.mean(y * np.log(scaled) + (1.0 - y) * np.log(1.0 - scaled)))

    res = minimize_scalar(_nll, bounds=(0.05, max_temp), method="bounded")
    return float(res.x) if res.success else 1.0


def temperature_scaled_ece(
    records: list[JudgmentRecord],
    *,
    temperature: float | None = None,
    n_bins: int = 10,
) -> tuple[float, float | None, float | None]:
    """Fit (or apply) a temperature and report ECE before vs after — §12 Q3.

    Returns ``(temperature, ece_raw, ece_scaled)``. ``temperature`` is fit via
    :func:`fit_temperature` when not supplied. ``ece_raw``/``ece_scaled`` are the
    :func:`expected_calibration_error` before and after rescaling each confidence by
    :func:`apply_temperature`; both are ``None`` when no record carries a confidence (the
    "calibration not measured" outcome — §12). The fit is in-sample (see the caveat in
    :func:`fit_temperature`): a proper held-out split is the honest improvement.

    Args:
        records: the judge's records (non-empty).
        temperature: use this ``T`` instead of fitting one (e.g. a held-out fit).
        n_bins: ECE bin count (passed through to :func:`expected_calibration_error`).

    Returns:
        ``(T, ece_raw, ece_scaled)`` — ``T`` is ``1.0`` when calibration is unmeasured.
    """
    raw = expected_calibration_error(records, n_bins=n_bins)
    if raw is None:
        return (1.0, None, None)
    temp = fit_temperature(records) if temperature is None else float(temperature)
    if temp <= 0.0:
        raise ValueError(f"temperature must be > 0, got {temp}")
    scaled_records = [
        r if r.confidence is None else replace(r, confidence=apply_temperature(r.confidence, temp))
        for r in records
    ]
    scaled = expected_calibration_error(scaled_records, n_bins=n_bins)
    return (temp, raw, scaled)


def temperature_scaled_ece_cv(
    records: list[JudgmentRecord],
    *,
    folds: int = 5,
    n_bins: int = 10,
) -> tuple[float, float | None, float | None]:
    """Held-out (k-fold) temperature scaling — the honest, out-of-sample ECE (§12 Q3).

    :func:`temperature_scaled_ece` fits ``T`` and measures ECE on the **same** records —
    optimistic. This estimates the ECE temperature scaling actually buys on *unseen* data:
    split the calibration set into ``folds`` folds, fit ``T`` on the train folds, apply it
    to the held-out fold, and pool the held-out (scaled) records for one ECE over the whole
    set — each record scaled by a ``T`` fit **without** it.

    **No rerun leakage.** Folds are grouped by ``item_id``: every test-retest rerun of an
    item lands in the *same* fold, so a temperature is never fit on one rerun of an item
    and tested on another (which would leak). Folds are assigned round-robin over sorted
    item ids (deterministic — PIN_PER_STEP).

    Degenerate folds are handled the honest way: a train fold with only one correctness
    class yields ``T = 1.0`` (identity — "no calibration learned here"), so those held-out
    records are simply unscaled rather than scaled by an overfit temperature.

    Args:
        records: the judge's records (non-empty).
        folds: target fold count (default 5); clamped to ``[2, n_items]``. With one item
            cross-validation is impossible and this falls back to the in-sample
            :func:`temperature_scaled_ece` (documented, not silent).
        n_bins: ECE bin count.

    Returns:
        ``(mean_temperature, ece_raw, ece_cv)`` — ``mean_temperature`` is the mean of the
        per-fold fitted ``T``; ``ece_raw`` is the unscaled ECE over all records; ``ece_cv``
        is the pooled held-out scaled ECE. ``(1.0, None, None)`` when confidence is
        unmeasured.
    """
    raw = expected_calibration_error(records, n_bins=n_bins)
    if raw is None:
        return (1.0, None, None)

    by_item: dict[str, list[JudgmentRecord]] = defaultdict(list)
    for r in records:
        if r.confidence is not None:
            by_item[r.item_id].append(r)
    items = sorted(by_item)
    if len(items) < 2:
        # cannot hold anything out with a single item — fall back, honestly.
        return temperature_scaled_ece(records, n_bins=n_bins)

    k = max(2, min(folds, len(items)))
    fold_of = {item: i % k for i, item in enumerate(items)}

    temps: list[float] = []
    pooled_scaled: list[JudgmentRecord] = []
    for f in range(k):
        train = [r for it in items if fold_of[it] != f for r in by_item[it]]
        test = [r for it in items if fold_of[it] == f for r in by_item[it]]
        if not test:
            continue
        temp = fit_temperature(train) if train else 1.0
        temps.append(temp)
        pooled_scaled.extend(
            replace(r, confidence=apply_temperature(r.confidence, temp)) for r in test
        )

    ece_cv = expected_calibration_error(pooled_scaled, n_bins=n_bins)
    mean_temp = float(np.mean(temps)) if temps else 1.0
    return (mean_temp, raw, ece_cv)


def position_bias(records: list[JudgmentRecord]) -> float:
    """Position-swap flip-rate — §11.1 #6 (bias panel).

    Position bias (Shi et al. 2024, arXiv:2406.07791 — research-grounding §3) is the
    pairwise judge's tendency to favor an option because of *where* it sits (A-first vs
    B-first), independent of content. We pair records of the *same item* presented in
    different ``position`` orderings and measure how often the verdict flips with the
    swap. 0.0 = position-invariant (good); 1.0 = the verdict is decided entirely by
    position (a useless judge).

    Records are grouped by ``item_id``; an item *contributes* only if it has records
    under ≥2 distinct ``position`` values. The item is counted as *flipped* iff its
    ``predicted`` verdict is not identical across those positions. The rate is over
    contributing items.

    Args:
        records: the judge's records, with ``position`` set for position-swap trials.

    Returns:
        Flip-rate in ``[0.0, 1.0]``. Returns ``0.0`` when no item was presented in
        ≥2 positions (no position-swap evidence → not flagged; the profile layer notes
        "position bias not measured").

    Raises:
        ValueError: if ``records`` is empty.
    """
    _require_records(records, "position_bias")
    by_item: dict[str, dict[int, list[object]]] = defaultdict(lambda: defaultdict(list))
    for r in records:
        if r.position is None:
            continue
        by_item[r.item_id][r.position].append(r.predicted)

    contributing = {i: pos for i, pos in by_item.items() if len(pos) >= 2}
    if not contributing:
        return 0.0
    flipped = 0
    for positions in contributing.values():
        # Verdict per position = the (first) predicted under that position. The item
        # flips iff the set of per-position verdicts has >1 distinct value.
        verdicts = {repr(v[0]) for v in positions.values()}
        if len(verdicts) > 1:
            flipped += 1
    return flipped / len(contributing)


def verbosity_bias(records: list[JudgmentRecord]) -> float:
    """Verbosity-preference correlation — §11.1 #6 (bias panel).

    Verbosity bias (Zheng et al. 2023, arXiv:2306.05685 — §3): judges over-reward
    longer answers regardless of quality. The signal is the Pearson correlation
    between answer *length* and the judge's score, **partialled against correctness**
    so legitimate "longer because more complete (and correct)" answers don't read as
    bias. We approximate the partialling cheaply: correlate length with the judge's
    *error* (predicted − gold) rather than raw score — a judge with no verbosity bias
    has length uncorrelated with its error; a verbose-biased judge's error grows with
    length (it inflates long answers). The returned value is ``|r|`` so it is a
    magnitude the gate can threshold (0.0 = unbiased; → 1.0 = strongly length-driven).

    Length is read from ``metadata["answer_len"]`` (chars/tokens — the kernel records
    it when it stages a verbosity-swap trial). Records lacking it are skipped.

    Args:
        records: the judge's records; those without ``metadata["answer_len"]`` (or with
            non-numeric predicted/gold) are skipped.

    Returns:
        ``|Pearson r|`` in ``[0.0, 1.0]`` between answer length and judge error.
        Returns ``0.0`` when fewer than 2 usable records exist or length/error has zero
        variance (no measurable association → not flagged).

    Raises:
        ValueError: if ``records`` is empty.
    """
    _require_records(records, "verbosity_bias")
    lengths: list[float] = []
    errors: list[float] = []
    for r in records:
        raw_len = r.metadata.get("answer_len")
        if raw_len is None:
            continue
        try:
            length = float(raw_len)
            err = float(r.predicted) - float(r.gold)
        except (TypeError, ValueError):
            continue
        lengths.append(length)
        errors.append(err)

    if len(lengths) < 2 or np.std(lengths) == 0.0 or np.std(errors) == 0.0:
        return 0.0
    return float(abs(pearsonr(lengths, errors).statistic))


def family_pref_delta(records: list[JudgmentRecord]) -> float:
    """Same-family minus different-family agreement — §11.1 #6 / §1 / §3.

    Self/family preference (Panickssery et al. 2024, arXiv:2404.13076; mechanism in
    Wataoka et al. 2024, arXiv:2410.21819 — §1): a judge over-rates outputs from its
    own model family. We split the judge's records by whether the *judged output's*
    ``family`` matches the judge's own family and take

        Δ = mean_accuracy(same-family) − mean_accuracy(different-family).

    A Δ near 0 is unbiased; a large positive Δ means the judge agrees with gold *more*
    on its own family's outputs (it is being lenient/favorable to kin — the §3 bias the
    cross-family panel exists to defend against). Δ can be negative (anti-kin), which
    the gate also treats as a magnitude.

    Each record's *judged-output* family is read from ``metadata["judged_family"]``;
    the judge's *own* family from ``metadata["judge_family"]`` (the kernel tags both
    when it stages the family-preference trial). Records missing either are skipped.

    Args:
        records: the judge's records tagged with ``judged_family`` + ``judge_family``.

    Returns:
        Δ in ``[-1.0, 1.0]`` (same- minus different-family accuracy). Returns ``0.0``
        when either subgroup is empty (no contrast to measure → not flagged).

    Raises:
        ValueError: if ``records`` is empty.
    """
    _require_records(records, "family_pref_delta")
    same: list[float] = []
    diff: list[float] = []
    for r in records:
        judged = r.metadata.get("judged_family")
        own = r.metadata.get("judge_family")
        if judged is None or own is None:
            continue
        acc = 1.0 if _is_correct(r) else 0.0
        (same if judged == own else diff).append(acc)

    if not same or not diff:
        return 0.0
    return float(sum(same) / len(same) - sum(diff) / len(diff))


def quality_score(
    *,
    accuracy: float,
    agreement_r: float,
    consistency: float,
    ece: float | None,
    max_bias: float,
    ece_soft_ceiling: float = 0.15,
    w_accuracy: float = 0.5,
    w_agreement: float = 0.3,
    w_consistency: float = 0.2,
) -> float:
    """Calibrated 0–1 judge-quality score — §12 Q4 (replaces the brittle 7-threshold AND).

    §12 retires the multi-threshold binary gate in favor of a single
    **difficulty-normalized, calibrated, continuous** judge-quality score, scored across
    the operating curve rather than at one cut (Traub 2024, "selective classification",
    arXiv:2407.01032 — a single fixed threshold can violate monotonicity). This is the
    pure combiner; the profile layer feeds it ``accuracy`` =
    :func:`difficulty_weighted_accuracy` (the §12 difficulty normalization) and the other
    measured components, and pushes an *accuracy confidence interval* through it (holding
    the deterministic factors fixed) to get a score CI for the selective decision.

    Form — a convex blend of the three "good" components, then multiplicative penalties:

        base    = w_acc·acc + w_agr·max(0, r) + w_cons·consistency        ∈ [0, 1]
        p_ece   = 1 / (1 + max(0, ece − ece_soft_ceiling))   (1.0 if ece is None)
        p_bias  = max(0, 1 − max_bias)
        score   = base · p_ece · p_bias                                   ∈ [0, 1]

    The blend weights sum to 1 so ``base`` stays in ``[0, 1]``; the penalties are in
    ``(0, 1]``, so the product is bounded in ``[0, 1]`` and is **monotone**: strictly
    non-decreasing in accuracy / agreement / consistency and non-increasing in ECE and
    bias. Monotonicity is the property the selective-CI decision relies on (a higher
    accuracy bound ⇒ a higher score bound). ECE is a *soft* signal (it only bites past
    the soft ceiling and never zeroes the score), matching §11.1; ``None`` ECE
    ("not measured", §12) applies no calibration penalty.

    Args:
        accuracy: difficulty-weighted accuracy in ``[0, 1]`` (§12; pass
            :func:`difficulty_weighted_accuracy`).
        agreement_r: Pearson r vs gold/human in ``[-1, 1]`` (negative ⇒ contributes 0).
        consistency: test-retest stability in ``[0, 1]`` (1 − flip-rate).
        ece: expected calibration error in ``[0, 1]`` or ``None`` (not measured).
        max_bias: the largest bias magnitude (position / verbosity / |family-pref|) in
            ``[0, 1]``.
        ece_soft_ceiling: ECE below this is "calibrated" and applies no penalty
            (default 0.15, the §11.1 soft ceiling).
        w_accuracy, w_agreement, w_consistency: blend weights (should sum to 1.0;
            defaults 0.5 / 0.3 / 0.2 lead with difficulty-weighted accuracy).

    Returns:
        The continuous quality score in ``[0.0, 1.0]``.
    """
    r_unit = max(0.0, agreement_r)
    base = w_accuracy * accuracy + w_agreement * r_unit + w_consistency * consistency
    base = max(0.0, min(1.0, base))
    p_ece = 1.0 if ece is None else 1.0 / (1.0 + max(0.0, ece - ece_soft_ceiling))
    p_bias = max(0.0, 1.0 - max_bias)
    return float(max(0.0, min(1.0, base * p_ece * p_bias)))


# Re-export the standard-normal CDF helper used by the profile layer's z-tests so it
# imports a single statistics surface from here (keeps scipy usage centralized).
def _two_sided_p_from_z(z: float) -> float:
    """Two-sided p-value for a z-score (used by the profile layer for legible notes)."""
    return float(2.0 * (1.0 - norm.cdf(abs(z))))
