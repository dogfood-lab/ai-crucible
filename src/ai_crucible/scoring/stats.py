"""Small-N statistics for the AI Crucible scoring layer.

AI Crucible grades a handful of attempts per puzzle, never hundreds, so the
classical CLT-based normal-approximation interval is inadmissible here
(Bowyer, Aitchison & Ivanova 2025, "Don't Use the CLT in LLM Evals With Fewer
Than a Few Hundred Datapoints", arXiv:2503.01747, ICML 2025 Spotlight —
research-grounding §1). This module ships the interval estimators that *are*
admissible at small N (Wilson score, Clopper-Pearson exact), the primary paired
significance test (McNemar exact, research-grounding §9.3), the reliability
statistic pass^k (τ-bench, Yao et al. 2024, arXiv:2406.12045 — §1), and the §1
graduation rule that rules out trivial *and* impossible puzzles in one test.

All functions are pure and deterministic: same inputs grade identically tomorrow
(research-grounding §1, "Live external dependencies are forbidden").
"""

from __future__ import annotations

import math

from scipy.stats import beta, binomtest, fisher_exact, norm

__all__ = [
    "pass_hat_k",
    "wilson_interval",
    "clopper_pearson",
    "mcnemar_exact",
    "graduates",
    "conformal_coverage_interval",
    "newcombe_wilson_diff",
    "fisher_difference_pvalue",
    "benjamini_hochberg",
    "bernoulli_log_eprocess",
]


def _validate_counts(successes: int, n: int) -> None:
    """Shared guard for the binomial estimators.

    Raises ``ValueError`` (not a raw assertion) so the failure carries a stable,
    legible message — the structured-error discipline the kernel relies on.
    """
    if n <= 0:
        raise ValueError(f"n must be a positive integer, got {n}")
    if successes < 0 or successes > n:
        raise ValueError(f"successes must be in [0, n]={n}, got {successes}")


def pass_hat_k(successes: int, n: int, k: int) -> float:
    """pass^k — the probability that *all* k i.i.d. attempts succeed.

    pass^k (Yao et al. 2024, τ-bench, arXiv:2406.12045 — research-grounding §1)
    measures *consistency*, not best-of-k: it is the chance that k independent
    trials at the observed success rate would all pass, which decays
    exponentially and exposes the unreliability pass@k hides (GPT-4o is <50%
    pass@1 but <25% pass^8 in τ-bench retail).

    **Estimator.** We use the plug-in estimate ``p_hat ** k`` where
    ``p_hat = successes / n`` is the empirical single-attempt success rate. This
    is the simple consistency projection from an observed rate (the §1 "pass^k
    from an empirical rate p=successes/n is p**k" estimator); it assumes the k
    trials are i.i.d. at ``p_hat``. It is *not* the unbiased U-statistic estimator
    pass@k uses for sampling-without-replacement — pass^k asks a different
    question (consistency of independent draws), so the i.i.d. projection is the
    intended quantity here.

    Args:
        successes: number of observed successes (0 <= successes <= n).
        n: number of observed attempts (n > 0).
        k: the consistency horizon (k >= 1) — how many independent successes in
            a row we are estimating the probability of.

    Returns:
        ``(successes / n) ** k`` in ``[0.0, 1.0]``.

    Raises:
        ValueError: if counts are out of range or ``k < 1``.
    """
    _validate_counts(successes, n)
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    p_hat = successes / n
    return p_hat**k


def wilson_interval(successes: int, n: int, conf: float = 0.95) -> tuple[float, float]:
    """Wilson score confidence interval for a binomial proportion.

    The Wilson score interval (Wilson 1927) is the small-N-admissible interval
    ai_crucible's graduation rule is built on (research-grounding §1; Agresti &
    Coull 1998 for the practical recommendation over the raw normal interval).
    Unlike the Wald/normal-approximation interval it never escapes ``[0, 1]`` and
    stays sane at the 0/n and n/n boundaries.

    Args:
        successes: observed successes (0 <= successes <= n).
        n: observed attempts (n > 0).
        conf: confidence level in ``(0, 1)``; default 0.95.

    Returns:
        ``(lower, upper)`` clamped to ``[0.0, 1.0]``.

    Raises:
        ValueError: if counts are out of range or ``conf`` is not in ``(0, 1)``.
    """
    _validate_counts(successes, n)
    if not 0.0 < conf < 1.0:
        raise ValueError(f"conf must be in (0, 1), got {conf}")

    # Two-sided z for the requested confidence.
    z = norm.ppf(1.0 - (1.0 - conf) / 2.0)
    p_hat = successes / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p_hat + z2 / (2.0 * n)) / denom
    half = (z / denom) * (((p_hat * (1.0 - p_hat)) / n + z2 / (4.0 * n * n)) ** 0.5)
    lower = center - half
    upper = center + half
    # Coerce numpy scalars (z from norm.ppf is numpy) to built-in float per the
    # ``tuple[float, float]`` contract.
    return (float(max(0.0, lower)), float(min(1.0, upper)))


def clopper_pearson(successes: int, n: int, conf: float = 0.95) -> tuple[float, float]:
    """Clopper-Pearson exact confidence interval for a binomial proportion.

    The conservative "exact" interval (Clopper & Pearson 1934 — research-grounding
    §1) inverts the binomial CDF via the beta distribution. It guarantees at-least
    nominal coverage (wider than Wilson) and is the conservative bound ai_crucible
    reports when it must not understate uncertainty.

    Boundary handling: at ``successes == 0`` the lower bound is exactly 0; at
    ``successes == n`` the upper bound is exactly 1 (the corresponding beta
    quantile is undefined there).

    Args:
        successes: observed successes (0 <= successes <= n).
        n: observed attempts (n > 0).
        conf: confidence level in ``(0, 1)``; default 0.95.

    Returns:
        ``(lower, upper)`` in ``[0.0, 1.0]``.

    Raises:
        ValueError: if counts are out of range or ``conf`` is not in ``(0, 1)``.
    """
    _validate_counts(successes, n)
    if not 0.0 < conf < 1.0:
        raise ValueError(f"conf must be in (0, 1), got {conf}")

    alpha = 1.0 - conf
    lower = 0.0 if successes == 0 else beta.ppf(alpha / 2.0, successes, n - successes + 1)
    upper = 1.0 if successes == n else beta.ppf(1.0 - alpha / 2.0, successes + 1, n - successes)
    return (float(lower), float(upper))


def mcnemar_exact(b: int, c: int) -> float:
    """Exact McNemar paired two-sided p-value for two systems on the same items.

    McNemar's exact test is ai_crucible's *primary* significance test for comparing
    two models/configs on the same puzzle set (paired binary outcomes — Dror et
    al. 2018, ACL P18-1128; research-grounding §9.3). Only the discordant pairs
    carry information:

    * ``b`` — items the first system got right and the second wrong.
    * ``c`` — items the first system got wrong and the second right.

    The exact test conditions on ``b + c`` discordant pairs and asks whether the
    split is consistent with a fair coin: it is the two-sided exact binomial test
    of ``b`` successes in ``b + c`` trials at ``p = 0.5``. When there are no
    discordant pairs there is no evidence of a difference, so ``p = 1.0``.

    Args:
        b: count of (first-correct, second-wrong) discordant pairs (b >= 0).
        c: count of (first-wrong, second-correct) discordant pairs (c >= 0).

    Returns:
        Two-sided exact p-value in ``[0.0, 1.0]``.

    Raises:
        ValueError: if ``b`` or ``c`` is negative.
    """
    if b < 0 or c < 0:
        raise ValueError(f"discordant counts must be non-negative, got b={b}, c={c}")
    n_discordant = b + c
    if n_discordant == 0:
        return 1.0
    return float(binomtest(b, n_discordant, 0.5, alternative="two-sided").pvalue)


def conformal_coverage_interval(
    n: int, alpha: float, conf: float = 0.95
) -> tuple[float, float, float]:
    """Realized-coverage interval for split conformal from ONE calibration set of size n.

    The coverage actually realized from a single small calibration set is a RANDOM
    variable, not the nominal ``1 − alpha``: it is Beta(n+1−l, l)-distributed with
    ``l = floor((n+1)·alpha)`` (Vovk 2012, arXiv:1209.2673; Angelopoulos & Bates 2021
    §3.2, arXiv:2107.07511), and that distribution is UNIVERSAL — determined solely by
    ``(alpha, n)``, independent of the data (Marques F. 2023, arXiv:2303.02770). At
    N≈30–50 it is wide: a nominal 90% can realize ~80–97%. This is the honest small-N
    coverage statement ai_crucible attaches to a SEAT DECISION's false-seat rate (never to ω
    itself, which is already an aggregate) — a single-set point "90% coverage" claim at
    N<50 is statistically indefensible, so we report the interval the chosen N actually
    buys. The mean lies in ``[1−alpha, 1−alpha + 1/(n+1)]``.

    Pure + deterministic (PIN_PER_STEP), reusing the already-imported ``scipy.stats.beta``
    (no new dependency) — the same admissible-at-small-N discipline as Wilson/Clopper-
    Pearson above. The risk-controlled DERIVATION of the seat cut itself (conformal risk
    control + Learn-Then-Test; Angelopoulos 2022 arXiv:2208.02814 / 2021 arXiv:2110.01052)
    and the SSBC level inflation (Zwart 2025, arXiv:2509.15349) are a documented deferred
    slice (§12.1); this primitive supplies the coverage spread that statement quotes.

    Args:
        n: calibration-set size (number of human-labeled items; n > 0).
        alpha: target miscoverage (e.g. 0.10 for a 90% target); in ``(0, 1)``.
        conf: central probability mass of the reported interval over calibration draws
            (default 0.95).

    Returns:
        ``(lower, upper, mean)`` realized coverage — the central ``conf`` interval of the
        Beta(n+1−l, l) coverage distribution plus its mean. When ``alpha`` is too small
        for ``n`` to admit a finite-sample guarantee (``l < 1``), returns
        ``(1.0, 1.0, 1.0)`` (the degenerate full-coverage regime).

    Raises:
        ValueError: ``n <= 0``, ``alpha`` not in ``(0, 1)``, or ``conf`` not in ``(0, 1)``.
    """
    if n <= 0:
        raise ValueError(f"n must be a positive integer, got {n}")
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    if not 0.0 < conf < 1.0:
        raise ValueError(f"conf must be in (0, 1), got {conf}")
    # floor((n+1)·alpha), nudged so a product that is mathematically an integer but
    # represented just below it in float64 (e.g. 63.0 → 62.999…) floors correctly — an
    # off-by-one in the l-th order statistic otherwise (re-audit fork-c-stats LOW).
    ell = math.floor((n + 1) * alpha + 1e-9)
    if ell < 1:
        # alpha too small for this n: no l-th order statistic exists; coverage → 1.
        return (1.0, 1.0, 1.0)
    a = n + 1 - ell
    b = ell
    tail = (1.0 - conf) / 2.0
    lower = float(beta.ppf(tail, a, b))
    upper = float(beta.ppf(1.0 - tail, a, b))
    mean = float(a / (a + b))
    return (lower, upper, mean)


def graduates(successes: int, n: int) -> bool:
    """The §1 graduation rule — rules out trivial *and* impossible in one test.

    A Lab puzzle graduates to Arena only when its Wilson 95% interval shows it is
    neither trivially easy nor effectively impossible (research-grounding §1):

        ``0.10 <= wilson_lower``  AND  ``wilson_upper <= 0.90``

    The lower-bound clause kills *impossible* puzzles (a 0/20 puzzle has a low
    Wilson lower bound — no evidence anyone can clear 10%); the upper-bound clause
    kills *trivial* puzzles (a 20/20 puzzle has a high Wilson upper bound — no
    evidence it ever stumps anyone). A mid-rate puzzle (~5/20) satisfies both and
    graduates. The Wilson interval (not Wald) is load-bearing: at the 0/n and n/n
    boundaries the normal interval degenerates, but Wilson stays informative.

    Args:
        successes: observed successes for the puzzle (0 <= successes <= n).
        n: observed attempts (n > 0); §1 uses N=20 per puzzle for graduation.

    Returns:
        ``True`` iff the puzzle clears both Wilson bounds.

    Raises:
        ValueError: if counts are out of range (propagated from
            :func:`wilson_interval`).
    """
    lower, upper = wilson_interval(successes, n, conf=0.95)
    # Coerce to a built-in bool: the bounds come from numpy/scipy, so the naive
    # comparison would yield numpy.bool_, breaking the ``-> bool`` contract and
    # ``is True`` identity checks downstream.
    return bool(lower >= 0.10 and upper <= 0.90)


# --------------------------------------------------------------------------- #
# Epic-4 catalog primitives (the differential + saturation math)
# --------------------------------------------------------------------------- #


def newcombe_wilson_diff(
    s1: int, n1: int, s2: int, n2: int, conf: float = 0.95
) -> tuple[float, float, float]:
    """Newcombe-1998 hybrid-score CI for the difference of two INDEPENDENT proportions.

    The differential typology (research-grounding §4) asks whether Claude's solve
    rate differs from the cross-family cohort's on a puzzle. Claude and the cohort
    run INDEPENDENT attempt sets, so this is an UNPAIRED two-proportion comparison —
    :func:`mcnemar_exact` is the WRONG tool here (it is paired). The right small-N
    estimator is Newcombe's "method 10": build a CI for ``p1 − p2`` by COMBINING the
    two single-proportion Wilson intervals (Newcombe 1998, "Interval estimation for
    the difference between independent proportions: comparison of eleven methods",
    Statistics in Medicine 17(8):873-890,
    doi:10.1002/(SICI)1097-0258(19980430)17:8<873::AID-SIM779>3.0.CO;2-I — "performs
    well and is readily implemented irrespective of sample size").

    **Do NOT** decide a difference by checking whether two separate Wilson intervals
    overlap: the interval-overlap heuristic is over-conservative and fails to reject
    when it should (Schenker & Gentleman 2001, The American Statistician
    55(3):182-186, doi:10.1198/000313001317097960). Test the DIFFERENCE — this CI.

    With ``p1 = s1/n1``, ``p2 = s2/n2``, ``δ = p1 − p2`` and the two Wilson intervals
    ``(l1, u1)`` and ``(l2, u2)`` (at confidence ``conf``)::

        lower = δ − sqrt((p1 − l1)² + (u2 − p2)²)
        upper = δ + sqrt((u1 − p1)² + (p2 − l2)²)

    Args:
        s1, n1: successes / attempts for system 1 (Claude). ``0 <= s1 <= n1``, ``n1>0``.
        s2, n2: successes / attempts for system 2 (the cohort). ``0 <= s2 <= n2``, ``n2>0``.
        conf: confidence level in ``(0, 1)``; default 0.95.

    Returns:
        ``(lower, upper, delta)`` with ``lower``/``upper`` clamped to ``[-1, 1]``.

    Raises:
        ValueError: if either count pair is out of range or ``conf`` not in ``(0, 1)``.
    """
    _validate_counts(s1, n1)
    _validate_counts(s2, n2)
    if not 0.0 < conf < 1.0:
        raise ValueError(f"conf must be in (0, 1), got {conf}")

    p1 = s1 / n1
    p2 = s2 / n2
    delta = p1 - p2
    l1, u1 = wilson_interval(s1, n1, conf=conf)
    l2, u2 = wilson_interval(s2, n2, conf=conf)
    lower = delta - math.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2)
    upper = delta + math.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2)
    return (max(-1.0, lower), min(1.0, upper), delta)


def fisher_difference_pvalue(s1: int, n1: int, s2: int, n2: int) -> float:
    """Two-sided Fisher exact p-value for H0: the two proportions are equal.

    The per-puzzle significance handle the catalog-level BH-FDR pass consumes
    (:func:`benjamini_hochberg`). Fisher's EXACT test on the 2×2 contingency table
    ``[[s1, n1−s1], [s2, n2−s2]]`` is the small-N-admissible choice (matching the
    exact-at-small-N discipline of :func:`clopper_pearson` / :func:`mcnemar_exact`;
    a normal-approximation z-test is inadmissible at N≈20, research-grounding §1).

    Args:
        s1, n1: successes / attempts for system 1. ``0 <= s1 <= n1``, ``n1 > 0``.
        s2, n2: successes / attempts for system 2. ``0 <= s2 <= n2``, ``n2 > 0``.

    Returns:
        Two-sided exact p-value in ``[0.0, 1.0]``.

    Raises:
        ValueError: if either count pair is out of range.
    """
    _validate_counts(s1, n1)
    _validate_counts(s2, n2)
    table = [[s1, n1 - s1], [s2, n2 - s2]]
    return float(fisher_exact(table, alternative="two-sided")[1])


def benjamini_hochberg(pvalues: list[float], q: float = 0.10) -> list[bool]:
    """Benjamini-Hochberg FDR control — which hypotheses survive at level ``q``.

    Across many puzzles the differential runs many comparisons, so a per-puzzle
    "significant" call needs multiple-comparison control or the catalog fills with
    false typology labels. BH controls the false-discovery rate at ``q`` for
    independent (or PRDS) tests (Benjamini & Hochberg 1995, JRSS-B 57(1):289-300).
    The catalog classifier downgrades non-survivors to INCONCLUSIVE_UNDERPOWERED.

    (For dependent tests the more conservative Benjamini-Yekutieli is the right
    choice — it already ships in ``characterize.metrics`` for the alt-test; across
    largely-independent puzzles BH is the appropriate, less conservative control.)

    Args:
        pvalues: the per-hypothesis p-values (each in ``[0, 1]``). Order preserved.
        q: target FDR in ``(0, 1)``; default 0.10.

    Returns:
        A list of bools, ``survived[i]`` True iff hypothesis ``i`` is rejected
        (a discovery) under BH at level ``q``. An empty input returns ``[]``.

    Raises:
        ValueError: if ``q`` not in ``(0, 1)`` or any p-value is out of ``[0, 1]``.
    """
    if not 0.0 < q < 1.0:
        raise ValueError(f"q must be in (0, 1), got {q}")
    m = len(pvalues)
    if m == 0:
        return []
    for p in pvalues:
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"p-values must be in [0, 1], got {p}")
    # Rank ascending; the BH threshold is the largest k with p_(k) <= (k/m)·q.
    order = sorted(range(m), key=lambda i: pvalues[i])
    max_k = 0
    for rank, idx in enumerate(order, start=1):
        if pvalues[idx] <= (rank / m) * q:
            max_k = rank
    survived = [False] * m
    # Step-up: reject all hypotheses with rank <= max_k (i.e. the max_k smallest p's).
    for rank, idx in enumerate(order, start=1):
        if rank <= max_k:
            survived[idx] = True
    return survived


def bernoulli_log_eprocess(successes: int, n: int, p0: float, p1: float) -> float:
    """Log e-value increment for ONE run, testing H0: p ≤ ``p0`` vs the point H1: p = ``p1``.

    The anytime-valid building block of the saturation rule (CONTRACT §B). A
    catalog that peeks after every run cannot act on a FIXED-N Wilson interval at a
    data-dependent stopping time without inflating the false-demotion rate; an
    e-process is valid at EVERY sample count (Howard, Ramdas, McAuliffe & Sekhon,
    "Time-uniform Chernoff bounds via nonnegative supermartingales", Annals of
    Statistics 2021, 49(2):1055-1080, doi:10.1214/20-AOS1991; the e-value framing of
    Ramdas/Grünwald/Vovk/Shafer).

    The per-run e-value is the likelihood ratio at the H0 boundary::

        E_run = (p1/p0)^s · ((1−p1)/(1−p0))^(n−s)

    which has expectation ≤ 1 under any ``p ≤ p0`` (it is a test supermartingale),
    so the PRODUCT of per-run e-values across runs is a valid e-process for the
    composite H0: p ≤ p0. This returns its NATURAL LOG (summed across runs by the
    caller, avoiding overflow); demote when the accumulated log-e crosses
    ``ln(1/alpha)``. With ``p1 > p0`` the statistic accumulates evidence that p is
    HIGH (saturation); the re-promotion direction reuses this on the failure counts
    (``successes → n−successes``, ``p0 → 1−p0``).

    Args:
        successes: successes in this run (0 <= successes <= n).
        n: attempts in this run (n > 0).
        p0: the H0-boundary rate, in ``(0, 1)``.
        p1: the H1 point rate, in ``(0, 1)`` (``!= p0``; ``> p0`` for the saturation
            direction).

    Returns:
        The natural-log e-value increment for this run (a float; may be negative).

    Raises:
        ValueError: if counts are out of range or ``p0``/``p1`` not in ``(0, 1)``.
    """
    _validate_counts(successes, n)
    if not 0.0 < p0 < 1.0:
        raise ValueError(f"p0 must be in (0, 1), got {p0}")
    if not 0.0 < p1 < 1.0:
        raise ValueError(f"p1 must be in (0, 1), got {p1}")
    failures = n - successes
    return successes * math.log(p1 / p0) + failures * math.log((1.0 - p1) / (1.0 - p0))
