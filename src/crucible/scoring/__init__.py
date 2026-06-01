"""Crucible scoring layer — oracle gate, judge panel, small-N statistics.

The scoring domain owns three concerns (research-grounding §10.6 "Scoring"):

* :mod:`crucible.scoring.stats` — small-N admissible statistics: ``pass_hat_k``
  (consistency, τ-bench §1), ``wilson_interval`` / ``clopper_pearson`` (CIs that
  hold below ~300 items, §1), ``mcnemar_exact`` (primary paired test, §9.3), and
  ``graduates`` (the §1 Lab→Arena graduation rule).
* :mod:`crucible.scoring.oracle` — the §8.3 conjunctive hard gate
  (solved-AND-no-regression-AND-quality-AND-no-critical-penalty-AND-in-budget-AND
  -novelty-validated) with a tiebreaker net score.
* :mod:`crucible.scoring.judge_panel` — the cross-family PoLL panel with
  EXTERNAL_VERIFIER generator-family exclusion (§10.2).

All public types cross module boundaries via :mod:`crucible.types`; this layer
implements against them and never redefines them.
"""

from __future__ import annotations

from crucible.scoring.judge_panel import JudgePanel, judge_family, reduce_scores
from crucible.scoring.oracle import CRITICAL_FLAVOR, OracleOutcome, grade
from crucible.scoring.stats import (
    clopper_pearson,
    graduates,
    mcnemar_exact,
    pass_hat_k,
    wilson_interval,
)

__all__ = [
    # stats
    "pass_hat_k",
    "wilson_interval",
    "clopper_pearson",
    "mcnemar_exact",
    "graduates",
    # oracle
    "OracleOutcome",
    "grade",
    "CRITICAL_FLAVOR",
    # judge panel
    "JudgePanel",
    "reduce_scores",
    "judge_family",
]
