"""Observability rollups (``observability`` module, research-grounding §1, swarm-17 finding 6).

The stats-handoff surface: it rolls individual attempts up into per-puzzle
histories and per-model profiles, and computes the reliability metric the
literature settles on — **pass^k, not pass@k** (τ-bench, Yao et al. 2024,
arXiv:2406.12045; §1). pass^k = "all k independent trials succeeded," which
decays exponentially and exposes the consistency gap pass@k hides (GPT-4o was
<50% pass@1 but <25% pass^8 in retail).

Per finding 6, pass^k is recorded **natively**: k sibling attempts under one
puzzle-history record, not k samples in one log. :class:`PuzzleHistory`
collects the siblings; :class:`ModelProfile` rolls a model's attempts across
puzzles.

**Wilson intervals at small N.** Below ~300 datapoints the CLT-based normal
interval underestimates uncertainty (Bowyer, Aitchison & Ivanova 2025,
arXiv:2503.01747, ICML 2025 Spotlight); the Wilson score interval is the
practical small-N choice (Agresti & Coull 1998). A minimal Wilson is
implemented **inline** here (:func:`wilson_interval`) so this module is
self-contained for the Wave-1 parallel build — it deliberately does **not**
import :mod:`ai_crucible.scoring` at module load, avoiding a cross-agent
dependency. The kernel (Wave 2) may later inject the scoring module's richer
interval via the ``interval`` parameter; the signature accepts an injected
callable so the wiring is non-breaking.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

from ai_crucible.types import AttemptState, TerminatedBy

__all__ = [
    "ModelProfile",
    "PuzzleHistory",
    "WilsonInterval",
    "aggregate_pass_hat_k",
    "roll_up",
    "wilson_interval",
]

# A 95% interval by default (z for two-sided 95%). Kept as a module constant so
# callers and the inline Wilson agree on the confidence level.
_Z_95 = 1.959963984540054


@dataclass(frozen=True, slots=True)
class WilsonInterval:
    """A binomial proportion interval: point estimate + [lower, upper] bounds."""

    estimate: float
    lower: float
    upper: float

    @property
    def width(self) -> float:
        return self.upper - self.lower


def wilson_interval(successes: int, n: int, *, z: float = _Z_95) -> WilsonInterval:
    """Wilson score interval for ``successes`` out of ``n`` trials.

    The small-N-correct binomial interval (Agresti & Coull 1998; preferred over
    the CLT normal interval per Bowyer et al. 2025). Reimplemented inline rather
    than imported from :mod:`ai_crucible.scoring` to keep the Wave-1 build free of
    cross-agent imports (see module docstring).

    For ``n == 0`` the proportion is undefined; we return the maximally
    uninformative ``estimate=0.0, lower=0.0, upper=1.0`` rather than dividing by
    zero, so downstream rollups never raise on an empty puzzle.
    """
    if n < 0 or successes < 0 or successes > n:
        raise ValueError(f"invalid Wilson inputs: successes={successes}, n={n}")
    if n == 0:
        return WilsonInterval(estimate=0.0, lower=0.0, upper=1.0)

    p_hat = successes / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p_hat + z2 / (2 * n)) / denom
    margin = (z / denom) * math.sqrt(p_hat * (1 - p_hat) / n + z2 / (4 * n * n))
    lower = max(0.0, center - margin)
    upper = min(1.0, center + margin)
    return WilsonInterval(estimate=p_hat, lower=lower, upper=upper)


def aggregate_pass_hat_k(outcomes: Sequence[bool], k: int) -> float:
    """Empirical pass^k over a sequence of per-attempt boolean ``outcomes``.

    pass^k is the probability that **all k** independent trials succeed
    (τ-bench, Yao 2024). With ``n`` observed outcomes and an empirical per-trial
    success rate ``p = successes / n``, the standard plug-in estimator is
    ``p ** k`` (Chen et al. 2021 introduced the pass@k estimator family; pass^k
    is its all-must-succeed sibling). Special cases:

    - empty ``outcomes`` -> ``0.0`` (no evidence of success).

    The consistency horizon is ``k >= 1`` — you cannot be *consistent* over
    fewer than one trial. We REJECT ``k < 1`` with a ``ValueError`` to share the
    one boundary contract :func:`ai_crucible.scoring.stats.pass_hat_k` already
    enforces (and that ``run_pass_hat_k`` assumes via ``k >= 1``); the two
    implementations of this same named metric must not diverge at the boundary
    (scoring-stats-004 — the Wave-2 unification injects one ``interval`` impl, so
    a silent ``k<=0 -> 1.0`` here vs a raise there would be a latent trap).

    Args:
        outcomes: per-attempt success flags (e.g. one bool per sibling attempt).
        k: the consistency depth; how many independent successes are required
            (``k >= 1``).

    Returns:
        pass^k as a float in [0, 1].

    Raises:
        ValueError: if ``k < 1`` (a consistency horizon below one trial is
            undefined — matches :func:`ai_crucible.scoring.stats.pass_hat_k`).
    """
    if k < 1:
        raise ValueError(f"pass^k consistency horizon must be >= 1, got k={k}")
    n = len(outcomes)
    if n == 0:
        return 0.0
    successes = sum(1 for o in outcomes if o)
    p = successes / n
    return p**k


def _attempt_solved(attempt: AttemptState, *, score_key: str = "oracle") -> bool:
    """Decide whether an attempt counts as a solve.

    A solve requires a clean terminal status (``COMPLETED``) AND a passing oracle
    verdict. A budget/time/hard-kill/error termination is *not* a solve even if a
    partial score leaked in — pass^k must reflect end-to-end success
    (terminal-world-state grading, τ-bench / finding 6).

    **The authoritative solve signal is the §8.3 conjunctive *gate*, not the
    tiebreaker net** (scoring-stats-001). :func:`ai_crucible.scoring.oracle.grade`
    records the gate verdict in ``score.metadata['gate_passed']`` and sets
    ``score.value`` to the net only as a within-passing-region tiebreaker (it can
    be ≤ 0 on a passing gate when a non-critical REGRESSIONAL penalty outweighs
    solve+bonuses, since penalties are unbounded relative to solve). Reading
    ``value > 0`` would mis-derive such a clean solve as UNSOLVED and silently
    corrupt pass^k / solve-rate / graduation rollups — the grading authority and
    the reported reliability metric must not disagree. So we prefer the explicit
    boolean ``gate_passed`` flag when the score carries the oracle gate envelope,
    and only fall back to the tolerant truthy interpretation for scores that lack
    it (e.g. a bool ``"C"``/``True`` Inspect-style score).
    """
    if attempt.terminated_by not in (None, TerminatedBy.COMPLETED):
        return False
    score = attempt.scores.get(score_key)
    if score is None:
        return False
    gate_passed = score.metadata.get("gate_passed")
    if isinstance(gate_passed, bool):
        return gate_passed
    return _score_is_truthy(score.value)


def _score_is_truthy(value: Any) -> bool:
    """Interpret a heterogeneous Score.value as solved/unsolved.

    bool -> itself; numbers -> > 0; "C"/"correct"/"pass"/"true" -> True; other
    strings -> False. Mirrors Inspect's tolerant CORRECT/INCORRECT handling
    without importing it.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value > 0
    if isinstance(value, str):
        return value.strip().lower() in {"c", "correct", "pass", "passed", "true", "1"}
    return False


def _is_novel(attempt: AttemptState, *, score_key: str = "oracle") -> bool:
    """Whether the (validated) novelty bonus applies to this attempt.

    Reads the structured ``metadata`` on the oracle/panel score — novelty is a
    panel-validated flag (§8.7), never Solver-self-asserted, so we look for a
    kernel/panel-set ``novelty_validated`` truthy marker rather than trusting a
    raw ``novelty: true`` claim.
    """
    for key in (score_key, "panel"):
        score = attempt.scores.get(key)
        if score is not None and bool(score.metadata.get("novelty_validated")):
            return True
    return False


@dataclass
class PuzzleHistory:
    """Sibling attempts for a single puzzle — the pass^k unit (finding 6).

    Collects the k i.i.d. attempts run against one ``puzzle_id`` and exposes the
    reliability views the catalog graduation rule needs (§1): pass^k and a
    Wilson interval on the per-attempt solve rate.
    """

    puzzle_id: str
    attempts: list[AttemptState] = field(default_factory=list)
    score_key: str = "oracle"

    def add(self, attempt: AttemptState) -> None:
        """Append a sibling attempt (must share this ``puzzle_id``)."""
        if attempt.puzzle_id != self.puzzle_id:
            raise ValueError(
                f"attempt puzzle_id {attempt.puzzle_id!r} != history {self.puzzle_id!r}"
            )
        self.attempts.append(attempt)

    @property
    def outcomes(self) -> list[bool]:
        """Per-attempt solve flags, in attempt order."""
        return [_attempt_solved(a, score_key=self.score_key) for a in self.attempts]

    @property
    def n_attempts(self) -> int:
        return len(self.attempts)

    @property
    def n_solved(self) -> int:
        return sum(self.outcomes)

    def solve_rate(self) -> float:
        """Per-attempt empirical solve rate (pass@1 estimate)."""
        if not self.attempts:
            return 0.0
        return self.n_solved / self.n_attempts

    def pass_hat_k(self, k: int) -> float:
        """Empirical pass^k over the collected siblings (finding 6)."""
        return aggregate_pass_hat_k(self.outcomes, k)

    def wilson(
        self,
        *,
        interval: Callable[[int, int], WilsonInterval] | None = None,
    ) -> WilsonInterval:
        """Wilson 95% interval on the per-attempt solve rate.

        Uses the inline :func:`wilson_interval` by default. Wave 2 may inject the
        scoring module's interval via ``interval`` (a callable
        ``(successes, n) -> WilsonInterval``) to share one implementation without
        a Wave-1 import.
        """
        compute = interval or (lambda s, n: wilson_interval(s, n))
        return compute(self.n_solved, self.n_attempts)

    def is_graduation_candidate(self) -> bool:
        """The §1 Lab-graduation rule: ``0.10 <= Wilson-lower`` and
        ``Wilson-upper <= 0.90`` — rules out trivial and impossible in one test.
        """
        ci = self.wilson()
        return ci.lower >= 0.10 and ci.upper <= 0.90


@dataclass
class ModelProfile:
    """Per-model rollup across many puzzles (the leaderboard row, §8.7).

    Exposes solve_rate, novelty_rate, mean_latency, and n_attempts — the
    leaderboard surfaces novelty-rate alongside solve-rate because a model that
    consistently finds legitimate novel paths is demonstrating the capability
    ai_crucible most wants to measure (§8.7).
    """

    model: str
    n_attempts: int = 0
    n_solved: int = 0
    n_novel: int = 0
    total_latency: float = 0.0
    score_key: str = "oracle"

    def add(self, attempt: AttemptState) -> None:
        """Fold one attempt for this model into the running profile."""
        if attempt.model != self.model:
            raise ValueError(
                f"attempt model {attempt.model!r} != profile {self.model!r}"
            )
        self.n_attempts += 1
        if _attempt_solved(attempt, score_key=self.score_key):
            self.n_solved += 1
        if _is_novel(attempt, score_key=self.score_key):
            self.n_novel += 1
        self.total_latency += attempt.wall_time

    @property
    def solve_rate(self) -> float:
        return self.n_solved / self.n_attempts if self.n_attempts else 0.0

    @property
    def novelty_rate(self) -> float:
        return self.n_novel / self.n_attempts if self.n_attempts else 0.0

    @property
    def mean_latency(self) -> float:
        return self.total_latency / self.n_attempts if self.n_attempts else 0.0

    def to_dict(self) -> dict[str, Any]:
        """Render the profile as a JSON-able leaderboard row."""
        return {
            "model": self.model,
            "n_attempts": self.n_attempts,
            "solve_rate": self.solve_rate,
            "novelty_rate": self.novelty_rate,
            "mean_latency": self.mean_latency,
        }


def roll_up(attempts: Iterable[AttemptState], *, score_key: str = "oracle") -> dict[str, Any]:
    """Roll a flat list of attempts up into per-puzzle and per-model views.

    The top-level stats-handoff surface (swarm-17 ``observability`` module): an
    eval batch produces many attempts spanning several puzzles and models; this
    folds them into the two rollups the catalog and the leaderboard consume.

    Returns a dict::

        {
          "n_attempts": int,
          "puzzles": { puzzle_id: {n_attempts, n_solved, solve_rate,
                                   pass_hat_3, wilson_lower, wilson_upper} },
          "models":  { model: {n_attempts, solve_rate, novelty_rate,
                               mean_latency} },
        }

    ``pass_hat_3`` is included as the default reliability headline (k=3 gives
    clean separation given the 71% identical-failure rate, §1); callers needing
    other k can use :meth:`PuzzleHistory.pass_hat_k` directly.
    """
    attempts = list(attempts)
    histories: dict[str, PuzzleHistory] = {}
    profiles: dict[str, ModelProfile] = {}

    for attempt in attempts:
        hist = histories.get(attempt.puzzle_id)
        if hist is None:
            hist = PuzzleHistory(puzzle_id=attempt.puzzle_id, score_key=score_key)
            histories[attempt.puzzle_id] = hist
        hist.add(attempt)

        prof = profiles.get(attempt.model)
        if prof is None:
            prof = ModelProfile(model=attempt.model, score_key=score_key)
            profiles[attempt.model] = prof
        prof.add(attempt)

    puzzles_out: dict[str, Any] = {}
    for pid, hist in histories.items():
        ci = hist.wilson()
        puzzles_out[pid] = {
            "n_attempts": hist.n_attempts,
            "n_solved": hist.n_solved,
            "solve_rate": hist.solve_rate(),
            "pass_hat_3": hist.pass_hat_k(3),
            "wilson_lower": ci.lower,
            "wilson_upper": ci.upper,
        }

    models_out = {model: prof.to_dict() for model, prof in profiles.items()}

    return {
        "n_attempts": len(attempts),
        "puzzles": puzzles_out,
        "models": models_out,
    }
