"""Cross-family judge panel — the EXTERNAL_VERIFIER mitigation (§10.2, §3).

The single strongest mitigation against same-model self-preference is a panel of
judges drawn from *disjoint model families* (Verga et al. 2024, "Replacing Judges
with Juries / PoLL", arXiv:2404.18796 — research-grounding §3): a 3-model
cross-family panel correlates with humans better than a single GPT-4 judge at 1/7
the cost. The bias being defended against is mechanistic — a model over-rates its
own low-perplexity text (Wataoka et al. 2024, arXiv:2410.21819 — §1) — so
"different prompt" is not enough; the judge must come from a different
*distribution*. Crucible enforces this **structurally**: any judge whose family
equals the generator's family is excluded from the panel before aggregation
(EXTERNAL_VERIFIER, workflow-standard 6 + research-grounding §10.2,§10.7).

In Phase 2 the judges are real cross-family local models (Qwen + Mistral +
Command-R, via ollama-intern + RTX 5090 — §8.6). Here, in Phase 1, judges are
**injected async callables** so the panel logic is unit-testable without a model
runtime. Each judge is a ``Callable[[AttemptState], Awaitable[Score]]``; its model
family is read from a ``family`` attribute on the callable (set by the kernel when
it wires a model into a judge), falling back to ``None`` (treated as "unknown
family", never excluded — an untagged judge is assumed external).

Aggregation is a reducer (PoLL pattern): ``"majority"`` vote for boolean/discrete
verdicts, ``"median"`` for numeric scores (median is the robust central estimate
that resists a single outlier judge — the reason a panel beats one judge).

**Novelty adjudication (§8.7).** The panel is *the authority* on whether a Solver's
claimed novel path is a legitimate alternative or a disguised bypass — novelty
validation needs distributional perspective from outside the Solver's family (PoLL
applied to a different question). A judge expresses its per-judge novelty verdict as
``novelty_validated: bool`` in its returned :class:`~crucible.types.Score.metadata`;
:func:`reduce_scores` aggregates those votes (majority of *cast* votes) into a panel
``novelty_validated`` verdict on the panel score. The kernel feeds THAT verdict into
the oracle gate — the oracle_runner's self-reported ``novelty_validated`` is never
trusted for the gate. Phase 1 uses *injected* judges (so the wiring is testable
without a model runtime); real cross-family panels (Qwen + Mistral + Command-R via
ollama-intern + RTX 5090) plug into the same shape at Phase 2 with no kernel change.
"""

from __future__ import annotations

import asyncio
import statistics
from collections import Counter
from collections.abc import Awaitable, Callable

from crucible.types import AttemptState, Score

__all__ = ["JudgePanel", "reduce_scores", "judge_family"]

#: Type alias for an injected judge (Phase-2 wires real cross-family models here).
JudgeFn = Callable[[AttemptState], Awaitable[Score]]


def judge_family(judge: JudgeFn) -> str | None:
    """Return the model family a judge belongs to, or ``None`` if untagged.

    The kernel tags a judge callable with a ``family`` attribute when it binds a
    concrete model to it (e.g. ``"qwen"``, ``"mistral"``, ``"claude"``). An
    untagged judge returns ``None`` and is treated as an unknown — and therefore
    *external* — family, so it is never excluded. This keeps the exclusion rule
    conservative: we only drop a judge we can *prove* shares the generator's
    family.
    """
    return getattr(judge, "family", None)


def _aggregate_novelty(scores: list[Score]) -> dict[str, object]:
    """Aggregate per-judge ``novelty_validated`` votes into a panel verdict (§8.7).

    Each judge MAY carry ``novelty_validated: bool`` in its ``Score.metadata`` (its
    vote on whether the Solver's claimed novel path is a legitimate alternative or a
    disguised bypass — §8.7). Judges that omit the key abstain and cast no vote.

    Returns a dict with:

    * ``novelty_votes`` — the list of cast votes (in judge order), for auditability.
    * ``novelty_validated`` — the panel verdict: ``True`` iff a strict majority of
      *cast* votes are ``True``. With no cast votes it is ``False`` — a missing
      adjudication is never a validation (the gate must not open on silence, §8.3).

    Why majority of *cast* votes rather than of the whole panel: an abstaining judge
    expresses no opinion; counting it as a "no" would let one silent panelist veto a
    unanimous pair. Majority-of-voters matches the PoLL spirit (the panel decides
    among those who ruled) while still failing closed when nobody ruled.
    """
    votes = [
        bool(s.metadata["novelty_validated"])
        for s in scores
        if "novelty_validated" in s.metadata
    ]
    validated = bool(votes) and sum(votes) * 2 > len(votes)
    return {"novelty_votes": votes, "novelty_validated": validated}


def reduce_scores(scores: list[Score], method: str) -> Score:
    """Aggregate a list of judge scores into one panel score (PoLL reducer).

    Args:
        scores: the per-judge scores to aggregate (must be non-empty).
        method: ``"majority"`` for boolean/discrete ``Score.value`` (modal vote),
            or ``"median"`` for numeric ``Score.value`` (robust central estimate).

    Returns:
        A single :class:`~crucible.types.Score` whose ``value`` is the reduced
        verdict and whose ``metadata`` records the method, the individual votes,
        and the panel size for auditability.

    Raises:
        ValueError: if ``scores`` is empty, ``method`` is unknown, or the values
            are the wrong type for the chosen method (e.g. ``"median"`` over
            non-numeric values).
    """
    if not scores:
        raise ValueError("cannot reduce an empty list of scores")

    values = [s.value for s in scores]
    base_meta: dict[str, object] = {
        "reducer": method,
        "panel_size": len(scores),
        "votes": values,
    }
    # Aggregate the per-judge novelty verdict (§8.7). reduce_scores previously
    # DROPPED per-judge metadata entirely; that is the H2 bug — the novelty
    # adjudication lives in each judge's metadata and the panel is the authority,
    # so it must be carried up. Each judge MAY carry a ``novelty_validated`` bool in
    # its own ``Score.metadata``; an abstaining judge (no key) casts no vote. The
    # panel verdict is the majority of the *cast* votes, defaulting to False when no
    # judge voted (a missing vote is never a validation — conservative).
    base_meta.update(_aggregate_novelty(scores))

    if method == "majority":
        # Modal vote. Bools and discrete strings/ints are all hashable; on a tie
        # Counter.most_common preserves first-seen order, so the panel is
        # deterministic given a fixed judge order (replayability, PIN_PER_STEP).
        try:
            counts = Counter(values)
        except TypeError as exc:  # unhashable value (e.g. a float NaN list)
            raise ValueError(
                "majority reducer requires hashable (bool/discrete) values"
            ) from exc
        winner, win_count = counts.most_common(1)[0]
        base_meta["tally"] = dict(counts)
        base_meta["agreement"] = win_count / len(values)
        return Score(value=winner, metadata=base_meta)

    if method == "median":
        try:
            numeric = [float(v) for v in values]  # type: ignore[arg-type]
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "median reducer requires numeric Score.value entries"
            ) from exc
        med = statistics.median(numeric)
        base_meta["median"] = med
        return Score(value=med, metadata=base_meta)

    raise ValueError(f"unknown reducer method: {method!r} (use 'majority' or 'median')")


class JudgePanel:
    """A cross-family panel of judges with EXTERNAL_VERIFIER exclusion (§10.2).

    The panel runs every eligible judge over an :class:`~crucible.types.AttemptState`
    and reduces their scores. A judge is *eligible* iff its family is not equal to
    ``generator_family`` — this is the structural guarantee that no model verifies
    its own family's output (workflow-standard 6, EXTERNAL_VERIFIER).

    Args:
        judges: the injected judges (Phase-2: real cross-family models). Each is
            an async ``Callable[[AttemptState], Awaitable[Score]]`` optionally
            tagged with a ``family`` attribute.
        reducer: aggregation method passed to :func:`reduce_scores` —
            ``"majority"`` (default) or ``"median"``.
        generator_family: the family of the model that produced the attempt being
            judged. Judges of this family are excluded. ``None`` disables
            exclusion (no generator family known → no judge can be proven to
            share it).
    """

    def __init__(
        self,
        judges: list[JudgeFn],
        reducer: str = "majority",
        generator_family: str | None = None,
    ) -> None:
        self.judges = judges
        self.reducer = reducer
        self.generator_family = generator_family

    def eligible_judges(self) -> list[JudgeFn]:
        """The judges that will actually vote — generator-family judges removed.

        Exclusion only fires when ``generator_family`` is set *and* a judge's
        family equals it. Untagged judges (family ``None``) are always kept.
        """
        if self.generator_family is None:
            return list(self.judges)
        return [j for j in self.judges if judge_family(j) != self.generator_family]

    async def score(self, attempt: AttemptState) -> Score:
        """Run the eligible judges concurrently and reduce to a panel score.

        Args:
            attempt: the attempt to judge.

        Returns:
            The reduced panel :class:`~crucible.types.Score`. Its ``metadata``
            additionally records ``generator_family``, ``excluded`` (the families
            dropped for sharing the generator family), ``eligible_count``, and the
            aggregated novelty verdict ``novelty_validated`` (+ the per-judge
            ``novelty_votes``) — the panel is the novelty authority (§8.7) and the
            kernel feeds this verdict into the oracle gate, never the Solver's or
            oracle_runner's self-report.

        Raises:
            ValueError: if exclusion removes *every* judge (a panel of zero
                cannot return a verdict — the kernel must supply at least one
                cross-family judge, §3 PoLL requires ≥3 in practice). The message
                names the offending generator family so the misconfiguration is
                legible.
        """
        eligible = self.eligible_judges()
        if not eligible:
            raise ValueError(
                "no eligible judges remain after excluding generator family "
                f"{self.generator_family!r}: a cross-family panel needs at least "
                "one judge from a different family (EXTERNAL_VERIFIER, §10.2)"
            )

        # One observable fan-out; judges are independent so we gather concurrently.
        scores = await asyncio.gather(*(judge(attempt) for judge in eligible))

        panel = reduce_scores(list(scores), self.reducer)
        excluded = sorted(
            {
                fam
                for j in self.judges
                if (fam := judge_family(j)) is not None
                and fam == self.generator_family
            }
        )
        panel.metadata.update(
            {
                "generator_family": self.generator_family,
                "excluded": excluded,
                "eligible_count": len(eligible),
            }
        )
        return panel
