"""Cross-family judge panel — the EXTERNAL_VERIFIER mitigation (§10.2, §3).

The single strongest mitigation against same-model self-preference is a panel of
judges drawn from *disjoint model families* (Verga et al. 2024, "Replacing Judges
with Juries / PoLL", arXiv:2404.18796 — research-grounding §3): a 3-model
cross-family panel correlates with humans better than a single GPT-4 judge at 1/7
the cost. The bias being defended against is mechanistic — a model over-rates its
own low-perplexity text (Wataoka et al. 2024, arXiv:2410.21819 — §1) — so
"different prompt" is not enough; the judge must come from a different
*distribution*. AI Crucible enforces this **structurally**: any judge whose family
equals the generator's family is excluded from the panel before aggregation
(EXTERNAL_VERIFIER, workflow-standard 6 + research-grounding §10.2,§10.7).

In Phase 2 the judges are real cross-family local models (Qwen + Mistral +
Command-R, via ollama-intern + RTX 5090 — §8.6). Here, in Phase 1, judges are
**injected async callables** so the panel logic is unit-testable without a model
runtime. Each judge is a ``Callable[[AttemptState], Awaitable[Score]]``; its model
family is read from a ``family`` attribute on the callable (set by the kernel when
it wires a model into a judge), falling back to ``None`` (treated as "unknown
family", never excluded — an untagged judge is admitted as external on the
operator's responsibility, since it cannot be *proven* cross-family; the family
comparison is casefold/strip-normalized so a casing drift cannot smuggle a
same-family judge through, scoring-stats-002).

Aggregation is a reducer (PoLL pattern): ``"majority"`` vote for boolean/discrete
verdicts, ``"median"`` for numeric scores (median is the robust central estimate
that resists a single outlier judge — the reason a panel beats one judge).

**Novelty adjudication (§8.7).** The panel is *the authority* on whether a Solver's
claimed novel path is a legitimate alternative or a disguised bypass — novelty
validation needs distributional perspective from outside the Solver's family (PoLL
applied to a different question). A judge expresses its per-judge novelty verdict as
``novelty_validated: bool`` in its returned :class:`~ai_crucible.types.Score.metadata`;
:func:`reduce_scores` aggregates those votes (majority of *cast* votes) into a panel
``novelty_validated`` verdict on the panel score. The kernel feeds THAT verdict into
the oracle gate — the oracle_runner's self-reported ``novelty_validated`` is never
trusted for the gate. Phase 1 uses *injected* judges (so the wiring is testable
without a model runtime); real cross-family panels (Qwen + Mistral + Command-R via
ollama-intern + RTX 5090) plug into the same shape at Phase 2 with no kernel change.
"""

from __future__ import annotations

import asyncio
import math
import statistics
from collections import Counter
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING

from ai_crucible.types import AttemptState, Score

if TYPE_CHECKING:
    from ai_crucible.characterize.aggregate import SeatedPanel

__all__ = ["JudgePanel", "reduce_scores", "judge_family", "weighted_judge"]

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

    An untagged family MUST be Python ``None`` — never a colliding sentinel
    string like ``"unknown"`` (the SHARED FAMILY CONTRACT: a colliding literal
    would make two genuinely-distinct untagged models compare *equal* and silently
    weaken EXTERNAL_VERIFIER exclusion; the CLI stamp in characterize/run.py yields
    ``None`` for an untagged ``--models`` entry, scoring-stats-002 cluster).
    """
    return getattr(judge, "family", None)


def _norm_family(family: str | None) -> str | None:
    """Canonicalize a family label for comparison: ``casefold() + strip()``.

    EXTERNAL_VERIFIER exclusion compares the generator's family against each
    judge's family. The labels are set by two independent tag sites (the kernel's
    generator tag and a seat's ``family``), so a casing/whitespace drift —
    ``"Claude"`` vs ``"claude"`` vs ``"  CLAUDE "`` — would let a same-family judge
    survive a raw string inequality and vote on its own family's output (the exact
    self-preference bias §10.2 prevents). Normalizing both sides closes that drift
    (scoring-stats-002). ``None`` (untagged) is returned unchanged: it never equals
    any concrete family, so an untagged judge is never *proven* same-family.
    """
    return family.casefold().strip() if family is not None else None


def judge_model_id(judge: JudgeFn) -> str:
    """Best-effort human-readable id for a judge, for audit metadata.

    Reads a ``model_id`` attribute (set by the kernel / :func:`weighted_judge`)
    when present, else falls back to the callable's ``__name__``. Used only to
    populate the ``untagged_judges_seated`` audit list (scoring-stats-002) so the
    operator can SEE which untagged (None-family) judge was admitted on trust; it
    never participates in exclusion.
    """
    mid = getattr(judge, "model_id", None)
    if isinstance(mid, str) and mid:
        return mid
    return getattr(judge, "__name__", repr(judge))


def weighted_judge(judge: JudgeFn, weight: float, *, family: str | None = None) -> JudgeFn:
    """Wrap a judge so its returned Score carries a ``judge_weight`` (CARE, §11.4).

    The composed panel (:func:`ai_crucible.characterize.aggregate.compose_panel`) assigns
    each seat a reliability weight; binding it here lets the ``"weighted"`` reducer
    weight that judge's vote without the judge needing to know its own weight. The model
    ``family`` is preserved (or overridden via the keyword) so EXTERNAL_VERIFIER
    generator-family exclusion still fires on the wrapped judge.

    Args:
        judge: the judge callable to wrap.
        weight: the reliability weight to stamp into each Score's metadata (≥ 0).
        family: override the wrapped judge's family tag; defaults to the inner judge's.

    Returns:
        A judge callable that runs ``judge`` and adds ``judge_weight`` to the Score.

    Raises:
        ValueError: if ``weight`` is negative.
    """
    if weight < 0.0:
        raise ValueError(f"judge weight must be >= 0, got {weight}")

    async def _weighted(attempt: AttemptState) -> Score:
        score = await judge(attempt)
        score.metadata["judge_weight"] = float(weight)
        return score

    _weighted.family = family if family is not None else judge_family(judge)  # type: ignore[attr-defined]
    # Carry the inner judge's model id through the wrapper so the audit metadata
    # (untagged_judges_seated, scoring-stats-002) can name a seated judge.
    _weighted.model_id = judge_model_id(judge)  # type: ignore[attr-defined]
    return _weighted


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
        A single :class:`~ai_crucible.types.Score` whose ``value`` is the reduced
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
        # Drop non-finite (NaN/±inf) judge values BEFORE reducing (scoring-numerics-002).
        # ``statistics.median`` over a list containing a NaN returns an
        # ORDER-DEPENDENT NaN (the partition comparisons against NaN are all False),
        # so a single broken judge would silently poison the panel score — exactly the
        # outlier the median reducer exists to resist (§3, the panel-beats-one-judge
        # property). We reduce over the finite peers and flag how many were dropped so
        # the degradation is visible, not hidden. If EVERY value is non-finite there is
        # no finite central estimate to report, so we fail with a clear reason rather
        # than return a silent NaN (fail-closed, mirroring the empty-panel floor).
        finite = [v for v in numeric if math.isfinite(v)]
        dropped = len(numeric) - len(finite)
        if not finite:
            raise ValueError(
                "median reducer: every judge value is non-finite (NaN/inf) — "
                "no finite central estimate exists (scoring-numerics-002)"
            )
        base_meta["non_finite_dropped"] = dropped
        med = statistics.median(finite)
        base_meta["median"] = med
        return Score(value=med, metadata=base_meta)

    if method == "weighted":
        # Reliability-weighted vote (CARE, §11.4): each judge's vote is weighted by the
        # ``judge_weight`` it carries in its Score.metadata (a seat's reliability weight,
        # bound by :func:`weighted_judge`; default 1.0 = equal vote). The weighted
        # analogue of "majority" — for DISCRETE verdicts (numeric scores use "median").
        # Surfaces ``escalate``: a too-divided panel routes to the Claude Designer (the
        # §11.1 Trust-or-Escalate path) rather than committing to a thin weighted winner.
        from ai_crucible.characterize.aggregate import reliability_weighted_vote

        weights = [float(s.metadata.get("judge_weight", 1.0)) for s in scores]
        try:
            result = reliability_weighted_vote(list(zip(values, weights, strict=True)))
        except ValueError as exc:
            raise ValueError(f"weighted reducer: {exc}") from exc
        base_meta["weights"] = weights
        base_meta["weighted_tally"] = result.weighted_tally
        base_meta["total_weight"] = result.total_weight
        base_meta["margin"] = result.margin
        base_meta["escalate"] = result.escalate
        return Score(value=result.value, metadata=base_meta)

    raise ValueError(
        f"unknown reducer method: {method!r} (use 'majority', 'median', or 'weighted')"
    )


class JudgePanel:
    """A cross-family panel of judges with EXTERNAL_VERIFIER exclusion (§10.2).

    The panel runs every eligible judge over an :class:`~ai_crucible.types.AttemptState`
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
            judged. Judges of this family are excluded (case/whitespace-normalized
            comparison — ``"Claude"`` matches ``"claude"``, scoring-stats-002).
            ``None`` disables exclusion (no generator family known → no judge can
            be proven to share it).
        strict_cross_family: when ``True``, also exclude *untagged* (None-family)
            judges that cannot be PROVEN cross-family — they are admitted by
            default (``False``) on operator responsibility (see
            :meth:`eligible_judges`). Default ``False`` preserves the existing
            panel-emptying semantics (an all-untagged panel still scores).
    """

    def __init__(
        self,
        judges: list[JudgeFn],
        reducer: str = "majority",
        generator_family: str | None = None,
        *,
        strict_cross_family: bool = False,
    ) -> None:
        self.judges = judges
        self.reducer = reducer
        self.generator_family = generator_family
        self.strict_cross_family = strict_cross_family

    @classmethod
    def from_seated(
        cls,
        panel: SeatedPanel,
        judge_for: Callable[[str], JudgeFn],
        *,
        generator_family: str | None = None,
        strict_cross_family: bool = False,
    ) -> JudgePanel:
        """Build a reliability-weighted panel from a composed :class:`SeatedPanel`.

        The bridge from characterization to scoring: each seated judge is instantiated
        via ``judge_for`` (the kernel injects this — e.g.
        ``lambda mid: OllamaModel(mid, fam).as_judge()`` — so this module stays decoupled
        from the model adapters, DECOMPOSE_BY_SECRETS) and bound to its seat's reliability
        weight + family (:func:`weighted_judge`). The reducer is ``"weighted"`` (CARE,
        §11.4); same-family-as-generator judges are still excluded at score time.

        Note the composition already enforced the ρ-submodularity gate and the PoLL ≥3
        quorum (:func:`ai_crucible.characterize.aggregate.compose_panel`); a sub-quorum
        ``panel.escalate`` panel should be routed to the Designer by the caller rather
        than seated here.

        Args:
            panel: the composed :class:`SeatedPanel` (its ``seats`` are instantiated).
            judge_for: maps a seated ``model_id`` to a concrete judge callable.
            generator_family: the attempt's generator family (excluded judges, §10.2).
            strict_cross_family: forwarded to the panel — exclude None-family seats
                (default ``False``; see :meth:`eligible_judges`).

        Returns:
            A :class:`JudgePanel` with the seated judges, ``"weighted"`` reducer.
        """
        judges = []
        for seat in panel.seats:
            wrapped = weighted_judge(
                judge_for(seat.model_id), seat.reliability_weight, family=seat.family
            )
            # Carry the seat's model id onto the wrapper so an untagged seat is
            # nameable in untagged_judges_seated (scoring-stats-002).
            wrapped.model_id = seat.model_id  # type: ignore[attr-defined]
            judges.append(wrapped)
        return cls(
            judges=judges,
            reducer="weighted",
            generator_family=generator_family,
            strict_cross_family=strict_cross_family,
        )

    def eligible_judges(self) -> list[JudgeFn]:
        """The judges that will actually vote — generator-family judges removed.

        Exclusion fires when ``generator_family`` is set *and* a judge's family
        equals it under a **normalized** (``casefold() + strip()``) comparison, so
        a casing/whitespace drift in the family vocabulary — ``"Claude"`` vs
        ``"claude"`` vs ``"  CLAUDE "`` — can no longer let a same-family judge
        survive (scoring-stats-002).

        **The honest residual on untagged judges.** A judge whose family is
        ``None`` (untagged) is admitted as cross-family **on the operator's
        responsibility** — it CANNOT be *proven* cross-family, so we do not silently
        drop it (that would over-exclude), but we also cannot certify it is not the
        generator's own family. :meth:`score` records every admitted untagged judge
        in ``untagged_judges_seated`` so this trust assumption is visible, not
        hidden. Set ``strict_cross_family=True`` to exclude None-family judges
        instead (routing an emptied panel through the §3 empty-panel ``ValueError``);
        the default ``False`` preserves the existing panel-emptying semantics.
        """
        if self.generator_family is None:
            # No generator family known → nothing can be proven to share it. Even
            # in strict mode there is no concrete family to exclude against.
            return list(self.judges)

        gen = _norm_family(self.generator_family)
        eligible: list[JudgeFn] = []
        for j in self.judges:
            fam = _norm_family(judge_family(j))
            if fam is None:
                # Untagged: admitted unless strict mode excludes the unprovable.
                if not self.strict_cross_family:
                    eligible.append(j)
                continue
            if fam != gen:
                eligible.append(j)
        return eligible

    async def score(self, attempt: AttemptState) -> Score:
        """Run the eligible judges concurrently and reduce to a panel score.

        Args:
            attempt: the attempt to judge.

        Returns:
            The reduced panel :class:`~ai_crucible.types.Score`. Its ``metadata``
            additionally records ``generator_family``, ``excluded`` (the families
            dropped for sharing the generator family, normalized), ``eligible_count``,
            ``untagged_judges_seated`` (model ids of admitted None-family judges —
            seated on operator trust, not proven cross-family, scoring-stats-002),
            ``judges_errored`` (id + error of any judge that RAISED and was dropped
            from the reduction — a non-empty list means a DEGRADED partial panel,
            scoring-numerics-001), and the aggregated novelty verdict
            ``novelty_validated`` (+ the per-judge
            ``novelty_votes``) — the panel is the novelty authority (§8.7) and the
            kernel feeds this verdict into the oracle gate, never the Solver's or
            oracle_runner's self-report.

        Raises:
            ValueError: if exclusion removes *every* judge (a panel of zero
                cannot return a verdict — the kernel must supply at least one
                cross-family judge, §3 PoLL requires ≥3 in practice). The message
                names the offending generator family so the misconfiguration is
                legible.
            ValueError: if every eligible judge RAISED (a dead/flaky model
                runtime), so zero scores survived — the empty-panel floor. A
                *partial* failure (≥1 surviving judge) degrades gracefully instead
                (scoring-numerics-001, SHARED GRADING-SEAM CONTRACT): the survivors
                decide and the errored judges are recorded in
                ``metadata['judges_errored']`` (id + error), never re-raised.
        """
        eligible = self.eligible_judges()
        if not eligible:
            raise ValueError(
                "no eligible judges remain after excluding generator family "
                f"{self.generator_family!r}: a cross-family panel needs at least "
                "one judge from a different family (EXTERNAL_VERIFIER, §10.2)"
            )

        # DEGRADE on a flaky judge (scoring-numerics-001, SHARED GRADING-SEAM
        # CONTRACT). One observable fan-out; judges are independent so we gather
        # concurrently with ``return_exceptions=True`` so ONE dead judge (e.g. the
        # ollama daemon down, a load timeout) does not abort the whole panel and
        # discard the survivors' completed work. We reduce over the SURVIVING scores
        # and record each errored judge (id + error) in ``judges_errored`` for
        # auditability. The §3 empty-panel ValueError is raised ONLY when ZERO judges
        # survived (the floor) — a partial panel returns a Score, never a bare
        # exception. (Agent A's kernel seam then catches the zero-survivor raise into
        # a traced, degraded attempt; this half makes the partial-panel case never
        # reach it.)
        results = await asyncio.gather(
            *(judge(attempt) for judge in eligible),
            return_exceptions=True,
        )
        scores: list[Score] = []
        judges_errored: list[dict[str, str]] = []
        for judge, result in zip(eligible, results, strict=True):
            if isinstance(result, BaseException):
                judges_errored.append(
                    {"judge": judge_model_id(judge), "error": str(result)}
                )
            else:
                scores.append(result)
        if not scores:
            # Zero judges survived — the floor. No verdict exists to give; raise the
            # empty-panel ValueError naming the failure so the kernel's andon (Agent
            # A) can catch it into a traced, degraded attempt (§3, ANDON_AUTHORITY).
            errs = "; ".join(f"{e['judge']}: {e['error']}" for e in judges_errored)
            raise ValueError(
                "no judges survived scoring — every eligible judge raised, so the "
                f"panel has zero scores to reduce (errors: {errs}) "
                "(EXTERNAL_VERIFIER floor, §3 / scoring-numerics-001)"
            )

        panel = reduce_scores(scores, self.reducer)
        gen = _norm_family(self.generator_family)
        excluded = sorted(
            {
                fam
                for j in self.judges
                if (fam := judge_family(j)) is not None
                and _norm_family(fam) == gen
            }
        )
        # The honest residual (scoring-stats-002): list the untagged (None-family)
        # judges that VOTED — admitted as cross-family on operator responsibility,
        # since they cannot be proven cross-family. Surfacing the ids keeps the
        # trust assumption visible rather than hidden behind a clean panel verdict.
        untagged_seated = [
            judge_model_id(j) for j in eligible if judge_family(j) is None
        ]
        panel.metadata.update(
            {
                "generator_family": self.generator_family,
                "excluded": excluded,
                "eligible_count": len(eligible),
                "untagged_judges_seated": untagged_seated,
                # The judges that RAISED and were dropped from the reduction
                # (scoring-numerics-001). Empty when every eligible judge scored;
                # a non-empty list means the panel ruled on a DEGRADED (partial)
                # set of survivors — surfaced so the degradation is visible to the
                # kernel + audit, not hidden behind a clean verdict.
                "judges_errored": judges_errored,
            }
        )
        return panel
