"""Event-sourced catalog persistence + the tier-fold projection (CONTRACT §D).

The catalog is **event-sourced** on the existing :class:`ai_crucible.attestation.JsonlEventStore`:
a single hash-chained JSON-lines log of RUN and TRANSITION events is the **SOURCE OF
TRUTH** (the audit artifact itself — RFC-8785 canonical JSON, ``prev_hash``/``hash``,
externally verifiable). Tier state (Lab / Arena / Regression) is a **DERIVED projection**
rebuilt by folding the transitions; it is **never** mutated in place. A tier change is an
appended TRANSITION event and a re-fold (CONTRACT §D, cross-cutting lock — "log wins").

:class:`CatalogStore` is the only writer. It wraps the store with three concerns:

1. **Idempotent appends** — :meth:`record_run` / :meth:`record_transition` skip an event
   whose deterministic id (``run_id`` / ``transition_id``) is already in the log. Replaying
   the same evidence appends nothing new (PIN_PER_STEP / CONTRACT §D idempotency).
2. **Folds** — :meth:`read_runs` / :meth:`read_transitions` (dedup, first-wins),
   :meth:`current_tiers` (last-transition-per-puzzle wins; LAB default),
   :meth:`aggregate` (per-puzzle Claude-vs-cohort split + ``per_model`` series + latest
   present panel) — the views graduation + differential consume.
3. **The projection** — :meth:`derive_tiers` is the FOLD MECHANISM. It reads the current
   tier per puzzle and, taking the lifecycle decisions as **injected callables**
   (:data:`~ai_crucible.catalog.types.PromoteFn` / :data:`~ai_crucible.catalog.types.DemoteFn`
   / :data:`~ai_crucible.catalog.types.ClassifyFn`), emits the appropriate TRANSITION
   events. It **never imports** ``graduation`` or ``differential`` (DECOMPOSE_BY_SECRETS):
   the store knows event mechanics; the leaves know the rules; the integrator wires them.

The store records **outcomes, not oracles** — the grading secret never enters the catalog.

Standards compliance (the six — workflow-standards.md)
------------------------------------------------------
- **PIN_PER_STEP — 3:** every appended event carries ``rule_version`` + ``schema_version``
  + a deterministic id; :meth:`derive_tiers` is a pure function of the folded log + the
  injected decision fns + the pinned ``now`` timestamp (no ``datetime.now()`` / ``random``
  inside) so a fold is byte-replayable, and re-running it with the same evidence appends
  nothing (the idempotency tests prove it).
- **ANDON_AUTHORITY — 2:** a record stamped with a HIGHER ``schema_version`` than this build
  understands HALTS the fold loud with a structured ``[CATALOG_SCHEMA_FUTURE]`` error (a
  downgrade refusing to mis-read newer data, mirroring ``characterize.panel_store``);
  :meth:`verify` delegates to ``verify_hash_chain`` so a tampered/broken log is detectable
  before any tier state is trusted. Bad tier state never silently propagates.
- **NAMED_COMPENSATORS — 2 (NO skip — the store performs durable writes):**

  - **action:** append a RUN/TRANSITION event to the JSONL log.
  - **command-to-undo:** the log is append-only — the logical undo is to APPEND a
    compensating ``MANUAL`` transition, never edit or delete a line.
  - **post-rollback state:** the log carries an explicit reversing transition; the tier
    re-folds; history is preserved (the demote-not-delete ethos).
  - **owner:** ``catalog.store``.

  (The single-writer lock-file compensator named in the CONTRACT is deferred with the
  sqlite cache; this build is the JSONL fold. The append compensator above is honored:
  :meth:`record_run` / :meth:`record_transition` never edit or delete a line.)
- **DECOMPOSE_BY_SECRETS — 3:** the store takes ``promote_fn`` / ``demote_fn`` /
  ``classify_fn`` / ``is_frontier`` / ``min_k_for`` as injected callables and imports
  neither ``graduation`` nor ``differential``; its tests inject fakes.
- **UNCERTAINTY_GATED_HUMANS — 2:** :meth:`derive_tiers` records the DEFER_TO_DESIGNER
  receipt (a LAB→LAB no-tier-change escalation) and :meth:`catalog_summary` surfaces the
  ``defer_fraction`` instrument-health metric (CONTRACT cross-cutting lock 4).
- **EXTERNAL_VERIFIER — 3:** the cross-family split the store computes IS the external
  anchor — ``aggregate`` separates the generator family from the cohort so graduation +
  the differential certify against non-self families; the store never self-certifies.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from ai_crucible.attestation import JsonlEventStore
from ai_crucible.catalog.types import (
    CATALOG_SCHEMA_VERSION,
    ClassifyFn,
    Decision,
    DemoteFn,
    FrontierFn,
    ModelRunSeries,
    PanelSignal,
    PromoteFn,
    PuzzleAggregate,
    RuleConfig,
    RunRecord,
    TransitionReason,
    TransitionRecord,
)
from ai_crucible.types import CatalogTier

__all__ = [
    "CATALOG_STORE_FILENAME",
    "CatalogStore",
    "CatalogStoreError",
]

#: Conventional file name for the catalog log under a project's state dir (the
#: integrator chooses the parent dir; this keeps the name stable across the codebase).
CATALOG_STORE_FILENAME = "catalog.jsonl"


class CatalogStoreError(Exception):
    """Raised on a catalog fold defect (a schema-future record, a malformed event).

    Carries the repo's Ship-Gate-B structured message shape (``[CODE] message
    (hint: ...)``) — mirrors :class:`ai_crucible.attestation.HashChainError` and
    :class:`ai_crucible.characterize.panel_store.PanelArtifactError` — so a bad log
    surfaces an actionable error, never a raw stack. (A broken HASH CHAIN is a
    ``verify()`` ``False`` return, not this exception — see :meth:`verify`.)
    """


def _fail(code: str, message: str, hint: str) -> CatalogStoreError:
    return CatalogStoreError(f"[{code}] {message} (hint: {hint})")


def _check_schema(event: dict, *, kind: str) -> None:
    """ANDON: refuse a record from a NEWER schema than this build understands.

    A downgrade reading newer data would silently mis-interpret fields it does not
    know about; halt loud instead (CONTRACT §D forward-incompat guard, mirroring
    ``characterize.panel_store``'s version guard).
    """
    version = event.get("schema_version", CATALOG_SCHEMA_VERSION)
    try:
        version_i = int(version)
    except (TypeError, ValueError) as exc:
        raise _fail(
            "CATALOG_SCHEMA_BAD",
            f"a {kind} event has a non-integer schema_version {version!r}",
            "the log is corrupt or hand-edited; restore from a verified copy",
        ) from exc
    if version_i > CATALOG_SCHEMA_VERSION:
        raise _fail(
            "CATALOG_SCHEMA_FUTURE",
            f"a {kind} event has schema_version {version_i}, newer than this build "
            f"understands ({CATALOG_SCHEMA_VERSION})",
            "upgrade ai_crucible to read this catalog; an older build must not "
            "silently mis-read events written by a newer one",
        )


class CatalogStore:
    """The single writer + folder over the hash-chained catalog log (CONTRACT §D).

    Wraps a :class:`~ai_crucible.attestation.JsonlEventStore` at ``path``. All public
    folds are pure functions of the stored events; all writes are idempotent appends.
    """

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._store = JsonlEventStore(self.path)

    # ------------------------------------------------------------- internals
    def _events(self) -> list[dict]:
        """The raw event payloads in log order (envelope stripped)."""
        return self._store.read_events()

    def _seen_ids(self, *, kind: str, id_field: str) -> set[str]:
        return {
            ev[id_field]
            for ev in self._events()
            if ev.get("kind") == kind and id_field in ev
        }

    # ------------------------------------------------------------ idempotent writes
    def record_run(self, run: RunRecord) -> str | None:
        """Append ``run`` (idempotent on ``run.run_id``).

        Returns the new chain hash on a genuine append, or ``None`` if a run with the
        same deterministic ``run_id`` is already in the log (skipped — CONTRACT §D
        idempotency; append-only, never an edit).
        """
        if run.run_id in self._seen_ids(kind="run", id_field="run_id"):
            return None
        return self._store.append(run.to_payload())

    def record_transition(self, transition: TransitionRecord) -> str | None:
        """Append ``transition`` (idempotent on ``transition.transition_id``).

        Returns the new chain hash on a genuine append, or ``None`` if a transition
        with the same deterministic ``transition_id`` is already in the log (so the
        same evidence under the same rule fires the transition at most once —
        CONTRACT §D; this is what makes :meth:`derive_tiers` replay-idempotent).
        """
        if transition.transition_id in self._seen_ids(
            kind="transition", id_field="transition_id"
        ):
            return None
        return self._store.append(transition.to_payload())

    # ------------------------------------------------------------------ folds
    def read_runs(self) -> list[RunRecord]:
        """Fold the log into RUN records (dedup by ``run_id``, first wins, log order).

        Raises:
            CatalogStoreError: a RUN event carries a schema_version newer than this
                build (``[CATALOG_SCHEMA_FUTURE]`` — ANDON, see :func:`_check_schema`).
        """
        out: list[RunRecord] = []
        seen: set[str] = set()
        for ev in self._events():
            if ev.get("kind") != "run":
                continue
            _check_schema(ev, kind="run")
            rid = ev.get("run_id")
            if rid in seen:
                continue
            if rid is not None:
                seen.add(rid)
            out.append(RunRecord.from_payload(ev))
        return out

    def read_transitions(self) -> list[TransitionRecord]:
        """Fold the log into TRANSITION records (dedup by ``transition_id``, log order).

        Raises:
            CatalogStoreError: a TRANSITION event carries a future schema_version.
        """
        out: list[TransitionRecord] = []
        seen: set[str] = set()
        for ev in self._events():
            if ev.get("kind") != "transition":
                continue
            _check_schema(ev, kind="transition")
            tid = ev.get("transition_id")
            if tid in seen:
                continue
            if tid is not None:
                seen.add(tid)
            out.append(TransitionRecord.from_payload(ev))
        return out

    def verify(self) -> bool:
        """Delegate to the underlying hash-chain verifier (CONTRACT §D / ANDON check).

        Returns ``True`` when the chain is intact (or empty), ``False`` when any link
        is broken (a payload edited / a line inserted/removed/reordered). A broken
        chain is a *return value*, not an exception — ``catalog verify`` is the andon
        gate the operator reads before trusting the derived tier state.
        """
        return self._store.verify_hash_chain()

    def current_tiers(self) -> dict[str, CatalogTier]:
        """Fold the transitions into the current tier per puzzle (the projection).

        In log order, the LAST transition recorded for a puzzle sets its tier (a
        LAB→LAB DEFER receipt leaves the tier at LAB — it records the escalation, it
        does not move the puzzle). A puzzle that has RUN events but no TRANSITION
        defaults to :attr:`~ai_crucible.types.CatalogTier.LAB` (the seed tier). Tier
        is never stored mutably; it is always this fold (CONTRACT §D).
        """
        tiers: dict[str, CatalogTier] = {}
        # Seed every puzzle that has any run at LAB.
        for run in self.read_runs():
            tiers.setdefault(run.puzzle_id, CatalogTier.LAB)
        # Apply transitions in order; the last per puzzle wins.
        for t in self.read_transitions():
            tiers[t.puzzle_id] = t.to_tier
        return tiers

    def aggregate(
        self,
        *,
        generator_family: str = "claude",
        is_frontier: FrontierFn,
        min_k_for: Callable[[str], int],
    ) -> dict[str, PuzzleAggregate]:
        """Fold RUN records into one :class:`PuzzleAggregate` per puzzle (CONTRACT §A/§C).

        For each puzzle:

        * **Family split (case-insensitive):** runs whose ``family`` equals
          ``generator_family`` accumulate into ``claude_successes``/``claude_n``; ALL
          other families accumulate into ``cohort_successes``/``cohort_n`` (the
          cross-family cohort — the external anchor; both at the same N floor,
          cross-cutting lock 1).
        * **per_model:** a :class:`ModelRunSeries` per ``model_id`` whose ``runs`` is the
          chronological ``[(successes, n), ...]`` (per-RUN, the saturation unit —
          CONTRACT §B), with ``is_frontier`` from the injected predicate and ``min_k``
          from ``min_k_for(puzzle_id)``.
        * **puzzle_content_hash:** the latest run's content hash (DVC lineage).
        * **panel:** the LATEST run (by ``started_at``, ties broken by log order) that
          carried a panel with ``present=True``; if none present, ``PanelSignal(present=False)``.
        * **contributing_run_ids:** every ``run_id`` for the puzzle.
        """
        gen = generator_family.lower()
        # Buckets per puzzle, populated in log order.
        claude_s: dict[str, int] = {}
        claude_n: dict[str, int] = {}
        cohort_s: dict[str, int] = {}
        cohort_n: dict[str, int] = {}
        content_hash: dict[str, str] = {}
        per_model: dict[str, dict[str, ModelRunSeries]] = {}
        run_ids: dict[str, list[str]] = {}
        # Latest-present-panel tracking: (started_at, log_index) of the winner.
        best_panel: dict[str, tuple[str, int]] = {}
        panel_winner: dict[str, PanelSignal] = {}

        for idx, run in enumerate(self.read_runs()):
            pid = run.puzzle_id
            claude_s.setdefault(pid, 0)
            claude_n.setdefault(pid, 0)
            cohort_s.setdefault(pid, 0)
            cohort_n.setdefault(pid, 0)
            per_model.setdefault(pid, {})
            run_ids.setdefault(pid, [])

            fam = (run.family or "").lower()
            if fam == gen:
                claude_s[pid] += run.successes
                claude_n[pid] += run.n
            else:
                cohort_s[pid] += run.successes
                cohort_n[pid] += run.n

            series = per_model[pid].get(run.model_id)
            if series is None:
                series = ModelRunSeries(
                    puzzle_id=pid,
                    model_id=run.model_id,
                    family=run.family,
                    is_frontier=is_frontier(run.model_id),
                    min_k=min_k_for(pid),
                    runs=[],
                )
                per_model[pid][run.model_id] = series
            series.runs.append((run.successes, run.n))

            content_hash[pid] = run.puzzle_content_hash  # latest wins (log order)
            run_ids[pid].append(run.run_id)

            if run.panel is not None and run.panel.present:
                key = (run.started_at, idx)
                if pid not in best_panel or key > best_panel[pid]:
                    best_panel[pid] = key
                    panel_winner[pid] = run.panel

        aggregates: dict[str, PuzzleAggregate] = {}
        for pid in claude_s:
            aggregates[pid] = PuzzleAggregate(
                puzzle_id=pid,
                puzzle_content_hash=content_hash.get(pid, ""),
                claude_successes=claude_s[pid],
                claude_n=claude_n[pid],
                cohort_successes=cohort_s[pid],
                cohort_n=cohort_n[pid],
                panel=panel_winner.get(pid, PanelSignal(present=False)),
                per_model=per_model[pid],
                contributing_run_ids=run_ids[pid],
            )
        return aggregates

    # ----------------------------------------------------------- the projection
    def derive_tiers(
        self,
        *,
        promote_fn: PromoteFn,
        demote_fn: DemoteFn,
        classify_fn: ClassifyFn,
        rule_config: RuleConfig,
        is_frontier: FrontierFn,
        min_k_for: Callable[[str], int],
        now: str,
        decided_by: str = "auto",
        repromote_fn: DemoteFn | None = None,
    ) -> list[TransitionRecord]:
        """THE FOLD / PROJECTION: emit the right TRANSITION per puzzle (CONTRACT §D).

        Reads the current tier per puzzle (:meth:`current_tiers`) and, per tier, calls
        the **injected** lifecycle decision fns (the store imports neither ``graduation``
        nor ``differential`` — DECOMPOSE_BY_SECRETS):

        * **LAB** — ``v = promote_fn(agg, rule_config)``:
          - ``PROMOTE`` → emit LAB→ARENA / ``GRADUATE_WILSON``, stamping the promote
            clause flags AND the born typology (``classify_fn(agg, rule_config).typology``,
            CONTRACT §A: classify at graduation to stamp the Arena entry's born label).
          - ``DEFER_TO_DESIGNER`` → emit a LAB→LAB / ``DEFER_TO_DESIGNER`` *receipt*
            (records the escalation; does NOT change tier — the uncertainty-gated human
            checkpoint).
          - ``HOLD`` → emit nothing.
        * **ARENA** — for each FRONTIER model series, ``d = demote_fn(series, rule_config)``;
          on the first ``d.demote`` emit ARENA→REGRESSION / ``SATURATED_REGRESSION`` with
          the saturating ``model_id`` + ``e_value``. The frontier gate is applied by the
          store before calling ``demote_fn`` (a weak local model passing never demotes —
          CONTRACT §B).
        * **REGRESSION** — if ``repromote_fn`` is given, for each frontier model series,
          ``r = repromote_fn(series, rule_config)``; on the first ``r.demote`` emit
          REGRESSION→ARENA / ``REPROMOTE_REGRESSION`` (a genuine new-version capability
          regression re-promotes — the symmetric e-process).

        Every emitted record pins ``rule_version=rule_config.rule_version``,
        ``recorded_at=now`` (injected — PIN_PER_STEP, no wall clock here),
        ``contributing_run_ids=agg.contributing_run_ids``, and ``decided_by``. Each is
        appended via the idempotent :meth:`record_transition`, so re-running with the
        same evidence appends nothing new. Returns ONLY the transitions actually
        appended (``record_transition`` returned non-``None``).
        """
        tiers = self.current_tiers()
        aggregates = self.aggregate(
            is_frontier=is_frontier, min_k_for=min_k_for
        )
        rule_version = rule_config.rule_version
        appended: list[TransitionRecord] = []

        for puzzle_id, tier in tiers.items():
            agg = aggregates.get(puzzle_id)
            if agg is None:  # pragma: no cover - tiers seeded from runs, so always present
                continue

            if tier is CatalogTier.LAB:
                t = self._lab_transition(
                    agg, promote_fn, classify_fn, rule_config, now, decided_by, rule_version
                )
            elif tier is CatalogTier.ARENA:
                t = self._frontier_demote_transition(
                    agg,
                    demote_fn,
                    rule_config,
                    now,
                    decided_by,
                    rule_version,
                    from_tier=CatalogTier.ARENA,
                    to_tier=CatalogTier.REGRESSION,
                    reason=TransitionReason.SATURATED_REGRESSION,
                )
            elif tier is CatalogTier.REGRESSION and repromote_fn is not None:
                t = self._frontier_demote_transition(
                    agg,
                    repromote_fn,
                    rule_config,
                    now,
                    decided_by,
                    rule_version,
                    from_tier=CatalogTier.REGRESSION,
                    to_tier=CatalogTier.ARENA,
                    reason=TransitionReason.REPROMOTE_REGRESSION,
                )
            else:
                t = None

            if t is not None and self.record_transition(t) is not None:
                appended.append(t)

        return appended

    # ------------------------------------------------ per-tier transition builders
    def _lab_transition(
        self,
        agg: PuzzleAggregate,
        promote_fn: PromoteFn,
        classify_fn: ClassifyFn,
        rule_config: RuleConfig,
        now: str,
        decided_by: str,
        rule_version: str,
    ) -> TransitionRecord | None:
        v = promote_fn(agg, rule_config)
        if v.decision is Decision.PROMOTE:
            born = v.born_typology
            if born is None:
                born = classify_fn(agg, rule_config).typology
            evidence = {
                "claude_band_ok": v.claude_band_ok,
                "cohort_nontrivial": v.cohort_nontrivial,
                "fairness_ok": v.fairness_ok,
                "deferred": v.deferred,
                "born_typology": str(born),
                **dict(v.evidence),
            }
            return TransitionRecord(
                puzzle_id=agg.puzzle_id,
                from_tier=CatalogTier.LAB,
                to_tier=CatalogTier.ARENA,
                reason_code=TransitionReason.GRADUATE_WILSON,
                recorded_at=now,
                decided_by=decided_by,
                contributing_run_ids=list(agg.contributing_run_ids),
                evidence=evidence,
                rule_version=rule_version,
            )
        if v.decision is Decision.DEFER_TO_DESIGNER:
            evidence = {
                "claude_band_ok": v.claude_band_ok,
                "cohort_nontrivial": v.cohort_nontrivial,
                "fairness_ok": v.fairness_ok,
                "deferred": v.deferred,
                **dict(v.evidence),
            }
            return TransitionRecord(
                puzzle_id=agg.puzzle_id,
                from_tier=CatalogTier.LAB,
                to_tier=CatalogTier.LAB,  # receipt: escalation, no tier change
                reason_code=TransitionReason.DEFER_TO_DESIGNER,
                recorded_at=now,
                decided_by=decided_by,
                contributing_run_ids=list(agg.contributing_run_ids),
                evidence=evidence,
                rule_version=rule_version,
            )
        return None  # HOLD → nothing

    def _frontier_demote_transition(
        self,
        agg: PuzzleAggregate,
        decision_fn: DemoteFn,
        rule_config: RuleConfig,
        now: str,
        decided_by: str,
        rule_version: str,
        *,
        from_tier: CatalogTier,
        to_tier: CatalogTier,
        reason: TransitionReason,
    ) -> TransitionRecord | None:
        """Apply the frontier gate, then ``decision_fn`` per frontier series.

        Used for both ARENA→REGRESSION (saturation ``demote_fn``) and
        REGRESSION→ARENA (symmetric ``repromote_fn``). The store filters to frontier
        models BEFORE calling ``decision_fn`` (CONTRACT §B frontier gate — a weak local
        model passing never moves a puzzle). The FIRST series whose verdict fires wins.
        """
        for series in agg.per_model.values():
            if not series.is_frontier:
                continue
            d = decision_fn(series, rule_config)
            if d.demote:
                evidence = {
                    "model_id": series.model_id,
                    "family": series.family,
                    "e_value": d.e_value,
                    "dwell_runs": d.dwell_runs,
                    "post_trigger_n": d.post_trigger_n,
                    "frontier_ok": d.frontier_ok,
                    "reason": d.reason,
                    **dict(d.evidence),
                }
                return TransitionRecord(
                    puzzle_id=agg.puzzle_id,
                    from_tier=from_tier,
                    to_tier=to_tier,
                    reason_code=reason,
                    recorded_at=now,
                    decided_by=decided_by,
                    contributing_run_ids=list(agg.contributing_run_ids),
                    evidence=evidence,
                    rule_version=rule_version,
                )
        return None

    # ----------------------------------------------------------- instrument health
    def catalog_summary(self) -> dict:
        """A small summary for the CLI: tier counts + the DEFER fraction (lock 4).

        ``defer_fraction`` is the CONTRACT cross-cutting lock-4 instrument-health metric:
        the fraction of LAB puzzles whose LATEST graduation attempt was a DEFER (a rising
        rate means *recruit a third disjoint family*, NOT relax the threshold). A puzzle's
        latest graduation attempt is its last LAB→LAB DEFER receipt OR LAB→ARENA promote;
        a puzzle still in LAB whose last attempt was a DEFER counts toward the numerator.
        """
        tiers = self.current_tiers()
        counts: dict[str, int] = {t.value: 0 for t in CatalogTier}
        for tier in tiers.values():
            counts[tier.value] = counts.get(tier.value, 0) + 1

        # Latest graduation-attempt reason per LAB puzzle, in log order.
        latest_attempt: dict[str, TransitionReason] = {}
        for t in self.read_transitions():
            if t.reason_code in (
                TransitionReason.DEFER_TO_DESIGNER,
                TransitionReason.GRADUATE_WILSON,
            ):
                latest_attempt[t.puzzle_id] = t.reason_code

        lab_puzzles = [pid for pid, tier in tiers.items() if tier is CatalogTier.LAB]
        if lab_puzzles:
            deferred = sum(
                1
                for pid in lab_puzzles
                if latest_attempt.get(pid) is TransitionReason.DEFER_TO_DESIGNER
            )
            defer_fraction = deferred / len(lab_puzzles)
        else:
            defer_fraction = 0.0

        return {
            "tiers": counts,
            "total_puzzles": len(tiers),
            "defer_fraction": defer_fraction,
            "chain_verified": self.verify(),
        }
