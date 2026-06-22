"""AI Crucible catalog layer (Epic 4) — persistence + graduation + differential.

The catalog accumulates per-(puzzle, model) diagnostic history across runs as an
EVENT-SOURCED, hash-chained log (the source of truth), derives the
Lab→Arena→Regression tier state by folding it, and classifies each puzzle into the
differential typology (Claude vs cross-family cohort). Design lock:
``swarm/epic-4/CONTRACT.md``.

Three leaves implement against the :mod:`ai_crucible.catalog.types` contract and do
NOT import each other — the store takes the lifecycle decisions as INJECTED callables:

* :mod:`ai_crucible.catalog.store` — event-sourced persistence + the tier-fold mechanism.
* :mod:`ai_crucible.catalog.graduation` — the Lab→Arena→Regression lifecycle rules.
* :mod:`ai_crucible.catalog.differential` — the diagnostic typology statistics.

This package ``__init__`` is the INTEGRATOR layer (the only place that imports all three
leaves): it binds the default decision functions and exposes :func:`apply_lifecycle`, the
one-call wiring the CLI drives. :mod:`ai_crucible.catalog.ingest` holds the run→RunRecord
glue.
"""

from __future__ import annotations

from ai_crucible.catalog.differential import (
    classify_catalog,
    classify_counts,
    classify_puzzle,
)
from ai_crucible.catalog.graduation import (
    DEFAULT_FRONTIER_IDS,
    demote_decision,
    make_frontier_fn,
    promote_decision,
    repromote_decision,
)
from ai_crucible.catalog.ingest import (
    build_run_record,
    min_k_map,
    panel_signal_from_seated,
    puzzle_content_hash,
)
from ai_crucible.catalog.store import (
    CATALOG_STORE_FILENAME,
    CatalogStore,
    CatalogStoreError,
)
from ai_crucible.catalog.types import (
    CATALOG_SCHEMA_VERSION,
    ClassifyResult,
    Decision,
    DemoteVerdict,
    FrontierFn,
    ModelRunSeries,
    PanelSignal,
    PromoteVerdict,
    PuzzleAggregate,
    RuleConfig,
    RunRecord,
    TransitionReason,
    TransitionRecord,
    Typology,
)

__all__ = [
    # contract
    "CATALOG_SCHEMA_VERSION",
    "Decision",
    "Typology",
    "TransitionReason",
    "RuleConfig",
    "PanelSignal",
    "RunRecord",
    "TransitionRecord",
    "ModelRunSeries",
    "PuzzleAggregate",
    "PromoteVerdict",
    "DemoteVerdict",
    "ClassifyResult",
    "FrontierFn",
    # store
    "CatalogStore",
    "CatalogStoreError",
    "CATALOG_STORE_FILENAME",
    # leaf rules (the default wiring binds these)
    "promote_decision",
    "demote_decision",
    "repromote_decision",
    "make_frontier_fn",
    "DEFAULT_FRONTIER_IDS",
    "classify_counts",
    "classify_puzzle",
    "classify_catalog",
    # ingest glue
    "build_run_record",
    "puzzle_content_hash",
    "panel_signal_from_seated",
    "min_k_map",
    # default wiring
    "default_frontier_fn",
    "apply_lifecycle",
]

#: The default frontier predicate (the known capable cross-family subjects). POLICY —
#: a deployment can build its own via ``make_frontier_fn(extra=[...])`` (CONTRACT §B).
default_frontier_fn: FrontierFn = make_frontier_fn()


def apply_lifecycle(
    store: CatalogStore,
    *,
    rule_config: RuleConfig | None = None,
    now: str,
    decided_by: str = "auto",
    frontier_fn: FrontierFn | None = None,
    enable_repromote: bool = True,
) -> list[TransitionRecord]:
    """Run the full Lab→Arena→Regression projection with the DEFAULT decision wiring.

    The one-call integration the CLI drives: folds the store and emits the tier
    transitions promotion/saturation/re-promotion dictate, with the real leaf decisions
    injected (``promote_decision`` / ``demote_decision`` / ``classify_puzzle`` / and —
    when ``enable_repromote`` — ``repromote_decision``). ``min_k_for`` is recovered from
    the catalog itself (:func:`~ai_crucible.catalog.ingest.min_k_map`) so the saturation
    floor is computable from the log alone.

    Idempotent: re-running with no new evidence appends nothing (the store's
    deterministic transition ids). Returns the transitions actually appended.

    Args:
        store: the :class:`CatalogStore` to fold + append to.
        rule_config: the pinned thresholds (default :class:`RuleConfig`).
        now: the ISO timestamp stamped on emitted transitions (injected at the edge).
        decided_by: ``"auto"`` for a rule-derived transition (an attested Designer
            override uses the store's :meth:`record_transition` directly with a MANUAL
            reason — see the ``catalog graduate --override`` CLI path).
        frontier_fn: the frontier gate (default :data:`default_frontier_fn`).
        enable_repromote: wire the REGRESSION→ARENA edge (default True).

    Returns:
        The list of :class:`TransitionRecord` actually appended this call.
    """
    rule = rule_config if rule_config is not None else RuleConfig()
    frontier = frontier_fn if frontier_fn is not None else default_frontier_fn
    return store.derive_tiers(
        promote_fn=promote_decision,
        demote_fn=demote_decision,
        classify_fn=classify_puzzle,
        rule_config=rule,
        is_frontier=frontier,
        min_k_for=min_k_map(store),
        now=now,
        decided_by=decided_by,
        repromote_fn=repromote_decision if enable_repromote else None,
    )
