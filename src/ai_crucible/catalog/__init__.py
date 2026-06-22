"""AI Crucible catalog layer (Epic 4) — persistence + graduation + differential.

The catalog accumulates per-(puzzle, model) diagnostic history across runs as an
EVENT-SOURCED, hash-chained log (the source of truth), derives the
Lab→Arena→Regression tier state by folding it, and classifies each puzzle into the
differential typology (Claude vs cross-family cohort). Design lock:
``swarm/epic-4/CONTRACT.md``.

Three leaves implement against the :mod:`ai_crucible.catalog.types` contract and do
not import each other — the store takes the lifecycle decisions as INJECTED callables:

* :mod:`ai_crucible.catalog.store` — event-sourced persistence + the tier-fold mechanism.
* :mod:`ai_crucible.catalog.graduation` — the Lab→Arena→Regression lifecycle rules.
* :mod:`ai_crucible.catalog.differential` — the diagnostic typology statistics.

The integrator wires the default decision functions + the CLI surface; this
``__init__`` re-exports the contract so callers have one import home.
"""

from __future__ import annotations

from ai_crucible.catalog.types import (
    CATALOG_SCHEMA_VERSION,
    ClassifyResult,
    Decision,
    DemoteVerdict,
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
]
