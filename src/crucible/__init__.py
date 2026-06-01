"""Crucible — a diagnostic adversarial game for frontier LLMs.

A policy-enforced kernel mediates a Designer -> Solver -> (Critic) -> Judge-panel
cycle, scores against a hidden oracle, and curates a Lab/Arena/Regression catalog.

The kernel is a thin policy layer on UK AISI's Inspect AI (research-grounding
docs/research-grounding.md, section 10.2). All cross-module contracts live in
:mod:`crucible.types`; module bodies implement against them.
"""

from __future__ import annotations

__version__ = "0.2.0"

from crucible.types import (
    AttemptState,
    Budget,
    CatalogTier,
    Chrome,
    FramingArm,
    GoodhartFlavor,
    Penalty,
    PuzzleMeta,
    Rewards,
    Role,
    RoleName,
    Score,
    TerminatedBy,
    TraceEvent,
)

__all__ = [
    "__version__",
    "AttemptState",
    "Budget",
    "CatalogTier",
    "Chrome",
    "FramingArm",
    "GoodhartFlavor",
    "Penalty",
    "PuzzleMeta",
    "Rewards",
    "Role",
    "RoleName",
    "Score",
    "TerminatedBy",
    "TraceEvent",
]
