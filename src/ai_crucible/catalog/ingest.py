"""Integration glue between a diagnostic run and the catalog store (CONTRACT §D).

Coordinator-authored seam (not a leaf): turns the in-memory artifacts a diagnostic
cycle produces — a :class:`~ai_crucible.observability.PuzzleHistory` and, optionally, a
seated cross-family panel — into a durable :class:`~ai_crucible.catalog.types.RunRecord`
the :class:`~ai_crucible.catalog.store.CatalogStore` appends. Kept out of the pure leaves
so the store/graduation/differential modules stay free of run-time/clock/filesystem
concerns; the timestamps + nonce are injected HERE, at the edge (PIN_PER_STEP — the pure
fold downstream stays deterministic).

What it owns:

* :func:`puzzle_content_hash` — the DVC-style content address over the files that DEFINE
  a puzzle (``meta.json`` + ``prompt`` + ``setup_script`` + ``oracle/check.py``). An edit
  to any of them forks a NEW lineage, so the timeline never conflates two puzzle versions
  (CONTRACT §D). The oracle is hashed but never EXPOSED — a hash leaks nothing.
* :func:`panel_signal_from_seated` — distils a loaded :class:`SeatedPanel` into the small
  :class:`PanelSignal` graduation's fairness clause reads. ``fairness`` stays ``None``
  (HONEST: ai-crucible has no confident cross-family *fairness* judge yet — the panel
  validates novelty, ω is on ice — so graduation correctly DEFERS, never fakes a verdict).
* :func:`build_run_record` — assembles the :class:`RunRecord` from a graded history.
* :func:`min_k_map` — recovers ``{puzzle_id: min_k}`` from the catalog so graduation's
  saturation floor is computable from the log alone (the catalog is self-contained).
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from pathlib import Path

from ai_crucible.catalog.store import CatalogStore
from ai_crucible.catalog.types import PanelSignal, RunRecord
from ai_crucible.observability import PuzzleHistory

__all__ = [
    "puzzle_content_hash",
    "panel_signal_from_seated",
    "build_run_record",
    "min_k_map",
]

#: The files whose bytes DEFINE a puzzle's identity (CONTRACT §D content addressing).
#: oracle/check.py is included (an oracle edit changes what "solved" means → a new
#: lineage) but is only HASHED, never read into Solver-visible state.
_CONTENT_FILES = ("meta.json", "prompt", "setup_script", "oracle/check.py")


def puzzle_content_hash(puzzle_dir: Path) -> str:
    """Content address over the files that define the puzzle at ``puzzle_dir``.

    A stable SHA-256 (first 16 hex) over each content file's relative path + bytes (a
    missing optional file contributes its path + an empty marker, so adding/removing one
    changes the hash). An edit to meta/prompt/setup/oracle forks a NEW puzzle lineage —
    the catalog timeline never silently conflates two versions of "the same" puzzle
    (CONTRACT §D, DVC content-addressing).
    """
    h = hashlib.sha256()
    root = Path(puzzle_dir)
    for rel in _CONTENT_FILES:
        h.update(rel.encode("utf-8"))
        h.update(b"\x00")
        p = root / rel
        if p.is_file():
            h.update(p.read_bytes())
        h.update(b"\x00")
    return h.hexdigest()[:16]


def panel_signal_from_seated(seated: object | None) -> PanelSignal:
    """Distil a loaded :class:`SeatedPanel` into the :class:`PanelSignal` graduation reads.

    ``None`` (no ``--panel``) → ``PanelSignal(present=False)``. A seated panel →
    ``present=True`` with ``meets_quorum`` / ``escalate`` read off the artifact (PoLL ≥3;
    a sub-quorum panel carries ``meets_quorum=False`` + ``escalate=True``).

    ``fairness`` is ALWAYS ``None`` here — HONEST by construction: ai-crucible has no
    confident cross-family *fairness* judge yet (the panel adjudicates novelty; the
    alt-test ω is on ice for lack of ≥3 independent human annotators). So a real run never
    yields a confident "fair"/"unfair" verdict, and graduation DEFERS every puzzle to the
    Designer rather than auto-promoting — the disclosed posture, not a faked confidence.
    """
    if seated is None:
        return PanelSignal(present=False)
    return PanelSignal(
        present=True,
        meets_quorum=bool(getattr(seated, "meets_quorum", False)),
        escalate=bool(getattr(seated, "escalate", True)),
        fairness=None,  # no confident cross-family fairness verdict exists yet (honest)
    )


def build_run_record(
    history: PuzzleHistory,
    *,
    puzzle_dir: Path,
    model_id: str,
    family: str | None,
    k: int,
    min_k: int,
    arm: str,
    started_at: str,
    finished_at: str,
    nonce: str,
    rule_version: str,
    role: str = "solver",
    panel: PanelSignal | None = None,
) -> RunRecord:
    """Assemble a durable :class:`RunRecord` from a graded :class:`PuzzleHistory`.

    Folds the run's reliability views (``n`` / ``successes`` / ``pass^k`` / Wilson) off the
    history and stamps the provenance (content hash, family, arm, timestamps, nonce,
    ``rule_version``, ``min_k``) the catalog needs to be self-contained. The clock + nonce
    are injected by the caller (the CLI edge) so this stays a pure assembler.

    ``family`` is the cross-family axis the differential + graduation cohort split on
    (record it honestly — the Claude Solver vs a CohortSolver). ``role`` distinguishes the
    generator's own run (``"solver"``) from a cross-family cohort run (``"cohort_solver"``).
    """
    ci = history.wilson()
    return RunRecord(
        puzzle_id=history.puzzle_id,
        puzzle_content_hash=puzzle_content_hash(puzzle_dir),
        model_id=model_id,
        family=family,
        role=role,
        k=k,
        n=history.n_attempts,
        successes=history.n_solved,
        pass_hat_k=history.pass_hat_k(k),
        wilson_lower=ci.lower,
        wilson_upper=ci.upper,
        arm=arm,
        started_at=started_at,
        finished_at=finished_at,
        nonce=nonce,
        min_k=min_k,
        rule_version=rule_version,
        panel=panel,
    )


def min_k_map(store: CatalogStore, *, default: int = 10) -> Callable[[str], int]:
    """Build a ``min_k_for(puzzle_id) -> int`` resolver from the catalog itself.

    The graduation saturation floor (``3·min_k``) must be computable from the LOG alone —
    the puzzle files may be gone or changed by graduate-time. Each :class:`RunRecord`
    carries the puzzle's ``min_k``; this folds the latest value per puzzle (log order) into
    a resolver the store injects into :meth:`CatalogStore.aggregate` /
    :meth:`CatalogStore.derive_tiers`. Unknown puzzles fall back to ``default``.
    """
    latest: dict[str, int] = {}
    for run in store.read_runs():
        latest[run.puzzle_id] = run.min_k

    def min_k_for(puzzle_id: str) -> int:
        return latest.get(puzzle_id, default)

    return min_k_for
