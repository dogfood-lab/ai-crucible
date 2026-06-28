"""Calibration-set CURATION — the Phase-A harder-set pipeline (study-swarm grounded).

Operationalizes the harder / less-saturated discriminating-admission-set design
(``swarm/openrouter-quorum/STUDY-SWARM-harder-calibration-set.md``). Two steps:

* **Discrimination selection** (AFLite, Le Bras et al. 2020, arXiv:2002.04108; Fisher-information
  selection, Zhou et al. 2025, arXiv:2505.15055): from a candidate item pool + a recorded panel
  grade-matrix, KEEP the items strong judges DISAGREE on (the discriminators) and DROP the ones
  every judge already passes (saturated). This moves the §12 IRT saturation screen UPSTREAM —
  from a post-hoc report into a forward CURATION step. Reuses :func:`irt.prune_items`.

* **Ambiguity gate** (GPQA difficulty-vs-ambiguity, Rein et al. 2023, arXiv:2311.12022; dual
  verification, JudgeBench, Tan et al. 2024, arXiv:2410.12784): a harder item must stay
  DEFENSIBLE — its disagreement must be DIFFICULTY (a defensible key exists), not AMBIGUITY (no
  key). Given independent verifier verdicts on an item, the key is DEFENSIBLE only when the
  verifiers AGREE with each other AND with the item's gold; verifier disagreement = AMBIGUOUS
  (omit); verifiers-agree-but-differ-from-gold = MISLABELED (omit). The crown-jewel honesty
  constraint: a harder set must never become an ambiguous set.

Phase A is the pipeline + the gate; Phase B feeds it a larger pool of new-construct items. The
discrimination step needs only a recorded run's grade-matrix (no model calls); the ambiguity gate
needs >=2 independent verifier verdicts per item (ideally cross-family models that each answered it
fresh — the same shape as a panel judge), supplied by the caller.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from ai_crucible.calibration import irt
from ai_crucible.calibration.types import CalibrationItem

__all__ = [
    "AmbiguityStatus",
    "AmbiguityVerdict",
    "CurationResult",
    "ambiguity_gate",
    "curate",
    "select_discriminators",
]


class AmbiguityStatus(StrEnum):
    """The ambiguity-gate verdict for one item (§12 difficulty-vs-ambiguity)."""

    DEFENSIBLE = "defensible"      # verifiers agree with each other AND gold -> difficulty, keep
    AMBIGUOUS = "ambiguous"        # verifiers disagree with each other -> no defensible key, omit
    MISLABELED = "mislabeled"      # verifiers agree with each other but DIFFER from gold -> omit
    NOT_VERIFIED = "not_verified"  # < 2 verifier verdicts supplied -> gate not run; keep + flag


@dataclass(slots=True)
class AmbiguityVerdict:
    """One item's ambiguity-gate result."""

    item_id: str
    status: AmbiguityStatus
    gold: str
    verifier_choices: list[str]
    note: str = ""


def ambiguity_gate(item: CalibrationItem, verifier_choices: list[str]) -> AmbiguityVerdict:
    """Classify whether an item has a DEFENSIBLE single key (difficulty) vs AMBIGUITY/MISLABEL.

    ``verifier_choices`` are independent verdicts on the item from the verifiers (ideally >=2
    cross-family models that each answered it fresh). The gate is parser-agnostic — it compares the
    raw choice tokens (stripped + upper-cased) to each other and to the item's ``gold``:

    * **< 2** verdicts -> ``NOT_VERIFIED`` (the gate cannot run; the item is kept but flagged).
    * verifiers **do not all agree** -> ``AMBIGUOUS`` (the disagreement is the ITEM's, not a
      judge's difficulty — there is no defensible single key) -> omit.
    * verifiers **all agree but != gold** -> ``MISLABELED`` (stored gold likely wrong) -> omit.
    * verifiers **all agree AND == gold** -> ``DEFENSIBLE`` (a real key; judge disagreement on
      it is difficulty, exactly the discrimination signal we want) -> keep.
    """
    gold = str(item.gold).strip().upper()
    choices = [str(c).strip().upper() for c in verifier_choices if str(c).strip()]
    if len(choices) < 2:
        return AmbiguityVerdict(
            item.id, AmbiguityStatus.NOT_VERIFIED, gold, choices,
            "fewer than 2 verifier verdicts — ambiguity gate not run",
        )
    unique = set(choices)
    if len(unique) > 1:
        return AmbiguityVerdict(
            item.id, AmbiguityStatus.AMBIGUOUS, gold, choices,
            f"verifiers disagree {sorted(unique)} — no defensible key (ambiguity, not difficulty)",
        )
    agreed = next(iter(unique))
    if agreed != gold:
        return AmbiguityVerdict(
            item.id, AmbiguityStatus.MISLABELED, gold, choices,
            f"verifiers agree on {agreed!r} != stored gold {gold!r} — likely mislabeled",
        )
    return AmbiguityVerdict(
        item.id, AmbiguityStatus.DEFENSIBLE, gold, choices,
        "verifiers agree with gold — defensible key (judge disagreement here is difficulty)",
    )


@dataclass(slots=True)
class CurationResult:
    """The outcome of curating a pool against a panel grade-matrix (+ optional ambiguity gate)."""

    kept: list[str]                        # surviving discriminating (+ defensible when gated) ids
    dropped_saturated: list[str]           # every judge agreed -> zero variance, no discrimination
    dropped_low_discrimination: list[str]  # point-biserial below floor (doesn't track ability)
    dropped_ambiguous: list[str]           # ambiguity gate: verifiers disagreed (no key)
    dropped_mislabeled: list[str]          # ambiguity gate: verifiers agree, gold differs
    not_verified: list[str]                # kept but ambiguity gate not run (no verifier verdicts)
    dropped_ragged: list[str] = field(default_factory=list)  # not scored by every model (salvage)
    notes: list[str] = field(default_factory=list)


def select_discriminators(
    grade_matrix: dict[str, dict[str, bool]],
    *,
    min_variance: float = 0.0,
    min_point_biserial: float = 0.1,
) -> tuple[list[str], list[str], list[str], list[str]]:
    """The AFLite / Fisher discrimination filter as a forward step.

    Returns ``(kept, dropped_saturated, dropped_low_discrimination, dropped_ragged)``. Reuses
    :func:`irt.prune_items` (the variance + corrected-point-biserial screen) for the kept/dropped
    split, then re-derives WHY each dropped item dropped (saturated vs low-r_pb) so the curation is
    legible — the saturation screen is the cheaper/first one, so an item that fails it is reported
    saturated even if it would also fail point-biserial.

    A RAGGED grade-matrix (per-item salvage left a model missing a few items — e.g. a cloud-run
    timeout) is degraded to the shared item subset via :func:`irt.shared_item_matrix`, and the
    ragged drops are reported rather than crashing the whole screen with ``IRT_RAGGED_MATRIX`` (the
    same degradation the run's post-hoc ``irt_prune_report`` applies; caught by the 2026-06-28
    live-ρ demo).
    """
    restricted, dropped_ragged = irt.shared_item_matrix(grade_matrix)
    kept, dropped = irt.prune_items(
        restricted, min_variance=min_variance, min_point_biserial=min_point_biserial
    )
    saturated: list[str] = []
    low_disc: list[str] = []
    for iid in dropped:
        col = [bool(row[iid]) for row in restricted.values() if iid in row]
        if irt.variance_of_item(col) <= min_variance:
            saturated.append(iid)
        else:
            low_disc.append(iid)
    return kept, saturated, low_disc, dropped_ragged


def curate(
    items: list[CalibrationItem],
    grade_matrix: dict[str, dict[str, bool]],
    *,
    ambiguity_verdicts: dict[str, list[str]] | None = None,
    min_variance: float = 0.0,
    min_point_biserial: float = 0.1,
) -> CurationResult:
    """The Phase-A curation pipeline: discrimination-filter the pool against a recorded panel
    grade-matrix, then (when verifier verdicts are supplied) ambiguity-gate the survivors.

    Args:
        items: the candidate item pool.
        grade_matrix: ``{model_id: {item_id: correct_bool}}`` from a characterization run (the
            majority-collapsed per-item panel verdicts — :func:`run._grade_matrix`).
        ambiguity_verdicts: optional ``{item_id: [verifier_choice, ...]}`` to run the ambiguity
            gate over the discriminators. Omitted (or missing for an item) -> that survivor is kept
            and flagged ``not_verified`` (the gate is never silently skipped — its absence is
            reported).
        min_variance / min_point_biserial: the discrimination floors (forwarded to
            :func:`irt.prune_items`).

    Returns:
        A :class:`CurationResult` with the surviving curated ids + every drop reason.
    """
    by_id = {it.id: it for it in items}
    kept, saturated, low_disc, dropped_ragged = select_discriminators(
        grade_matrix, min_variance=min_variance, min_point_biserial=min_point_biserial
    )
    av = ambiguity_verdicts or {}
    final: list[str] = []
    dropped_amb: list[str] = []
    dropped_mis: list[str] = []
    not_ver: list[str] = []
    for iid in kept:
        item = by_id.get(iid)
        choices = av.get(iid)
        if item is None or not choices:
            not_ver.append(iid)
            final.append(iid)
            continue
        verdict = ambiguity_gate(item, choices)
        if verdict.status is AmbiguityStatus.DEFENSIBLE:
            final.append(iid)
        elif verdict.status is AmbiguityStatus.AMBIGUOUS:
            dropped_amb.append(iid)
        elif verdict.status is AmbiguityStatus.MISLABELED:
            dropped_mis.append(iid)
        else:  # NOT_VERIFIED (e.g. <2 choices) — keep but flag
            not_ver.append(iid)
            final.append(iid)
    notes = [
        f"{len(final)} curated discriminator(s) kept from {len(by_id)} pool item(s)",
        f"discrimination screen dropped {len(saturated)} saturated + {len(low_disc)} "
        "low-discrimination (AFLite/§12)",
        f"ambiguity gate dropped {len(dropped_amb)} ambiguous + {len(dropped_mis)} mislabeled; "
        f"{len(not_ver)} kept un-verified (no verifier verdicts supplied)",
    ]
    if dropped_ragged:
        notes.append(
            f"grade-matrix was RAGGED — {len(dropped_ragged)} item(s) not scored by every model "
            f"(per-item salvage) degraded to the shared subset: {dropped_ragged}"
        )
    return CurationResult(
        kept=final,
        dropped_saturated=saturated,
        dropped_low_discrimination=low_disc,
        dropped_ambiguous=dropped_amb,
        dropped_mislabeled=dropped_mis,
        not_verified=not_ver,
        dropped_ragged=dropped_ragged,
        notes=notes,
    )
