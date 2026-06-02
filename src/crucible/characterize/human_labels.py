"""Human alt-test label loader (research-grounding §12.1, Fork C).

Retires the **circular model-jury ω** by loading REAL human annotator labels and building
the ``records_per_annotator`` dict the audit-ready alt-test
(:func:`crucible.characterize.metrics.alt_test`) consumes. The model-jury bootstrap drew
its "annotators" from the same population being seated (Panickssery 2024 self-preference,
~20–40pp — see ``memory/crucible.md``); humans are an INDEPENDENT reference, so ω stops
being circular.

Grounding (study-swarm, citation-verified §12.1):

* **Schema is the un-aggregated per-annotator matrix** ``item_id → {annotator_id: verdict}``
  — the alt-test is aggregation-free by construction, so we must NOT collapse to one gold
  (Davani et al. 2022, arXiv:2110.05719: per-annotator labels lose no accuracy; Calderon
  2025, arXiv:2501.10970: leave-one-out compares against held-out INDIVIDUALS).
* **≥3 annotators** (Calderon FAQ: two degenerates the leave-one-out), **≥30 items** (the
  t-test normality floor; below it the metric is under-powered — surfaced as a loud note,
  honest-surface constraint #4).
* **Per-tier ε** (Calderon §B.1): 0.2 expert / 0.15 skilled / 0.1 crowd. We take the most
  conservative (smallest) ε across the declared tiers; if inter-annotator agreement is
  low we clamp ε ≤ 0.1 and flag "add items" (Calderon §B.2).
* **DISPUTED items are dropped from ω, never force-resolved** (Plank 2022, arXiv:2211.02570;
  Aroyo & Welty 2015: high disagreement marks ambiguous items, not annotator error).
* **A genuine "unsure"/tie is its own outcome, not coerced to a side** (Krosnick & Presser
  2010: forced binary manufactures ~14% acquiescence) — recorded as no-label for that
  (annotator, item) so it is simply absent from that fold.

The verdict→0/1 mapping is IDENTICAL to ``crucible.characterize.run._to_num`` (A/PASS → 1,
else → 0) so human records live in the SAME space as the judge's records and every existing
metric primitive runs on them unchanged.

MACE / Dawid-Skene competence-weighted pool QC (Hovy 2013; Dawid & Skene 1979) is a
documented LAZY-optional path (parallel to the Bayesian-IRT hook in ``calibration/irt.py``)
— the per-annotator alt-test needs no aggregation and must work without it; see §12.1.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from crucible.calibration.types import CalibrationItem
from crucible.characterize import metrics as M
from crucible.characterize.types import JudgmentRecord

__all__ = [
    "HumanLabelError",
    "HumanLabels",
    "load_human_labels",
    "build_records_per_annotator",
    "TIER_EPSILON",
    "MIN_ANNOTATORS",
    "MIN_ITEMS",
    "DISPUTE_FLOOR",
]

#: Per-tier substitution tolerance ε (Calderon, Reichart & Dror 2025, arXiv:2501.10970 §B.1).
TIER_EPSILON: dict[str, float] = {"expert": 0.2, "skilled": 0.15, "crowd": 0.1}
#: Leave-one-out needs ≥3 humans (the FAQ: two degenerates to "aligns more with A or B?").
MIN_ANNOTATORS = 3
#: t-test normality floor; below it the alt-test is under-powered (recommend Wilcoxon).
MIN_ITEMS = 30
#: Per-item human agreement below this → DISPUTED (excluded from the ω denominator).
DISPUTE_FLOOR = 0.67
#: Krippendorff α below this is "insufficient" reliability → clamp ε ≤ 0.1, flag add-items.
_ALPHA_INSUFFICIENT = 0.667
#: Verdicts that mean "no usable label" (a genuine tie/unsure, never coerced to a side).
_UNSURE = frozenset({"", "TIE", "UNSURE", "?", "NEITHER", "BOTH", "NA", "NONE"})
#: The only verdicts that map to a side (A/PASS → 1, B/FAIL → 0). Anything else is a typo,
#: rejected rather than silently coerced to 0 (re-audit fork-c-integration INFO).
_KNOWN_VERDICTS = frozenset({"A", "B", "PASS", "FAIL"})


class HumanLabelError(Exception):
    """Raised when a human-label source is missing or malformed.

    Carries the Ship-Gate-B structured shape (``[CODE] message (hint: ...)``) so a caller
    surfaces an actionable, redacted error without a raw stack (mirrors
    :class:`crucible.calibration.loader.CalibrationLoadError`).
    """


def _fail(code: str, message: str, hint: str) -> HumanLabelError:
    return HumanLabelError(f"[{code}] {message} (hint: {hint})")


def _to_num(verdict: str) -> int:
    """Map a categorical verdict to 0/1 — IDENTICAL to ``run._to_num`` (A/PASS → 1)."""
    return 1 if str(verdict).upper() in ("A", "PASS") else 0


@dataclass(slots=True)
class HumanLabels:
    """Loaded, validated human alt-test labels (Fork C, §12.1).

    Attributes:
        records_per_annotator: ``{annotator_id: [JudgmentRecord, ...]}`` — the per-annotator
            matrix, NO ``"judge"`` key yet (added by :func:`build_records_per_annotator`).
        annotators: ``{annotator_id: {"tier": ...}}`` roster.
        epsilon: the substitution tolerance to feed :func:`metrics.alt_test` (most
            conservative tier ε, clamped ≤0.1 when IAA is insufficient).
        iaa_alpha: human–human Krippendorff α (the κ-z baseline and the reported IAA).
        item_ids: every item with ≥1 usable human label.
        disputed: items with human agreement below :data:`DISPUTE_FLOOR` (dropped from ω).
        notes: honest-surface warnings (too few items, low IAA + ε clamp, disputed drops).
    """

    records_per_annotator: dict[str, list[JudgmentRecord]]
    annotators: dict[str, dict[str, Any]]
    epsilon: float
    iaa_alpha: float
    item_ids: list[str]
    disputed: list[str]
    notes: list[str] = field(default_factory=list)

    @property
    def n_annotators(self) -> int:
        return len(self.records_per_annotator)

    @property
    def n_items(self) -> int:
        return len(self.item_ids)


def load_human_labels(path: Path | str, items: list[CalibrationItem]) -> HumanLabels:
    """Load + validate ``human_labels.json`` against the calibration ``items``.

    Args:
        path: the ``human_labels.json`` file. Shape::

            {
              "schema_version": 1,
              "annotators": {"<id>": {"tier": "expert|skilled|crowd"}},
              "labels": {"<item_id>": {"<annotator_id>": "A"|"B"|"unsure", ...}}
            }

        items: the calibration items the labels are over (provides ``gold`` and the
            valid id set; ``gold`` is grading-side and never shown to annotators).

    Returns:
        A validated :class:`HumanLabels`.

    Raises:
        HumanLabelError: path missing / bad JSON / wrong shape; unknown ``tier``; a label
            referencing an unknown ``item_id`` or undeclared annotator; or fewer than
            :data:`MIN_ANNOTATORS` annotators with usable labels.
    """
    root = Path(path)
    if not root.is_file():
        raise _fail(
            "INPUT_PATH_MISSING",
            f"human-labels path does not exist: {root}",
            "pass a path to a human_labels.json file",
        )
    try:
        raw = json.loads(root.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise _fail("INPUT_BAD_JSON", f"{root.name} is not valid JSON: {exc}",
                    f"fix the JSON syntax in {root.name}") from exc
    if not isinstance(raw, dict):
        raise _fail("INPUT_JSON_SHAPE", f"{root.name} top-level JSON is not an object",
                    "the file is one object with 'annotators' and 'labels'")

    gold_by_id = {it.id: it.gold for it in items}
    annotators = _coerce_annotators(raw.get("annotators"))
    labels = raw.get("labels")
    if not isinstance(labels, dict) or not labels:
        raise _fail("INPUT_FIELD_MISSING", f"{root.name} has no non-empty 'labels' object",
                    "map each item_id to {annotator_id: verdict}")

    rpa: dict[str, list[JudgmentRecord]] = {a: [] for a in annotators}
    item_ids: list[str] = []
    disputed: list[str] = []
    for item_id, by_annotator in labels.items():
        if item_id not in gold_by_id:
            raise _fail(
                "INPUT_UNKNOWN_ITEM",
                f"label references item {item_id!r} not in the calibration set",
                "every labeled item_id must exist in the loaded calibration items",
            )
        if not isinstance(by_annotator, dict) or not by_annotator:
            raise _fail("CONFIG_FIELD_TYPE", f"labels[{item_id!r}] must be a non-empty object",
                        "map annotator_id -> verdict for the item")
        gold_num = _to_num(str(gold_by_id[item_id]))
        usable: list[int] = []
        for ann, verdict in by_annotator.items():
            if ann not in annotators:
                raise _fail(
                    "CONFIG_UNKNOWN_ANNOTATOR",
                    f"labels[{item_id!r}] names annotator {ann!r} not in 'annotators'",
                    "declare every annotator (with a tier) in the 'annotators' object",
                )
            v_up = str(verdict).strip().upper()
            if v_up in _UNSURE:
                continue  # genuine tie/unsure → no label for this (annotator, item)
            if v_up not in _KNOWN_VERDICTS:
                raise _fail(
                    "CONFIG_BAD_VERDICT",
                    f"labels[{item_id!r}][{ann!r}] verdict {verdict!r} is not a known label",
                    "use 'A'/'B' (or 'PASS'/'FAIL'); a tie/unsure marker; never a typo "
                    "(typos would be silently coerced to one side)",
                )
            pred = _to_num(v_up)
            usable.append(pred)
            rpa[ann].append(
                JudgmentRecord(
                    item_id=item_id, model_id=ann, predicted=pred, gold=gold_num,
                    metadata={"annotator_tier": annotators[ann]["tier"]},
                )
            )
        if usable:
            item_ids.append(item_id)
            # per-item agreement = modal share among the usable human labels.
            agree = max(usable.count(0), usable.count(1)) / len(usable)
            if agree < DISPUTE_FLOOR:
                disputed.append(item_id)

    active = {a: recs for a, recs in rpa.items() if recs}
    if len(active) < MIN_ANNOTATORS:
        raise _fail(
            "CONFIG_TOO_FEW_ANNOTATORS",
            f"only {len(active)} annotator(s) have usable labels; alt_test needs "
            f">= {MIN_ANNOTATORS} (Calderon 2025)",
            "supply labels from at least 3 distinct human annotators",
        )

    iaa = M.krippendorff_alpha(active)
    epsilon = min(TIER_EPSILON[annotators[a]["tier"]] for a in active)
    notes: list[str] = []
    if iaa < _ALPHA_INSUFFICIENT and epsilon > 0.1:
        notes.append(
            f"human IAA Krippendorff α={iaa:.3f} < {_ALPHA_INSUFFICIENT} (insufficient) "
            f"→ ε clamped {epsilon:.2f}→0.10 and 'increase N_items' flagged (Calderon §B.2)"
        )
        epsilon = 0.1
    if len(item_ids) < MIN_ITEMS:
        notes.append(
            f"only {len(item_ids)} labeled items < {MIN_ITEMS} (t-test normality floor) — "
            "ω is under-powered; prefer a Wilcoxon variant or add items (Calderon)"
        )
    if disputed:
        notes.append(
            f"{len(disputed)} DISPUTED item(s) (human agreement < {DISPUTE_FLOOR}) "
            "EXCLUDED from the ω denominator, not force-resolved (Plank 2022)"
        )

    return HumanLabels(
        records_per_annotator=active,
        annotators={a: annotators[a] for a in active},
        epsilon=epsilon,
        iaa_alpha=iaa,
        item_ids=sorted(item_ids),
        disputed=sorted(disputed),
        notes=notes,
    )


def _coerce_annotators(value: Any) -> dict[str, dict[str, Any]]:
    if not isinstance(value, dict) or not value:
        raise _fail("INPUT_FIELD_MISSING", "human-labels file has no non-empty 'annotators'",
                    "map annotator_id -> {\"tier\": \"expert|skilled|crowd\"}")
    out: dict[str, dict[str, Any]] = {}
    for ann, meta in value.items():
        if not isinstance(meta, dict):
            raise _fail("CONFIG_FIELD_TYPE", f"annotators[{ann!r}] must be an object",
                        "give each annotator at least a 'tier'")
        tier = meta.get("tier")
        if tier not in TIER_EPSILON:
            raise _fail(
                "CONFIG_UNKNOWN_TIER",
                f"annotators[{ann!r}] tier {tier!r} is not one of {sorted(TIER_EPSILON)}",
                "tier must be 'expert', 'skilled', or 'crowd' (Calderon ε tiers)",
            )
        out[ann] = dict(meta)
    return out


def build_records_per_annotator(
    human_labels: HumanLabels, judge_records: list[JudgmentRecord]
) -> dict[str, list[JudgmentRecord]]:
    """Assemble the ``records_per_annotator`` for :func:`metrics.alt_test`.

    Puts the candidate judge's own records under the reserved ``"judge"`` key and the
    human annotators as the peers — the NON-circular swap Fork C exists for. The
    DISPUTED items are left in the records; :func:`metrics.alt_test` is told to exclude
    them via ``exclude_items`` so the same matrix can be reported with and without them.
    """
    return {"judge": judge_records, **human_labels.records_per_annotator}
