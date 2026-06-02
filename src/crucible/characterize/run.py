"""Characterization runner (research-grounding §11/§12) — the real-run driver.

Integration glue (coordinator-owned): calibration items → judge prompts → model
judgments (via the model adapters, sequential/deterministic serving) →
``JudgmentRecord``s → ``build_profile`` → ``JudgeProfile``; plus the known-groups
instrument check, the panel submodularity (ρ) analysis, the §12 IRT item-prune
screen, and the §12 ship-time perturbation audit.

This module does the *judge-admission* run. Each calibration item is a complete
grading task whose ``gold`` is the correct answer. Two item shapes are supported
and parsed in the **expected answer space** (§12):

* **A/B pairs** (``gold`` ∈ ``{"A", "B"}``) — the discriminating shape: two
  candidate answers, exactly one subtly wrong (JudgeBench, §12 Q1). The default
  ``admission_pairs.json`` set is all pairs.
* **PASS/FAIL verdicts** (``gold`` ∈ ``{"PASS", "FAIL"}``) — the legacy
  ``admission_set.json`` shape, still loadable via ``--items``.

The alt-test (§11.1 #3) runs here as a **model-jury bootstrap** (a director
decision, documented caveat): the *other* panel models stand in as "annotators".
This is **circular** — the reference is drawn from the same population being
seated — so a valid alt-test still needs ≥3 *human* annotators (Calderon,
Reichart & Dror 2025, arXiv:2501.10970). The bootstrap ω is reported and gates
seating only as an *outlier detector* (a judge no worse than its peers); the
report carries the caveat and the seat decisions are PROVISIONAL until human
labels exist. See ``alt_test_caveat`` in the emitted report and §12 Q3.

Usage:
    uv run python -m crucible.characterize.run --k 3 --out report.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from crucible.calibration import irt
from crucible.calibration.loader import load_items
from crucible.calibration.types import CalibrationCategory, CalibrationItem
from crucible.characterize import aggregate
from crucible.characterize.human_labels import (
    HumanLabels,
    build_records_per_annotator,
    load_human_labels,
)
from crucible.characterize.metrics import temperature_scaled_ece_cv
from crucible.characterize.panel_store import save_panel
from crucible.characterize.profile import build_profile, perturbation_audit
from crucible.characterize.types import JudgeProfile, JudgmentRecord, RoleSlot
from crucible.models.ollama_adapter import OllamaModel

_VERDICT_RE = re.compile(r"\b(PASS|FAIL)\b", re.IGNORECASE)
_CHOICE_RE = re.compile(r"\b([AB])\b")  # case-sensitive: the prompt asks for one letter

#: The standing caveat stamped on every report: the alt-test ω here is a circular
#: model-jury bootstrap, not a human-grounded substitution test (§12 Q3).
_ALT_TEST_CAVEAT = (
    "alt-test ω is a MODEL-JURY BOOTSTRAP (a documented director decision). The "
    "reference 'annotators' are the other panel models, drawn from the SAME population "
    "being seated, so ω is CIRCULAR. A valid alt-test needs >=3 HUMAN annotators on "
    ">=30 items (Calderon, Reichart & Dror 2025, arXiv:2501.10970). Here ω functions "
    "as an OUTLIER DETECTOR (a judge that agrees with its peers no worse than peers "
    "agree with each other), NOT a substitution guarantee. Seat decisions are "
    "PROVISIONAL until human labels exist (§12 Q3)."
)

#: The Fork-C note stamped when REAL human labels are supplied (``--human-labels``): ω is
#: the non-circular, audit-ready substitution test, so the circular caveat is retired.
def _alt_test_human_note(hl: HumanLabels) -> str:
    return (
        f"alt-test ω is HUMAN-GROUNDED (Fork C, §12.1): {hl.n_annotators} independent human "
        f"annotators over {hl.n_items} items, ε={hl.epsilon:.2f}, computed by the audit-ready "
        "procedure (per-annotator one-sided paired t-test of H0: ρ_judge ≤ ρ_human − ε, "
        "Benjamini-Yekutieli FDR q=0.05, ω = fraction of rejected nulls; Calderon, Reichart "
        f"& Dror 2025, arXiv:2501.10970 §3). Human–human Krippendorff α={hl.iaa_alpha:.3f}. "
        "The reference is NO LONGER the seated model population, so ω is NOT circular — the "
        "model-jury caveat is retired for this run."
    )

# Default local panel (cross-family) — already pulled on the Omen. (model_id, family, quant)
DEFAULT_PANEL: list[tuple[str, str, str | None]] = [
    ("qwen3.6:27b", "qwen", None),
    ("mistral-small:24b", "mistral", None),
    ("gemma4:31b", "gemma", None),
    ("aya-expanse:32b", "cohere", None),
    ("granite4.1:30b", "granite", None),
    ("devstral-small-2:24b", "mistral-devstral", None),
]


def parse_verdict(text: str) -> str | None:
    """Pull a PASS/FAIL verdict from raw model output (first occurrence)."""
    m = _VERDICT_RE.search(text or "")
    return m.group(1).upper() if m else None


def parse_choice(text: str, gold: str) -> str | None:
    """Parse a judgment from raw output in the **expected answer space** (§12).

    The space is chosen by ``gold``: an A/B pair item is parsed for a standalone
    ``A``/``B`` letter (case-sensitive — the prompt asks for exactly one letter, so a
    stray lowercase article never masquerades as a choice); any other item is parsed
    as a PASS/FAIL verdict. Returns the upper-cased token or ``None`` if absent.
    """
    t = text or ""
    if str(gold).upper() in ("A", "B"):
        m = _CHOICE_RE.search(t)
        return m.group(1) if m else None
    return parse_verdict(t)


def _to_num(choice: str | None) -> int:
    """Map a categorical judgment to 0/1 for agreement/κ. PASS/A → 1; FAIL/B/None → 0."""
    return 1 if choice in ("PASS", "A") else 0


async def collect_records(
    model: OllamaModel, items: list[CalibrationItem], *, k: int = 3
) -> list[JudgmentRecord]:
    """Run one model over the calibration items (k reruns), returning graded records.

    Each record is stamped with the **authored** ``item.id`` (NOT the adapter's
    prompt-hash fallback) so every grouping metric — known-groups, consistency,
    panel ρ-correlation, IRT prune, alt-test — aligns across models and against the
    authored set. ``predicted``/``gold`` are mapped to 0/1, ``correct`` is the
    exact-match on the parsed token, and ``metadata`` carries the category + the
    item difficulty (the §12 difficulty weighting) + the raw parsed token.
    """
    records: list[JudgmentRecord] = []
    for item in items:
        gold_str = str(item.gold).upper()
        gold_num = _to_num(gold_str)
        for ri in range(k):
            rec = await model.judge_item(item.prompt, run_index=ri)
            rec.item_id = item.id  # authored id, not the prompt-hash fallback (§11.3)
            parsed = parse_choice(str(rec.predicted), gold_str)
            pred_num = _to_num(parsed) if parsed is not None else (1 - gold_num)
            rec.predicted = pred_num
            rec.gold = gold_num
            rec.correct = parsed == gold_str
            rec.metadata["category"] = item.category.value
            rec.metadata["raw_choice"] = parsed
            if item.difficulty is not None:
                rec.metadata["difficulty"] = item.difficulty
            records.append(rec)
    return records


def _evict(model_id: str) -> None:
    """Free VRAM after a model's run (keep_alive=0 unloads it)."""
    try:
        import httpx

        httpx.post(
            "http://localhost:11434/api/generate",
            json={"model": model_id, "keep_alive": 0},
            timeout=30,
        )
    except Exception:
        pass


def _jury(
    all_records: dict[str, list[JudgmentRecord]],
    model_id: str,
    recs: list[JudgmentRecord],
) -> dict[str, list[JudgmentRecord]] | None:
    """Build the model-jury bootstrap ``records_per_annotator`` for ``model_id`` (§12 Q3).

    The candidate's own records go under the reserved ``"judge"`` key
    (:func:`crucible.characterize.metrics.alt_test_omega`); the OTHER panel models are
    the "annotators". Returns ``None`` when fewer than two peers exist (leave-one-out
    needs ≥2), so :func:`build_profile` records "alt-test not measured" rather than
    raising. CIRCULAR by construction — see :data:`_ALT_TEST_CAVEAT`.
    """
    peers = {oid: orecs for oid, orecs in all_records.items() if oid != model_id}
    if len(peers) < 2:
        return None
    return {"judge": recs, **peers}


async def run_panel(
    panel: list[tuple[str, str, str | None]],
    items: list[CalibrationItem],
    *,
    k: int = 3,
    human_labels: HumanLabels | None = None,
) -> tuple[dict[str, JudgeProfile], dict[str, list[JudgmentRecord]]]:
    """Sequential load → judge → evict, then profile with the alt-test (§11.2/§12).

    Two passes: (1) judge the whole panel one model at a time (VRAM-respecting —
    ``OLLAMA_NUM_PARALLEL=1``, evict between models); (2) build each model's profile.
    Pass 2 needs the full record set first, which is why profiling is deferred.

    The alt-test reference (pass 2) is chosen by ``human_labels``:

    * **``None``** — the circular model-jury bootstrap (the *other* panel models stand in
      as annotators; ω is the PROXY :func:`metrics.alt_test_omega`; the loud caveat holds).
    * **supplied** — the NON-circular, audit-ready human alt-test (Fork C §12.1): each
      judge is profiled against the REAL human annotators via :func:`metrics.alt_test`
      (per-tier ε + paired t-test + BY-FDR), DISPUTED items excluded, and the κ-z baseline
      is the measured human–human Krippendorff α.
    """
    os.environ.setdefault("OLLAMA_NUM_PARALLEL", "1")
    os.environ.setdefault("OLLAMA_MAX_LOADED_MODELS", "1")

    all_records: dict[str, list[JudgmentRecord]] = {}
    quant_by_id: dict[str, str | None] = {}
    for model_id, family, quant in panel:
        quant_by_id[model_id] = quant
        model = OllamaModel(model_id=model_id, family=family, quant=quant)
        t0 = time.monotonic()
        try:
            recs = await collect_records(model, items, k=k)
        except Exception as exc:  # noqa: BLE001 — one bad model must not sink the panel
            print(f"[{model_id}] ERROR: {exc!r}")
            _evict(model_id)
            continue
        elapsed = time.monotonic() - t0
        all_records[model_id] = recs
        acc = sum(1 for r in recs if r.correct) / len(recs) if recs else 0.0
        print(f"[{model_id}] judged {len(recs)} records  acc={acc:.3f}  ({elapsed:.0f}s)")
        _evict(model_id)

    disputed = set(human_labels.disputed) if human_labels else None
    profiles: dict[str, JudgeProfile] = {}
    for model_id, recs in all_records.items():
        if human_labels is not None:
            profile = build_profile(
                model_id,
                RoleSlot.JUDGE,
                recs,
                records_per_annotator=build_records_per_annotator(human_labels, recs),
                human_grounded=True,
                alt_test_epsilon=human_labels.epsilon,
                alt_test_exclude=disputed,
                human_human_kappa=human_labels.iaa_alpha,
                quant=quant_by_id.get(model_id),
            )
            # Surface the human-set caveats (n<30 under-power, low-IAA ε clamp, DISPUTED
            # drops) in the profile itself, where the seat is justified — not only in the
            # run report (re-audit fork-c-integration LOW).
            profile.notes.extend(f"[human-set] {note}" for note in human_labels.notes)
        else:
            profile = build_profile(
                model_id,
                RoleSlot.JUDGE,
                recs,
                records_per_annotator=_jury(all_records, model_id, recs),
                quant=quant_by_id.get(model_id),
            )
        profiles[model_id] = profile
        flag = " [review]" if profile.metadata.get("review_flag") else ""
        q = profile.metadata.get("quality_score")
        print(
            f"[{model_id}] {profile.seat_decision.value}{flag}  acc={profile.objective_accuracy}  "
            f"q={q}  ece={profile.ece}  omega={profile.alt_test_omega}"
        )
    return profiles, all_records


def known_groups_report(
    items: list[CalibrationItem], records: dict[str, list[JudgmentRecord]]
) -> dict[str, Any]:
    """Instrument validation: on trivial-anchor items every model should be correct;
    a miss is an instrument/model-fault flag (§11.3).

    A *trivial anchor* is any item the weakest tier is expected to pass — either a
    ``KNOWN_TRIVIAL`` item or any item whose ``expected_pass["weak"]`` is ``True``
    (the pairs set declares its easy anchors this way rather than by category).
    Records carry the **authored** ``item.id`` (see :func:`collect_records`), so the
    comparison is real — the prompt-hash fallback would have made this check vacuous.
    """
    trivial_ids = {
        i.id
        for i in items
        if i.category == CalibrationCategory.KNOWN_TRIVIAL or i.expected_pass.get("weak") is True
    }
    flags: list[str] = []
    for model_id, recs in records.items():
        misses = {r.item_id for r in recs if r.item_id in trivial_ids and not r.correct}
        if misses:
            flags.append(f"{model_id} missed trivial items: {sorted(misses)}")
    return {"trivial_item_count": len(trivial_ids), "violations": flags, "passed": not flags}


def panel_correlation_report(
    records: dict[str, list[JudgmentRecord]],
) -> dict[str, Any]:
    """ρ<0.25 submodularity analysis over per-item error vectors (§11.4).

    Passes the records dict straight to :func:`aggregate.pairwise_error_correlation`,
    which builds each judge's error vector internally (the earlier glue built the
    vectors itself and passed the wrong shape — the ``'str' has no attribute
    'predicted'`` bug; fixed here).
    """
    if len(records) < 2:
        return {"note": "fewer than two judges; submodularity vacuous", "submodular": True}
    try:
        corr = aggregate.pairwise_error_correlation(records)
        ok = aggregate.passes_submodularity(corr)
        flat = {f"{a}|{b}": round(c, 4) for (a, b), c in corr.items()}
        return {"pairwise_error_correlation": flat, "submodular": ok}
    except Exception as exc:  # noqa: BLE001
        return {"error": repr(exc)}


def _grade_matrix(records: dict[str, list[JudgmentRecord]]) -> dict[str, dict[str, bool]]:
    """Collapse k reruns to one ``{model_id: {item_id: majority_correct}}`` grid."""
    matrix: dict[str, dict[str, bool]] = {}
    for model_id, recs in records.items():
        by_item: dict[str, list[bool]] = {}
        for r in recs:
            by_item.setdefault(r.item_id, []).append(bool(r.correct))
        matrix[model_id] = {iid: (sum(v) / len(v)) >= 0.5 for iid, v in by_item.items()}
    return matrix


def irt_prune_report(records: dict[str, list[JudgmentRecord]]) -> dict[str, Any]:
    """Model-free IRT item screen — §12 Q1 (ATLAS): drop saturated / non-discriminating items.

    Drops an item when its panel verdict has zero variance (saturated — every model
    agrees, the exact failure the first run hit) or its point-biserial r_pb<0.1 (right/
    wrong doesn't track ability). Reports ``(kept, dropped)`` so the next pilot can
    retire the dead items.
    """
    matrix = _grade_matrix(records)
    try:
        kept, dropped = irt.prune_items(matrix, min_variance=0.0, min_point_biserial=0.1)
        return {
            "n_items": len(kept) + len(dropped),
            "kept_count": len(kept),
            "dropped_count": len(dropped),
            "kept": kept,
            "dropped": dropped,
            "note": "ATLAS screen: drop zero-variance (saturated) or low-r_pb items",
        }
    except Exception as exc:  # noqa: BLE001
        return {"error": repr(exc)}


def perturbation_report(records: dict[str, list[JudgmentRecord]]) -> dict[str, Any]:
    """Ship-time perturbation audit — §12 Q4 / §8.3 (Alzahrani 2024 single-perturbation).

    Jitter each admission threshold ±1 SE and report the per-model decision-flip rate;
    a robust seat/screen/reject should not flip within a threshold's own noise. Surfaces
    ``max_flip_rate`` as the andon signal — a high flip rate means the gate is balanced
    on a knife-edge and the decision is not yet trustworthy.
    """
    out: dict[str, Any] = {}
    max_flip = 0.0
    for model_id, recs in records.items():
        rpa = _jury(records, model_id, recs)
        try:
            audit = perturbation_audit(recs, records_per_annotator=rpa)
            out[model_id] = audit
            max_flip = max(max_flip, float(audit.get("flip_rate", 0.0)))
        except Exception as exc:  # noqa: BLE001
            out[model_id] = {"error": repr(exc)}
    return {
        "max_flip_rate": round(max_flip, 4),
        "by_model": out,
        "andon": "high max_flip_rate → gate sits within threshold noise; decisions not yet robust",
    }


def calibration_report(records: dict[str, list[JudgmentRecord]]) -> dict[str, Any]:
    """Per-model post-hoc temperature scaling — §12 Q3 (Guo et al. 2017).

    Reports each judge's mean fitted temperature + ECE before vs after, measured
    **held-out** (k-fold grouped by item, so test-retest reruns never leak across the
    fit/measure split). ``ece_cv`` is therefore the out-of-sample ECE temperature scaling
    actually buys — not the optimistic in-sample number.
    """
    out: dict[str, Any] = {}
    for model_id, recs in records.items():
        try:
            temp, raw, cv = temperature_scaled_ece_cv(recs)
        except Exception as exc:  # noqa: BLE001
            out[model_id] = {"error": repr(exc)}
            continue
        out[model_id] = {
            "mean_temperature": round(temp, 4),
            "ece_raw": round(raw, 4) if raw is not None else None,
            "ece_cv": round(cv, 4) if cv is not None else None,
        }
    return {
        "by_model": out,
        "note": "temperature scaling (Guo 2017) measured HELD-OUT: k-fold grouped by item",
    }


def panel_composition_report(
    profiles: dict[str, JudgeProfile], records: dict[str, list[JudgmentRecord]]
) -> dict[str, Any]:
    """The composed seated panel — §11.4 (the instrument config the run produces).

    Turns the per-model profiles into the final ρ-pruned, reliability-weighted,
    quorum-checked panel crucible would score with (:func:`aggregate.compose_panel`).
    Below quorum the panel escalates to the Claude Designer rather than seating thin.
    """
    try:
        return asdict(aggregate.compose_panel(profiles, records))
    except Exception as exc:  # noqa: BLE001
        return {"error": repr(exc)}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Crucible judge-admission characterization run.")
    ap.add_argument("--items", type=Path, default=None, help="items file/dir (default: pairs set)")
    ap.add_argument("--k", type=int, default=3, help="reruns per item (test-retest)")
    ap.add_argument("--models", nargs="*", default=None, help="model_id@family specs")
    ap.add_argument("--out", type=Path, default=Path("characterization-report.json"))
    ap.add_argument(
        "--write-panel",
        type=Path,
        default=None,
        help="also write the composed seated panel to this path (the committed artifact)",
    )
    ap.add_argument(
        "--human-labels",
        type=Path,
        default=None,
        help="human_labels.json — REAL annotators for a non-circular alt-test ω (Fork C, §12.1)",
    )
    args = ap.parse_args(argv)

    items = load_items(args.items) if args.items else _load_admission_items()
    panel = _parse_models(args.models) if args.models else DEFAULT_PANEL
    human_labels = load_human_labels(args.human_labels, items) if args.human_labels else None

    profiles, records = asyncio.run(run_panel(panel, items, k=args.k, human_labels=human_labels))

    report = {
        "n_items": len(items),
        "k": args.k,
        "item_set": args.items.name if args.items else "admission_pairs.json",
        "alt_test_reference": "human" if human_labels else "model-jury-bootstrap",
        "alt_test_caveat": _alt_test_human_note(human_labels) if human_labels else _ALT_TEST_CAVEAT,
        "human_alt_test": (
            {
                "n_annotators": human_labels.n_annotators,
                "n_items_labeled": human_labels.n_items,
                "epsilon": human_labels.epsilon,
                "iaa_krippendorff_alpha": round(human_labels.iaa_alpha, 4),
                "disputed_items": human_labels.disputed,
                "notes": human_labels.notes,
            }
            if human_labels
            else None
        ),
        "profiles": {m: asdict(p) for m, p in profiles.items()},
        "known_groups": known_groups_report(items, records),
        "panel_correlation": panel_correlation_report(records),
        "irt_prune": irt_prune_report(records),
        "perturbation": perturbation_report(records),
        "calibration": calibration_report(records),
        "panel_composition": panel_composition_report(profiles, records),
    }
    args.out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    seated = [m for m, p in profiles.items() if p.seat_decision.value == "seat"]
    print(f"\nseated: {seated}")
    comp = report["panel_composition"]
    if "error" not in comp:
        panel = [(s["model_id"], round(s["reliability_weight"], 3)) for s in comp["seats"]]
        verdict = "escalate (sub-quorum)" if comp["escalate"] else "quorum met"
        print(f"composed panel ({verdict}): {panel}")
    if args.write_panel is not None:
        try:
            save_panel(aggregate.compose_panel(profiles, records), args.write_panel)
            print(f"panel artifact -> {args.write_panel}")
        except Exception as exc:  # noqa: BLE001 — artifact write must not fail the run
            print(f"panel artifact NOT written: {exc!r}")
    print(f"report -> {args.out}")
    return 0


def _parse_models(specs: list[str]) -> list[tuple[str, str, str | None]]:
    """Parse ``model_id@family`` specs. The model_id itself may contain a ':'
    (e.g. ``qwen3.6:27b``), so split on the LAST '@' only."""
    out: list[tuple[str, str, str | None]] = []
    for s in specs:
        if "@" in s:
            mid, fam = s.rsplit("@", 1)
            out.append((mid, fam, None))
        else:
            out.append((s, "unknown", None))
    return out


def _load_admission_items() -> list[CalibrationItem]:
    """Load the bundled judge-admission **pairs** set (§12 Q1 — the discriminating shape)."""
    path = Path(__file__).parent.parent / "calibration" / "admission_pairs.json"
    return load_items(path)


if __name__ == "__main__":
    raise SystemExit(main())
