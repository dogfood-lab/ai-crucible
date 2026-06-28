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
    uv run python -m ai_crucible.characterize.run --k 3 --out report.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from ai_crucible.calibration import irt
from ai_crucible.calibration.loader import load_items
from ai_crucible.calibration.types import CalibrationCategory, CalibrationItem
from ai_crucible.characterize import aggregate
from ai_crucible.characterize.human_labels import (
    HumanLabels,
    build_records_per_annotator,
    load_human_labels,
)
from ai_crucible.characterize.metrics import temperature_scaled_ece_cv
from ai_crucible.characterize.panel_store import save_panel
from ai_crucible.characterize.profile import build_profile, perturbation_audit
from ai_crucible.characterize.types import (
    JudgeProfile,
    JudgmentRecord,
    RoleSlot,
    SeatDecision,
)
from ai_crucible.models.ollama_adapter import OllamaModel
from ai_crucible.models.openrouter_adapter import OpenRouterModel

_VERDICT_RE = re.compile(r"\b(PASS|FAIL)\b", re.IGNORECASE)
_CHOICE_RE = re.compile(r"\b([AB])\b")  # case-sensitive: the prompt asks for one letter

#: The hardcoded human–human κ baseline the default (no-human-labels) path falls back to —
#: mirrors :func:`ai_crucible.characterize.profile.build_profile`'s ``human_human_kappa``
#: default. Surfaced in the report's ``kappa_baseline`` block as ``provisional=True`` because
#: it is FABRICATED (no humans in the model-jury path) — characterize-002.
_DEFAULT_HUMAN_KAPPA = 0.80

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

# Default local panel (cross-family) — already pulled on the Omen. (model_id, family, quant).
# ``family`` is ``str | None``: the DEFAULT_PANEL is fully tagged, but the ``--models`` CLI
# path may stamp ``None`` for an untagged entry (models-cli-001 / SHARED FAMILY CONTRACT).
DEFAULT_PANEL: list[tuple[str, str | None, str | None]] = [
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


#: Per-item wall-clock budget around a single ``judge_item`` call (characterize-degradation-005).
#: A single hung Ollama call (transient daemon stall during a model swap) must not stall the
#: whole expensive run; this caps it well under the adapter's transport-level 600s. PIN_PER_STEP:
#: the effective budget is stamped into each record's ``metadata["per_item_timeout_s"]``.
_PER_ITEM_TIMEOUT_S = 180.0

#: After this many CONSECUTIVE per-item failures, ``collect_records`` stops salvaging and
#: re-raises (characterize-degradation-002): a sustained run of failures genuinely signals the
#: daemon is down / the tag is unservable, so the whole-model drop in :func:`run_panel` is the
#: correct outcome — burning the full item set against a dead daemon is not graceful, it is slow.
_MAX_CONSECUTIVE_ITEM_FAILURES = 3


async def collect_records(
    model: OllamaModel,
    items: list[CalibrationItem],
    *,
    k: int = 3,
    per_item_timeout: float | None = _PER_ITEM_TIMEOUT_S,
) -> tuple[list[JudgmentRecord], dict[str, Any]]:
    """Run one model over the calibration items (k reruns), returning graded records + stats.

    Each record is stamped with the **authored** ``item.id`` (NOT the adapter's
    prompt-hash fallback) so every grouping metric — known-groups, consistency,
    panel ρ-correlation, IRT prune, alt-test — aligns across models and against the
    authored set. ``predicted``/``gold`` are mapped to 0/1, ``correct`` is the
    exact-match on the parsed token, and ``metadata`` carries the category + the
    item difficulty (the §12 difficulty weighting) + the raw parsed token.

    Graceful degradation (characterize-degradation-002/004/005):

    * **Per-item salvage** — a single ``judge_item`` raise (or per-item timeout) drops only
      THAT item, not the whole model: a partial profile on 17/20 items (a slightly wider
      Wilson CI) is strictly better than dropping the model. Dropped items are counted +
      named in the returned stats. A *sustained* run of ≥:data:`_MAX_CONSECUTIVE_ITEM_FAILURES`
      consecutive failures re-raises (the daemon is down — let :func:`run_panel` drop the model).
    * **Per-item timeout** — each call is bounded by ``per_item_timeout`` (``None`` disables it,
      e.g. for the offline tests). A hung call is treated as a dropped item, not a hang.
    * **Parse-failure accounting** — an unparseable output is still scored *incorrect*
      (conservative default), but it is also COUNTED (``stats["n_unparsed"]``) and the record is
      flagged ``metadata["parse_failure"] = True`` so a format break is diagnosable rather than
      silently depressing accuracy as if the judge were merely weak.

    Returns ``(records, stats)`` where ``stats`` carries ``dropped_items`` /
    ``dropped_item_ids`` / ``n_unparsed`` for the report (honesty-only: attempted vs measured).
    """
    records: list[JudgmentRecord] = []
    dropped_item_ids: list[str] = []
    n_unparsed = 0
    consecutive_failures = 0
    for item in items:
        gold_str = str(item.gold).upper()
        gold_num = _to_num(gold_str)
        item_failed = False
        for ri in range(k):
            try:
                if per_item_timeout is not None:
                    rec = await asyncio.wait_for(
                        model.judge_item(item.prompt, run_index=ri), timeout=per_item_timeout
                    )
                else:
                    rec = await model.judge_item(item.prompt, run_index=ri)
            except Exception as exc:  # noqa: BLE001 — any per-item fault is salvaged below
                # Per-item salvage: drop only this item's reruns; keep the rest of the model.
                # asyncio.wait_for raises TimeoutError (==asyncio.TimeoutError since 3.11), a
                # subclass of Exception, so this single handler covers the hung-call case too.
                consecutive_failures += 1
                reason = "timeout" if isinstance(exc, TimeoutError) else repr(exc)
                dropped_item_ids.append(f"{item.id} ({reason})")
                print(f"[{model.model_id}] item {item.id} dropped (run {ri}): {reason}")
                item_failed = True
                if consecutive_failures >= _MAX_CONSECUTIVE_ITEM_FAILURES:
                    # Sustained failure → daemon-down signal: let run_panel drop the whole model.
                    raise
                break  # skip remaining reruns of this item; move on
            consecutive_failures = 0  # a success breaks the consecutive-failure streak
            rec.item_id = item.id  # authored id, not the prompt-hash fallback (§11.3)
            parsed = parse_choice(str(rec.predicted), gold_str)
            pred_num = _to_num(parsed) if parsed is not None else (1 - gold_num)
            rec.predicted = pred_num
            rec.gold = gold_num
            rec.correct = parsed == gold_str
            rec.metadata["category"] = item.category.value
            rec.metadata["raw_choice"] = parsed
            # characterize-degradation-004: an unparseable output is scored wrong (conservative)
            # but flagged + counted so a format break is distinct from a genuinely-weak judge.
            rec.metadata["parse_failure"] = parsed is None
            if parsed is None:
                n_unparsed += 1
            if item.difficulty is not None:
                rec.metadata["difficulty"] = item.difficulty
            if per_item_timeout is not None:
                rec.metadata["per_item_timeout_s"] = per_item_timeout  # PIN_PER_STEP
            records.append(rec)
        if item_failed:
            continue
    stats = {
        "dropped_items": len(dropped_item_ids),
        "dropped_item_ids": dropped_item_ids,
        "n_unparsed": n_unparsed,
    }
    return records, stats


def _decision_why(profile: JudgeProfile) -> str | None:
    """Pull the contrastive WHY for a profile's seat/screen/reject — the deciding reason.

    characterize-result-legibility-005: ``build_profile`` records a human-legible
    ``DECISION: REJECT — <reason>`` / ``DECISION: SCREEN — <why>`` note (profile.py) that
    names the gate that moved the decision. The terminal line should carry that WHY for any
    non-SEAT decision so a REJECT driven by a format break is distinguishable on screen from
    one driven by a genuinely-weak judge — without the operator opening the JSON. Returns the
    reason text after ``DECISION: <WORD> — `` (em-dash), or ``None`` if no such note exists.
    """
    for note in profile.notes:
        if note.startswith("DECISION:"):
            # "DECISION: REJECT — κ<floor ..." → "κ<floor ..." (split on the em-dash once)
            _head, _sep, tail = note.partition(" — ")
            return tail.strip() if tail else note[len("DECISION:"):].strip()
    return None


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
    (:func:`ai_crucible.characterize.metrics.alt_test_omega`); the OTHER panel models are
    the "annotators". Returns ``None`` when fewer than two peers exist (leave-one-out
    needs ≥2), so :func:`build_profile` records "alt-test not measured" rather than
    raising. CIRCULAR by construction — see :data:`_ALT_TEST_CAVEAT`.
    """
    peers = {oid: orecs for oid, orecs in all_records.items() if oid != model_id}
    if len(peers) < 2:
        return None
    return {"judge": recs, **peers}


async def run_panel(
    panel: list[tuple[str, str | None, str | None]],
    items: list[CalibrationItem],
    *,
    k: int = 3,
    human_labels: HumanLabels | None = None,
) -> tuple[dict[str, JudgeProfile], dict[str, list[JudgmentRecord]], list[dict[str, Any]]]:
    """Sequential load → judge → evict, then profile with the alt-test (§11.2/§12).

    Two passes: (1) judge the whole panel one model at a time (VRAM-respecting —
    ``OLLAMA_NUM_PARALLEL=1``, evict between models); (2) build each model's profile.
    Pass 2 needs the full record set first, which is why profiling is deferred.

    Returns ``(profiles, all_records, failed)``. ``failed`` is the
    characterize-degradation-001 honesty signal: a model that fails mid-run must NOT vanish
    from the report — it is captured as ``{"model_id", "family", "error", "n_records_lost"}``
    so a thinned panel (one OOM / one unservable tag — the common partial-failure case) is
    distinguishable from a smaller-by-design one. Per-item salvage (characterize-degradation-002)
    means a model with a few dropped items still seats with a partial profile; only a
    daemon-down-level failure lands a model in ``failed``.

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
    stats_by_id: dict[str, dict[str, Any]] = {}
    failed: list[dict[str, Any]] = []
    n_panel = len(panel)
    for idx, (model_id, family, quant) in enumerate(panel, start=1):
        quant_by_id[model_id] = quant
        # characterize-result-legibility-003: emit per-model progress on stderr BEFORE the
        # pass so a multi-hour run shows liveness instead of a ~25-min black hole per model.
        # stderr (not stdout) so a consumer parsing the stdout summary lines stays clean.
        print(
            f"[{idx}/{n_panel}] loading + judging {model_id} "
            f"({len(items)} item(s) x k={k}) ...",
            file=sys.stderr,
            flush=True,
        )
        if model_id.startswith("openrouter:"):
            model = OpenRouterModel(model_id=model_id, family=family or "", quant=quant)
        else:
            model = OllamaModel(model_id=model_id, family=family, quant=quant)
        t0 = time.monotonic()
        try:
            recs, stats = await collect_records(model, items, k=k)
        except Exception as exc:  # noqa: BLE001 — one bad model must not sink the panel
            # characterize-degradation-001: do NOT let the failure vanish into stdout alone —
            # capture it (id + reason) so the report can list the model + flag the run degraded.
            print(f"[{model_id}] ERROR: {exc!r}")
            failed.append(
                {
                    "model_id": model_id,
                    "family": family,
                    "error": repr(exc),
                    "n_records_lost": 0,
                }
            )
            _evict(model_id)
            continue
        elapsed = time.monotonic() - t0
        if not recs:
            # characterize-degradation-001/002: every item was salvaged-away (dropped/timed out)
            # WITHOUT tripping the consecutive-failure ceiling — e.g. an item-set shorter than the
            # ceiling, or alternating failures. Zero records is not a profileable model and would
            # crash the profiler downstream; treat it as a full failure so it lands in `failed`
            # (surfaced + degraded), not in all_records as an empty, un-profileable set.
            print(f"[{model_id}] no records salvaged ({stats.get('dropped_items', 0)} dropped)")
            failed.append(
                {
                    "model_id": model_id,
                    "family": family,
                    "error": (
                        f"no records salvaged ({stats.get('dropped_items', 0)} item(s) dropped: "
                        f"{stats.get('dropped_item_ids', [])})"
                    ),
                    "n_records_lost": stats.get("dropped_items", 0),
                }
            )
            _evict(model_id)
            continue
        all_records[model_id] = recs
        stats_by_id[model_id] = stats
        acc = sum(1 for r in recs if r.correct) / len(recs) if recs else 0.0
        # characterize-degradation-004: surface the unparseable count next to acc so a format
        # break is visible at a glance, not hidden inside a depressed accuracy.
        unparsed = stats.get("n_unparsed", 0)
        dropped = stats.get("dropped_items", 0)
        print(
            f"[{model_id}] judged {len(recs)} records  acc={acc:.3f}  "
            f"unparsed={unparsed}  dropped_items={dropped}  ({elapsed:.0f}s)"
        )
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
        # characterize-degradation-004 / -002: make the unparseable rate + salvaged-item count
        # first-class, report-visible profile fields so a REJECT driven by a format break (high
        # unparsed rate) is diagnosable, and a partial profile (items dropped) is flagged as such
        # rather than read as a confident full-set measurement.
        stats = stats_by_id.get(model_id, {})
        n_total = len(recs)
        n_unparsed = stats.get("n_unparsed", 0)
        profile.metadata["n_unparsed"] = n_unparsed
        profile.metadata["parse_failure_rate"] = round(n_unparsed / n_total, 4) if n_total else 0.0
        profile.metadata["dropped_items"] = stats.get("dropped_items", 0)
        if stats.get("dropped_items", 0):
            profile.notes.append(
                f"[degraded] profiled on a PARTIAL item set — {stats['dropped_items']} item(s) "
                f"dropped: {stats.get('dropped_item_ids', [])}"
            )
        if n_total and n_unparsed / n_total >= 0.10:
            profile.notes.append(
                f"[warning] {n_unparsed}/{n_total} outputs were UNPARSEABLE and scored as "
                "incorrect — check the prompt format before trusting this decision"
            )
        profiles[model_id] = profile
        flag = " [review]" if profile.metadata.get("review_flag") else ""
        q = profile.metadata.get("quality_score")
        print(
            f"[{model_id}] {profile.seat_decision.value}{flag}  acc={profile.objective_accuracy}  "
            f"q={q}  ece={profile.ece}  omega={profile.alt_test_omega}"
        )
        # characterize-result-legibility-005: for a non-SEAT decision, echo the contrastive
        # WHY (the deciding gate from build_profile's DECISION note) to stderr so the reason is
        # legible on screen — a format-break REJECT reads differently from a weak-judge REJECT.
        if profile.seat_decision is not SeatDecision.SEAT:
            why = _decision_why(profile)
            if why:
                print(
                    f"[{model_id}]   └─ {profile.seat_decision.value.upper()}: {why}",
                    file=sys.stderr,
                )
        # surface the [warning]/[degraded] interpretation notes inline too (not only the raw
        # unparsed=/dropped_items= counts), so the operator sees the actionable reading.
        for note in profile.notes:
            if note.startswith(("[warning]", "[degraded]")):
                print(f"[{model_id}]   {note}", file=sys.stderr)
    return profiles, all_records, failed


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
    """Collapse k reruns to one ``{model_id: {item_id: majority_correct}}`` grid.

    The majority vote breaks an exact tie CONSERVATIVELY toward *incorrect* (``> 0.5``, not
    ``>= 0.5``) — calibration-instrument-003. This grid feeds :func:`irt.prune_items` (the
    §12 IRT calibration screen), so an optimistic ``>= 0.5`` tie-break (rounding a 50/50
    even-k split UP to a pass) could misclassify a knife-edge item as all-pass (zero variance
    → dropped) or push it across the point-biserial floor, systematically biasing which
    calibration items survive. The item bank is the foundation of every later measurement, so
    a tie must be deterministic and conservative: an item is graded "correct" only when a
    STRICT majority of its reruns were correct; an exact tie (or minority) is "incorrect".
    """
    matrix: dict[str, dict[str, bool]] = {}
    for model_id, recs in records.items():
        by_item: dict[str, list[bool]] = {}
        for r in recs:
            by_item.setdefault(r.item_id, []).append(bool(r.correct))
        matrix[model_id] = {iid: (sum(v) / len(v)) > 0.5 for iid, v in by_item.items()}
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
    quorum-checked panel ai_crucible would score with (:func:`aggregate.compose_panel`).
    Below quorum the panel escalates to the Claude Designer rather than seating thin.
    """
    try:
        return asdict(aggregate.compose_panel(profiles, records))
    except Exception as exc:  # noqa: BLE001
        return {"error": repr(exc)}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="AI Crucible judge-admission characterization run.")
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

    item_set_name = args.items.name if args.items else "admission_pairs.json"
    reference = "human" if human_labels else "model-jury-bootstrap (PROVISIONAL)"
    # characterize-result-legibility-004: announce the plan on stderr BEFORE the (long) run so
    # the operator can confirm it started + abort a wrong/unintended job before sinking the
    # time. Emitted on stderr to keep the stdout summary stream clean for any consumer.
    print(
        f"characterizing {len(panel)} model(s) over {len(items)} item(s), k={args.k}, "
        f"item_set={item_set_name}, reference={reference}; "
        f"this is a long run (minutes per model). report -> {args.out}",
        file=sys.stderr,
        flush=True,
    )

    profiles, records, failed = asyncio.run(
        run_panel(panel, items, k=args.k, human_labels=human_labels)
    )

    report = {
        "n_items": len(items),
        "k": args.k,
        "item_set": args.items.name if args.items else "admission_pairs.json",
        # characterize-degradation-001: a model that failed mid-run must NOT vanish — list the
        # requested panel + the failures (id + reason) and flag the run `degraded` so a thinned
        # panel is distinguishable from a smaller-by-design one. Honesty-only: surfacing what was
        # attempted vs measured, no fabricated data. ANDON_AUTHORITY: `degraded` is the signal a
        # downstream gate can halt on rather than consuming a quietly-thinned panel as full.
        "attempted_panel": [m for m, _f, _q in panel],
        "failed_models": failed,
        "degraded": bool(failed),
        "alt_test_reference": "human" if human_labels else "model-jury-bootstrap",
        "alt_test_caveat": _alt_test_human_note(human_labels) if human_labels else _ALT_TEST_CAVEAT,
        # characterize-002: the κ one-sided floor + κ-z gate run against a human–human κ
        # baseline. On the DEFAULT (model-jury) path there are no humans, so build_profile's
        # 0.80 default is a FABRICATED baseline — surface that in a machine-readable form
        # (parallel to ``alt_test_caveat`` for the circular ω) so a consumer parsing the JSON
        # can SEE a κ-floor REJECT / a human-like SEAT is provisional, not human-grounded.
        # Honesty-only: no human labels are fabricated. On the human-grounded path the
        # baseline IS the measured human–human Krippendorff α, so it is NOT provisional.
        "kappa_baseline": (
            {
                "value": round(human_labels.iaa_alpha, 4),
                "provisional": False,
                "source": (
                    f"measured human–human Krippendorff α over {human_labels.n_annotators} "
                    "annotators (Fork C, §12.1) — the κ floor/z gates are human-grounded"
                ),
            }
            if human_labels
            else {
                "value": _DEFAULT_HUMAN_KAPPA,
                "provisional": True,
                "alt_test_reference": "model-jury-bootstrap",
                "source": (
                    "hardcoded default — NOT human-estimated; the κ one-sided floor and κ-z "
                    "gates (which can SEAT or REJECT a judge) are PROVISIONAL until "
                    "--human-labels supplies a measured human–human Krippendorff α (§12 Q3)"
                ),
            }
        ),
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

    # models-cli-003: a measurement instrument that measured NOTHING is a FAILURE, not a
    # green pass. The per-model try/except (run_panel) keeps one bad model from sinking the
    # panel, but if EVERY model failed (Ollama down / wrong host / every tag unservable) both
    # ``profiles`` and ``records`` are empty — emit a structured error (code/message/hint) and
    # exit NON-ZERO so an automation/CI gate (or the dogfood-swarm harness) that checks the
    # exit code does not consume a stale/empty characterization report as if valid. Exit 1 is
    # distinct from the dispatcher's 2 (unknown command).
    if not profiles or not records:
        err = {
            "code": "CHARACTERIZE_NO_JUDGMENTS",
            "message": (
                "characterization collected zero judgments — every model failed "
                "(empty profiles/records)"
            ),
            "hint": (
                "check that Ollama is running and reachable (default http://localhost:11434), "
                "that each --models tag is pulled/servable, and re-run; the report at "
                f"{args.out} contains only empty profiles and must not be consumed as valid"
            ),
        }
        sys.stderr.write(json.dumps(err) + "\n")
        print(f"report -> {args.out}")
        return 1

    seated = [m for m, p in profiles.items() if p.seat_decision.value == "seat"]
    print(f"\nseated: {seated}")
    # characterize-result-legibility-005: the final summary carries the contrastive WHY for
    # every NON-SEAT decision (the deciding gate from build_profile's DECISION note) so a
    # REJECT driven by a format break reads differently on screen from a weak-judge REJECT —
    # the operator does not have to open the JSON to learn why a model was not seated. stderr
    # (the JSON report's profiles[*].notes carry the same text machine-readably).
    for model_id, profile in profiles.items():
        if profile.seat_decision is not SeatDecision.SEAT:
            why = _decision_why(profile)
            tail = f" — {why}" if why else ""
            print(
                f"  {profile.seat_decision.value.upper()}: {model_id}{tail}",
                file=sys.stderr,
            )
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

    # characterize-result-legibility-002: a DEGRADED run (failed models from Stage B) must
    # SAY so to the operator — a thinned panel that still met quorum is not a full-strength
    # result. Surfaced on stderr (the JSON report already carries `degraded`/`failed_models`).
    if failed:
        ids = [f["model_id"] for f in failed]
        print(
            f"\nDEGRADED RUN: {len(failed)} of {len(report['attempted_panel'])} model(s) "
            f"failed and were dropped: {ids} — this panel is PARTIAL. "
            f"See failed_models in {args.out}.",
            file=sys.stderr,
            flush=True,
        )

    # characterize-result-legibility-001 + error-hint-sweep-002: the load-bearing PROVISIONAL
    # caveat is the LAST thing the operator sees. On the default (no --human-labels) path the
    # alt-test ω is a CIRCULAR model-jury bootstrap and the κ baseline is hardcoded (0.80), so
    # every seat is PROVISIONAL and a sub-quorum panel escalates to the Claude Designer. The
    # operator must NOT read the seated list as authoritative. On the human-grounded path the
    # circular caveat is RETIRED — print the retired-caveat note instead. stderr keeps stdout
    # clean; the same caveat lives machine-readably in report['alt_test_caveat'].
    if human_labels is None:
        print(
            "\n*** PROVISIONAL RESULT — DO NOT treat this seated panel as authoritative ***\n"
            "    alt-test ω is a CIRCULAR model-jury bootstrap (the reference 'annotators'\n"
            "    are the other panel models) and the κ baseline is hardcoded (0.80, NOT\n"
            "    human-measured). All seat decisions are PROVISIONAL until a human-labeling\n"
            "    round runs; a sub-quorum panel escalates to the Claude Designer.\n"
            "    Pass --human-labels for a non-circular, human-grounded alt-test.\n"
            f"    See alt_test_caveat in {args.out}.",
            file=sys.stderr,
            flush=True,
        )
    else:
        print(
            "\nNOTE: alt-test ω is HUMAN-GROUNDED this run (Fork C) — the circular "
            "model-jury caveat is RETIRED; seats are NOT provisional on the alt-test axis.",
            file=sys.stderr,
            flush=True,
        )
    return 0


def _parse_models(specs: list[str]) -> list[tuple[str, str | None, str | None]]:
    """Parse ``model_id@family`` specs. The model_id itself may contain a ':'
    (e.g. ``qwen3.6:27b``), so split on the LAST '@' only.

    An UNTAGGED entry (no ``@family``) stamps ``family=None`` — NEVER the literal string
    ``"unknown"`` (models-cli-001, the SHARED EXTERNAL_VERIFIER family contract). ``None``
    is the honest representation of an unknown family: :func:`judge_family` documents that an
    untagged judge returns ``None`` and is admitted as cross-family on OPERATOR
    responsibility (it cannot be *proven* cross-family). A literal ``"unknown"`` is a genuine
    family value that COLLIDES with every other ``"unknown"`` and is matched by the exact
    exclusion comparison, so a mixed tagged/untagged SAME-family judge could survive
    same-family exclusion against a generator also tagged ``"unknown"`` — silently weakening
    EXTERNAL_VERIFIER (research-grounding §3/§10.2). ``None`` never equals any concrete
    family, so judge_panel's None-handling applies and the residual is honest (the operator
    sees the untagged judge was seated on trust via ``panel.metadata["untagged_judges_seated"]``).
    """
    out: list[tuple[str, str | None, str | None]] = []
    for s in specs:
        if "@" in s:
            mid, fam = s.rsplit("@", 1)
        else:
            mid, fam = s, None
        # An OpenRouter seat MUST declare its @family — the served id exposes the vendor, but the
        # cross-family attribution is the operator's call. Refuse fail-fast (before the long run)
        # rather than seat an unattributable judge (models-cli-001 / EXTERNAL_VERIFIER §10.2).
        if mid.startswith("openrouter:") and not (fam and fam.strip()):
            raise ValueError(
                f"[OPENROUTER_NO_FAMILY] the OpenRouter spec {s!r} needs an explicit @family for "
                "cross-family attribution (hint: e.g. openrouter:deepseek/deepseek-chat@deepseek)"
            )
        out.append((mid, fam, None))  # untagged → fam is None (NOT the colliding "unknown")
    return out


def _load_admission_items() -> list[CalibrationItem]:
    """Load the bundled judge-admission **pairs** set (§12 Q1 — the discriminating shape)."""
    path = Path(__file__).parent.parent / "calibration" / "admission_pairs.json"
    return load_items(path)


if __name__ == "__main__":
    raise SystemExit(main())
