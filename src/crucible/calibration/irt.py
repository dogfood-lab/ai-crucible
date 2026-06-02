"""IRT prune / discrimination screen for the calibration set (research-grounding §12, Q1).

The first characterization run **saturated** — three of six local models scored 1.00 on
the 20-item set, so the top of the panel had no discrimination (§12). The fix is two-part:
author plausible-vs-subtly-wrong PAIRS (``admission_pairs.json``) AND screen the bank for
items that actually separate strong from weak judges. This module is that screen.

Why model-free is PRIMARY here
------------------------------
Classical 2PL IRT MLE needs ~200–500 respondents (Schroeders & Gnambs 2025); crucible's
panel is ~6 models, far below that floor. So the **model-free point-biserial / variance
screen is the primary instrument** (ATLAS, Peiyu Li et al. 2025, arXiv:2511.04689: drop
items with >95% accuracy / <1% variance / point-biserial r_pb < 0.1), and a Bayesian-prior
IRT fit (py-irt; Lalor, Wu & Yu 2019, EMNLP) is an *optional* secondary path
(:func:`fit_irt_bayesian`) used only when py-irt is installed. py-irt stays a LAZY optional
import — importing this module never requires it (build-law 4).

What "discrimination" means operationally
-----------------------------------------
* **Variance** — across the panel, how split is the verdict on this item? An item every
  model gets right (or every model gets wrong) has ~zero variance: it cannot separate
  models, so it is **saturated** and pruned. This is the §12 failure made measurable.
* **Point-biserial** — does getting *this* item right correlate with being a *stronger*
  model overall (higher total score)? A high r_pb means the item tracks ability; r_pb≈0
  means right/wrong on it is unrelated to skill (a coin-flip item) and it is pruned. The
  item is excluded from the total it is scored against (corrected item-total correlation,
  the standard discrimination index), so an item never inflates its own r_pb.

Standards compliance (the six — workflow-standards.md)
------------------------------------------------------
* **PIN_PER_STEP — 2:** the screen is a pure, deterministic function of the response
  matrix — identical input → identical kept/dropped split, so a prune decision is
  replayable from the recorded matrix without any model call.
* **ANDON_AUTHORITY — n/a:** a leaf analysis; the harness halts on its output, the screen
  raises a structured ``ValueError`` on malformed input rather than returning a silent
  wrong split.
* **NAMED_COMPENSATORS — n/a:** pure read/compute; no irreversible action to undo.
* **DECOMPOSE_BY_SECRETS — 2:** consumes only the already-graded boolean matrix
  (``correct``), never gold labels nor model internals — grading lives upstream with the
  scorer (§11.6).
* **UNCERTAINTY_GATED_HUMANS — n/a:** no human checkpoint at this layer; the thresholds
  are the director's calibrated knobs (§12 perturbation audit lives in the profiler).
* **EXTERNAL_VERIFIER — 2:** this *is* the item-quality verifier — it screens the
  instrument itself for the saturation the §12 verification pass caught on the first run.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

__all__ = ["point_biserial", "prune_items", "fit_irt_bayesian", "IRTError"]


class IRTError(ValueError):
    """Raised on malformed input to the discrimination screen.

    Carries the Ship-Gate-B structured shape (``[CODE] message (hint: ...)``) so a caller
    surfaces an actionable, redacted error without a raw stack (mirrors the calibration
    loader's :class:`~crucible.calibration.loader.CalibrationLoadError`).
    """


def _fail(code: str, message: str, hint: str) -> IRTError:
    return IRTError(f"[{code}] {message} (hint: {hint})")


def point_biserial(
    item_correct: Sequence[bool] | Sequence[int],
    total_scores: Sequence[float] | Sequence[int],
) -> float:
    """Point-biserial discrimination of one item: corr(got-it-right, overall ability).

    The point-biserial correlation is Pearson's r between a **binary** variable (did each
    respondent get this item right) and a **continuous** one (each respondent's total
    score). It is the classical item-discrimination index: high → the item separates
    strong from weak respondents; ≈0 → right/wrong on it is unrelated to ability.

    Args:
        item_correct: per-model correctness on this item (``True``/``1`` = correct),
            one entry per model, aligned with ``total_scores``.
        total_scores: per-model overall ability score (e.g. total items correct),
            aligned element-wise with ``item_correct``. For an unbiased index, pass the
            **corrected** total (this item's contribution removed); :func:`prune_items`
            does this for you.

    Returns:
        r_pb in ``[-1, 1]``. Returns ``0.0`` when discrimination is undefined — fewer than
        two models, an all-correct or all-wrong item (no item variance), or zero variance
        in the totals — because in every one of those cases the item provably cannot
        separate models, which is exactly the "drop it" signal the screen wants.

    Raises:
        IRTError: ``item_correct`` and ``total_scores`` differ in length.
    """
    item = np.asarray(item_correct, dtype=float)
    totals = np.asarray(total_scores, dtype=float)
    if item.shape != totals.shape:
        raise _fail(
            "IRT_LENGTH_MISMATCH",
            f"item_correct (n={item.size}) and total_scores (n={totals.size}) must align",
            "pass one correctness value and one total score per model, in the same order",
        )
    if item.size < 2:
        return 0.0
    # No item variance (all-correct / all-wrong) or no spread in ability → r is undefined;
    # report 0.0 (no discrimination) rather than a NaN the caller must special-case.
    if item.std() == 0.0 or totals.std() == 0.0:
        return 0.0
    # np.corrcoef of a binary vector vs a continuous one IS the point-biserial coefficient.
    r = float(np.corrcoef(item, totals)[0, 1])
    if np.isnan(r):  # defensive; the std() guards above should preclude this.
        return 0.0
    return r


def variance_of_item(item_correct: Sequence[bool] | Sequence[int]) -> float:
    """Population variance of the verdicts on one item across the panel.

    For a boolean item this is ``p*(1-p)`` where ``p`` is the pass-rate: 0.0 when every
    model agrees (saturated — the §12 failure), peaking at 0.25 at a 50/50 split. Used by
    :func:`prune_items` as the first (cheapest) saturation filter.
    """
    arr = np.asarray(item_correct, dtype=float)
    if arr.size == 0:
        return 0.0
    return float(arr.var())  # population variance (ddof=0): p*(1-p) for a boolean item.


def prune_items(
    matrix: Mapping[str, Mapping[str, bool]],
    *,
    min_variance: float = 0.0,
    min_point_biserial: float = 0.1,
) -> tuple[list[str], list[str]]:
    """Screen a response matrix, returning ``(kept, dropped)`` item ids (§12, ATLAS).

    The **model-free discrimination screen** — primary because ~6 models cannot MLE-fit a
    2PL (§12). An item is **dropped** when EITHER:

    * its verdict **variance ≤ ``min_variance``** — saturated: every model agrees, so it
      cannot separate the panel (the exact failure the first characterization run hit); or
    * its **point-biserial < ``min_point_biserial``** — getting it right doesn't track
      overall ability, so it is a non-discriminating (coin-flip / mislabeled) item.

    Point-biserial uses the **corrected item-total**: each model's total is the count of
    *other* items it got right (this item removed), so an item never discriminates against
    a total it is part of. Totals are computed over the full matrix once, then the item's
    own contribution is subtracted per item.

    Args:
        matrix: ``{model_id: {item_id: correct_bool}}`` — the graded panel × item grid
            (e.g. assembled from :class:`~crucible.characterize.types.JudgmentRecord`
            ``correct`` flags by the scorer). Every model must report the *same* item-id
            set; ``correct`` must be coercible to bool.
        min_variance: items with variance at or below this are saturated and dropped.
            Default ``0.0`` drops only perfectly-saturated items (all-agree); raise toward
            the ATLAS ``<1% variance`` to prune near-saturated ones too.
        min_point_biserial: items below this discrimination are dropped. Default ``0.1``
            is the ATLAS r_pb floor.

    Returns:
        ``(kept, dropped)`` — two lists of item ids in the matrix's item order. Their union
        is the full item set; they are disjoint.

    Raises:
        IRTError: the matrix is empty, models disagree on the item-id set, or a cell is
            missing.
    """
    if not matrix:
        raise _fail(
            "IRT_EMPTY_MATRIX",
            "response matrix has no models",
            "pass {model_id: {item_id: correct_bool}} with at least one model",
        )

    model_ids = list(matrix)
    # All models must cover the identical item set, else totals/columns don't align.
    item_id_sets = [frozenset(matrix[m]) for m in model_ids]
    item_ids = sorted(item_id_sets[0])
    if not item_ids:
        raise _fail(
            "IRT_EMPTY_ITEMS",
            "response matrix has models but no items",
            "each model's row must map item_id -> correct_bool for the same items",
        )
    for m, ids in zip(model_ids, item_id_sets, strict=True):
        if ids != item_id_sets[0]:
            missing = sorted(item_id_sets[0] - ids)
            extra = sorted(ids - item_id_sets[0])
            raise _fail(
                "IRT_RAGGED_MATRIX",
                f"model {m!r} item set differs (missing={missing}, extra={extra})",
                "every model must be scored on exactly the same item ids",
            )

    # Boolean matrix M[model, item]; column j is item_ids[j].
    grid = np.empty((len(model_ids), len(item_ids)), dtype=float)
    for i, m in enumerate(model_ids):
        row = matrix[m]
        for j, item_id in enumerate(item_ids):
            grid[i, j] = 1.0 if bool(row[item_id]) else 0.0

    full_totals = grid.sum(axis=1)  # per-model total correct across ALL items.

    kept: list[str] = []
    dropped: list[str] = []
    for j, item_id in enumerate(item_ids):
        column = grid[:, j]
        var = float(column.var())
        if var <= min_variance:
            dropped.append(item_id)  # saturated: no power to separate models.
            continue
        corrected_total = full_totals - column  # remove this item from the ability score.
        r_pb = point_biserial(column, corrected_total)
        if r_pb < min_point_biserial:
            dropped.append(item_id)  # non-discriminating: doesn't track ability.
            continue
        kept.append(item_id)

    return kept, dropped


def fit_irt_bayesian(matrix: Mapping[str, Mapping[str, bool]]) -> Any:
    """Bayesian-prior 2PL IRT fit (SECONDARY screen) — requires the optional ``py-irt``.

    With ~6 respondents, classical 2PL MLE is infeasible (needs ~200–500 — Schroeders &
    Gnambs 2025), so the supported small-N path is **variational IRT with priors** via
    **py-irt** (Lalor, Wu & Yu 2019, EMNLP; ``github.com/nd-ball/py-irt``), which returns
    per-item difficulty/discrimination posteriors with uncertainty. py-irt is a LAZY
    optional import: it is resolved here, at call time, so importing this module (and
    running the primary :func:`prune_items` screen) never requires it.

    This function is intentionally **not implemented** in this slice. When ``py-irt`` is
    absent it raises :class:`NotImplementedError` documenting the integration path
    (rather than ``ImportError``), so the missing-capability contract is explicit. When
    ``py-irt`` *is* present it still raises — wiring the Pyro dataset/model is a separate,
    deliberately-scoped step — but the message confirms the dependency was found.

    Args:
        matrix: the same ``{model_id: {item_id: correct_bool}}`` grid
            :func:`prune_items` consumes (validated there before any fit).

    Raises:
        NotImplementedError: always — either "install py-irt to enable this path" (when
            absent) or "py-irt found; the Bayesian fit wiring is a future slice — use
            prune_items as the primary screen" (when present). The model-free
            :func:`prune_items` is the primary instrument per §12; this is the documented
            secondary hook.
    """
    try:
        import py_irt  # type: ignore[import-untyped]  # noqa: F401
    except ImportError as exc:
        raise NotImplementedError(
            "fit_irt_bayesian needs the optional 'py-irt' package (Lalor, Wu & Yu 2019, "
            "EMNLP; github.com/nd-ball/py-irt) for small-N Bayesian-prior 2PL IRT — "
            "classical 2PL MLE is infeasible at crucible's ~6-model scale (Schroeders & "
            "Gnambs 2025). Install it (`uv add --optional irt py-irt`) to enable the "
            "secondary path; the model-free prune_items() screen is the primary §12 "
            "instrument and needs no extra dependency."
        ) from exc

    raise NotImplementedError(
        "py-irt is installed, but the Bayesian-prior 2PL fit wiring (Pyro dataset + "
        "variational inference over the response matrix → per-item difficulty/"
        "discrimination posteriors) is a deliberately-scoped future slice. Use the "
        "model-free prune_items() screen as the primary §12 discrimination instrument; "
        "this hook reserves the secondary Bayesian path documented in §12 / §11.3."
    )
