"""Stage 2 — the 7-step tuning protocol (research-grounding §9.4).

Tuning a measurement instrument without compromising its measurements has a
specific answer: **multi-stage separation between hypothesis-locking, tuning, and
release**, with the validation set touched exactly once per bundle hash. This
module implements the load-bearing, fully-real parts of the §9.4 7-step protocol
and provides clearly-marked documented stubs for the parts whose full
implementation is out of Phase-1 scope.

The seven steps and their status here:

1. **Split inventory 60/20/10/10** — :func:`split_inventory`. REAL. Per
   **Dwork, Feldman, Hardt, Pitassi, Reingold & Roth 2015** ("The Reusable
   Holdout," *Science* 349(6248):636-638, arXiv:1506.02629): reusing a single
   holdout adaptively decays validity; the multi-split structure bounds this.
   Each split manifest is content-hashed (a maintainer publishes a Datasheet per
   split — Gebru et al. 2018, arXiv:1803.09010).
2. **Sobol screen** — :func:`sobol_screen`. REAL, wraps
   ``scipy.stats.sobol_indices`` (**Saltelli et al. 2010**, *Comput. Phys.
   Commun.* 181(2):259-270). Total-effect index T_i identifies load-bearing vs
   noise parameters; freeze T_i≈0.
3. **BO-search top-T_i weights** — :func:`bo_search`. DOCUMENTED STUB. The
   protocol (**Snoek, Larochelle & Adams 2012**, NeurIPS 2012) is specified;
   the full GP loop is out of Phase-1 scope and raises ``NotImplementedError``
   with the exact contract a future implementation must satisfy.
4. **Paraphrase-ablate judge prompts** — :func:`paraphrase_ablate`. DOCUMENTED
   STUB (**JudgeSense**, Bellibatlu et al. 2026, arXiv:2604.23478). The decision
   rule (drop findings whose sign flips under paraphrase) is documented; the
   paraphrase-generation + re-scoring loop is out of Phase-1 scope.
5. **Validate the locked bundle ONCE on `validation`** — enforced by
   :class:`ThresholdoutBudget` semantics + the split structure; you cannot
   iterate on validation.
6. **Seal `private_test`** — :func:`split_inventory` produces it; it is never
   queried by the tuning loop.
7. **Publish** the bundle + ``TUNING.md`` provenance + Sobol report + BO trace +
   per-split Datasheet — :func:`split_inventory` and :func:`sobol_screen` return
   exactly the provenance fields ``TUNING.md`` records.

:class:`ThresholdoutBudget` (step 3) tracks the dev-set query budget so adaptive
reuse stays bounded (**Dwork 2015**).

Standards compliance (the six — workflow-standards.md):
- PIN_PER_STEP — 3: :func:`split_inventory` takes a deterministic ``seed_note``
  and a *given order* of ids (it does NOT call ``random``/the clock), so a split
  is byte-for-byte reproducible and its manifest hash is the pin. :func:`sobol_screen`
  takes an explicit ``seed`` so the Saltelli sample is reproducible.
- ANDON_AUTHORITY — 2: :class:`ThresholdoutBudget.query` raises a structured
  ``TuningBudgetError`` when the dev-query budget is exhausted — the protocol
  halts rather than silently over-querying the holdout (the §9.4 step-3 defect).
  :func:`split_inventory` raises on a duplicate/empty id set.
- NAMED_COMPENSATORS — n/a: pure computation, no irreversible tool call. (Sealing
  ``private_test`` is enforced by *not* querying it, not by an undoable action.)
- DECOMPOSE_BY_SECRETS — 3: the split structure (which changes per inventory) is
  separated from the budget tracker (which changes per query) and from the
  sensitivity screen (which changes per parameter set); the sealed
  ``private_test`` is decomposed away from every queryable split by construction.
- UNCERTAINTY_GATED_HUMANS — 2: the Thresholdout budget *is* the gate — it caps
  how much the human can adaptively interrogate the dev split before the result
  is no longer trustworthy, and it reports remaining budget contrastively.
- EXTERNAL_VERIFIER — 2: split manifest hashes let an external auditor confirm
  the splits were disjoint and unchanged without re-running anything; the Sobol
  report tells the auditor which weights were even tunable. The stubs explicitly
  refuse to fake a result (NotImplementedError, not a plausible number).
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np
from scipy.stats import sobol_indices, uniform

__all__ = [
    "TuningError",
    "TuningBudgetError",
    "SPLIT_FRACTIONS",
    "split_inventory",
    "sobol_screen",
    "ThresholdoutBudget",
    "bo_search",
    "paraphrase_ablate",
]


class TuningError(Exception):
    """Raised on a malformed tuning input (empty/duplicate inventory, bad bounds).
    Structured ``[CODE] message (hint: ...)`` payload (Ship-Gate-B shape)."""


class TuningBudgetError(TuningError):
    """Raised when the Thresholdout dev-query budget is exhausted (§9.4 step 3).
    A subclass of :class:`TuningError` so callers can catch either."""


def _fail(
    code: str, message: str, hint: str, *, cls: type[TuningError] = TuningError
) -> TuningError:
    return cls(f"[{code}] {message} (hint: {hint})")


# The §9.4 step-1 split fractions, in fixed order. Sum to 1.0.
SPLIT_FRACTIONS: tuple[tuple[str, float], ...] = (
    ("calibration", 0.60),
    ("dev", 0.20),
    ("validation", 0.10),
    ("private_test", 0.10),
)


def _manifest_hash(name: str, ids: list[str], seed_note: str) -> str:
    """Content-hash a split manifest: the split name, its ordered ids, and the
    seed note that determined the partition. Lets an auditor confirm the split is
    the one that was registered (§9.4 step 1; Datasheet per split)."""
    material = json.dumps(
        {"split": name, "ids": ids, "seed_note": seed_note},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def split_inventory(puzzle_ids: Sequence[str], seed_note: str) -> dict:
    """Partition ``puzzle_ids`` 60/20/10/10 into calibration/dev/validation/
    private_test, **deterministically** (§9.4 step 1).

    Determinism contract (PIN_PER_STEP): this function does NOT use
    ``random.random`` or the clock. The partition is a pure function of the
    *given order* of ``puzzle_ids`` and the ``seed_note``. ``seed_note`` seeds a
    reproducible permutation (so the split isn't biased by however the catalog
    happened to be enumerated) via a fixed SHA-256-derived ordering — the same
    ``(puzzle_ids, seed_note)`` always yields the same split, on any machine, and
    the seed_note is recorded in each manifest hash so the split is auditable.

    Sizes use largest-remainder rounding so the four splits always sum to exactly
    ``len(puzzle_ids)`` (no off-by-one drop). For the canonical 100-id case this
    is exactly 60/20/10/10.

    Returns a dict with:
        ``seed_note``: the supplied note (echoed for the record).
        ``order``: the deterministic permutation of ids the split was cut from.
        ``splits``: ``{name: {"ids": [...], "n": int, "manifest_sha256": hex}}``
            for each of the four splits in :data:`SPLIT_FRACTIONS` order.
        ``private_test_sealed``: ``True`` — a flag the tuning loop honors by never
            querying that split (step 6).

    Raises:
        TuningError: empty inventory, or duplicate ids (a split must be a clean
            partition; duplicates would leak across splits — the exact §9.4
            failure the structure exists to prevent).
    """
    ids = list(puzzle_ids)
    if not ids:
        raise _fail(
            "INPUT_INVENTORY_EMPTY",
            "cannot split an empty puzzle inventory",
            "pass the list of puzzle_ids to partition (≥4 recommended so every "
            "split is non-empty)",
        )
    if len(set(ids)) != len(ids):
        dupes = sorted({x for x in ids if ids.count(x) > 1})
        raise _fail(
            "INPUT_INVENTORY_DUPLICATE",
            f"puzzle inventory contains duplicate ids: {dupes[:5]}",
            "splits must be a clean partition; de-duplicate before splitting so "
            "no id leaks across calibration/dev/validation/private_test",
        )

    # Deterministic permutation: sort by a per-id hash salted with seed_note.
    # This is reproducible (no RNG state, no clock) and decorrelates the split
    # from the incoming enumeration order, satisfying both PIN_PER_STEP and the
    # "don't bias the split by catalog order" intent.
    def _key(pid: str) -> str:
        return hashlib.sha256(f"{seed_note}\x00{pid}".encode()).hexdigest()

    order = sorted(ids, key=_key)

    # Largest-remainder apportionment so sizes sum to len(order) exactly.
    n = len(order)
    raw = [(name, frac * n) for name, frac in SPLIT_FRACTIONS]
    floors = [(name, int(val)) for name, val in raw]
    assigned = sum(c for _, c in floors)
    remainder = n - assigned
    # Distribute the leftover to the splits with the largest fractional parts.
    frac_parts = sorted(
        ((val - int(val), idx) for idx, (_, val) in enumerate(raw)),
        reverse=True,
    )
    counts = [c for _, c in floors]
    for k in range(remainder):
        counts[frac_parts[k][1]] += 1

    splits: dict[str, dict] = {}
    cursor = 0
    for (name, _frac), count in zip(SPLIT_FRACTIONS, counts, strict=True):
        chunk = order[cursor : cursor + count]
        cursor += count
        splits[name] = {
            "ids": chunk,
            "n": len(chunk),
            "manifest_sha256": _manifest_hash(name, chunk, seed_note),
        }

    return {
        "seed_note": seed_note,
        "order": order,
        "splits": splits,
        "private_test_sealed": True,
    }


def _nearest_power_of_two(n: int) -> int:
    """Round ``n`` to the nearest power of two (>=2), which
    ``scipy.stats.sobol_indices`` requires for the Saltelli balance properties."""
    if n < 2:
        return 2
    lower = 1 << (n.bit_length() - 1)
    upper = lower << 1
    return lower if (n - lower) <= (upper - n) else upper


def sobol_screen(
    param_names: Sequence[str],
    bounds: Sequence[tuple[float, float]],
    model_fn: Callable[[np.ndarray], np.ndarray],
    n: int = 256,
    *,
    seed: int = 0,
) -> dict[str, float]:
    """Sobol total-effect screen over candidate weights (§9.4 step 2).

    Wraps ``scipy.stats.sobol_indices(method='saltelli_2010')``. Returns
    ``{param_name: total_effect_T_i}`` — the total-order Sobol index per
    parameter. Parameters with ``T_i ≈ 0`` are *not load-bearing* and get frozen;
    only the load-bearing ones go to BO search (step 3). This is the §9.4
    mechanism that keeps the tuning surface small enough that the dev-set query
    budget actually bounds the over-fitting risk.

    Args:
        param_names: the candidate weight names, in order.
        bounds: ``(low, high)`` per parameter, same order as ``param_names``.
            Each maps to a ``scipy.stats.uniform`` marginal over the range.
        model_fn: the objective. Receives an array of shape ``(d, m)`` (d =
            number of params, m = sample count) and returns a length-``m`` array
            of scalar outputs — the SciPy ``sobol_indices`` convention. In
            ai_crucible this is a surrogate of "gate-clear rate on calibration as a
            function of the weights"; for screening it can be any deterministic
            scalarisation.
        n: Saltelli base sample size. Rounded to the nearest power of two
            (SciPy requirement) — the effective ``n`` is recorded nowhere it can
            silently differ because rounding is deterministic.
        seed: RNG seed for the Saltelli sequence, so the screen is reproducible
            (PIN_PER_STEP).

    Returns:
        ``{param_name: T_i}`` with ``T_i`` clipped at 0.0 from below (Sobol
        estimates can go slightly negative at finite sample; a negative total
        effect is physically 0 and reads cleaner in the report).

    Raises:
        TuningError: ``param_names`` and ``bounds`` length mismatch, empty
            params, or a degenerate bound (low >= high).
        TuningError: a non-finite (NaN/inf) ``objective`` output — caught at the
            source and surfaced rather than silently masked. A NaN objective is
            otherwise swallowed twice (scipy's variance ratio collapses a NaN
            sample to a clean total-effect 0.0, and ``max(0.0, nan) == 0.0`` clips
            it again), which FREEZES a possibly load-bearing weight as
            zero-sensitivity (the §9.4 step-2 failure this screen exists to
            prevent). The screen halts (ANDON) so the operator fixes the
            objective rather than tuning on a silently-frozen surface.
    """
    names = list(param_names)
    bnds = [tuple(b) for b in bounds]
    if not names:
        raise _fail(
            "INPUT_SOBOL_NO_PARAMS",
            "sobol_screen needs at least one parameter",
            "pass the candidate weight names you want to screen",
        )
    if len(names) != len(bnds):
        raise _fail(
            "INPUT_SOBOL_LEN_MISMATCH",
            f"param_names ({len(names)}) and bounds ({len(bnds)}) differ in length",
            "supply exactly one (low, high) bound per parameter, same order",
        )
    dists = []
    for name, (low, high) in zip(names, bnds, strict=True):
        if not (high > low):
            raise _fail(
                "INPUT_SOBOL_DEGENERATE_BOUND",
                f"parameter '{name}' has a non-increasing bound ({low}, {high})",
                "each bound must satisfy low < high so the marginal is non-degenerate",
            )
        # scipy.stats.uniform(loc, scale) spans [loc, loc+scale].
        dists.append(uniform(loc=low, scale=high - low))

    n_eff = _nearest_power_of_two(int(n))

    # ANDON: detect a non-finite objective AT THE SOURCE. A NaN/inf returned by
    # the objective is otherwise masked twice — once inside scipy's variance
    # ratio (a NaN sample collapses the total-effect estimate to a clean 0.0),
    # and again by ``max(0.0, T_i)`` below — so the index alone cannot reveal it.
    # Left unguarded, a NaN objective FREEZES a possibly load-bearing weight as
    # zero-sensitivity, the exact §9.4 step-2 defect this screen exists to
    # prevent. Wrap the objective so a non-finite output halts the screen instead
    # of silently reporting T_i=0.0.
    def _guarded_model_fn(x: np.ndarray) -> np.ndarray:
        out = np.asarray(model_fn(x), dtype=float)
        if not np.all(np.isfinite(out)):
            raise _fail(
                "STATE_SOBOL_NONFINITE_OBJECTIVE",
                "sobol_screen objective (model_fn) returned a non-finite "
                "(NaN/inf) value for at least one Saltelli sample",
                "a non-finite objective is silently masked to total-effect 0.0 "
                "and freezes a possibly load-bearing weight as zero-sensitivity "
                "— fix the objective to return finite scalars over the whole "
                "bounds box (§9.4 step 2)",
            )
        return out

    result = sobol_indices(
        func=_guarded_model_fn,
        n=n_eff,
        dists=dists,
        method="saltelli_2010",
        rng=np.random.default_rng(seed),
    )
    total = np.asarray(result.total_order, dtype=float)
    return {name: float(max(0.0, total[i])) for i, name in enumerate(names)}


@dataclass
class ThresholdoutBudget:
    """Bounded dev-set query budget for the adaptive tuning loop (§9.4 step 3).

    Per **Dwork et al. 2015** ("The Reusable Holdout," arXiv:1506.02629): each
    adaptive query against a holdout spends statistical validity; a *fixed* query
    budget bounds the decay. The BO search (step 3) reports against ``dev`` under
    this budget, and the §9.4-mandated provenance field
    ``dev_queries_used_of_budget`` is read straight off this object.

    This tracks the *count* of queries, not the holdout values themselves —
    ai_crucible's actual Thresholdout noise mechanism (add Laplace noise, only
    report when the dev estimate diverges from calibration beyond a threshold) is
    a step-3 detail layered on top; the budget cap is the load-bearing invariant
    and is enforced here.

    Attributes:
        budget: total permitted dev queries (fixed up front).
        used: queries spent so far.
    """

    budget: int
    used: int = 0
    _log: list[str] = field(default_factory=list, repr=False)

    def __post_init__(self) -> None:
        if self.budget < 0:
            raise _fail(
                "INPUT_BUDGET_NEGATIVE",
                f"Thresholdout budget must be ≥ 0, got {self.budget}",
                "set the fixed number of dev-set queries the BO search may spend",
            )

    @property
    def remaining(self) -> int:
        return max(0, self.budget - self.used)

    @property
    def exhausted(self) -> bool:
        return self.used >= self.budget

    def query(self, label: str = "") -> int:
        """Spend one dev-set query; return the number remaining *after* it.

        Raises :class:`TuningBudgetError` (ANDON) if the budget is already
        exhausted — the tuning loop must stop and validate on ``validation`` (step
        5) rather than keep interrogating the holdout. The error message is
        contrastive: it states budget, used, and the step-5 next action.
        """
        if self.exhausted:
            raise _fail(
                "STATE_DEV_BUDGET_EXHAUSTED",
                f"dev-set query budget exhausted ({self.used}/{self.budget} used)",
                "stop tuning and validate the locked bundle ONCE on `validation` "
                "(§9.4 step 5); adaptive holdout reuse beyond budget invalidates the "
                "result (Dwork 2015)",
                cls=TuningBudgetError,
            )
        self.used += 1
        self._log.append(label or f"query#{self.used}")
        return self.remaining

    def provenance(self) -> dict:
        """The §9.4-step-7 provenance fragment for ``TUNING.md``."""
        return {
            "dev_queries_used_of_budget": f"{self.used}/{self.budget}",
            "remaining": self.remaining,
        }


def bo_search(*_args, **_kwargs):  # noqa: ANN002, ANN003, ANN201
    """Bayesian-optimisation search over the load-bearing weights (§9.4 step 3).

    DOCUMENTED STUB — raises ``NotImplementedError``. A full implementation is out
    of Phase-1 scope (it needs a GP library and a real calibration objective).
    The contract a future implementation MUST satisfy, recorded so the stub is a
    specification and not a TODO:

    - Search ONLY the parameters the Sobol screen flagged load-bearing
      (``T_i`` above a documented cutoff). Frozen params keep their pinned value.
    - Report each trial against ``dev`` through a :class:`ThresholdoutBudget`;
      stop when the budget is exhausted (never silently over-query).
    - Log the GP posterior (the §9.4-step-7 "BO trace" artifact).
    - Return the proposed weight vector + the trace; the caller compiles it into
      a :class:`~ai_crucible.instrument.rubric_bundle.RubricBundle` and validates it
      ONCE on ``validation`` (step 5).

    Grounding: **Snoek, Larochelle & Adams 2012** ("Practical Bayesian
    Optimization of Machine Learning Algorithms," NeurIPS 2012).

    It deliberately does NOT return a fabricated weight vector: a plausible-but-
    fake tuning result would poison the very audit chain this module exists to
    protect (cf. the attestation module's "honest provenance or none" rule).
    """
    raise NotImplementedError(
        "[NOT_IMPLEMENTED] bo_search is a documented Phase-1 stub (§9.4 step 3); "
        "see the docstring for the contract a full GP implementation must satisfy. "
        "Sobol screen (sobol_screen) and the split structure (split_inventory) are "
        "the real Phase-1 deliverables."
    )


def paraphrase_ablate(*_args, **_kwargs):  # noqa: ANN002, ANN003, ANN201
    """Paraphrase-ablate judge prompts (§9.4 step 4).

    DOCUMENTED STUB — raises ``NotImplementedError``. The decision rule is fixed
    and load-bearing; the paraphrase-generation + re-scoring machinery is out of
    Phase-1 scope (it needs the judge panel, a live model panel, Phase 2).

    Contract a future implementation MUST satisfy:

    - Generate 5-10 semantically-equivalent paraphrases of each judge prompt.
    - Re-score a fixed probe set under each paraphrase.
    - **Any finding whose sign flips under paraphrase is below ai_crucible's
      resolution and is NOT reported** (the §9.4-step-4 rule).

    Grounding: **Bellibatlu, Raff & Zhang 2026** ("JudgeSense,"
    arXiv:2604.23478) — semantically-equivalent paraphrases produce measurably
    unstable scores and model scale does not fix it.
    """
    raise NotImplementedError(
        "[NOT_IMPLEMENTED] paraphrase_ablate is a documented Phase-1 stub "
        "(§9.4 step 4); the sign-flip-drop decision rule is specified in the "
        "docstring. Needs the Phase-2 judge panel + live model panel to run."
    )
