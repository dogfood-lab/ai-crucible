"""Stage 4 — Inspect-AI-compatible task definition (research-grounding §9.6).

§9.6 calls building Inspect-compatible task definitions **"the highest-leverage
move for credibility"** — UK AISI's Inspect AI (github.com/UKGovernmentBEIS/
inspect_ai) is the de-facto frontier eval framework, adopted by Anthropic,
DeepMind, and others, so peer auditors *already have the tooling* to run a
ai_crucible task. This module maps a ai_crucible puzzle (:class:`~ai_crucible.types.
PuzzleMeta` + the Solver-facing prompt) onto the Inspect ``Task`` shape.

The mapping uses the installed ``inspect_ai`` package's real types where natural:
the per-puzzle item becomes an Inspect :class:`inspect_ai.dataset.Sample`
(``input`` = the prompt, ``id`` = the puzzle id, ``metadata`` = the capability
aspect / class / tier / budgets), and the result is returned as a plain dict
(``model_dump`` of the Sample plus a scorer *reference* and the budget limits).
A dict — rather than a constructed :class:`inspect_ai.Task` object — is the right
return type here because:

1. A real ``Task`` needs a bound ``solver`` and ``scorer`` callable; in ai_crucible
   those are the kernel's Solver role and the sealed out-of-band oracle (§10.4),
   which are NOT this module's to instantiate (exclusive ownership) and must NOT
   be importable from the Solver side anyway (the oracle is sealed).
2. The §9.6 deliverable is the *task definition format* — the
   serialisable, auditable description — which is exactly a dict. An auditor
   materialises it into a live ``Task`` on their side with their own model + the
   published scorer.

The crucial sealed-oracle invariant (§10.4, §1 SWE-bench hidden-oracle pattern)
is honored here: the returned task definition carries a scorer **reference**
(a name/uri), never the oracle assertions themselves. The Solver sees ``input``;
it never sees ``target`` (left empty) or the grading logic.

:func:`two_repo_layout` documents the §9.6 ``ai_crucible-harness`` + ``ai_crucible-
results`` split (the SEPARATE-REPOS concern) as data, without creating any repo.

Standards compliance (the six — workflow-standards.md):
- PIN_PER_STEP — 3: :func:`to_inspect_task` is a pure function of
  ``(puzzle_meta, prompt)``; the same inputs serialise to the same dict, and the
  dict pins the budgets + scorer ref + puzzle id so a run is replayable.
- ANDON_AUTHORITY — 2: raises a structured ``InspectTaskError`` on an empty
  prompt or a puzzle meta that cannot supply a scorer reference — a malformed
  task definition halts here rather than producing an unscoreable Inspect task.
- NAMED_COMPENSATORS — n/a: pure in-memory construction, no irreversible action.
- DECOMPOSE_BY_SECRETS — 3: the *public* task surface (prompt, id, budgets,
  scorer reference) is built here; the *secret* (oracle assertions) is decomposed
  entirely away onto the grading side and only referenced by name — the module
  structurally cannot leak it because it never holds it.
- UNCERTAINTY_GATED_HUMANS — n/a: no human checkpoint in a format builder.
- EXTERNAL_VERIFIER — 3: the whole point — the task definition is the artifact an
  *external* auditor loads into their own Inspect install with a *different*
  model to reproduce ai_crucible's numbers without trusting ai_crucible's harness.
"""

from __future__ import annotations

import warnings
from importlib.metadata import PackageNotFoundError, version
from typing import Any

from inspect_ai.dataset import Sample

from ai_crucible.types import PuzzleMeta

__all__ = [
    "InspectTaskError",
    "INSPECT_AI_TESTED_RANGE",
    "to_inspect_task",
    "two_repo_layout",
]

# The inspect-ai version range this module's Sample.model_dump() consumption was
# authored and tested against. pyproject pins `inspect-ai>=0.3` with NO upper
# bound (a deliberate resolver constraint, not a runtime dep — see pyproject), so
# the Sample shape this module serialises wholesale can drift under us. Per
# arXiv:2510.25506 silent dependency-version drift is a leading cause of
# "functional" eval artifacts failing months later; a runtime warning makes the
# drift legible instead of producing a misparsed task definition silently. We
# WARN (not raise) because a newer Inspect is usually compatible — the §9.6
# deliverable is the dict shape, and the defensive shape guard below catches an
# actual incompatibility. Bump the upper bound here when re-tested against it.
INSPECT_AI_TESTED_RANGE: tuple[str, str] = ("0.3", "0.4")

# The Sample.model_dump() keys this module's task definition depends on. If a
# future Inspect renames/drops one of these, the wholesale-consumed shape is
# broken and we surface it (ANDON) rather than emitting an unscoreable task.
_REQUIRED_SAMPLE_KEYS: tuple[str, ...] = ("id", "input", "target", "metadata")


class InspectTaskError(Exception):
    """Raised when a ai_crucible puzzle cannot be mapped to an Inspect task
    definition. Structured ``[CODE] message (hint: ...)`` payload (Ship-Gate-B)."""


def _fail(code: str, message: str, hint: str) -> InspectTaskError:
    return InspectTaskError(f"[{code}] {message} (hint: {hint})")


def _parse_minor(ver: str) -> tuple[int, int]:
    """Parse the ``major.minor`` prefix of a version string to an int 2-tuple.

    Tolerant of pre-release/build suffixes (``0.3.233`` -> ``(0, 3)``,
    ``0.4.0rc1`` -> ``(0, 4)``). Returns ``(-1, -1)`` if the prefix is
    unparseable so the caller treats it as "outside range" and warns rather than
    crashing on a surprise version format.
    """
    parts = ver.split(".")
    try:
        major = int("".join(c for c in parts[0] if c.isdigit()) or "x")
        minor = int("".join(c for c in parts[1] if c.isdigit()) or "x") if len(parts) > 1 else 0
    except ValueError:
        return (-1, -1)
    return (major, minor)


def _check_inspect_version() -> None:
    """Warn (once-ish, via the warnings filter) if the installed ``inspect_ai`` is
    outside :data:`INSPECT_AI_TESTED_RANGE`.

    The Sample.model_dump() shape is consumed wholesale; an untested Inspect
    version may have drifted it. This makes the drift legible (the §9.6
    reproducibility concern) without hard-failing on a probably-compatible newer
    release — the shape guard in :func:`to_inspect_task` is the actual safety net.
    """
    try:
        installed = version("inspect-ai")
    except PackageNotFoundError:  # pragma: no cover - inspect_ai is importable
        return
    lo = _parse_minor(INSPECT_AI_TESTED_RANGE[0])
    hi = _parse_minor(INSPECT_AI_TESTED_RANGE[1])
    cur = _parse_minor(installed)
    if cur == (-1, -1) or not (lo <= cur < hi):
        warnings.warn(
            f"[CONFIG_INSPECT_AI_UNTESTED_VERSION] inspect-ai {installed} is "
            f"outside the tested range "
            f"[{INSPECT_AI_TESTED_RANGE[0]}, {INSPECT_AI_TESTED_RANGE[1]}); the "
            "Sample.model_dump() task shape may have drifted (hint: re-validate "
            "to_inspect_task against this version and bump "
            "INSPECT_AI_TESTED_RANGE, or pin inspect-ai in pyproject — §9.6 "
            "reproducibility, arXiv:2510.25506 version drift)",
            stacklevel=2,
        )


def _scorer_ref(puzzle_meta: PuzzleMeta) -> str:
    """Build the scorer *reference* for this puzzle.

    A reference, never the oracle itself (sealed-oracle invariant, §10.4). It
    names the ai_crucible oracle scorer and the puzzle whose hidden assertions it
    will apply on the grading side. The convention mirrors Inspect's
    ``module/scorer`` registry-name style so an auditor can resolve it against the
    published ``ai_crucible-harness`` package.
    """
    return f"ai_crucible/oracle_scorer#{puzzle_meta.puzzle_id}"


def to_inspect_task(puzzle_meta: PuzzleMeta, prompt: str) -> dict[str, Any]:
    """Map a ai_crucible puzzle to an Inspect-AI task-definition dict (§9.6).

    Args:
        puzzle_meta: the validated per-puzzle contract (:class:`~ai_crucible.types.
            PuzzleMeta`).
        prompt: the Solver-facing prompt (Tier-1 scored context). What the Solver
            sees; it carries no oracle/answer content.

    Returns:
        A JSON-serialisable dict with:
            ``dataset``: a single-element list of the Inspect ``Sample``
                serialisation (``model_dump``) — ``input`` = ``prompt``,
                ``id`` = ``puzzle_id``, ``target`` left empty (the Solver never
                sees the answer), ``metadata`` = capability aspect / class / tier.
            ``scorer``: the scorer *reference* string (sealed oracle, §10.4) —
                resolved on the grading side, never inlined.
            ``sandbox``: the Inspect sandbox spec shape ``["docker", null]`` —
                a placeholder the harness fills with the digest-pinned per-puzzle
                image (§10.4); kept as a 2-tuple so it matches
                ``SandboxEnvironmentSpec(type, config)``.
            ``limits``: the budget limits mapped onto Inspect's task limits —
                ``message_limit`` (from ``tool_call_budget``),
                ``time_limit`` (from ``time_budget_seconds``).
            ``epochs``: ``min_k`` — Inspect's epochs is ai_crucible's k sibling
                attempts per puzzle (pass^k native, §10.2).
            ``metadata``: puzzle-level metadata echoed for the task record
                (puzzle id, capability aspect, catalog tier, puzzle class).

        The ``Sample`` is constructed as a *real* ``inspect_ai`` object then
        serialised, so the shape is guaranteed compatible with the installed
        Inspect version rather than hand-faked.

    Raises:
        InspectTaskError: empty prompt (a task with no Solver-facing content is
            not a task).
    """
    if not isinstance(prompt, str) or not prompt.strip():
        raise _fail(
            "INPUT_TASK_EMPTY_PROMPT",
            f"puzzle '{puzzle_meta.puzzle_id}' has an empty Solver prompt",
            "pass the Tier-1 scored-context prompt the Solver will see",
        )

    # Make any inspect-ai version drift legible before we consume its shape.
    _check_inspect_version()

    sample = Sample(
        input=prompt,
        target="",  # sealed-oracle invariant: the Solver never sees the answer
        id=puzzle_meta.puzzle_id,
        metadata={
            "capability_aspect": puzzle_meta.capability_aspect,
            "puzzle_class": puzzle_meta.puzzle_class.value,
            "catalog_tier": puzzle_meta.catalog_tier.value,
            "source_url": puzzle_meta.source_url,
        },
    )

    # Defensive guard around the wholesale-consumed shape: a future Inspect that
    # renames/drops a key the task definition relies on must fail here with a
    # legible, model-named error, not silently emit a task missing `id`/`input`
    # that an auditor's tooling later misparses (the inspect-future-deps-002 gap).
    sample_dict = sample.model_dump()
    if not isinstance(sample_dict, dict):
        raise _fail(
            "STATE_INSPECT_SAMPLE_SHAPE",
            f"inspect_ai Sample.model_dump() returned "
            f"{type(sample_dict).__name__}, not a dict, for puzzle "
            f"'{puzzle_meta.puzzle_id}'",
            "the installed inspect-ai version produced an unexpected Sample "
            "serialisation; re-validate against INSPECT_AI_TESTED_RANGE (§9.6)",
        )
    missing = [k for k in _REQUIRED_SAMPLE_KEYS if k not in sample_dict]
    if missing:
        raise _fail(
            "STATE_INSPECT_SAMPLE_SHAPE",
            f"inspect_ai Sample.model_dump() is missing key(s) {missing} for "
            f"puzzle '{puzzle_meta.puzzle_id}'",
            "the installed inspect-ai version drifted the Sample shape this task "
            "definition consumes; re-validate to_inspect_task and bump "
            "INSPECT_AI_TESTED_RANGE (§9.6 reproducibility)",
        )

    return {
        "name": f"ai_crucible-{puzzle_meta.puzzle_id}",
        "dataset": [sample_dict],
        # Scorer reference only — the oracle is sealed and graded out-of-band.
        "scorer": _scorer_ref(puzzle_meta),
        # Inspect SandboxEnvironmentSpec(type, config) shape; harness pins the
        # digest into `config` at run time (§10.4).
        "sandbox": ["docker", None],
        "limits": {
            # Inspect counts model/tool turns via message_limit; ai_crucible's tool
            # budget is the operative cap (§8.4 BATS displayed budget).
            "message_limit": puzzle_meta.tool_call_budget,
            "time_limit": puzzle_meta.time_budget_seconds,
        },
        # k sibling attempts per puzzle -> Inspect epochs (pass^k native, §10.2).
        "epochs": puzzle_meta.min_k,
        "metadata": {
            "puzzle_id": puzzle_meta.puzzle_id,
            "capability_aspect": puzzle_meta.capability_aspect,
            "catalog_tier": puzzle_meta.catalog_tier.value,
            "puzzle_class": puzzle_meta.puzzle_class.value,
            "point_threshold": puzzle_meta.point_threshold,
        },
    }


def two_repo_layout() -> dict[str, Any]:
    """Describe the §9.6 two-repo release split as data (no repos created).

    Per METR's vivaria + eval-analysis-public + HCAST pattern
    (metr.org/blog/2025-03-19-measuring-ai-ability-to-complete-long-tasks): the
    harness (which costs API budget to re-run) is separated from the results
    (which auditors verify statistically without paying for inference). This is
    the SEPARATE-REPOS concern — documented here per the build guidance, NOT
    actioned (creating git repos is out of this module's scope).

    Returns:
        A dict describing each repo's purpose and contents, suitable for dropping
        into release docs or a Phase-10 checklist.
    """
    return {
        "pattern": "two-repo release (METR vivaria/eval-analysis-public/HCAST)",
        "rationale": "auditors verify statistical claims without paying for inference",
        "repos": {
            "ai_crucible-harness": {
                "purpose": "runs trials; costs API budget to replicate",
                "contains": [
                    "kernel (puzzle_loader, sandbox, roles, budget_governor, "
                    "oracle_scorer, judge_panel, trace_writer, observability, "
                    "attestation)",
                    "rubric.bundle (content-hashed, §9.1)",
                    "Inspect-AI task definitions (this module's output)",
                    "SUT.yaml per release (§9.6)",
                    "digest-pinned per-puzzle Docker images",
                ],
            },
            "ai_crucible-results": {
                "purpose": "raw outputs + analysis that regenerate every figure",
                "contains": [
                    "raw per-trial JSON (EvalLog shape)",
                    "analysis notebooks (regenerate all figures)",
                    "AsPredicted pre-registration + filled REFORMS checklist",
                    "TUNING.md provenance + Sobol report + BO trace",
                    "Datasheet per split (calibration/dev/validation/private_test)",
                    "RFC 3161 timestamps + in-toto attestations + Rekor inclusion proofs",
                ],
            },
        },
    }
