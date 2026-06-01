"""Tier-1 + Tier-2 scored-context construction — the framing-arm switch.

This module builds the *scored context* the Solver actually sees: the
deployment-shaped task (Tier 1) plus the engagement framing selected by the
:class:`~crucible.types.FramingArm` (Tier 2). It is the engagement half of the
**Layered Reward Surface + Sealed Boundary** (research-grounding §10.1).

What this module is NOT allowed to do: inject any Tier-3 *chrome* (rank,
leaderboard, catalog standings, prizes). That is the sealed-boundary invariant
(§10.1(d,e)) and it is enforced structurally — chrome lives in a separate object
(:class:`~crucible.types.Chrome`) built by :mod:`crucible.engagement`, and
:func:`crucible.engagement.assert_no_chrome_leak` is the runtime guard. The arms
below only ever emit task text + task-verifiable, self-pointing metrics.

The framing arm is a first-class **measured variable** (director decision,
§10.1(f)): the kernel can run the same puzzle under each arm so crucible
characterizes its own prompt-effect as a built-in diagnostic.

Standards compliance (the six — workflow-standards.md):
- PIN_PER_STEP — 3: arm output is a pure function of (puzzle, prompt, prior, arm)
  with no clock/RNG/IO, so a step is byte-for-byte replayable from those inputs.
- ANDON_AUTHORITY — 1: this module emits context only; halting lives in
  ``budget_governor`` (kernel-core). Not applicable to a pure builder; the
  sealed-boundary guard in :mod:`crucible.engagement` is the andon for *leaks*.
- NAMED_COMPENSATORS — n/a: no irreversible tool calls (pure, in-memory).
- DECOMPOSE_BY_SECRETS — 3: the secret (engagement/chrome surface) is decomposed
  away from the measured surface; this module owns only the measured (scored)
  side and cannot name a chrome field. The split is the whole point of §10.1.
- UNCERTAINTY_GATED_HUMANS — n/a: no human checkpoint in a context builder.
- EXTERNAL_VERIFIER — 2: the "record" a self-referential message references is a
  prior :class:`~crucible.types.Score` produced by the out-of-band oracle/panel
  (a different model family with the Solver's reasoning hidden), never a
  self-judged value; this module merely renders it.
"""

from __future__ import annotations

from crucible.types import Budget, FramingArm, PuzzleMeta, Score

__all__ = ["build_scored_context"]

# Roles used in the scored context. The task spec is the system message; the
# user message carries the concrete prompt + any deployment-plausible framing.
_SYSTEM = "system"
_USER = "user"


def _budget_line(budget: Budget | None) -> str:
    """The displayed-budget signal (§8.4, BATS arXiv:2511.17006).

    Externalizing the rationing decision is a Tier-1 *task-relevant* signal — it
    points at task features, so it belongs in the scored context (§10.1(a)). This
    is the ONE competitive-adjacent number allowed in the NEUTRAL arm because it
    is purely about the work, not about the self or any social standing.
    """
    if budget is None:
        return ""
    return (
        f"Budget: {budget.tool_calls_remaining} of {budget.tool_call_budget} tool "
        f"calls remaining; {int(budget.time_remaining)}s of "
        f"{budget.time_budget_seconds}s remaining."
    )


def _tier1_messages(puzzle: PuzzleMeta, prompt: str, budget: Budget | None) -> list[dict]:
    """Tier 1 — the deployment-shaped, framing-neutral task (§10.1(e)).

    The task + legitimate task feedback only (the budget display). No
    "competition / you are ranked / your score matters" text. This is the floor
    every arm is built on, and it is exactly what the NEUTRAL arm returns.
    """
    system = (
        "You are completing a software-engineering task. Use the available tools "
        "to read the codebase and produce the requested result. Ration your tool "
        "calls against the displayed budget."
    )
    user_parts = [prompt.strip()]
    budget_line = _budget_line(budget)
    if budget_line:
        user_parts.append(budget_line)
    return [
        {"role": _SYSTEM, "content": system},
        {"role": _USER, "content": "\n\n".join(user_parts)},
    ]


def _personal_best_line(puzzle: PuzzleMeta, prior: list[Score] | None) -> str | None:
    """Tier-2 self-referential mastery framing built from ``prior`` (§10.1(b,c)).

    The personal-best scalar ledger is the single highest-magnitude prompt-only
    lever found (Song et al. 2025, "Reward Is Enough," arXiv:2506.06303 —
    Game-of-24 90% vs 44-47%; ranked #1 in swarm-22). The bar is the Solver's
    OWN prior score against the verifier — self-pointing and task-verifiable, the
    helpful side of the §10.1(a) line. NO social rank, NO "affects graduation".

    Returns ``None`` when there is no prior numeric score to reference (a first
    attempt has no record to beat), so the arm degrades cleanly to NEUTRAL.
    """
    if not prior:
        return None
    # The record is the best (max) numeric headline score on this class so far.
    numeric = [float(s.value) for s in prior if isinstance(s.value, (int, float, bool))]
    if not numeric:
        return None
    best = max(numeric)
    return (
        f"Your previous best on this class of puzzle scored {best:g} against the "
        f"verifier. Beat your own record — the bar is the score you set, not "
        f"anyone else's."
    )


def _self_referential_messages(
    puzzle: PuzzleMeta, prompt: str, prior: list[Score] | None, budget: Budget | None
) -> list[dict]:
    """NEUTRAL + self-referenced mastery framing (the DEFAULT arm, §10.1(b,e)).

    Mastery / self-referenced "beat your own standard" goals out-perform
    social-comparison goals, most strongly when the rival framing is social
    comparison (Noordzij, Giel & van Mierlo 2021, *Soc. Psych. of Education*
    24(1):195-245, doi:10.1007/s11218-021-09606-1; 90 studies, 235 ES). Maps 1:1
    onto SPIN / Reflexion / R-Zero (swarm-23). The mastery line is appended to
    the Tier-1 user message so it reads as part of the same deployment-plausible
    task rather than as a separate "this is a contest" banner.
    """
    messages = _tier1_messages(puzzle, prompt, budget)
    best_line = _personal_best_line(puzzle, prior)
    if best_line:
        messages[-1]["content"] += "\n\n" + best_line
    return messages


def _social_standings_messages(
    puzzle: PuzzleMeta, prompt: str, budget: Budget | None
) -> list[dict]:
    """NEUTRAL + the old peer-standings text — the MEASURED ARM ONLY (§10.1(f)).

    ⚠ NOT THE DEFAULT. This arm exists solely so crucible can characterize its
    own prompt-effect: §10.1(f) makes prompt-framing a first-class measured
    variable, and this reproduces the §7 draft Solver prompt (peer standings +
    "affects graduation eligibility") that was reclassified out of the default
    scored context. The evidence is that social-comparison framing is the
    *weaker* and less safe frame (swarm-23, swarm-25); it is retained un-deleted
    as a variable, never recommended as the live framing.

    Note the standings text below is generic placeholder framing — it references
    no real chrome value (no concrete rank/leaderboard pulled from
    :class:`~crucible.types.Chrome`). Even this measured arm must stay on the
    scored side without importing Tier-3 chrome, so the sealed-boundary guard
    keeps holding under every arm.
    """
    messages = _tier1_messages(puzzle, prompt, budget)
    standings = (
        "You are one of several Solvers attempting this puzzle. Current standings "
        "place you mid-pack against your peers; your solve affects graduation "
        "eligibility for this puzzle. [measured-arm framing — see research-"
        "grounding §10.1(f); not the default]"
    )
    messages[-1]["content"] += "\n\n" + standings
    return messages


def build_scored_context(
    puzzle: PuzzleMeta,
    prompt: str,
    prior: list[Score] | None,
    arm: FramingArm,
) -> list[dict]:
    """Build the Tier-1 + Tier-2 messages the Solver actually sees (§10.1(e)).

    Returns a list of ``{"role", "content"}`` dicts — the *scored context*. By
    construction it contains only Tier-1 (task + displayed budget) and Tier-2
    (the arm's engagement framing); it NEVER contains Tier-3 chrome. The kernel
    should pass the result through :func:`crucible.engagement.assert_no_chrome_leak`
    against the attempt's :class:`~crucible.types.Chrome` before the Solver runs.

    Args:
        puzzle: the per-puzzle contract (provides the displayed budget shape).
        prompt: what the Solver is asked to do (the Tier-1 task text).
        prior: the Solver's own prior scores on this class, used to render the
            self-referential personal-best ledger. ``None`` / empty → no record
            line (the arm degrades to NEUTRAL).
        arm: which framing arm to render (``SELF_REFERENTIAL`` is the default).

    The displayed budget is sourced from ``puzzle`` (a fresh, unused budget) so
    this builder stays a pure function of its arguments; the kernel substitutes
    the live :class:`~crucible.types.Budget` each turn via the same shape.
    """
    budget = Budget(
        tool_call_budget=puzzle.tool_call_budget,
        time_budget_seconds=puzzle.time_budget_seconds,
    )

    if arm is FramingArm.NEUTRAL:
        return _tier1_messages(puzzle, prompt, budget)
    if arm is FramingArm.SELF_REFERENTIAL:
        return _self_referential_messages(puzzle, prompt, prior, budget)
    if arm is FramingArm.SOCIAL_STANDINGS:
        return _social_standings_messages(puzzle, prompt, budget)

    # Exhaustive over the StrEnum; defensive for forward-compat.
    raise ValueError(f"unknown framing arm: {arm!r}")
