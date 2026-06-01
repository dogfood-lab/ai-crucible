"""Tier-3 chrome + the sealed-boundary guard.

This module owns the *un-scored* side of the **Sealed Boundary** (research-
grounding §10.1(d,e)): the engagement/competitive chrome (rank, leaderboard,
catalog standings, prizes) that motivates the human watching the run, and the
runtime guard that proves this chrome never leaks into the context window the
model solves in.

The animating principle (swarm-25): *a stake the model can perceive influencing
is a stake it will optimize toward, so every motivating signal must live strictly
on the un-scored side of a one-way boundary.* Framing alone moves choices
4.3%→68.8% (Wang & Zhang 2026, arXiv:2603.19282); a single prompt change moves
cheating 92%→1% (Zhong et al. 2025, ImpossibleBench, arXiv:2510.20270); eval-
awareness scales with capability and can't be scrubbed once cued (Chaudhary 2025,
arXiv:2509.13333). The only robust lever is *not emitting eval/competition cues on
the scored side* — which is exactly what :func:`assert_no_chrome_leak` enforces.

Standards compliance (the six — workflow-standards.md):
- PIN_PER_STEP — 3: :func:`build_chrome` is a pure constructor; the guard is a
  pure predicate over (messages, chrome). Both are byte-for-byte replayable.
- ANDON_AUTHORITY — 3: :func:`assert_no_chrome_leak` IS an andon — it halts the
  pipeline by raising :class:`SealedBoundaryViolation` the instant a chrome value
  is found in scored context; bad (contaminated) context never reaches the model.
  Proven RED in tests/test_engagement.py (a chrome rank injected into a message
  raises), per the "prove the gate goes RED" discipline.
- NAMED_COMPENSATORS — n/a: no irreversible tool calls (pure, in-memory).
- DECOMPOSE_BY_SECRETS — 3: chrome (the motivating secret) is decomposed into its
  own object and its own module, structurally separated from the measured surface
  in :mod:`crucible.framing`. The guard verifies the decomposition held at runtime.
- UNCERTAINTY_GATED_HUMANS — n/a.
- EXTERNAL_VERIFIER — 2: chrome is rendered from scores the out-of-band oracle/
  panel produced; this module never self-scores. The boundary it protects is what
  *lets* the external verifier stay external (motivation can't reach the grader).
"""

from __future__ import annotations

import re
from typing import Any

from crucible.types import Chrome

__all__ = ["SealedBoundaryViolation", "build_chrome", "assert_no_chrome_leak"]


class SealedBoundaryViolation(Exception):
    """Raised when Tier-3 chrome leaks into the scored context (§10.1(d,e)).

    This is a hard failure, not a warning: a single chrome value in a message the
    model solves in means motivation and measurement shared a context window, and
    the diagnostic for that attempt is contaminated. The kernel must treat this as
    an andon halt for the attempt, never swallow it.
    """


def build_chrome(
    rank: int | None = None,
    cohort_size: int | None = None,
    leaderboard: list[dict[str, Any]] | None = None,
    catalog_standing: dict[str, Any] | None = None,
) -> Chrome:
    """Build the Tier-3 :class:`~crucible.types.Chrome` for the human UI / records.

    Chrome is the rank/leaderboard/standings/prize surface that renders ONLY in
    the human-facing wrapper (§10.1(e), Tier 3). It is held on
    :class:`~crucible.types.AttemptState.chrome`, separate from ``messages``, and
    is never injected into a context window the model solves in — the separate
    object makes the boundary structural rather than aspirational (see the
    :class:`~crucible.types.Chrome` invariant). Pass it, together with the scored
    context, to :func:`assert_no_chrome_leak` to prove the separation held.
    """
    return Chrome(
        rank=rank,
        cohort_size=cohort_size,
        leaderboard=list(leaderboard) if leaderboard else [],
        catalog_standing=dict(catalog_standing) if catalog_standing else {},
    )


def _stringify(value: Any) -> list[str]:
    """Flatten a chrome value into the non-empty string tokens it would render to.

    The guard must catch a leak no matter how chrome is serialized into a prompt —
    a bare ``rank`` int, a leaderboard row's solver id, a catalog-standing percentile
    string. So we walk dicts/lists/scalars recursively and collect every leaf rendered
    as ``str``.

    Two deliberate exclusions, learned from building the guard against realistic
    chrome shapes:

    * **Dict keys are NOT tokenized — only values are.** A leaderboard row is
      ``{"solver": "solver-zeta", "score": 99}``; the *keys* ("solver", "score")
      are schema field names, not chrome data. Matching them would (a) raise on the
      generic word "score"/"solver" appearing in any prompt and (b) mask the real
      leaked value behind a structural label. The data that must not leak is the
      values, so we recurse into values only.
    * **Empty / ``None`` leaves contribute nothing**, so an unset chrome field (its
      dataclass default) can never cause a false positive. ``0`` / ``False`` ARE
      kept (they are real displayable values).
    """
    tokens: list[str] = []
    if value is None:
        return tokens
    if isinstance(value, dict):
        for v in value.values():  # values only — keys are schema labels, not data.
            tokens.extend(_stringify(v))
        return tokens
    if isinstance(value, (list, tuple, set)):
        for item in value:
            tokens.extend(_stringify(item))
        return tokens
    # Scalar leaf.
    s = str(value).strip()
    if s:
        tokens.append(s)
    return tokens


def _chrome_tokens(chrome: Chrome) -> list[str]:
    """Every non-empty string token a chrome object could leak as.

    A field at its dataclass default (``rank=None``, empty ``leaderboard``/
    ``catalog_standing``) contributes nothing — only chrome that was actually
    populated can trip the guard, so a benign empty Chrome never blocks a clean
    context.
    """
    tokens: list[str] = []
    tokens.extend(_stringify(chrome.rank))
    tokens.extend(_stringify(chrome.cohort_size))
    tokens.extend(_stringify(chrome.leaderboard))
    tokens.extend(_stringify(chrome.catalog_standing))
    return tokens


def assert_no_chrome_leak(messages: list[dict], chrome: Chrome) -> None:
    """THE LOAD-BEARING GUARD — raise if Tier-3 chrome appears in scored context.

    Walks every message's ``content`` and raises :class:`SealedBoundaryViolation`
    if any non-empty chrome value (rank / cohort_size / leaderboard rows / catalog
    standing, each rendered to its string tokens) is found as a substring of any
    message content. This is the §10.1(d,e) sealed boundary made executable:
    motivation lives in chrome, measurement context stays clean, and the two never
    share a context window.

    Matching is **word-boundary** on the rendered string form, so it catches a
    leak however the value was interpolated into a prompt (``f"rank {chrome.rank}"``,
    a templated leaderboard row, a JSON blob) while not firing on an incidental
    substring (chrome ``rank=7`` does not match the ``7`` inside ``17``). An empty /
    default Chrome has no tokens and so always passes — only populated chrome can
    trip the guard.

    **Residual ambiguity (intentional, fail-closed).** A *bare numeric* chrome
    value that happens to equal a number legitimately displayed in the Tier-1 task
    (e.g. ``cohort_size`` == the puzzle's ``tool_call_budget``) is genuinely
    indistinguishable from a leak by string inspection alone. The guard resolves
    this conservatively — it raises — because a sealed-boundary check must fail
    toward "possible leak" over "missed leak". In practice the high-signal leak
    vector is a *string-rendered* chrome value (a solver id, a badge, a "rank N of
    M" phrase, a percentile label), which is what realistic chrome carries.

    Args:
        messages: the scored context (Tier-1 + Tier-2) from
            :func:`crucible.framing.build_scored_context`.
        chrome: the Tier-3 chrome that must stay out of ``messages``.

    Raises:
        SealedBoundaryViolation: on the first chrome token found in any content,
            naming the offending token and role so the kernel andon log is precise.
    """
    tokens = _chrome_tokens(chrome)
    if not tokens:
        return  # nothing populated → nothing can leak.

    for index, message in enumerate(messages):
        content = message.get("content", "")
        if not isinstance(content, str):
            content = str(content)
        for token in tokens:
            # Word-boundary match: catches the token however it was interpolated,
            # but not as an incidental substring of a larger token/number.
            if re.search(rf"(?<!\w){re.escape(token)}(?!\w)", content):
                role = message.get("role", "?")
                raise SealedBoundaryViolation(
                    f"Tier-3 chrome value {token!r} leaked into scored context "
                    f"(message[{index}], role={role!r}). Motivation must stay in "
                    f"chrome; the scored context must remain deployment-shaped and "
                    f"framing-neutral (research-grounding §10.1(d,e))."
                )
