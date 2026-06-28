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
  in :mod:`ai_crucible.framing`. The guard verifies the decomposition held at runtime.
- UNCERTAINTY_GATED_HUMANS — n/a.
- EXTERNAL_VERIFIER — 2: chrome is rendered from scores the out-of-band oracle/
  panel produced; this module never self-scores. The boundary it protects is what
  *lets* the external verifier stay external (motivation can't reach the grader).
"""

from __future__ import annotations

import dataclasses
import re
from typing import Any

from ai_crucible.types import Chrome

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
    """Build the Tier-3 :class:`~ai_crucible.types.Chrome` for the human UI / records.

    Chrome is the rank/leaderboard/standings/prize surface that renders ONLY in
    the human-facing wrapper (§10.1(e), Tier 3). It is held on
    :class:`~ai_crucible.types.AttemptState.chrome`, separate from ``messages``, and
    is never injected into a context window the model solves in — the separate
    object makes the boundary structural rather than aspirational (see the
    :class:`~ai_crucible.types.Chrome` invariant). Pass it, together with the scored
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

    Field-AGNOSTIC by construction: it walks EVERY dataclass field of ``chrome``
    (``dataclasses.fields``), not a hand-maintained allow-list. The ``Chrome`` docstring
    already names ``prizes`` as a plausible future Tier-3 signal; a new field added there
    must NOT be able to leak into scored context while the guard reports clean — the exact
    silently-misleading result an eval-integrity instrument must never produce, and the
    asymmetry with the already-field-agnostic :func:`_message_text` (which walks every
    message value). A field at its default (``rank=None``, empty ``leaderboard``/
    ``catalog_standing``) contributes nothing — only chrome that was actually populated can
    trip the guard, so a benign empty Chrome never blocks a clean context.
    """
    tokens: list[str] = []
    for field in dataclasses.fields(chrome):
        tokens.extend(_stringify(getattr(chrome, field.name)))
    return tokens


def _message_text(message: dict) -> list[str]:
    """Every string a message could leak chrome through — ALL its string-valued
    fields, not just ``content``.

    The scored context is NOT uniform-shaped: the Solver appends
    ``{role, content}`` messages, but the Critic appends
    ``{role, critique, anonymized}`` (roles.py ``Critic.message``) — its
    model-produced text lives under ``critique``, with no ``content`` key. Scanning
    only ``content`` left an entire class of scored-context message unscanned, so a
    chrome value in a critique bypassed both the role guard and the kernel
    re-check. We therefore mirror :func:`_stringify`'s value-walk and collect every
    string leaf of every field value — content, critique, or any future
    message-mutating site's key — so the boundary check covers the whole message
    however its text was carried. Scanning the ``role`` label value too is harmless
    (it can at worst match a generic word, never a leaked chrome value).
    """
    texts: list[str] = []
    for value in message.values():
        texts.extend(_stringify(value))
    return texts


def assert_no_chrome_leak(messages: list[dict], chrome: Chrome) -> None:
    """THE LOAD-BEARING GUARD — raise if Tier-3 chrome appears in scored context.

    Walks every string-valued field of every message (``content``, the Critic's
    ``critique``, or any other carried text — see :func:`_message_text`) and raises
    :class:`SealedBoundaryViolation` if any non-empty chrome value (rank /
    cohort_size / leaderboard rows / catalog standing, each rendered to its string
    tokens) is found in any of them. This is the §10.1(d,e) sealed boundary made
    executable: motivation lives in chrome, measurement context stays clean, and
    the two never share a context window.

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
            :func:`ai_crucible.framing.build_scored_context`.
        chrome: the Tier-3 chrome that must stay out of ``messages``.

    Raises:
        SealedBoundaryViolation: on the first chrome token found in any content,
            naming the offending token and role so the kernel andon log is precise.
    """
    tokens = _chrome_tokens(chrome)
    if not tokens:
        return  # nothing populated → nothing can leak.

    for index, message in enumerate(messages):
        # Scan EVERY string-valued field, not only ``content`` — the Critic's text
        # rides in ``critique`` (no ``content`` key), so a content-only scan left
        # that scored-context field unguarded (and the kernel post-Critic re-check
        # inherited the blind spot). The value-walk mirrors :func:`_stringify`.
        for content in _message_text(message):
            for token in tokens:
                # Word-boundary match: catches the token however it was
                # interpolated, but not as an incidental substring of a larger
                # token/number.
                if re.search(rf"(?<!\w){re.escape(token)}(?!\w)", content):
                    role = message.get("role", "?")
                    raise SealedBoundaryViolation(
                        f"[SEALED_BOUNDARY_LEAK] Tier-3 chrome value {token!r} leaked "
                        f"into scored context (message[{index}], role={role!r}): "
                        f"motivation and measurement shared a context window, so this "
                        f"attempt's diagnostic is contaminated (research-grounding "
                        f"§10.1(d,e)) "
                        f"(hint: keep chrome (rank/leaderboard/standings) out of the "
                        f"scored context — render it only in the human-facing wrapper "
                        f"and pass it via AttemptState.chrome, never interpolated into a "
                        f"message the model solves in)"
                    )
