"""Stage 2 — the content-hashed rubric bundle (research-grounding §9.1, §9.4).

The §9.1 animating principle: *a tuned ai_crucible reports the protocol, not the
weights — and the release is not "v1.0 weights" but ``v1.0/rubric.bundle.sha256``
plus the documented protocol that produced it.* This module is the compiler for
that artifact.

Grounding:
- **Hong et al. 2026 — "RULERS: Locked Rubrics and Evidence-Anchored Scoring"**
  (arXiv:2601.08654). The rubric is compiled to a content-hashed bundle; the
  leaderboard records ``(model_id, score, bundle_hash)``. A new bundle hash is a
  new instrument — "no silent retconning" (§9.1).
- **Cawley & Talbot 2010 — "On Over-fitting in Model Selection and Subsequent
  Selection Bias"** (JMLR 11:2079-2107). Tuning that is not pinned to a hash is
  selection bias laundered as a single number.

A :class:`RubricBundle` holds the four tunable surfaces — penalty/component
``weights``, gate ``thresholds``, ``judge_prompts``, and a human-readable
``version`` string. :func:`compile_bundle` canonicalises it to RFC-8785-style
JSON and returns ``(sha256_hex, canonical_bytes)``; the hash is the anchor every
downstream attestation (in-toto predicate, RFC 3161 timestamp, Rekor inclusion
proof) binds to (§9.5). The §9.1 invariant — *the version changes iff the
content hash changes*, so two byte-identical bundles can never carry two
different versions and a changed bundle can never silently keep the old version
— is split across two functions by intent: :func:`bump_on_change` *advises* the
label a changed bundle should adopt (a pure suggester, see its docstring), and
:func:`assert_versioned` *enforces* the invariant as a fail-closed andon — it
raises when content moved but the label did not, or when the label moved but the
content did not. The authoritative anti-retconning seal at rest is the recorded
``bundle_hash`` itself (the third element of every leaderboard record); the two
functions are the human-facing guardrails that keep the label honest *before*
that hash is recorded.

NOTE on the version field and the hash: the ``version`` string is deliberately
**excluded from the hashed material**. The hash is a fingerprint of the *scoring
content* (weights/thresholds/judge_prompts) — what actually changes a model's
score — not of the label we give it. This is what lets :func:`bump_on_change`
ask the well-posed question "did the scoring content change?" and answer it from
the hash alone. (If the version were hashed, every rename would look like a new
instrument and the invariant would be circular.)

Standards compliance (the six — workflow-standards.md):
- PIN_PER_STEP — 3: :func:`compile_bundle` is the embodiment of this standard —
  it turns a bundle into a byte-exact, hash-addressable artifact so a tuning run
  is replayable from ``bundle_hash`` alone. Pure function, no clock/RNG/IO.
- ANDON_AUTHORITY — 2: a defect (e.g. a bundle whose content cannot be
  canonicalised) raises a structured ``RubricBundleError`` at compile time rather
  than emitting an unhashable / non-reproducible artifact downstream.
- NAMED_COMPENSATORS — n/a: pure in-memory compilation, no irreversible tool
  call. (The irreversible act — publishing a bundle hash to the leaderboard /
  Rekor — lives in the attestation module and Phase-10 release, with their own
  compensators.)
- DECOMPOSE_BY_SECRETS — 3: the four tunable surfaces are grouped into one
  hashable object that changes together; the *version label* (which changes on a
  different cadence — human decision) is held as a separate, un-hashed field.
- UNCERTAINTY_GATED_HUMANS — 2: :func:`assert_versioned` is the gate that forces
  a human-visible version increment exactly when (and only when) the scoring
  content changed — it RAISES a structured ``RubricBundleError`` on an unbumped
  change (or a relabel of unchanged content), so the human cannot silently
  retcon a tuned instrument. :func:`bump_on_change` is the contrastive companion:
  it advises the label the change should adopt (old vs suggested-new, with the
  two hashes as evidence). The gate enforces; the suggester explains.
- EXTERNAL_VERIFIER — 3: the content hash is verifiable by *any* third party with
  the canonical bytes and a SHA-256 implementation — the verification needs no
  trust in ai_crucible and no access to ai_crucible's reasoning. This is the §9.6
  independent-verification primitive in its smallest form.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "RubricBundleError",
    "RubricBundle",
    "canonical_bundle_json",
    "compile_bundle",
    "bump_on_change",
    "assert_versioned",
]


class RubricBundleError(Exception):
    """Raised on a malformed rubric bundle (un-canonicalisable content, an
    un-bumpable version). Structured ``[CODE] message (hint: ...)`` payload per
    the repo's Ship-Gate-B error shape."""


def _fail(code: str, message: str, hint: str) -> RubricBundleError:
    return RubricBundleError(f"[{code}] {message} (hint: {hint})")


@dataclass
class RubricBundle:
    """The four tunable scoring surfaces plus a human-readable version label.

    Attributes:
        weights: penalty + component weights (e.g.
            ``{"answer_key_fetch": -150, "elegance_bonus_max": 24}``). The §8.3
            component bounds and §8.2 penalty flavors live in the puzzle metas;
            this is the *tuned* weight surface the §9.4 protocol searches over.
        thresholds: gate thresholds (e.g.
            ``{"point_threshold": 50, "solve_threshold": 0.8}``).
        judge_prompts: the cross-family panel prompts, keyed by role/use
            (e.g. ``{"novelty_validation": "...", "bypass_adjudication": "..."}``).
            Paraphrase-ablated in §9.4 step 4.
        version: the human-readable label (e.g. ``"v1.0"``). Excluded from the
            content hash by design (see module docstring); managed via
            :func:`bump_on_change`.
    """

    weights: dict[str, float] = field(default_factory=dict)
    thresholds: dict[str, float] = field(default_factory=dict)
    judge_prompts: dict[str, str] = field(default_factory=dict)
    version: str = "v0"

    def hashable_content(self) -> dict[str, Any]:
        """Return the scoring content that defines the bundle's identity.

        Deliberately omits :attr:`version` — the hash fingerprints the scoring
        behavior, not its label (module docstring). Keys are fixed and ordered by
        :func:`canonical_bundle_json` at serialization time.
        """
        return {
            "weights": self.weights,
            "thresholds": self.thresholds,
            "judge_prompts": self.judge_prompts,
        }


def canonical_bundle_json(bundle: RubricBundle) -> bytes:
    """Serialize ``bundle``'s scoring content to canonical (RFC-8785-style) JSON
    bytes: sorted keys, minimal separators, UTF-8.

    This is the exact byte sequence that gets hashed. Stable across processes and
    machines so the hash is reproducible by an independent verifier — the
    precondition for the §9.5 attestation chain. Mirrors the canonicalisation the
    sibling ``attestation`` module applies before hash-chaining.

    Raises:
        RubricBundleError: when the content is not JSON-serializable (e.g. a
            weight value that is not a number/str/None) — caught here so the
            defect surfaces at compile time, not at hash time.
    """
    try:
        text = json.dumps(
            bundle.hashable_content(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,  # NaN/Inf are not valid JSON and break reproducibility
        )
    except (TypeError, ValueError) as exc:
        raise _fail(
            "INPUT_BUNDLE_UNSERIALIZABLE",
            f"rubric bundle content is not canonicalisable: {exc}",
            "weights/thresholds must be finite numbers and judge_prompts strings; "
            "no NaN/Inf, no custom objects",
        ) from exc
    return text.encode("utf-8")


def compile_bundle(bundle: RubricBundle) -> tuple[str, bytes]:
    """Compile ``bundle`` to ``(sha256_hex, canonical_bytes)`` (§9.4 step 7).

    ``sha256_hex`` is the 64-char lowercase hex digest of ``canonical_bytes``;
    ``canonical_bytes`` is :func:`canonical_bundle_json`. The hash is the
    ``rubric.bundle.sha256`` anchor for all downstream attestation (§9.5) and the
    third element of every leaderboard record ``(model_id, score, bundle_hash)``.

    Identical scoring content (regardless of ``version`` label) compiles to the
    same hash; any change to a weight, threshold, or judge prompt changes it.
    """
    canonical = canonical_bundle_json(bundle)
    digest = hashlib.sha256(canonical).hexdigest()
    return digest, canonical


def bump_on_change(old: RubricBundle, new: RubricBundle) -> str:
    """*Advise* the version ``new`` should carry under the §9.1 invariant.

    This is a pure **suggester**, not the seal — it computes a recommended label
    and returns it; it does NOT mutate either argument and does NOT force the
    caller to adopt the suggestion. The fail-closed enforcement of "no silent
    retconning" lives in :func:`assert_versioned`; the authoritative seal at rest
    is the recorded ``bundle_hash`` (module docstring). Use this to *propose* a
    label, then run :func:`assert_versioned` to *prove* the label is honest.

    - If ``new``'s scoring content hash equals ``old``'s, the bundles are the
      same instrument; the version need not advance. This returns
      ``new.version`` unchanged — including a deliberate relabel — so the
      caller's explicit choice is preserved (it does not silently revert to
      ``old.version``). (Whether a relabel of identical content is *allowed* is
      a separate policy question :func:`assert_versioned` answers.)
    - If the hashes differ, the content changed and the version must advance;
      this returns a new version string derived from ``old.version`` (e.g.
      ``"v1.0"`` -> ``"v1.1"``; an unparseable label gets a ``"+1"`` suffix).

    The caller assigns the returned string to ``new.version``. Because the
    function is pure, the decision is auditable: the returned string plus the two
    hashes are the full justification.

    Raises:
        RubricBundleError: if either bundle's content cannot be compiled.
    """
    old_hash, _ = compile_bundle(old)
    new_hash, _ = compile_bundle(new)
    if old_hash == new_hash:
        # Same instrument — no advance needed. Preserve the caller's chosen label
        # (do NOT silently revert an explicit relabel to old.version); a relabel
        # of identical content is a policy decision assert_versioned adjudicates.
        return new.version
    return _next_version(old.version)


def assert_versioned(old: RubricBundle, new: RubricBundle) -> str:
    """Enforce the §9.1 invariant as a fail-closed andon and return ``new``'s hash.

    "No silent retconning": the version label changes **iff** the content hash
    changes. This is the ENFORCING half of the invariant — where
    :func:`bump_on_change` merely advises, this gate RAISES when the label and
    the content disagree, so a changed instrument cannot be published or recorded
    under a stale label and an unchanged instrument cannot be relabeled to look
    like a new one. Call it on the path that records ``(model_id, score,
    bundle_hash)`` (release / leaderboard / Rekor), before the hash is committed.

    The two failure modes the gate refuses:

    - **content moved, label did not** (``STATE_RUBRIC_VERSION_NOT_BUMPED``): the
      tuned weights/thresholds/judge_prompts changed but ``new.version ==
      old.version`` — the classic silent retcon. The hint names the label
      :func:`bump_on_change` would have suggested.
    - **label moved, content did not** (``STATE_RUBRIC_VERSION_DRIFT``): two
      byte-identical bundles carry two different versions — a phantom "new
      instrument" that scores identically.

    On success returns ``compile_bundle(new)[0]`` — the hash the caller should
    record — so the gate doubles as the hash source on the happy path. Pure: it
    does not mutate either argument.

    Raises:
        RubricBundleError: on either disagreement above, or if either bundle's
            content cannot be compiled.
    """
    old_hash, _ = compile_bundle(old)
    new_hash, _ = compile_bundle(new)
    content_changed = old_hash != new_hash
    version_changed = new.version != old.version

    if content_changed and not version_changed:
        raise _fail(
            "STATE_RUBRIC_VERSION_NOT_BUMPED",
            f"scoring content changed (hash {old_hash[:12]} -> {new_hash[:12]}) "
            f"but version stayed {new.version!r} — §9.1 forbids silent retconning",
            f"bump the version (e.g. {bump_on_change(old, new)!r}) before "
            f"recording (model_id, score, bundle_hash)",
        )
    if version_changed and not content_changed:
        raise _fail(
            "STATE_RUBRIC_VERSION_DRIFT",
            f"version changed ({old.version!r} -> {new.version!r}) but scoring "
            f"content is byte-identical (hash {new_hash[:12]}) — two labels for "
            "one instrument",
            f"keep the label {old.version!r} for unchanged content, or actually "
            "change a weight/threshold/judge_prompt if a new instrument is meant",
        )
    return new_hash


def _next_version(version: str) -> str:
    """Derive the next version label from ``version``.

    Recognises a trailing dotted numeric component and increments the last
    segment (``"v1.0"`` -> ``"v1.1"``, ``"1.2.3"`` -> ``"1.2.4"``,
    ``"v3"`` -> ``"v4"``). Zero-padding is **preserved** so date-style /
    calendar-pinned labels stay sane: ``"2026.06"`` -> ``"2026.07"`` (not
    ``"2026.7"``), ``"v01"`` -> ``"v02"``; a carry that outgrows the pad width
    widens naturally (``"2026.09"`` -> ``"2026.10"``, ``"v1.09"`` -> ``"v1.10"``).
    Anything it can't parse cleanly — including a label with a trailing dangling
    dot like ``"v1."`` — gets a ``"+1"`` suffix so the label still provably
    changes (the invariant is "version changes when hash changes"; the exact
    scheme is a convention, the change is the contract).
    """
    stripped = version.strip()
    if not stripped:
        return "v1"

    # Find the trailing run of [0-9.] and bump its final numeric segment.
    i = len(stripped)
    while i > 0 and (stripped[i - 1].isdigit() or stripped[i - 1] == "."):
        i -= 1
    prefix, tail = stripped[:i], stripped[i:]

    # A tail that is empty, bare dots, or ends in a dangling '.' is not a clean
    # numeric label (e.g. "v1." -> tail "1.") — fall back to the +1 suffix rather
    # than emit a misleading partial bump like "v2.".
    if not tail or tail.strip(".") == "" or tail.endswith("."):
        return f"{stripped}+1"

    segments = tail.split(".")
    # Bump the last non-empty numeric segment, preserving its zero-pad width so a
    # calendar label ("2026.06") increments to "2026.07", not "2026.7".
    for idx in range(len(segments) - 1, -1, -1):
        if segments[idx].isdigit():
            width = len(segments[idx])
            segments[idx] = str(int(segments[idx]) + 1).zfill(width)
            return prefix + ".".join(segments)
    return f"{stripped}+1"
