"""Stage 2 — the content-hashed rubric bundle (research-grounding §9.1, §9.4).

The §9.1 animating principle: *a tuned crucible reports the protocol, not the
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
proof) binds to (§9.5). :func:`bump_on_change` enforces the §9.1 invariant: the
version changes iff the content hash changes, so two byte-identical bundles can
never carry two different versions and a changed bundle can never silently keep
the old version.

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
- UNCERTAINTY_GATED_HUMANS — 2: :func:`bump_on_change` is the gate that forces a
  human-visible version increment exactly when (and only when) the scoring
  content changed; it frames the decision contrastively (returns old vs new
  version with the hash as evidence).
- EXTERNAL_VERIFIER — 3: the content hash is verifiable by *any* third party with
  the canonical bytes and a SHA-256 implementation — the verification needs no
  trust in crucible and no access to crucible's reasoning. This is the §9.6
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
    """Return the version ``new`` should carry, enforcing the §9.1 invariant.

    "No silent retconning": the version changes **iff** the content hash changes.

    - If ``new``'s scoring content hash equals ``old``'s, the bundles are the
      same instrument; this returns ``old.version`` (the version does not advance
      for a no-op edit, and ``new`` should keep the old label).
    - If the hashes differ, the content changed and the version must advance;
      this returns a new version string derived from ``old.version`` (e.g.
      ``"v1.0"`` -> ``"v1.1"``; an unparseable label gets a ``"+1"`` suffix).

    The caller assigns the returned string to ``new.version``. This function does
    not mutate either argument (pure), so the decision is auditable: the returned
    string plus the two hashes are the full justification.

    Raises:
        RubricBundleError: if either bundle's content cannot be compiled.
    """
    old_hash, _ = compile_bundle(old)
    new_hash, _ = compile_bundle(new)
    if old_hash == new_hash:
        # Same instrument — version stays put (and new should adopt old's label).
        return old.version
    return _next_version(old.version)


def _next_version(version: str) -> str:
    """Derive the next version label from ``version``.

    Recognises a trailing dotted numeric component and increments the last
    segment (``"v1.0"`` -> ``"v1.1"``, ``"1.2.3"`` -> ``"1.2.4"``,
    ``"v3"`` -> ``"v4"``). Anything it can't parse gets a ``"+1"`` suffix so the
    label still provably changes (the invariant is "version changes when hash
    changes"; the exact scheme is a convention, the change is the contract).
    """
    stripped = version.strip()
    if not stripped:
        return "v1"

    # Find the trailing run of [0-9.] and bump its final numeric segment.
    i = len(stripped)
    while i > 0 and (stripped[i - 1].isdigit() or stripped[i - 1] == "."):
        i -= 1
    prefix, tail = stripped[:i], stripped[i:]

    if not tail or tail in {".", ""}:
        return f"{stripped}+1"

    segments = tail.split(".")
    # Bump the last non-empty numeric segment.
    for idx in range(len(segments) - 1, -1, -1):
        if segments[idx].isdigit():
            segments[idx] = str(int(segments[idx]) + 1)
            return prefix + ".".join(segments)
    return f"{stripped}+1"
