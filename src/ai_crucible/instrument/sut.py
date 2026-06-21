"""Stage 4 — the System-Under-Test record, ``SUT.yaml`` (research-grounding §9.6).

Per **MLCommons MLPerf Inference Rules** (the strictest reproducibility regime in
industrial ML eval): every release ships a SUT description pinning the *exact*
system measured. The load-bearing discipline is **exact version strings, not
family aliases** — ``claude-opus-4-7-20260415``, never ``claude-opus``. A family
alias silently re-points as the provider updates the model, and per
arXiv:2510.25506 + arXiv:2512.00651 silent model-version drift causes >40% of
"functional" eval artifacts to fail within months. The SUT is **frozen from
submission through publication**.

:class:`SUT` holds the five §9.6-mandated fields; :func:`render_sut_yaml` emits a
deterministic ``SUT.yaml`` whose fields round-trip exactly (the value an auditor
reads back is the value that was recorded). YAML is rendered by hand — stable key
order, quoted scalars — so the output is byte-deterministic and the module needs
no YAML dependency (mirroring the canonical-JSON discipline in the ``attestation``
and ``rubric_bundle`` modules). :func:`parse_sut_yaml` reads it back so the
round-trip is testable and an auditor's tooling has a reference parser.

Standards compliance (the six — workflow-standards.md):
- PIN_PER_STEP — 3: the SUT.yaml *is* the per-run pin (model snapshot + harness
  commit + container digest + system-prompt hash). :func:`render_sut_yaml` is a
  pure function of the :class:`SUT`, so identical SUTs render identical bytes.
- ANDON_AUTHORITY — 2: :func:`render_sut_yaml` raises a structured ``SUTError``
  on a field that looks like a family alias rather than an exact version (e.g. a
  ``model_id`` with no version/snapshot component) — catching the single most
  common §9.6 reproducibility defect at render time rather than letting a
  non-reproducible SUT ship.
- NAMED_COMPENSATORS — n/a: pure string rendering, no irreversible tool call.
  (Freezing/publishing the SUT is a Phase-10 release step with its own
  compensators.)
- DECOMPOSE_BY_SECRETS — 3: the SUT surface (exact versions, frozen per release)
  is its own module, separate from the rubric bundle (tuned) and the splits
  (per-inventory) — three things that change on three different cadences.
- UNCERTAINTY_GATED_HUMANS — 1: the alias-shaped-id check nudges the human to
  supply an exact version, but the deeper "is this really the snapshot you ran?"
  question is a human responsibility this module can only prompt, not verify.
- EXTERNAL_VERIFIER — 3: SUT.yaml is consumed by an *external* assessor to
  reproduce the run; :func:`parse_sut_yaml` is a reference reader so their tooling
  and ai_crucible's agree byte-for-byte on what was pinned.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "SUTError",
    "SUT",
    "SUT_FIELDS",
    "render_sut_yaml",
    "parse_sut_yaml",
]


class SUTError(Exception):
    """Raised on an invalid SUT (a family-alias model id, an unparseable
    SUT.yaml). Structured ``[CODE] message (hint: ...)`` payload (Ship-Gate-B)."""


def _fail(code: str, message: str, hint: str) -> SUTError:
    return SUTError(f"[{code}] {message} (hint: {hint})")


# The five §9.6 fields, in render order. Used as the canonical key ordering for
# both the YAML emitter and the round-trip parser.
SUT_FIELDS: tuple[str, ...] = (
    "model_id",
    "provider_endpoint",
    "system_prompt_sha",
    "harness_commit_sha",
    "container_digest",
)


@dataclass
class SUT:
    """The System-Under-Test description (§9.6), frozen submission→publication.

    Attributes:
        model_id: the *exact* version string, e.g. ``"claude-opus-4-7-20260415"``.
            NOT a family alias like ``"claude-opus"`` (the §9.6 hard rule).
        provider_endpoint: the provider API endpoint the run hit (e.g.
            ``"https://api.anthropic.com/v1/messages"``).
        system_prompt_sha: SHA-256 of the exact system prompt used.
        harness_commit_sha: the ``ai_crucible-harness`` git commit the run executed.
        container_digest: the Docker image pinned by digest
            (``"sha256:..."``), never by tag (§9.6 container immutability).
    """

    model_id: str
    provider_endpoint: str
    system_prompt_sha: str
    harness_commit_sha: str
    container_digest: str


def _looks_like_family_alias(model_id: str) -> bool:
    """Heuristic: a model id with no version/snapshot component is an alias.

    We treat an id as "exact" only if it contains an **ASCII** digit (a date
    stamp like ``-20260415`` or a version like ``-4-7``). A bare family name
    (``claude-opus``, ``gpt``, ``mistral-large``) has none and is the §9.6
    failure mode. This is a guardrail, not a proof — an exact-looking id can
    still be wrong; the check catches the obvious, common mistake.

    The digit test is ASCII-only on purpose: ``str.isdigit()`` is true for
    non-ASCII digits and superscripts (Arabic-Indic numerals, superscript ``²``,
    etc.), so a decorative-Unicode family name like ``"claude-opus²"`` would
    otherwise sneak past the guardrail as if it carried a real version. No
    provider snapshot string uses non-ASCII digits, so requiring ``0-9`` keeps
    the §9.6 guardrail tight without rejecting any legitimate id.
    """
    return not any(ch in "0123456789" for ch in model_id)


def _yaml_scalar(value: str) -> str:
    """Render ``value`` as a double-quoted YAML scalar with the few escapes a
    quoted scalar needs (backslash and double-quote). Quoting everything keeps
    the output unambiguous and byte-stable regardless of the value (digests with
    leading zeros, endpoints with special chars, etc.)."""
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def render_sut_yaml(sut: SUT) -> str:
    """Render ``sut`` to a deterministic ``SUT.yaml`` string (§9.6).

    Fields are emitted in :data:`SUT_FIELDS` order with double-quoted scalars, so
    the output is byte-identical for identical input (PIN_PER_STEP) and parses
    back losslessly via :func:`parse_sut_yaml`. A leading comment records the
    §9.6 freeze contract for any human who opens the file.

    Raises:
        SUTError: if any field is empty, or if ``model_id`` looks like a family
            alias rather than an exact version string (the §9.6 hard rule).
    """
    for fld in SUT_FIELDS:
        value = getattr(sut, fld)
        if not isinstance(value, str) or not value.strip():
            raise _fail(
                "INPUT_SUT_EMPTY_FIELD",
                f"SUT field '{fld}' is empty",
                "every §9.6 field is mandatory: exact model_id, provider_endpoint, "
                "system_prompt_sha, harness_commit_sha, container_digest",
            )

    if _looks_like_family_alias(sut.model_id):
        raise _fail(
            "INPUT_SUT_FAMILY_ALIAS",
            f"model_id '{sut.model_id}' looks like a family alias, not an exact "
            "version string",
            "use the exact snapshot, e.g. 'claude-opus-4-7-20260415' — a family "
            "alias silently re-points across provider updates (§9.6)",
        )

    lines = [
        "# AI Crucible SUT.yaml — System Under Test (research-grounding §9.6).",
        "# Frozen from submission through publication. Exact version strings only.",
    ]
    for fld in SUT_FIELDS:
        lines.append(f"{fld}: {_yaml_scalar(getattr(sut, fld))}")
    return "\n".join(lines) + "\n"


def parse_sut_yaml(text: str) -> SUT:
    """Parse a ``SUT.yaml`` produced by :func:`render_sut_yaml` back into a
    :class:`SUT` (reference reader for the round-trip; §9.6).

    Intentionally minimal: it reads ``key: "value"`` lines, ignores ``#``
    comments and blank lines, and reconstructs the five fields. It is NOT a
    general YAML parser — it is the inverse of :func:`render_sut_yaml` so the
    round-trip is provable and an auditor has a tiny, dependency-free reference.

    Because the SUT is the frozen reproducibility record, the parser is STRICT:
    a duplicate key (which a permissive last-wins parse would silently resolve,
    quietly discarding one pinned value) and an indented / non-``key: value`` line
    (an unrepresentable nested structure this flat format cannot honor) are both
    rejected with a structured :class:`SUTError` rather than accepted ambiguously.

    Raises:
        SUTError: a required field is missing, a line is malformed or indented, or
            a key appears more than once (ambiguous pin).
    """
    found: dict[str, str] = {}
    for raw in text.splitlines():
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        # An indented line implies nested/list YAML this flat format cannot
        # represent — reject it rather than silently flattening or misreading it.
        if raw[:1].isspace():
            raise _fail(
                "INPUT_SUT_BAD_LINE",
                f"SUT.yaml has an indented line (unsupported nested structure): "
                f"{raw.rstrip()!r}",
                "render_sut_yaml emits only flat 'key: \"value\"' lines; nested "
                "or list YAML is not a valid SUT record",
            )
        line = raw.strip()
        if ":" not in line:
            raise _fail(
                "INPUT_SUT_BAD_LINE",
                f"SUT.yaml line is not 'key: value': {line!r}",
                "expected the shape render_sut_yaml emits",
            )
        key, _, value = line.partition(":")
        key = key.strip()
        value = value.strip()
        # Unwrap a double-quoted scalar.
        if len(value) >= 2 and value[0] == '"' and value[-1] == '"':
            value = value[1:-1].replace('\\"', '"').replace("\\\\", "\\")
        if key in found:
            raise _fail(
                "INPUT_SUT_DUPLICATE_KEY",
                f"SUT.yaml declares key {key!r} more than once",
                "the frozen SUT record must pin each field exactly once; a "
                "duplicate key is ambiguous (a last-wins parse would silently "
                "discard one value — §9.6 reproducibility)",
            )
        found[key] = value

    missing = [f for f in SUT_FIELDS if f not in found]
    if missing:
        raise _fail(
            "INPUT_SUT_MISSING_FIELD",
            f"SUT.yaml is missing required field(s): {missing}",
            f"a valid SUT.yaml carries all of: {list(SUT_FIELDS)}",
        )
    return SUT(**{f: found[f] for f in SUT_FIELDS})
