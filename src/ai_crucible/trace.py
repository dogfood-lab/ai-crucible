"""Trace writer (``trace_writer`` module, research-grounding §10.2, swarm-17 finding 5).

Every model/tool/score call the kernel makes is appended here as a structured
:class:`ai_crucible.types.TraceEvent`. Per Vivaria (swarm-17 finding 8) the budget
and trace accounting is **kernel-side, never Solver-self-reported** — the Solver
cannot write its own trace, so transcripts are audit-ready by construction.

:meth:`TraceWriter.to_eval_log` renders the per-attempt record in the **Inspect
AI EvalLog shape** (finding 5): top-level ``eval`` / ``plan`` / ``results`` /
``stats`` / ``samples[]``, each sample carrying
``id, input, target, output, scores, events, metadata, error``.

Large blobs are *not* inlined into events. They are de-duplicated into an
``attachments`` map keyed by their SHA-256 digest (Inspect EvalLog stores large
event content as attachments referenced by digest, finding 5); an event holds
only the short ``ref`` string. Two identical blobs collapse to one attachment.

This module is intentionally Inspect-shape-*compatible* rather than
Inspect-typed: it emits a plain ``dict`` matching the documented EvalLog v2
envelope, so it neither pins nor breaks against Inspect's evolving pydantic API
surface (swarm-17 verification note). The kernel can hand the dict straight to
an Inspect log writer or persist it via :mod:`ai_crucible.attestation`.
"""

from __future__ import annotations

import hashlib
from typing import Any

from ai_crucible.types import AttemptState, RoleName, Score, TraceEvent

__all__ = ["EVAL_LOG_SCHEMA_VERSION", "TraceWriter", "attachment_ref", "sha256_hex"]

# Self-describing schema marker stamped onto every emitted eval-log (kernel-runtime-005).
# The envelope is hand-shaped to match "Inspect EvalLog v2" but is a plain dict (it
# neither pins nor breaks against Inspect's evolving pydantic surface). inspect-ai is
# pinned >=0.3 with an open upper bound, so the shape this mirrors can move. Without a
# version key a future envelope change (a renamed top-level field, a consumer expecting a
# marker) would be SILENT — a reader would misparse or drop fields with no error, and
# persisted attestation JSONL records could not be told apart by schema version after the
# fact. The marker is ai_crucible-namespaced so it is unambiguous which schema authored a
# persisted record; bump it whenever the top-level envelope shape changes.
EVAL_LOG_SCHEMA_VERSION = "crucible/evallog/2"

# Blobs at or above this size (chars) are spilled to an attachment rather than
# left inline in the event payload. Below it, inlining is cheaper than a digest
# round-trip. The threshold mirrors Inspect's "large content -> attachment"
# behaviour without copying its exact constant (which is an internal detail).
ATTACHMENT_THRESHOLD = 1024

# Stable scheme for attachment references stored inside events, so a reader can
# tell "this is an attachment pointer" from "this is literal content".
_ATTACHMENT_SCHEME = "attachment://sha256/"


def sha256_hex(content: str) -> str:
    """Return the hex SHA-256 of ``content`` (UTF-8). The attachment key."""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def attachment_ref(digest: str) -> str:
    """Return the canonical reference string for an attachment ``digest``."""
    return f"{_ATTACHMENT_SCHEME}{digest}"


def _role_value(role: RoleName | None) -> str | None:
    """Normalise a role enum to its string value (or ``None``)."""
    return role.value if role is not None else None


def _score_to_dict(score: Score) -> dict[str, Any]:
    """Render a :class:`Score` as an Inspect-shaped score dict."""
    return {
        "value": score.value,
        "answer": score.answer,
        "explanation": score.explanation,
        "metadata": dict(score.metadata),
    }


class TraceWriter:
    """Append-only collector of per-attempt :class:`TraceEvent` records.

    The writer owns sequence numbering (callers never set ``seq``) and the
    attachment store. It is deliberately small and pure: it holds events +
    attachments in memory and renders them on demand. Durable persistence is the
    job of :class:`ai_crucible.attestation.JsonlEventStore`; this class only shapes
    the record.
    """

    def __init__(self) -> None:
        self._events: list[TraceEvent] = []
        self._attachments: dict[str, str] = {}
        self._next_seq = 0

    # ----------------------------------------------------------------- events
    @property
    def events(self) -> list[TraceEvent]:
        """The appended events, in order. Returns the live list (read-only by
        convention — callers append via :meth:`append`)."""
        return self._events

    @property
    def attachments(self) -> dict[str, str]:
        """The digest -> content attachment map accumulated so far."""
        return self._attachments

    def append(self, event: TraceEvent) -> None:
        """Append ``event`` to the transcript, auto-assigning its ``seq``.

        The caller's ``seq`` is ignored and overwritten with the next monotonic
        sequence number, so ordering is owned by the writer (kernel-side), not by
        whatever constructed the event.
        """
        event.seq = self._next_seq
        self._next_seq += 1
        self._events.append(event)

    def attach(self, content: str) -> tuple[str, str]:
        """Register ``content`` as an attachment; return ``(sha256_hex, ref)``.

        Idempotent: registering identical content twice yields the same digest
        and stores it once (content-addressed de-duplication). ``ref`` is the
        short pointer string callers embed in an event payload in place of the
        blob.
        """
        digest = sha256_hex(content)
        self._attachments.setdefault(digest, content)
        return digest, attachment_ref(digest)

    def append_blob_event(
        self,
        kind: str,
        content: str,
        *,
        role: RoleName | None = None,
        payload_key: str = "content",
        extra_payload: dict[str, Any] | None = None,
    ) -> TraceEvent:
        """Convenience: append an event whose large ``content`` is spilled to an
        attachment when it exceeds :data:`ATTACHMENT_THRESHOLD`.

        Small content is inlined under ``payload_key``; large content is replaced
        by ``{payload_key + "_ref": ref}`` and the event records the digest in
        its ``attachments`` map (name ``payload_key`` -> digest). This is the
        path the kernel uses for model completions / tool outputs so the trace
        stream stays compact and replayable (finding 5).
        """
        payload: dict[str, Any] = dict(extra_payload or {})
        attachments: dict[str, str] = {}
        if len(content) >= ATTACHMENT_THRESHOLD:
            digest, ref = self.attach(content)
            payload[f"{payload_key}_ref"] = ref
            attachments[payload_key] = digest
        else:
            payload[payload_key] = content
        event = TraceEvent(kind=kind, role=role, payload=payload, attachments=attachments)
        self.append(event)
        return event

    # ------------------------------------------------------------- rendering
    def _event_to_dict(self, event: TraceEvent) -> dict[str, Any]:
        """Render a single :class:`TraceEvent` to a JSON-able dict."""
        return {
            "seq": event.seq,
            "kind": event.kind,
            "role": _role_value(event.role),
            "payload": dict(event.payload),
            "attachments": dict(event.attachments),
        }

    def to_eval_log(self, attempt: AttemptState) -> dict[str, Any]:
        """Render ``attempt`` + the collected trace as an Inspect-EvalLog-shaped
        ``dict`` (finding 5).

        The returned dict has the documented top-level keys ``eval`` / ``plan``
        / ``results`` / ``stats`` / ``samples`` plus an ``attachments`` map and a
        self-describing ``schema_version`` marker (:data:`EVAL_LOG_SCHEMA_VERSION`,
        kernel-runtime-005) so a downstream consumer can branch on envelope drift and
        a persisted record stays version-legible. The single sample carries
        ``id, input, target, output, scores, events, metadata, error`` — AI Crucible
        runs one attempt per record (pass^k is k *sibling* records, not k samples in
        one log, per finding 6, handled by :mod:`ai_crucible.observability`).

        ``input`` is the scored message context (Tier-1/Tier-2 only); Tier-3
        ``chrome`` is **never** serialized here — it is not part of the scored
        record (§10.1(e)).
        """
        budget = attempt.budget
        scores = {name: _score_to_dict(s) for name, s in attempt.scores.items()}
        events = [self._event_to_dict(e) for e in self._events]

        sample: dict[str, Any] = {
            "id": attempt.attempt_id,
            "input": list(attempt.messages),  # scored context only (no chrome)
            "target": None,  # the oracle/target stays grading-side (§10.4)
            "output": attempt.output,
            "scores": scores,
            "events": events,
            "metadata": dict(attempt.metadata),
            "error": attempt.error,
        }

        eval_spec: dict[str, Any] = {
            "task": "ai_crucible",
            "model": attempt.model,
            "puzzle_id": attempt.puzzle_id,
            "framing_arm": attempt.framing_arm.value,
        }

        plan: dict[str, Any] = {
            "steps": [self._event_to_dict(e) for e in self._events if e.kind == "model"],
        }

        results: dict[str, Any] = {
            "scores": scores,
            "terminated_by": attempt.terminated_by.value if attempt.terminated_by else None,
        }

        stats: dict[str, Any] = {
            "usage": dict(attempt.usage),
            "wall_time": attempt.wall_time,
            "tool_calls_used": budget.tool_calls_used if budget else None,
            "tool_call_budget": budget.tool_call_budget if budget else None,
        }

        return {
            # Self-describing schema marker so a consumer can detect/branch on drift
            # and persisted records are version-legible after the fact (kernel-runtime-005).
            "schema_version": EVAL_LOG_SCHEMA_VERSION,
            "eval": eval_spec,
            "plan": plan,
            "results": results,
            "stats": stats,
            "samples": [sample],
            "attachments": dict(self._attachments),
        }

    def __len__(self) -> int:
        return len(self._events)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"TraceWriter(events={len(self._events)}, "
            f"attachments={len(self._attachments)})"
        )
