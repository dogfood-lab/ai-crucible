"""Attestation edge (``attestation`` module, research-grounding §9.5, §10.2).

This module is the *polyglot edge* of the kernel. In production the durable
trajectory log and the cryptographic signing are provided by two non-Python
dependencies, isolated here so the language boundary is one module rather than a
cross-cutting concern (swarm-17 recommended skeleton):

1. **Persistence** — the sibling org's npm package
   `@attestia/event-store`
   (https://github.com/mcp-tool-shop-org/Attestia/tree/main/packages/event-store):
   append-only persistence with SHA-256 hash chaining over **RFC 8785** canonical
   JSON, a ``JsonlEventStore`` for durable file-based logs, and a
   ``verifyHashChain()`` integrity check (§9.5).

2. **Signing** — the **cosign** Go binary (`cosign sign-blob`), Sigstore +
   Rekor, OIDC-keyless (§9.5).

:class:`JsonlEventStore` here is a **pure-Python mirror** of the Attestia hash
chain so Crucible's Phase-1 build has a working, testable durable log without
the Node dependency on the critical path. It is byte-compatible in *intent*
(same canonical-JSON + ``prev_hash``/``hash`` chain) and is the surface the
kernel writes per-batch trajectory logs to. The full eval-domain extension and
the in-toto / RFC 3161 / Sigstore integration is a future dogfood swarm on
Attestia (see ``docs/attestia-integration-roadmap.md``).

:func:`cosign_sign_blob` is a **documented no-op stub** — it returns ``None``
unless explicitly enabled, and even when enabled it shells out to the real
``cosign`` binary. It NEVER fakes a signature: faking cryptographic provenance
would be worse than having none (§9.5 is explicit that the value is a real
transparency-log inclusion proof, not a plausible-looking string).
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

__all__ = [
    "GENESIS_HASH",
    "HashChainError",
    "JsonlEventStore",
    "canonical_json",
    "chain_hash",
    "cosign_sign_blob",
]

# The chain anchor: prev_hash of the first entry. 64 zero hex chars = "no
# predecessor", mirroring a genesis block. Stable so an independent verifier
# can reconstruct the chain from byte zero.
GENESIS_HASH = "0" * 64


class HashChainError(Exception):
    """Raised on a structural defect in the stored chain (malformed JSON, a line
    missing its ``hash``/``prev_hash`` envelope fields).

    NOTE: a *tamper* (a content edit that breaks the chain) is reported by
    :meth:`JsonlEventStore.verify_hash_chain` returning ``False`` — it is an
    expected verification outcome, not an exception. This exception is reserved
    for a log that is not a well-formed hash-chained JSONL file at all.

    Carries a structured (code/message/hint) message per the repo's Ship-Gate-B
    error shape.
    """


def _fail(code: str, message: str, hint: str) -> HashChainError:
    return HashChainError(f"[{code}] {message} (hint: {hint})")


def canonical_json(obj: Any) -> str:
    """Return RFC-8785-style canonical JSON for ``obj``.

    Stable key ordering + minimal separators so the byte string is reproducible
    across processes and machines — the precondition for a meaningful hash chain
    (§9.5). This is the same canonicalisation `@attestia/event-store` applies
    before hashing.

    (Full RFC 8785 also pins number formatting; Python's ``json`` emits
    canonical integer forms and we keep payload numbers integer/float-stable, so
    ``sort_keys`` + tight separators is sufficient for the JSON shapes Crucible
    writes. Documented here so a future maintainer knows the boundary.)

    Non-finite floats (``NaN``, ``Infinity``, ``-Infinity``) are **not legal
    JSON** and have no RFC-8785 representation. Python's ``json`` would otherwise
    emit the bare tokens ``NaN``/``Infinity``, which a conformant verifier (the
    ``@attestia/event-store`` this mirrors) rejects while Crucible's own lenient
    ``json.loads`` accepts — a chain that silently verifies here yet no external
    auditor can verify. We pass ``allow_nan=False`` so such a value fails LOUDLY
    as a structured :class:`HashChainError` instead of writing an unverifiable
    line (§9.5: the stored value is a real integrity proof, not a plausible
    string).
    """
    try:
        return json.dumps(
            obj,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except ValueError as exc:
        # json raises ValueError ("Out of range float values are not JSON
        # compliant") for NaN/Infinity when allow_nan=False.
        raise _fail(
            "INPUT_NON_FINITE_NUMBER",
            f"cannot canonicalise a non-finite number (NaN/Infinity): {exc}",
            "RFC 8785 / JSON has no NaN/Infinity literal; a conformant verifier "
            "rejects such a line — sanitise the value before logging it",
        ) from exc


def chain_hash(prev_hash: str, entry: dict[str, Any]) -> str:
    """Compute the chain hash for ``entry`` given its predecessor's hash.

    ``hash = sha256(prev_hash + canonical_json(entry))`` — the entry hashed is
    the *payload* (without its own ``hash``/``prev_hash`` envelope), bound to the
    predecessor so any reorder, insertion, deletion, or content edit anywhere in
    the chain changes every subsequent hash.
    """
    material = prev_hash + canonical_json(entry)
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


# Envelope field names. The stored line is the entry payload plus these two.
_F_PREV = "prev_hash"
_F_HASH = "hash"
_F_PAYLOAD = "payload"


class JsonlEventStore:
    """Append-only, SHA-256 hash-chained JSONL event store.

    Each stored line is a JSON object::

        {"prev_hash": "<64 hex>", "payload": <the event dict>, "hash": "<64 hex>"}

    where ``hash == sha256(prev_hash + canonical_json(payload))`` and
    ``prev_hash`` is the previous line's ``hash`` (or :data:`GENESIS_HASH` for
    the first line). This is the durable backing the kernel streams per-batch
    trajectory logs to (§9.5), and a faithful mirror of the Attestia Node
    package's ``JsonlEventStore`` semantics.

    The store is *append-only by contract*: :meth:`append` only ever writes to
    the end of the file. Rewriting earlier lines is exactly the tamper that
    :meth:`verify_hash_chain` detects.
    """

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        # Ensure the parent dir exists so the first append doesn't explode; the
        # file itself is created lazily on first append.
        self.path.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------- internals
    def _read_lines(self) -> list[str]:
        if not self.path.exists():
            return []
        text = self.path.read_text(encoding="utf-8")
        return [ln for ln in text.splitlines() if ln.strip()]

    def _last_hash(self) -> str:
        """Return the ``hash`` of the final stored entry, or GENESIS if empty."""
        lines = self._read_lines()
        if not lines:
            return GENESIS_HASH
        try:
            last = json.loads(lines[-1])
        except json.JSONDecodeError as exc:
            raise _fail(
                "STATE_CHAIN_CORRUPT_TAIL",
                f"final line of {self.path.name} is not valid JSON: {exc}",
                "the log is not a well-formed hash-chained JSONL file",
            ) from exc
        if not isinstance(last, dict) or _F_HASH not in last:
            raise _fail(
                "STATE_CHAIN_MISSING_HASH",
                f"final line of {self.path.name} has no '{_F_HASH}' field",
                "every line must carry the chain envelope (prev_hash/payload/hash)",
            )
        return str(last[_F_HASH])

    # ---------------------------------------------------------------- public
    def append(self, event: dict[str, Any]) -> str:
        """Append ``event`` and return the new entry's chain hash.

        The event is wrapped in the chain envelope (``prev_hash`` bound to the
        current tail) and one JSONL line is appended. The returned hash becomes
        the ``prev_hash`` of the next append — callers may keep it for an
        external receipt.
        """
        prev = self._last_hash()
        digest = chain_hash(prev, event)
        line = {
            _F_PREV: prev,
            _F_PAYLOAD: event,
            _F_HASH: digest,
        }
        # Canonical JSON for the stored line too, so the file is reproducible.
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(canonical_json(line) + "\n")
        return digest

    def read_events(self) -> list[dict[str, Any]]:
        """Return the stored event payloads in order (envelope stripped).

        Does not verify the chain — call :meth:`verify_hash_chain` for that.
        """
        out: list[dict[str, Any]] = []
        for raw in self._read_lines():
            obj = json.loads(raw)
            out.append(obj.get(_F_PAYLOAD, {}))
        return out

    def verify_hash_chain(self) -> bool:
        """Recompute the whole chain and return whether it is intact.

        Walks every stored line, checking that each line's ``prev_hash`` equals
        the running hash and that its ``hash`` equals
        ``sha256(prev_hash + canonical_json(payload))``. Returns:

        - ``True``  — every link verifies (or the log is empty).
        - ``False`` — any link is broken: a payload was edited, a line was
          inserted/removed/reordered, or a stored hash was altered. This is the
          tamper-detection contract; a broken chain is a *return value*, not an
          exception.

        Raises :class:`HashChainError` only if the file is not well-formed
        hash-chained JSONL (a line is not JSON, or lacks the envelope fields) —
        i.e. it was never a valid chain, as opposed to a valid chain that was
        subsequently tampered.
        """
        running = GENESIS_HASH
        for lineno, raw in enumerate(self._read_lines(), start=1):
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise _fail(
                    "STATE_CHAIN_BAD_JSON",
                    f"line {lineno} of {self.path.name} is not valid JSON: {exc}",
                    "the log is not a well-formed hash-chained JSONL file",
                ) from exc
            if not isinstance(obj, dict) or obj.keys() != {_F_PREV, _F_PAYLOAD, _F_HASH}:
                raise _fail(
                    "STATE_CHAIN_BAD_ENVELOPE",
                    f"line {lineno} of {self.path.name} has a malformed chain envelope",
                    f"each line must have EXACTLY '{_F_PREV}', '{_F_PAYLOAD}', and "
                    f"'{_F_HASH}' — no missing and no extra top-level keys (extra "
                    "keys are unhashed and would otherwise survive verification)",
                )
            # Link 1: prev_hash must match the running hash.
            if obj[_F_PREV] != running:
                return False
            # Link 2: stored hash must match the recomputed hash of the payload.
            expected = chain_hash(obj[_F_PREV], obj[_F_PAYLOAD])
            if obj[_F_HASH] != expected:
                return False
            running = obj[_F_HASH]
        return True

    def __len__(self) -> int:
        return len(self._read_lines())

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"JsonlEventStore(path={self.path!s}, entries={len(self)})"


def cosign_sign_blob(path: Path, *, enabled: bool = False) -> str | None:
    """Sign ``path`` with Sigstore ``cosign sign-blob`` (§9.5) — stub by default.

    This is the second half of the polyglot edge. When ``enabled`` is ``False``
    (the default for the Phase-1 build and every test), it is a **no-op that
    returns ``None``** — the production cryptographic-provenance pipeline
    (RFC 3161 + in-toto + Sigstore/Rekor) is a future dogfood swarm on Attestia.

    When ``enabled`` is ``True`` it shells out to the real ``cosign`` binary
    (OIDC-keyless) and returns the signature bundle as a string. It does **not**
    fabricate a signature under any circumstance: if ``cosign`` is not installed
    or fails, it raises rather than returning a fake. Honest provenance or no
    provenance — never plausible-looking provenance (§9.5).

    Returns:
        The cosign signature bundle (str) when enabled and successful; ``None``
        when disabled.

    Raises:
        HashChainError: when enabled but ``cosign`` is unavailable or the
            target path does not exist (structured code/message/hint).
    """
    if not enabled:
        return None

    target = Path(path)
    if not target.is_file():
        raise _fail(
            "INPUT_SIGN_TARGET_MISSING",
            f"cannot sign a path that is not a file: {target}",
            "pass the path to the sealed blob you want cosign to sign",
        )

    cosign = shutil.which("cosign")
    if cosign is None:
        raise _fail(
            "DEP_COSIGN_MISSING",
            "cosign binary not found on PATH",
            "install Sigstore cosign, or call with enabled=False to skip signing",
        )

    # Keyless OIDC signing; emits the signature to stdout as a bundle.
    completed = subprocess.run(  # noqa: S603 - args are fixed, not user-shell
        [cosign, "sign-blob", "--yes", str(target)],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise _fail(
            "RUNTIME_COSIGN_FAILED",
            f"cosign sign-blob exited {completed.returncode}: {completed.stderr.strip()}",
            "check cosign OIDC configuration; do not ship an unsigned blob as signed",
        )
    return completed.stdout.strip()
