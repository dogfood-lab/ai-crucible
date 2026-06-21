"""Claude model adapter — the Designer/Solver bridge and a cross-family panel anchor.

Claude is the role ai_crucible is built around: the :class:`~ai_crucible.types.RoleName.DESIGNER`
stays Claude by design, and Claude is the default :class:`~ai_crucible.types.RoleName.SOLVER`
(types §10.3). This adapter backs the kernel's injected ``generate`` choke point with
the Anthropic Messages API, and — because the cross-family panel needs a Claude-family
member to *exclude* when Claude is the generator (EXTERNAL_VERIFIER, §10.2) — also
exposes the judge surface so a :class:`~ai_crucible.scoring.judge_panel.JudgePanel` can
drop it structurally when ``generator_family == "claude"``.

It mirrors :class:`~ai_crucible.models.ollama_adapter.OllamaModel`'s public surface
(``complete`` / ``generate`` / ``judge`` / ``as_judge`` / ``judge_item``) so the two
are drop-in interchangeable in the kernel and on the panel. ``family`` is fixed to
``"claude"``.

Injectable client (testability) + secrets
------------------------------------------
The Anthropic client is a **constructor parameter** (``client=None``). When ``None``
the adapter builds a real :class:`anthropic.AsyncAnthropic` lazily at call time —
reading ``ANTHROPIC_API_KEY`` from the environment *then*, never capturing it at
import — so importing this module never requires the ``anthropic`` SDK to be present
and no key is baked in. Tests inject a fake client exposing
``messages.create(*, model, system, messages, max_tokens, temperature, ...)`` and the
suite makes no network call (build-law 3).

Determinism
-----------
Like the Ollama adapter, every request pins ``temperature=0`` for replayability
(PIN_PER_STEP). The Anthropic API has no user-set seed knob, so determinism rests on
``temperature=0`` plus the pinned model id; that pin is recorded in every
:class:`~ai_crucible.characterize.types.JudgmentRecord`'s metadata so a profile run is
reproducible to the extent the API allows.

Standards compliance (the six — workflow-standards.md)
------------------------------------------------------
* **PIN_PER_STEP — 3:** model id + ``temperature=0`` + ``max_tokens`` pinned per
  request and echoed into :attr:`JudgmentRecord.metadata`. Proven in
  ``tests/test_models.py``.
* **ANDON_AUTHORITY — n/a / EXTERNAL_VERIFIER — 3 / DECOMPOSE_BY_SECRETS — 2 /
  UNCERTAINTY_GATED_HUMANS — n/a:** identical rationale to the Ollama adapter (a leaf
  model client; the panel reads ``.family`` for same-family exclusion; gold labels
  live with the grader, not the adapter). The one addition: this adapter reads a
  **secret** (``ANTHROPIC_API_KEY``) — it does so at call time from the env and never
  logs or stores it. Like the Ollama adapter, :meth:`complete` verifies the served
  model (the Messages API ``model`` echo) matches the requested one and raises
  :class:`~ai_crucible.models.ollama_adapter.ModelMismatchError` on a silent route to a
  different model, so the ``"claude"`` family tag is never attributed to a judgment a
  different model produced (§11.6 / models-cli-002).
"""

from __future__ import annotations

import time
from typing import Any, Protocol

from ai_crucible.characterize.types import JudgmentRecord
from ai_crucible.models.ollama_adapter import ModelMismatchError, _norm_model
from ai_crucible.scoring.judge_panel import JudgeFn
from ai_crucible.types import AttemptState, Score

__all__ = ["ClaudeModel", "ClaudeClient", "ModelMismatchError"]

#: The fixed model family for every ClaudeModel — the panel excludes a judge of this
#: family when Claude is the generator (EXTERNAL_VERIFIER, §10.2).
CLAUDE_FAMILY = "claude"

#: Default ceiling on generated tokens per request (pinned for replayability).
_DEFAULT_MAX_TOKENS = 4096


class ClaudeClient(Protocol):
    """Structural type for the Anthropic client the adapter drives.

    Only the ``messages.create`` async method is used. The real
    :class:`anthropic.AsyncAnthropic` satisfies this; tests pass a fake exposing the
    same ``messages.create(...)`` coroutine. Kept as a Protocol (not a concrete
    import) so this module imports without the ``anthropic`` SDK installed.
    """

    class _Messages(Protocol):
        async def create(self, **kwargs: Any) -> Any: ...

    @property
    def messages(self) -> _Messages: ...


def _split_system(messages: list[dict[str, Any]]) -> tuple[str | None, list[dict[str, Any]]]:
    """Split out leading ``system`` turns — the Anthropic API takes ``system`` top-level.

    Unlike the Ollama chat shape (system is just another message), the Messages API
    wants ``system`` as a separate string parameter and ``messages`` containing only
    ``user``/``assistant`` turns. Any ``system`` messages are concatenated (newline-
    joined) into the returned system string and stripped from the turn list; the rest
    pass through unchanged.
    """
    system_parts = [str(m.get("content", "")) for m in messages if m.get("role") == "system"]
    turns = [m for m in messages if m.get("role") != "system"]
    system = "\n\n".join(p for p in system_parts if p) or None
    return system, turns


def _extract_text(response: Any) -> str:
    """Pull assistant text out of an Anthropic Messages response.

    The Messages API returns ``content`` as a list of typed blocks; the text is the
    concatenation of ``.text`` over the ``text`` blocks. Tolerates three shapes so a
    fake client may return any of them: an object with ``.content`` blocks (real SDK),
    a dict with ``"content"`` block list, or a plain string. Never raises on a missing
    field — an empty completion is a valid answer the caller scores.
    """
    if isinstance(response, str):
        return response

    content = response.get("content") if isinstance(response, dict) else getattr(
        response, "content", None
    )
    if content is None:
        return ""
    if isinstance(content, str):
        return content

    parts: list[str] = []
    for block in content:
        # Block may be an object (.type/.text) or a dict ({"type":..,"text":..}).
        if isinstance(block, dict):
            if block.get("type", "text") == "text" and "text" in block:
                parts.append(str(block["text"]))
        else:
            if getattr(block, "type", "text") == "text" and hasattr(block, "text"):
                parts.append(str(block.text))
    return "".join(parts)


class ClaudeModel:
    """A Claude model via the Anthropic Messages API — Designer/Solver + panel anchor.

    Public surface mirrors :class:`~ai_crucible.models.ollama_adapter.OllamaModel` so the
    two are interchangeable in the kernel and on the panel. ``family`` is fixed to
    ``"claude"``.

    Args:
        model_id: the Anthropic model id, e.g. ``"claude-opus-4-8"``.
        client: an injected client of the :class:`ClaudeClient` shape. ``None``
            (default) builds a real :class:`anthropic.AsyncAnthropic` lazily at call
            time, reading ``ANTHROPIC_API_KEY`` from the env then. Injected so tests
            pass a fake and the suite makes no network call.
        max_tokens: the generation ceiling pinned per request (PIN_PER_STEP).
        quant: kept for surface-parity with the Ollama adapter and recorded in the
            JudgmentRecord; hosted Claude is not user-quantized, so it is ``None`` in
            practice.
    """

    family = CLAUDE_FAMILY

    def __init__(
        self,
        model_id: str,
        *,
        client: ClaudeClient | None = None,
        max_tokens: int = _DEFAULT_MAX_TOKENS,
        quant: str | None = None,
    ) -> None:
        self.model_id = model_id
        self._client = client
        self.max_tokens = max_tokens
        self.quant = quant

    # -- provenance (PIN_PER_STEP) ------------------------------------------ #

    def pin_metadata(self) -> dict[str, Any]:
        """The PIN_PER_STEP provenance block stamped on characterization records.

        Records the model id, family, and the pinned request knobs (``temperature=0``
        + ``max_tokens``). The Anthropic API exposes no seed, so determinism rests on
        ``temperature=0`` + the model pin — captured here so the profile run is as
        replayable as the API allows (§11.6).
        """
        return {
            "model_id": self.model_id,
            "family": self.family,
            "quant": self.quant,
            "options": {"temperature": 0, "max_tokens": self.max_tokens},
        }

    # -- client resolution (lazy, secret read at call time) ----------------- #

    def _resolve_client(self) -> ClaudeClient:
        """Return the injected client, or lazily build a real one at call time.

        Builds :class:`anthropic.AsyncAnthropic` only if no client was injected,
        importing the SDK lazily (so this module imports without it) and letting the
        SDK read ``ANTHROPIC_API_KEY`` from the environment *now*. The key is never
        captured at import nor stored on the instance.
        """
        if self._client is not None:
            return self._client
        import anthropic  # type: ignore[import-not-found]

        # AsyncAnthropic reads ANTHROPIC_API_KEY from the env at construction.
        self._client = anthropic.AsyncAnthropic()
        return self._client

    # -- core completion ---------------------------------------------------- #

    async def complete(self, messages: list[dict[str, Any]]) -> str:
        """Call the Messages API with pinned determinism and return assistant text.

        Splits leading ``system`` turns to the top-level ``system`` parameter
        (:func:`_split_system`), sends ``temperature=0`` + the pinned ``max_tokens``,
        and concatenates the returned text blocks (:func:`_extract_text`).
        """
        client = self._resolve_client()
        system, turns = _split_system(messages)
        kwargs: dict[str, Any] = {
            "model": self.model_id,
            "messages": turns,
            "max_tokens": self.max_tokens,
            "temperature": 0,
        }
        if system is not None:
            kwargs["system"] = system
        response = await client.messages.create(**kwargs)
        self._assert_served_matches(response)
        return _extract_text(response)

    def _assert_served_matches(self, response: Any) -> None:
        """Raise :class:`ModelMismatchError` if the served model != the requested one.

        The Messages API echoes the model that actually answered in a top-level ``model``
        field (``response.model`` on the SDK object, ``response["model"]`` on a dict). If
        the API silently routes to a different model (e.g. an alias/snapshot resolving
        elsewhere), this surfaces it as an exception rather than letting a JudgmentRecord
        be stamped with the requested ``model_id`` while a different model produced the
        judgment (§11.6 / models-cli-002 — the same provenance guard the Ollama adapter
        enforces). A *missing* ``model`` field is tolerated (older/fake responses that do
        not echo it), so the guard fails closed only on a *positive* mismatch.
        """
        served = (
            response.get("model")
            if isinstance(response, dict)
            else getattr(response, "model", None)
        )
        if isinstance(served, str) and served and _norm_model(served) != _norm_model(
            self.model_id
        ):
            raise ModelMismatchError(self.model_id, served)

    # -- kernel generate plug (§10.2) --------------------------------------- #

    async def generate(self, state: AttemptState) -> str:
        """Kernel ``generate`` choke point: complete on ``state.messages`` → text.

        Matches :data:`ai_crucible.roles.GenerateFn` so a bound ``model.generate`` drops
        into ``run_attempt(generate=...)``. The kernel records the call + owns
        budget/andon; the adapter only produces the completion (§10.2).
        """
        return await self.complete(state.messages)

    # -- panel judge plug (§10.2, EXTERNAL_VERIFIER) ------------------------ #

    async def judge(self, attempt: AttemptState) -> Score:
        """Score an attempt as a panel judge → :class:`~ai_crucible.types.Score`.

        Thin parse here (the §11.3 rubric lands with the calibration set). The verdict
        text and the judging ``model_id``/``family`` are recorded in ``Score.metadata``;
        the score may carry a ``novelty_validated`` vote the panel aggregates (§8.7).
        Use :meth:`as_judge` to get the panel-ready callable carrying ``.family``.
        """
        verdict = await self.complete(self._judge_messages(attempt))
        return Score(
            value=verdict,
            explanation=verdict,
            metadata={"judge_model": self.model_id, "judge_family": self.family},
        )

    def as_judge(self) -> JudgeFn:
        """Return a panel-ready judge callable tagged with ``family="claude"``.

        :class:`~ai_crucible.scoring.judge_panel.JudgePanel` reads ``.family`` off the
        callable to enforce same-family exclusion (§10.2). A bound method cannot carry
        the attribute, so this wraps :meth:`judge` and stamps ``.family`` — the
        supported way to seat (and thus to *exclude*) this Claude model on a panel.
        """

        async def _judge(attempt: AttemptState) -> Score:
            return await self.judge(attempt)

        _judge.family = self.family  # type: ignore[attr-defined]
        return _judge

    def _judge_messages(self, attempt: AttemptState) -> list[dict[str, Any]]:
        """Build the messages for a judging call (placeholder rubric, §11.3).

        Reuses the attempt's scored context + Solver output. NEVER reads
        ``attempt.chrome`` — Tier-3 is sealed out of every model context (§10.1(e)).
        """
        return [
            *attempt.messages,
            {
                "role": "user",
                "content": (
                    "You are an impartial judge. Given the candidate output below, "
                    "respond with your verdict.\n\n"
                    f"OUTPUT:\n{attempt.output or ''}"
                ),
            },
        ]

    # -- characterization probe (§11.1) ------------------------------------- #

    async def judge_item(
        self,
        prompt: str,
        *,
        run_index: int = 0,
        position: int | None = None,
    ) -> JudgmentRecord:
        """Run one calibration item → a :class:`JudgmentRecord` (the §11.1 metric unit).

        Identical contract to :meth:`OllamaModel.judge_item`: records ``predicted`` +
        measured ``latency_s``, stamps ``model_id``/``family``/``quant`` and the
        PIN_PER_STEP provenance into ``metadata``, and leaves ``gold``/``correct``
        ``None`` for the scorer that holds the gold labels (DECOMPOSE_BY_SECRETS,
        §11.6). ``run_index`` drives test-retest (§11.1 #4); ``position`` drives
        position-swap bias (#6).
        """
        messages = [{"role": "user", "content": prompt}]
        start = time.monotonic()
        predicted = await self.complete(messages)
        latency_s = time.monotonic() - start
        return JudgmentRecord(
            item_id=_item_id(prompt),
            model_id=self.model_id,
            predicted=predicted,
            gold=None,
            quant=self.quant,
            latency_s=latency_s,
            run_index=run_index,
            position=position,
            family=self.family,
            metadata=self.pin_metadata(),
        )


def _item_id(prompt: str) -> str:
    """A stable short id for a calibration item, derived from its prompt text.

    Mirrors the Ollama adapter so a probe is self-identifying without a registry at
    this layer (the authored ids arrive with the §11.3 calibration set).
    """
    import hashlib

    return "item-" + hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:12]
