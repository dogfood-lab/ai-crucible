"""OpenRouter model adapter — a cross-family verifier/Solver seat via ONE OpenAI-compatible key.

OpenRouter fronts the whole catalog of model families — the open ones (deepseek / qwen / cohere /
nvidia / meta-llama / …) AND the closed frontier (real GPT, Gemini) — behind a single
OpenAI-compatible endpoint. That makes it a cheap, broad **cross-family supply** for the panel
(§8.6 / §11.2) and the banked cross-family cloud ω-anchor jury: one key, many genuinely-different
families, including ones the local Ollama panel and the open-only Ollama-Cloud roster cannot give.

Why this is simpler than prism's gateway seat
----------------------------------------------
crucible's model spec already declares the family explicitly (``<id>@family`` — cli ``_build_model``
/ ``characterize._parse_models``), so an OpenRouter-served model carries its **true** family by
construction. No lineage-derivation guard is structurally required — the operator's ``@family`` is
the declared cross-family axis, exactly as it already is for :class:`OllamaModel`. The one
OpenRouter-specific affordance: because an OpenRouter id *does* expose the vendor (``deepseek/…``),
:func:`_warn_on_family_mismatch` emits a NON-FATAL warning when the vendor prefix clearly
contradicts the declared ``@family`` (e.g. ``deepseek/deepseek-chat@qwen``) — a cheap typo-catcher
for a mislabel that would silently corrupt the cross-family differential / ω, without a brittle
vendor→family table or a refusal (unknown vendors/families pass through untouched).

Selector + identity
-------------------
A model id prefixed ``openrouter:`` routes here (``openrouter:deepseek/deepseek-chat@deepseek``).
The prefix is kept as part of :attr:`model_id` (the stable identity recorded in every JudgmentRecord
and seat, so a seated OpenRouter judge round-trips through ``panel.json`` → ``run --panel``); only
the HTTP call strips it (:attr:`_wire_id`) to send the bare ``deepseek/deepseek-chat`` OpenRouter
expects.

Public surface / determinism / provenance
------------------------------------------
Mirrors :class:`~ai_crucible.models.ollama_adapter.OllamaModel` (``complete`` / ``generate`` /
``judge`` / ``as_judge`` / ``judge_item`` / ``pin_metadata`` / ``.family``) so the two are
interchangeable in the kernel and on the panel. Every request pins ``temperature=0`` (PIN_PER_STEP);
``OPENROUTER_API_KEY`` is read from the env at call time, never captured at import or logged. The
served model OpenRouter echoes (``response.model``) is checked against the requested id modulo an
OpenRouter dated snapshot / variant suffix (:func:`_norm_openrouter`), raising the shared
:class:`~ai_crucible.models.ollama_adapter.ModelMismatchError` on a genuine cross-vendor fallback so
a mis-served judge is never attributed to (and excluded as) the wrong family (§11.6 /
models-cli-002). Per-token logprobs (OpenAI ``logprobs``/``top_logprobs``) drive the verdict-token
confidence (§12) via the SHARED
:func:`~ai_crucible.models.ollama_adapter._first_token_logprob` (which already handles the OpenAI
``choices[0].logprobs.content[0].logprob`` shape).
"""

from __future__ import annotations

import asyncio
import os
import re
import time
import warnings
from collections.abc import Awaitable, Callable
from typing import Any

from ai_crucible.characterize.types import JudgmentRecord
from ai_crucible.models.ollama_adapter import (
    ModelMismatchError,
    _call_with_retry,
    _first_token_logprob,
    _item_id,
    _logprob_to_probability,
    _normalize_harmony,
)
from ai_crucible.scoring.judge_panel import JudgeFn
from ai_crucible.types import AttemptState, Score

__all__ = [
    "OpenRouterModel",
    "OpenRouterClient",
    "OpenRouterUnreachableError",
    "OpenRouterBadResponseError",
    "OPENROUTER_PREFIX",
    "is_openrouter_spec",
]

#: The model-id sentinel that routes a spec to this adapter (kept as part of the id).
OPENROUTER_PREFIX = "openrouter:"

#: Default OpenRouter base; already includes ``/v1`` so requests POST to ``/chat/completions``.
#: Read (with ``OPENROUTER_BASE_URL``) at call time, never baked in.
DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"

_NAME = "openrouter"

#: Generation ceiling pinned per request (PIN_PER_STEP). Mirrors the Claude adapter's default.
_DEFAULT_MAX_TOKENS = 4096

#: Bounded retry policy for a single call (§8.6, PIN_PER_STEP). Mirrors the Ollama/Claude adapters.
_DEFAULT_MAX_RETRIES = 2
_DEFAULT_RETRY_BACKOFF_BASE = 0.5

#: OpenRouter variant suffixes (the ``:free`` / ``:nitro`` / … routing flavors). The served model
#: echo and the requested id may differ only by one of these — the SAME model, not a fallback.
_OR_VARIANTS = (":free", ":nitro", ":floor", ":extended", ":thinking", ":online", ":beta")

#: An embedded dated snapshot OpenRouter inserts in the served id (``north-mini-code-20260617``).
_OR_DATE = re.compile(r"-\d{8}\b")

#: Best-effort vendor/family canonicalization for the NON-FATAL mismatch warning ONLY. Maps both a
#: vendor prefix and an ``@family`` token to a canonical family; the warn fires only when BOTH are
#: recognized AND disagree, so an unknown vendor or family never produces a false positive — a
#: typo-catcher, not an authoritative table (the operator's ``@family`` remains the declared axis).
_VENDOR_FAMILY_CANON = {
    "deepseek": "deepseek",
    "qwen": "qwen",
    "mistral": "mistral",
    "mistralai": "mistral",
    "google": "google",
    "gemini": "google",
    "openai": "openai",
    "gpt": "openai",
    "anthropic": "anthropic",
    "claude": "anthropic",
    "meta-llama": "meta",
    "meta": "meta",
    "llama": "meta",
    "x-ai": "xai",
    "xai": "xai",
    "grok": "xai",
    "cohere": "cohere",
    "nvidia": "nvidia",
    "nemotron": "nvidia",
    "microsoft": "microsoft",
    "phi": "microsoft",
}


def is_openrouter_spec(model_id: str) -> bool:
    """Does ``model_id`` carry the ``openrouter:`` sentinel (i.e. route to this adapter)?"""
    return model_id.startswith(OPENROUTER_PREFIX)


def _norm_openrouter(model_id: str | None) -> str:
    """Normalize an OpenRouter id for served-vs-requested comparison (§11.6).

    Strips the ``openrouter:`` selector prefix, a trailing variant suffix (:data:`_OR_VARIANTS`),
    and an embedded dated snapshot (:data:`_OR_DATE`) — so
    ``openrouter:cohere/north-mini-code:free`` and the served
    ``cohere/north-mini-code-20260617:free`` compare EQUAL (the same model), while a genuine
    cross-vendor fallback (a ``deepseek/…`` served for a requested ``cohere/…``) survives and is
    caught.
    """
    m = (model_id or "").strip().lower()
    if m.startswith(OPENROUTER_PREFIX):
        m = m[len(OPENROUTER_PREFIX):]
    for suffix in _OR_VARIANTS:
        if m.endswith(suffix):
            m = m[: -len(suffix)]
            break
    return _OR_DATE.sub("", m)


def _warn_on_family_mismatch(wire_id: str, family: str) -> None:
    """Emit a NON-FATAL warning if the id's vendor prefix clearly contradicts the ``@family`` tag.

    Fires ONLY when both the ``vendor`` of a ``vendor/model`` id and the declared ``family`` resolve
    to KNOWN-but-DIFFERENT canonical families (:data:`_VENDOR_FAMILY_CANON`) — e.g.
    ``deepseek/deepseek-chat@qwen``. An unknown vendor or family, or an empty family (the seated
    placeholder), never warns. The operator's ``@family`` is still authoritative; this only surfaces
    a likely mislabel that would corrupt the cross-family attribution.
    """
    if "/" not in wire_id or not family:
        return
    vendor = wire_id.split("/", 1)[0].strip().lower()
    canon_v = _VENDOR_FAMILY_CANON.get(vendor)
    canon_f = _VENDOR_FAMILY_CANON.get(family.strip().lower())
    if canon_v and canon_f and canon_v != canon_f:
        warnings.warn(
            f"OpenRouter model {wire_id!r} has vendor {vendor!r} (≈ family {canon_v!r}) but was "
            f"tagged @{family} (≈ {canon_f!r}); the cross-family attribution may be wrong — verify "
            "the @family tag.",
            UserWarning,
            stacklevel=3,
        )


class OpenRouterUnreachableError(RuntimeError):
    """OpenRouter could not be reached (transport/connect failure, §8.6).

    Raised when the httpx call fails to *connect* — a refused connection or connect timeout before
    any HTTP status — so a bare ``httpx.ConnectError`` never escapes with no operator guidance
    (Ship-Gate B: code/message/hint, no raw stacks). It is transient
    (:func:`_is_transient_openrouter`) so the bounded retry absorbs a blip.
    """

    def __init__(self, base_url: str, model_id: str, cause: BaseException | None = None) -> None:
        self.base_url = base_url
        self.model_id = model_id
        self.__cause__ = cause
        super().__init__(
            f"[OPENROUTER_UNREACHABLE] cannot reach OpenRouter at {base_url!r} "
            f"for model {model_id!r}"
            + (f": {cause}" if cause else "")
            + " (hint: check network egress and OPENROUTER_BASE_URL; the endpoint is "
            "https://openrouter.ai/api/v1)"
        )


class OpenRouterBadResponseError(RuntimeError):
    """OpenRouter answered but the body is not a well-formed JSON object (§8.6).

    Raised at the transport/parse boundary on an HTTP 200 whose body is not decodable JSON (a proxy
    error page) or decodes to a non-mapping — model-named and hinted instead of an opaque
    ``JSONDecodeError`` / ``AttributeError`` downstream (Ship-Gate B).
    """

    def __init__(self, model_id: str, base_url: str, detail: str) -> None:
        self.model_id = model_id
        self.base_url = base_url
        self.detail = detail
        super().__init__(
            f"[OPENROUTER_BAD_RESPONSE] OpenRouter at {base_url!r} returned a malformed body for "
            f"model {model_id!r}: {detail} "
            "(hint: a proxy may be returning a non-JSON body, or OPENROUTER_BASE_URL is not an "
            "OpenAI-compatible /v1 surface)"
        )


def _is_transient_openrouter(exc: BaseException) -> bool:
    """Is ``exc`` a transient OpenRouter call failure worth retrying (§8.6)?

    Retryable: :class:`OpenRouterUnreachableError`, the underlying httpx connect/read/protocol
    errors, a **429 rate limit**, and a 5xx ``HTTPStatusError``. A 429 is the *expected*
    transient on OpenRouter's shared free pool (and under a sustained panel run's parallel
    cross-family calls) — the bounded backoff exists to absorb it, so treating it as fatal would
    drop a whole model's run on a momentary burst (the exact rate-limit-as-fatal gap the Claude
    adapter shares). NOT retryable: any other 4xx (a request error retrying won't fix), a
    :class:`ModelMismatchError` (a provenance breach the andon must see), or an
    :class:`OpenRouterBadResponseError` (a shape problem, not a blip). httpx is imported lazily so
    the module still imports without it.
    """
    if isinstance(exc, (ModelMismatchError, OpenRouterBadResponseError)):
        return False
    if isinstance(exc, OpenRouterUnreachableError):
        return True
    try:
        import httpx
    except ImportError:  # pragma: no cover - httpx is a declared dependency
        return False
    if isinstance(
        exc,
        (
            httpx.ConnectError,
            httpx.ConnectTimeout,
            httpx.ReadError,
            httpx.ReadTimeout,
            httpx.RemoteProtocolError,
        ),
    ):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        status = exc.response.status_code
        return status == 429 or 500 <= status < 600
    return False


def _extract_text(response: dict[str, Any]) -> str:
    """Pull the assistant text out of an OpenAI-compatible ``/chat/completions`` response.

    Reads ``choices[0].message.content``. Falls back to ``""`` on any missing field — an empty
    completion is a legitimate (if unhelpful) answer the caller scores, matching the Ollama
    adapter's graceful contract (never raises on a shape gap; the served-match + transport guards
    live elsewhere).

    The content is normalized through the SHARED :func:`~ai_crucible.models.ollama_adapter.
    _normalize_harmony` (identity pass-through for clean text), so a gpt-oss model served via
    OpenRouter that leaks OpenAI **Harmony** chat-template control tokens (``<|channel|>analysis
    <|message|>…``) into ``content`` is collapsed to its final-channel answer BEFORE the judge /
    solver-loop parser sees it — the same defense the Ollama adapter already applies (the OpenAI
    family is reachable on this gateway, so the leak the Ollama path hit is reachable here too).
    A deepseek/qwen/cohere response carries no Harmony tokens and is returned unchanged.
    """
    choices = response.get("choices")
    if isinstance(choices, list) and choices and isinstance(choices[0], dict):
        message = choices[0].get("message")
        if isinstance(message, dict):
            content = message.get("content")
            if isinstance(content, str):
                return _normalize_harmony(content)
    return ""


#: Structural type for the OpenAI-compatible chat client the adapter drives. The built-in httpx
#: caller and any injected fake satisfy it; ``__call__`` returns the raw OpenAI response mapping
#: (``{"choices": [{"message": {"content": ...}}], "model": ..., ...}``).
OpenRouterClient = Callable[..., Awaitable[dict[str, Any]]]


class OpenRouterModel:
    """An OpenRouter-served model: kernel ``generate`` / panel judge / characterization probe.

    Args:
        model_id: the spec id, e.g. ``"openrouter:deepseek/deepseek-chat"`` (the ``openrouter:``
            prefix is kept as the stable identity; the bare ``deepseek/deepseek-chat`` is what is
            sent to OpenRouter — :attr:`_wire_id`).
        family: the model *family* for EXTERNAL_VERIFIER exclusion (the operator's declared axis,
            §10.2). May be ``""`` at seated-panel reconstruction (re-bound from the seat record).
        quant: surface-parity with the Ollama adapter; hosted models are not user-quantized, so it
            is ``None`` in practice but carried into every :class:`JudgmentRecord`.
        api_key: the OpenRouter key; ``None`` (default) reads ``OPENROUTER_API_KEY`` from the env at
            call time, never captured at import.
        base_url: the OpenAI-compatible base; ``None`` reads ``OPENROUTER_BASE_URL`` then
            :data:`DEFAULT_BASE_URL`.
        client: an injected client of the :data:`OpenRouterClient` shape; ``None`` resolves the
            built-in httpx caller lazily. Injected so tests pass a fake and make no network call.
        max_tokens: the generation ceiling pinned per request (PIN_PER_STEP).
        max_retries / retry_backoff_base / sleep: the bounded transient-retry policy (§8.6),
            mirrored from the other adapters and recorded in :meth:`pin_metadata`.
    """

    def __init__(
        self,
        model_id: str,
        family: str,
        quant: str | None = None,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        client: OpenRouterClient | None = None,
        max_tokens: int = _DEFAULT_MAX_TOKENS,
        max_retries: int = _DEFAULT_MAX_RETRIES,
        retry_backoff_base: float = _DEFAULT_RETRY_BACKOFF_BASE,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self.model_id = model_id
        # The bare id sent to OpenRouter + used for the served-match (the prefix is identity-only).
        self._wire_id = (
            model_id[len(OPENROUTER_PREFIX):] if is_openrouter_spec(model_id) else model_id
        )
        self.family = family
        self.quant = quant
        self._api_key = api_key
        self._base_url = base_url
        self._client = client
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.retry_backoff_base = retry_backoff_base
        self._sleep = sleep
        _warn_on_family_mismatch(self._wire_id, family)

    # -- provenance (PIN_PER_STEP) ------------------------------------------ #

    def pin_metadata(self) -> dict[str, Any]:
        """The PIN_PER_STEP provenance block stamped on characterization records.

        Records the (prefixed) identity, family, quant, and the pinned request knobs
        (``temperature=0`` + ``max_tokens`` + the logprob request). OpenRouter exposes no seed, so
        determinism rests on ``temperature=0`` + the model pin — captured here so the profile run is
        as replayable as the API allows (§11.6).
        """
        return {
            "model_id": self.model_id,
            "family": self.family,
            "quant": self.quant,
            "options": {
                "temperature": 0,
                "max_tokens": self.max_tokens,
                "logprobs": True,
                "top_logprobs": 1,
            },
            "transport": "openrouter (/v1/chat/completions)",
            "retry": {
                "max_retries": self.max_retries,
                "backoff_base": self.retry_backoff_base,
            },
        }

    # -- client resolution (lazy, secret read at call time) ----------------- #

    def _base(self) -> str:
        return self._base_url or os.environ.get("OPENROUTER_BASE_URL", DEFAULT_BASE_URL)

    def _resolve_client(self) -> OpenRouterClient:
        """Return the injected client, or lazily build the built-in httpx caller at call time.

        Reads ``OPENROUTER_API_KEY`` / ``OPENROUTER_BASE_URL`` from the env NOW (never at import),
        POSTs the OpenAI body to ``<base>/chat/completions`` with a Bearer header, and surfaces a
        connect failure as the structured :class:`OpenRouterUnreachableError` (transient) and a
        malformed body as :class:`OpenRouterBadResponseError`. A 4xx/5xx ``HTTPStatusError`` from
        ``raise_for_status`` propagates to the retry classifier.
        """
        if self._client is not None:
            return self._client
        import httpx

        base = self._base()
        api_key = self._api_key or os.environ.get("OPENROUTER_API_KEY", "")
        wire_id = self._wire_id

        async def _via_httpx(**kwargs: Any) -> dict[str, Any]:
            payload = {k: v for k, v in kwargs.items() if v is not None}
            payload.setdefault("model", wire_id)
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            }
            try:
                async with httpx.AsyncClient(base_url=base, timeout=600.0, headers=headers) as http:
                    r = await http.post("/chat/completions", json=payload)
                    r.raise_for_status()
            except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
                raise OpenRouterUnreachableError(base, wire_id, cause=exc) from exc
            try:
                body = r.json()
            except (ValueError, httpx.DecodingError) as exc:
                raise OpenRouterBadResponseError(
                    wire_id, base, f"response body is not valid JSON: {r.text[:200]!r}"
                ) from exc
            if not isinstance(body, dict):
                raise OpenRouterBadResponseError(
                    wire_id, base, f"response body is {type(body).__name__}, not an object"
                )
            return body

        self._client = _via_httpx
        return self._client

    # -- core completion ---------------------------------------------------- #

    async def _complete_raw(
        self, messages: list[dict[str, Any]], *, tools: list[dict[str, Any]] | None = None
    ) -> dict[str, Any]:
        """Call OpenRouter chat with the pinned options and return the **raw response**.

        Sends the OpenAI body (``temperature=0`` + ``max_tokens`` + ``logprobs``/``top_logprobs``,
        plus ``tools`` when supplied) and returns the mapping untouched so :meth:`judge_item` can
        read the verdict-token logprob for confidence (§12). The single client call is wrapped in
        the SHARED bounded retry (:func:`_call_with_retry`) with the OpenRouter classifier, then the
        served-model provenance guard runs (§11.6).
        """
        client = self._resolve_client()
        extra = {"tools": tools} if tools else {}

        async def _call() -> dict[str, Any]:
            return await client(
                model=self._wire_id,
                messages=messages,
                temperature=0,
                max_tokens=self.max_tokens,
                logprobs=True,
                top_logprobs=1,
                **extra,
            )

        response = await _call_with_retry(
            _call,
            max_retries=self.max_retries,
            backoff_base=self.retry_backoff_base,
            sleep=self._sleep,
            is_transient=_is_transient_openrouter,
        )
        if not isinstance(response, dict):
            raise OpenRouterBadResponseError(
                self._wire_id,
                self._base(),
                f"client returned {type(response).__name__}, not a mapping",
            )
        self._assert_served_matches(response)
        return response

    def _assert_served_matches(self, response: dict[str, Any]) -> None:
        """Raise :class:`ModelMismatchError` if the served model != the requested one (§11.6).

        OpenRouter (OpenAI shape) echoes the served model in a top-level ``model`` field, often a
        dated snapshot (``…-20260617``) or a variant (``:free``); :func:`_norm_openrouter` folds
        both out before comparing to :attr:`_wire_id`. A *missing* ``model`` field is tolerated
        (fakes that do not echo it) so the guard fails CLOSED only on a *positive* cross-vendor
        mismatch.
        """
        served = response.get("model")
        if (
            isinstance(served, str)
            and served
            and _norm_openrouter(served) != _norm_openrouter(self._wire_id)
        ):
            raise ModelMismatchError(self._wire_id, served)

    async def complete(self, messages: list[dict[str, Any]]) -> str:
        """Call OpenRouter chat with pinned determinism and return the assistant text."""
        return _extract_text(await self._complete_raw(messages))

    # -- kernel generate plug (§10.2) --------------------------------------- #

    async def generate(self, state: AttemptState) -> str:
        """Kernel ``generate`` choke point: complete on ``state.messages`` → text (§10.2)."""
        return await self.complete(state.messages)

    # -- panel judge plug (§10.2, EXTERNAL_VERIFIER) ------------------------ #

    async def judge(self, attempt: AttemptState) -> Score:
        """Score an attempt as a panel judge → :class:`~ai_crucible.types.Score`.

        Thin parse here (the §11.3 rubric lands with the calibration set); the verdict text and the
        judging ``model_id``/``family`` are recorded in ``Score.metadata``. Use :meth:`as_judge` for
        the panel-ready callable carrying ``.family``.
        """
        verdict = await self.complete(self._judge_messages(attempt))
        return Score(
            value=verdict,
            explanation=verdict,
            metadata={"judge_model": self.model_id, "judge_family": self.family},
        )

    def as_judge(self) -> JudgeFn:
        """Return a panel-ready judge callable tagged with this model's ``family``.

        :class:`~ai_crucible.scoring.judge_panel.JudgePanel` reads ``.family`` off the callable to
        enforce same-family exclusion (§10.2). A bound method cannot carry the attribute, so this
        wraps :meth:`judge` and stamps ``.family`` — the supported way to seat (and thus to exclude)
        this model on a panel.
        """

        async def _judge(attempt: AttemptState) -> Score:
            return await self.judge(attempt)

        _judge.family = self.family  # type: ignore[attr-defined]
        return _judge

    def _judge_messages(self, attempt: AttemptState) -> list[dict[str, Any]]:
        """Build the messages for a judging call (placeholder rubric, §11.3).

        Reuses the attempt's scored context + Solver output. NEVER reads ``attempt.chrome`` — Tier-3
        is sealed out of every model context (§10.1(e)).
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

        Identical contract to :meth:`OllamaModel.judge_item`: records ``predicted`` + measured
        ``latency_s`` + the **verdict-token logprob → confidence** (§12, via the shared
        :func:`_first_token_logprob` which handles the OpenAI logprob shape), stamps
        ``model_id``/``family``/``quant`` + the PIN_PER_STEP provenance into ``metadata``, and
        leaves ``gold``/``correct`` ``None`` for the scorer that holds the labels
        (DECOMPOSE_BY_SECRETS, §11.6). ``run_index`` drives test-retest; ``position`` drives
        position-swap bias.
        """
        messages = [{"role": "user", "content": prompt}]
        start = time.monotonic()
        response = await self._complete_raw(messages)
        latency_s = time.monotonic() - start
        predicted = _extract_text(response)
        confidence = _logprob_to_probability(_first_token_logprob(response))
        return JudgmentRecord(
            item_id=_item_id(prompt),
            model_id=self.model_id,
            predicted=predicted,
            gold=None,
            quant=self.quant,
            confidence=confidence,
            latency_s=latency_s,
            run_index=run_index,
            position=position,
            family=self.family,
            metadata=self.pin_metadata(),
        )
