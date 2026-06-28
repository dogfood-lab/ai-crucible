"""Tests for the OpenRouter model adapter + its wiring into cli/characterize.

Like ``tests/test_models.py``, every test passes a **fake client** of the adapter's shape, so no
network call is made and no SDK is needed (build-law 3). Async methods are driven with
:func:`asyncio.run` (no ``pytest-asyncio`` dependency).

Covered: the OpenAI-compatible transport (wire-id strips the ``openrouter:`` prefix; identity keeps
it), ``generate``/``judge``/``judge_item`` parity with the other adapters, the verdict-token logprob
→ confidence path (§12), the OpenRouter-lenient served-model guard (dated snapshot ≈ same model; a
cross-vendor fallback raises), the NON-FATAL vendor↔family mismatch warning, and the
cli/characterize routing (the ``openrouter:`` sentinel + the required-@family fail-fast).
"""

from __future__ import annotations

import asyncio
import math
import warnings
from typing import Any

import pytest

from ai_crucible.characterize.run import _parse_models
from ai_crucible.characterize.types import JudgmentRecord
from ai_crucible.cli import _build_model
from ai_crucible.models import OpenRouterModel, is_openrouter_spec
from ai_crucible.models.ollama_adapter import ModelMismatchError
from ai_crucible.models.openrouter_adapter import (
    OpenRouterBadResponseError,
    _norm_openrouter,
)
from ai_crucible.scoring.judge_panel import judge_family
from ai_crucible.types import AttemptState, Budget, FramingArm

_SPEC = "openrouter:deepseek/deepseek-chat"
_WIRE = "deepseek/deepseek-chat"


# --------------------------------------------------------------------------- #
# Fake client — OpenAI /chat/completions shape; records call kwargs.
# --------------------------------------------------------------------------- #


class FakeOpenRouterClient:
    """A fake of the :data:`OpenRouterClient` shape — returns the OpenAI response mapping."""

    def __init__(
        self,
        content: str = "or-says-hi",
        *,
        served_model: str = _WIRE,
        first_logprob: float | None = None,
    ) -> None:
        self.content = content
        self.served_model = served_model
        self.first_logprob = first_logprob
        self.calls: list[dict[str, Any]] = []

    async def __call__(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(kwargs)
        choice: dict[str, Any] = {"message": {"role": "assistant", "content": self.content}}
        if self.first_logprob is not None:
            choice["logprobs"] = {
                "content": [{"token": self.content[:1] or " ", "logprob": self.first_logprob}]
            }
        return {"model": self.served_model, "choices": [choice], "usage": {}}


@pytest.fixture
def attempt() -> AttemptState:
    return AttemptState(
        attempt_id="att-or-1",
        puzzle_id="seed-or",
        model="under-test",
        framing_arm=FramingArm.SELF_REFERENTIAL,
        messages=[{"role": "user", "content": "solve this"}],
        output="the candidate answer",
        budget=Budget(tool_call_budget=8, time_budget_seconds=300),
    )


def _model(client: FakeOpenRouterClient, family: str = "deepseek") -> OpenRouterModel:
    return OpenRouterModel(_SPEC, family=family, client=client)


# --------------------------------------------------------------------------- #
# transport: identity keeps the prefix, the wire id strips it
# --------------------------------------------------------------------------- #


def test_identity_keeps_prefix_wire_strips_it(attempt: AttemptState) -> None:
    client = FakeOpenRouterClient("out")
    model = _model(client)
    assert model.model_id == _SPEC  # stable identity (round-trips through the seat)
    out = asyncio.run(model.generate(attempt))
    assert out == "out"
    # The bare id (no openrouter: prefix) is what reaches the OpenRouter endpoint.
    assert client.calls[0]["model"] == _WIRE
    assert client.calls[0]["messages"] == attempt.messages


def test_generate_equals_complete(attempt: AttemptState) -> None:
    client = FakeOpenRouterClient("same")
    model = _model(client)
    via_generate = asyncio.run(model.generate(attempt))
    via_complete = asyncio.run(model.complete(attempt.messages))
    assert via_generate == via_complete == "same"


def test_pins_temperature_zero_and_max_tokens(attempt: AttemptState) -> None:
    client = FakeOpenRouterClient()
    model = OpenRouterModel(_SPEC, family="deepseek", client=client, max_tokens=256)
    asyncio.run(model.generate(attempt))
    call = client.calls[0]
    assert call["temperature"] == 0
    assert call["max_tokens"] == 256
    assert call["logprobs"] is True


def test_empty_content_is_graceful_empty_string() -> None:
    # An empty completion is a legitimate (if unhelpful) answer the caller scores — matches the
    # Ollama adapter's graceful contract, never raises.
    model = _model(FakeOpenRouterClient(""))
    assert asyncio.run(model.complete([{"role": "user", "content": "hi"}])) == ""


# --------------------------------------------------------------------------- #
# judge_item → JudgmentRecord (+ confidence from the OpenAI logprob shape, §12)
# --------------------------------------------------------------------------- #


def test_judge_item_record_shape() -> None:
    model = OpenRouterModel(_SPEC, family="deepseek", quant=None, client=FakeOpenRouterClient("A"))
    rec = asyncio.run(model.judge_item("which is better, A or B?", run_index=2, position=1))
    assert isinstance(rec, JudgmentRecord)
    assert rec.model_id == _SPEC  # the prefixed identity is recorded
    assert rec.family == "deepseek"
    assert rec.predicted == "A"
    assert rec.run_index == 2
    assert rec.position == 1
    assert rec.gold is None and rec.correct is None  # filled by the scorer (§11.6)
    assert rec.confidence is None  # no logprob channel → never fabricated (§12)
    assert rec.metadata["model_id"] == _SPEC
    assert rec.metadata["options"]["temperature"] == 0


def test_judge_item_confidence_from_openai_logprob() -> None:
    client = FakeOpenRouterClient("A", first_logprob=-0.10536)  # exp(-0.10536) ≈ 0.90
    model = _model(client)
    rec = asyncio.run(model.judge_item("rate this"))
    assert rec.confidence is not None
    assert rec.confidence == pytest.approx(math.exp(-0.10536), abs=1e-4)


def test_item_id_stable_for_same_prompt() -> None:
    model = _model(FakeOpenRouterClient())
    r1 = asyncio.run(model.judge_item("identical"))
    r2 = asyncio.run(model.judge_item("identical"))
    assert r1.item_id == r2.item_id


# --------------------------------------------------------------------------- #
# as_judge carries .family for the panel's same-family exclusion (§10.2)
# --------------------------------------------------------------------------- #


def test_as_judge_carries_family_tag() -> None:
    judge = _model(FakeOpenRouterClient(), family="deepseek").as_judge()
    assert judge_family(judge) == "deepseek"


# --------------------------------------------------------------------------- #
# served-model provenance guard — OpenRouter-lenient (§11.6 / models-cli-002)
# --------------------------------------------------------------------------- #


def test_dated_snapshot_and_variant_are_the_same_model() -> None:
    # OpenRouter echoes a dated snapshot + variant for the requested bare id — NOT a fallback.
    client = FakeOpenRouterClient(served_model="deepseek/deepseek-chat-20260617:free")
    out = asyncio.run(_model(client).complete([{"role": "user", "content": "x"}]))
    assert out == "or-says-hi"  # no ModelMismatchError raised


def test_cross_vendor_fallback_raises_mismatch() -> None:
    client = FakeOpenRouterClient(served_model="qwen/qwen-2.5-72b-instruct")
    with pytest.raises(ModelMismatchError):
        asyncio.run(_model(client).complete([{"role": "user", "content": "x"}]))


def test_missing_served_model_is_tolerated() -> None:
    class _NoModel(FakeOpenRouterClient):
        async def __call__(self, **kwargs: Any) -> dict[str, Any]:
            return {"choices": [{"message": {"content": "ok"}}]}  # no top-level "model"

    out = asyncio.run(_model(_NoModel()).complete([{"role": "user", "content": "x"}]))
    assert out == "ok"


def test_non_mapping_response_raises_bad_response() -> None:
    class _Bad:
        async def __call__(self, **kwargs: Any) -> Any:
            return ["not", "a", "mapping"]

    with pytest.raises(OpenRouterBadResponseError):
        asyncio.run(_model(_Bad()).complete([{"role": "user", "content": "x"}]))  # type: ignore[arg-type]


def test_norm_openrouter_folds_prefix_variant_and_date() -> None:
    assert _norm_openrouter("openrouter:cohere/north-mini-code:free") == "cohere/north-mini-code"
    assert _norm_openrouter("cohere/north-mini-code-20260617:free") == "cohere/north-mini-code"
    assert _norm_openrouter("deepseek/deepseek-chat") == "deepseek/deepseek-chat"


# --------------------------------------------------------------------------- #
# NON-FATAL vendor↔family mismatch warning (the typo-catcher)
# --------------------------------------------------------------------------- #


def test_warns_on_clear_vendor_family_mismatch() -> None:
    with pytest.warns(UserWarning, match="cross-family attribution may be wrong"):
        OpenRouterModel("openrouter:deepseek/deepseek-chat", family="qwen")


def test_no_warning_when_vendor_matches_family() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any UserWarning would fail the test
        OpenRouterModel("openrouter:deepseek/deepseek-chat", family="deepseek")


def test_no_warning_for_unknown_vendor_or_family() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        OpenRouterModel("openrouter:somevendor/some-model", family="whatever")
        OpenRouterModel("openrouter:deepseek/deepseek-chat", family="")  # seated placeholder


# --------------------------------------------------------------------------- #
# routing: the openrouter: sentinel + the required-@family fail-fast
# --------------------------------------------------------------------------- #


def test_is_openrouter_spec() -> None:
    assert is_openrouter_spec(_SPEC)
    assert not is_openrouter_spec("mistral-small:24b")


def test_cli_build_model_routes_openrouter() -> None:
    model = _build_model("openrouter:deepseek/deepseek-chat@deepseek")
    assert isinstance(model, OpenRouterModel)
    assert model.model_id == _SPEC
    assert model.family == "deepseek"


def test_cli_build_model_openrouter_requires_family() -> None:
    with pytest.raises(ValueError, match="OPENROUTER_NO_FAMILY"):
        _build_model("openrouter:deepseek/deepseek-chat")


def test_parse_models_openrouter_requires_family() -> None:
    with pytest.raises(ValueError, match="OPENROUTER_NO_FAMILY"):
        _parse_models(["openrouter:deepseek/deepseek-chat"])


def test_parse_models_openrouter_with_family_ok() -> None:
    assert _parse_models(["openrouter:deepseek/deepseek-chat@deepseek"]) == [
        (_SPEC, "deepseek", None)
    ]


def test_parse_models_untagged_local_unchanged() -> None:
    # The existing untagged-local contract is preserved (family None, not the colliding "unknown").
    assert _parse_models(["mistral-small:24b"]) == [("mistral-small:24b", None, None)]
