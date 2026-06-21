"""Tests for the Phase-2 model adapters (Ollama + Claude).

These prove the adapters bridge Phase-1's injected stubs to real models *without ever
touching the network* (build-law 3): every test passes a **fake client** of the
adapter's expected shape, so no live Ollama or Anthropic call is made and neither SDK
needs to be installed.

Covered behaviors:

* ``generate()`` returns the fake client's text and routes through ``complete()``
  (one model-I/O path, §10.2) — for both adapters.
* ``judge_item()`` returns a well-formed :class:`~ai_crucible.characterize.types.JudgmentRecord`
  with ``family`` / ``model_id`` / ``quant`` / ``predicted`` set, ``gold``/``correct``
  left ``None`` (filled later by the scorer, §11.6), and the PIN_PER_STEP provenance in
  ``metadata`` (§11.1, §11.6).
* the ``.family`` tag from ``as_judge()`` drives
  :class:`~ai_crucible.scoring.judge_panel.JudgePanel` same-family exclusion — a ``ClaudeModel``
  judge is dropped when the panel excludes ``"claude"`` (EXTERNAL_VERIFIER, §10.2).
* the pinned deterministic options (``temperature=0`` + ``seed`` for Ollama, §11.2;
  ``temperature=0`` for Claude) are actually passed to the client on every call.

Async methods are driven with :func:`asyncio.run` so the suite needs no
``pytest-asyncio`` plugin (matching ``tests/test_scoring.py``; it is not a declared
dependency).
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from ai_crucible.characterize.types import JudgmentRecord
from ai_crucible.models import ClaudeModel, OllamaModel
from ai_crucible.models.ollama_adapter import ModelMismatchError, _norm_model
from ai_crucible.scoring.judge_panel import JudgePanel, judge_family
from ai_crucible.types import AttemptState, Budget, FramingArm, Score

# --------------------------------------------------------------------------- #
# Fake clients — match each adapter's expected client shape, record call kwargs.
# --------------------------------------------------------------------------- #


class FakeOllamaClient:
    """A fake of the :data:`ai_crucible.models.ollama_adapter.OllamaClient` shape.

    Callable as ``await client(model=..., messages=..., options=...)`` and returns the
    Ollama ``/api/chat`` mapping. Records every call's kwargs so tests can assert the
    deterministic options were passed.
    """

    def __init__(self, content: str = "ollama-says-hi") -> None:
        self.content = content
        self.calls: list[dict[str, Any]] = []

    async def __call__(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(kwargs)
        return {"message": {"role": "assistant", "content": self.content}}


class _FakeMessages:
    """The ``messages`` sub-object of the fake Anthropic client."""

    def __init__(self, parent: FakeClaudeClient) -> None:
        self._parent = parent

    async def create(self, **kwargs: Any) -> dict[str, Any]:
        self._parent.calls.append(kwargs)
        # Messages API content is a list of typed blocks.
        return {"content": [{"type": "text", "text": self._parent.content}]}


class FakeClaudeClient:
    """A fake of the :class:`ai_crucible.models.claude_adapter.ClaudeClient` shape.

    Exposes ``.messages.create(...)`` and records call kwargs for assertions.
    """

    def __init__(self, content: str = "claude-says-hi") -> None:
        self.content = content
        self.calls: list[dict[str, Any]] = []
        self.messages = _FakeMessages(self)


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture
def attempt() -> AttemptState:
    """A minimal attempt with a scored context + an output for the judge to rule on."""
    return AttemptState(
        attempt_id="att-models-1",
        puzzle_id="seed-models",
        model="under-test",
        framing_arm=FramingArm.SELF_REFERENTIAL,
        messages=[{"role": "user", "content": "solve this"}],
        output="the candidate answer",
        budget=Budget(tool_call_budget=8, time_budget_seconds=300),
    )


# --------------------------------------------------------------------------- #
# generate() routes through complete() and returns the client's text
# --------------------------------------------------------------------------- #


def test_ollama_generate_routes_through_complete(attempt: AttemptState) -> None:
    client = FakeOllamaClient("ollama-out")
    model = OllamaModel("mistral-small:24b", family="mistral", client=client)

    out = asyncio.run(model.generate(attempt))

    assert out == "ollama-out"
    # generate() must have called the client exactly once with the attempt's messages.
    assert len(client.calls) == 1
    assert client.calls[0]["messages"] == attempt.messages
    assert client.calls[0]["model"] == "mistral-small:24b"


def test_ollama_generate_equals_complete(attempt: AttemptState) -> None:
    """generate(state) is complete(state.messages) — the single-choke-point contract."""
    client = FakeOllamaClient("same-text")
    model = OllamaModel("mistral-small:24b", family="mistral", client=client)

    via_generate = asyncio.run(model.generate(attempt))
    via_complete = asyncio.run(model.complete(attempt.messages))

    assert via_generate == via_complete == "same-text"


def test_claude_generate_routes_through_complete(attempt: AttemptState) -> None:
    client = FakeClaudeClient("claude-out")
    model = ClaudeModel("claude-opus-4-8", client=client)

    out = asyncio.run(model.generate(attempt))

    assert out == "claude-out"
    assert len(client.calls) == 1
    # The user turn survives; system turns (none here) would be split out.
    assert client.calls[0]["messages"] == attempt.messages
    assert client.calls[0]["model"] == "claude-opus-4-8"


def test_claude_splits_system_message() -> None:
    """A leading system turn is lifted to the top-level ``system`` param (Messages API)."""
    client = FakeClaudeClient("ok")
    model = ClaudeModel("claude-opus-4-8", client=client)
    messages = [
        {"role": "system", "content": "be terse"},
        {"role": "user", "content": "hi"},
    ]

    asyncio.run(model.complete(messages))

    call = client.calls[0]
    assert call["system"] == "be terse"
    assert call["messages"] == [{"role": "user", "content": "hi"}]


# --------------------------------------------------------------------------- #
# Deterministic params are passed to the client (§11.2 / PIN_PER_STEP)
# --------------------------------------------------------------------------- #


def test_ollama_passes_deterministic_options(attempt: AttemptState) -> None:
    """temp 0 + fixed seed + fixed num_ctx must reach the client (§11.2)."""
    client = FakeOllamaClient()
    model = OllamaModel(
        "qwen3:32b", family="qwen", quant="q4_K_M", client=client, num_ctx=4096, seed=7
    )

    asyncio.run(model.generate(attempt))

    options = client.calls[0]["options"]
    assert options["temperature"] == 0
    assert options["seed"] == 7
    assert options["num_ctx"] == 4096


def test_claude_passes_temperature_zero(attempt: AttemptState) -> None:
    """Claude has no seed knob; determinism rests on temperature=0 + the model pin."""
    client = FakeClaudeClient()
    model = ClaudeModel("claude-opus-4-8", client=client, max_tokens=256)

    asyncio.run(model.generate(attempt))

    call = client.calls[0]
    assert call["temperature"] == 0
    assert call["max_tokens"] == 256


# --------------------------------------------------------------------------- #
# judge_item() → a well-formed JudgmentRecord (the §11.1 metric unit)
# --------------------------------------------------------------------------- #


def test_ollama_judge_item_returns_record() -> None:
    client = FakeOllamaClient("predicted-A")
    model = OllamaModel("qwen3:32b", family="qwen", quant="q5_K_M", client=client, seed=3)

    record = asyncio.run(model.judge_item("which is better, A or B?", run_index=2, position=1))

    assert isinstance(record, JudgmentRecord)
    assert record.model_id == "qwen3:32b"
    assert record.family == "qwen"
    assert record.quant == "q5_K_M"
    assert record.predicted == "predicted-A"
    assert record.run_index == 2
    assert record.position == 1
    # gold/correct are filled later by the scorer that holds the labels (§11.6).
    assert record.gold is None
    assert record.correct is None
    # The base fake returns no logprob channel → confidence stays None, never fabricated
    # (§12: confidence comes from the verdict-token logprob, else None).
    assert record.confidence is None
    assert record.latency_s >= 0.0
    # PIN_PER_STEP: the metadata makes the profile run replayable.
    assert record.metadata["model_id"] == "qwen3:32b"
    assert record.metadata["quant"] == "q5_K_M"
    assert record.metadata["options"]["temperature"] == 0
    assert record.metadata["options"]["seed"] == 3


def test_claude_judge_item_returns_record() -> None:
    client = FakeClaudeClient("verdict-B")
    model = ClaudeModel("claude-opus-4-8", client=client)

    record = asyncio.run(model.judge_item("rate this answer"))

    assert isinstance(record, JudgmentRecord)
    assert record.model_id == "claude-opus-4-8"
    assert record.family == "claude"
    assert record.predicted == "verdict-B"
    assert record.run_index == 0
    assert record.gold is None
    assert record.correct is None
    assert record.metadata["family"] == "claude"
    assert record.metadata["options"]["temperature"] == 0


def test_judge_item_id_is_stable_for_same_prompt() -> None:
    """Same prompt → same item_id (records are keyed by item for the profiler)."""
    model = OllamaModel("qwen3:32b", family="qwen", client=FakeOllamaClient())
    r1 = asyncio.run(model.judge_item("identical prompt"))
    r2 = asyncio.run(model.judge_item("identical prompt"))
    assert r1.item_id == r2.item_id


# --------------------------------------------------------------------------- #
# Verdict-token logprob → JudgmentRecord.confidence (§12 — the metric the first
# characterization run could not compute). confidence = exp(first-token logprob);
# None when the server returns no logprob channel. Every client is faked (no network).
# --------------------------------------------------------------------------- #


class FakeLogprobClient:
    """Ollama native ``/api/chat`` fake that ALSO returns per-token ``logprobs``.

    Shape: ``{"message": {...}, "logprobs": [{"token", "logprob"}, ...]}`` — the native
    shape Ollama emits with ``logprobs: true`` since v0.12.11. The first entry is the
    verdict token, so its ``logprob`` drives confidence.
    """

    def __init__(self, content: str, first_logprob: float) -> None:
        self.content = content
        self.first_logprob = first_logprob
        self.calls: list[dict[str, Any]] = []

    async def __call__(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(kwargs)
        return {
            "message": {"role": "assistant", "content": self.content},
            "logprobs": [
                {"token": self.content[:1] or " ", "logprob": self.first_logprob},
                {"token": "x", "logprob": -3.0},
            ],
        }


class FakeNestedLogprobClient:
    """Native fake with ``logprobs`` nested under ``message`` (some Ollama builds)."""

    def __init__(self, content: str, first_logprob: float) -> None:
        self.content = content
        self.first_logprob = first_logprob

    async def __call__(self, **kwargs: Any) -> dict[str, Any]:
        return {
            "message": {
                "role": "assistant",
                "content": self.content,
                "logprobs": [{"token": self.content[:1], "logprob": self.first_logprob}],
            }
        }


class FakeOpenAILogprobClient:
    """OpenAI-compatible fake: ``choices[0].logprobs.content[0].logprob``."""

    def __init__(self, content: str, first_logprob: float) -> None:
        self.content = content
        self.first_logprob = first_logprob

    async def __call__(self, **kwargs: Any) -> dict[str, Any]:
        token = {"token": self.content[:1], "logprob": self.first_logprob}
        return {
            "message": {"role": "assistant", "content": self.content},
            "choices": [{"logprobs": {"content": [token]}}],
        }


def test_judge_item_confidence_from_native_logprob() -> None:
    """confidence = exp(verdict-token logprob) on the native /api/chat shape (§12)."""
    import math

    client = FakeLogprobClient("A", first_logprob=-0.105360516)  # exp(...) ≈ 0.90
    model = OllamaModel("qwen3:32b", family="qwen", client=client)

    record = asyncio.run(model.judge_item("Q: which? A or B"))

    assert record.predicted == "A"
    assert record.confidence is not None
    assert record.confidence == pytest.approx(math.exp(-0.105360516))
    assert record.confidence == pytest.approx(0.9, abs=1e-3)
    # the request actually asked for logprobs (PIN_PER_STEP). Ollama 0.24.0 honours
    # these only at the request TOP LEVEL, not nested under options — so that is where
    # the adapter sends them, and what we assert (the live-server contract).
    assert client.calls[0]["logprobs"] is True
    assert client.calls[0]["top_logprobs"] == 1
    assert "logprobs" not in client.calls[0]["options"]


def test_judge_item_confidence_from_nested_logprob() -> None:
    """A confident verdict (logprob≈0) → confidence≈1.0; nested-under-message shape."""
    client = FakeNestedLogprobClient("B", first_logprob=0.0)
    model = OllamaModel("qwen3:32b", family="qwen", client=client)

    record = asyncio.run(model.judge_item("pick A or B"))

    assert record.predicted == "B"
    assert record.confidence == pytest.approx(1.0)


def test_judge_item_confidence_from_openai_logprob() -> None:
    """confidence is also read from the OpenAI-compatible logprobs shape (§12)."""
    import math

    client = FakeOpenAILogprobClient("PASS", first_logprob=-0.6931472)  # exp ≈ 0.5
    model = OllamaModel("mistral-small:24b", family="mistral", client=client)

    record = asyncio.run(model.judge_item("grade: PASS or FAIL"))

    assert record.predicted == "PASS"
    assert record.confidence == pytest.approx(math.exp(-0.6931472))
    assert record.confidence == pytest.approx(0.5, abs=1e-3)


def test_judge_item_confidence_none_when_logprobs_absent() -> None:
    """No logprob channel (older server / logprobs off) → confidence is None, not faked."""
    client = FakeOllamaClient("A")  # base fake: message only, no logprobs
    model = OllamaModel("qwen3:32b", family="qwen", client=client)

    record = asyncio.run(model.judge_item("Q: which? A or B"))

    assert record.predicted == "A"
    assert record.confidence is None


def test_judge_item_confidence_clamped_to_unit_interval() -> None:
    """A tiny positive logprob (fp noise at ≈0) clamps to ≤ 1.0, never overshoots."""
    client = FakeLogprobClient("A", first_logprob=1e-9)  # exp ≈ 1.000000001
    model = OllamaModel("qwen3:32b", family="qwen", client=client)

    record = asyncio.run(model.judge_item("A or B"))

    assert record.confidence is not None
    assert 0.0 <= record.confidence <= 1.0


# --------------------------------------------------------------------------- #
# .family exclusion works with the real JudgePanel (EXTERNAL_VERIFIER, §10.2)
# --------------------------------------------------------------------------- #


def test_as_judge_carries_family_attribute() -> None:
    ollama = OllamaModel("qwen3:32b", family="qwen", client=FakeOllamaClient())
    claude = ClaudeModel("claude-opus-4-8", client=FakeClaudeClient())

    assert judge_family(ollama.as_judge()) == "qwen"
    assert judge_family(claude.as_judge()) == "claude"


def test_panel_excludes_claude_family_judge(attempt: AttemptState) -> None:
    """A ClaudeModel judge is dropped when the panel excludes the generator family
    ``"claude"``; the cross-family Ollama judges decide (EXTERNAL_VERIFIER, §10.2)."""
    qwen = OllamaModel("qwen3:32b", family="qwen", client=FakeOllamaClient("yes"))
    mistral = OllamaModel("mistral-small:24b", family="mistral", client=FakeOllamaClient("yes"))
    claude = ClaudeModel("claude-opus-4-8", client=FakeClaudeClient("no"))

    panel = JudgePanel(
        judges=[claude.as_judge(), qwen.as_judge(), mistral.as_judge()],
        reducer="majority",
        generator_family="claude",
    )

    # Eligibility: the claude judge is gone; two cross-family judges remain.
    eligible = panel.eligible_judges()
    assert len(eligible) == 2
    assert all(judge_family(j) != "claude" for j in eligible)

    result = asyncio.run(panel.score(attempt))
    assert result.metadata["excluded"] == ["claude"]
    assert result.metadata["eligible_count"] == 2
    # The two surviving cross-family judges both voted "yes" → majority "yes".
    assert result.value == "yes"


def test_panel_keeps_claude_when_generator_is_ollama(attempt: AttemptState) -> None:
    """When the generator is an Ollama family (``qwen``), the Claude judge is a valid
    cross-family member and is kept, while the same-family ``qwen`` judge is excluded —
    the symmetry of EXTERNAL_VERIFIER (§10.2): exclusion is keyed to the *generator's*
    family, whichever family that is."""
    claude = ClaudeModel("claude-opus-4-8", client=FakeClaudeClient("approve"))
    qwen = OllamaModel("qwen3:32b", family="qwen", client=FakeOllamaClient("approve"))
    mistral = OllamaModel("mistral-small:24b", family="mistral", client=FakeOllamaClient("approve"))

    panel = JudgePanel(
        judges=[claude.as_judge(), qwen.as_judge(), mistral.as_judge()],
        reducer="majority",
        generator_family="qwen",
    )

    eligible = panel.eligible_judges()
    # qwen (same family as the generator) is dropped; claude + mistral survive.
    assert len(eligible) == 2
    assert {judge_family(j) for j in eligible} == {"claude", "mistral"}
    result = asyncio.run(panel.score(attempt))
    assert result.metadata["excluded"] == ["qwen"]
    assert result.value == "approve"


# --------------------------------------------------------------------------- #
# judge() Score shape + sealed-boundary (never reads chrome)
# --------------------------------------------------------------------------- #


def test_judge_returns_tagged_score(attempt: AttemptState) -> None:
    model = OllamaModel("qwen3:32b", family="qwen", client=FakeOllamaClient("VALID"))
    score = asyncio.run(model.judge(attempt))
    assert isinstance(score, Score)
    assert score.value == "VALID"
    assert score.metadata["judge_family"] == "qwen"
    assert score.metadata["judge_model"] == "qwen3:32b"


def test_judge_does_not_read_chrome(attempt: AttemptState) -> None:
    """The judge messages are built from the scored context + output only — Tier-3
    chrome must never enter a model context (§10.1(e)). We assert the chrome content
    does not appear in what was sent to the client."""
    from ai_crucible.types import Chrome

    attempt.chrome = Chrome(rank=1, leaderboard=[{"name": "SECRET_RIVAL", "score": 999}])
    client = FakeOllamaClient("ok")
    model = OllamaModel("qwen3:32b", family="qwen", client=client)

    asyncio.run(model.judge(attempt))

    sent = str(client.calls[0]["messages"])
    assert "SECRET_RIVAL" not in sent


# --------------------------------------------------------------------------- #
# Served-vs-requested model provenance guard (models-cli-002, §11.6).
#
# Under load->judge->evict / OLLAMA_NUM_PARALLEL=1, a load timeout / alias drift /
# previously-loaded model answering makes Ollama return HTTP 200 with a DIFFERENT
# `model` field. The adapter must NOT silently stamp the JudgmentRecord with the
# requested model_id/family (that corrupts the measurement AND the EXTERNAL_VERIFIER
# family tag) — it raises ModelMismatchError so the kernel's andon can halt. A served
# tag that differs only by a trailing cloud suffix (the daemon serves `glm-5` for
# `glm-5:cloud`) is the SAME model and must NOT raise.
# --------------------------------------------------------------------------- #


class FakeOllamaServedClient:
    """Ollama ``/api/chat`` fake that echoes a ``model`` field (real-server shape).

    Ollama's response carries the *served* model at the top level. ``served`` defaults
    to whatever model was requested (faithful server); a test sets it to a DIFFERENT tag
    to drive the silent-fallback path the provenance guard must catch (models-cli-002).
    """

    def __init__(self, content: str = "ok", served: str | None = None) -> None:
        self.content = content
        self.served = served
        self.calls: list[dict[str, Any]] = []

    async def __call__(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(kwargs)
        served = self.served if self.served is not None else kwargs["model"]
        return {
            "model": served,
            "message": {"role": "assistant", "content": self.content},
        }


def test_ollama_served_mismatch_raises(attempt: AttemptState) -> None:
    """A served model != requested model raises ModelMismatchError (RED before the fix).

    The fake daemon is asked for ``qwen3:32b`` but answers with ``hermes3:8b`` (a real
    tier-timeout fallback). The adapter MUST refuse to attribute this judgment to the
    requested model — the provenance + same-family-exclusion seal (models-cli-002)."""
    client = FakeOllamaServedClient(content="contaminated", served="hermes3:8b")
    model = OllamaModel("qwen3:32b", family="qwen", client=client)

    with pytest.raises(ModelMismatchError) as exc:
        asyncio.run(model.generate(attempt))

    assert exc.value.requested == "qwen3:32b"
    assert exc.value.served == "hermes3:8b"


def test_ollama_judge_item_served_mismatch_raises() -> None:
    """The guard also fires on the characterization path: a mis-served judge_item never
    yields a JudgmentRecord stamped with the (wrong) requested model_id/family."""
    client = FakeOllamaServedClient(content="A", served="mistral-small:24b")
    model = OllamaModel("qwen3:32b", family="qwen", client=client)

    with pytest.raises(ModelMismatchError):
        asyncio.run(model.judge_item("Q: which? A or B"))


def test_ollama_served_matches_does_not_raise(attempt: AttemptState) -> None:
    """A faithful server (served == requested) flows through unchanged — the guard fails
    closed only on a POSITIVE mismatch, not on every response."""
    client = FakeOllamaServedClient(content="clean")  # echoes the requested model
    model = OllamaModel("qwen3:32b", family="qwen", client=client)

    out = asyncio.run(model.generate(attempt))
    assert out == "clean"


def test_ollama_cloud_suffix_is_not_a_mismatch(attempt: AttemptState) -> None:
    """The daemon serves a cloud tag WITHOUT its suffix (``glm-5`` for ``glm-5:cloud``,
    ``qwen3-coder:480b`` for ``qwen3-coder:480b-cloud``); that is the SAME model and must
    NOT raise — only a genuine fallback to a different model does."""
    for requested, served in (
        ("glm-5:cloud", "glm-5"),
        ("qwen3-coder:480b-cloud", "qwen3-coder:480b"),
        ("qwen3:32b", "qwen3:32b:latest"),
    ):
        client = FakeOllamaServedClient(content="fine", served=served)
        model = OllamaModel(requested, family="glm", client=client)
        # Must complete without raising.
        assert asyncio.run(model.generate(attempt)) == "fine"


def test_ollama_missing_served_field_tolerated(attempt: AttemptState) -> None:
    """An older server / minimal fake that omits the ``model`` field is tolerated — the
    guard fails closed only on a positive mismatch, never on absence of provenance."""
    client = FakeOllamaClient("no-model-field")  # base fake: message only, no `model`
    model = OllamaModel("qwen3:32b", family="qwen", client=client)

    assert asyncio.run(model.generate(attempt)) == "no-model-field"


def test_norm_model_strips_cloud_and_latest_suffixes() -> None:
    """The normalization shape (mirrors swarm/verify_findings.py._norm): trailing
    ``:latest`` and a trailing ``-cloud``/``:cloud`` collapse; a different model does not."""
    assert _norm_model("glm-5:cloud") == _norm_model("glm-5")
    assert _norm_model("qwen3-coder:480b-cloud") == _norm_model("qwen3-coder:480b")
    assert _norm_model("qwen3:32b:latest") == _norm_model("qwen3:32b")
    assert _norm_model("qwen3:32b") != _norm_model("hermes3:8b")
    assert _norm_model(None) == ""


class FakeClaudeServedClient:
    """Anthropic Messages fake that echoes a ``model`` field on the response.

    The real Messages API returns the model that answered in a top-level ``model`` field;
    ``served`` defaults to the requested model (faithful) or is set to a different id to
    drive the silent-route-to-a-different-model path the provenance guard must catch."""

    def __init__(self, content: str = "ok", served: str | None = None) -> None:
        self.content = content
        self.served = served
        self.calls: list[dict[str, Any]] = []
        self.messages = _FakeServedMessages(self)


class _FakeServedMessages:
    def __init__(self, parent: FakeClaudeServedClient) -> None:
        self._parent = parent

    async def create(self, **kwargs: Any) -> dict[str, Any]:
        self._parent.calls.append(kwargs)
        served = self._parent.served if self._parent.served is not None else kwargs["model"]
        return {
            "model": served,
            "content": [{"type": "text", "text": self._parent.content}],
        }


def test_claude_served_mismatch_raises(attempt: AttemptState) -> None:
    """Claude must surface a silent route to a different model too — the ``"claude"``
    family tag can never be attributed to a judgment a different model produced
    (models-cli-002, the shared family contract point C)."""
    client = FakeClaudeServedClient(content="x", served="claude-haiku-3-5")
    model = ClaudeModel("claude-opus-4-8", client=client)

    with pytest.raises(ModelMismatchError) as exc:
        asyncio.run(model.generate(attempt))

    assert exc.value.requested == "claude-opus-4-8"
    assert exc.value.served == "claude-haiku-3-5"


def test_claude_served_matches_does_not_raise(attempt: AttemptState) -> None:
    """A faithful Messages response (served == requested) flows through unchanged."""
    client = FakeClaudeServedClient(content="fine")  # echoes requested model
    model = ClaudeModel("claude-opus-4-8", client=client)

    assert asyncio.run(model.generate(attempt)) == "fine"


def test_claude_missing_served_field_tolerated(attempt: AttemptState) -> None:
    """The existing FakeClaudeClient omits ``model`` — tolerated, not a mismatch."""
    client = FakeClaudeClient("legacy-shape")  # no `model` key in its response
    model = ClaudeModel("claude-opus-4-8", client=client)

    assert asyncio.run(model.generate(attempt)) == "legacy-shape"


def test_claude_family_tag_is_correct() -> None:
    """ClaudeModel exposes a correct fixed family tag for EXTERNAL_VERIFIER exclusion —
    on the instance, the pin metadata, and the as_judge() callable (shared contract C)."""
    model = ClaudeModel("claude-opus-4-8", client=FakeClaudeClient())
    assert model.family == "claude"
    assert model.pin_metadata()["family"] == "claude"
    assert judge_family(model.as_judge()) == "claude"
