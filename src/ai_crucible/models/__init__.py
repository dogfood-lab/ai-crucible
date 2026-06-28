"""Model adapters — the Phase-2 bridge from Phase-1's injected stubs to real models.

The Phase-1 kernel takes every model-shaped dependency as an *injected callable*
(``generate``, ``judges``, ``oracle_runner``) so its policy logic is testable without
a model runtime (research-grounding §10.2, §11.6). This package supplies the concrete
backings for those slots:

* :class:`~ai_crucible.models.ollama_adapter.OllamaModel` — a local model served by
  Ollama on the RTX 5090 (the cross-family panel members, §8.6 / §11.2).
* :class:`~ai_crucible.models.claude_adapter.ClaudeModel` — Claude via the Anthropic
  Messages API (the Designer/Solver, and the Claude-family anchor the panel excludes
  when Claude is the generator — EXTERNAL_VERIFIER, §10.2).

Both expose the same surface — ``complete`` / ``generate`` (kernel choke point) /
``judge`` + ``as_judge`` (panel, ``.family``-tagged) / ``judge_item`` (a
:class:`~ai_crucible.characterize.types.JudgmentRecord` for the §11.1 admission test) —
so they are interchangeable in the kernel and on the panel. Both keep the model client
injectable for mocking and pin deterministic request options (PIN_PER_STEP, §11.2).
"""

from __future__ import annotations

from ai_crucible.models.claude_adapter import ClaudeModel
from ai_crucible.models.ollama_adapter import OllamaModel
from ai_crucible.models.openrouter_adapter import OpenRouterModel, is_openrouter_spec

__all__ = ["ClaudeModel", "OllamaModel", "OpenRouterModel", "is_openrouter_spec"]
