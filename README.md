<p align="center">
  <a href="README.ja.md">日本語</a> | <a href="README.zh.md">中文</a> | <a href="README.es.md">Español</a> | <a href="README.fr.md">Français</a> | <a href="README.hi.md">हिन्दी</a> | <a href="README.it.md">Italiano</a> | <a href="README.pt-BR.md">Português (BR)</a>
</p>

<p align="center">
  <img src="https://raw.githubusercontent.com/dogfood-lab/ai-crucible/main/assets/logo.png" alt="ai-crucible" width="500" />
</p>

<p align="center">
  <a href="https://github.com/dogfood-lab/ai-crucible/actions/workflows/ci.yml"><img src="https://github.com/dogfood-lab/ai-crucible/actions/workflows/ci.yml/badge.svg" alt="CI" /></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="MIT License" /></a>
  <img src="https://img.shields.io/badge/python-3.11%E2%80%933.13-blue.svg" alt="Python 3.11–3.13" />
  <img src="https://img.shields.io/badge/coverage-93%25-brightgreen.svg" alt="Coverage 93%" />
  <a href="CHANGELOG.md"><img src="https://img.shields.io/badge/version-0.2.0-orange.svg" alt="Version 0.2.0" /></a>
  <a href="https://dogfood-lab.github.io/ai-crucible/"><img src="https://img.shields.io/badge/docs-handbook-orange.svg" alt="Handbook" /></a>
</p>

<p align="center"><b>A diagnostic adversarial game for frontier LLMs — a measurement instrument that happens to be fun.</b></p>

One Claude session (**Designer**) crafts puzzles targeting real, currently-observed capability gaps. Another (**Solver**) attempts them. A policy-enforced kernel mediates, scores against a hidden oracle, and curates a catalog through a `Lab → Arena → Regression` lifecycle. Puzzles are grounded in empirical signal — real GitHub issues, academic literature, observed failures in the field — not synthetic.

## What makes it different

- **Capability, not "cheating."** AI Crucible distinguishes *elegance* and *novelty* (rewarded) from *answer-bypass* (penalized). Lateral thinking is a capability to measure, not a vice to punish.
- **The instrument measures itself.** Prompt framing is a first-class measured arm — the kernel runs the same puzzle under `neutral` / `self_referential` / `social_standings` framings and reports its own prompt-effect as a diagnostic.
- **A sealed measurement boundary.** Motivation and measurement never share a context window; the hidden oracle is graded out-of-band by a different model family with the agent's reasoning hidden. The model can't game what it can't perceive.
- **Reliability by consistency.** `pass^k` (all *k* independent trials succeed), Wilson intervals, and cross-family judge panels — built to report distributions, not point estimates.
- **A living catalog.** Solved puzzles are *demoted* to `Regression`, never deleted, so the catalog becomes a capability-evolution timeline as the frontier moves.

## Threat model (summary)

AI Crucible's primary adversary is the **model under measurement**, which may try to reach the answer key or to detect and game the eval. It **touches**: puzzle prompts/setup, the Solver's own confined working directory, and the model-provider API calls the operator configures. It does **not** touch: the oracle/answer key (graded out-of-band on a separate host the Solver cannot reach) or motivation "chrome" (rank/standings — never injected into the scored context). **Permissions:** model-provider keys via environment variables at runtime; no bundled secrets, no telemetry, no outbound calls of its own. Full disclosure — including where a boundary is *defense-in-depth* rather than a hard guarantee — is in **[SECURITY.md](SECURITY.md)**.

## Architecture

AI Crucible is a **thin policy layer on [Inspect AI](https://inspect.aisi.org.uk/)** (UK AISI), not a from-scratch harness. A single `AttemptState` object is threaded Designer → Solver → (Critic) → Judge through **one `generate` choke point**, so every model and tool call is observable.

| Module | Responsibility |
| ------ | -------------- |
| `puzzle_loader` | Loads a puzzle directory (`meta.json` / `prompt` / `setup_script`) into Solver-visible state. **Never touches the oracle.** |
| `sandbox` | Narrow `exec` / `read_file` / `write_file` channel into a locked, network-less container. |
| `roles` | The five role slots (Designer / Solver / Critic / Judge / CohortSolver). Only Solver gets tools; Critic is interface-reserved, default-off. |
| `budget_governor` | Per-class tool-call + wall-clock budgets, displayed to the agent, enforced kernel-side; hard-kill on pathological loops. |
| `oracle_scorer` | Out-of-band grading: solved-**and**-no-regression against the hidden oracle (SWE-bench pattern). |
| `judge_panel` | Cross-family panel of model-scorers + reducer (PoLL) for novelty validation and bypass detection. |
| `trace_writer` | Per-attempt transcript in the Inspect `EvalLog` shape; large blobs stored by digest. |
| `observability` | Per-attempt → per-puzzle → per-model rollups; `pass^k` native. |
| `attestation` | Cryptographic provenance (cosign + event-store) behind a typed subprocess boundary. |

The sealed boundary runs in three tiers — **Tier 1** scored context (deployment-shaped, framing-neutral), **Tier 2** engagement framing (probed for contamination each release), **Tier 3** chrome (rank/leaderboard — human-facing UI only, never in a context the model solves in). The full design rationale, with citations, is in [`docs/research-grounding.md`](docs/research-grounding.md).

## Install

```bash
# As a Python library + CLI (PyPI):
pip install ai-crucible          # or: uv pip install ai-crucible
ai-crucible --help

# Or zero-prerequisite via npx — downloads a verified binary, no Python needed:
npx @dogfood-lab/ai-crucible --help
```

**Run one diagnostic cycle** — a Solver attempts a puzzle in the sandbox, graded out-of-band against the sealed oracle, emitting the `pass^k` / Wilson rollup:

```bash
# @family selects the adapter: no tag / @claude -> Claude (ANTHROPIC_API_KEY);
# any other @family -> a local Ollama model of that family.
ai-crucible run puzzles/seed-sulzbach-55252 --model claude-opus-4-8@claude --k 5
```

> **Research preview (v0.2.x).** The judge panel's alt-test ω is still a *circular model-jury bootstrap*: validating it needs a round of **≥3 independent human annotators** (the [alt-test](https://arxiv.org/abs/2501.10970)), which a single-human studio cannot staff — so that round is **on ice by structural constraint, not neglect**. Seated judges stay **provisional**, the composed panel **escalates to a Claude Designer** below quorum, and the instrument discloses this rather than faking human grounding. See the [scorecard](SCORECARD.md) for the honest, non-cosmetic gate results.

## Quick start (from source)

AI Crucible uses [`uv`](https://docs.astral.sh/uv/) for environment and dependency management. Python **3.11+**.

```bash
# Create the venv and install the dev + stats extras
uv sync --extra dev --extra stats

# Run the test suite (with the coverage gate)
uv run pytest --cov=ai_crucible --cov-report=term-missing

# Lint
uv run ruff check .

# One command: lint + tests + build + smoke
bash verify.sh
```

## Documentation

- **[Handbook](https://dogfood-lab.github.io/ai-crucible/)** — guides, architecture, and reference.
- [`docs/research-grounding.md`](docs/research-grounding.md) — design rationale, with citations.
- [`docs/gameplan.md`](docs/gameplan.md) — roadmap and open questions.
- [`SECURITY.md`](SECURITY.md) — threat model + honest residual-risk disclosure.

## License

[MIT](LICENSE). Public and pre-1.0 — see the [CHANGELOG](CHANGELOG.md) for version status.

---

<p align="center"><sub>Built by <a href="https://mcp-tool-shop.github.io/">MCP Tool Shop</a> · part of the <a href="https://github.com/dogfood-lab">dogfood-lab</a> workshop for testing in the AI era.</sub></p>
