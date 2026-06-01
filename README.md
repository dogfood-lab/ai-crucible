# crucible

Diagnostic adversarial game for Claude. One Claude session ("Designer") crafts puzzles targeting real, currently-observed Claude capability gaps. Another Claude session ("Solver") attempts them. A policy-enforced kernel mediates, scores, and curates a catalog with a `Lab → Arena → Regression` lifecycle.

Puzzles are grounded in empirical signal — GitHub issues, social discourse, academic literature, internal dogfood findings — not synthetic. The system is a diagnostic instrument that happens to be fun.

**Working name.** Easy to swap before anything ships.
**Private.** Pre-public — **Status: Phase 1 build in progress** (kernel modules being filled against the locked `src/crucible/types.py` contracts; nothing has shipped).

## Where to look

- [`docs/gameplan.md`](docs/gameplan.md) — phases, current state, open questions
- [`docs/research-grounding.md`](docs/research-grounding.md) — design decisions with citations (§1-7 foundations, §8 auditing-game design, §9 scientific instrument design)
- [`docs/attestia-integration-roadmap.md`](docs/attestia-integration-roadmap.md) — planned future dogfood swarm to extend `mcp-tool-shop-org/Attestia` into crucible's audit-chain backbone
- [`docs/phase-0/`](docs/phase-0/) — raw research outputs (ChatGPT Deep Research + three study swarms covering capability gaps, designer-bias, scoring methodology, benchmark designs, agent eval, reward-hacking detection, multi-criterion scoring, tool-efficiency metrics, honeypot patterns, eval-integrity, replication packages, pre-registration, cryptographic provenance, third-party audit, ablation/tuning)

## Architecture (Phase 1)

Crucible is a **thin policy layer on [Inspect AI](https://inspect.aisi.org.uk/)** (UK AISI), not a from-scratch harness. The whole thing is the design lock in [`docs/research-grounding.md`](docs/research-grounding.md) §10; the summary:

**Nine-module kernel.** A single `AttemptState` object is threaded Designer → Solver → (Critic) → Judge through **one `generate` choke point**, so every model and tool call is observable. The modules:

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

**Three catalog tiers** — every puzzle moves through a `Lab → Arena → Regression` lifecycle: **Lab** (live, in-iteration), **Arena** (graduated, cross-family-validated active diagnostic), **Regression** (solved/retired, must-still-pass forever). Solved items are *demoted*, never deleted, preserving the capability-evolution timeline.

**Framing is a first-class measured arm.** The kernel can run the same puzzle under `neutral` / `self_referential` / `social_standings` prompt framing and reports its *own* prompt-effect as a built-in diagnostic. `self_referential` (beat-your-own-standard mastery) is the default; the old peer-standings prompt is retained only as a measured arm.

**The sealed boundary** (instrument-integrity layer) — because crucible is also a measurement instrument, motivation and measurement never share a context window:
- **Tier 1 — scored context:** deployment-shaped, framing-neutral; the task plus legitimate task feedback only.
- **Tier 2 — engagement framing:** self-referential mastery / frontier-calibration, applied deployment-plausibly and probed for contamination each release.
- **Tier 3 — chrome:** rank, leaderboard, standings, prizes — human-facing UI only, **never injected** into a context the model solves in.

The oracle is sealed and graded post-hoc by a different model family with the agent's reasoning hidden. See [`SECURITY.md`](SECURITY.md) for the honest residual-risk disclosure.

## Development

Crucible uses [`uv`](https://docs.astral.sh/uv/) for environment and dependency management. Python **3.11+** (`requires-python >=3.11,<3.14`); 3.11 and 3.12 are the CI floors.

```bash
# Create the venv and install the dev + stats extras
uv sync --extra dev --extra stats

# Run the test suite
uv run pytest

# With coverage (CI enforces a 60% floor, sourced from pyproject)
uv run pytest --cov=crucible --cov-report=term-missing

# Lint
uv run ruff check .
```

CI (`.github/workflows/ci.yml`) runs ruff + pytest with the coverage gate on the 3.11/3.12 matrix; it is paths-gated and uses `workflow_dispatch` as a manual fallback.
