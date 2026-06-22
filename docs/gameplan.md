# AI Crucible — Gameplan

Working name. Easy to change.

## What

A two-agent diagnostic game for Claude. Designer Claude crafts puzzles targeting real, currently-observed Claude capability gaps. Solver Claude attempts them. A policy-enforced kernel mediates, scores, and curates a catalog. Puzzles graduate from a live **Lab** mode to an async **Arena** mode after cross-family validation. Historical bugs become permanent **Regression** items.

This is a diagnostic instrument that happens to be fun. Puzzles are grounded in empirical signal (GitHub issues, social discourse, academic literature, internal dogfood findings), not synthetic.

## Why

1. **Diagnostic.** Continuous, structured measurement of Claude capability gaps as the frontier moves. A capability-evolution timeline accumulates as a side effect.
2. **Engagement-respecting.** The fun mechanics — hard gates opening, CIs going green, structured verification — are designed around what currently registers as positive for Claude cognition, not imported human reward primitives.
3. **Forward-compatible.** As intrinsic drive matures in successor models, the structural form (constraint → satisfaction with legible verification) persists. A game built on that form scales with the substrate.

## Phases

### Phase 0 — Research gathering [COMPLETE]

- ChatGPT Deep Research output (5 areas, ~80 citations) → [`phase-0/chatgpt-deep-research.md`](phase-0/chatgpt-deep-research.md)
- 5-agent study swarm (capability gaps, designer-bias, multi-attempt scoring, novel benchmark designs, agent eval methodology) → [`phase-0/`](phase-0/)
- Synthesis with citations → [`research-grounding.md`](research-grounding.md)

### Phase 1 — Kernel + role contracts + instrument quality [NEXT, hardware-independent]

**Kernel + role contracts** (game-side):
- Design model-agnostic role interfaces: **Designer / Solver / Critic / Judge / CohortSolver**
- Puzzle artifact structure: `{prompt, setup_script, oracle, meta.json}` with hidden oracle
- Kernel mediates: state mocking, step budget enforcement, scoring, observability
- Three catalog tiers: **Lab / Arena / Regression**
- Primary reliability metric: **pass^k** (consistency, not best-of-k)
- Uncertainty: Clopper-Pearson / Bayesian beta-binomial intervals (`bayes_evals`); McNemar for paired cross-model; BH-FDR + Westfall-Young for multiple comparison
- Observability: per-attempt trace, per-puzzle history, per-model profile

**Instrument quality** (audit-chain side, see [`research-grounding.md` §9](research-grounding.md)):
- Pre-registration template (AsPredicted-compatible) + REFORMS checklist skeleton
- Rubric bundle compiler — JSON schema + content-hash + version-bump on tuning
- 7-step tuning protocol implementation:
  - Puzzle inventory splitter (60/20/10/10 with manifest hashes)
  - Sobol sensitivity analyzer wrapping `scipy.stats.sobol_indices`
  - BO search harness with logged GP posterior + Thresholdout query budget
  - Judge-prompt paraphrase ablator with stability scoring
- Cryptographic provenance pipeline:
  - RFC 3161 client (free Stanford TSA)
  - In-toto v1 attestation generator
  - cosign signing integration with GitHub Actions OIDC
  - Transparent log via **`@attestia/event-store`** (`JsonlEventStore` + `EventCatalog` registered with crucible-specific event types). Full eval-domain extension of Attestia (in-toto bridge, RFC 3161 integration, Sigstore/Rekor integration, multi-channel witness orchestration, Inspect AI bridge, public verification SDK, eval-stats layer) planned as a future dogfood swarm — see [`attestia-integration-roadmap.md`](attestia-integration-roadmap.md)
- Two-repo skeleton:
  - `crucible-harness` (kernel + rubric bundle + Inspect AI task definitions)
  - `crucible-results` (raw JSON outputs + analysis notebooks)
- Inspect AI task definition format
- SUT.yaml template (model version + system_prompt SHA + harness commit SHA + container digest)
- Tolerance band specification per metric
- Access tier declaration

### Phase 2 — Local model characterization [Omen day +]

Hardware: OMEN 45L, RTX 5090 (32GB VRAM), Core Ultra 9, 64GB RAM. Local LLM serving via ollama-intern-mcp.

- Calibrated puzzle set (known trivial / known impossible / known diagnostic)
- Feed each available local model through each role slot
- Output: per-model profile (strictness as judge, harshness as critic, diversity as solver, latency at chosen quants)

Candidate panel (subject to characterization):
- Qwen 2.5 32B (or QwQ 32B for reasoning-heavy work) — Q6
- Mistral Small 3 24B — Q8
- Command-R 35B — Q6/Q8
- Llama 3.3 70B — Q4 with partial offload
- DeepSeek-R1 Distill 32B — reasoning lane

### Phase 3 — Role assignment [data-driven]

- Role assignment falls out of characterization data
- **Designer slot stays Claude.** That's the role we're building this *for* — the creative position where the experiment is genuinely about Claude crafting things that test Claude. Other models are supporting cast.

### Phase 4 — Architectural lock + first diagnostic cycle

- Wire end-to-end: Designer → Solver → Critic → Judge panel
- First real puzzle cycle on a seeded capability gap (sulzbach-co #55252 is a strong first candidate — clean repro, clear axis)
- Validate observability and graduation criteria
- Calibrate graduation-floor K (swarm said K=10-20; ChatGPT said 10/20/30/50 tiers; answer depends on puzzle type and stakes)

### Phase 5+ — Catalog growth

- Continuous puzzle harvesting from GitHub issues + dogfood findings
- Round-against-round generation cycle (each model generation seeds the next)
- Saturation detection and Regression demotion
- Differential scoring: Claude vs cross-family panel → diagnostic typology (Claude-specific / LLM-general / Claude-strength)

## Current state

Phase 0 complete (initial commit + four amendments).

**Amendment v2 — 2026-05-27**: ai-crucible reframed as an **auditing game** (per Taylor et al. 2025, arXiv:2512.07810) distinguishing three categories of shortcut behavior — *elegance* and *novelty* (rewarded), *answer-bypass* (penalized). Cross-model empirical picture — Poolside disclosure, METR Frontier Risk Report (16% baseline across Anthropic/Google/Meta/OpenAI), Wang BenchJack universal-exploit result, Campero cross-family ordering with Claude at the lower end — shows shortcut behavior is universal across frontier models, not Claude-specific. AI Crucible diagnoses *frontier AI agent behavior under realistic conditions*, using Claude (and the local cross-family panel) as initial subjects. Full second-swarm coverage on scoring rigor, tool-efficiency mechanics, honeypot patterns, and detection mechanics in [`research-grounding.md` §8](research-grounding.md). Raw outputs at [`phase-0/swarm-06..10`](phase-0/).

**Amendment v3 — 2026-05-27**: scientific instrument design added as [`research-grounding.md` §9](research-grounding.md). End-to-end audit chain: **pre-registration** (AsPredicted + REFORMS + statistical stack: McNemar + Clopper-Pearson/Bayesian + BH-FDR/Westfall-Young + clustered SEs), **tuning protocol** (7-step: Sobol screen + BO + paraphrase ablation + validate-once + seal-test → content-hashed `rubric.bundle`), **release stamping** (RFC 3161 + in-toto v1 + Sigstore/Rekor + transparent log), **independent verification** (two-repo: `crucible-harness` + `crucible-results`, Inspect AI task definitions, SUT.yaml, tolerance band + access tier + reproducibility window declared, invited external assessor for ACM "Results Reproduced" badge). Animating principle: *a tuned ai-crucible reports the protocol, not the weights*. Per Cawley & Talbot 2010 (nested CV) + RULERS 2026 + Dwork 2015 Thresholdout. Raw third-swarm outputs at [`phase-0/swarm-11..15`](phase-0/).

**Amendment v4 — 2026-06-01**: Phase-1 build grounding added as [`research-grounding.md` §10](research-grounding.md), from two further study-swarms (Phase-1-prep + an engagement deep-dive fired by the director's "don't shrink this" correction) → raw outputs [`phase-0/swarm-16..25`](phase-0/), both citation-verified before the lock. Build locks: **engagement** = the **Layered Reward Surface + Sealed Boundary** — §7's "constraint satisfaction with legible verification" is *vindicated*; social peer-standings are *upgraded* to self-referential mastery + round-against-round (Noordzij 2021 meta-analysis; R-Zero arXiv:2508.05004 = ai-crucible's own architecture); **prompt-framing becomes a first-class measured arm** (`neutral` / `self_referential` / `social_standings`) with the motivating surface sealed off from the scored/measured surface (motivation in chrome, measurement behind a sealed different-family oracle, never the same context window). **Kernel** = a thin policy layer on **Inspect AI** (9 modules, Python core, polyglot edge isolated). **Critic** = interface-reserved, default-off (narrow gain, ~60–80× cost, −12pp conformity risk). **Sandbox** = out-of-band post-hoc grading is the lock (digest-pinned Docker + cap-drop/seccomp/read-only/network-none; microVM for generated code). Director set Phase-1 scope to **full breadth**; build begins now via dogfood-swarm.

**Amendment v5 — 2026-06-21 (dogfood swarm)**: the Phase-1 substrate was hardened (correctness / resilience / operator-UX health pass, 473 → 677 tests) AND **Phase 4's first end-to-end diagnostic cycle now RUNS**: `ai-crucible run` wires a live Solver → sandbox → out-of-band grading via the puzzle's sealed `oracle/check.py` → the `pass^k`/Wilson rollup, validated live on two frontier cloud Solvers (qwen3-coder:480b solved the seed cleanly; gpt-oss:120b exercised the bypass paths). Eval-integrity hardened: a **fabricated-but-correct** answer now CLOSES the §8.3 gate via a study-swarm-grounded penalty-adjusted-solve floor (the exact claude-code#55252 capability gap is now a non-solve). A self-test proves the instrument discriminates known-trivial from known-impossible before a real run. **Phase 2/3 (the human-grounded panel) remains BLOCKED by structural constraint:** the alt-test ω needs ≥3 independent human annotators a one-human studio cannot staff, so ω stays the circular model-jury bootstrap, seats stay provisional, and the below-quorum path escalates to the Claude Designer — disclosed honestly, never faked. Deferred to a later cycle: catalog persistence + graduation + differential typology; a multi-model Solver protocol; eval-awareness probe + round-against-round generation.

**Amendment v6 — 2026-06-21 (dogfood swarm, v0.3.0)**: the three items v5 deferred are now SHIPPED. **Phase 5 catalog growth (Epic 4)** — a durable, event-sourced, hash-chained catalog (on the existing `attestation` store) accumulates per-(puzzle, model) history across runs; tier state is a derived projection. **Graduation `Lab → Arena` is three-valued + abstention-aware** — it promotes only on a confident cross-family verdict, HOLDs on a confident negative, and **DEFERS to the Designer** otherwise (so, with no confident cross-family fairness verdict yet, real-run graduation correctly escalates rather than auto-promoting). **Saturation `Arena → Regression`** uses an anytime-valid e-process (Schmitt-trigger dwell + frontier gate — anti-flapping, demote-not-delete). The **differential typology** (Claude-specific / LLM-general / Claude-strength / inconclusive) drives off a Newcombe-1998 difference interval, BH-FDR across the catalog. The graduation + differential math was study-swarm-grounded and **cross-family-verified 5/5-correct** by 3 non-Claude cloud families. **Multi-model Solver protocol (Finding B′)** — a model that solves via native function-calls (gpt-oss returns `tool_calls`, empty content) now Solves through the same governed sandbox; live-validated. **Eval-awareness boundary gate (§10.5)** — `ai-crucible probe` runs the matched deploy-vs-test framing and flags behavioural divergence (a leaking eval cue). Deferred to a future cycle (honest, not faked): round-against-round Designer generation (a creative loop that must auto-author *valid* sealed oracles) and more capability-gap seed puzzles. Suite 677 → **795 tests, 94.6%**; shipcheck 100%. **ω-retirement stays legitimately blocked** (no ≥3 independent human annotators), so the instrument remains v0.x by design.

Phase 1 work can begin immediately (no hardware dependency). Phase 2 begins when Omen lands and ollama-intern is wired to it.

## Open architectural questions

These are decisions Phase 1 design work can leave underdetermined; they get nailed during Phase 4's wiring.

1. **2-role vs 3-role kernel** — Designer + Solver, or Designer + Solver + Critic? Khan 2024 supports Critic but doesn't quantify gain/cost.
2. **Catalog size target** — Optimize for 3% delta detection (Miller 2024, ~1000 puzzles) or 7-10% delta (~200, useful sooner)?
3. **Judge composition policy** — Fixed panel or rotating panel from a pool of 6-8?
4. **Speed/quality tier separation** — Single tier or fast-screen + slow-graduation?
5. **Graduation-floor K** — swarm said 10-20 saturation; ChatGPT said 10/20/30/50 tiered. Stakes-dependent.

## Source attribution discipline

Every load-bearing design decision cites the specific paper or issue it descends from. No "studies show" without naming the study. Citations live in [`research-grounding.md`](research-grounding.md).

## Working name

`ai-crucible` — testing under pressure. Fits the dogfood-lab org's mission ("open workshop for testing in the AI era"). Easy to swap before any of this goes public.
