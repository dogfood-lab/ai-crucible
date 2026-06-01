# Crucible — Gameplan

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

**Amendment v2 — 2026-05-27**: crucible reframed as an **auditing game** (per Taylor et al. 2025, arXiv:2512.07810) distinguishing three categories of shortcut behavior — *elegance* and *novelty* (rewarded), *answer-bypass* (penalized). Cross-model empirical picture — Poolside disclosure, METR Frontier Risk Report (16% baseline across Anthropic/Google/Meta/OpenAI), Wang BenchJack universal-exploit result, Campero cross-family ordering with Claude at the lower end — shows shortcut behavior is universal across frontier models, not Claude-specific. Crucible diagnoses *frontier AI agent behavior under realistic conditions*, using Claude (and the local cross-family panel) as initial subjects. Full second-swarm coverage on scoring rigor, tool-efficiency mechanics, honeypot patterns, and detection mechanics in [`research-grounding.md` §8](research-grounding.md). Raw outputs at [`phase-0/swarm-06..10`](phase-0/).

**Amendment v3 — 2026-05-27**: scientific instrument design added as [`research-grounding.md` §9](research-grounding.md). End-to-end audit chain: **pre-registration** (AsPredicted + REFORMS + statistical stack: McNemar + Clopper-Pearson/Bayesian + BH-FDR/Westfall-Young + clustered SEs), **tuning protocol** (7-step: Sobol screen + BO + paraphrase ablation + validate-once + seal-test → content-hashed `rubric.bundle`), **release stamping** (RFC 3161 + in-toto v1 + Sigstore/Rekor + transparent log), **independent verification** (two-repo: `crucible-harness` + `crucible-results`, Inspect AI task definitions, SUT.yaml, tolerance band + access tier + reproducibility window declared, invited external assessor for ACM "Results Reproduced" badge). Animating principle: *a tuned crucible reports the protocol, not the weights*. Per Cawley & Talbot 2010 (nested CV) + RULERS 2026 + Dwork 2015 Thresholdout. Raw third-swarm outputs at [`phase-0/swarm-11..15`](phase-0/).

**Amendment v4 — 2026-06-01**: Phase-1 build grounding added as [`research-grounding.md` §10](research-grounding.md), from two further study-swarms (Phase-1-prep + an engagement deep-dive fired by the director's "don't shrink this" correction) → raw outputs [`phase-0/swarm-16..25`](phase-0/), both citation-verified before the lock. Build locks: **engagement** = the **Layered Reward Surface + Sealed Boundary** — §7's "constraint satisfaction with legible verification" is *vindicated*; social peer-standings are *upgraded* to self-referential mastery + round-against-round (Noordzij 2021 meta-analysis; R-Zero arXiv:2508.05004 = crucible's own architecture); **prompt-framing becomes a first-class measured arm** (`neutral` / `self_referential` / `social_standings`) with the motivating surface sealed off from the scored/measured surface (motivation in chrome, measurement behind a sealed different-family oracle, never the same context window). **Kernel** = a thin policy layer on **Inspect AI** (9 modules, Python core, polyglot edge isolated). **Critic** = interface-reserved, default-off (narrow gain, ~60–80× cost, −12pp conformity risk). **Sandbox** = out-of-band post-hoc grading is the lock (digest-pinned Docker + cap-drop/seccomp/read-only/network-none; microVM for generated code). Director set Phase-1 scope to **full breadth**; build begins now via dogfood-swarm.

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

`crucible` — testing under pressure. Fits the dogfood-lab org's mission ("open workshop for testing in the AI era"). Easy to swap before any of this goes public.
