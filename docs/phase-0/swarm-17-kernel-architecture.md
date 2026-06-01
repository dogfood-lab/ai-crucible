# Swarm Agent 17 — Multi-Agent Eval-Harness Kernel Architecture

**Date:** 2026-06-01
**Source:** Fourth study swarm dispatch (Phase-1-prep, research-grounded advisor protocol)
**Question:** How do production agentic-eval harnesses architect the mediating kernel (state, budget enforcement, scoring, observability, multi-role orchestration)? Ground crucible's Phase 1 kernel implementation — module boundaries, role-interface contract, trace schema, primary stack — in proven harness architecture rather than inventing it.

---

# Crucible Kernel Architecture: Findings Brief

How production agentic-eval harnesses architect the mediating kernel. Each finding carries a concrete kernel implication.

**1. Inspect AI — Task = (Dataset, Solver-chain, Scorer) with `TaskState` as the single mutable bus.** ([docs](https://inspect.aisi.org.uk/), [solvers](https://inspect.aisi.org.uk/solvers.html), 2025) A solver is `async def solve(state: TaskState, generate: Generate) -> TaskState`; solvers compose via `chain(...)`. `TaskState` carries `messages, output, input, metadata, tools, target, sample_id`. The `generate` closure calls the model, appends the assistant message, sets output — the *only* sanctioned model-touch point. **Kernel implication:** make Crucible's per-attempt object a single `AttemptState` dataclass threaded through Designer→Solver→Critic→Judge; the only function permitted to mutate `state.output` or call a model is an injected `generate`/`step` closure, so all model I/O funnels through one observable choke point.

**2. Inspect Scorer = `async def score(state, target) -> Score`, `Score(value, answer, explanation, metadata)`.** ([scorers](https://inspect.aisi.org.uk/scorers.html), 2025) `model_graded_qa()` fills a `{question}/{answer}/{criterion}/{instructions}` template, regex-extracts `GRADE: C/I`, and **supports multiple grader models with majority voting**. Multiple scorers run as a list (independent values) or a multi-scorer with a **reducer** (e.g. majority vote). **Kernel implication:** the hidden-oracle check is a `Scorer` returning `Score(value, explanation, metadata)`; the Judge-panel is N model-scorers + a reducer module — do not hand-roll panel aggregation.

**3. Inspect limits are stackable context managers, not loop counters.** ([util ref](https://inspect.aisi.org.uk/reference/inspect_ai.util.html), [errors-and-limits](https://inspect.aisi.org.uk/errors-and-limits.html), 2025) `message_limit()`, `token_limit()`, `time_limit()`, `working_limit()` (wall-clock minus rate-limit waiting), `cost_limit()`. They wrap *either* a whole sample *or* a scoped block/agent; breach raises `LimitExceededError` **at the level the manager was opened**. **Kernel implication:** enforce Crucible's per-attempt step/tool budget with a `with step_limit(n), token_limit(t):` wrapper around the Solver block only (not Designer/Judge), and catch the breach at the attempt boundary — emit a `terminated_by: budget` trace field rather than silently truncating.

**4. Inspect agents: narrow `AgentState`, composed via `handoff()` / `as_tool()` / `as_solver()`.** ([agents](https://inspect.aisi.org.uk/agents.html), 2025) One harness orchestrates multiple agents: `handoff()` forwards full conversation history to a sub-agent; `as_tool()` gives a string-in/string-out boundary. **Kernel implication:** model Designer, Solver, Critic as three agents behind a uniform `Role` interface; Solver↔tools use the tool boundary (sandboxed), Critic uses `as_tool` (no env access). This gives the "multiple roles in one harness" contract without bespoke plumbing.

**5. Inspect EvalLog schema — the trace contract downstream stats depend on.** ([eval-logs](https://inspect.aisi.org.uk/eval-logs.html), format v2, 2025) Top-level: `eval` (spec: task/model/time), `plan`, `results` (metrics), `stats` (token usage), `samples[]`. Per-`EvalSample`: `id, input, target, output, scores{}, messages[], events[]` (full transcript), `metadata`, `model_usage`, `total_time`, `error`. Events de-dup large content into **attachments**. **Kernel implication:** adopt this exact envelope for per-attempt records — `attempt_id, puzzle_id, model, scores{oracle, panel}, events[], usage{tokens,cost}, wall_time, terminated_by, error`; store blobs as attachments so the `@attestia/event-store` stream stays compact and replayable.

**6. τ-bench — LLM-simulated *user* + tool API + DB-state reward, capped by `max_num_steps`.** ([arXiv 2406.12045](https://arxiv.org/abs/2406.12045); [sierra-research/tau-bench](https://github.com/sierra-research/tau-bench), 2024) Reward compares **final database state** against ground-truth action sequences (plus output checks); `pass^k` measures consistency across k i.i.d. trials; turn-based loop with a hard step cap. **Kernel implication:** Crucible's oracle scores **terminal world-state** (mocked/cached external state), not chat text; record `pass^k` natively by running k attempts per puzzle and storing each as a sibling attempt under one puzzle-history record.

**7. SWE-bench / terminal-bench — per-instance container, gold patch hidden, verdict from a test oracle.** ([SWE-bench](https://github.com/SWE-bench/SWE-bench), ICLR 2024; [terminal-bench](https://github.com/laude-institute/terminal-bench), 2025) Instance = `{instance_id, repo, base_commit, patch(gold), test_patch, FAIL_TO_PASS, PASS_TO_PASS, problem_statement, environment_setup_commit, version}`. Harness builds a Docker env, applies the **separately-supplied `model_patch`** (prediction file: `instance_id, model_name_or_path, model_patch`), runs tests; **resolved iff all FAIL_TO_PASS now pass AND all PASS_TO_PASS still pass** (no regression). **Kernel implication:** Crucible's puzzle dir mirrors this — `prompt, setup_script, oracle, meta.json`; the gold/oracle is *never* in the Solver's context (a separate prediction object is collected), and the scorer requires both a positive (solved) and a negative (no-regression) condition, encoded in `meta.json`.

**8. METR Vivaria — agent↔server via hooks; every LLM call + action/observation is a trace entry in Postgres.** ([METR/vivaria](https://github.com/METR/vivaria), 2024) `pyhooks` is the *only* path an agent uses to call LLM APIs and record trace entries; runs use the METR Task Standard (`TaskFamily`); UI annotates traces. Budget/scoring live server-side, not in the agent. **Kernel implication:** keep budget accounting and trace-writing **kernel-side**, never trusting the Solver to self-report; every model call and tool call is appended as a structured event by the kernel, giving audit-ready transcripts by construction.

**9. LLM-as-judge needs a *panel*, not a single judge — biases are documented.** (MT-bench, [arXiv 2306.05685](https://arxiv.org/abs/2306.05685), NeurIPS 2023; "Replacing Judges with Juries / PoLL", [arXiv 2404.18796](https://arxiv.org/abs/2404.18796), 2024) Single GPT-4 judges show position, verbosity, and self-enhancement bias; a Panel-of-LLM-evaluators (PoLL) across model *families* is cheaper and less biased. **Kernel implication:** Crucible's Judge-panel must draw from ≥2 model families, randomize answer position per judge, and aggregate by majority/median reducer — and the *generator's* judge must differ from the Solver's family (matches the `EXTERNAL_VERIFIER` standard).

---

## Recommended kernel skeleton

Primary stack: **Python** — Inspect AI, oracle scorers, and `scipy.stats.sobol_indices` are all Python; build Crucible as a thin policy layer *on top of* Inspect's Task/Solver/Scorer/limits primitives rather than reimplementing them.

| Module | Responsibility |
|---|---|
| `puzzle_loader` | Parse `{prompt, setup_script, oracle, meta.json}`; hold oracle out of any Solver-visible state. |
| `sandbox` | Mock/cache external state; run `setup_script`; reset world per attempt (Vivaria/SWE-bench model). |
| `roles` (Designer/Solver/Critic) | Uniform `Role` interface behind `as_tool`/`handoff`; only `Solver` gets sandbox tools. |
| `budget_governor` | Scoped `step_limit/token_limit/time_limit` context managers around the Solver block; raises at attempt boundary, sets `terminated_by`. |
| `oracle_scorer` | `Scorer` comparing terminal world-state to hidden oracle; emits `Score(value, explanation, metadata)`; requires solved-AND-no-regression. |
| `judge_panel` | N cross-family model-scorers + reducer (majority/median); position-randomized; generator-family excluded. |
| `trace_writer` | Append every model/tool call as a structured event; Inspect EvalLog-shaped record; blobs → attachments. |
| `observability` | Roll attempts → per-puzzle history → per-model profile; `pass^k` aggregation; the stats handoff surface. |
| `attestation` (polyglot edge) | Sign each sealed trace via the **cosign** Go binary; stream events to **`@attestia/event-store`** (Node) over a stable JSON envelope. |

**Polyglot reality:** keep the kernel core Python; isolate the two non-Python deps behind the `attestation` module as subprocess (cosign) + a typed JSON contract (`@attestia/event-store`), so the language boundary is one module, not a cross-cutting concern.

Sources: [Inspect AI docs](https://inspect.aisi.org.uk/) · [solvers](https://inspect.aisi.org.uk/solvers.html) · [scorers](https://inspect.aisi.org.uk/scorers.html) · [agents](https://inspect.aisi.org.uk/agents.html) · [eval-logs](https://inspect.aisi.org.uk/eval-logs.html) · [util/limits](https://inspect.aisi.org.uk/reference/inspect_ai.util.html) · [τ-bench arXiv 2406.12045](https://arxiv.org/abs/2406.12045) / [repo](https://github.com/sierra-research/tau-bench) · [SWE-bench](https://github.com/SWE-bench/SWE-bench) · [terminal-bench](https://github.com/laude-institute/terminal-bench) · [METR Vivaria](https://github.com/METR/vivaria) · [OpenAI Evals](https://github.com/openai/evals) · [MT-bench arXiv 2306.05685](https://arxiv.org/abs/2306.05685) · [PoLL arXiv 2404.18796](https://arxiv.org/abs/2404.18796)

---

## ⚠ Verification note (second pass — 2026-06-01)

This brief's claims rest mostly on **official Inspect AI documentation** (TaskState/Solver/Scorer, stackable limit context managers, EvalLog v2 schema, SandboxEnvironment) plus well-established benchmark papers (τ-bench 2406.12045, SWE-bench, MT-bench 2306.05685, PoLL 2404.18796, METR Vivaria) — all verified to exist. The benchmark IDs and the PoLL/MT-bench claims were cross-checked in the engagement/critic verification passes and resolve correctly. **The exact Inspect AI API surface (function signatures, field names, `as_tool`/`handoff` semantics) is the one thing to re-verify against the live docs at implementation time** — it is the foundation of the build and Inspect's API has moved across releases. Pin the Inspect version in `pyproject.toml` and confirm signatures against that pinned version before relying on them.
