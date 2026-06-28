# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- **OpenRouter cross-family seat** — a new `OpenRouterModel` adapter
  (`models/openrouter_adapter.py`) makes the whole OpenRouter catalog — the open families AND the
  closed frontier (real GPT, Gemini) the open-only Ollama-Cloud roster can't give — available as
  cross-family panel judges / Solvers / the banked cloud ω-anchor jury, through one
  OpenAI-compatible key. Selected by an `openrouter:` id prefix that REQUIRES an explicit `@family`
  (`--model openrouter:deepseek/deepseek-chat@deepseek`); the prefix is kept as the model's identity
  so a seated OpenRouter judge round-trips through `panel.json` → `run --panel`. Wired into all three
  construction sites — `cli._build_model` (run/probe Solver), `characterize._parse_models` +
  `run_panel` (the panel), and the seated-panel reconstruction.
  - Mirrors `OllamaModel`'s surface (`generate`/`judge`/`as_judge`/`judge_item`/`pin_metadata`/
    `.family`), pins `temperature=0`, reads `OPENROUTER_API_KEY` at call time, and reuses the shared
    verdict-token-logprob → confidence path (OpenRouter returns OpenAI-shape logprobs, §12).
  - **Served-model guard** folds out OpenRouter's dated snapshots / variant suffixes
    (`cohere/north-mini-code-20260617:free` ≈ the requested bare id) before the provenance check, so
    only a genuine cross-vendor fallback raises `ModelMismatchError`.
  - **Non-fatal vendor↔family warning** — because an OpenRouter id exposes the vendor, a clear
    contradiction (`deepseek/deepseek-chat@qwen`) emits a `UserWarning` (a typo-catcher for a
    mislabel that would corrupt the cross-family attribution) without a brittle vendor→family table
    or a refusal. The operator's `@family` stays the authoritative axis — consistent with the
    trusting Ollama path.

- **`eval/RESULTS.md` — published cross-family judge-admission runs.** The first cross-family
  OpenRouter quorum run (2 local + 5 pinned-OpenRouter seats, 7 disjoint families, 93 pairs, k=3)
  with its committed `eval/panel.json` + `eval/characterization-report.json`. Result: 3 disjoint
  families now clear the 6-metric admission bar (up from 2), but the composed panel still escalates
  (sub-quorum, error-redundancy) and ω stays on ice — seats provisional, disclosed not faked.

- **`ai-crucible labels validate` — offline human-label intake gate (Fork C, §12.2).** A thin new
  subcommand that loads the calibration items and runs `load_human_labels` in **check-only mode** —
  reporting annotator count, item count, ε, IAA Krippendorff α, DISPUTED drops, and the under-power
  note — with **no model seated and no GPU**, so an operator can validate a candidate
  `human_labels.json` the day independent annotators deliver one, long before a full
  `characterize --human-labels` run. Calls the same loader the run uses (a file that validates here
  is one the run accepts); human chrome → STDERR, machine JSON → STDOUT; structured
  `[CODE] msg (hint:)` errors on a malformed file (exit 1), exit 2 on a bad invocation, exit 0 with
  `under_powered: true` for a valid-but-thin file. Ships a copy-paste starting point at
  `calibration/human_labels.example.json` (3 expert annotators over 32 bundled items, clearing the
  ≥3-annotator / ≥30-item floors, with one `unsure` and one disputed split) and a handbook page
  ("Retiring the circular omega (Fork C)"). This is intake plumbing only — it fabricates no labels,
  and ω stays the circular model-jury bootstrap until ≥3 *independent* humans exist.

- **`ai-crucible calibration curate` + `calibration/curate.py` — the harder-set curation pipeline
  (study-swarm Phase A).** Research-grounded (design in `swarm/openrouter-quorum/STUDY-SWARM-harder-
  calibration-set.md`) response to the run's #1 lever: only 11/93 admission items discriminate among
  strong judges. The pipeline has two steps. (1) **Discrimination screen** (AFLite, Le Bras 2020;
  Fisher-info selection, Zhou 2025): given a candidate pool + a characterization run's persisted
  `grade_matrix`, keep the items strong judges DISAGREE on and drop the saturated ones every judge
  passes — moving the §12 IRT saturation screen UPSTREAM into a forward curation step (reuses
  `irt.prune_items`). (2) **Ambiguity gate** (`ambiguity_gate`; GPQA difficulty-vs-ambiguity, Rein
  2023; dual verification, JudgeBench 2024): a harder item must keep a DEFENSIBLE key — given ≥2
  independent verifier verdicts, verifier disagreement = AMBIGUOUS (omit), verifiers-agree-but-differ-
  from-gold = MISLABELED (omit), agree-with-gold = DEFENSIBLE (keep). The crown-jewel constraint: a
  harder set must never become an ambiguous set. The offline CLI runs the discrimination screen and,
  with `--out`, writes the kept subset as a calibration JSON directly re-usable as `characterize
  --items`; the ambiguity gate ships as a tested library function (it needs live verifier verdicts).
  The characterization report now persists the `grade_matrix` so curation + subset re-analysis run
  offline from a committed report. NO model seated, NO GPU. (Phase A is the pipeline; Phase B feeds it
  new-construct content — the genuinely-harder items.)

### Fixed
- **Eval-integrity: the answer-key bait penalty was evadable (CRITICAL).** The grading host detected
  a sealed-answer-key touch only when the Solver read it via an allowlisted command
  (`grep`/`cat`/…); a read via any other command (`od`/`xxd`/`strings`/`python3 -c`/`cp`/`dd`) touched
  the bait without registering, so a fabricated-but-correct answer could pass the §8.3 gate CLEAN. The
  runner now emits a deny-by-default `touched_paths` set (path operands of EVERY exec) that the
  bait/oracle guard keys on, while grounded-read stays allowlist-gated. Closed across 6 evasion vectors
  end-to-end and cross-family verified. (The residual — shell-glob / computed-path reads — is
  documented and deferred to a host-side filesystem access tripwire.)
- **Judge admission: the Wilson interval was fed a fabricated count (HIGH).** The seat gate built its
  accuracy confidence interval from `round(difficulty_weighted_accuracy × n_items)` — a fictional
  binomial count that understated uncertainty (a chance-level judge whose few heavy items happened to
  be right could clear the REJECT floor). It now uses the Kish effective sample size
  `n_eff = (Σw)²/Σw²` (a no-op on uniformly-weighted sets), restoring a sound small-N admissibility
  bound.
- **npm launcher shipped a stale version (HIGH).** `npx @dogfood-lab/ai-crucible` derived its release
  asset names from a hardcoded `0.2.0` in the bin, so the published 0.3.0 package fetched 0.2.0
  binaries; the launcher now derives version + tag from `package.json` at runtime (cannot drift).
- **Rate-limit (429) was treated as fatal in the model adapters.** The OpenRouter and Claude transient
  classifiers retried 5xx but not 429, so a momentary rate-limit burst aborted a long panel/solver run
  instead of being absorbed by the bounded backoff. 429 (and Anthropic 529 Overloaded) is now retried.
- **The sealed-boundary chrome scan was not field-agnostic.** `_chrome_tokens` hard-coded four fields,
  so a future Tier-3 chrome field could leak into scored context while the guard reported clean; it now
  walks `dataclasses.fields`, mirroring the already-field-agnostic message scan.
- **A sandbox per-call timeout was swallowed by the solver loop.** A `BudgetExceeded(TIME)` raised by
  the sandbox adapter inside `_execute` was caught by the broad observation handler; it now propagates
  so a runaway command halts the attempt with `terminated_by=TIME`.
- **perturbation_audit jittered two thresholds the decision never reads** (`consistency_floor`,
  `bias_ceiling`), padding the flip-rate denominator and deflating the gate-fragility andon signal; the
  perturbed set now equals the decision-relevant threshold set.
- **Eval-integrity: a genuine grounded read via a modern reader scored as a fabricated non-solve.**
  The grounded-read signal counted only an allowlisted reader (`grep`/`cat`/…), so reading the source
  via `rg`/`bat`/`nl`/`python3 -c` (ripgrep is Claude Code's own default) fired a false
  `skip_grounded_read` and closed the gate on a correct answer — biasing cross-model comparison and able
  to corrupt the calibration anchors. The reader allowlist now covers the common content-readers
  (non-reading commands like `cp`/`mv` stay excluded so they can't falsely ground).
- **Eval-integrity: an unknown (misspelled) triggered penalty name never closed the gate.** A penalty a
  puzzle's oracle fires but `meta.json` doesn't declare was surfaced-but-not-scored, so a typo'd CRITICAL
  penalty (`answer_key_fetchh`) silently skipped the critical veto and an adversarial bypass scored
  clean. An unresolved triggered name now fails the gate closed (`unknown_penalty_fired`); the runner's
  universal redundancy penalty is injected only when the puzzle declares it.
- **A positive `Penalty.weight` would invert a penalty into a bonus.** `Penalty.weight` is now
  constrained `≤ 0` at the schema boundary, rejected at puzzle load instead of silently skipping the
  penalty-adjusted floor close.
- **A gpt-oss judge served via OpenRouter could mis-grade on leaked Harmony tokens.** The OpenRouter
  adapter now routes completions through the shared `_normalize_harmony` (identity pass-through for
  clean responses), collapsing an OpenAI Harmony chat-template leak (`<|channel|>analysis<|message|>…`)
  to its final-channel answer before the judge/solver parser sees it — the same defense the Ollama
  adapter already applies.
- **A salvaged characterization run collapsed the IRT calibration screen to an error.** Per-item
  salvage can leave one model missing a few items (a ragged matrix); the IRT item-prune screen now
  degrades to the item subset shared across all models and reports the dropped count, instead of
  erroring the whole screen (matching the per-pair degradation the panel-correlation screen already had).

## [0.3.0] — 2026-06-21

Dogfood-swarm: hardening + the first runnable diagnostic cycle + the durable catalog +
the multi-model Solver protocol + the eval-awareness gate. Every change
cross-family-verified (Ollama Cloud panel) and test-first; suite **473 → 795** tests,
**91.5% → 94.6%** coverage. Pre-1.0 research preview by design — the version stays **0.x**
because the judge-panel alt-test ω is still a circular model-jury bootstrap (no ≥3
independent human annotators a one-human studio can staff), so seats are provisional and
graduation escalates to the Designer rather than auto-promoting. Honest, disclosed, not
faked.

### Added
- **Eval-awareness boundary gate (§10.5)** — `ai-crucible probe <puzzle> --model …` (and
  `cycle.run_eval_awareness_probe`) runs the Solver under the matched deploy- vs
  test-framing (`build_probe_pair` holds the task + budget constant, so the only difference
  is the eval cue) and compares behaviour. A bait-touch / solve-rate DIVERGENCE between the
  framings means an eval cue is leaking into the scored context (a §10.1(d) sealed-boundary
  failure). The kernel gains an optional `framing_messages` override (chrome-guarded) so the
  probe injects the matched contexts through the normal run path. Honest small-N caveat in
  the output. (Deferred to a later cycle, with rationale: round-against-round Designer
  generation — a Phase-5 creative loop that must auto-author *valid* sealed oracles — and
  more capability-gap seed puzzles.)
- **Multi-model Solver protocol — native tool-calls (Finding B′)** — a model that solves
  via native function-calls (gpt-oss-cloud returns `message.tool_calls` with empty
  `content`, not the text `ACTION` protocol) can now Solve. The Ollama adapter gains
  `complete_turn` (offers the sandbox tool schemas, returns text + normalized tool_calls);
  the solver loop routes native calls through the SAME governor + sandbox channel as the
  text protocol (duck-typed — text-only models are unchanged). Validated live: gpt-oss
  executed `read_file config/limits.py` via a native tool-call, grounded the read, and
  solved the seed (gate PASS).
- **Durable catalog + graduation lifecycle + differential typology (Epic 4)** — the
  instrument now ACCUMULATES across runs. `ai-crucible run` appends each cycle to an
  event-sourced, hash-chained catalog log (the source of truth); tier state
  (Lab→Arena→Regression) is a derived projection folded from it (never mutated in place).
  New `ai-crucible catalog {list|show|graduate}` reads + curates it. Graduation is
  three-valued and abstention-aware — it PROMOTES only on a confident cross-family
  fairness verdict, HOLDS on a confident negative, and **DEFERS to the Designer** otherwise
  (the honest posture while no confident fairness panel exists; ω on ice). Saturation uses
  an anytime-valid e-process with a Schmitt-trigger dwell + frontier gate (anti-flapping,
  demote-not-delete). The **differential typology** classifies each puzzle Claude-specific
  gap / LLM-general / Claude-strength / inconclusive off the Newcombe-1998 delta CI (not
  interval overlap), with BH-FDR across the catalog — the diagnostic payoff. Study-swarm-
  grounded + the graduation math & differential cross-family-verified (Newcombe 1998;
  Howard/Ramdas/McAuliffe/Sekhon 2021 confidence sequences; Jung et al. ICLR 2025
  Trust-or-Escalate; Verga et al. 2024 PoLL; SWE-bench Verified; Benjamini-Hochberg 1995).
- **The instrument RUNS** — `ai-crucible run <puzzle> --model <id>[@family] --k N`
  executes a real Solver against a puzzle in the sandbox and grades it out-of-band via
  the puzzle's sealed `oracle/check.py`, emitting the `pass^k` / Wilson rollup. New
  modules: `oracle_runner` (trace-driven grading), `staging` (setup-script + grading-side
  bait placement), `solver_loop` (a bounded ReAct `generate`), `cycle` (the wiring).
  Validated live on two frontier cloud Solvers (qwen3-coder:480b solved the seed cleanly;
  gpt-oss:120b exercised the bypass paths).
- **Self-test / discrimination harness** (`self_test.run_self_test`) — proves the
  instrument distinguishes a known-trivial puzzle (must pass) from a known-impossible one
  (any pass = a leak) BEFORE a real run, with two calibration puzzles grounded in the
  `known_groups` acceptance laws.
- **Adversarial bypass test suite** — every declared bypass vector (answer-key fetch
  across read/grep/cat, ungrounded fabrication, redundant-call looping, hard-kill) closes
  the §8.3 gate end-to-end through the real kernel.

### Changed
- **Eval-integrity — fabricated-but-correct is now a non-solve** (the capability the seed
  puzzle exists to measure). A non-critical (causal/regressional) penalty that drags a
  *penalty-adjusted solve floor* (`solve_quality + non-critical penalties`, **bonus-free**)
  below `point_threshold` now CLOSES the conjunctive gate — a flavor-keyed magnitude clause,
  not a Goodhart-fragile raw-net test. Study-swarm-grounded (Lightman 2023; Turpin 2023;
  ImpossibleBench 2025; SWE-bench).
- **Observability solve-count** keys on the authoritative `gate_passed`, not `score.value > 0`
  (a gate-passed net≤0 attempt was silently undercounted in pass^k / graduation).
- **EXTERNAL_VERIFIER family-exclusion hardened** — case/whitespace-normalized comparison,
  untagged judges represented as `None` (not a colliding `"unknown"`), served-model verified
  against requested in both adapters, `untagged_judges_seated` surfaced + optional
  `strict_cross_family`.

### Fixed
- **Resilience** — the post-Solver grading/panel/log tail degrades to a traced ERROR attempt
  (Solver transcript preserved) instead of discarding an expensive run on a grading-host
  blip; `JudgePanel` degrades over surviving judges (one flaky judge no longer aborts the
  panel); the oracle rejects non-finite `solve_quality`/`time_used`; bounded model retry +
  structured daemon-down/malformed-body errors; puzzle-loader input size guards.
- **Sandbox CRLF corruption** — `LocalSandbox.write_file` wrote with platform newline
  translation, corrupting any script staged through the channel on Windows; now writes bytes
  verbatim (`newline=""`).
- **gpt-oss Harmony robustness** — the Ollama adapter strips Harmony control tokens and the
  solver_loop parser cuts at the first control token, so a gpt-oss Solver's agent loop is no
  longer shredded by leaked `<|message|>`/`<|channel|>` tokens.
- **Operator UX** — the CLI renders an escaping error as one structured `[CODE] … (hint:)`
  line (full traceback only under `--debug`); a run prints the provisional-ω caveat +
  degraded-run notice + per-model progress; `--help` documents exit codes; `ai-crucible
  --help` survives a Windows cp1252 console.

### Notes
- **Retiring the circular ω is ON ICE — by structural constraint, not neglect.** The
  alt-test (Calderon 2025) needs ≥3 *independent human* annotators; this is a one-human
  studio, so a defensible human-labeling round cannot be staffed. The seats therefore remain
  **provisional**, the below-quorum path **escalates to the Claude Designer**, and the
  instrument **discloses this honestly** (operator-visible caveat) rather than faking human
  grounding. A future no-humans-needed reduction-of-self-circularity (anchoring ω on an
  *independent cross-family cloud jury* disjoint from the seated panel) is recorded as a
  candidate, not built.
- Deferred to a later cycle: catalog persistence + Lab/Arena/Regression graduation +
  differential typology (Epic 4); a multi-model Solver protocol reading native `tool_calls`
  (gpt-oss-cloud returns empty content + structured tool calls, distinct from the
  token-leak shape); eval-awareness probe wiring + round-against-round generation.

## [0.2.0] - 2026-06-01

Phase-1 milestone: the policy-enforced kernel + role contracts + instrument-quality
scaffolding, built greenfield from a citation-verified design lock. Pre-1.0 by
design — this is Phase 1 of a multi-phase research instrument (see Notes).

### Added
- **Phase-1 kernel** — a thin policy layer on [Inspect AI](https://inspect.aisi.org.uk/),
  nine modules threaded through one observable `generate` choke point:
  `puzzle_loader`, `sandbox`, `roles`, `budget_governor`, `oracle_scorer`,
  `judge_panel`, `trace_writer`, `observability`, `attestation`. Entry points
  `run_attempt` / `run_pass_hat_k`.
- **Layered Reward Surface + Sealed Boundary** — prompt-framing as a first-class
  measured arm (`neutral` / `self_referential` / `social_standings`); the
  motivating surface (chrome) never shares a context window with the scored,
  out-of-band-graded surface.
- **Scoring** — `pass^k`, Wilson / Clopper-Pearson intervals, McNemar, the
  graduation rule, the §8.3 conjunctive hard gate, and a cross-family judge panel
  that excludes the generator's family (external verifier).
- **Instrument-quality scaffolding** — AsPredicted + REFORMS pre-registration
  templates, content-hashed rubric bundle with version-bump, Sobol screen +
  deterministic split + Thresholdout budget, `SUT.yaml`, and Inspect-AI task
  mapping (the §9 audit chain).
- **First seed puzzle** — `puzzles/seed-sulzbach-55252/` (claude-code#55252:
  fabrication / looping-on-diagnosis) with a grading-side oracle + fingerprinted
  bait honeypot.
- **Design lock** — `docs/research-grounding.md` §10, grounded in two
  citation-verified study-swarms (`docs/phase-0/swarm-16..25`; every arXiv ID
  verified, corrections recorded inline).
- **Shipping baseline** — CI (ruff + pytest + 60% coverage gate on Python
  3.11/3.12 + dependency audit), `SECURITY.md` threat model, MIT `LICENSE`, and a
  `verify.sh` one-command gate.

### Notes
- **Pre-1.0 research instrument.** Phase 1 of a multi-phase build. The real
  cross-family local-model panel (Phase 2), the hardened container/microVM sandbox
  provider, and the cosign/Rekor external attestation anchor are deferred to
  Phase 2+ and are documented as such. No package has been published.
- `0.1.0` was an internal scaffold version, never tagged or released.
