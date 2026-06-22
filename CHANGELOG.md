# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

Dogfood-swarm hardening + the first runnable diagnostic cycle + the durable catalog
(2026-06-21). Every change cross-family-verified (Ollama Cloud panel) and test-first;
suite **473 → 783** tests, **91.5% → 94.6%** coverage. Pre-1.0 research preview unchanged.

### Added
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
