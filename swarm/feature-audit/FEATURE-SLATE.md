# Feature Audit — Prioritized Slate (user review gate, Law 8)

16 candidates (3-agent read-only audit, grounded in gameplan + research-grounding + code) collapse into **4 epics**. The instrument is health-complete (594 tests, 94.51%, 0 CRIT/0 HIGH) but is still a **substrate**: it can *characterize judges* but cannot yet *run a single diagnostic cycle*. The build is deliberately seam-first — most of this is **wiring existing shapes**, not building from scratch.

## Epic 1 — First real diagnostic cycle  ⭐ RECOMMENDED FIRST (foundational; unlocks 2 & 4)
**Composes:** roadmap-capability-002 + operability-glue-001/002/003/004 + challenge-test-surface-001.
**What:** the glue that makes the instrument actually *measure a model on a puzzle* end-to-end:
- a real `OracleRunner` that copies the workdir out (`copy_workdir_out` exists) and executes the seed puzzle's sealed `oracle/check.py` against it → a real `OracleOutcome`;
- setup-script execution to stage the Solver workdir in the sandbox;
- an `ai-crucible run <puzzle> --model <id> --k N` CLI subcommand wiring a live model `.generate` (Solver) + the seated panel into `run_pass_hat_k` → the `pass^k`/Wilson rollup;
- a small run-config surface (model/family, puzzle, budgets, panel).
**Grounding:** gameplan **Phase 4** ("wire end-to-end; first real cycle on sulzbach-co #55252" — already authored on disk) + research-grounding §10.4/§10.6. Kernel docstring explicitly says "Phase-2 wires the real harness behind the same shape."
**State:** injected seam present (kernel choke point, copy_workdir_out, LocalSandbox, model adapters, panel bridge all built+green); the real `OracleRunner`, setup exec, and `run` command are genuinely absent.
**Effort:** L. **Open question (study-swarm target):** how are penalty `trigger`s (answer-key fetch, redundant-tool-calls ratio) actually *detected* from the trace/workdir — trace-driven, FS-access-driven, or both?

## Epic 2 — Adversarial + self-test harness  (your "ways to challenge & test")
**Composes:** challenge-test-surface-002 (bypass) + 003 (trivial/impossible discrimination) + 004 (eval-awareness probe) + 006 (more seed puzzles).
**What:** exercise + stress the instrument — prove the **ADVERSARIAL critical penalty actually fires** through the kernel (scripted Solver fetches the answer key → gate closes); a **self-test** that proves the instrument discriminates a known-trivial (all pass) from a known-impossible (any pass = leak) puzzle *before* trusting a real run; the **eval-awareness probe** (paired deploy/test framing arms as the per-release boundary gate); **2-3 more seed puzzles** on distinct capability axes (catalog is currently 1 puzzle).
**Grounding:** research-grounding §8 (auditing-game/bypass/honeypots) + §10.5 (eval-awareness gate) + §5 (catalog harvesting). Seams: declared penalties, `known_groups`, the framing arms, `eval_awareness.py`, the seed template.
**Effort:** M each. **Builds on Epic 1** (needs the real run path).

## Epic 3 — Retire the circular ω (human-labeling round)  (the known top product gap)
**Composes:** roadmap-capability-001 + operability-glue-005 + challenge-test-surface-007.
**What:** run a real ≥3-annotator / ≥30-item human-labeling round through the Fork-C harness that **already exists** (loader, `metrics.alt_test` BY-FDR, Krippendorff α, the `--human-labels` flag that flips `alt_test_reference` off "model-jury-bootstrap") — plus the missing elicitation glue (`label export` of gold-stripped items → `human_labels.json`).
**Why:** every seat in `panel.json` is PROVISIONAL because ω is circular; this is the single change that makes the seats defensible. Harness complete, **never used** (no real `human_labels.json` exists).
**Effort:** L (mostly process + elicitation glue). **This one has a decision that's genuinely yours + a study-swarm target:** who are the ≥3 annotators for a 1-human studio (director + 2 crew? external?), at which tier ε — and does that reintroduce a correlated-rater problem the alt-test assumes away? *I should not presume the annotator model.*

## Epic 4 — Catalog growth (Phase 5; later)
**Composes:** roadmap-capability-003 (persistent catalog + Lab→Arena→Regression graduation lifecycle) + roadmap-capability-004 (differential typology: Claude-vs-cross-family delta → Claude-gap / LLM-general / Claude-strength).
**State:** `CatalogTier` enum + graduation rule exist; persistence + transitions + the differential computation are absent (`differential` grep is empty). **Depends on Epic 1** (needs accumulated run histories to graduate/differentiate).
**Effort:** M.

## Recommendation
**Epic 1 first** — it's the foundational unlock (everything else needs a real run path), the most grounded (Phase 4, named seed puzzle), and largely wiring. Then **Epic 2** (proves it works + is your challenge/test surface). **Epic 3** in parallel-ish but gated on your annotator-panel call. **Epic 4** last (Phase 5). For any epic with a design question (Epic 1 penalty-detection, Epic 3 annotator panel), I fire a **study-swarm** before building, per your authorization.
