# Epic 4 — Wave-0 Contract Lock

**Catalog persistence + graduation lifecycle + differential typology.** Greenfield
build: this contract is frozen first (coordinator), then parallel leaf agents build
NEW files against it, then the coordinator integrates + re-audits + cross-family
verifies. Grounded in the Epic-4 study-swarm (`swarm/epic-4/STUDY-SWARM.json`, 4
parallel web-research agents + adversarial synthesis, 2026-06-21). Every load-bearing
decision below carries its citation.

## The seam that already exists (do not rebuild)

- `types.CatalogTier` (LAB / ARENA / REGRESSION) — **reuse**, do not redefine.
- `scoring.stats.graduates(successes, n)` — the existing fixed-N Wilson band
  (`0.10 ≤ wilson_lower ∧ wilson_upper ≤ 0.90`). **Clause 1 of graduation; unchanged.**
- `scoring.stats.wilson_interval / clopper_pearson / mcnemar_exact` — reuse.
- `attestation.JsonlEventStore` — `append(event)->hash`, `read_events()->list[dict]`,
  `verify_hash_chain()->bool`, `canonical_json`, `chain_hash`, `GENESIS_HASH`. **The
  catalog log IS a JsonlEventStore.** No edits to `attestation.py`.
- `meta.PuzzleMeta.{catalog_tier, min_k}` — the per-puzzle floor + tier seed.
- `observability.PuzzleHistory` / `roll_up` — the in-memory per-batch rollup a run
  produces; Epic 4 *persists* what it currently throws away.

`grep -r differential src/` is empty; `PuzzleHistory` is in-memory per-batch only.
Persistence, tier transitions, and the differential are genuinely absent.

---

## The four design locks (study-swarm, web-confirmed citations)

### A. Graduation — Lab→Arena is THREE-VALUED, abstention-aware

`promote_decision(...) -> Decision{PROMOTE, HOLD, DEFER_TO_DESIGNER}`. Gates on THREE
clauses that catch orthogonal failure modes:

1. **Claude band (unchanged):** `stats.graduates(claude_succ, claude_n)`. Fail → `HOLD`.
   (Rules out trivial AND impossible on Claude's own rate — research-grounding §1.)
2. **Cohort non-trivial (NEW, solve):** `wilson_interval(cohort_succ, cohort_n).upper
   ≤ 0.90`. If the cross-family cohort trivially aces it → `HOLD`. Catches the mode
   solve-rate-on-Claude-alone is blind to: a puzzle easy for everyone else.
   *(BBH live-prune: keep only items genuinely hard vs a strong external baseline —
   Suzgun et al. 2022, arXiv:2210.09261, web-confirmed.)*
3. **Judge-fairness (NEW, abstention-aware keystone):** consume the panel signals that
   ALREADY exist. `deferred = (not panel.meets_quorum) or panel.escalate`. A
   low-confidence fairness verdict → `DEFER_TO_DESIGNER` (Trust-or-Escalate); a
   confident "unfair/broken" → `HOLD`; confident "fair" → `fairness_ok`.
   *(SWE-bench Verified: human screening flagged 38.3% underspecified + 61.1% broken
   tests, 68.3% filtered — fairness is a graduation criterion solve-rate cannot see;
   OpenAI/Chowdhury et al. 2024, web-confirmed. Trust-or-Escalate: a low-confidence
   jury must abstain+escalate, never fake confidence — Jung/Brahman/Choi, ICLR 2025,
   arXiv:2407.18370, web-confirmed. PoLL: 2<3 disjoint families is genuinely weaker —
   Verga et al. 2024, arXiv:2404.18796, web-confirmed.)*

`PROMOTE iff claude_band_ok ∧ cohort_nontrivial ∧ fairness_ok ∧ not deferred`.
Promotion out of `DEFER` requires an **explicit attested Designer override** — never
auto-promote on a deferred fairness verdict, never synthesize a fake quorum. Use
**fixed-N Wilson** here (graduation is a one-shot decision at the pre-chosen N=20 floor)
— do NOT swap in the saturation confidence-sequence. At graduation, call the differential
classifier to stamp the Arena entry's **born typology label** (INCONCLUSIVE_UNDERPOWERED
is a valid born-state).

> **Honesty caveat (locked):** the DEFER threshold is a HEURISTIC posture borrowed from
> Trust-or-Escalate, NOT its provable human-agreement guarantee (that assumes a
> *calibrated* confidence signal; a raw 2-seat weighted margin is not one). Do not market
> the gate as "provably bounded false-graduation risk." Report the spread; do not
> manufacture calibration.

### B. Saturation — anti-flapping demotion (Schmitt-trigger hysteresis + dwell + frontier gate)

Saturation is **NOT** `wilson_upper > 0.90` on a single batch (it flaps). DEMOTE
Arena→Regression for `(puzzle, model m)` iff ALL of:

1. **High bar (anytime-valid):** an **e-process** for `H0: p ≤ 0.90` vs `H1: p ≥ 0.97`
   crosses `1/α = 20` (α=0.05). A confidence SEQUENCE, not fixed-N `wilson_lower` —
   the catalog peeks every run, so acting on a fixed-N interval at a data-dependent
   stopping time inflates the false-demotion rate.
2. **Dwell:** the e-process unit is the **k-attempt RUN, not the attempt**, and dwell
   is required across `≥ dwell_K` SEPARATE runs (default 2) AND `post_trigger_n ≥
   3·min_k`. Dwell is **emergent** from the e-process (can't cross on one lucky run),
   not a magic constant. Separate runs/seeds/dates respect Liu-2025's 71% within-batch
   identical-failure correlation.
3. **Frontier gate:** `m` is a current-frontier / panel-admitted capable subject. A
   weak local model passing never demotes; **everyone-fails stays in Arena** (it is the
   LLM-general frontier gap — maximally diagnostic). `is_frontier` is an injected
   predicate (policy input, default-configurable).

RE-PROMOTE Regression→Arena iff a symmetric e-process for `H0: p ≥ 0.75` crosses the
threshold on the best current subject (the rate has dropped below the band). The
**0.75–0.90 dead band** is the guard band that kills demote/re-promote oscillation. Wire
re-promotion (a genuine new-version capability REGRESSION must be able to re-promote),
not just demotion. Record **which model + date** saturated each puzzle (capability-
evolution timeline); demote-not-delete.

*(Page CUSUM 1954, integrate-before-signal; Howard/Ramdas/McAuliffe/Sekhon
time-uniform confidence sequences, Annals of Statistics 2021 49(2):1055-1080,
doi:10.1214/20-AOS1991, web-confirmed; "When AI Benchmarks Plateau" Akhtar/Reuel et al.
2026 arXiv:2602.16763, web-confirmed — act only when an uncertainty-aware index
CONSISTENTLY exceeds high thresholds.)*

> **Locked distinction (do NOT unify with graduation):** graduation uses FIXED-N Wilson
> (one-shot at pre-chosen N); saturation uses a confidence SEQUENCE / e-process (peeks
> every run). A maintainer who "simplifies" both to `wilson_interval()` silently breaks
> saturation's anytime-validity. Flag loudly in code.

### C. Differential typology — drive from the CI on the DELTA, four-valued

Drive the typology from the confidence interval on `δ = claude_rate − cohort_rate`, NOT
by comparing two separate intervals (non-overlapping-Wilson is statistically wrong:
over-conservative, loses power — **REJECT it with a docstring warning**). Claude and the
cohort run independent attempt sets → this is an **UNPAIRED** two-proportion comparison.
**McNemar is the WRONG tool here** (reserve it strictly for genuinely paired runs; gate
behind `paired=True` requiring explicit b/c counts — the architecture does NOT produce
trial-level pairing between Claude and the CohortSolver).

- Primary estimator: **Newcombe-1998 hybrid-score CI** for `p1−p2`, built from two
  `wilson_interval()` calls.
- **Fourth class `INCONCLUSIVE_UNDERPOWERED`** is first-class (NOT a forced 3-way). At
  N=20/k=3 the MDE is large, so most puzzles legitimately land here — correct behavior,
  not a bug.
- Classify directionally ONLY when the delta-CI **excludes zero** AND `|δ| ≥ mde_floor`
  (default 0.30, config-pinned, from Miller's MDE ≈ 1/√N at N=20/k=3):
  - `upper < 0` → `CLAUDE_SPECIFIC_GAP` (Claude underperforms cohort — highest value)
  - `lower > 0` → `CLAUDE_STRENGTH` (anti-regression)
  - `LLM_GENERAL_GAP` requires **positive** shared-failure evidence (delta-CI contains 0
    AND both Wilson uppers ≤ `general_fail_ceiling`, default 0.20) — never merely a null δ.
- Unit of analysis is the **PUZZLE** (one cluster of N attempts), never pooled attempts.
- Catalog level: `classify_catalog()` applies **BH-FDR (q=0.10)** across per-puzzle delta
  p-values and downgrades non-survivors to `INCONCLUSIVE_UNDERPOWERED` even if their solo
  delta-CI excluded zero. Same classifier stamps the born label at graduation (one
  measurement, two uses).

*(Newcombe 1998, Statistics in Medicine 17(8):873-890, web-confirmed — the
Wilson-combining hybrid "performs well irrespective of sample size"; Schenker &
Gentleman 2001, Am. Statistician 55:182-186, web-confirmed — interval-overlap is
over-conservative, test the difference; Miller "Adding Error Bars to Evals" 2024
arXiv:2411.00640, web-confirmed scope — MDE ≈ 1/√N, cluster by puzzle.)*

### D. Persistence — EVENT-SOURCED on the existing JsonlEventStore

Build the catalog as event-sourced on `attestation.JsonlEventStore`. Do not invent a new
store, do not add a dependency, do not maintain a mutable tier table.

- The **JSON-lines hash-chained log is the SOURCE OF TRUTH** (the audit artifact itself):
  RFC-8785 canonical JSON, `prev_hash`/`hash`, `verify_hash_chain`. Git-diffable,
  externally verifiable.
- **Tier state (Lab/Arena/Regression) is a DERIVED projection** rebuilt by folding events.
  NEVER update a tier in place; append a TRANSITION event and re-fold.
- An optional stdlib `sqlite3` view is a **rebuildable cache only** ("log wins"). **Deferred**
  for this build unless trivially free — the JSONL fold is the deliverable.
- **Two event types in one log:**
  - `RUN {run_id, puzzle_id, puzzle_content_hash, model_id, family, role, k, n,
    successes, pass_hat_k, wilson_lower, wilson_upper, started_at, finished_at,
    schema_version, rule_version, provenance_level, ...}`
  - `TRANSITION {transition_id, puzzle_id, from_tier, to_tier, reason_code, evidence{...},
    decided_by, recorded_at, schema_version, rule_version}`
  - `reason_code ∈ {GRADUATE_WILSON, DEFER_TO_DESIGNER, SATURATED_REGRESSION,
    DEMOTE_NOT_DELETE, REPROMOTE_REGRESSION, MANUAL}`.
- **Idempotency via deterministic ids:** `run_id = sha256(puzzle_content_hash +
  model_id + started_at + nonce)`; `transition_id = sha256(puzzle_id + from_tier +
  to_tier + sorted(contributing_run_ids) + rule_version)`. The folder skips
  already-folded transition ids so replay is idempotent and a transition fires at most
  once per evidence set.
- **PIN `rule_version` on every RUN and TRANSITION** (keystone): graduation/saturation
  thresholds are policy knobs. `rule_version` makes the timeline reflect the decision made
  AT THE TIME, not a retroactive re-judgment. `derive_tiers` records the active
  `rule_version` on each emitted transition; full per-event rule-version *replay* (a
  `rule_resolver` seam) is left for when rules first change — disclosed, not faked.
- **Reference puzzles by `puzzle_content_hash`** (hash over meta.json + oracle +
  setup_script) so an edit forks a NEW lineage; the timeline never conflates two versions.
- **Provenance honesty:** stamp `provenance_level` honestly (1 = hash-chain-only ≈ SLSA
  L1, 2 = cosign-signed). The chain is **tamper-EVIDENT, not tamper-PROOF** — a single
  writer can rewrite it consistently. Never stamp level 2 without a real signature
  (matches `attestation.py`'s "honest provenance or none").

*(Event Sourcing — Fowler 2005; in-toto USENIX Security 2019 + SLSA spec — pin the
provenance tuple, declare an honest level; LiveCodeBench Jain et al. 2024 arXiv:2403.07974
— date-stamp + record training-cutoff for a contamination axis; DVC content-addressing
— edits fork lineage.)*

---

## Cross-cutting LOCKS (the synthesis caught these; honor them)

1. **Cohort runs at the SAME N floor as Claude.** The most load-bearing consistency
   requirement: Clause-2 Wilson-upper, the frontier gate, and the Newcombe delta-CI all
   degrade if the cohort N < Claude N. Default floor N=20 (min_k drives it per puzzle).
2. **Two Wilson-family estimators, do NOT unify** (graduation fixed-N vs saturation
   e-process). Code comments flag this loudly.
3. **`rule_version` on every event** (replay determinism vs mutable knobs).
4. **Monitor the DEFER fraction as instrument health.** Routing DEFER to the Claude
   Designer reintroduces a same-family fairness judge — acceptable ONLY as the disclosed,
   attested escalation of last resort under structural sub-quorum (ω on ice). A RISING
   defer rate means *recruit a third disjoint family*, NOT relax the threshold. Surface
   `defer_fraction` in the catalog summary.
5. **Abstention threshold is HEURISTIC, not provably bounded** — state honestly (see A).
6. **e-process unit is the RUN, not the attempt; dwell across SEPARATE runs** (Liu-2025).
7. **Tamper-EVIDENT ≠ tamper-PROOF** — provenance_level honest (see D).
8. **McNemar misuse guard** — `classify_puzzle` defaults `paired=False`; `paired=True`
   asserts explicit b/c counts from a single shared attempt log.

---

## Module layout + EXCLUSIVE FILE OWNERSHIP

Wave-0 (coordinator, committed BEFORE leaves):
- `src/ai_crucible/catalog/types.py` — the contract: enums + dataclasses + the injected
  decision-fn Protocols. Reuses `types.CatalogTier`.
- `scoring/stats.py` **extension** (coordinator) — the pure, general primitives the leaves
  consume: `newcombe_wilson_diff`, `benjamini_hochberg`, `bernoulli_log_eprocess`. Tests
  in `tests/test_scoring.py` (extend) — coordinator-owned.
- `src/ai_crucible/catalog/__init__.py` — placeholder (integrator finalizes exports).

Leaves (parallel, each owns ONE new src file + ONE new test file — no overlap):

| Leaf | Owns (src) | Owns (test) | Imports (stable) |
|------|-----------|-------------|------------------|
| **store** | `catalog/store.py` | `tests/test_catalog_store.py` | `catalog.types`, `attestation` |
| **graduation** | `catalog/graduation.py` | `tests/test_catalog_graduation.py` | `catalog.types`, `scoring.stats` |
| **differential** | `catalog/differential.py` | `tests/test_catalog_differential.py` | `catalog.types`, `scoring.stats` |

**The decoupling that keeps the leaves independent (DECOMPOSE_BY_SECRETS):** `store.py`
defines the event schema + the fold *mechanism* `derive_tiers(events, *, promote_fn,
demote_fn, classify_fn, rule_config)` taking the decision predicates as **injected
callables** — it never imports `graduation`/`differential`. The store's tests inject FAKE
predicates. `graduation.py` and `differential.py` provide the real predicates. The
**integrator** (coordinator) wires them in `__init__.py` + the CLI. So: store knows event
mechanics; graduation knows the lifecycle rules; differential knows the typology stats;
none import each other.

Integration (coordinator, after leaves green):
- `catalog/__init__.py` exports + the default wiring (`default_promote_fn`, etc.).
- `cycle.run_diagnostic` / `cli run` → append a RUN event to the store after a cycle.
- `cli catalog {list|show|graduate}` subcommand. Operator chrome → STDERR, machine JSON →
  STDOUT (Stage-C convention).

## Build laws (greenfield, non-negotiable)

1. **Exclusive new-file ownership** — a leaf edits ONLY its two files. No leaf edits
   `types.py`, `stats.py`, `cycle.py`, `cli.py`, or another leaf's file.
2. **Contracts-only deps** — leaves import only `catalog.types` + the named stable
   primitives. No leaf imports another leaf.
3. **Test-first RED→GREEN** — write the failing test first, then the impl. Targeted test
   only (`uv run --reinstall-package ai-crucible pytest tests/test_catalog_<x>.py`); the
   coordinator runs the ONE serial `verify.sh`.
4. **Determinism / PIN_PER_STEP** — no `datetime.now()`/`random` inside pure logic; inject
   timestamps + nonces (default to a clock at the edge). Tests pin them. No network.
5. **Structured errors** — `[CODE] message (hint: ...)` house shape, like every loader.
6. **Public surfaces are coordinator-authored** — leaves do NOT touch README / CHANGELOG /
   SCORECARD / docs / `--help` prose (the coordinator writes those).
7. **CI-green ≠ host-green** — the coordinator gates on CI after push and owns any
   cross-platform fix (a Windows-green sandbox test was Linux-red on Phase-1).

---

## Standards compliance (the six — workflow-standards.md)

- **PIN_PER_STEP — 3 (target):** every RUN/TRANSITION pins `rule_version` + `schema_version`
  + the deterministic ids; pure decision fns take injected clock/nonce so a fold is
  byte-replayable. Leaves prove it (deterministic-id + fold-idempotency tests).
- **ANDON_AUTHORITY — 2:** a malformed event / broken chain HALTS the fold loud
  (`verify_hash_chain` False → structured error; an unknown `reason_code` raises). Bad tier
  state never silently propagates; `catalog verify` is the andon check.
- **NAMED_COMPENSATORS — 2 (table below; NO skip — the store performs durable writes):**

  | action | command-to-undo | post-rollback state | owner |
  |--------|-----------------|---------------------|-------|
  | append RUN/TRANSITION event to JSONL | append-only; logical undo = append a compensating MANUAL transition — never edit/delete a line | the log carries an explicit reversing transition; tier re-folds; history preserved (demote-not-delete ethos) | `catalog.store` |
  | acquire single-writer lock file (`O_CREAT\|O_EXCL`) | release on clean exit / context-manager `__exit__` | lock file removed; next writer proceeds | `catalog.store` |
  | (Phase 10) `gh release create` / PyPI+npm OIDC publish | `gh release delete` + yank; **immutable — see release-ordering rule** | release withdrawn; tag remains | coordinator (Phase 10) |

- **DECOMPOSE_BY_SECRETS — 3:** store ↔ graduation ↔ differential decoupled via injected
  decision fns (above); the grading secret never enters the catalog (it records outcomes,
  not oracles).
- **UNCERTAINTY_GATED_HUMANS — 3:** the DEFER_TO_DESIGNER path IS an uncertainty-gated
  human checkpoint (gates on panel sub-quorum/escalate, not step count);
  INCONCLUSIVE_UNDERPOWERED surfaces underpowered differentials instead of forcing a noisy
  call.
- **EXTERNAL_VERIFIER — 3:** graduation's fairness clause consumes a CROSS-FAMILY panel
  (generator family excluded); the differential's cohort rate is the cross-family solve
  anchor. Claude never self-certifies graduation/typology. The graduation math + the
  differential logic are themselves cross-family-verified before commit (Ollama Cloud,
  ≥2 families) per `cross-family-cloud-verification`.
