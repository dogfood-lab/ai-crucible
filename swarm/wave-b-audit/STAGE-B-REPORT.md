# Stage B — Proactive Health Audit (defensive / degradation / observability / numerical / future-proofing)

**Repo:** dogfood-lab/ai-crucible @ `dogfood-swarm` (post Stage-A: 520 tests, 93.01% cov).
**Method:** 5-domain proactive audit → cross-family Ollama Cloud verify → coordinator synthesis-against-source.
**Cross-family result:** 22 found → **21 real, 1 false-positive** (kernel-runtime-004 dropped — terminating BUDGET before the 3-strike HARD_KILL window is *correct*, not a mislabel; cloud + coordinator agree). Severity is the noisy axis (the panel single-vote up-rated several MEDIUMs to HIGH — not authoritative; coordinator-calibrated below).

## Coordinator-calibrated → AMEND (graceful degradation / observability / defensive / numerical)

### HIGH (resilience crux — 3/3 cloud real)
| ID | File | Issue |
|----|------|-------|
| **kernel-runtime-001** | kernel.py:356-406 | Post-Solver pipeline (oracle grade + panel score + eval_log write) has NO failure handling — a grading-host blip or one judge error discards the whole completed attempt with no trace, no `terminated_by`, no diagnosable cause; in pass^k one sibling's grading failure aborts all *k*. → wrap in try/except, stamp ERROR + structured code/message/hint naming the failing stage, still render+persist the eval_log, return the degraded attempt; isolate per-sibling in run_pass_hat_k. |
| **scoring-numerics-001** (= kernel-runtime-002, deduped) | scoring/judge_panel.py:419 | `asyncio.gather` without `return_exceptions=True` — one flaky judge (Ollama timeout) cancels the whole panel, discarding every healthy verdict. → return_exceptions, reduce over survivors, record `judges_errored` in metadata, keep the empty/below-floor ValueError as the floor. |
| **characterize-degradation-001** | characterize/run.py:229-275,460-513 | Failed models vanish from the report — a partial panel is indistinguishable from a smaller-by-design one (silent measurement degradation). → surface failed models + reason in the output; mark the run degraded. |

### MEDIUM (amend)
- **scoring-numerics-003** (oracle.py:169) — gate accepts NaN/inf solve_quality/time_used: NaN passes every threshold compare and emits a non-finite Score.value with no failed condition (eval-integrity-adjacent). → reject non-finite as a failed condition.
- **scoring-numerics-002** (judge_panel.py:224) — median reducer over a NaN judge value returns an order-dependent NaN that poisons the panel score. → drop/flag non-finite before reduce.
- **kernel-runtime-003** (puzzle.py:107,146) — unbounded meta.json/prompt reads at the puzzle trust boundary (sandbox caps output at 1 MiB but the loader has no ceiling). → size guard + structured PuzzleLoadError.
- **characterize-degradation-002** (run.py:145-162) — a single item failure discards a model's entire transcript (no partial salvage). → salvage per-item.
- **characterize-degradation-004** (run.py:152-158) — per-item parse failures scored as wrong, never aggregated into a legible signal. → count + surface parse failures distinctly.
- **models-ollama-resilience-001** (ollama_adapter/claude_adapter) — no retry/backoff: a transient blip aborts a model's multi-hour run. → bounded retry/backoff.
- **models-ollama-resilience-002** (ollama_adapter) — daemon-down surfaces as bare httpx ConnectError, no operator hint. → structured error + hint.
- **models-ollama-resilience-003** (ollama_adapter) — malformed HTTP-200 body raises opaque JSONDecodeError; no mapping guard. → guard + structured error.
- **instrument-future-deps-001** (tuning.py:300) — sobol_screen masks a NaN objective to T_i=0.0, silently freezing a load-bearing weight. → surface NaN, don't freeze silently.
- **instrument-future-deps-002** (inspect_task.py) — inspect-ai pinned only `>=0.3`, no upper bound, no version assertion; Sample.model_dump consumed wholesale. → version assertion + documented compat note.

### LOW (amend, batched)
- kernel-runtime-005 (trace.py) — eval_log shape has no version tag → add `schema_version`.
- characterize-degradation-005 (run.py) — no per-item timeout around judge_item → bounded timeout.
- models-ollama-resilience-004 (ollama_adapter) — `_via_sdk` assumes dict()-convertible SDK response → guard.
- instrument-future-deps-003 (sut.py) — parse_sut_yaml silent duplicate-key last-wins → reject/warn.
- instrument-future-deps-004 (sut.py) — `_looks_like_family_alias` treats non-ASCII digits as valid version → ASCII-only guard.
- instrument-future-deps-005 (calibration/loader.py) — no size/file-count bound on operator dir → bound.

## DEFERRED to Feature Pass (with rationale, not dropped)
- **characterize-degradation-003** — checkpoint/resume of the multi-hour run. This is a genuine *feature* (durable progress + resume), not a spot-fix; it belongs in the feature pass (Phases 5-8) where it gets a designed contract, not a patch. Logged here so it is not lost.

## DROPPED
- **kernel-runtime-004** (false-positive) — BUDGET firing before HARD_KILL on a budget < 3-strike window is correct termination, not a mislabel.

## Amend clusters (v2 mechanics — exclusive file ownership)
- **A:** kernel.py + trace.py + puzzle.py + tests/test_kernel.py + tests/test_puzzle.py → kernel-runtime-001, 003, 005.
- **B:** scoring/judge_panel.py + scoring/oracle.py + tests/test_scoring.py → scoring-numerics-001(=kernel-runtime-002), 002, 003.
- **D:** characterize/run.py + tests/test_characterize_run.py → characterize-degradation-001, 002, 004, 005.
- **E:** models/ollama_adapter.py + models/claude_adapter.py + tests/test_models.py → models-ollama-resilience-001, 002, 003, 004.
- **F:** instrument/tuning.py + inspect_task.py + sut.py + calibration/loader.py + tests/test_instrument.py + tests/test_calibration.py → instrument-future-deps-001, 002, 003, 004, 005.

---

## OUTCOME (closed 2026-06-21)

5 fix agents, exclusive file ownership, **19/19 fixes red→green** (each test SIMULATES the failure — a raising oracle_runner/judge, NaN value, oversized file, daemon-down/malformed-body stub, hung call — and asserts the graceful outcome). Coordinator added the inspect-ai `<1` pin (Agent F's out-of-set recommendation) + re-locked.

**Cross-stage catch (test-first discipline working):** Agent A's first grading-wrap cut swallowed the Critic-turn `ChromeAccessError` → would have silently undone the Stage-A sealed-boundary fix (a contaminated attempt degraded instead of halting). Caught by the pre-existing Stage-A test going RED; fixed by re-raising `SealedBoundaryViolation`/`ChromeAccessError` unwrapped at both the stage and outer catch. A contaminated attempt halts; everything else degrades.

**Resilience now in place:** kernel post-Solver tail degrades to a traced ERROR attempt naming the failing stage (Solver transcript + eval_log preserved); pass^k isolates per-sibling grading failures; JudgePanel degrades over surviving judges (`judges_errored` metadata) with a zero-survivor floor; oracle rejects non-finite solve_quality/time_used; median reducer drops non-finite; puzzle loader size-guards the trust boundary; characterize surfaces failed models + salvages partial transcripts + bounds per-item time; model adapters get bounded retry/backoff + structured daemon-down/malformed-body/SDK-shape errors; sobol surfaces NaN; inspect-ai version drift warns at runtime + caps at <1; SUT rejects duplicate keys + non-ASCII versions; calibration loader bounds dir size; eval_log carries schema_version.

**Build gate:** `verify.sh` GREEN — **520 → 576 tests** (+56), **93.01% → 94.50% coverage**, ruff clean, build+smoke OK.

**Deferred to feature pass:** characterize-degradation-003 (checkpoint/resume — a designed durability feature, not a patch). **Dropped:** kernel-runtime-004 (FP). **Noted non-gating:** a pre-existing mypy nit in judge_panel.py (the 'weighted' reducer Score(value=object)) — mypy is not a CI/verify gate; left for a future typing cleanup.

Stage B closed: **0 HIGH residual.** Proceed to Stage C (behavioral humanization).
