# Stage A — Health Audit (bug / security / correctness / eval-integrity)

**Repo:** dogfood-lab/ai-crucible @ branch `dogfood-swarm` (save-point tag `swarm-baseline-2026-06-21`)
**Baseline:** 473 tests green, 91.5% coverage, build/smoke OK.
**Method:** 7-domain parallel Claude audit → cross-family Ollama Cloud verify (deepseek-v4-pro / qwen3-coder:480b / glm-5 / kimi-k2.6 / gpt-oss:120b, refute-by-default, model-fallback-guarded) → coordinator synthesis-against-source.
**Cross-family result:** 22/22 findings rated **real, 0 false positives**. Severity is the noisy axis — coordinator re-calibrated against source (the panel up-rated several single-reviewer MEDIUMs to HIGH; not authoritative without convergence).

## Coordinator-calibrated severities (synthesis-against-source)

### HIGH — eval-integrity (load-bearing; cheap fixes)
| ID | File | Issue | Verify |
|----|------|-------|--------|
| **scoring-stats-001** | observability.py:119-148 | `_attempt_solved` uses `score.value > 0`, ignoring the authoritative `gate_passed` metadata. A gate-PASSED attempt with net ≤ 0 (non-critical penalty ≥ solve+bonuses, or threshold-0 solve_quality=0) is silently counted UNSOLVED → undercounts pass^k / solve-rate / graduation. | 3/3 cloud HIGH + coordinator read |
| **kernel-core-001** | engagement.py:174-181 | `assert_no_chrome_leak` scans only `message['content']`; the Critic appends scored-context messages with text under `critique` (no `content` key) → chrome in a critique bypasses BOTH the role guard and the kernel post-Critic re-check. Fail-closed guard isn't fail-closed for a whole message field. | 3/3 cloud real (HIGH/HIGH/MED) + coordinator read |
| **family-exclusion cluster** (scoring-stats-002 + models-cli-001 + models-cli-002) | judge_panel.py:287-295, characterize/run.py:485-495, ollama_adapter.py:310-337 | EXTERNAL_VERIFIER same-family exclusion is fragile: exact-string case-sensitive `!=` (Claude≠claude); untagged judges stamped literal `"unknown"` (not None) so a same-family-but-untagged judge survives exclusion; served model never verified against requested (wrong family attribution). Together: a same-family judge can silently vote on the panel. | qwen+glm HIGH (deepseek correctly flagged the auditor's *example* as imperfect — real trigger is the mixed tagged/untagged same-family case) |

### MEDIUM
| ID | File | Issue |
|----|------|-------|
| kernel-core-002 | tests/test_kernel.py:520-584 | Post-Critic re-check test leaks via in-place `content` edit, not the Critic's real `critique`-field message → false assurance. This IS the test-first RED gate for kernel-core-001. |
| characterize-001 | characterize/profile.py:510-518 | Bias panel (position/verbosity/family-pref) is structurally inert in the default run yet reports a PASSING measured gate, not "not measured" — masks unmeasured judge bias. |
| calibration-instrument-001 | instrument/rubric_bundle.py:171-195 | `bump_on_change` is a non-enforcing suggester documented as an invariant ("seal that does not seal"). Real seal (compile_bundle content-hash) is sound → cloud deflated to LOW; coordinator: MEDIUM because it's mis-documented as enforcing. |
| calibration-instrument-002 | instrument/rubric_bundle.py:198-226 | `_next_version` mangles date-style labels (2026.06 → 2026.7) on a real rubric change. |
| models-cli-003 | characterize/run.py:220-231 | `ai-crucible characterize` exits 0 when every model fails / 0 judgments collected (silent success on total failure). |
| test-integrity-002 | characterize/run.py:124-153 | Real-run driver (collect_records/run_panel/main) untested incl. a measurement-biasing parse-fail branch (run.py 44% covered). |
| honesty-coverage (test-integrity-001 = ci-supply-docs-001) | README.md:13 | Badge claims **96%**; actual **91.5%**; CI floor only 60%. COORDINATOR-AUTHORED fix. |
| honesty-security (ci-supply-docs-002) | SECURITY.md:19-22 | §1 describes the Solver container as a shipped hardened Docker container; only a local-subprocess sandbox ships. Overclaim vs sandbox.py's own honest disclosure. COORDINATOR-AUTHORED fix. |
| honesty-shipgate (ci-supply-docs-003) | SHIP_GATE.md:8 / SCORECARD.md | Classifies repo as "not a CLI / not an npm package"; it ships both. COORDINATOR-AUTHORED fix. |

### LOW
| ID | File | Issue |
|----|------|-------|
| scoring-stats-003 | observability.py:151-163 + oracle.py | `grade()` never surfaces a `novelty_validated` metadata key → observability novelty-rate reads 0 off the oracle score. |
| scoring-stats-004 | observability.py:109-116 | pass^k k≤0 contract diverges: `stats.pass_hat_k` raises, `observability.aggregate_pass_hat_k` returns 1.0. |
| calibration-instrument-003 | characterize/run.py:322 | `_grade_matrix` majority-vote uses `>=0.5`, breaking ties toward 'correct' → can shift which calibration items survive IRT screen. |
| characterize-002 | characterize/run.py:251-258 | Default model-jury path applies κ floor + κ-z gate vs hardcoded `human_human_kappa=0.80` with no machine-readable provisional caveat (ties to circular-ω disclosure). |
| models-cli-004 | cli.py:17-18 | CLI usage banner hardcodes "v0.2.0" while `--version` reads package metadata (drift on next bump). |
| ci-supply-docs-004 | SHIP_GATE.md:54 | States built wheel is `crucible-0.2.0`; actual is `ai_crucible-0.2.0`. COORDINATOR-AUTHORED fix. |
| test-integrity-003 | tests/test_sandbox.py:256-315 | Symlink-escape security tests skip on Windows w/o SeCreateSymbolicLinkPrivilege (win32-skip = regression marker, not fix). |

## Amend wave plan (v2 mechanics — exclusive file ownership)
- **A (sealed-boundary):** engagement.py + tests/test_engagement.py + tests/test_kernel.py + tests/test_sandbox.py → kernel-core-001, 002, test-integrity-003.
- **B (scoring/observability):** scoring/oracle.py + scoring/judge_panel.py + observability.py + tests/test_scoring.py + tests/test_observability.py → scoring-stats-001, 002, 003, 004 (+ family-exclusion normalization/fail-closed half).
- **C (model adapters):** models/ollama_adapter.py + models/claude_adapter.py + tests/test_models.py → models-cli-002 (served-model verify, reuse `_norm` guard).
- **D (characterize+cli):** characterize/run.py + characterize/profile.py + cli.py + tests/test_characterize*.py + tests/test_cli.py → models-cli-001 ("unknown"→None family stamp), models-cli-003, 004, characterize-001, 002, calibration-instrument-003.
- **E (instrument):** instrument/rubric_bundle.py + tests/test_instrument.py → calibration-instrument-001, 002.
- **Coordinator (personal, NOT delegated):** honesty surfaces — README badge, SECURITY.md §1, SHIP_GATE.md, SCORECARD.md.

Discipline: per-finding test-first RED gate; 3 post-fix verifier lenses (contract-completeness / cross-boundary info-flow / invariant-test-completeness) + family-of-call-sites probe; build green (473+ tests) after the wave; coordinator-as-judge on scope.

---

## OUTCOME (closed 2026-06-21)

Amend wave: 5 fix agents, exclusive file ownership, **16/16 fixes red→green** (each proved RED against HEAD — incl. honest out-of-tree RED proofs via `git stash` and throwaway-copy neutering for files an agent did not own). All 22 findings closed (the 2 coverage-badge dups merged).

**Coordinator family-of-call-sites probe caught a missed sibling:** `characterize/metrics.py::family_pref_delta` compared `judged_family == own` raw (un-normalized) — the same scoring-stats-002 pattern in the kin-bias *metric* (not the exclusion gate). A casing drift would mask kin-bias. Fixed by the coordinator (metrics.py unowned by any amend agent) with a test-first RED→GREEN gate (`test_family_pref_delta_normalizes_family_casing`).

**Honesty surfaces (coordinator-authored, per LAW):**
- Coverage badge **96% → 93%** across all 8 README locales (actual).
- SECURITY.md §1 reconciled: states `LocalSandbox` (process isolation + workdir confinement, NOT a security sandbox) is what ships in Phase 1; the hardened container config is the *planned* provider; the out-of-band grading edge is a pluggable boundary (separate-host deployment is Phase 2).
- SHIP_GATE.md reclassified `[all][pypi][cli][npm]`; un-SKIPped now-live CLI rows (exit codes, `--help` — checked; raw-stack-traces flagged as a Stage-C item); wheel name `crucible-` → `ai_crucible-`; version-matches-tag row corrected (v0.2.0 tag exists).
- SCORECARD.md type-tags corrected.

**Build gate:** `bash verify.sh` GREEN — **473 → 520 tests** (+47), **91.5% → 93.01% coverage**, ruff clean, wheel+sdist build OK, smoke OK. Version held at 0.2.0 (research-preview; v1.0.0 minimum is a director-decided SKIP while ω is circular).

Clean bill of health for Stage A: **0 CRITICAL, 0 HIGH residual.** Proceed to Stage B (proactive health).
