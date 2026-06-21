# Stage C — Behavioral Humanization (operator UX)

**Repo:** dogfood-lab/ai-crucible @ `dogfood-swarm` (post Stage-B: 576 tests, 94.5% cov).
**Scope:** operator UX for a headless library + thin CLI (no GUI). Stage D (visual polish) has no rendered surface here — its substance (landing page / handbook / README visual identity) is Phase 10.
**Method:** 3-agent operator-UX audit → cross-family verify (11/11 real, 0 FP) → 3-agent amend (test-first) + coordinator follow-up.

## Closed (11 findings + 1 coordinator follow-up)
- **CLI error wrapper (the #1 fix)** — `cli.main()` now renders an escaping exception as a SINGLE structured `[CODE] … (hint: …)` stderr line + clean exit 1 (operator-input errors like a bad `--items` path no longer dump a raw traceback); full traceback is opt-in via `--debug`/`-v`; `KeyboardInterrupt` → clean 130. Mirrors the kernel's operator-vs-developer contrast (sealed-boundary andons still propagate unwrapped). [cli-operator-001, error-hint-sweep-001]
- **Terminal honesty** — the run now PRINTS (to stderr, leaving the JSON/file report machine-clean) the load-bearing PROVISIONAL caveat (circular-ω, all seats provisional, sub-quorum escalates to the Claude Designer), the DEGRADED-run notice (failed models), and the per-model seat/screen/reject WHY — so an operator can't read the seated list as authoritative. [characterize-result-legibility-001/002/005, error-hint-sweep-002]
- **Run announcement + progress** — up-front "N models × M items × k, this may take a while" + per-model progress/elapsed during the multi-hour run (was silent). [characterize-result-legibility-003/004]
- **Exit-code discoverability** — `--help` documents the 0/1/2 contract + the `--debug` flag. [cli-operator-002, error-hint-sweep-004]
- **Error-code consistency** — `SealedBoundaryViolation` + `ChromeAccessError` now carry stable `[SEALED_BOUNDARY_LEAK]` / `[CHROME_ACCESS]` codes + actionable hints (types/control-flow unchanged — the kernel re-raises them unwrapped by design). [error-hint-sweep-003]
- **Coordinator follow-up (pre-existing bug Agent A flagged):** `ai-crucible --help` crashed on a stock Windows cp1252 console (`UnicodeEncodeError` on the banner `ω`). `_ensure_utf8_streams()` reconfigures stdout/stderr to UTF-8 at CLI entry (guarded, best-effort). RED proven via a strict-cp1252 `TextIOWrapper` integration test; GREEN after.

## Build gate
`verify.sh` GREEN — **576 → 594 tests** (+18), **94.5% → 94.51% coverage**, ruff clean, build+smoke OK. 1 pre-existing benign statsmodels RuntimeWarning (tiny synthetic records; not introduced, not gating).

**HEALTH PASS COMPLETE (A + B + C).** 0 CRIT / 0 HIGH residual; correct, resilient, and honest-to-the-operator. Stage D → Phase 10. Next: feature audit (user review gate before any feature code).
