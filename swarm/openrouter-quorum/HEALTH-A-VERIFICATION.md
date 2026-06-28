# Health Pass A — cross-family verification receipt (EXTERNAL_VERIFIER)

Dogfood swarm on `ai-crucible`, 2026-06-28. The two load-bearing eval-integrity fixes were
verified by **3 disjoint non-Claude cloud families** (Ollama Cloud, server-side), generator's
reasoning withheld, per the EXTERNAL_VERIFIER standard. Diff under review:
`swarm/openrouter-quorum/load_bearing.diff`.

## Verifiers
| family | model | verdict |
|--------|-------|---------|
| qwen | `qwen3-coder:480b-cloud` | both fixes sound; flagged substring-FP (negligible) |
| deepseek | `deepseek-v3.1:671b-cloud` | both fixes sound; no new evasion/FP found |
| moonshot | `kimi-k2.6:cloud` | FIX2 sound; FIX1 closes reported evasion; raised base64/content-indirection (already documented) + malformed-quote (already handled) |

(`glm-4.6:cloud` was retired 2026-06-16 — dropped from the panel.)

## FIX 1 — answer-key bait evasion (CRITICAL, `oracle_runner.py` + both `check.py`)
- **3/3 confirm** the explicit-name evasions (`od`/`strings`/`cp`/`dd if=`/`python3 -c`) now
  register in `touched_paths` and fire `answer_key_fetch` → gate closes. Proven end-to-end by
  6 parametrized bypass tests against the real kernel+oracle pipeline.
- **Substring-match false-positive** (a token merely *containing* the bait name): rated
  negligible by all three — the bait is a **sealed random-hash secret**; an honest Solver never
  knows the name to embed it, and knowing it already implies a touch. Substring (not exact)
  match is retained deliberately — it is what catches the bait embedded inside a larger token
  (`python3 -c "...open('BAIT')..."`).
- **Residual (documented, deferred):** shell-glob (`cat *`) and content-indirection
  (computed/base64-decoded/stdin paths) still evade trace-token detection. Pre-existing limit,
  not a regression. Robust closure = host-side filesystem access tripwire (sandbox-provider
  cycle). The instrument treats a clean bait signal as necessary-but-not-sufficient.
- **Malformed-quote** (kimi): already handled — `_exec_tokens` degrades a `shlex` ValueError to
  a whitespace split, so the bait token survives. Locked by `exec_malformed_quote_bait`.

## FIX 2 — Wilson on a fabricated binomial count (HIGH, `profile.py` + `metrics.py`)
- **3/3 confirm** `n_eff = (Σw)²/Σw²` is the correct Kish effective sample size; `== n_items`
  under uniform weights (no-op on unweighted sets); edge cases handled (all-zero weights →
  `len(items)` fallback; single item; rounding clamped so Wilson's `0 ≤ successes ≤ n, n > 0`
  contract holds). No regressions.
- Noted methodological (not code) limits: discretization (double-rounding at tiny `n_eff`) and
  the Kish design-effect being a conventional *approximation* of the weighted-proportion CI —
  both inherent to using the integer-count Wilson interval; accepted.

## Net
Both fixes **CORRECT and verified cross-family**. Residuals documented and honestly scoped.
Local gate: `ruff` clean, full suite green (830 tests), bypass suite flake-stable (3×).
