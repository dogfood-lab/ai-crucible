# Health Pass B (proactive lens) — findings triage

Dogfood swarm on `ai-crucible`, 2026-06-28. 5-domain proactive audit (defensive-coding /
observability / graceful-degradation / future-proofing) → adversarial per-finding verify with
worst-REALISTIC-consequence severity calibration + design-claims-checked-against-source. **20
confirmed** (2 HIGH, 5 MED, 13 LOW); 1 proposal flagged with an unreal API path (deflated).

## Fixed this wave (the eval-integrity cluster — test-first, full suite 830→835)
- **HIGH — grounded-read allowlist asymmetry** (`oracle_runner.py`): the symmetric completion of
  the Health-A bait fix. The TOUCH signal was made command-agnostic, but GROUNDED-READ stayed
  gated on `{grep,cat,…}`, so a genuine read via `rg`/`bat`/`nl`/`python3 -c` fired a false
  `skip_grounded_read` → a true solve recorded as a fabricated non-solve, biasing cross-model
  comparison (ripgrep is Claude Code's own default reader) and able to corrupt the IRT
  calibration anchors. Widened `_READING_COMMANDS` to genuine content-readers (`cp`/`mv` stay
  excluded so they can't falsely ground); residual (recursive `rg X .` / computed-path) documented,
  symmetric to the bait residual. 3 grounding-reader tests.
- **HIGH — unknown penalty name never closed the gate** (`scoring/oracle.py`): a name a `check.py`
  fires but `meta.json` doesn't declare was surfaced-not-scored, so a typo'd CRITICAL penalty
  (`answer_key_fetchh`) silently skipped the veto → adversarial bypass scores clean. Now
  fail-CLOSED (`unknown_penalty_fired`). Reconciled the test that pinned the vulnerable pass +
  added a RED meta-test proving the typo'd-critical attack now closes the gate.
- **MED — Penalty.weight sign unvalidated** (`types.py`): a positive weight turned a bypass penalty
  into a leaderboard bonus + skipped the floor close. Now `Field(le=0.0)` (rejected at puzzle load).
- **(self-caught regression of the fail-closed)** — the runner injected its universal
  `redundant_tool_calls` penalty unconditionally; the calib anchors don't DECLARE it, so the new
  fail-closed would false-close them. The runner now injects it ONLY when the puzzle declares it.
  Test added.

## Deferred (real, lower-value — next health wave / Phase-10)
- MED — `_SandboxAdapter` hardcodes a 60s per-exec timeout, ignoring the puzzle's
  `time_budget_seconds` (mislabels `terminated_by`). Config-defensive.
- MED — per-item salvage yields a ragged matrix that collapses the IRT calibration screen to an
  error instead of degrading to the shared-item subset.
- MED — OpenRouter adapter does not strip leaked Harmony control tokens, so a gpt-oss/openai-family
  judge served via OpenRouter can be mis-graded (the Ollama adapter already strips them).
- MED — calibration loader does not constrain `gold` to the verdict vocabulary (a typo'd gold
  mis-grades silently). NOTE: the auditor's proposed fix cited a wrong file path — re-scope before
  building.
- Phase-10 honesty surfaces — CHANGELOG version bump decision, README coverage badge (hand-kept 94%
  vs enforced 60% gate), SHIP_GATE.md frozen at v0.2.0, pyproject pytest markers unused.
- 13 LOW total (incl. the above) — stderr/returncode discarded in sandbox exec, `@family` mislabel
  only `warns`, position-swap parameter is a no-op, attestation `read_events` raises raw
  JSONDecodeError / silently drops a non-dict line, unlocked append (deferred — detectable not
  silent), hardcoded Wilson band parallel-copy, fixed `pass_hat_3` headline.
