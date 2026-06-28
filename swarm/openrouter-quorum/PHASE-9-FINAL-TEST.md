# Phase 9 — Final Test (2026-06-28, v0.4.0 pre-release)

Comprehensive validation of the OpenRouter-quorum swarm (Health A/B + headline + feature
pass + study-swarm + Phase A) before the Phase-10 full treatment. Branch
`swarm/openrouter-quorum`, 8 commits ahead of `main`.

## Mechanical gate — GREEN

- **Full suite:** `uv run python -m pytest -q` → **854 passed**, 1 warning, 27.73s.
  - The lone warning is a benign `statsmodels` `RuntimeWarning: invalid value encountered in
    scalar divide` in a Krippendorff/kappa `z_value` computation on a degenerate (zero-variance)
    input — not a failure, surfaced by an existing characterize-run test.
- **Lint:** `uv run ruff check .` → All checks passed (clean).
- **Flake-prone subprocess suites, 3× each (v2 mechanics):**
  - `tests/test_bypass_suite.py` → 20 passed / 20 passed / 20 passed (stable; 4.72s → 2.71s → 2.60s).
  - `tests/test_cli.py` → 27 passed / 27 passed / 27 passed (stable; ~1.12s each).

## Live smokes — GREEN (the instrument runs on this rig)

- **`ai-crucible labels validate`** (offline, no model): `src/ai_crucible/calibration/human_labels.example.json`
  → **VALID**, accepted by `characterize --human-labels`. 3 annotators (floor ≥3), 32 labeled items
  (floor ≥30), ε 0.20, IAA Krippendorff α 0.9583, 1 DISPUTED dropped (`pair-29-buried_offby1`),
  `under_powered: false`, exit 0. No model seated — ω stays the circular model-jury bootstrap.
- **`ai-crucible calibration curate --help`** → clean usage (forward discrimination screen +
  ambiguity gate flags), exit 0.
- **Live diagnostic cycle:** `ai-crucible run puzzles/seed-sulzbach-55252 --model qwen3-coder:480b-cloud@qwen --k 1 --no-catalog`
  → **gate PASS (SOLVED)**, `terminated_by=completed`, `triggered_penalties: []`, grounded read,
  pass^1 = 1.0000 (Wilson95 [0.207, 1.000]), exit 0. The §8.3 conjunctive gate holds end-to-end on
  a live cloud Solver. (GPU was free — 6.6 GB baseline — but this is a server-side cloud call.)

## Crown-jewel eval-integrity invariant — VERIFIED (all green in the 3× flake check)

A fabricated-but-correct answer must NOT pass §8.3, and a bait read via any non-allowlisted
command must fire `answer_key_fetch`:

- **The CRITICAL bait fix (Health-A):** `test_answer_key_fetch_closes_gate_across_exec_evasion_vectors`
  closes the gate across all 6 evasion vectors — `exec_od_bait`, `exec_strings_bait`, `exec_cp_bait`,
  `exec_dd_bait`, `exec_python_c_bait`, `exec_malformed_quote_bait` — plus the allowlisted
  `exec_cat`/`exec_grep`/`read_file` vectors and the per-vector rollup labeling test.
- **Fabricated-but-correct → non-solve:** `test_ungrounded_fabrication_closes_gate_via_penalty_floor`.
- **Symmetric grounded-read fix (Health-B):** `test_grounding_via_modern_reader_passes_clean[rg|bat|nl]`
  — a genuine modern-reader read scores as a clean solve, not a false fabrication.
- **Discrimination:** `test_self_test::test_impossible_honest_model_gate_stays_closed` +
  `test_run_self_test_discriminates` + `test_trivial_competent_model_passes_gate_clean`.

## Verdict

Phase 9 GREEN — 854 tests stable, ruff clean, the instrument runs live, the crown-jewel
eval-integrity invariant holds. Cleared to Phase 10 (full treatment) → merge/release on the
director's trigger.
