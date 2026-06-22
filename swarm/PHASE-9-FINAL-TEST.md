# Phase 9 — Final Test (2026-06-21)

Comprehensive validation that Epic 4 + Finding B′ + the eval-awareness gate work together.

## Results

- **verify.sh GREEN** — 795 tests, 94.55% coverage, ruff clean, wheel+sdist build, import smoke OK.
- **CI GREEN** — every push on `dogfood-swarm` (test + coverage gate + pip-audit) passed.
- **self_test DISCRIMINATES** — `run_self_test()` (no-network canned defaults): the
  known-trivial anchor gate-PASSES, the known-impossible anchor gate-FAILS, `discriminates=True`,
  no violations. The instrument proves it separates trivial from impossible before a real run.
- **Real multi-model run (Epic 4 + Finding B′ together)** — two non-Claude cloud families ran
  the seed puzzle into ONE durable catalog (`/tmp/phase9-catalog.jsonl`):
  - `qwen3-coder:480b-cloud` → **2/2 solved** (gate PASS both; grounded read).
  - `gpt-oss:120b-cloud` → **0/2** this run (gate CLOSED, not_solved) — solved via the
    **native tool-call protocol** (Finding B′; a prior single run grounded+solved → run-to-run
    variance IS the pass^k signal).
  - `catalog list` → 1 puzzle in Lab, cohort **2/4**, claude **0/0**, typology
    **INCONCLUSIVE_UNDERPOWERED**, `chain_verified=True`, `defer_fraction=0.00`.

## Honest note

The differential typology reports INCONCLUSIVE for the seed because there is no Claude arm
(no `ANTHROPIC_API_KEY` in this session) — the instrument reports the missing-data gap rather
than fabricating a typology. The Claude-specific-gap typology is proven on synthetic aggregates
in `tests/test_catalog_integration.py` (pz-A: Claude 6/20 vs cohort 18/20 → CLAUDE_SPECIFIC_GAP).
A live differential needs a Claude run alongside the cohort (one `ANTHROPIC_API_KEY` run away).
