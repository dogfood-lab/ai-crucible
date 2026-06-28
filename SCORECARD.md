# Scorecard

> Pre/post assessment for the full treatment (2026-06-01), refreshed for the v0.3.0
> dogfood swarm (2026-06-21) and the v0.4.0 OpenRouter-quorum swarm (2026-06-28).

**Repo:** dogfood-lab/ai-crucible
**Date:** 2026-06-01 (treatment) · 2026-06-21 (v0.3.0 refresh) · 2026-06-28 (v0.4.0 refresh)
**Type tags:** `[all]` `[pypi]` `[cli]` `[npm]` (Python library/framework that also ships a thin `ai-crucible` CLI + an `@dogfood-lab/ai-crucible` npm launcher — corrected 2026-06-21; not MCP/desktop/vsix)

## Pre-Remediation Assessment

| Category | Score | Notes |
|----------|-------|-------|
| A. Security | 8/10 | SECURITY.md + threat model present; no secrets/telemetry. Missing inline README threat-model paragraph. |
| B. Error Handling | 7/10 | Typed, coded exceptions with hints throughout; library-appropriate. |
| C. Operator Docs | 4/10 | README good, but no LICENSE and no CHANGELOG. |
| D. Shipping Hygiene | 5/10 | CI + coverage gate + lockfile, clean build; but no `verify` script and no dependency scanning. |
| E. Identity (soft) | 1/10 | No logo, landing page, handbook, translations, or repo metadata. |
| **Overall** | **25/50** | Solid engineering substrate; unshipped polish. |

## Key Gaps

1. No LICENSE / CHANGELOG (hard gate C).
2. No `verify` script; no CI dependency scanning (hard gate D).
3. No README threat-model paragraph (hard gate A).
4. Soft-gate identity entirely absent (logo, landing page, handbook, translations, metadata).

## Remediation Priority

| Priority | Item | Estimated effort |
|----------|------|-----------------|
| 1 | LICENSE + CHANGELOG + verify.sh + README threat-model para (hard gates A/C/D) | small |
| 2 | CI `pip-audit` dependency scan (hard gate D) | small |
| 3 | Logo + landing page + handbook + translations + metadata (soft gate E) | medium |

## Post-Remediation

| Category | Before | After |
|----------|--------|-------|
| A. Security | 8/10 | 10/10 |
| B. Error Handling | 7/10 | 9/10 |
| C. Operator Docs | 4/10 | 10/10 |
| D. Shipping Hygiene | 5/10 | 9/10 |
| E. Identity (soft) | 1/10 | 9/10 |
| **Overall** | 25/50 | **47/50** |

Hard gates A–D: **all pass** (checked or SKIP-with-reason). Residual points: D's "version-matches-tag" is deferred (treatment is *minus publishing*, no tag cut) and "automated dep updates" is SKIPped per the org's GitHub Actions rule (no Dependabot unless requested); E reflects an honest pre-1.0 research instrument.

## v0.3.0 dogfood swarm refresh (2026-06-21)

`npx @mcptoolshop/shipcheck audit` → **100%** (21/21 checked, 0 unchecked) — the two prior
PARTIAL/APPLICABLE items closed (the CLI `--debug` opt-in + one-line structured-error
contract; `npm pack --dry-run` clean). Suite **795 tests, 94.6% coverage**, ruff clean, CI
green on every push. This release adds the durable catalog + abstention-aware graduation +
the differential typology (Epic 4), the multi-model native-tool-call Solver protocol
(Finding B′), and the eval-awareness boundary gate — each cross-family-verified (the
graduation/differential math got a 5/5-correct review by 3 non-Claude cloud families) or
live-validated (gpt-oss solved the seed via native tool-calls). D's "version-matches-tag" is
now satisfied by the v0.3.0 release.

**Why still v0.x, not v1.0.0:** the judge-panel alt-test ω remains a circular model-jury
bootstrap (no ≥3 independent human annotators a one-human studio can staff), so seats are
provisional and graduation escalates to the Designer. Forcing v1.0.0 would overclaim a
pre-Phase-2 instrument; pre-1.0 is the honest version. This is disclosed, not faked.

## v0.4.0 OpenRouter-quorum swarm refresh (2026-06-28)

`npx @mcptoolshop/shipcheck audit` → **100%** (21/21 checked, 0 unchecked, 14 SKIP). Suite
**795 → 854** tests, **94%** coverage, ruff clean; CI green on Python 3.11 / 3.12 / **3.13** —
3.13 was added to the matrix this swarm (verified 854-green locally) so every Python version
advertised in the PyPI classifiers is exercised, not assumed. This release ships the first
**published** cross-family judge-admission run (`eval/RESULTS.md`: 2 local + 5 pinned-OpenRouter
seats, 7 disjoint families, 93 pairs, k=3, 1,395 paid calls, **zero** rate-limit drops), two
eval-integrity health passes (1 CRITICAL bait-evasion close + 4 HIGH, every fix test-first and
cross-family verified), the offline `ai-crucible labels validate` intake gate, and the
`ai-crucible calibration curate` harder-set pipeline (study-swarm Phase A). D's
"version-matches-tag" is satisfied by the v0.4.0 release.

**Why still v0.x:** the cross-family run admitted 3 disjoint families (up from 2 local-only), but
the composed panel still escalates (sub-quorum + ρ≈1.0 error-redundancy), and the alt-test ω
remains a circular model-jury bootstrap (no ≥3 independent human annotators). "3 admit" ≠ "ω
solved" — seats stay provisional, graduation defers, pre-1.0 stays the honest version. Disclosed,
not faked.
