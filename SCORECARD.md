# Scorecard

> Pre/post assessment for the full treatment (2026-06-01).

**Repo:** dogfood-lab/crucible
**Date:** 2026-06-01
**Type tags:** `[all]` `[pypi]` (Python library/framework — not CLI/MCP/desktop/npm)

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
