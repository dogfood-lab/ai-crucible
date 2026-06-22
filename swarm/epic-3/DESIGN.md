# Epic 3 — Retire the Circular ω — Design (study-swarm-grounded)

**Goal:** run a real human-labeling round so the judge-panel alt-test ω is grounded in HUMANS, not the circular model-jury bootstrap — flipping provisional seats to defensible. Harness exists (human_labels.py, metrics.alt_test per Calderon 2025); this designs the round + the elicitation glue.

## Locked findings (citations in swarm/epic-3/study-swarm.json)
- **The alt-test does not require annotator INDEPENDENCE — it requires a representative human consensus** (leave-one-out paired t-test + BY-FDR). BUT correlated raters (shared context) bias ω + FDR toward FALSE confidence — the leverage of N raters is independent errors cancelling, destroyed by shared rationale (Calderon 2025 arXiv:2501.10970; Snow 2008). → **require ≥1 out-of-studio reviewer; tier internal-only crew "crowd" (ε=0.1), not expert.**
- **ε is set by annotator QUALITY, not panel size** (expert 0.2 / skilled 0.15 / crowd 0.1). Min 3 raters × 30 items is the FLOOR; target 4–5 × 50–100. Don't shrink ε for small N — report per-fold effect size + power; guard against "passed-but-every-fold-marginal" (Calderon 2025; BY 2001).
- **Elicitation: the EXPORTER owns randomization.** Persist a per-(annotator,item) `shown_order` (AB/BA); the ingester inverts it. A CSV the human can re-sort silently voids order-persistence → `shown_order` is a machine field the human never edits (Shi 2024 position bias arXiv:2406.07791; Eckman 2024 arXiv:2403.01208).
- **"unsure" is first-class + nearly free** (<3% chosen; tracks model-hard items) — keep A/B/unsure, validated on ingest; unsure/DISPUTED drop from ω (Eckman 2024; the loader already drops them).
- **Quality controls:** separate sessions / no shared rationale / no cross-visibility + a per-annotator isolation attestation; embedded known-ambiguous attention/gold-check items to catch inattentive raters; balance the gold marginal ~50/50; report **Gwet AC1 beside Krippendorff α** (α is prevalence-skew-fragile — the kappa paradox; Cook 2025 EffiARA arXiv:2504.00589; Wongpakaran 2013).
- **Item set:** the 28 IRT-discriminators are < the ≥30 floor → author/stratify a 50–100-item labeling set (include hard items above the 28 survivors), not just the discriminators.

## Glue to build (the code; tier-agnostic, safeguards baked in)
1. `label export` — emit gold-stripped items with a persisted per-(annotator,item) `shown_order`, A/B/unsure schema, embedded attention checks, + a round manifest (annotators, tier, isolation attestation fields).
2. `label ingest` — invert `shown_order`, validate A/B/unsure, drop disputed/unsure, run gold-check attention filter, compute Krippendorff α **+ Gwet AC1**, assemble `human_labels.json`.
3. Round runner — wire into `characterize --human-labels` (flips `alt_test_reference` off model-jury-bootstrap) + per-fold power/effect-size reporting + a clear internal-only-vs-external **caveat in the output** (ω is only as defensible as the panel's independence).

## The one director decision (gates the round's defensibility, not the code)
Who annotates + at which tier — see the AskUserQuestion. The glue supports all options; the choice sets the default tier + whether the resulting ω is fully defensible (≥1 external) or an honest internal pilot (caveated).
