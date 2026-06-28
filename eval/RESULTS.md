# AI Crucible — Evaluation Results

Published runs of the judge-admission instrument. Each section is a dated, reproducible run with
its committed artifacts. Numbers here are **provisional** in the precise sense the instrument
stamps on every report: the judge-admission alt-test ω is a **circular model-jury bootstrap**, not
a human-grounded substitution test, so seat decisions are PROVISIONAL until a human-labeling round
exists. See the caveat block in each section.

---

## Cross-family OpenRouter quorum (2026-06-28)

**Artifacts:** [`eval/panel.json`](panel.json) (committed seated panel) · [`eval/cross-family-quorum-2026-06-28.json`](cross-family-quorum-2026-06-28.json) (full characterization report)
**Command:**
```
uv run python -m ai_crucible.characterize.run \
  --models gemma4:31b@gemma granite4.1:30b@granite \
    openrouter:deepseek/deepseek-v3.2@deepseek \
    openrouter:cohere/command-a-03-2025@cohere \
    openrouter:meta-llama/llama-3.3-70b-instruct@meta \
    openrouter:qwen/qwen3.7-plus@qwen \
    openrouter:nvidia/nemotron-3-super-120b-a12b@nvidia \
  --k 3 --out eval/cross-family-quorum-2026-06-28.json --write-panel eval/panel.json
```
7 seats (2 local + 5 pinned OpenRouter), 7 genuinely disjoint families, 93 admission pairs, k=3.
Run was **not degraded** — every seat returned the full 279 records, 0 unparsed, 0 dropped (the
5 OpenRouter seats made 1,395 pinned/paid calls with **zero** rate-limit drops).

### The question

Can a **broader cross-family pool** get **≥3 disjoint-family judges** through the 6-metric
judge-admission test — chipping at the ≥3-seat quorum gap the local-only panel could not clear
(only 2/6 local models cleared the de-saturated bar)? This is the **QUORUM** question. It does
**not** retire ω: cloud/cross-family judges reduce self-circularity, not human-validity, so the
Calderon (2025) ≥3-independent-humans gate stays unmet. "≥3 seated" is **not** "ω solved."

### Verdict: quorum NOT met (`escalate: true`, `meets_quorum: false`) — but the pool moved the needle

The broader pool **admitted 3 disjoint-family judges** (up from 2 local-only), but the **composed
independent panel is still 2** after the submodularity ρ-prune, which is below the PoLL ≥3 quorum.
The blocker has **shifted** — from "too few seat-worthy judges" to "the 3rd seat is error-redundant
with an existing one, and 4 strong judges are screened only by the circular ω."

### The panel (per-judge admission)

| judge | family | accuracy | quality | held-out ECE | alt-test ω | decision |
|---|---|---:|---:|---:|---:|---|
| `gemma4:31b` (local) | gemma | 0.985 | 0.986 | 0.011 | 0.67 | **SEAT** |
| `qwen/qwen3.7-plus` (OR) | qwen | **1.000** | **1.000** | ~0.00004 | **1.00** | **SEAT** |
| `nvidia/nemotron-3-super-120b` (OR) | nvidia | 0.985 | 0.986 | ~1e-6 | 0.67 | **SEAT** |
| `granite4.1:30b` (local) | granite | 0.960 | 0.954 | 0.044 | 0.33 | SCREEN — ω<0.5 |
| `cohere/command-a-03-2025` (OR) | cohere | 0.927 | 0.926 | n/a | 0.33 | SCREEN — ω<0.5 |
| `meta-llama/llama-3.3-70b-instruct` (OR) | meta | 0.927 | 0.920 | 0.093 | 0.33 | SCREEN — ω<0.5 |
| `deepseek/deepseek-v3.2` (OR) | deepseek | 0.914 | 0.837 | 0.230 | 0.00 | SCREEN — ω<0.5 |

Every model carries a `review_flag` (super-consistent κ vs. the hardcoded human baseline — a flag,
never a downgrade, Han 2025 Tier-1B). Perturbation audit `max_flip_rate = 0.0`: no seat/screen
decision flips under ±1 SE threshold jitter, so the decisions are **robust within their own
threshold noise** (not knife-edge).

### Composed panel: 3 admitted → 2 seated (sub-quorum)

```
seated (ρ-pruned, reliability-weighted): qwen/qwen3.7-plus (1.00), gemma4:31b (0.99)
dropped as ρ-redundant:                   nvidia/nemotron-3-super-120b  (ρ=1.00 with gemma4:31b)
quorum 2/3: NOT MET → escalate to the Claude Designer (too few independent judges)
```

The submodularity gate (ρ < 0.25, Codex-Verify 2025) drops `nemotron` because its **error vector
is identical to `gemma4`'s (ρ = 1.0)** — the two near-perfect judges erred on the *same* items, so
seating both adds no independent signal. The gate keeps the higher-reliability `gemma4` and the
lone genuinely-uncorrelated `qwen`.

### Why the quorum is still short — the honest reading

1. **The pool is good enough; the gates are the limit.** All six non-perfect judges score
   accuracy ≥ 0.91. The four that screen are screened **solely on the circular model-jury ω < 0.5**
   — their accuracy (0.91–0.96) and quality (0.84–0.95) clear every other gate. So the quorum
   bottleneck is the **un-validated ω axis**, not judge capability. This is the on-ice constraint
   made visible: a broader, stronger pool does not lift the quorum while the screening gate is a
   circular bootstrap.
2. **The discriminating signal is thin.** The IRT screen keeps only **11 of 93 items** (82 saturate
   — every judge agrees) for this panel of strong frontier judges. So both the ρ=1.0 `gemma↔nvidia`
   correlation and the seat/screen margins rest on a small discriminating subset; the set is near
   its ceiling for judges this strong.
3. **One "independence" is partly an artifact.** `qwen/qwen3.7-plus` correlates ρ=0.0 with every
   peer **because it made zero errors** — a zero error-vector correlates with nothing. Its
   independence is real for the panel math but is a ceiling effect, not demonstrated error-diversity.
4. **Seats are jury-relative.** `granite4.1:30b` was SEAT in the prior local-only run and SCREENs
   here — its leave-one-out ω fell to 0.33 against the broader jury. Seat decisions depend on panel
   composition because ω is a leave-one-out bootstrap over the seated population: changing the pool
   changes who seats. This is a property (and a limitation) of the circular ω.

### ⚠ Caveat block — ω is still on ice; seats are PROVISIONAL (load-bearing, never faked)

- The alt-test ω here is a **CIRCULAR model-jury bootstrap**: the reference "annotators" are the
  other panel models, drawn from the same population being seated. The κ baseline is **hardcoded
  (0.80), not human-measured**. Both are stamped PROVISIONAL in the report (`alt_test_caveat`,
  `kappa_baseline.provisional = true`).
- A valid alt-test needs **≥3 independent human annotators on ≥30 items** (Calderon, Reichart &
  Dror 2025, arXiv:2501.10970). A one-human studio cannot staff that without correlated-annotator
  bias, so **ω stays on ice by structural constraint**; the below-quorum panel **escalates to the
  Claude Designer** rather than seating thin — disclosed, never faked.
- Adding cross-family OpenRouter seats **reduces self-circularity** (more genuinely-different
  families in the jury) but does **not** establish human-validity. **"3 admitted" ≠ "ω solved."**
- Therefore every seat above is provisional, and **Arena graduation's cross-family fairness judge is
  not yet seatable**: the composed panel still escalates, so real-run graduation correctly DEFERS
  every puzzle to the Designer rather than faking a promotion.

### What changed vs. the local-only run

| | local-only (6 models) | + cross-family OpenRouter (this run) |
|---|---|---|
| seat-worthy admitted | 2 (gemma, granite) | **3** (gemma, qwen, nvidia) |
| composed independent panel | 2 (gemma, granite) | 2 (qwen, gemma) — nvidia ρ-dropped |
| quorum (≥3) | NOT met → escalate | NOT met → escalate |
| new cross-family seat | — | **`qwen3.7-plus`** (acc 1.0, ω 1.0) |
| binding blocker | too few clear the bar | circular-ω screens 4 strong judges + error-redundancy drops 1 |

**Bottom line:** the cross-family pool **seated a 3rd disjoint family** (a real milestone for the
admission question) and shifted the quorum blocker off "judge quality" — but the composed panel
still escalates, and the result does **not** move ω off the ice. The next lever for the quorum is
either (a) a harder, less-saturated discriminating set so error-diversity is measurable on more than
11 items, or (b) the banked independent cross-family **cloud ω-anchor jury** (disjoint from the
seats) to reduce — not remove — the ω circularity. Human validity still requires the human round
the studio's structure can't staff.

## Live ρ-curation demonstration (2026-06-28)

**Artifacts:** [`eval/rho-demo-full-2026-06-28.json`](rho-demo-full-2026-06-28.json) (full-set,
grade-matrix-bearing) · [`eval/rho-demo-curated-2026-06-28.json`](rho-demo-curated-2026-06-28.json)
(curated re-run) · [`eval/rho-demo-harder-subset-2026-06-28.json`](rho-demo-harder-subset-2026-06-28.json)
(the curated discriminating subset, re-usable as `characterize --items`).

The bottom line above named lever **(a)** — *a harder, less-saturated discriminating set so
error-diversity is measurable*. This run **tests** that lever end-to-end with the Phase-A curation
pipeline: a fresh grade-matrix-bearing `characterize` run → `ai-crucible calibration curate` → a
re-run on just the discriminators → compare the inter-judge error-correlation ρ.

### Setup

A fast 4-seat disjoint-family **cloud** panel (`qwen3-coder:480b@qwen`, `glm-4.6@zhipu`,
`gpt-oss:120b@openai`, `gemini-3-flash-preview@google`), full 93 admission pairs, k=3. One seat
(`glm-4.6:cloud`) dropped mid-run on a `410 Gone` — the volatile-cloud-roster reality — leaving 3
graded judges. (A different, faster panel than the headline run above; the goal here is the curation
*mechanism* + the ρ direction, not a new seating verdict.)

### Result — the redundancy is a saturation artifact, and the bank is the bottleneck

| metric | full set (93 items) | curated (3 discriminators) |
| --- | --- | --- |
| discriminating items | **3 of 91** kept (88 saturated) | 3 (the kept set) |
| ρ(gpt-oss, qwen) | **+0.42** (> 0.25 ceiling) | **−0.50** |
| ρ(gemini, gpt-oss) / ρ(gemini, qwen) | 0.0 / 0.0 | 0.0 / 0.0 |
| judge accuracy (qwen / gpt-oss / gemini) | 0.97 / 1.00 / 1.00 | 0.64 / 0.74 / 1.00 |
| seats → verdict | 2 (gemini, gpt-oss) → escalate | 0 → escalate |

1. **The full-set redundancy does NOT survive curation.** The one above-ceiling pair (gpt-oss↔qwen,
   ρ=+0.42 on the saturated 93-set) flips to ρ=−0.50 on the 3 discriminators — the positive
   error-correlation was a **saturation artifact** (two strong judges co-passing the easy items),
   exactly the Kohli 2026 (arXiv:2605.29800) "effective votes collapse on saturated sets" prediction.
   The redundancy lived in the *bank*, not the *judges*.
2. **The discrimination screen is brutal: 3 of 91.** Even more saturated than the headline's 11/93,
   because these frontier cloud judges score acc 0.97–1.00 — almost everything is a co-pass. On the 3
   hard items accuracy collapses (qwen 0.97→0.64, gpt-oss 1.00→0.74): they genuinely separate the
   panel, which is *why* they survive the screen.
3. **The curated ρ is not a measurement.** N=3 — one item flips |ρ| between 0 and 1, so −0.50 carries
   no signal beyond "≠ +0.42". You cannot decorrelate a panel by sub-setting a saturated bank; you
   run out of items first. **That is the finding: Phase B — new harder-construct content — is the
   load-bearing lever, not re-screening the existing 93 pairs.** The pipeline confirms the
   study-swarm's central claim empirically.

### Dogfood win — the demo caught a real bug (fixed in this release)

`ai-crucible calibration curate` **crashed** on this run's report (`[IRT_RAGGED_MATRIX]`): a single
`gemini` per-item timeout left the persisted `grade_matrix` ragged, and the forward screen called
`irt.prune_items` without the shared-item degradation the post-hoc run screen already applies. Since
a per-item salvage is normal on any cloud run, curate would have crashed in the field. Fixed
test-first (a new public `irt.shared_item_matrix`, reused by `select_discriminators`; the ragged
drops are now reported, not silently swallowed): 854 → 857 tests. The two ragged items
(`pair-23-boollogic`, `stat-03-simpson`) are surfaced in the curate output rather than crashing it.

### Bottom line

The Phase-A pipeline runs end-to-end and **empirically settles** the lever-(a) question:
re-screening the existing bank can't fix the panel-redundancy problem, because the bank saturates to
~3 discriminators for frontier judges. The fix is **content** (Phase B), not curation knobs. ω stays
on ice; this run changes nothing about seating — it characterizes the *instrument's* own limit. And
the demonstration paid for itself by surfacing and fixing a crash in the shipped curation path.

### Reproduce

```
# 1. full-set run (any disjoint-family panel; cloud shown), persisting the grade matrix
uv run ai-crucible characterize \
  --models qwen3-coder:480b-cloud@qwen glm-4.6:cloud@zhipu \
    gpt-oss:120b-cloud@openai gemini-3-flash-preview:cloud@google \
  --k 3 --out eval/rho-demo-full-2026-06-28.json
# 2. curate the discriminators offline (no model; degrades a ragged matrix to the shared subset)
ai-crucible calibration curate --from-run eval/rho-demo-full-2026-06-28.json \
  --out eval/rho-demo-harder-subset-2026-06-28.json
# 3. re-run on just the discriminators (drop the dead glm tag), compare panel_correlation
uv run ai-crucible characterize --items eval/rho-demo-harder-subset-2026-06-28.json \
  --models qwen3-coder:480b-cloud@qwen gpt-oss:120b-cloud@openai gemini-3-flash-preview:cloud@google \
  --k 3 --out eval/rho-demo-curated-2026-06-28.json
```
