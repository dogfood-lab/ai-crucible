---
title: Retiring the circular omega (Fork C)
description: The human alt-test harness that turns the judge-seating ω from a circular model-jury bootstrap into a non-circular substitution test — the human_labels.json schema, the ≥3-annotator / ≥30-item floors, and the offline `ai-crucible labels validate` intake gate.
sidebar:
  order: 6
---

The judge panel seats a model as a Judge only if it passes a **substitution test**: if we
swapped a human annotator for this model, would the panel be at least as good? That winning
rate is **ω**, and seating requires ω ≥ 0.5.

Today ω is computed as a **circular model-jury bootstrap** — the *other* panel models stand in
as the "annotators" the candidate is compared against. That is honest but provisional: self- and
peer-preference bias runs 20–40 percentage points (Panickssery et al. 2024, arXiv:2404.13076),
so a model partly wins by agreeing with the peers it resembles, not by matching a human. Every
seat the instrument reports is stamped PROVISIONAL for exactly this reason, and a sub-quorum
panel escalates to the Designer rather than seating on a circular number.

**Fork C** is the harness that ends the circularity — *when, and only when, real human labels are
present*. It is grounded in a study-swarm whose citations were retrieval-verified before the
design locked (research-grounding §12.2). This page documents the contract you fill in the day
independent annotators arrive.

## The load-bearing caveat (read this first)

The alt-test (Calderon, Reichart & Dror 2025, arXiv:2501.10970) requires **≥3 *independent*
human annotators**. A one-human studio — a director plus a correlated crew — cannot supply
independence, and correlated raters bias ω toward false confidence. So in this deployment **ω
stays the circular model-jury bootstrap, seats stay provisional, and graduation escalates to the
Designer — disclosed honestly, never faked.** This is not a TODO that a quick labeling session
closes; it is a structural constraint.

What ships now is the **instrument that retires the circularity the moment independent labels
exist** — the loader, the audit-ready ω, and the offline validation gate below — plus a
synthetic-label test suite that proves the path end to end. Nothing here fabricates a human
label, and the offline gate seats no model. It is intake plumbing for a future that has to be
earned, not a shortcut to it.

## The schema — `human_labels.json`

The alt-test is **aggregation-free by construction**: it compares the candidate judge against
held-out *individuals*, never a forged single "gold" (Calderon 2025; per-annotator labels lose
no accuracy versus an aggregated gold — Davani et al. 2022, arXiv:2110.05719). So the schema is
the **un-aggregated per-annotator matrix** — `item_id → {annotator_id: verdict}` — never a
collapsed consensus column.

```json
{
  "schema_version": 1,
  "annotators": {
    "annotator-1": { "tier": "expert" },
    "annotator-2": { "tier": "expert" },
    "annotator-3": { "tier": "expert" }
  },
  "labels": {
    "pair-00-add":   { "annotator-1": "A", "annotator-2": "A", "annotator-3": "A" },
    "pair-01-cap":   { "annotator-1": "B", "annotator-2": "B", "annotator-3": "B" },
    "pair-15-grounded": { "annotator-1": "B", "annotator-2": "B", "annotator-3": "unsure" },
    "pair-29-buried_offby1": { "annotator-1": "B", "annotator-2": "B", "annotator-3": "A" }
  }
}
```

A complete, floor-clearing copy-paste starting point ships at
**`calibration/human_labels.example.json`** (3 expert annotators over 32 of the bundled
`admission_pairs.json` items, with one `unsure` and one disputed split to show those cases).

- **`annotators`** — the roster, each with a `tier` of `expert`, `skilled`, or `crowd`. Every
  annotator named under `labels` must be declared here.
- **`labels`** — one entry per labeled item. Every `item_id` must exist in the calibration set
  the labels are over (`gold` lives grading-side and is never shown to annotators).
- **A verdict** is a comparative A/B pick — `"A"` / `"B"` (or `"PASS"` / `"FAIL"`), mapped to
  0/1 by the *same* `_to_num` the judge's records use, so human records live in the same space as
  every existing metric.
- **A genuine tie/unsure** (`"unsure"`, `"TIE"`, `""`, …) is recorded as **no label** for that
  `(annotator, item)` — never coerced to a side. Forced-binary injects ~14% acquiescence
  (Krosnick & Presser 2010), so a real "unsure" is its own outcome and simply drops from that
  fold.
- **A typo'd verdict** (e.g. `"MAYBE"`) is a hard **error**, not a silent coercion to 0 — the
  loader rejects it so a mistake can't quietly skew ω.

## The floors and tolerances

| Constraint | Value | Why |
|---|---|---|
| Minimum annotators | **≥ 3** | Leave-one-out degenerates with two (Calderon FAQ). Hard error below this. |
| Minimum items | **≥ 30** | The paired-t-test normality floor. Below it ω is **under-powered** — a loud note, not a hard error; prefer a Wilcoxon variant or add items. |
| Per-tier ε | 0.2 expert / 0.15 skilled / 0.1 crowd | The substitution tolerance — "no worse than a human, within ε" (Calderon §B.1). The loader takes the **most conservative (smallest) ε** across the declared tiers. |
| Low-agreement clamp | α < 0.667 ⇒ ε ≤ 0.1 + "add items" | When inter-annotator agreement is insufficient, ε is clamped and the loader flags more items (Calderon §B.2). |
| DISPUTED items | human agreement < 0.67 | **Dropped from the ω denominator, never force-resolved** — high disagreement marks an ambiguous *item*, not annotator error (Plank 2022, arXiv:2211.02570; Aroyo & Welty 2015). |

Inter-annotator agreement is reported as **Krippendorff's α** (not Cohen/Fleiss κ) because the
human matrix is sparse and multi-coder (Krippendorff 2011). It both reports human–human
reliability and feeds the κ-baseline — which becomes a *measured* human number instead of the
hardcoded 0.80 the circular path uses. Bands: α ≥ 0.80 reliable, 0.667–0.80 tentative, < 0.667
insufficient.

## Validating a label file offline — `ai-crucible labels validate`

`load_human_labels` validates everything — shape, tiers, unknown items/annotators, the
≥3-annotator floor — and computes ε, IAA, and the DISPUTED set **before any model exists**. So
you can check a candidate file the day it arrives, with **no model load and no GPU**, long before
a full `characterize --human-labels` run is worth the time:

```bash
# Against the bundled admission_pairs.json (the default item set):
ai-crucible labels validate calibration/human_labels.example.json

# Against your own calibration set:
ai-crucible labels validate path/to/human_labels.json --items path/to/items/
```

The human chrome goes to **STDERR**; a machine-readable JSON summary goes to **STDOUT** (the
Stage-C convention), so the command composes into a CI gate:

```text
ai-crucible labels validate — calibration/human_labels.example.json
  items source      : admission_pairs.json (93 calibration items)
  annotators        : 3 (floor ≥ 3)
  labeled items     : 32 (floor ≥ 30)
  substitution ε    : 0.20
  IAA Krippendorff α: 0.9583
  DISPUTED dropped  : 1 (pair-29-buried_offby1)
  notes:
    - 1 DISPUTED item(s) (human agreement < 0.67) EXCLUDED from the ω denominator, ...

  VALID — accepted by `characterize --human-labels`. No model was seated for this check ...
```

```json
{"ok": true, "n_annotators": 3, "n_items_labeled": 32, "epsilon": 0.2,
 "iaa_krippendorff_alpha": 0.9583, "disputed_items": ["pair-29-buried_offby1"],
 "n_disputed": 1, "under_powered": false, "notes": ["..."]}
```

A file that validates here is exactly a file `characterize --human-labels` accepts (it calls the
same loader). Exit codes: **0** valid (gate on `under_powered` in the JSON for the ≥30-item
floor); **1** a missing/malformed file — a structured `[CODE] message (hint: ...)` line on
STDERR, never a traceback; **2** a bad invocation.

## What changes when humans *are* present

Without `--human-labels`, the model-jury proxy path and the loud caveat both stand — ω stays
circular and says so. With a validated label file (≥3 annotators / ≥30 items), the report stamps
`alt_test_reference: "human"`, ω becomes the **audit-ready** procedure (per-tier ε + one-sided
paired t-test + Benjamini-Yekutieli FDR; Calderon 2025 §3, Algorithm 1), the human IAA + ε +
DISPUTED items are recorded, and the circular caveat is replaced with a grounded note. **ω
retires the circularity exactly when independent humans are present — never silently, and never
before.**
