# Kickoff — crucible Phase-2 forks (dogfood swarm)

**Paste this as a new session's first message (or: "Read `docs/phase-2/kickoff-forks-swarm.md` and run the dogfood swarm").**
**Authored:** end of the 2026-06-01 calibration+panel session. **Branch:** `phase-2` @ `5c3d6f6` (origin: `dogfood-lab/crucible`).

---

## What crucible is (frame — read before acting)

crucible is a **diagnostic auditing game / measurement instrument**: a thin policy layer over Inspect AI that seats a cross-family panel of local judges to score Solver attempts under a sealed measurement boundary. It is **not** a toy — it is studio infrastructure (the measurement arm). Canon-first; do **not** shrink scope; do **not** scaffold GDOS; do **not** propose smaller-tool shortcuts. Full mission frame in `E:/AI/.claude/CLAUDE.md` + `C:/Users/mikey/.claude/projects/F--AI/memory/user_profile.md`.

**Path rule (Robot rig):** every `F:/AI/...` in any memory/skill/doc is read as `E:/AI/...`. Workspace is `E:/AI/dogfood-lab/crucible`.

## Where the last session left it (state)

A complete vertical slice shipped on `phase-2` (9 commits): **characterize → compose → persist → score**, end to end.

- **Characterization** (`src/crucible/characterize/`): `run.py` driver (A/B pairs + PASS/FAIL, two-pass model-jury bootstrap, authored `item_id`, IRT-prune + perturbation + calibration + panel-composition report sections), `metrics.py` (the §11.1 six + ECE + **temperature scaling, held-out k-fold** §12 Q3), `profile.py` (one-sided κ gate, continuous quality score, selective CI seat/screen/reject, `perturbation_audit`), `aggregate.py` (ρ-submodularity, reliability-weighted vote, minority-veto, **`compose_panel`** + `SeatedPanel`), `panel_store.py` (durable artifact save/load), `calibration/irt.py` (model-free point-biserial/variance prune).
- **Scoring** (`src/crucible/scoring/judge_panel.py`): `"weighted"` reducer (CARE), `weighted_judge`, `JudgePanel.from_seated`.
- **Kernel** (`src/crucible/kernel.py`): `run_attempt(..., panel=...)` accepts a pre-composed panel.
- **Grounding:** `docs/research-grounding.md` §11/§12/§12.1 (design lock + corrected-run receipts); raw study-swarms in `docs/phase-2/{calibration.md, grounding-research.md}`.
- **Gates:** `uv run ruff check .` clean; `uv run pytest -q` = **443 passed**, green ×3.
- **The corrected run (6 models × 51 pairs × k=3):** seated = qwen3.6:27b, gemma4:31b, granite4.1:30b (the three κ≈1.0 judges the *old inverted* gate wrongly screened); ECE measured via logprob; ρ/known-groups/IRT-prune/perturbation all real. Raw report is gitignored (`char-*.json`).

## Standing constraints (load-bearing — do not violate)

1. **No publishing.** No `npm publish`, no `gh release`, no release tag. This is dev-branch work.
2. **Never merge `phase-2` → `main` without explicit user authorization.** Commit + push the feature branch only.
3. **The model jury CANNOT be the alt-test reference** — it is circular (Calderon 2025). Fork C exists precisely to replace it with humans. Keep the `_ALT_TEST_CAVEAT` loud until then.
4. **Honesty surfaces:** any shortcut/cap/assumption is documented loudly in code + report + §12, never buried.
5. **Six workflow standards** apply (`E:/AI/.claude/rules/workflow-standards.md`); every new runner/workflow file carries a "Standards compliance" section.
6. **Preview-plugin override:** crucible is a CLI/library, not web — never call preview/screenshot tools.
7. Translations (if any docs ship) run **locally** via Ollama — never on Claude tokens. (Not expected here.)

## How to run things

```
# characterization panel (sequential serving — VRAM-respecting on the Omen RTX 5090):
OLLAMA_NUM_PARALLEL=1 OLLAMA_MAX_LOADED_MODELS=1 \
  uv run python -m crucible.characterize.run --k 3 \
  --out char-panel-report.json --write-panel docs/phase-2/panel.json
# gates:
uv run ruff check .
uv run pytest -q          # run 3x for flake-safety
```
Panel models (all pulled): `qwen3.6:27b`, `mistral-small:24b`, `gemma4:31b`, `aya-expanse:32b`, `granite4.1:30b`, `devstral-small-2:24b`.

---

## The swarm

Run as a **dogfood swarm**: Wave 0 (coordinator locks the cross-cutting contracts below) → parallel leaf agents on **disjoint files** → coordinator integrates the shared files → composed re-audit by a *different* agent (EXTERNAL_VERIFIER) → amend. **Fork C opens with a `study-swarm`** (research-grounded-advisor protocol — it introduces a new product layer + qualitative design questions). Forks A and the C-study can run concurrently; Fork B is gated on A + a director GPU go-ahead.

### Fork A — Expand the discriminating pair set (instrument quality) — **no GPU to author**

- **Why:** the IRT prune kept only **15 of 51** items; the set saturates at the top for the strong judges. Top judges should sit near the ~64% ceiling (JudgeBench, arXiv:2410.12784), not 100%.
- **Goal:** author **~30–50 more** plausible-vs-subtly-wrong A/B pairs into `src/crucible/calibration/admission_pairs.json`. Difficulty centered ~0.5–0.65 with tails (PSN-IRT, arXiv:2505.15055); balanced `gold` A/B and `correct_side` position; diverse, *new* `flaw_family` coverage and constructs.
- **Contract (Wave 0):** item schema is fixed by `calibration/loader.py` + `calibration/types.py` — `id` (unique), `category` (use `difficulty_anchor`; reserve a few `known_trivial`/`known_impossible` anchors), `construct`, `confound_controlled`, `prompt` (ends "Reply with exactly one letter: A or B."), `gold` ∈ {A,B}, `difficulty`, `expected_pass{strong,weak}`, `metadata{flaw_family, correct_side}`. The loader rejects unknown keys + duplicate ids — it is the verifier.
- **Leaf split (DECOMPOSE_BY_SECRETS):** one agent per flaw-family slice (e.g. off-by-one, unit-conversion, quantifier/logic, code-semantics, fabricated-fact-among-real, order-of-operations, statistical-trap). Each writes a disjoint block of new items; the coordinator merges into one JSON and re-balances position.
- **Verify:** `load_default`/`load_items` loads clean; position balance ≈50/50; difficulty histogram has tails; (later, GPU-gated) re-pilot keeps **>15** discriminating items.

### Fork B — Mint the committed `panel.json` artifact — **GPU (~40 min), director-gated**

- **Do:** one characterization run with `--write-panel docs/phase-2/panel.json` (after Fork A if expanding, so the artifact reflects the better set).
- **Commit:** `docs/phase-2/panel.json` (canonical seated panel) + refresh §12.1 with the now-populated `calibration` (held-out `ece_cv`) and `panel_composition` sections.
- **Verify:** `panel_store.load_panel("docs/phase-2/panel.json")` round-trips; report carries `calibration` + `panel_composition`; seats sane.

### Fork C — Human alt-test harness (retire the circular jury-ω) — **study-swarm FIRST**

- **Study-swarm (fire the research-grounded-advisor protocol):** design questions — annotation format + UI, how many/which humans (Calderon 2025: ≥3 annotators, ≥30 items, ε=0.2 expert / 0.15 skilled / 0.1 crowd), adjudicating jury-disagreements (MACE, arXiv:2508.07827; multi-LLM+human, arXiv:2503.17620), conformal coverage on a small set. Grounding already in §12 Q3 + `calibration.md`.
- **Then build:** a human-label schema + loader (`human_labels.json`: `item_id → {annotator_id: verdict}`); feed REAL human annotators into `build_profile(..., records_per_annotator=...)` (the existing `alt_test_omega` already takes a reserved `"judge"` key + annotators — swap the model peers for humans). Make ω non-circular; **remove/replace `_ALT_TEST_CAVEAT`** in `run.py` and update §12.1.
- **Verify:** ω computed against humans on synthetic-human-label tests; the circular-bootstrap caveat is gone; gates green.

### Coordinator sequencing

1. Wave 0: lock the Fork-A item contract + the Fork-C human-label schema. 2. Parallel: Fork-A leaf authors **and** the Fork-C study-swarm. 3. Integrate Fork A → (director GPU go-ahead) → Fork B run + commit `panel.json`. 4. Build Fork C against the study findings. 5. Composed re-audit (different agent) → amend. **No merge to `main`.**

## Standards compliance (the six — scored for this swarm plan)

- **PIN_PER_STEP — 2:** Wave 0 pins each leaf's contract; the run command pins serving env + `--k`. Remediation: each leaf records its model+prompt in its commit (owner: coordinator, this swarm).
- **ANDON_AUTHORITY — 2:** the loader (Fork A) and the gates halt on a bad item/regression before integration; the composed re-audit can halt before amend. Remediation: add an explicit "stop on loader/ruff/pytest failure" line to each leaf brief.
- **NAMED_COMPENSATORS — 2 (no skip; irreversible calls present):** see table below.
- **DECOMPOSE_BY_SECRETS — 3:** leaves are disjoint flaw-family slices + disjoint files; the human-label schema is isolated from the item set.
- **UNCERTAINTY_GATED_HUMANS — 3:** the GPU run (Fork B) and any `main` merge are director-gated; Fork C is gated behind a study-swarm because the design is uncertain.
- **EXTERNAL_VERIFIER — 3:** composed re-audit by a different agent; Fork C's entire purpose is replacing the circular model-jury reference with a non-circular human one.

### Compensators (irreversible tool calls — NO skip)

| Action | Command to undo | Post-rollback state | Owner |
|---|---|---|---|
| `git commit` on `phase-2` | `git revert <sha>` (or `git reset --soft HEAD~1` pre-push) | commit removed, files restored | coordinator |
| `git push origin phase-2` | `git revert <sha> && git push` (avoid force on shared branch) | remote branch back to prior tip | coordinator |
| write `docs/phase-2/panel.json` | `git checkout -- docs/phase-2/panel.json` (it is committed) | prior artifact restored by git | Fork B owner |

(No `npm publish` / `gh release` / tag / Pages deploy in this swarm — those remain explicitly out of scope.)

## Pointers

- Code: `src/crucible/characterize/{run,metrics,profile,aggregate,panel_store}.py`, `src/crucible/calibration/{admission_pairs.json,loader,types}.py`, `calibration/irt.py`, `src/crucible/scoring/judge_panel.py`, `src/crucible/kernel.py`
- Tests: `tests/test_{characterize,characterize_run,calibration,models,scoring,kernel,panel_store}.py`
- Grounding: `docs/research-grounding.md` §11/§12/§12.1; raw: `docs/phase-2/{calibration,grounding-research}.md`
- Protocols: dogfood-swarm + `study-swarm` (research-grounded-advisor) in `C:/Users/mikey/.claude/projects/F--AI/memory/research-grounded-advisor-protocol.md`; workflow standards in `E:/AI/.claude/rules/workflow-standards.md`
