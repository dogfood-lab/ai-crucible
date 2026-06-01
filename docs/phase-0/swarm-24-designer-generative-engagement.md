# Swarm Agent 24 — Designer-Side Generative Engagement

**Date:** 2026-06-01
**Source:** Fifth study swarm dispatch (engagement-surface deep-dive, research-grounded advisor protocol)
**Question:** What makes an LLM GENERATE harder, more novel, higher-quality, well-calibrated artifacts (eval puzzles)? Design the Designer-Claude reward surface to maximize generative quality + diversity + difficulty-calibration without degenerate or gamed output.

---

# Designer-Claude reward surface: generating harder, novel, well-calibrated puzzles

**Frame:** the Designer's "fun" is *crafting puzzles other Claudes still fail at*. The evidence says that target is reachable only if the reward couples three things — diversity, validity, and frontier-calibration — and explicitly caps the cheats for each.

## Findings

**1. Quality-Diversity beats quality-alone (Bradley 2023, "Quality-Diversity through AI Feedback," arXiv:2310.13032)** — https://arxiv.org/abs/2310.13032 — QDAIF, using an LLM both to mutate and to *judge* diversity, "covers more of a specified search space with high-quality samples than do non-QD controls." *Design implication:* reward the Designer on QD coverage (filling a behavior-descriptor archive of distinct puzzle types), not just per-puzzle quality — a quality-only objective collapses to one winning template.

**2. LLMs are competent in-context QD operators (Lim 2024, "LLMs as In-context AI Generators for Quality-Diversity," arXiv:2404.15794)** — https://arxiv.org/pdf/2404.15794 — feeding the current archive into the prompt lets an LLM generate targeted, archive-aware offspring without online training. *Design implication:* show Designer-Claude the existing puzzle archive (which capability cells are full/empty) and reward it for filling *empty* cells — framing "this region of solver-failure is unexplored."

**3. Evolutionary search + reflection raises generated-artifact quality past human experts (Ma 2023, Eureka, arXiv:2310.12931)** — https://arxiv.org/abs/2310.12931 — iteratively proposing batches and reflecting on *fitness feedback* beat human-designed reward code on 83% of 29 tasks. *Design implication:* the Designer's fitness signal must be the puzzle's *actual measured solver outcome* (pass^k), fed back for the next generation — generation without a grounded fitness loop drifts.

**4. Difficulty/complexity can be explicitly evolved (Xu 2023, WizardLM/Evol-Instruct, arXiv:2304.12244)** — https://arxiv.org/abs/2304.12244 — step-wise "deepening/complicating" rewrites produce instructions humans rate as harder and higher-quality than human-written ones. *Design implication:* give the Designer explicit difficulty-increasing operators (add-constraint, add-reasoning-hop, remove-scaffold) and reward measured difficulty gain — but verify the gain is real (see #6).

**5. Adversarial human-and-model-in-the-loop produces durably hard, non-saturating items (Nie 2020, ANLI, arXiv:1910.14599)** — https://arxiv.org/abs/1910.14599 — items kept *only when current models fail* yield a "moving target," not a static benchmark that saturates. *Design implication:* a puzzle should only "graduate" (and earn the Designer reward) once a fresh solver pool empirically fails it at the target rate — calibrate to a solve-rate band, exactly the "pass^3 = 18%" framing the team already uses.

**6. FAILURE MODE — adversarial filtering encodes spurious artifacts, not real difficulty (Zellers 2019, HellaSwag, arXiv:1905.07830; Bras 2020, AFLite, arXiv:2002.04108)** — https://arxiv.org/abs/1905.07830 — filtering to "models fail" can produce items hard only because of stylistic/distractor artifacts (BERT drops only 11.9 pts with context removed). *Design implication:* HARD CAP — reward only puzzles that are hard for *diverse* solver families and that a verifier confirms are well-posed and genuinely solvable; penalize items whose difficulty vanishes under a sanity/ablation check (the "fake difficulty" cheat).

**7. FAILURE MODE — auto-generated eval items leak/contaminate and inflate scores (Xu 2024 survey, arXiv:2406.04244)** — https://arxiv.org/html/2406.04244v1 — static auto-generated benchmarks get memorized; contamination inflates measured performance up to ~45%. *Design implication:* reward *freshly-constructed, non-templated* puzzles; penalize near-duplicates of prior items (embedding-distance gate) so the Designer can't farm reward by paraphrasing one solved puzzle.

**8. FAILURE MODE — any proxy objective gets Goodharted into degenerate output (Skalse 2022 reward-hacking; Ribeiro 2020 CheckList, arXiv:2005.04118)** — https://arxiv.org/abs/2005.04118 — generators satisfy the proxy (verbosity, repetition, "diversity" as trivial perturbation) without the intended property; held-out "looks hard" overestimates real capability-probing. *Design implication:* never let the Designer self-score; use an **external verifier of a different model family** (with the Designer's reasoning hidden) for validity/well-posedness, and define diversity by *behavioral capability cell*, not surface text.

## Recommended Designer reward surface

**Reward (multiplicatively gated, so a zero on any gate is zero reward):**
1. **Validity gate (binary, external verifier, different family):** puzzle is well-posed, has a verifiable solution, is not impossible-by-construction. Gates everything. (#6, #8)
2. **Frontier calibration (peaked, not monotonic):** reward maximized when measured solver pass^k lands in a target band (e.g. 10–30%); near-0% and near-100% earn little. The core "craft a puzzle they still fail" signal. (#5, #3)
3. **Novelty / QD coverage:** reward for occupying an *empty capability cell* in the puzzle archive (behavioral descriptor = which reasoning skill it probes), scaled down sharply for near-duplicates by embedding distance. (#1, #2, #7)
4. **Robust-difficulty bonus:** extra reward when the puzzle is hard across *multiple* solver families and survives an artifact/ablation sanity check. (#6)

**Cap / penalize:** near-duplicate or paraphrase farming (#7); difficulty that collapses under ablation = "fake hard" (#6); self-assessed validity (#8); single-cell over-mining (diminishing returns past a coverage threshold per cell) (#1).

**Engagement framing that supports it:** keep the graduation narrative — *"puzzle #47 graduated; pass^3 = 18%; two solvers found approaches you didn't anticipate."* It rewards the three load-bearing axes at once: **graduated** (passed validity + frontier gates), **18%** (hit the calibration band), **unanticipated approaches** (genuine, non-artifact difficulty + diversity). Surface an archive heatmap of solver-failure cells as the Designer's "frontier map" so its intrinsic drive is to *expand the unexplored frontier of Claude's failures* — diversity and difficulty become the game, with the verifier as the guardrail that makes the score honest.

---

## ⚠ Verification corrections (second pass — 2026-06-01)

The Designer reward surface verified clean across the board: **QDAIF** (Bradley et al. 2023, arXiv:2310.13032), **Lim 2024** (arXiv:2404.15794), **Eureka** (Ma et al. 2023, arXiv:2310.12931 — 83% of 29 tasks, exact), **Evol-Instruct/WizardLM** (Xu et al. 2023, arXiv:2304.12244), **HellaSwag** (Zellers et al. 2019, arXiv:1905.07830 — BERT −11.9pt without context, exact) and **AFLite** (Le Bras et al. 2020, arXiv:2002.04108) all confirmed. One number correction:

- **Finding 7 (contamination, Xu et al. 2024, arXiv:2406.04244):** the paper is real and its thesis (auto-generated/static benchmarks get memorized; contamination inflates measured performance) is correct — but the specific **"~45% inflation" figure does not appear in it** (it is a survey with no single headline number; the nearest figure is a 39.4% *reduction* from a mitigation tool, an inverse framing). Drop the "~45%" number; keep the qualitative claim and the near-duplicate-penalty design implication, which stand.
