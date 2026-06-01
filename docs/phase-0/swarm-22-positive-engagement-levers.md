# Swarm Agent 22 — Positive Engagement Levers That Raise LLM-Agent Performance

**Date:** 2026-06-01
**Source:** Fifth study swarm dispatch (engagement-surface deep-dive, research-grounded advisor protocol)
**Question:** What prompt/reward-surface levers have DEMONSTRABLE positive effects on LLM-agent performance, persistence, or exploration on hard tasks (challenge calibration, mastery/self-referential goals, curiosity/novelty rewards, legible-progress feedback) — beyond goal-gradient + IMAGINE?

---

# Engagement levers that raise LLM-agent effort, persistence & exploration

Findings 2024–2026, LLM-native unless flagged.

**(a) Challenge calibration / desirable difficulty**

1. **Li et al. 2025, "Staying in the Sweet Spot" (SEELE), arXiv:2509.06923** — https://arxiv.org/abs/2509.06923 — Keeping problems in a capability-matched "sweet spot" via adaptive hint scaffolding + per-problem accuracy prediction beat GRPO by **+11.8 points avg** on six math benchmarks. *Design implication:* the oracle/kernel should target puzzles whose predicted solve-probability sits near the Solver's frontier (≈40–60%), not max difficulty — and say so.

2. **ADARFT (Shi et al. 2025), "Efficient RL Finetuning via Adaptive Curriculum," OpenReview T795V77s00** — https://openreview.net/pdf?id=T795V77s00 — Continuously steering toward problems matched to *current* capability gave **up to 2× faster convergence + higher final accuracy** on competition math. *Implication:* difficulty should track the Solver as it improves within a session — a moving frontier, not a fixed ladder.

3. **"Probing the Difficulty Perception Mechanism," 2025, arXiv:2510.05969** — https://arxiv.org/abs/2510.05969 — LLMs *internally encode* problem difficulty (linearly decodable from hidden states before solving) and modulate effort accordingly — but often under-sustain effort on hard items. *Implication:* an explicit "calibrated to your frontier" framing is legible to the model and can counter the documented effort-collapse; pair the claim with sustained-effort cues. **(Validates the calibration framing premise directly.)**

**(b) Mastery / self-referential improvement**

4. **Chen et al. 2025, "Reflect, Retry, Reward," arXiv:2505.24726** — https://arxiv.org/abs/2505.24726 — Reward only the *self-reflection that produced an improved retry*: **+9.0 pp** (function-calling) and **+16.0 pp** (math equations); a trained 7B beat an untrained 72B. *Implication:* Crucible's kernel should give explicit credit to the reflection-between-attempts, not just the final solve — score "what you changed and why it helped."

5. **Song et al. 2025, "Reward Is Enough: LLMs Are In-Context RL Learners" (ICRL prompting), arXiv:2506.06303** — https://arxiv.org/abs/2506.06303 — Showing prior attempts **plus scalar rewards** in-context, with alternating explore/exploit instructions, lifted Game-of-24 to **90% vs 44–47%** for Reflexion/Self-Refine, and won 86% of creative-writing comparisons. Scalar feedback beat verbal. *Implication:* feed the Solver its own prior solve/elegance/novelty *numbers* across attempts (a personal-best ledger) — the single strongest in-prompt lever found.

**(c) Curiosity / novelty intrinsic rewards (beyond IMAGINE)**

6. **CDE (Curiosity-Driven Exploration), 2025, arXiv:2509.09675** — https://arxiv.org/abs/2509.09675 — Actor-perplexity + critic-disagreement bonuses over GRPO: **+6.6 (AMC23), +2.5 (AIME24), +2.3 (AIME25)** Pass@16. *Implication:* award the novelty axis for low-likelihood-yet-valid approaches — operationalizes "novelty" as the model's own surprise.

7. **MERCI, "Count Counts," 2025, arXiv:2510.16614** — https://arxiv.org/abs/2510.16614 — Count-based state-novelty bonus, GRPO-compatible, beats strong baselines on MATH + SQL with more robust solutions. *Implication:* a cheap visit-count novelty term (penalize re-treading the same solution skeleton) is composable with Crucible's existing info-gain reward.

(Context: IMAGINE — Gao et al. 2025, arXiv:2505.17621 — gave **+22% AIME** via intrinsic exploration; CDE/MERCI are complementary, finer-grained novelty signals.)

**(d) Legible progress / structured feedback**

8. **VCRL, 2025, arXiv:2509.19803** — https://arxiv.org/abs/2509.19803 — Reward *variance* across rollouts is a clean difficulty signal; moderate-difficulty (high variance) items drive the most learning. *Implication:* the kernel can surface "this puzzle is at your high-variance edge" as a legible progress/difficulty readout — making calibration visible, not just internal.

9. **CLPO, 2025, arXiv:2509.25004** — https://arxiv.org/abs/2509.25004 — Coupling difficulty-aware curriculum signals to optimization gave **+6.96% pass@1** across eight benchmarks. *Implication:* tie the displayed tool/time budget to predicted difficulty so the budget itself reads as a calibrated "how much is enough" signal.

**Human→LLM transfer flags:** all nine above are native LLM/agent studies — no transfer assumption needed. The prior swarm's goal-gradient (Kivetz 2006) is the main human-transfer lever; treat it as weaker-evidenced than these.

---

## Recommended engagement-lever stack (ranked by evidence strength)

1. **Personal-best scalar ledger in-context (ICRL, +43–46 pp on Game-of-24).** Show the Solver its own prior solve/elegance/novelty scores across attempts with explicit explore/exploit cues. Highest-magnitude, prompt-only, zero training.
2. **Reward the reflection-that-improved (Reflect-Retry-Reward, +9–16 pp).** Kernel gives credit to the between-attempt reasoning change, not just the final solve — supports persistence.
3. **Frontier-calibrated difficulty targeting (SEELE +11.8 / ADARFT 2×).** Oracle selects puzzles at ~40–60% predicted solve-rate and tells the Solver it's calibrated to its frontier — model-legible (difficulty-probe finding).
4. **Surprise/novelty bonus on the novelty axis (CDE +2.5–6.6 / MERCI).** Score low-likelihood-valid solutions and penalize re-treading skeletons; composes with existing info-gain reward.
5. **Legible variance/budget readout (VCRL + CLPO ~+7%).** Display "high-variance edge" difficulty + difficulty-scaled budget so calibration and partial-progress are visible signals.

Levers 1–2 are prompt/kernel-surface changes deployable immediately; 3–5 require the oracle to expose a calibrated difficulty estimate. Mastery framing (1–2) and calibration (3) are the two highest-confidence, mutually reinforcing pillars.

---

## ⚠ Verification corrections (second pass — 2026-06-01)

The two highest-confidence pillars verified clean: **SEELE** (Li et al. 2025, arXiv:2509.06923, +11.8 over GRPO — exact) and **Song ICRL / Reward-Is-Enough** (arXiv:2506.06303, Game-of-24 90% vs 44–47% — exact, the personal-best-ledger lever). **ADARFT** confirmed (Shi et al. 2025, also arXiv:2504.05520, ~2×). Corrections:

- **Finding 4 (Reflect-Retry-Reward, arXiv:2505.24726):** first author is **Bensal** (not "Chen"), and the magnitudes are **+18.1% (function-calling) / +34.7% (math)** — the cited +9.0/+16.0 *understate* the paper. The 7B-beats-untrained-72B result is confirmed. Use the corrected author + numbers.
- **Finding 3 (Difficulty-perception, arXiv:2510.05969, Lee et al. 2025):** "difficulty is linearly decodable from hidden states" is SUPPORTED; the "LLMs under-sustain effort on hard items" clause is **not in this paper** — drop that clause (the calibration-is-model-legible point stands).
- **Findings 6–7 (CDE arXiv:2509.09675 / MERCI arXiv:2510.16614):** CDE is real but its MATH gain is ~flat (+0.4 Avg@1) — gains concentrate on AMC/AIME Pass@16; MERCI's exact magnitudes were not confirmable from the abstract. Soften the novelty-RL lever to "modest, math-flat for CDE; MERCI magnitudes unverified" — it stays in the stack as a secondary lever, not a headline.
