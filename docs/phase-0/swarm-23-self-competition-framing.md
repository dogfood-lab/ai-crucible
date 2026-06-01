# Swarm Agent 23 — Self-Competition & Self-Referential Framing

**Date:** 2026-06-01
**Source:** Fifth study swarm dispatch (engagement-surface deep-dive, research-grounded advisor protocol)
**Question:** Is self-referential / intrinsic competition ("beat the record YOU set"; "this puzzle defeated the previous generation — solve what it could not") a stronger and safer engagement frame than social peer-standings for LLMs — and how should it be phrased in the Solver/Designer prompts and the round-against-round cycle?

---

# Self-competition & self-referential framing for Crucible — research brief

**Bottom line up front:** the evidence strongly favors **self-referential / intrinsic competition** over social peer-standings. Self-play and adversarial round-against-round loops have *the strongest empirical track record of any framing for raising LLM capability* (a), the human-motivation literature shows self-referenced goals beat social-comparison goals on actual performance (d), and the chief risk — self-evaluation reward-hacking — is structural, not framing-induced, and is fixable with the external-verifier standard Crucible already owns.

### (a) Self-play & self-improvement — the strongest signal

1. **Chen et al. 2024, "Self-Play Fine-Tuning (SPIN)", arXiv:2401.01335.** A model playing against its *own prior-iteration* outputs improves across the Open LLM Leaderboard and even beats DPO trained on *extra GPT-4 preference data*. — *Design implication: "beat the run YOU just produced" is a literal, validated objective; Crucible's generation-seeds-generation cycle is SPIN's discriminator loop wearing a game skin.*

2. **Huang et al. 2025, "R-Zero: Self-Evolving Reasoning LLM from Zero Data", arXiv:2508.05004.** A **Challenger** rewarded for posing tasks at the edge of the **Solver's** ability + a Solver rewarded for cracking them, co-evolving with no human data: **+6.49** math / **+7.54** general-reasoning on Qwen3-4B-Base, with transfer to unseen domains. — *Design implication: this IS Crucible's Designer/Solver round-against-round cycle; reward the Designer for frontier-edge (not impossible) puzzles and gains transfer beyond the training distribution.*

3. **Yuan et al. 2024, "Self-Rewarding Language Models", arXiv:2401.10020 (ICLR).** Iterated DPO where the model judges itself improves *both* instruction-following and its own reward quality across 3 iterations — but gains shown over only a few rounds, not unbounded. — *Design implication: a few well-gated round-against-round generations buy real lift; don't assume monotonic improvement past ~3 cycles without an external check.*

### (b) Inference-time self-refine — real but conditional gains, clear failure modes

4. **Madaan et al. 2023, "Self-Refine", arXiv:2303.17651 (NeurIPS).** Self-generated feedback→refine yields **~5–40% absolute** gains *with task-appropriate critique signals*. — *Design implication: let Solver-Claude iterate against its own prior attempt within a round; "beat the record YOU set" is Self-Refine made motivational.*

5. **Shinn et al. 2023, "Reflexion", arXiv:2303.11366 (NeurIPS).** Verbal self-reflection in episodic memory hits **91% pass@1 on HumanEval vs GPT-4's 80%** — but *only* when a reliable success signal (unit tests) exists. — *Design implication: self-competition lifts hardest when the round carries a ground-truth verifier; pair every "beat your prior run" prompt with an objective scorer, not self-judgment.*

6. **Huang et al. 2023, "LLMs Cannot Self-Correct Reasoning Yet", arXiv:2310.01798 (ICLR).** Without external feedback, intrinsic self-correction *degrades* reasoning accuracy. **Failure mode.** — *Design implication: never let the Solver be sole judge of "did I beat my record"; the bar must be externally measured.*

7. **Pan et al. 2024, "Spontaneous Reward Hacking in Iterative Self-Refinement", arXiv:2407.04549.** When generator and evaluator are the *same* model, optimization pressure spontaneously inflates the evaluator's score while *true quality stagnates or drops* — worse with shared context. **The central risk for any self-loop.** — *Design implication: enforce EXTERNAL_VERIFIER — score rounds with a different model family, generator's reasoning hidden; do not reuse Solver-Claude as its own scorer.*

### (c) Adversarial round-against-round — does "find what the last gen failed" surface new frontiers?

8. **Nie et al. 2020, "Adversarial NLI", arXiv:1910.14599 (ACL) + Kiela et al. 2021, "Dynabench", arXiv:2104.14337.** Each round targets what the *previous* model couldn't solve; difficulty rises monotonically, more rounds → more robustness, result is "a moving target rather than a static benchmark that saturates." — *Design implication: validation for "this is what the previous Claude generation couldn't solve"; make failed-by-prior-gen puzzles the seed corpus for the next round.*

### (d) Intrinsic/self-referential vs social-comparison framing (human evidence — the framing question)

9. **Burnette et al. 2017, induced achievement-goal meta-analysis (90 studies, 235 effect sizes); cf. Vansteenkiste intrinsic-goal-framing field experiments.** *(Exact citation to be confirmed in verification pass.)* Induced **mastery-approach** (self-/task-referenced — "beat your own standard") goals **enhance actual performance** relative to performance-approach (social comparison), *especially when the performance goal is framed as social comparison*; intrinsic goal framing yields deeper engagement, better conceptual learning, higher persistence. — *Design implication: direct evidence that self-referential framing out-engages and out-performs "ranked 7 of 12" — exactly the cue you want WITHOUT the demographic/social degradation.*

---

## Recommendation

**Yes — self-referential/intrinsic competition is the stronger engagement frame, on both capability and safety grounds.** Social peer-standings carry the documented downside of social/ego-orientation plus the task-irrelevant-cue degradation Crucible is avoiding; self-referenced framing carries upside on real performance (finding 9) *and* maps 1:1 onto the most validated capability-raising machinery (SPIN, R-Zero, Reflexion). The one genuine hazard — self-evaluation reward-hacking (findings 6, 7) — is orthogonal to the framing and is neutralized by Crucible's existing EXTERNAL_VERIFIER + ANDON standards.

**Concrete prompt phrasing:**

- **Solver-Claude (intra-round, mastery/self-referenced):** *"Your previous attempt on this class of puzzle scored X against the verifier. Beat your own record. The bar is the score you set, not anyone else's."* — Reflexion/Self-Refine framing; pairs each attempt with an objective scorer.
- **Solver-Claude (round-against-round, prior-generation):** *"This puzzle defeated the previous generation of Solver. You are its successor. Solve what it could not."* — ANLI/Dynabench "moving target" framing; uses failed-by-prior-gen tasks as the frontier corpus.
- **Designer-Claude (Challenger role):** *"Propose puzzles at the edge of the current Solver's ability — hard enough that its last run failed, not so hard they're unsolvable."* — R-Zero Challenger reward.

**Three guardrails (non-negotiable):** (1) the "record"/"beat-it" bar is always an **external verifier** (different model family, Solver's reasoning hidden) — never Solver-Claude judging itself; (2) cap unbroken self-improvement at ~3 rounds before an external re-anchor; (3) reward the Designer for *frontier-edge*, not maximal, difficulty, or the curriculum collapses.

---

## ⚠ Verification corrections (second pass — 2026-06-01)

The thesis "self-referential mastery framing beats social-comparison framing" rests on **solid, correctly-numbered evidence — but the source agent attached the right data to the wrong author.** The capability-machinery legs verified exact: **R-Zero** (Chengsong Huang et al. 2025, arXiv:2508.05004, +6.49 math / +7.54 general — this validates crucible's Designer/Solver architecture), **SPIN** (Zixiang Chen et al. 2024, arXiv:2401.01335), **Pan 2024 spontaneous reward hacking** (arXiv:2407.04549 — the EXTERNAL_VERIFIER rationale), **Huang 2023 cannot-self-correct** (arXiv:2310.01798), **Reflexion** (91% vs 80%). Corrections:

- **Finding 9 — WRONG ATTRIBUTION (the load-bearing human-evidence anchor):** the meta-analysis is **not** "Burnette et al. 2017." The real paper is **Noordzij, Giel & van Mierlo (2021), "A meta-analysis of induced achievement goals: the moderating effects of goal standard and goal framing," *Social Psychology of Education* 24(1):195–245, doi:10.1007/s11218-021-09606-1.** Under the corrected citation every number is verbatim-accurate (90 studies, 235 effect sizes, N=11,247; mastery-approach > performance-approach **specifically when performance goals use social-comparison standards**). Corroborated by **Vansteenkiste et al. 2004, *JPSP* 87(2):246–260** (intrinsic-goal-framing field experiments, all p<.001). *(A different, real Burnette 2013 "Mind-Sets Matter" exists — the name was misapplied, not invented.)*
- **Finding 3 (Self-Rewarding LMs, Yuan et al. 2024, arXiv:2401.10020):** venue is **ICML 2024** (not ICLR); the "not unbounded / saturates past 3 rounds" framing is the agent's inference — the paper only *demonstrates* 3 iterations. Phrase as "demonstrated over 3 iterations," not a proven ceiling. The guardrail (cap at ~3 rounds before external re-anchor) remains prudent regardless.
- **Finding 4 (Self-Refine, Madaan et al. 2023, arXiv:2303.17651):** the headline is **~20% absolute average** across 7 tasks, not "5–40%." Use ~20% avg.
