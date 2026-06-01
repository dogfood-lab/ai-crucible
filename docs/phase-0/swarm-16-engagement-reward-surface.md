# Swarm Agent 16 — Engagement & Reward-Surface Design ("Fun for Claude")

**Date:** 2026-06-01
**Source:** Fourth study swarm dispatch (Phase-1-prep, research-grounded advisor protocol)
**Question:** Is "constraint satisfaction with legible verification" (research-grounding §7) a defensible reward surface for the Solver/Designer prompt templates, or naive? What does the empirical literature on intrinsic motivation, structured-signal framing, and game design say should change the Phase 1 prompt-template design?

---

# Findings brief — "Fun-for-Claude": is constraint-satisfaction-with-legible-verification a defensible reward surface?

**Short answer: the literature is NOT empty.** There is no work on "fun for an LLM," but there is substantial direct evidence on (b) and (c) and strong transferable evidence on (a) and (d). It converges on a clear correction: the *verification half* of the team's assumption is well-supported; the *social/stakes enrichment* half (peer standings, "affects graduation eligibility," role identity) is weakly supported at best and carries documented backfire risk on agentic tasks.

### (a) Intrinsic motivation / curiosity — strong transfer, one direct LLM result

1. **Aubret 2023 — "An Information-Theoretic Perspective on Intrinsic Motivation in RL: A Survey"** (Entropy 25(2):327) — https://www.mdpi.com/1099-4300/25/2/327 — Canonical taxonomy: the durable intrinsic-reward families are *surprise/novelty* (prediction error), *information gain*, and *empowerment* (control over future states). **Design implication:** the legible-verification signal is itself an information-gain/empowerment signal — a green oracle *resolves uncertainty* the Solver could not resolve by reasoning alone. Frame the verifier as the epistemic payoff, not as a grade.

2. **Oudeyer & Kaplan / Oudeyer, Gottlieb & Lopes 2016 — "Intrinsic motivation, curiosity, and learning"** (Progress in Brain Research) — http://www.pyoudeyer.com/oudeyerGottliebLopesPBR16.pdf — The *learning-progress* (LP) signal — reward proportional to the *rate of decrease* in prediction error — produces automatic curricula (easy→hard) and is the most behaviorally robust curiosity variant. **Design implication (Designer prompt):** target the Solver's LP frontier — puzzles at the edge of solvability, not max difficulty. A "this puzzle is calibrated to your current frontier" cue is more motivating-by-design than raised stakes.

3. **Wang 2025 — "Navigate the Unknown: Enhancing LLM Reasoning with Intrinsic Motivation Guided Exploration" (IMAGINE)** — arXiv:2505.17621 — https://arxiv.org/abs/2505.17621 — Adding an intrinsic exploration reward to LLM reasoning improved AIME-2024 by **+22.23%**. This is the rare *direct* LLM-agent result: curiosity-style dense reward measurably raises hard-reasoning performance. **Design implication:** an explicit "explore alternative solution paths before committing" instruction has empirical support as a performance lever, independent of stakes.

### (b) Which structured signals change LLM behavior — and which backfire

4. **Li 2023 — "Large Language Models Understand and Can Be Enhanced by Emotional Stimuli" (EmotionPrompt)** — arXiv:2307.11760 — https://arxiv.org/abs/2307.11760 — Appending stakes/self-esteem cues ("This is very important to my career") gave **+8% (Instruction Induction) to +115% (BIG-Bench)** on *2023-era* models (GPT-3.5/4, Vicuna, Llama-2). This is the team's working assumption's strongest support — but note the model generation.

5. **Meincke / Wharton 2025 — "Prompting Science Report 1: Prompt Engineering is Complicated and Contingent"** — arXiv:2503.04818 — https://arxiv.org/pdf/2503.04818 — Asking each question **100×** per condition, they found *no reliable* benefit from generic prompt tweaks on hard benchmarks; effects were inconsistent and model-specific. **This directly tempers finding 4.** **Design implication:** treat any stakes/framing line as an *A/B-tunable parameter measured per Solver model*, never as a load-bearing assumption baked into the template.

6. **Zheng 2023 — "When 'A Helpful Assistant' Is Not Really Helpful: Personas in System Prompts Do Not Improve Performance"** — arXiv:2311.10054 — https://arxiv.org/pdf/2311.10054 — Across many personas × factual tasks, adding a role persona did **not** reliably improve (often slightly hurt) accuracy. Corroborated by **Meincke / Wharton "Report 4: Expert Personas Don't Improve Factual Accuracy"** (arXiv:2512.05858, https://arxiv.org/pdf/2512.05858). **Design implication:** the "role identity" enrichment in the Solver prompt is *not* a performance lever; keep it only if it serves catalog/legibility, not as motivation.

7. **Cao 2026 — "From Biased Chatbots to Biased Agents: Examining Role Assignment Effects on LLM Agent Robustness"** — arXiv:2602.12285 — https://arxiv.org/pdf/2602.12285 — Task-irrelevant persona cues caused **up to −26.2% degradation** on *agentic* benchmarks (planning, strategic reasoning, technical ops); **only harm, no benefit reported.** **This is the load-bearing backfire result** — it's agentic, recent, and matches crucible's setting exactly. **Design implication:** the richest-loaded parts of the draft Solver prompt (peer standings, "graduation eligibility," demographic-flavored identity) are precisely the task-irrelevant cues shown to *destabilize agent decisions.* High-leverage cut.

   *(Supporting, weaker:)* **Woolf 2024 — tipping-prompt analysis** (https://minimaxir.com/2024/02/chatgpt-tips-analysis/): monetary stakes moved *response length* a few % but most effects had high p-values — i.e., stakes change verbosity/affect, not reliably correctness.

### (c) Engagement / self-evaluation drive in LLM agents

No work establishes endogenous "engagement." The closest mechanisms are the curiosity rewards above (3) plus multi-agent-debate findings (**Smit 2023, "Should we be going MAD?"** arXiv:2311.17371 — https://arxiv.org/pdf/2311.17371) showing competitive/peer framing is *highly sensitive* to topology and can be beaten by a single strong-prompted agent. **Design implication:** competitive peer-standings are a fragile lever; a clean verifier beats a leaderboard.

### (d) Game-design / behavioral transfer (flagged as transfer, not LLM evidence)

8. **Kivetz, Urminsky & Zheng 2006 — "The Goal-Gradient Hypothesis Resurrected"** (J. Marketing Research 43(1):39) — https://journals.sagepub.com/doi/abs/10.1509/jmkr.43.1.39 — Effort *accelerates* with visible proximity to a goal; *illusory* progress (pre-stamped cards) works too. **Transfer claim, human-only.** **Design implication:** a *progress/budget indicator* ("3 of 4 hard-gates open; 2 oracle tests remain") is the highest-EV addition to the verification surface — it operationalizes goal-gradient + learning-progress, and unlike stakes it carries no backfire evidence. **Schultz (RPE/variable reward)** — https://pmc.ncbi.nlm.nih.gov/articles/PMC8116345/ — *unpredictable* reward timing maximizes the prediction-error signal; **transfer:** vary which gate opens first / occasional bonus "elegant-solution" recognition, rather than a fixed checklist.

### Verdict

**"Constraint satisfaction with legible verification" is the *right half* and is well-supported** — by intrinsic-motivation theory (info-gain/empowerment, finding 1), the only direct LLM curiosity result (finding 3), and goal-gradient transfer (finding 8). **The *enrichment* half — role identity + peer standings + "affects graduation" stakes — is naive-to-counterproductive:** persona/role cues show null-to-**−26% agentic backfire** (findings 6–7) and stakes effects don't survive rigorous replication on modern models (finding 5). **Single highest-leverage revision:** *strip the social/stakes loading from the Solver prompt and replace it with a live goal-gradient progress meter over the verification surface* — gates opened / oracle tests remaining / "you are near a complete solve" — keeping verification legible and incremental while removing the exact task-irrelevant cues the agent-robustness literature shows degrade performance. Treat any residual stakes line as a per-model A/B parameter, not a design commitment.

---

## ⚠ Verification corrections (second pass — 2026-06-01)

All 7 arXiv IDs resolve to real papers; no phantom citations. The engagement-prompt fork rests on genuine, correctly-identified evidence. Corrections to attribution/scope:

- **Finding 5 (IMAGINE, arXiv:2505.17621):** first author is **Gao et al. 2025** (Jingtong Gao); "Wang" is a co-author, not the lead. Cite as Gao et al. The "+22.23% on AIME-2024" figure is verbatim-correct.
- **Finding 1 (Cao 2026, arXiv:2602.12285) — load-bearing, calibrate carefully:** the paper is real, agentic, and the "up to −26.2% degradation, harm-only" figure is verbatim. BUT it is explicitly a **bias/vulnerability case study on task-irrelevant *demographic* persona cues** ("first systematic case study," "overlooked vulnerability"). It strongly supports *cutting task-irrelevant cues* (peer-standings, stakes framing, demographic identity) — which is the design fork — but it does **not** by itself license the broader claim that *any* role/persona framing harms agents. The synthesis must scope the claim to task-irrelevant cues, not all framing.
- **Finding 3 (Wharton Report 4, arXiv:2512.05858):** first author is **Basil et al.**; Meincke is a co-author, so "Meincke/Wharton" is defensible shorthand but not the lead.
- **Finding 4 (Report 1, arXiv:2503.04818):** the "100×-per-condition" methodology detail is from the report body, not the abstract — treat as report-sourced (the no-reliable-benefit conclusion is supported).
- Findings 2 (Zheng), 6 (Li EmotionPrompt), 7/8 (Smit MAD; Kivetz goal-gradient) verified; EmotionPrompt's "+8% to +115%" are the endpoint extremes, not a uniform gain.
