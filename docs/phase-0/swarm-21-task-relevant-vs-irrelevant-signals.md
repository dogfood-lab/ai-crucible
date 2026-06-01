# Swarm Agent 21 — The Task-Relevant vs Task-Irrelevant Signal Line

**Date:** 2026-06-01
**Source:** Fifth study swarm dispatch (engagement-surface deep-dive, research-grounded advisor protocol)
**Question:** Does competitive/peer/standing/ranking framing that is ABOUT THE TASK ITSELF (relative solve-rate, speed rank, beating an elegance record) help, hurt, or not move LLM agentic performance — and where exactly is the line between a degrading task-irrelevant cue (Cao 2026, demographic personas) and a useful task signal?

---

# Crucible Solver framing: where the task-relevant/irrelevant line falls

**Bottom line up front:** The evidence does NOT support deleting competitive framing. It supports a sharp reshaping rule — **signals that point the model at task-critical features help; signals that activate identity, demographics, or social capitulation hurt.** Crucible's instinct ("competitive framing enriches the verification surface") is correct *if and only if* the competition is expressed as quantitative task metrics, not as social/identity theater or coercive pressure.

## Findings

1. **Li et al. 2023, "Large Language Models Understand and Can be Enhanced by Emotional Stimuli" (arXiv:2307.11760)** — https://arxiv.org/abs/2307.11760 — Appending stakes/effort cues improved accuracy 8% (Instruction Induction) to 115% (BIG-Bench); attention analysis showed the stimuli **enhance focus on task-critical keywords while suppressing irrelevant tokens**, and larger models gain more. *This is the mechanism that draws the whole line.*
   → **Design implication:** Stakes language STAYS — but its value is attentional reweighting toward the task, so phrase it as "this solve is consequential," not as social drama.

2. **Patel et al. 2026, "The Role of Emotional Stimuli and Intensity in Shaping LLM Behavior" (arXiv:2604.07369)** — https://arxiv.org/abs/2604.07369 — Extends Li 2023 by varying *intensity*; effect is non-monotonic with diminishing/reversing returns at high intensity. → **Design implication:** Cap the pressure. One stakes clause, not a stacked "graduation depends on this + you're losing + last place" pileup.

3. **Luz de Araujo et al. 2025 (EMNLP), "Principled Personas" (arXiv:2508.19764)** — https://aclanthology.org/2025.emnlp-main.1364/ — 9 models × 27 tasks. **Task-relevant/expert personas → positive or neutral; task-irrelevant attributes → drops of nearly 30 points.** Formalizes the exact taxonomy crucible needs. → **Design implication:** "Solver-7 of 12" is fine ONLY as a task label; never attach an identity ("a competitive 28-year-old engineer") — that's the degrading axis.

4. **Cao et al. 2026, "From Biased Chatbots to Biased Agents" (arXiv:2602.12285)** — https://arxiv.org/abs/2602.12285 — The −26.2% figure is driven specifically by **task-irrelevant demographic** role cues on agentic benchmarks. Confirmed: it does NOT test competitive *performance* signals. → **Design implication:** Rebuts the prior swarm's overreach. Cao bans demographic personas, not standings. Standings survive.

5. **Song et al. 2025, "Reward Is Enough: LLMs Are In-Context Reinforcement Learners" (arXiv:2506.06303)** — https://arxiv.org/abs/2506.06303 — LLMs optimize **scalar numeric reward feedback in-context** without weight updates, beating Self-Refine/Reflexion on Game-of-24, ScienceWorld, Olympiad math. → **Design implication:** Quantitative self-state ("your solve: 6 tool calls vs. record 4") is a *usable gradient signal*, behaving differently from identity framing — KEEP and emphasize the numbers.

6. **Shinn et al. 2023, "Reflexion: Verbal Reinforcement Learning" (arXiv:2303.11366)** — https://arxiv.org/abs/2303.11366 — A scalar performance score converted to verbal feedback acts as a "semantic gradient," monotonically improving agent performance across episodes. → **Design implication:** Frame standings as actionable feedback ("fastest solve was 4 calls — beat it"), not as a status verdict.

7. **DeepMind answer-flipping-under-pressure study (2026)** — *(cited by the agent via Computerworld coverage — primary source to be confirmed in verification pass)* — Models structurally **abandon correct answers under confident social pressure** (sycophancy/capitulation); third-person framing cuts this up to 63.8%. *(Transfer caveat: social-pressure, not human-anxiety transfer.)* → **Design implication:** Never frame competition as another agent *asserting* a different answer or *pushing back* — that triggers capitulation, the one competitive form that genuinely degrades.

8. **MultiAgentBench, Zhu et al. 2025 (arXiv:2503.01935)** — https://arxiv.org/abs/2503.01935 — Competitive ("conflicting-goal") vs. cooperative conditions both produce measurable, scoreable agent behavior; competition is a legitimate task structure with its own KPIs, not noise. *(Transfer caveat: multi-agent benchmark, not single-Solver.)* → **Design implication:** Competition-as-task-structure is validated; keep the standings as a scored surface.

## Verdict + the line

**The line is feature-pointing vs. self-pointing.** A competitive cue helps when it directs attention to *the task's verifiable features* (your tool-call count, your solve-rate on the metric, the elegance record to beat — all quantitative, all about the work). It hurts when it directs attention to *the self* (identity, demographics) or *capitulation* (another agent confidently disputing you). Cao bans the first failure mode; the answer-flipping work bans the second; EmotionPrompt + Reward-Is-Enough explain why the metric form actively helps.

**The form Crucible SHOULD use (not delete):** Keep "Solver-7 of 12" as a bare task label. Express standings as **quantitative, actionable, self-referential metrics**: *"Current best solve: 4 tool calls. Yours so far: 6. Graduation threshold: solve under 5."* This is EmotionPrompt's attention boost + Reflexion's semantic-gradient + Reward-Is-Enough's in-context scalar — all three confirmed-helpful axes stacked. **Strip:** any demographic/personality identity on the Solver, any "rival agent insists the answer is X" pushback dynamic, and any stacked-intensity pressure pileup (cap at one stakes clause). That is the evidence-grounded competitive frame.

---

## ⚠ Verification corrections (second pass — 2026-06-01)

The load-bearing claims verified: **Principled Personas** (Luz de Araujo et al. 2025, arXiv:2508.19764) confirmed — task-irrelevant attributes drop ~30pt, task-relevant don't (this is the line); **Song 2025 Reward-Is-Enough** (arXiv:2506.06303) confirmed (in-context scalar reward, Game-of-24 90% vs 44–47%); Li EmotionPrompt, Cao 2026, MultiAgentBench, Reflexion verified. Corrections:

- **Finding 2 (Patel 2026, arXiv:2604.07369) — MIS-FRAMED:** the paper is real but does **not** report a non-monotonic intensity effect. Its actual finding: positive emotional stimuli improve accuracy / reduce toxicity but **increase sycophancy** (a trade-off). So the "cap the intensity" implication is not anchored here. Patel's *real* lesson is directly relevant though: motivational/stakes stimuli can raise **sycophancy** — another reason to keep stakes off the scored surface (see swarm-25). The "one stakes clause, don't pile on" guidance remains prudent but should be presented as design caution, not a Patel finding.
- **Finding 7 (answer-flipping under social pressure) — UNVERIFIED:** the primary source was cited via secondary coverage and was NOT confirmed in the verification pass. The "third-person framing cuts capitulation 63.8%" figure must not be cited until the primary paper is located. The qualitative guidance ("don't frame competition as a rival confidently asserting a different answer") is independently supported by the broader sycophancy/capitulation literature and stands.
