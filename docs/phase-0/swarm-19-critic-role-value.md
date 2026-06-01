# Swarm Agent 19 — Marginal Value vs. Cost of an Adversarial Critic / Debate Stage

**Date:** 2026-06-01
**Source:** Fourth study swarm dispatch (Phase-1-prep, research-grounded advisor protocol)
**Question:** Is there 2024–2026 evidence quantifying the gain (accuracy/reliability) and cost (latency/token/$) of adding an adversarial Critic / debate stage to an LLM judging pipeline, vs single-judge and vs jury/PoLL? Decide whether crucible's Phase 1 role contract should make Critic first-class, optional, or a deferred stub.

---

# Findings: Marginal Value vs. Cost of an Adversarial Critic / Debate Stage in LLM Judging (2024–2026)

**(a) Does debate reliably improve JUDGMENT accuracy? Mixed — it's task-conditional, and the canonical "yes" doesn't generalize.**

1. **Kenton et al. 2024**, *On scalable oversight with weak LLMs judging strong LLMs* — [arXiv:2407.04622](https://arxiv.org/abs/2407.04622). Debate beat consultancy on all tasks, and beat direct-QA **only on extractive QA with information asymmetry** (judge can't see the source); on math, coding, logic, and multimodal reasoning (no asymmetry) debate's advantage over plain direct-QA was **inconsistent**, and stronger-debater gains were "more modest than in previous studies" — a direct, partial walk-back of Khan 2024.

2. **Sternlicht et al. 2025**, *Debatable Intelligence: Benchmarking LLM Judges via Debate Speech Evaluation* — [arXiv:2506.05062](https://arxiv.org/pdf/2506.05062). Judging argument strength is a distinct, harder competency than QA/summarization judging — relevant because a Critic role asks the Judge to evaluate adversarial argumentation, not just answers.

**(b) Single-judge vs. debate vs. jury — accuracy AND cost.**

3. **Hu et al. 2025**, *Multi-Agent Debate for LLM Judges with Adaptive Stability Detection* — [arXiv:2510.12697](https://arxiv.org/html/2510.12697v1). Debate over single-judge: **+5.15pp** (LLMBar), **+4.81pp** (TruthfulQA); but **zero gain on simpler tasks** (JudgeAnything tied majority vote; one MLLM case slightly *underperformed*, 60.38 vs 60.88). Verdict: *"debate excels where accuracy justifies costs; [majority vote] suffices for straightforward tasks."* Adaptive early-stopping halted at 2–7 of 10 rounds with <1pp loss — i.e., naïve fixed-round debate over-spends.

4. **Cost multiplier (consistent across the literature):** multi-agent debate consumes **~3–5× the tokens** of single-pass CoT for **+1.5–5.3pp** accuracy ([S2-MAD, NAACL 2025](https://aclanthology.org/2025.naacl-long.475.pdf); see also [GroupDebate, arXiv:2409.14051](https://arxiv.org/pdf/2409.14051)). A Critic stage is not a marginal add-on; it roughly multiplies per-item judging cost.

**(c) NEGATIVE / NULL results — flagged, and decisive here.**

5. **Estornell et al. 2025**, *Talk Isn't Always Cheap: Understanding Failure Modes in Multi-Agent Debate* — [arXiv:2509.05396](https://arxiv.org/html/2509.05396v1). Debate **lowered** accuracy even when stronger models outnumbered weaker ones: **−9.2pp** (MMLU, 3×Mistral), **−12.0pp** (MMLU mixed), **−5 to −8pp** (CommonSenseQA). Mechanism: correct→incorrect flips exceed incorrect→correct — agents favor agreement over challenging flawed reasoning.

6. **2026**, *Single-Agent LLMs Outperform Multi-Agent Systems Under Equal Thinking-Token Budgets* — [arXiv:2604.02460](https://arxiv.org/html/2604.02460v1). The methodological gut-punch: at **matched token budget**, a single agent matches or beats debate on multi-hop reasoning; "many reported advantages…are better explained by unaccounted computation…rather than architectural benefits." Most pro-debate numbers compare unequal compute.

**(d) When a Critic pays off:** information-asymmetry tasks (judge lacks ground truth the debaters access), high-ambiguity/high-initial-variance items, and genuine adversarial settings. Wasted on: low-ambiguity scoring, reasoning tasks where the Judge could just think longer, and any case where matched single-judge compute is the honest baseline (D3 — [arXiv:2410.04663](https://arxiv.org/pdf/2410.04663) — reports SOTA human-agreement + bias reduction but, tellingly, only with *budgeted stopping*).

**(c′) Sycophancy/conformity mitigation if you do build it** — [*Peacemaker or Troublemaker*, arXiv:2509.23055](https://arxiv.org/pdf/2509.23055) and [CONSENSAGENT, ACL Findings 2025](https://aclanthology.org/2025.findings-acl.1141/). RLHF-induced sycophancy makes minority-correct agents conform to incorrect majorities (echo chamber); **anonymizing speaker identity nearly eliminates** identity-driven sycophancy.

---

## Recommendation for Crucible's Phase 1 role contract: **Optional stage** (interface-reserved, default-off, gated).

Make Critic a **first-class slot in the role-interface contract but a non-default, toggleable stage** — not a fully built first-class role, not a pure deferred stub. Evidence: the *value* of a Critic is real but **narrow and conditional** (Hu 2025: +5pp on hard/ambiguous items, ~0 on easy ones; Kenton 2024: gains concentrate on information-asymmetry tasks), while the *cost* is a flat **3–5× token multiplier** (S2-MAD) AND the *risk* is a measurable accuracy *regression* via conformity/answer-flipping (Estornell 2025: up to −12pp) that worsens precisely when the Solver is strong and the Critic/Judge weaker — Crucible's frontier-model setting. Worse, matched-budget analysis (2604.02460) suggests much of debate's apparent edge is just extra compute a stronger single Judge could spend directly. So: define the Critic message schema now (cheap, prevents a Phase-2 contract break), wire it as a per-puzzle opt-in stage, and **default it off**, routing to single-Judge or jury/PoLL. **Revisit when** you have logged Crucible puzzles showing (i) high Judge disagreement/variance or low human-agreement on a task class, or (ii) genuine Solver-Judge information asymmetry — the two conditions the literature says actually convert Critic spend into reliability. If you do enable it, anonymize debater identity (**D3, arXiv:2410.04663** — see correction below) and use adaptive early-stopping (Hu 2025), and always benchmark against a **token-matched** single Judge, never an unmatched one.

---

## ⚠ Verification corrections (second pass — 2026-06-01)

All IDs resolve (including the 2026 future-IDs). The "Critic default-off" decision rests on solid, verified negative evidence — the load-bearing numbers all check out. Corrections to attribution/figures before this is locked into architecture:

- **Finding 5 (arXiv:2509.05396) — WRONG AUTHOR:** the paper "Talk Isn't Always Cheap" is by **Wynn, Satija & Hadfield 2025**, *not* "Estornell." The negative effect sizes (MMLU 3×Mistral −9.2pp, MMLU mixed −12.0pp, CSQA −3 to −8pp) are confirmed from Table 1. Use the correct authors everywhere.
- **Finding 4 (arXiv:2604.02460):** real; correct title is *"Single-Agent LLMs Outperform Multi-Agent Systems on **Multi-Hop Reasoning** Under Equal Thinking Token Budgets"* by **Tran & Kiela 2026**. Scope is **multi-hop reasoning specifically** — note the scope when generalizing.
- **Finding 4 token-cost / S2-MAD (= arXiv:2502.04790, Zeng et al., NAACL 2025):** the "~3–5× tokens" figure is **materially wrong** — S2-MAD Table 1 shows multi-agent debate at **~60–80× CoT tokens** (e.g. GSM8K 0.25k→20.4k); S2-MAD's *own* contribution is cutting that by up to 94.5%. Debate is far **more** expensive than the brief stated, which only *strengthens* default-off.
- **Finding 6 (arXiv:2509.23055, Yao 2025, "Peacemaker or Troublemaker"):** clause 1 (RLHF-sycophancy → minority-correct agents conform, r=0.902) is supported. Clause 2 (**anonymizing identity nearly eliminates identity-driven sycophancy**) is **not in this paper** — that mechanism belongs to **D3 (arXiv:2410.04663)**. The "anonymize debater identity" recommendation is correctly anchored to D3 above.
- **Finding 2 (Hu 2025, arXiv:2510.12697):** adaptive early-stop range is **2–8** rounds (brief said 2–7); +5.15pp/+4.81pp figures confirmed.
- **Finding 1 (Kenton 2024):** "partial walk-back of Khan 2024" is our gloss, not a stated in-paper claim — keep it framed as interpretation.
