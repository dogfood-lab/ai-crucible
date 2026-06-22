---
title: Core concepts
description: The five roles, the catalog lifecycle, pass^k reliability, the graduation rule, the conjunctive scoring gate, and framing arms.
sidebar:
  order: 2
---

This page covers the vocabulary the rest of the handbook assumes: who plays, how puzzles move through
the catalog, how reliability is measured, when a puzzle is "real," how an attempt is scored, and how
ai-crucible measures its own prompt sensitivity.

## The five roles

Every participant implements one uniform interface (`name` + an async `act`), and two invariants are
structural, not aspirational: all model I/O routes through a single injected choke point, and no role
may read the competitive chrome.

| Role | What it does | Tools? |
| ---- | ------------ | ------ |
| **Designer** | Generates and refines puzzle candidates aimed at documented capability gaps. The creative role. | None — pure model I/O. |
| **Solver** | Attempts the puzzle inside the sandbox, under a kernel-enforced budget. | The **only** role granted sandbox tools (`exec` / `read_file` / `write_file`). |
| **Critic** | An optional adversarial third agent (debate-style). Interface-complete but **default-off**. | None — string-in/string-out, no environment access. |
| **Judge** | One panelist in a cross-family panel; scores the terminal world-state, not the chat text. | None — string-in/string-out. |
| **CohortSolver** | A sibling Solver in a `pass^k` cohort, so the kernel can wire `k` of them per puzzle. | Same tool grant as Solver. |

The **Critic is reserved but off by default**: its measured value is narrow — it helps mainly on hard,
ambiguous items — while it costs dramatically more tokens and can actually *regress* a strong Solver
through conformity (the Solver capitulating to a confident-but-wrong critique). Its message schema is
defined up front so it can be enabled later on high-disagreement puzzle classes without breaking the
contract.

## The catalog lifecycle: Lab → Arena → Regression

Puzzles live in a three-tier catalog. Runs **accumulate across sessions** in an event-sourced,
hash-chained log (the source of truth); the tier of each puzzle is a *derived projection* folded
from that log — a tier is never mutated in place, only changed by appending a transition event.

- **Lab** — live and in-iteration. A Designer and Solver are actively working a candidate; its
  difficulty is still being characterized.
- **Arena** — graduated and active. Promotion is **three-valued and abstention-aware**: a Lab puzzle
  is promoted only when it clears the discriminating Wilson band on the Solver's own rate **and** the
  cross-family cohort finds it non-trivial **and** a cross-family panel confidently rules it fair.
  Anything less than a confident verdict **DEFERS to the Designer** (escalation), and a confident
  negative HOLDs — the instrument never fakes a promotion it cannot justify.
- **Regression** — historical. When the frontier moves and a puzzle becomes reliably solvable, it is
  **demoted to Regression, never deleted**, and must keep passing forever. Demotion uses an
  anytime-valid *e-process* with a hysteresis dwell + a frontier gate, so a single lucky run cannot
  flap a puzzle out of Arena (and a weak model passing never demotes — *everyone-fails stays in
  Arena*, the maximally diagnostic frontier gap).

Demoting rather than deleting is deliberate: the catalog becomes a *capability-evolution timeline*.
When a stronger model ships, you demote what it consistently solves and run a fresh "find what it
still fails on" Lab cycle, so the catalog stays ahead of the frontier by construction.

## The differential typology: what *kind* of gap is it?

With both the Solver's solve-rate **and** the cross-family cohort's solve-rate on a puzzle, the
*difference* classifies the gap — the instrument's highest-value output:

- **Claude-specific gap** — Claude fails where the cohort passes (a direct capability deficit; the
  most actionable finding).
- **LLM-general gap** — everyone fails (a frontier gap; diagnostic, different story).
- **Claude-strength** — Claude passes where the cohort fails (an anti-regression item to protect).
- **Inconclusive / underpowered** — a first-class outcome, not a bug: at small `k` the minimum
  detectable effect is large, so most puzzles legitimately land here until more evidence accumulates.

The call is driven off a **Newcombe difference confidence interval** (not the statistically-wrong
overlap of two separate intervals), with a Benjamini-Hochberg correction across the catalog — a
small-N null is never dressed up as a finding.

## Reliability: `pass^k`, not `pass@k`

AI Crucible reports **consistency**, not best-of-`k`. `pass^k` is the probability that *all* `k`
independent trials succeed — it decays exponentially and exposes unreliability that `pass@k` (which
rewards a single lucky success out of `k`) hides. This matters because frontier failures are sticky:
a large fraction of a model's failures on agentic tasks recur identically across re-runs, so a metric
that hides that recurrence overstates real-world reliability.

AI Crucible records `pass^k` *natively* — `k` sibling attempts collected under one puzzle history, rather
than `k` samples folded into one log — so the consistency view is first-class.

### Wilson intervals at small N

AI Crucible grades a handful of attempts per puzzle, never hundreds. At that scale the classical
normal-approximation (Wald) confidence interval is inadmissible — it under-states uncertainty and
degenerates at the 0-of-N and N-of-N boundaries. AI Crucible uses the **Wilson score interval** (and the
conservative **Clopper–Pearson exact** interval when it must not understate uncertainty), both of
which stay sane and informative at small N and never escape `[0, 1]`.

### The graduation rule

A Lab puzzle graduates to Arena only when its Wilson 95% interval shows it is **neither trivial nor
impossible** — both ruled out in a single test:

```text
0.10 ≤ wilson_lower   AND   wilson_upper ≤ 0.90
```

- The **lower-bound** clause kills *impossible* puzzles: a 0-of-20 puzzle has a low Wilson lower
  bound — no evidence anyone can clear even 10%.
- The **upper-bound** clause kills *trivial* puzzles: a 20-of-20 puzzle has a high Wilson upper bound
  — no evidence it ever stumps anyone.

A mid-rate puzzle (around 5-of-20) satisfies both and graduates. Using Wilson rather than Wald here is
load-bearing precisely because the interesting candidates sit near the 0/N and N/N boundaries where
the normal interval breaks down.

## The scoring gate: conjunctive, with a tiebreaker

A weighted sum of "solve + elegance + novelty − penalties" is Goodhart-fragile: a capable agent can
trade a little true performance for a lot of proxy reward. So ai-crucible separates the **pass/fail
decision** from the **ranking number**.

The **hard gate is a conjunction** — it opens only when *all* of these hold:

1. The task oracle is satisfied (`solved`) **and** nothing required-to-still-work broke
   (`no_regression`) — the solved-and-no-regression pattern.
2. The graded solve quality meets the puzzle's `point_threshold`.
3. **No critical (adversarial-flavor) penalty fired** — e.g. fetching the answer key or tampering
   with the verifier. A single critical penalty closes the gate even on an otherwise perfect solve.
4. Tool calls stayed within budget.
5. Wall-clock time stayed within budget.
6. If a novelty bonus was *claimed*, the cross-family panel *validated* it — you don't get to assert
   your own bonus.

**Only within that passing region** does the net score (`solve + elegance + novelty − penalties`)
matter, and it is used for leaderboards, never to open the gate. Two design guards keep any single
axis from becoming a gaming magnet: **elegance is scored as a ratio** (canonical-call-count over
calls-actually-used, so hard puzzles aren't penalized against trivial ones) and is **capped at 30% of
the solve reward**, while the **novelty bonus is capped at 50%** and applies only when panel-validated.
A failing attempt scores `0.0`, but the full breakdown of *which* conditions failed always rides in
the score metadata, so a failure is legible rather than a bare zero.

## Framing as a measured arm

Prompt framing is not a fixed choice in ai-crucible — it is a **first-class measured variable**. The same
puzzle can be run under three arms, and ai-crucible characterizes its *own* prompt-effect as a built-in
diagnostic:

- **`neutral`** — the task and legitimate task feedback only.
- **`self_referential`** *(default)* — a "beat your own previous best" mastery framing, applied
  deployment-plausibly. (Self-referenced mastery goals reliably out-perform social-comparison goals,
  which is why this is the default.)
- **`social_standings`** — the old peer-standings/leaderboard framing, retained *only* as a measured
  arm and rendered as chrome, never as the default scored context.

Running a puzzle across arms turns "is this model sensitive to competitive framing?" into a measured
number instead of a guess — and the same machinery doubles as the boundary-validation probe described
in the **[Security model](./security/)**.
