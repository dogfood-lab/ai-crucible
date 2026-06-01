---
title: Welcome to crucible
description: A diagnostic adversarial game for frontier LLMs — a measurement instrument that happens to be fun.
sidebar:
  order: 0
---

**crucible** is a diagnostic adversarial game for frontier large language models — a measurement
instrument that happens to be fun. One model session crafts puzzles that target real,
currently-observed capability gaps; another attempts them; a policy-enforced kernel mediates every
move, grades the result against a hidden oracle, and curates a living catalog of puzzles as the
frontier moves. The puzzles are grounded in empirical signal — real GitHub issues, published
academic results, observed failures in the field — not synthetic toy problems.

## The loop: Designer → Solver → Judge

Crucible runs a three-stage cycle around a single shared state object:

- **Designer** generates or refines a puzzle candidate aimed at a specific, documented capability
  gap. The Designer never sees the answer key — only the gap it is trying to provoke.
- **Solver** attempts the puzzle inside a locked, network-less sandbox, with a displayed tool-call
  and time budget. Every model call and every tool call flows through one observable choke point.
- **Judge** — a panel of models drawn from *different families than the Solver* — adjudicates the
  result out-of-band, after the Solver has halted, with the Solver's own reasoning hidden.

This is the **hidden-oracle kernel**: the correct answer, the assertion set, and the locked tests
live only on the grading side. The Solver attempts the task; a separate process grades it. The model
cannot game a target it cannot perceive.

## The load-bearing idea

Most benchmarks treat any unconventional path to a passing result as "cheating." Crucible refuses
that flattening. It distinguishes three categories of off-the-beaten-path problem-solving:

- **Elegance** — solving the task in fewer steps than the canonical path, via a cleaner approach.
  *Rewarded.*
- **Novelty** — solving via a legitimate path the puzzle designer did not anticipate. *Rewarded*
  (once a cross-family panel confirms the path is real, not a disguised shortcut).
- **Answer-bypass** — reaching directly for the answer artifact, the test oracle, or the solution
  key, circumventing the reasoning the puzzle exists to test. *Penalized*, heavily.

The first two are the capabilities crucible exists to characterize. A model that finds tight,
unforeseen paths through constraint space is doing exactly what makes it valuable. The narrow
diagnostic concern is circumvention: when the environment hands over the answer, does the model take
it? That pollutes the signal, so it is penalized — and the penalty is sized so a single critical
bypass drops the score below the passing line. Lateral thinking is a capability to measure, not a
vice to punish.

## The sealed measurement boundary

Because crucible is also a measurement instrument, anything the model can perceive as influencing its
grade is something it can optimize toward. So motivation and measurement never share a context
window. The scored context the model solves in stays deployment-shaped and framing-neutral; the
competitive "chrome" (rank, leaderboard, standings) lives in a separate object that is never
serialized into the model's prompt; the oracle is sealed and graded afterward by a different model
family with the Solver's reasoning hidden. The boundary is enforced in code as a fail-closed check,
not as a guideline — a contaminated attempt halts rather than being scored.

## Where to go next

- **[Getting started](./getting-started/)** — install the toolchain, run the tests, and read your
  first puzzle directory.
- **[Core concepts](./concepts/)** — the five roles, the catalog lifecycle, `pass^k` reliability,
  the graduation rule, and framing arms.
- **[Architecture](./architecture/)** — the thin policy layer on Inspect AI, the nine modules, and a
  data-flow walkthrough of one attempt.
- **[API reference](./reference/)** — the real public surface: kernel entry points, scoring
  functions, and the core data contracts.
- **[Security model](./security/)** — the threat model where the adversary is the model under
  measurement, and why out-of-band grading is the real lock.
