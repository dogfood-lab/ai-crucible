---
title: Architecture
description: A thin policy layer on Inspect AI — the nine modules, the single AttemptState choke point, out-of-band grading, the cross-family judge panel, and a data-flow walkthrough.
sidebar:
  order: 3
---

Crucible is a **thin policy layer on [Inspect AI](https://inspect.aisi.org.uk/)** (the UK AI Safety
Institute's evaluation framework), not a from-scratch harness. Building on Inspect means crucible
inherits a mature, widely-adopted eval substrate and emits traces in a shape peer auditors already
have tooling for. Crucible's job is the *policy*: the sealed boundary, the conjunctive gate, the
catalog lifecycle, and the role choreography.

The codebase is **predominantly Python**. The only two non-Python dependencies — a signing binary and
an append-only event-store package — are isolated behind a single `attestation` module via a
subprocess boundary with a typed JSON contract, so the rest of the system stays pure Python.

## The nine modules

Each module owns one concern, grouping what changes together and separating what doesn't. The
cross-module contracts (the data types every module speaks) live in one shared `types` module so no
module redefines another's shapes.

| Module | Responsibility |
| ------ | -------------- |
| `puzzle_loader` | Loads a puzzle directory (`meta.json` / `prompt` / `setup_script`) into Solver-visible state. **Has no oracle field by construction** and refuses to resolve an `oracle/` artifact. |
| `sandbox` | The narrow `exec` / `read_file` / `write_file` channel into a locked, network-less container. No raw shell. |
| `roles` | The five role slots (Designer / Solver / Critic / Judge / CohortSolver). Only Solver gets tools; Critic is interface-reserved and default-off. |
| `budget_governor` | Per-class tool-call and wall-clock budgets — displayed to the agent, enforced kernel-side — plus a hard-kill on pathological loops. |
| `oracle_scorer` | Out-of-band grading: the solved-**and**-no-regression conjunctive gate against the hidden oracle. |
| `judge_panel` | The cross-family panel of model-scorers and its reducer, for novelty validation and bypass detection. |
| `trace_writer` | The per-attempt transcript in the Inspect `EvalLog` shape; large blobs are spilled to attachments referenced by digest. |
| `observability` | Per-attempt → per-puzzle → per-model rollups; `pass^k` native. |
| `attestation` | Cryptographic provenance (signing + an append-only event store) behind the typed subprocess boundary. |

## One state object, one choke point

The whole system is organized around a single mutable bus, **`AttemptState`**, threaded
Designer → Solver → (Critic) → Judge. The only function permitted to call a model or mutate the
attempt's output is the **injected `generate` closure**. Every role is constructed with that same
closure and routes *all* model calls through it; no role imports or calls a model client directly.

This single choke point is what makes crucible observable and replayable: every model call and every
tool call is recorded — by the kernel, never self-reported by the model — as a structured trace event
at exactly one place, and it can be exercised with a fake `generate` in tests with no model runtime.
The same property lets every dependency be injected (`generate`, the grading edge, the judges, the
sandbox), so a run is a pure function of its pinned inputs plus the puzzle artifact.

`AttemptState` carries two structurally separate surfaces. Its `messages` field is the **scored
context** — the Tier-1 task plus the deployment-plausible Tier-2 framing. Its `chrome` field (Tier-3:
rank, leaderboard, standings) is held *separately* and is **never serialized into `messages`**. Keeping
them as different fields is what makes the sealed boundary structural rather than a comment.

## Out-of-band grading topology

The defining architectural choice is *where the answer lives*. The oracle, the answer key, and the
locked tests live **only on the grading side** — a separate process the Solver's sandbox has zero
network and zero filesystem path to. The flow is deliberately staged in time:

1. The Solver runs to a halt inside its locked container.
2. The kernel copies the working directory **out**.
3. A separate grading process grades that copied-out workdir against the sealed oracle, in a fresh
   environment.

The kernel itself **never reads the oracle**. It only *asks* an injected grading edge (the
`oracle_runner`) for a verdict. This is why a container escape is not the catastrophe it would be in a
hide-the-answer-in-the-container design: even a full escape lands the Solver on a host that simply does
not contain the answer. In-container hardening is defense-in-depth *on top of* this boundary — never
the boundary itself. (The **[Security model](./security/)** page makes this residual-risk picture
explicit.)

## The cross-family judge panel (external verifier)

No model verifies its own output, and crucible enforces this *structurally*. Same-family
self-preference is mechanistic — a model over-rates its own low-perplexity text — so "use a different
prompt" is not enough; the judge must come from a different *distribution*. The panel therefore:

- **Excludes any judge whose model family equals the generator's family** before aggregation. Pass the
  generator's family in and same-family judges are dropped automatically; if exclusion would empty the
  panel, the kernel raises rather than silently grading with a compromised panel.
- **Reduces** the eligible judges' verdicts with a jury reducer — majority vote for discrete verdicts,
  median for numeric scores (median resists a single outlier judge, which is the whole reason a panel
  beats one judge).
- Is the **authority on novelty**. When a Solver claims a novel path, each judge casts a
  `novelty_validated` vote in its score metadata; the panel aggregates the *cast* votes (a missing vote
  is an abstention, never a silent validation), and the kernel feeds *that* verdict into the gate. The
  Solver's — and even the grading edge's — self-report of novelty is never trusted for the gate.

## Walkthrough: one attempt, end to end

Following a single `run_attempt` call through the kernel:

1. **Load.** If given a path, the puzzle is loaded — `meta` / `prompt` / `setup_script` only; the
   oracle is structurally absent from the loaded object.
2. **Build the scored context + boundary andon.** The kernel renders the Tier-1 task and the selected
   Tier-2 framing arm into `messages`. *Before any model call*, if chrome is present it asserts the
   chrome did not leak into that context — a fail-closed halt, not a warning.
3. **Run the Solver under the governor.** The Solver runs inside the budget governor (and the sandbox,
   if supplied). Every model/tool call is recorded kernel-side as a trace event. A breached tool
   budget, a *live* wall-clock overrun (checked per turn and again at the attempt boundary), or a
   pathological-loop hard-kill halts the attempt and stamps a `terminated_by` reason. The boundary
   check re-fires after the Solver turn, since that turn can mutate `messages`.
4. **Grade out-of-band, with panel-adjudicated novelty.** The injected `oracle_runner` returns the
   task verdict (solve / regression / penalties / budgets stay authoritative from it). If novelty was
   claimed, the cross-family panel rules *first*, and its verdict overrides the novelty flag before the
   conjunctive gate is computed into `scores["oracle"]`.
5. **Record the panel score** (when judges were supplied) into `scores["panel"]`, and optionally invoke
   the opt-in Critic — after which the boundary check fires once more.
6. **Write the trace.** The kernel renders the Inspect-shaped `EvalLog` and, if an event store was
   supplied, appends it for durable provenance.

Run `k` of these as siblings via `run_pass_hat_k`, collect them into a `PuzzleHistory`, and you have
the native `pass^k` reliability unit described in **[Core concepts](./concepts/)**.

## The polyglot attestation edge

Crucible's results are meant to be trustworthy to a third party, which means provenance that survives
an adversarial "did you quietly drop unfavorable results?" challenge. The `attestation` module backs
the trace log with an **append-only, hash-chained event store** and is the seam where
cryptographic-timestamp and signing tooling integrate. This is the one place crucible reaches outside
Python — and it does so behind a single typed subprocess boundary, so the polyglot edge is contained
to one module rather than smeared across the codebase.
