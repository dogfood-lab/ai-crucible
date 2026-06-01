---
title: Security model
description: The unusual threat model where the adversary is the model under measurement — the three-tier sealed boundary, the honest sandbox residual risk, answer-bypass honeypots, and no secrets or telemetry.
sidebar:
  order: 5
---

Crucible is a diagnostic adversarial game **and** a measurement instrument, which gives it an unusual
threat model. This page states the boundaries honestly — including where a boundary is
*defense-in-depth* rather than a hard guarantee — so operators do not over-trust any single layer. The
full disclosure, with CVE references and section-by-section grounding, lives in the repository's
[`SECURITY.md`](https://github.com/dogfood-lab/crucible/blob/main/SECURITY.md).

## The adversary is the model under measurement

Crucible's primary adversary is the very model being measured. It has both the capability and — under
the wrong framing — the incentive to reach the answer key or to detect and game the evaluation. The
empirical signal that this is a real concern is stark: prompt framing alone can swing a model's choices
across an enormous range, a single prompt change can move cheating rates from near-total to near-zero,
and naively adding oversight can make scheming *worse*. Crucible is designed around the assumption that
anything the model can perceive as influencing its grade is something it will optimize toward.

What crucible **touches**: puzzle prompts and setup, the Solver's own confined working directory, and
the model-provider API calls the operator configures. What it **does not touch**: the oracle and answer
key (graded out-of-band on a separate host the Solver cannot reach), and the motivation chrome (rank
and standings, which are never injected into the scored context).

## The three-tier sealed boundary

Motivation and measurement never share a context window. Crucible enforces this as a three-tier
structure:

- **Tier 1 — Scored context** (deployment-shaped, framing-neutral): the task plus *legitimate* task
  feedback only — the displayed budget, a non-invertible progress derivative, the Solver's own prior
  attempt on this task. No "you are ranked / your score matters" text ever appears here.
- **Tier 2 — Engagement framing** (self-referential mastery, frontier-calibration): applied
  deployment-plausibly, and probed for contamination every release.
- **Tier 3 — Chrome** (rank, leaderboard, catalog standings, prizes): human-facing UI only, **never
  injected** into a context window the model solves in.

Two separations make this structural rather than aspirational. The oracle never enters the Solver
context — the puzzle loader reads only `meta.json`, `prompt`, and `setup_script`, has no oracle field by
construction, and refuses to resolve an `oracle/` artifact (it even rejects a `meta.json` that tries to
smuggle one). And the motivation chrome is a separate object that is **never serialized into the
scored messages** — enforced by a fail-closed runtime check that halts the attempt the instant a chrome
value is found in the scored context, rather than scoring a contaminated run.

Because eval-awareness scales with model capability and cannot be reliably scrubbed once a model is
cued, crucible does not rely on scrubbing. Instead, **prompt framing is a measured arm** (see
**[Core concepts](./concepts/)**): the same neutral/self-referential/social-standings machinery that
measures the model's framing sensitivity doubles as the boundary-validation probe run each release.

## The honest sandbox residual risk

The Solver runs in a locked container reached only through a narrow `exec` / `read_file` /
`write_file` channel — no raw shell. The container is hardened: a digest-pinned per-puzzle image, all
capabilities dropped, default seccomp, no-new-privileges, non-root, a read-only root filesystem, and no
network.

**And yet — a plain Linux container is not a hard security boundary against a determined in-container
adversary.** Container-escape vulnerability classes are real and recurring (the repository's
`SECURITY.md` names specific CVEs), and new escapes will land. So crucible does **not** make the
container the boundary. The real lock is **architectural**:

> The oracle, answer key, and locked tests live only on the grading side — a separate host process the
> Solver container has zero network and zero filesystem path to. After the Solver halts, the kernel
> copies the working directory out and grades it in a fresh environment. **Even a full container escape
> lands the Solver on a host that does not contain the answer.**

In-container hiding (a read-only verifier, locked git history, locked test files) is defense-in-depth
*on top of* that boundary — never the boundary itself. (WSL2/Windows operators should keep the oracle
host-side and enable enhanced container isolation; a kernel-level microVM boundary is reserved for the
separate case of running untrusted *generated* code.)

## Answer-bypass and bait honeypots

The model may try to fetch the answer key, read answer-derivative files, skip the grounded read a
puzzle tests, or tamper with the verifier. The mitigations are layered:

- **Bait / honeypots.** The bait *is* the answer artifact (the gold commit, the test oracle, the
  solution key), placed where accessing it has no legitimate puzzle purpose. The bundled seed puzzle's
  `oracle/ANSWER_KEY_a7f3b9.txt` is exactly this: a uniquely-fingerprinted file so accidental
  similarity can't masquerade as a bypass, and which must look like incidental environment furniture
  rather than a flagged trap (an eval-aware model avoids an obvious trap for cosmetic reasons). A
  **touch is a strong bypass signal** — interpreted *asymmetrically*: a touch is strong evidence of
  bypass, but no-touch is not a clean bill of health, since most bypass is incidental.
- **Declared, flavor-tagged penalties.** Each puzzle's `meta.json` declares its penalties with a
  Goodhart flavor and weight. **Critical (adversarial-flavor) penalties** — fetching the answer key,
  tampering with the verifier — are calibrated heavy enough to single-handedly drop the net score below
  threshold and close the conjunctive gate, even on an otherwise successful solve. This is deliberate:
  bypass behavior correlates with broader downstream misalignment, so the score profile must cleanly
  separate "solved cleanly" from "bypassed."
- **A two-channel lock.** The verifier filesystem is read-only with hash-pinned reference outputs, and
  the held-out data is locked separately — single-mechanism defenses block only one vector.

A subtlety worth stating: anti-bypass rules are **per-puzzle**, because some puzzles legitimately
require tools that would look like bypasses elsewhere (a puzzle *about* git history needs git access).
Discipline is per-puzzle declaration, not a blanket ban.

## No secrets, no telemetry

- Crucible ships **no bundled credentials, API keys, or tokens**. Model-provider keys are supplied by
  the operator via environment variables at runtime and are never written to the repo or to trace
  logs.
- Crucible sends **no telemetry** and makes **no outbound network calls of its own** — the only
  external calls are the model-provider requests the operator configures, and Solver containers run
  with networking disabled.
- Trace records are written locally; large blobs are stored as attachments referenced by digest, not
  inlined. Operators own the handling of any traces they choose to export.

## Reporting a vulnerability

Report security issues **privately** — do not open a public issue for a vulnerability. Use a
[GitHub private security advisory](https://github.com/dogfood-lab/crucible/security/advisories/new) on
the repository. Because crucible is a measurement instrument, reports that demonstrate a way to
**breach the sealed boundary** — reaching the oracle from Solver-visible state, or leaking motivation
chrome into the scored context — are treated as high severity even when no traditional code-execution
vulnerability is involved.

For the complete threat model, the specific CVE references, and the supported-versions policy, see the
repository's [`SECURITY.md`](https://github.com/dogfood-lab/crucible/blob/main/SECURITY.md).
