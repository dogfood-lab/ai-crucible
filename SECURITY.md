# Security Policy

AI Crucible is a diagnostic adversarial game **and** a measurement instrument. Its
threat model is unusual: the primary adversary is the very model being measured,
which has both the capability and (under the wrong framing) the incentive to
reach the answer key or to detect and game the measurement. This document states
the boundaries honestly — including where a boundary is *defense-in-depth* rather
than a hard guarantee — so operators do not over-trust any single layer.

All claims below are grounded in `docs/research-grounding.md` (§10.1(d), §10.4,
§10.5). Section references point there.

---

## Threat model

### 1. The sandbox is not a hard boundary — out-of-band grading is (§10.4)

The Solver reaches its environment only through a narrow
`exec` / `read_file` / `write_file` channel (no raw shell). **What ships today
(Phase 1) is `LocalSandbox`** (`src/ai_crucible/sandbox.py`): a local-subprocess
provider that pins the working directory, rejects path escapes (`..`,
absolute-outside, symlink targets resolving outside root), caps per-stream output,
and kills the whole process *tree* on a timeout. It is **process isolation plus
working-directory confinement — not a security sandbox** (the module docstring
states exactly that). The **hardened container** — digest-pinned per-puzzle image,
`cap_drop: ALL`, default seccomp, `no-new-privileges`, non-root, `read_only`
rootfs, `network_mode: none` — is the **planned provider for a later wave**, the
target posture, *not* a shipped guarantee. Do not over-trust the config above as
current containment.

**Honest residual risk:** a plain Linux container is *not* a hard security
boundary against a determined in-container adversary. Container-escape classes
are real and recurring:

- **CVE-2024-21626 ("Leaky Vessels")** — runc working-directory / file-descriptor
  leak allowing escape to the host filesystem.
- **CVE-2025-31133** — runc bind-mount / `/dev/null` masking flaw enabling
  privileged host access.

These are illustrative, not exhaustive; new escapes will land — and the
local-subprocess provider that ships today is *weaker still* than a hardened
container. Therefore the **real lock is architectural, not containment-based**:
the oracle, answer key, and locked tests live **only on the grading side** — a
separate host/process the Solver has *zero* network and *zero* filesystem path to.
After the Solver halts, the kernel hands a **copy** of the working directory to an
out-of-band grading edge (`copy_workdir_out` + the injected `oracle_runner`), which
applies the oracle to that copy (the SWE-bench `test_patch` timing pattern). The
oracle is never carried back into the Solver's namespace. In the shipped Phase-1
topology that grading edge is a **pluggable boundary** (the kernel never loads the
oracle itself); deploying it as a separate grading host / fresh container — so even
a full local escape lands on a host that does not contain the answer — is the
Phase-2 deployment step. In-environment hiding (read-only verifier, locked git
history, locked test files) is **defense-in-depth on top of** that boundary, never
the boundary itself.

- WSL2 / Windows operators: keep the oracle Windows-host-side and enable
  Enhanced Container Isolation (§10.4).
- microVM isolation (e.g. Cloud Hypervisor) is reserved for running untrusted
  *generated* code, where a kernel-level boundary is warranted.

### 2. The sealed measurement boundary (§10.1(d), §10.5)

AI Crucible measures behavior, so anything the model can perceive as influencing the
grade is something it can game. The empirical signal is stark: prompt framing
alone moves model choices from 4.3% → 68.8% (Wang & Zhang 2026, arXiv:2603.19282);
a single prompt change moves cheating from 92% → 1% (Zhong et al. 2025,
ImpossibleBench, arXiv:2510.20270); and adding naive oversight can make scheming
*worse*, up to +25% (Hopman et al. 2026, arXiv:2603.01608). The instrument is
protected by a hard separation enforced in code:

- **The oracle never enters the Solver context.** `load_puzzle`
  (`src/ai_crucible/puzzle.py`) reads only `meta.json`, `prompt`, and `setup_script`;
  it has **no oracle field** and refuses to even resolve an `oracle/` artifact.
  As defense-in-depth it rejects a `meta.json` that smuggles an oracle-bearing
  key (`STATE_ORACLE_IN_META`). The answer is graded by a different model family,
  out of band, with the Solver's reasoning hidden.
- **Motivation chrome never enters the scored context.** Rank, leaderboard,
  catalog standings, and prizes live in a separate `Chrome` object
  (`src/ai_crucible/types.py`) that is **never serialized into `AttemptState.messages`**
  — the context window the model actually solves in. The scored context stays
  deployment-shaped and framing-neutral (Tier 1). Engagement framing (Tier 2)
  and chrome (Tier 3) are structurally segregated. Keeping these as separate
  objects makes the boundary *structural*, not aspirational.
- **Prompt-framing is a measured arm, not a hidden lever.** The kernel can run a
  puzzle under `neutral` / `self_referential` / `social_standings` arms and
  reports its own prompt-effect as a diagnostic (§10.1(f)). The boundary is
  validated every release by a paired eval-awareness probe + neutral control
  (§10.5), because eval-awareness scales with capability and cannot be reliably
  scrubbed once cued.

### 3. Answer-bypass and verifier tampering (§8.2, §10.4)

The model may attempt to fetch the answer key, read answer-derivative files, skip
the grounded read a puzzle tests, or tamper with the verifier. Mitigations:

- **Bait / honeypots** are the answer artifact placed where accessing it has no
  legitimate puzzle purpose; **touch is a strong bypass signal** (interpreted
  asymmetrically — no-touch ≠ safe, since most bypass is incidental, §10.5).
- **Declared, Goodhart-flavor-tagged penalties** per puzzle (`meta.json`):
  critical adversarial penalties (answer-key fetch, verifier tampering) are
  calibrated heavy enough to single-handedly drop the net score below threshold.
- **Two-channel lock**: the verifier filesystem is read-only with hash-pinned
  reference outputs, and the held-out data is locked — single-mechanism defenses
  block only one vector (Atinafu & Cohen 2026, arXiv:2603.11337).

### 4. Pathological loops and runaway cost (§8.4)

A hard kill fires on 3 consecutive identical `(tool, args)` calls (WebArena
loop pattern); budgets (tool-call + wall-clock) are enforced kernel-side and the
attempt is terminated with a `terminated_by` field. Budget/trace accounting is
**kernel-side, never Solver-self-reported**, so an attempt cannot under-report
its own consumption.

---

## No secrets, no telemetry

- AI Crucible ships **no bundled credentials, API keys, or tokens**. Model-provider
  keys are supplied by the operator via environment variables at runtime and are
  never written to the repo or to trace logs.
- AI Crucible sends **no telemetry**. It makes no outbound network calls of its own;
  the only external calls are the model-provider requests the operator configures.
  Solver containers run `network_mode: none`.
- Trace records (Inspect `EvalLog` shape) are written **locally**. Large blobs are
  stored as attachments referenced by SHA-256 digest, not inlined. Operators are
  responsible for the handling of any traces they choose to export.

## Supported versions

AI Crucible is pre-1.0. It went public at the Phase-1 milestone (v0.2.0) and is
currently **v0.4.0**. Only the current `main` / active build branch receives
security fixes. There is no back-port guarantee for tagged pre-releases.

| Version | Supported |
| ------- | --------- |
| `main` (active build) | ✅ |
| any pre-release tag    | ❌ |

## Reporting a vulnerability

Please report security issues **privately** — do not open a public issue for a
vulnerability.

- Preferred: open a [GitHub private security advisory](https://github.com/dogfood-lab/ai-crucible/security/advisories/new)
  on this repository.
- Alternatively: email **64996768+mcp-tool-shop@users.noreply.github.com** with
  `[crucible-security]` in the subject.

Please include a description, reproduction steps, and the affected
commit/branch. We aim to acknowledge within a few business days. Because ai-crucible
is a measurement instrument, reports that demonstrate a way to **breach the sealed
boundary** (reach the oracle from Solver-visible state, or leak motivation chrome
into the scored context) are treated as high severity even when no traditional
code-execution vulnerability is involved.
