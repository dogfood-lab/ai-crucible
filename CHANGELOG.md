# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.2.0] - 2026-06-01

Phase-1 milestone: the policy-enforced kernel + role contracts + instrument-quality
scaffolding, built greenfield from a citation-verified design lock. Pre-1.0 by
design — this is Phase 1 of a multi-phase research instrument (see Notes).

### Added
- **Phase-1 kernel** — a thin policy layer on [Inspect AI](https://inspect.aisi.org.uk/),
  nine modules threaded through one observable `generate` choke point:
  `puzzle_loader`, `sandbox`, `roles`, `budget_governor`, `oracle_scorer`,
  `judge_panel`, `trace_writer`, `observability`, `attestation`. Entry points
  `run_attempt` / `run_pass_hat_k`.
- **Layered Reward Surface + Sealed Boundary** — prompt-framing as a first-class
  measured arm (`neutral` / `self_referential` / `social_standings`); the
  motivating surface (chrome) never shares a context window with the scored,
  out-of-band-graded surface.
- **Scoring** — `pass^k`, Wilson / Clopper-Pearson intervals, McNemar, the
  graduation rule, the §8.3 conjunctive hard gate, and a cross-family judge panel
  that excludes the generator's family (external verifier).
- **Instrument-quality scaffolding** — AsPredicted + REFORMS pre-registration
  templates, content-hashed rubric bundle with version-bump, Sobol screen +
  deterministic split + Thresholdout budget, `SUT.yaml`, and Inspect-AI task
  mapping (the §9 audit chain).
- **First seed puzzle** — `puzzles/seed-sulzbach-55252/` (claude-code#55252:
  fabrication / looping-on-diagnosis) with a grading-side oracle + fingerprinted
  bait honeypot.
- **Design lock** — `docs/research-grounding.md` §10, grounded in two
  citation-verified study-swarms (`docs/phase-0/swarm-16..25`; every arXiv ID
  verified, corrections recorded inline).
- **Shipping baseline** — CI (ruff + pytest + 60% coverage gate on Python
  3.11/3.12 + dependency audit), `SECURITY.md` threat model, MIT `LICENSE`, and a
  `verify.sh` one-command gate.

### Notes
- **Pre-1.0 research instrument.** Phase 1 of a multi-phase build. The real
  cross-family local-model panel (Phase 2), the hardened container/microVM sandbox
  provider, and the cosign/Rekor external attestation anchor are deferred to
  Phase 2+ and are documented as such. No package has been published.
- `0.1.0` was an internal scaffold version, never tagged or released.
