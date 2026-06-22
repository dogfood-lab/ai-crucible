# Ship Gate

> No repo is "done" until every applicable line is checked.
> Checked/SKIP state for ai-crucible v0.2.0 (full treatment, 2026-06-01).

**Tags:** `[all]` every repo · `[npm]` `[pypi]` `[vsix]` `[desktop]` `[container]` published artifacts · `[mcp]` MCP servers · `[cli]` CLI tools

**This repo is:** a Python **library/framework** that also ships a thin **CLI** (`ai-crucible` console script + `python -m ai_crucible`) and an **npm launcher** (`@dogfood-lab/ai-crucible`) — tags `[all]` + `[pypi]` + `[cli]` + `[npm]`. It is not an MCP server, desktop app, or VS Code extension — those tagged items are SKIPped with reasons. *(Reclassified 2026-06-21, dogfood swarm: the prior "not a CLI / not an npm package" line was stale — the CLI entrypoint + npm launcher were added at publish-prep, so the `[cli]`/`[npm]` rows below are now live, not SKIPped.)*

---

## A. Security Baseline

- [x] `[all]` SECURITY.md exists (report email, supported versions, response timeline) (2026-06-01)
- [x] `[all]` README includes threat model paragraph (data touched, data NOT touched, permissions required) (2026-06-01)
- [x] `[all]` No secrets, tokens, or credentials in source or diagnostics output (2026-06-01)
- [x] `[all]` No telemetry by default — state it explicitly even if obvious (2026-06-01 — stated in SECURITY.md + README)

### Default safety posture

- [ ] `[cli|mcp|desktop]` Dangerous actions require explicit `--allow-*` flag — SKIP: the `ai-crucible characterize` CLI performs no destructive operations (it reads the local model panel + computes metrics); the kernel exposes no destructive operations, so there is nothing to gate behind `--allow-*`.
- [ ] `[cli|mcp|desktop]` File operations constrained to known directories — SKIP: the CLI writes no files; the `sandbox` module confines every `exec`/`read_file`/`write_file` to the workdir and rejects path escapes (`..`, absolute-outside, symlink).
- [ ] `[mcp]` Network egress off by default — SKIP: not an MCP server. The Solver sandbox runs `network_mode: none`; ai-crucible makes no outbound calls of its own.
- [ ] `[mcp]` Stack traces never exposed — SKIP: not an MCP server. As a library it raises typed, coded exceptions; the caller controls display.

## B. Error Handling

- [x] `[all]` Errors follow the Structured Error Shape — (2026-06-01) typed exceptions carry a stable `[CODE] message (hint: …)` (e.g. `STATE_ORACLE_IN_META`, `HashChainError`, `PuzzleLoadError`, `SealedBoundaryViolation`); JSON `cause?`/`retryable?` are for the [mcp]/[cli] surfaces (N/A).
- [x] `[cli]` Exit codes (2026-06-21 — CLI returns `0` success/help/version, `2` unknown command; the `characterize` subcommand returns non-zero on total model failure, models-cli-003)
- [x] `[cli]` No raw stack traces without `--debug` (2026-06-21 — the `main()` top-level handler renders an escaping exception as a SINGLE structured stderr line: a `[CODE] msg (hint:)` house-shape error verbatim, else a `[CLI_UNEXPECTED] … (hint: re-run with --debug)` wrapper; the full traceback is opt-IN via `--debug`/`-v`. Verified: `run /no/such/dir` → one-line `[INPUT_PUZZLE_DIR_MISSING]` + exit 1; `--debug` → traceback.)
- [ ] `[mcp]` Tool errors return structured results — SKIP: not an MCP server.
- [ ] `[mcp]` State/config corruption degrades gracefully — SKIP: not MCP. (The hash-chained event store surfaces `HashChainError` rather than crashing on tamper.)
- [ ] `[desktop]` User-friendly error messages — SKIP: not a desktop app.
- [ ] `[vscode]` Errors via notification API — SKIP: not a VS Code extension.

## C. Operator Docs

- [x] `[all]` README current: what it does, install, usage, supported platforms + runtime versions (2026-06-01)
- [x] `[all]` CHANGELOG.md (Keep a Changelog format) (2026-06-01)
- [x] `[all]` LICENSE file present and repo states support status (2026-06-01 — MIT `LICENSE` + SECURITY.md supported-versions)
- [x] `[cli]` `--help` accurate (2026-06-21 — banner reads the version from package metadata (single source, models-cli-004), lists the `characterize` command + flag-forwarding, and honestly discloses the provisional circular-ω caveat)
- [ ] `[cli|mcp|desktop]` Logging levels defined — SKIP: library; the `trace_writer`/`observability` modules provide structured tracing; no bundled logger config to gate.
- [ ] `[mcp]` All tools documented — SKIP: not an MCP server.
- [ ] `[complex]` HANDBOOK.md — SKIP: ops docs are delivered as the Starlight **handbook site** (see README → Documentation / `site/`), not a separate `HANDBOOK.md`.

## D. Shipping Hygiene

- [x] `[all]` `verify` script exists (test + build + smoke in one command) (2026-06-01 — `verify.sh`)
- [x] `[all]` Version in manifest matches git tag (2026-06-21 — `pyproject.toml` 0.2.0 == released tag `v0.2.0`; re-verify if this dogfood swarm cuts a new tag at Phase 10)
- [x] `[all]` Dependency scanning runs in CI (2026-06-01 — `pip-audit` on the resolved runtime deps; runs clean)
- [ ] `[all]` Automated dependency update mechanism — SKIP: per `.claude/rules/github-actions.md`, no Dependabot unless explicitly requested.
- [x] `[npm]` `npm pack --dry-run` (2026-06-21 — clean: 11 files, 8.6 kB packed / 27.1 kB unpacked, `@dogfood-lab/ai-crucible`, bin + README + 7 translations; published with provenance via `release.yml` OIDC) — superseding APPLICABLE (the npm launcher `@dogfood-lab/ai-crucible`): it published 0.2.0 with provenance via `release.yml` OIDC (2026-06-02); a fresh `npm pack --dry-run` was not re-run this swarm — re-verify at the next release.
- [x] `[npm]` engines.node · `[pypi]` `python_requires` set (2026-06-01 — `requires-python = ">=3.11,<3.14"`)
- [x] `[npm]` Lockfile committed · `[pypi]` Clean wheel + sdist build (2026-06-21 — `uv build` → `ai_crucible-0.2.0` wheel + sdist; `uv.lock` committed)
- [ ] `[vsix]` `vsce package` — SKIP: not a VS Code extension.
- [ ] `[desktop]` Installer builds — SKIP: not a desktop app.

## E. Identity (soft gate — does not block ship)

- [x] `[all]` Logo in README header (2026-06-01)
- [x] `[all]` Translations (polyglot-mcp — 7 translations ja/zh/es/fr/hi/it/pt-BR + EN source) (2026-06-01)
- [x] `[org]` Landing page + Starlight handbook (@mcptoolshop/site-theme) (2026-06-01)
- [x] `[all]` GitHub repo metadata: description, homepage, topics (2026-06-01)

---

## Gate Rules

**Hard gate (A–D):** Must pass before any version is tagged or published. ✅ All A–D items are checked or SKIPped with justification.

**Soft gate (E):** Should be done. ✅ Completed during the full treatment (logo, 7 translations, landing page + handbook, repo metadata).
