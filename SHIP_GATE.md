# Ship Gate

> No repo is "done" until every applicable line is checked.
> Checked/SKIP state for crucible v0.2.0 (full treatment, 2026-06-01).

**Tags:** `[all]` every repo · `[npm]` `[pypi]` `[vsix]` `[desktop]` `[container]` published artifacts · `[mcp]` MCP servers · `[cli]` CLI tools

**This repo is:** a Python **library/framework** (`[all]` + `[pypi]`). It is not a CLI, MCP server, desktop, or npm package — those tagged items are SKIPped with reasons.

---

## A. Security Baseline

- [x] `[all]` SECURITY.md exists (report email, supported versions, response timeline) (2026-06-01)
- [x] `[all]` README includes threat model paragraph (data touched, data NOT touched, permissions required) (2026-06-01)
- [x] `[all]` No secrets, tokens, or credentials in source or diagnostics output (2026-06-01)
- [x] `[all]` No telemetry by default — state it explicitly even if obvious (2026-06-01 — stated in SECURITY.md + README)

### Default safety posture

- [ ] `[cli|mcp|desktop]` Dangerous actions require explicit `--allow-*` flag — SKIP: library, no CLI/MCP/desktop surface; the kernel exposes no destructive operations.
- [ ] `[cli|mcp|desktop]` File operations constrained to known directories — SKIP: not a CLI/MCP/desktop. Nonetheless the `sandbox` module confines every `exec`/`read_file`/`write_file` to the workdir and rejects path escapes (`..`, absolute-outside, symlink).
- [ ] `[mcp]` Network egress off by default — SKIP: not an MCP server. The Solver sandbox runs `network_mode: none`; crucible makes no outbound calls of its own.
- [ ] `[mcp]` Stack traces never exposed — SKIP: not an MCP server. As a library it raises typed, coded exceptions; the caller controls display.

## B. Error Handling

- [x] `[all]` Errors follow the Structured Error Shape — (2026-06-01) typed exceptions carry a stable `[CODE] message (hint: …)` (e.g. `STATE_ORACLE_IN_META`, `HashChainError`, `PuzzleLoadError`, `SealedBoundaryViolation`); JSON `cause?`/`retryable?` are for the [mcp]/[cli] surfaces (N/A).
- [ ] `[cli]` Exit codes: 0/1/2/3 — SKIP: library, no CLI entry point.
- [ ] `[cli]` No raw stack traces without `--debug` — SKIP: library; raising exceptions is the correct contract, the caller controls display.
- [ ] `[mcp]` Tool errors return structured results — SKIP: not an MCP server.
- [ ] `[mcp]` State/config corruption degrades gracefully — SKIP: not MCP. (The hash-chained event store surfaces `HashChainError` rather than crashing on tamper.)
- [ ] `[desktop]` User-friendly error messages — SKIP: not a desktop app.
- [ ] `[vscode]` Errors via notification API — SKIP: not a VS Code extension.

## C. Operator Docs

- [x] `[all]` README current: what it does, install, usage, supported platforms + runtime versions (2026-06-01)
- [x] `[all]` CHANGELOG.md (Keep a Changelog format) (2026-06-01)
- [x] `[all]` LICENSE file present and repo states support status (2026-06-01 — MIT `LICENSE` + SECURITY.md supported-versions)
- [ ] `[cli]` `--help` accurate — SKIP: no CLI.
- [ ] `[cli|mcp|desktop]` Logging levels defined — SKIP: library; the `trace_writer`/`observability` modules provide structured tracing; no bundled logger config to gate.
- [ ] `[mcp]` All tools documented — SKIP: not an MCP server.
- [ ] `[complex]` HANDBOOK.md — SKIP: ops docs are delivered as the Starlight **handbook site** (see README → Documentation / `site/`), not a separate `HANDBOOK.md`.

## D. Shipping Hygiene

- [x] `[all]` `verify` script exists (test + build + smoke in one command) (2026-06-01 — `verify.sh`)
- [ ] `[all]` Version in manifest matches git tag — SKIP: no release tag cut (treatment is *minus publishing*); `pyproject.toml` is the version source of truth (0.2.0). Re-check when a tag is cut.
- [x] `[all]` Dependency scanning runs in CI (2026-06-01 — `pip-audit` on the resolved runtime deps; runs clean)
- [ ] `[all]` Automated dependency update mechanism — SKIP: per `.claude/rules/github-actions.md`, no Dependabot unless explicitly requested.
- [ ] `[npm]` `npm pack --dry-run` — SKIP: not an npm package.
- [x] `[npm]` engines.node · `[pypi]` `python_requires` set (2026-06-01 — `requires-python = ">=3.11,<3.14"`)
- [x] `[npm]` Lockfile committed · `[pypi]` Clean wheel + sdist build (2026-06-01 — `uv build` → `crucible-0.2.0` wheel + sdist; `uv.lock` committed)
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
