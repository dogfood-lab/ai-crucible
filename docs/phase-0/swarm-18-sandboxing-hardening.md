# Swarm Agent 18 — Sandboxing & Environmental Hardening for Agentic Eval

**Date:** 2026-06-01
**Source:** Fourth study swarm dispatch (Phase-1-prep, research-grounded advisor protocol)
**Question:** How to implement crucible's "environmental hardening" (named the single highest-leverage anti-bypass intervention in §8.6) on a single host — sandbox the Solver's tool/code execution AND lock the verifier+oracle+held-out data so the agent cannot read or tamper with them while still running legitimate tools?

---

# Sandboxing & verifier-locking for Crucible (single-host kernel)

**Framing:** Crucible's two-channel lock (hide the verifier AND the held-out data) maps to two orthogonal mechanisms below: (1) **isolation tech** keeps the agent's code from escaping/reading host state; (2) **grading topology** keeps the oracle outside the agent's reachable namespace. SWE-bench's design is the proof-existence that you can run *legitimate* tools inside the box while the oracle lives outside it.

## Patterns & tools

**1. Inspect AI `SandboxEnvironment` (Docker provider, network-off default).** [AISI Inspect docs, 2026](https://inspect.aisi.org.uk/sandboxing.html) — the agent reaches the box only through `exec()`/`read_file()`/`write_file()`; the auto-generated compose file sets `network_mode: none` and per-service `mem_limit`/`cpus`. *Crucible decision:* adopt the same narrow async API as the kernel's only agent→env channel — never hand the Solver a raw shell socket; every tool call routes through `exec()` with a timeout.

**2. Inspect `files` prefixing + setup script ordering.** [Same docs](https://inspect.aisi.org.uk/sandboxing.html) — files are copied per-named-environment (`"victim:flag.txt"`) and `setup` runs *after* copy; there is **no file-hiding primitive**. *Crucible decision:* do not try to "hide" the oracle inside the Solver's container — Inspect itself doesn't; put it in a second environment the Solver has no handle to (pattern 6).

**3. SWE-bench layered images, oracle applied by harness post-prediction.** [SWE-bench harness ref](https://www.swebench.com/SWE-bench/reference/harness/) + [issue #328 / Verified dataset](https://huggingface.co/datasets/SWE-bench/SWE-bench_Verified) — base→environment→instance layers; the `test_patch` and `FAIL_TO_PASS`/`PASS_TO_PASS` sets are **applied and executed by the harness after** the model emits its patch, not present in the edit loop. *Crucible decision:* this IS the lock — bake locked test files into an image layer the Solver's working copy doesn't include, and apply/run them only in the grading pass.

**4. Digest-pinned per-task images.** [Epoch AI, 2025](https://epoch.ai/blog/swebench-docker) — pinned dependency/toolchain images give byte-for-byte replay (500 Verified images, 30 GiB, 62-min full run). *Crucible decision:* pin each puzzle's image by digest (satisfies workflow standard **PIN_PER_STEP**) so a graded run is reproducible and the setup_script can't drift.

**5. cap-drop + seccomp + read-only bind, but beware `CAP_SYS_ADMIN`.** [OWASP Docker Cheat Sheet](https://cheatsheetseries.owasp.org/cheatsheets/Docker_Security_Cheat_Sheet.html) + [systemd/Kinvolk note](https://kinvolk.io/blog/2018/04/towards-unprivileged-container-builds/) — `--cap-drop ALL`, default seccomp (blocks `mount()` even *with* the cap), read-only mounts; **a read-only bind is only safe if the process can't `umount`/move it, which needs `CAP_SYS_ADMIN`**. *Crucible decision:* `cap_drop: [ALL]`, `read_only: true`, default seccomp, `no-new-privileges`, non-root UID — never re-add `SYS_ADMIN`/`NET_RAW`.

**6. Two-environment / out-of-sandbox grading.** [Inspect victim-pattern](https://inspect.aisi.org.uk/sandboxing.html); [Anthropic, "Demystifying evals," 2025](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents) (grader reads the transcript, runs separately). *Crucible decision:* the kernel/oracle runs as a **separate host process the Solver container has zero network/FS path to**; grading consumes the transcript + a copy of the post-run workdir — the Solver never shares a namespace with the scorer (also satisfies **EXTERNAL_VERIFIER**).

**7. Cryptographic capability binding for tool use.** ["Governing Dynamic Capabilities," arXiv:2603.14332, 2026](https://arxiv.org/pdf/2603.14332) — limited proxy tokens issued to the agent while real keys/grading authority stay outside; reproducibility verification on replay. *Crucible decision:* if any legitimate tool needs privilege (e.g. git over a locked history), hand the Solver a scoped proxy token, never the credential or the oracle's signing key.

**8. microVM "Matryoshka" for true escape resistance.** [emirb, "Your Container Is Not a Sandbox," 2026](https://emirb.github.io/blog/microvm-2026/) — plain containers are *not* a security boundary (CVE-2024-21626 "Leaky Vessels", CVE-2025-31133 runc `/dev/null` symlink race); nest the container inside Firecracker/Cloud Hypervisor so escape needs a hypervisor 0-day. *Crucible decision:* wrap the puzzle container in a Cloud Hypervisor microVM (it supports **GPU passthrough + Windows**, unlike Firecracker) when the Solver runs untrusted *generated* code, not just curated tools.

**9. WSL2 caveat: shared kernel + `\\wsl$` reachback.** [Trend Micro, "Cracking the Isolation," 2025](https://www.trendmicro.com/vinfo/us/security/news/virtualization-and-cloud/cracking-the-isolation-novel-docker-desktop-vm-escape-techniques-under-wsl2) + [Docker WSL2 docs](https://docs.docker.com/desktop/features/wsl/) — all WSL2 distros share one kernel; the FS is exposed to Windows via 9p at `\\wsl$`, and Docker Desktop's WSL2 backend has documented VM-escape techniques. *Crucible decision:* enable Docker Desktop **Enhanced Container Isolation** (needs WSL ≥2.6 / kernel ≥6.6); keep the oracle on the **Windows host side**, never inside the same WSL distro the Solver container runs in.

## Recommended sandbox design

- **Isolation:** digest-pinned per-puzzle Docker image, `cap_drop: ALL` + default seccomp + `no-new-privileges` + non-root + `read_only` rootfs + `network_mode: none`; for untrusted *generated* code, nest in a Cloud Hypervisor microVM (GPU + Windows-capable). On WSL2, turn on Enhanced Container Isolation.
- **Agent channel:** Inspect-style `exec`/`read_file`/`write_file` only — no raw shell; per-call timeouts and output caps.
- **Grading location:** **outside the sandbox.** Kernel runs as a separate host process; after the Solver halts, copy the workdir out, apply the locked test_patch/oracle in a *fresh* grading container, score there.
- **Oracle hiding:** oracle, answer key, and locked tests live only in the grading-side image layer / host process — never copied into the Solver's working environment (mirrors SWE-bench `test_patch` timing). Privileged tools get scoped proxy tokens, not real creds.
- **Standards:** PIN_PER_STEP (digest pin), EXTERNAL_VERIFIER + DECOMPOSE_BY_SECRETS (separate grader process/family), NAMED_COMPENSATOR (teardown ephemeral container/microVM per run).
- **Biggest residual risk (honest):** a plain Docker container is *not* a hard boundary — a runc/kernel escape (Leaky Vessels-class) lets a determined Solver reach the host and potentially the oracle. The container/microVM hides the oracle from a *cooperative* tool-user, but only the microVM layer (or fully out-of-band, post-hoc grading on a host the container can't route to) defends against an adversarial escape. Treat post-hoc out-of-sandbox grading as the real lock; treat in-container hiding as defense-in-depth, not the boundary.

---

## ⚠ Verification corrections (second pass — 2026-06-01)

The sandbox design's load-bearing anchors verified verbatim: **RHB environmental-hardening 87.7% relative exploit reduction** (Thaman 2026, arXiv:2605.02964 ✓), **evaluator-tampering in ~50% of episodes** (Atinafu & Cohen 2026, arXiv:2603.11337 ✓), and both container-escape CVEs (**CVE-2024-21626 "Leaky Vessels"** runc fd-leak ✓; **CVE-2025-31133** runc `/dev/null` symlink/masked-path TOCTOU ✓). The "honest residual risk" claim stands. One re-anchor:

- **Pattern 7 (arXiv:2603.14332, Zhou 2026, "Governing Dynamic Capabilities"):** the paper RESOLVES but it is about **cryptographic capability-binding (Ed25519 / SNARK reproducibility verification)**, NOT the "limited proxy tokens, real keys stay outside" detail attributed to it. The architectural point — *grading authority and secrets stay outside the agent's namespace* — is sound, but it should be anchored to the **two-channel evaluator-locking** results (Atinafu & Cohen 2026; Thaman 2026 RHB) rather than to 2603.14332. Keep 2603.14332 only as a (separate) pointer for cryptographic binding of tool capabilities, if used at all.
