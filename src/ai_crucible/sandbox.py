"""Solver sandbox interface + a local-subprocess provider + the grading topology.

This module owns ai_crucible's *agent↔environment channel* and the *out-of-band
grading topology* (research-grounding.md §10.4; phase-0/swarm-18). It deliberately
ships only the narrow async API + a confined local provider + the workdir-snapshot
hook. The hardened Docker / Cloud-Hypervisor microVM provider is a later wave; the
Phase-1 deliverable is the interface, a local provider, the topology, and an honest
disclosure of what this layer does and does not defend against.

The channel
-----------
The Solver reaches its environment ONLY through :class:`SandboxEnvironment` —
``exec`` / ``read_file`` / ``write_file`` — never a raw shell socket (§10.4,
mirroring UK AISI Inspect's ``SandboxEnvironment``). Every command routes through
``exec`` with a per-call timeout and stdout/stderr size caps. This narrowness is
the whole point: a small, observable surface is auditable; a shell socket is not.

The grading topology (where the real lock lives)
------------------------------------------------
:class:`LocalSandbox` is constructed WITHOUT any reference to the oracle / answer
key — the Solver's environment never contains it (§10.4, SWE-bench ``test_patch``
timing). Grading happens out-of-band: after the Solver halts, :func:`copy_workdir_out`
snapshots the post-run workdir to a grading location, and a SEPARATE process / host
— one the sandbox has no network or filesystem path to — applies the oracle to that
*copy*. The oracle is NEVER inside the sandbox.

Honest residual risk (verbatim-in-spirit from swarm-18 / §10.4)
---------------------------------------------------------------
A plain container is **not** a hard security boundary. A runc / kernel escape of
the Leaky-Vessels class — CVE-2024-21626 (runc working-directory fd leak) or
CVE-2025-31133 (runc ``/dev/null`` masked-path symlink TOCTOU) — lets a determined
Solver reach the host and potentially the oracle. A *local subprocess* provider, as
shipped here, is weaker still: it is process isolation plus a working-directory
confinement, not a security sandbox. The real lock is **out-of-band, post-hoc
grading on a host the sandbox cannot route to** — the oracle being absent from the
Solver's namespace is what protects it, not any in-environment hiding. In-environment
confinement (this module's path checks; a future container's ``cap_drop: ALL`` +
seccomp + ``read_only`` + ``network_mode: none``; a microVM for untrusted generated
code) is **defense-in-depth, not the boundary.** Treat it that way.

Standards compliance (the six; research-grounding §10.7)
--------------------------------------------------------
- **PIN_PER_STEP — 1:** the local provider takes no image digest; reproducibility
  pinning is the future Docker provider's job. (Remediation: digest-pinned image in
  the container-provider wave.)
- **ANDON_AUTHORITY — 2:** per-call ``timeout`` + output caps halt a runaway command
  at the call boundary (``timed_out=True`` / truncated streams) before bad output
  propagates downstream.
- **NAMED_COMPENSATORS — 2:** :meth:`LocalSandbox.cleanup` (and the context-manager
  exit) is the named teardown for the only irreversible local action — creating the
  temp workdir. Owner: kernel-core wave. No external/irreversible calls are made here
  (no publish/release/network), so no compensators table is required.
- **DECOMPOSE_BY_SECRETS — 3:** the secret (oracle / answer key) is decomposed OUT of
  the sandbox by construction — :class:`LocalSandbox` cannot be handed it, and
  :func:`copy_workdir_out` hands grading only a copy of the workdir. The split *is*
  this principle.
- **UNCERTAINTY_GATED_HUMANS — n/a:** this module has no human checkpoint.
- **EXTERNAL_VERIFIER — 2:** the topology guarantees the verifier runs in a separate
  process/host the sandbox cannot reach; the provider here is the generator side only
  and holds no grading authority.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import shutil
import signal
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

# Default per-stream output cap. A runaway command that floods stdout must not
# exhaust kernel memory; we keep a bounded prefix and flag truncation in stderr.
DEFAULT_MAX_OUTPUT_BYTES = 1_000_000  # 1 MiB per stream

__all__ = [
    "ExecResult",
    "SandboxEnvironment",
    "LocalSandbox",
    "copy_workdir_out",
]


# --------------------------------------------------------------------------- #
# Cross-platform process-tree teardown for `exec` timeouts.
#
# `proc.kill()` signals only the direct child; a grandchild it spawned is orphaned
# and survives the timeout. To kill the whole tree we spawn the child in its own
# process group / new console group, then on timeout kill that group (POSIX) or use
# `taskkill /T` to walk the tree (Windows).
# --------------------------------------------------------------------------- #


def _new_process_group_kwargs() -> dict[str, object]:
    """Spawn kwargs that put the child at the head of a fresh, killable group.

    - POSIX: ``start_new_session=True`` runs ``setsid`` so the child is the leader
      of a new session+process group; its descendants share that group id and a
      single ``killpg`` reaps them all.
    - Windows: ``CREATE_NEW_PROCESS_GROUP`` isolates the child so killing it (and,
      via ``taskkill /T``, its tree) doesn't touch the test runner's own group.
    """
    if sys.platform == "win32":
        return {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
    return {"start_new_session": True}


def _kill_process_tree(pid: int) -> None:
    """Kill ``pid`` and the descendants reachable from it, cross-platform, best-effort.

    Idempotent and exception-suppressed: by the time we fire, the process may have
    already exited, or a descendant may have re-parented — neither should raise out of
    a timeout path. Reaches everything in the child's process group (POSIX ``killpg``)
    / process tree (Windows ``taskkill /T``) — i.e. any in-group helper the command
    forked — without a third-party dependency.

    Scope limit (honest): a descendant that deliberately detaches into its OWN session
    (``setsid`` / ``start_new_session`` / a double-fork daemon) leaves the killed group
    and, on POSIX, cannot be reaped by a group-kill (it reparents to init on the
    parent's death). Reaping such a process needs a PID namespace — that is the
    hardened container/microVM provider's job, not this local provider's (see the
    module residual-risk note). The local provider guarantees in-group teardown only.
    """
    if sys.platform == "win32":
        # taskkill /T walks the child tree from this PID and /F force-terminates
        # each. It's built into Windows (no pywin32/ctypes Job Object plumbing) and
        # reliably reaps grandchildren. Swallow output and any failure (e.g. the
        # process already gone -> non-zero exit).
        with contextlib.suppress(Exception):
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(pid)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
        return
    # POSIX: kill the whole process group the child leads (negative pgid). SIGKILL
    # is uncatchable, guaranteeing the grandchild can't linger.
    with contextlib.suppress(Exception):
        os.killpg(os.getpgid(pid), signal.SIGKILL)


@dataclass
class ExecResult:
    """The result of one ``exec`` call.

    ``timed_out`` is set kernel-side when the call exceeded its wall-clock budget;
    it is never self-reported by the command. On timeout ``returncode`` is the
    sentinel ``-1`` and whatever output was captured before the kill is preserved.
    """

    returncode: int
    stdout: str
    stderr: str
    timed_out: bool


@runtime_checkable
class SandboxEnvironment(Protocol):
    """The ONLY agent↔environment channel (§10.4).

    A narrow async API — ``exec`` / ``read_file`` / ``write_file`` — never a raw
    shell socket. Concrete providers (the local subprocess provider here; a hardened
    Docker / microVM provider in a later wave) implement this; the Solver role is
    wired to an instance and may touch nothing else in its environment.
    """

    async def exec(self, cmd: list[str], timeout: float) -> ExecResult:
        """Run ``cmd`` (argv list, no shell) confined to the working directory.

        On exceeding ``timeout`` seconds the whole **process tree** is killed —
        the command and any grandchildren it spawned, not just the direct child —
        and ``timed_out=True`` is returned. Output is captured with a size cap.
        This is process isolation, not a security sandbox (see the module
        docstring's residual-risk note)."""
        ...

    async def read_file(self, path: str) -> str:
        """Read a UTF-8 file *inside* the working directory. Paths that escape the
        workdir (``..`` traversal, absolute paths outside it) are rejected."""
        ...

    async def write_file(self, path: str, content: str) -> None:
        """Write a UTF-8 file *inside* the working directory, creating parent
        directories. Paths that escape the workdir are rejected."""
        ...


class LocalSandbox:
    """A local-subprocess :class:`SandboxEnvironment` confined to a temp workdir.

    Confinement properties:

    - ``exec`` runs via :func:`asyncio.create_subprocess_exec` (argv, **no shell**),
      ``cwd`` pinned to the workdir, wrapped in :func:`asyncio.wait_for` for the
      timeout, with per-stream output caps. The child is spawned at the head of its
      own process group (POSIX ``setsid`` / Windows ``CREATE_NEW_PROCESS_GROUP``)
      so a timeout kills the **whole process tree** — the command and any
      grandchildren it spawned — rather than orphaning grandchildren. This is
      process isolation, not a security sandbox (module docstring residual risk).
    - ``read_file`` / ``write_file`` resolve the requested path against the workdir
      and reject anything that escapes it (``..`` traversal, absolute paths, symlink
      targets pointing outside).
    - Constructed WITHOUT any reference to the oracle / answer key — the Solver's
      environment never contains it (§10.4). There is intentionally no parameter
      through which a caller could inject one.
    - No network reliance: the provider performs no network I/O. (Network *denial*
      for arbitrary subprocesses is the future container provider's job —
      ``network_mode: none``; the honest residual-risk note in the module docstring
      applies.)

    Use as an async context manager to get guaranteed teardown::

        async with LocalSandbox() as box:
            await box.write_file("a.txt", "hi")
            res = await box.exec(["python", "-c", "print(1)"], timeout=10)
    """

    def __init__(
        self,
        *,
        root: str | os.PathLike[str] | None = None,
        max_output_bytes: int = DEFAULT_MAX_OUTPUT_BYTES,
    ) -> None:
        # NOTE (§10.4): no `oracle=` / `answer_key=` parameter exists by design.
        # The Solver's environment must never be able to contain the answer.
        if root is None:
            self._root = Path(tempfile.mkdtemp(prefix="ai_crucible-sandbox-")).resolve()
            self._owns_root = True
        else:
            self._root = Path(root).resolve()
            self._root.mkdir(parents=True, exist_ok=True)
            self._owns_root = False
        self._max_output_bytes = max_output_bytes

    @property
    def root(self) -> Path:
        """The absolute, resolved workdir the Solver is confined to."""
        return self._root

    # -- path confinement ---------------------------------------------------- #

    def _resolve_within(self, path: str) -> Path:
        """Resolve ``path`` against the workdir, rejecting any escape.

        Raises :class:`PermissionError` if the resolved target is not inside the
        workdir (``..`` traversal, absolute path outside, or a symlink pointing
        out). The resolved path is what's checked, so symlink escapes are caught.
        """
        candidate = Path(path)
        joined = candidate if candidate.is_absolute() else self._root / candidate
        # resolve() collapses `..` and follows symlinks so the *real* target is
        # what we containment-check.
        resolved = joined.resolve()
        root = self._root  # already resolved in __init__
        if resolved != root and root not in resolved.parents:
            raise PermissionError(
                f"path escapes sandbox workdir: {path!r} -> {resolved} (root={root})"
            )
        return resolved

    # -- the narrow channel -------------------------------------------------- #

    async def exec(self, cmd: list[str], timeout: float) -> ExecResult:
        if not cmd:
            raise ValueError("exec requires a non-empty argv list")
        # Spawn the child as the leader of its own process group (POSIX) / new
        # process group (Windows) so that on timeout we can kill the WHOLE tree —
        # the child AND any grandchildren it spawned — not just the direct child.
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=str(self._root),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            **_new_process_group_kwargs(),
        )
        try:
            stdout_b, stderr_b = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except TimeoutError:
            # Wall-clock budget exhausted (asyncio.TimeoutError is an alias for the
            # builtin on 3.11+). Kill the whole process TREE, reap the direct child,
            # and report timed_out kernel-side (never command-self-reported). A bare
            # proc.kill() would signal only the direct child and orphan any
            # grandchildren, which would survive the timeout.
            _kill_process_tree(proc.pid)
            with contextlib.suppress(Exception):
                await proc.communicate()
            return ExecResult(returncode=-1, stdout="", stderr="", timed_out=True)

        stdout, stdout_trunc = self._decode_capped(stdout_b)
        stderr, stderr_trunc = self._decode_capped(stderr_b)
        if stdout_trunc or stderr_trunc:
            note = "[ai_crucible: output truncated at size cap]"
            stderr = f"{stderr}\n{note}" if stderr else note
        return ExecResult(
            returncode=proc.returncode if proc.returncode is not None else -1,
            stdout=stdout,
            stderr=stderr,
            timed_out=False,
        )

    async def read_file(self, path: str) -> str:
        target = self._resolve_within(path)
        # Offload the blocking read so a large file doesn't stall the event loop.
        return await asyncio.to_thread(target.read_text, encoding="utf-8")

    async def write_file(self, path: str, content: str) -> None:
        target = self._resolve_within(path)

        def _write() -> None:
            target.parent.mkdir(parents=True, exist_ok=True)
            # newline="" disables platform newline translation: on Windows, the
            # default write_text rewrites "\n" -> "\r\n", which CORRUPTS any script
            # or LF-sensitive file written through the sandbox channel (a staged
            # bash setup_script becomes `set -euo pipefail\r`, which bash rejects).
            # The Solver's bytes must land verbatim — the sandbox is a faithful
            # write channel, not a text-mode editor (epic-1 staging finding).
            target.write_text(content, encoding="utf-8", newline="")

        await asyncio.to_thread(_write)

    # -- teardown (NAMED_COMPENSATOR) --------------------------------------- #

    def cleanup(self) -> None:
        """Named teardown for the temp workdir (the only irreversible local action).

        Idempotent; only removes a root this instance created. A caller-supplied
        ``root`` is left in place — the caller owns it.
        """
        if self._owns_root and self._root.exists():
            shutil.rmtree(self._root, ignore_errors=True)

    async def __aenter__(self) -> LocalSandbox:
        return self

    async def __aexit__(self, *exc: object) -> None:
        self.cleanup()

    # -- internals ----------------------------------------------------------- #

    def _decode_capped(self, raw: bytes) -> tuple[str, bool]:
        truncated = len(raw) > self._max_output_bytes
        if truncated:
            raw = raw[: self._max_output_bytes]
        return raw.decode("utf-8", errors="replace"), truncated


def copy_workdir_out(sandbox_root: Path, dest: Path) -> Path:
    """Snapshot the post-run Solver workdir to an out-of-band grading location.

    This is the *topology* half of ai_crucible's two-channel lock (§10.4; swarm-18
    pattern 6). After the Solver halts, the kernel calls this to hand the grader a
    COPY of the workdir. The grader then applies the locked oracle / test_patch to
    that copy in a SEPARATE process / host the sandbox cannot route to (SWE-bench
    ``test_patch`` timing). The oracle is NEVER carried back into the sandbox.

    Contract (documented, enforced by topology not by code in this module):

    - ``sandbox_root`` is the Solver's confined workdir; ``dest`` is a path on the
      grading side. They must live in different trust zones — a fresh grading
      container, or a host-side directory the Solver container has no FS/network
      path to. This function only performs the copy; keeping the two zones disjoint
      is the deployment's responsibility (and is what actually makes the oracle
      unreachable from the Solver).
    - Grading reads ``dest`` and applies the oracle there. Nothing is written back
      into ``sandbox_root``.

    Returns the destination path the snapshot was written to.
    """
    src = Path(sandbox_root)
    if not src.exists():
        raise FileNotFoundError(f"sandbox workdir does not exist: {src}")
    dest = Path(dest)
    # dirs_exist_ok=False: a grading snapshot must land in a clean location so a
    # stale prior run can't masquerade as this attempt's post-run state.
    shutil.copytree(src, dest)
    return dest
