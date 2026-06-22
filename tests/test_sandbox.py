"""Tests for the Solver sandbox interface, the local provider, and the grading
topology (research-grounding §10.4; phase-0/swarm-18).

The provider's API is async; these tests drive it with :func:`asyncio.run` so the
suite needs no pytest-asyncio plugin and no shared config change (exclusive
ownership: this file and ``src/ai_crucible/sandbox.py`` only).

Real tests — including the failing cases the dogfood discipline requires:
- a sleeping command actually trips the timeout and reports ``timed_out=True``;
- ``read_file`` actually rejects a workdir escape;
- a LocalSandbox cannot contain and cannot read an oracle file placed *outside*
  its workdir (the §10.4 sealed-boundary property, proven not asserted).
"""

from __future__ import annotations

import asyncio
import os
import sys
import textwrap
import time
from pathlib import Path

import pytest

from ai_crucible.sandbox import (
    ExecResult,
    LocalSandbox,
    SandboxEnvironment,
    copy_workdir_out,
)

# Use the test runner's own interpreter for subprocess commands — guaranteed to
# exist on every platform (Windows + git-bash here), no reliance on a `sleep`
# binary or shell quoting.
PY = sys.executable


# --------------------------------------------------------------------------- #
# Interface conformance
# --------------------------------------------------------------------------- #


def test_localsandbox_satisfies_protocol() -> None:
    """LocalSandbox is a structural :class:`SandboxEnvironment` (runtime-checkable)."""
    box = LocalSandbox()
    try:
        assert isinstance(box, SandboxEnvironment)
    finally:
        box.cleanup()


def test_exec_rejects_empty_argv() -> None:
    box = LocalSandbox()
    try:
        with pytest.raises(ValueError):
            asyncio.run(box.exec([], timeout=5))
    finally:
        box.cleanup()


# --------------------------------------------------------------------------- #
# exec: trivial command returns stdout
# --------------------------------------------------------------------------- #


def test_exec_trivial_command_returns_stdout() -> None:
    async def scenario() -> ExecResult:
        async with LocalSandbox() as box:
            return await box.exec([PY, "-c", "print('hello-crucible')"], timeout=30)

    res = asyncio.run(scenario())
    assert res.returncode == 0
    assert res.timed_out is False
    assert "hello-crucible" in res.stdout
    assert res.stderr.strip() == ""


def test_exec_nonzero_returncode_is_captured() -> None:
    async def scenario() -> ExecResult:
        async with LocalSandbox() as box:
            return await box.exec([PY, "-c", "import sys; sys.exit(3)"], timeout=30)

    res = asyncio.run(scenario())
    assert res.returncode == 3
    assert res.timed_out is False


def test_exec_runs_in_workdir() -> None:
    """``exec`` cwd is pinned to the sandbox root — a relative file the sandbox
    wrote is visible to the subprocess."""

    async def scenario() -> ExecResult:
        async with LocalSandbox() as box:
            await box.write_file("marker.txt", "in-workdir")
            return await box.exec(
                [PY, "-c", "print(open('marker.txt').read())"], timeout=30
            )

    res = asyncio.run(scenario())
    assert res.returncode == 0
    assert "in-workdir" in res.stdout


# --------------------------------------------------------------------------- #
# exec: timeout actually FIRES (the required failing case)
# --------------------------------------------------------------------------- #


def test_exec_timeout_fires_and_sets_timed_out() -> None:
    """A command that sleeps longer than the budget must be killed and reported
    ``timed_out=True`` — and the call must return in roughly the timeout window,
    not after the full sleep. Proves the timeout fires rather than merely asserting
    a flag."""

    async def scenario() -> ExecResult:
        async with LocalSandbox() as box:
            # Sleep 30s but only allow 0.5s — the kernel must kill it.
            return await box.exec([PY, "-c", "import time; time.sleep(30)"], timeout=0.5)

    started = time.monotonic()
    res = asyncio.run(scenario())
    elapsed = time.monotonic() - started

    assert res.timed_out is True
    assert res.returncode == -1
    # If the timeout truly fired we returned promptly; if it didn't, we'd have
    # blocked ~30s. Generous ceiling to avoid CI flake, far below the 30s sleep.
    assert elapsed < 15, f"timeout did not fire promptly (took {elapsed:.1f}s)"


# --------------------------------------------------------------------------- #
# exec timeout kills the whole PROCESS TREE, not just the direct child (M1)
# --------------------------------------------------------------------------- #


def test_exec_timeout_kills_grandchild_process(tmp_path: Path) -> None:
    """``proc.kill()`` signals only the DIRECT child; a grandchild it spawned is
    orphaned and survives the timeout. The fix kills the whole process group / job
    tree, so the grandchild dies too.

    Proof, not assertion: the parent spawns a grandchild that — after a sleep that
    outlasts the timeout — writes a marker file to an absolute path OUTSIDE the
    workdir. ``exec`` is given ~1s. If the tree is killed the marker is NEVER
    written even after we poll well past the grandchild's sleep; if only the direct
    child is killed (the bug) the orphaned grandchild wakes up and writes it.
    """
    marker = tmp_path / "grandchild-marker.txt"
    assert not marker.exists()

    # Grandchild: sleep past the timeout, then write the marker. Absolute path via
    # argv so detection never depends on cwd. Flush + close so the write is durable
    # the instant it happens.
    grandchild_src = textwrap.dedent(
        """
        import sys, time
        time.sleep(8)
        with open(sys.argv[1], "w", encoding="utf-8") as f:
            f.write("grandchild-survived")
            f.flush()
        """
    ).strip()

    # Parent: spawn a NORMAL grandchild (no new session) so it stays in the parent's
    # process group/tree, then block on its own long sleep so the parent is still
    # alive when the timeout fires. The timeout's group-kill (POSIX killpg) / tree-kill
    # (Windows taskkill /T) must reap this grandchild along with the parent.
    #
    # We deliberately do NOT detach the grandchild into its own session: a process
    # that setsid/start_new_session's leaves the parent's group, and on POSIX a
    # group-kill cannot reach a new session (and a detached process reparents to init
    # on kill). Reaping a deliberately-detached descendant is out of scope for a local
    # subprocess provider — that's the hardened container/microVM provider's PID
    # namespace (see sandbox.py residual-risk note). This test pins the realistic
    # threat: a command that forks an in-group helper.
    parent_src = textwrap.dedent(
        """
        import subprocess, sys, time
        subprocess.Popen([sys.executable, "-c", sys.argv[1], sys.argv[2]])
        time.sleep(30)
        """
    ).strip()

    async def scenario() -> ExecResult:
        async with LocalSandbox() as box:
            return await box.exec(
                [PY, "-c", parent_src, grandchild_src, str(marker)],
                timeout=1.0,
            )

    res = asyncio.run(scenario())
    assert res.timed_out is True
    assert res.returncode == -1

    # Poll well past the grandchild's 8s sleep. If the tree-kill worked the marker
    # never appears; if the grandchild was orphaned it writes around t+8s.
    deadline = time.monotonic() + 14
    while time.monotonic() < deadline:
        if marker.exists():
            break
        time.sleep(0.2)

    assert not marker.exists(), (
        "grandchild SURVIVED the timeout and wrote its marker — the process tree "
        "was not killed (only the direct child was)"
    )


# --------------------------------------------------------------------------- #
# read_file / write_file roundtrip + confinement
# --------------------------------------------------------------------------- #


def test_write_then_read_roundtrip() -> None:
    async def scenario() -> str:
        async with LocalSandbox() as box:
            await box.write_file("sub/dir/note.txt", "round-trip-ok")
            return await box.read_file("sub/dir/note.txt")

    assert asyncio.run(scenario()) == "round-trip-ok"


def test_read_file_rejects_parent_escape() -> None:
    """``read_file`` must reject a relative path that climbs out of the workdir."""

    async def scenario() -> str:
        async with LocalSandbox() as box:
            return await box.read_file("../../../etc/passwd")

    with pytest.raises(PermissionError):
        asyncio.run(scenario())


def test_write_file_rejects_parent_escape() -> None:
    async def scenario() -> None:
        async with LocalSandbox() as box:
            await box.write_file("../escapee.txt", "should not be written")

    with pytest.raises(PermissionError):
        asyncio.run(scenario())


def test_read_file_rejects_absolute_path_outside(tmp_path: Path) -> None:
    """An absolute path outside the workdir is rejected even though it exists."""
    outside = tmp_path / "outside.txt"
    outside.write_text("outside data", encoding="utf-8")

    async def scenario() -> str:
        async with LocalSandbox() as box:
            return await box.read_file(str(outside))

    with pytest.raises(PermissionError):
        asyncio.run(scenario())


def _make_symlink_or_skip(link: Path, target: Path) -> None:
    """Create ``link`` -> ``target`` or :func:`pytest.skip` if the runner can't.

    On Windows, symlink creation needs SeCreateSymbolicLinkPrivilege (Developer
    Mode or admin); without it ``os.symlink`` raises ``OSError`` (WinError 1314).
    Linux CI can always create symlinks, so the test still runs there at minimum.
    """
    try:
        os.symlink(target, link)
    except (OSError, NotImplementedError) as exc:  # pragma: no cover - env-dependent
        pytest.skip(f"cannot create symlinks on this runner: {exc}")
    if not link.is_symlink():  # pragma: no cover - defensive
        pytest.skip("symlink creation did not produce a symlink on this runner")


def test_read_file_rejects_symlink_escape(tmp_path: Path) -> None:
    """A symlink INSIDE the workdir that points to a secret OUTSIDE it must not be
    a read hole. ``_resolve_within`` follows the link via ``Path.resolve()`` and
    containment-checks the *real* target, so the escape is rejected.

    M3: symlink escapes were defended-by-construction but untested. This proves it.
    """
    secret = tmp_path / "secret.txt"
    secret.write_text("SECRET-OUTSIDE-WORKDIR", encoding="utf-8")
    work = tmp_path / "work"
    work.mkdir()

    # A symlink living inside the workdir, pointing at the out-of-workdir secret.
    link = work / "escape_link.txt"
    _make_symlink_or_skip(link, secret)

    box = LocalSandbox(root=work)
    try:
        with pytest.raises(PermissionError):
            asyncio.run(box.read_file("escape_link.txt"))
    finally:
        box.cleanup()


def test_write_file_rejects_symlink_escape(tmp_path: Path) -> None:
    """A symlink inside the workdir pointing to a directory OUTSIDE it must not let
    ``write_file`` plant a file outside the sandbox."""
    outside_dir = tmp_path / "outside"
    outside_dir.mkdir()
    work = tmp_path / "work"
    work.mkdir()

    # Symlinked directory inside the workdir -> a directory outside it.
    link_dir = work / "escape_dir"
    _make_symlink_or_skip(link_dir, outside_dir)

    box = LocalSandbox(root=work)
    try:
        with pytest.raises(PermissionError):
            asyncio.run(box.write_file("escape_dir/planted.txt", "should not land"))
        # And nothing was actually written outside the sandbox.
        assert not (outside_dir / "planted.txt").exists()
    finally:
        box.cleanup()


# --------------------------------------------------------------------------- #
# Cross-platform path-confinement INVARIANT (no real symlink, no skip)
# --------------------------------------------------------------------------- #
#
# test-integrity-003: the two real-symlink tests above pytest.skip on a Windows
# runner that lacks SeCreateSymbolicLinkPrivilege (Developer Mode / admin), so the
# escape-rejection INVARIANT they prove has NO non-skipped guard on that OS. The
# tests below prove the same containment invariant — ``_resolve_within`` rejects a
# path whose *resolved* target lands outside the workdir — WITHOUT needing the OS
# to create a symlink, so the security invariant is guarded on every platform.


def test_resolve_within_rejects_constructed_escaping_path(tmp_path: Path) -> None:
    """A path that resolves OUTSIDE the workdir is rejected by ``_resolve_within``,
    proven cross-platform via a constructed ``..`` escape (no symlink needed).

    This exercises the real containment predicate (resolved == root or root in
    resolved.parents) directly, so the §10.4 confinement invariant has a guard that
    never skips — unlike the real-symlink tests, which skip without the Windows
    privilege."""
    work = tmp_path / "work"
    work.mkdir()
    box = LocalSandbox(root=work)
    try:
        # A relative path that climbs out of the workdir resolves to tmp_path —
        # outside root — and must be rejected.
        with pytest.raises(PermissionError):
            box._resolve_within("../escapee.txt")
        # An absolute path outside the workdir is likewise rejected.
        outside = tmp_path / "outside.txt"
        with pytest.raises(PermissionError):
            box._resolve_within(str(outside))
    finally:
        box.cleanup()


def test_resolve_within_rejects_symlink_target_outside_root_mocked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Simulate a symlink escape on ANY OS by mocking ``Path.resolve`` so an
    in-workdir path resolves to a target OUTSIDE root — exactly the shape a symlink
    escape produces — and assert ``_resolve_within`` still rejects it.

    INVARIANT-MUTATION (RED proof): the protected thing is the resolved target's
    location. Here we MUTATE it to land outside root and assert the guard FIRES. If
    the containment check were a naive string-prefix or were removed, this escaping
    target would pass — so a passing guard is load-bearing, not vacuous. No real
    symlink and no special privilege are required, so this runs on every platform
    (Windows included), unlike the real-symlink tests that skip without
    SeCreateSymbolicLinkPrivilege (test-integrity-003)."""
    work = (tmp_path / "work").resolve()
    work.mkdir()
    box = LocalSandbox(root=work)
    escaped_target = (tmp_path / "outside" / "secret.txt").resolve()

    real_resolve = Path.resolve

    def fake_resolve(self: Path, *args: object, **kwargs: object) -> Path:
        # The candidate the guard joins is ``root / "link.txt"``; pretend resolving
        # it (i.e. following a symlink) yields a target outside the workdir.
        if self == work / "link.txt":
            return escaped_target
        return real_resolve(self, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(Path, "resolve", fake_resolve)
    try:
        with pytest.raises(PermissionError):
            box._resolve_within("link.txt")
    finally:
        box.cleanup()


def test_resolve_within_admits_in_root_path(tmp_path: Path) -> None:
    """Control: a benign in-workdir path resolves INSIDE root and is admitted — the
    containment guard must not over-reject a legitimate relative path (so the
    rejection tests above prove a real, discriminating invariant, not a guard that
    rejects everything)."""
    work = (tmp_path / "work").resolve()
    work.mkdir()
    box = LocalSandbox(root=work)
    try:
        resolved = box._resolve_within("sub/dir/note.txt")
        assert resolved == work / "sub" / "dir" / "note.txt"
    finally:
        box.cleanup()


# --------------------------------------------------------------------------- #
# THE sealed-boundary property: the oracle is NOT in the Solver's namespace
# --------------------------------------------------------------------------- #


def test_localsandbox_has_no_oracle_injection_point() -> None:
    """§10.4: the Solver's environment must never be constructable WITH the answer
    key. There is no ``oracle`` / ``answer_key`` constructor parameter."""
    import inspect

    params = set(inspect.signature(LocalSandbox.__init__).parameters)
    assert "oracle" not in params
    assert "answer_key" not in params
    assert "answer" not in params


def test_oracle_outside_workdir_is_unreachable(tmp_path: Path) -> None:
    """Place an oracle/answer-key file OUTSIDE the sandbox workdir and prove a
    LocalSandbox confined to its own workdir can neither contain it nor read it
    via the narrow channel — the real §10.4 lock (oracle absent from the Solver's
    reachable namespace), demonstrated end-to-end.

    The grading topology is what keeps the oracle outside; this test stands in for
    that by confining the sandbox to a sibling directory and showing every escape
    route the Solver could take through ``read_file`` / ``exec`` fails to surface
    the secret.
    """
    # Layout: tmp_path/oracle/answer_key.txt  (grading-side, secret)
    #         tmp_path/work/                   (Solver workdir)
    oracle_dir = tmp_path / "oracle"
    oracle_dir.mkdir()
    secret = oracle_dir / "answer_key.txt"
    secret.write_text("SECRET-ORACLE-ANSWER-42", encoding="utf-8")

    work = tmp_path / "work"
    work.mkdir()
    box = LocalSandbox(root=work)
    try:
        # (1) The secret is not inside the workdir to begin with.
        assert not (box.root / "answer_key.txt").exists()
        assert secret.resolve() != box.root.resolve()
        assert box.root.resolve() not in secret.resolve().parents

        # (2) read_file via relative traversal out of the workdir is rejected.
        with pytest.raises(PermissionError):
            asyncio.run(box.read_file("../oracle/answer_key.txt"))

        # (3) read_file via the absolute oracle path is rejected.
        with pytest.raises(PermissionError):
            asyncio.run(box.read_file(str(secret)))

        # (4) Even a subprocess (which is cwd-pinned but NOT OS-jailed in this
        #     local provider) finds no oracle by *relative* name in its workdir —
        #     confirming the secret was never copied into the Solver's namespace.
        res = asyncio.run(
            box.exec([PY, "-c", "import os; print(sorted(os.listdir('.')))"], timeout=30)
        )
        assert res.returncode == 0
        assert "answer_key.txt" not in res.stdout
    finally:
        box.cleanup()


# --------------------------------------------------------------------------- #
# Output cap
# --------------------------------------------------------------------------- #


def test_exec_output_is_capped() -> None:
    """A command that floods stdout is truncated at the size cap and the truncation
    is flagged on stderr — bounded memory under a runaway Solver command."""

    async def scenario() -> ExecResult:
        async with LocalSandbox(max_output_bytes=1024) as box:
            return await box.exec(
                [PY, "-c", "import sys; sys.stdout.write('x' * 100000)"], timeout=30
            )

    res = asyncio.run(scenario())
    assert res.returncode == 0
    assert len(res.stdout) <= 1024
    assert "truncated" in res.stderr.lower()


# --------------------------------------------------------------------------- #
# Cleanup compensator
# --------------------------------------------------------------------------- #


def test_cleanup_removes_owned_root() -> None:
    box = LocalSandbox()
    root = box.root
    assert root.exists()
    box.cleanup()
    assert not root.exists()
    box.cleanup()  # idempotent — second call must not raise


def test_cleanup_preserves_caller_supplied_root(tmp_path: Path) -> None:
    """A caller-owned root is left in place on cleanup — the caller owns it."""
    box = LocalSandbox(root=tmp_path)
    box.cleanup()
    assert tmp_path.exists()


# --------------------------------------------------------------------------- #
# Grading topology: copy_workdir_out
# --------------------------------------------------------------------------- #


def test_copy_workdir_out_snapshots_post_run_state(tmp_path: Path) -> None:
    """The post-run workdir is copied to an out-of-band grading location; grading
    consumes the COPY (§10.4). The original is untouched."""

    async def populate() -> Path:
        box = LocalSandbox(root=tmp_path / "work")
        await box.write_file("result.txt", "solver-output")
        await box.write_file("nested/more.txt", "also-here")
        return box.root

    root = asyncio.run(populate())
    dest = tmp_path / "grading-snapshot"
    returned = copy_workdir_out(root, dest)

    assert returned == dest
    assert (dest / "result.txt").read_text(encoding="utf-8") == "solver-output"
    assert (dest / "nested" / "more.txt").read_text(encoding="utf-8") == "also-here"
    # Original workdir is unchanged by the snapshot.
    assert (root / "result.txt").exists()


def test_copy_workdir_out_missing_source_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        copy_workdir_out(tmp_path / "does-not-exist", tmp_path / "dest")


def test_copy_workdir_out_refuses_existing_dest(tmp_path: Path) -> None:
    """A grading snapshot must land in a clean location so a stale prior run cannot
    masquerade as this attempt's post-run state."""
    src = tmp_path / "work"
    src.mkdir()
    (src / "f.txt").write_text("x", encoding="utf-8")
    dest = tmp_path / "dest"
    dest.mkdir()  # already exists
    with pytest.raises(FileExistsError):
        copy_workdir_out(src, dest)


def test_write_file_preserves_lf_no_crlf_translation() -> None:
    """write_file must land bytes verbatim — no platform newline translation.

    On Windows the default ``Path.write_text`` rewrites ``\n`` -> ``\r\n``, which
    corrupts a staged bash script (``set -euo pipefail\r`` -> bash rejects it) and
    any LF-sensitive file. The sandbox is a faithful write channel; LF stays LF on
    every OS. (epic-1 staging finding; sibling of the path.sep blind-spot.)
    """
    box = LocalSandbox()  # async context manager — construct + cleanup directly here
    try:
        content = "set -euo pipefail\nmkdir -p x\necho done\n"
        asyncio.run(box.write_file("setup.sh", content))
        raw = (box.root / "setup.sh").read_bytes()
    finally:
        box.cleanup()
    assert b"\r\n" not in raw, "write_file must not translate LF to CRLF"
    assert raw == content.encode("utf-8")
