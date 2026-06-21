"""ai-crucible unified CLI dispatcher tests (the console-script / npm-launcher entry point).

The dispatcher only routes ``argv[0]`` and forwards the remainder to the subcommand's own
parser, so these assert the routing — not the (GPU-bound) characterization itself, which is
exercised by a monkeypatched stand-in.
"""

from __future__ import annotations

from ai_crucible import cli


def test_version_prints_and_exits_zero(capsys) -> None:
    assert cli.main(["--version"]) == 0
    assert capsys.readouterr().out.startswith("ai-crucible ")


def test_help_and_no_args_show_usage(capsys) -> None:
    assert cli.main(["--help"]) == 0
    helptext = capsys.readouterr().out
    assert "characterize" in helptext
    assert "research preview" in helptext
    assert cli.main([]) == 0
    assert "usage: ai-crucible" in capsys.readouterr().out


def test_usage_banner_version_is_single_sourced(capsys, monkeypatch) -> None:
    """models-cli-004: the usage banner must read the version from package metadata (the
    same single source as ``--version``), NOT a hardcoded literal that drifts on the next
    bump. We monkeypatch the version resolver and assert the banner reflects it — proving no
    hardcoded version string survives in the banner path."""
    monkeypatch.setattr(cli, "_version", lambda: "9.9.9-test")
    assert cli.main(["--help"]) == 0
    helptext = capsys.readouterr().out
    assert "9.9.9-test" in helptext
    # the stale hardcoded literal must be gone from the banner.
    assert "v0.2.0" not in helptext


def test_unknown_command_exits_2_with_message(capsys) -> None:
    assert cli.main(["frobnicate"]) == 2
    assert "unknown command" in capsys.readouterr().err


def test_characterize_dispatches_and_forwards_args(monkeypatch) -> None:
    """`ai-crucible characterize <flags>` forwards <flags> verbatim to run.main and
    returns its exit code."""
    seen: dict[str, object] = {}

    def fake_main(argv: list[str]) -> int:
        seen["argv"] = argv
        return 7

    monkeypatch.setattr("ai_crucible.characterize.run.main", fake_main)
    assert cli.main(["characterize", "--k", "3", "--out", "x.json"]) == 7
    assert seen["argv"] == ["--k", "3", "--out", "x.json"]


# --- cli-operator-001 / error-hint-sweep-001: structured error rendering, no raw traceback ---


def test_structured_error_renders_as_one_line_not_traceback(monkeypatch, capsys) -> None:
    """A subcommand raising the repo's structured ``[CODE] msg (hint: ...)`` error must reach
    the operator as that single stderr line + a non-zero exit — NOT a multi-frame traceback.
    This mirrors the loader/Ollama errors that the lazy-imported run.main raises before its own
    per-model loop (cli-operator-001, error-hint-sweep-001)."""
    structured = (
        "[INPUT_PATH_MISSING] calibration path does not exist: /nope "
        "(hint: pass a path to a .json file or a directory of .json files)"
    )

    def boom(argv: list[str]) -> int:
        raise ValueError(structured)

    monkeypatch.setattr("ai_crucible.characterize.run.main", boom)
    rc = cli.main(["characterize", "--items", "/nope"])
    captured = capsys.readouterr()
    assert rc != 0
    # the ONE structured line is on stderr, verbatim
    assert structured in captured.err
    # and it is NOT dressed as a crash — no Python traceback chrome
    assert "Traceback (most recent call last)" not in captured.err
    assert "Traceback (most recent call last)" not in captured.out
    # exactly the structured line (no extra wrapper code) for an already-structured error
    assert "[CLI_UNEXPECTED]" not in captured.err


def test_unexpected_error_gets_generic_wrapper_with_debug_hint(monkeypatch, capsys) -> None:
    """An exception whose message is NOT in the structured shape (e.g. a packaging/import
    fault like a missing scientific dep) is wrapped in a generic ``[CLI_UNEXPECTED] ...
    (hint: re-run with --debug ...)`` line on stderr + non-zero exit, never a raw traceback
    (cli-operator-001)."""

    def boom(argv: list[str]) -> int:
        raise ModuleNotFoundError("No module named 'scipy'")

    monkeypatch.setattr("ai_crucible.characterize.run.main", boom)
    rc = cli.main(["characterize"])
    captured = capsys.readouterr()
    assert rc != 0
    assert "[CLI_UNEXPECTED]" in captured.err
    assert "No module named 'scipy'" in captured.err
    assert "--debug" in captured.err
    assert "Traceback (most recent call last)" not in captured.err


def test_lazy_import_failure_is_wrapped_not_a_traceback(monkeypatch, capsys) -> None:
    """The lazy ``from ai_crucible.characterize.run import main`` is itself inside the guard:
    an ImportError there (a real packaging fault, the exact probe in the audit) is rendered
    as the generic wrapper, not a raw ModuleNotFoundError traceback (cli-operator-001)."""
    import builtins

    real_import = builtins.__import__

    def fake_import(name: str, *a, **k):
        if name == "ai_crucible.characterize.run" or name.startswith(
            "ai_crucible.characterize.run"
        ):
            raise ModuleNotFoundError("No module named 'scipy'")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    rc = cli.main(["characterize"])
    captured = capsys.readouterr()
    assert rc != 0
    assert "[CLI_UNEXPECTED]" in captured.err
    assert "Traceback (most recent call last)" not in captured.err


def test_debug_flag_reraises_full_traceback(monkeypatch) -> None:
    """``--debug``/``-v`` opts INTO the developer traceback: the exception propagates so the
    interpreter prints the full multi-frame trace (cli-operator-001)."""

    def boom(argv: list[str]) -> int:
        raise RuntimeError("kaboom")

    monkeypatch.setattr("ai_crucible.characterize.run.main", boom)
    try:
        cli.main(["--debug", "characterize"])
    except RuntimeError as exc:
        assert "kaboom" in str(exc)
    else:  # pragma: no cover - the flag must re-raise
        raise AssertionError("--debug must re-raise the underlying exception")


def test_debug_flag_is_stripped_before_forwarding(monkeypatch) -> None:
    """``--debug`` is a top-level dispatcher flag: it is consumed before the remaining argv is
    forwarded to the subcommand, so the subcommand's own parser never sees it
    (cli-operator-001)."""
    seen: dict[str, object] = {}

    def fake_main(argv: list[str]) -> int:
        seen["argv"] = argv
        return 0

    monkeypatch.setattr("ai_crucible.characterize.run.main", fake_main)
    assert cli.main(["--debug", "characterize", "--k", "3"]) == 0
    assert seen["argv"] == ["--k", "3"]
    assert "--debug" not in seen["argv"]


def test_keyboard_interrupt_exits_130(monkeypatch) -> None:
    """A Ctrl-C during a run is a clean operator abort, not a crash: KeyboardInterrupt passes
    through to exit code 130 with no traceback wrapper (cli-operator-001)."""

    def boom(argv: list[str]) -> int:
        raise KeyboardInterrupt

    monkeypatch.setattr("ai_crucible.characterize.run.main", boom)
    assert cli.main(["characterize"]) == 130


# --- cli-operator-002 / error-hint-sweep-004: exit-code contract is discoverable from --help ---


def test_help_documents_exit_codes(capsys) -> None:
    """The exit-code contract (0 success; 2 unknown command; characterize's 1 = no judgments)
    must be discoverable from --help, not only from source — so a CI/dogfood-swarm author can
    gate on it (cli-operator-002, error-hint-sweep-004)."""
    assert cli.main(["--help"]) == 0
    helptext = capsys.readouterr().out
    assert "exit codes:" in helptext
    assert "0" in helptext and "success" in helptext
    assert "2" in helptext and "unknown command" in helptext
    # characterize's exit 1 = no judgments collected, cross-referenced to the stderr code
    assert "CHARACTERIZE_NO_JUDGMENTS" in helptext


def test_help_renders_on_cp1252_console(monkeypatch) -> None:
    """`ai-crucible --help` must not crash on a stock Windows (cp1252) console.

    The banner carries non-ASCII (``ω``); writing it to a strict-cp1252 stream raises
    UnicodeEncodeError. _ensure_utf8_streams() (called at the top of main) reconfigures the
    stream to UTF-8 so the operator's first command never dies on an encoding artifact.
    RED before the fix (UnicodeEncodeError escapes main); GREEN after.
    """
    import io
    import sys

    buf = io.BytesIO()
    cp1252 = io.TextIOWrapper(buf, encoding="cp1252", errors="strict", newline="")
    monkeypatch.setattr(sys, "stdout", cp1252)

    rc = cli.main(["--help"])  # would raise UnicodeEncodeError without the reconfigure
    cp1252.flush()

    assert rc == 0
    rendered = buf.getvalue().decode("utf-8")
    assert "ai-crucible" in rendered
    assert "ω" in rendered  # the ω survived (stream is now UTF-8), proving the path works
