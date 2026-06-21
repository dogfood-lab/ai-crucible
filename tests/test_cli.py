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
