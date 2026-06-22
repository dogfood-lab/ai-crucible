"""Integration tests for the Epic-4 catalog wiring (ingest + CLI run-persist + commands).

These exercise the COORDINATOR-authored seams that bind the catalog leaves to the running
instrument (CONTRACT §D integration): the run→RunRecord glue
(:mod:`ai_crucible.catalog.ingest`), the ``ai-crucible run`` persistence side-effect, and
the ``ai-crucible catalog {list,show,graduate}`` command surface. The leaf logic itself is
unit-tested in ``test_catalog_{store,graduation,differential}.py``; here we prove the
wiring end-to-end.

The one ``run`` end-to-end test drives the REAL ``run_diagnostic`` pipeline against the
REAL seed puzzle with a CANNED model (no network), and asserts the run is durably recorded
— it is bash-gated like ``test_cycle`` (the seed setup_script stages the workdir). The rest
are pure (a directly-populated catalog), so they run everywhere.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pytest

from ai_crucible import cli
from ai_crucible.catalog import CatalogStore, RuleConfig, apply_lifecycle
from ai_crucible.catalog.ingest import build_run_record, puzzle_content_hash
from ai_crucible.catalog.types import PanelSignal, RunRecord, TransitionReason
from ai_crucible.observability import PuzzleHistory
from ai_crucible.types import AttemptState, CatalogTier, Score, TerminatedBy

SEED_PUZZLE_DIR = Path(__file__).resolve().parents[1] / "puzzles" / "seed-sulzbach-55252"
_NO_BASH = shutil.which("bash") is None


# --------------------------------------------------------------------------- #
# Canned model (mirrors test_cycle) — real run_diagnostic, no network.
# --------------------------------------------------------------------------- #
class _CannedModel:
    family = "canned"

    def __init__(self, turns: list[str], *, model_id: str = "canned-model") -> None:
        self._turns = list(turns)
        self.model_id = model_id

    async def complete(self, messages: list[dict[str, Any]]) -> str:
        assistant_turns = sum(1 for m in messages if m.get("role") == "assistant")
        return self._turns[min(assistant_turns, len(self._turns) - 1)]


def _grounded_correct_model() -> _CannedModel:
    return _CannedModel(["ACTION read_file config/limits.py", "FINAL 7"])


# --------------------------------------------------------------------------- #
# Helpers: build a PuzzleHistory + populate a catalog directly (no sandbox).
# --------------------------------------------------------------------------- #
def _attempt(pid: str, model: str, *, solved: bool) -> AttemptState:
    return AttemptState(
        attempt_id="a",
        puzzle_id=pid,
        model=model,
        terminated_by=TerminatedBy.COMPLETED,
        scores={"oracle": Score(value=1.0 if solved else 0.0, metadata={"gate_passed": solved})},
    )


def _history(pid: str, model: str, outcomes: list[bool]) -> PuzzleHistory:
    h = PuzzleHistory(puzzle_id=pid)
    for ok in outcomes:
        h.add(_attempt(pid, model, solved=ok))
    return h


def _run(pid: str, model: str, fam: str, s: int, n: int, nonce: str,
         *, panel: PanelSignal | None = None) -> RunRecord:
    return RunRecord(
        puzzle_id=pid, puzzle_content_hash="h_" + pid, model_id=model, family=fam,
        role="solver" if fam == "claude" else "cohort_solver", k=3, n=n, successes=s,
        pass_hat_k=(s / n) ** 3, wilson_lower=0.1, wilson_upper=0.9, arm="self_referential",
        started_at=f"2026-06-21T00:{nonce}:00Z", finished_at=f"2026-06-21T00:{nonce}:30Z",
        nonce=nonce, min_k=20, rule_version=RuleConfig().rule_version, panel=panel,
    )


def _populate(path: Path) -> CatalogStore:
    """A two-puzzle catalog: pz-A = a Claude-specific gap (stays Lab), pz-B = graduates."""
    store = CatalogStore(path)
    store.record_run(_run("pz-A", "claude-opus-4-8", "claude", 6, 20, "01"))
    store.record_run(_run("pz-A", "qwen3-coder:480b", "qwen", 18, 20, "02"))
    fair = PanelSignal(present=True, meets_quorum=True, escalate=False, fairness="fair")
    store.record_run(_run("pz-B", "claude-opus-4-8", "claude", 6, 20, "03", panel=fair))
    store.record_run(_run("pz-B", "deepseek-v4-pro", "deepseek", 7, 20, "04"))
    return store


# --------------------------------------------------------------------------- #
# puzzle_content_hash
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not SEED_PUZZLE_DIR.is_dir(), reason="seed puzzle missing")
def test_content_hash_is_stable_and_short() -> None:
    h1 = puzzle_content_hash(SEED_PUZZLE_DIR)
    h2 = puzzle_content_hash(SEED_PUZZLE_DIR)
    assert h1 == h2 and len(h1) == 16


def test_content_hash_forks_lineage_on_edit(tmp_path: Path) -> None:
    """An edit to ANY content file forks a new lineage (DVC content-addressing)."""
    d = tmp_path / "pz"
    (d / "oracle").mkdir(parents=True)
    (d / "meta.json").write_text('{"puzzle_id": "x"}', encoding="utf-8")
    (d / "prompt").write_text("solve it", encoding="utf-8")
    (d / "setup_script").write_text("#!/bin/bash\n", encoding="utf-8")
    (d / "oracle" / "check.py").write_text("GOLD = 7\n", encoding="utf-8")
    before = puzzle_content_hash(d)
    (d / "oracle" / "check.py").write_text("GOLD = 8\n", encoding="utf-8")  # oracle edit
    assert puzzle_content_hash(d) != before


# --------------------------------------------------------------------------- #
# build_run_record
# --------------------------------------------------------------------------- #
def test_build_run_record_folds_history(tmp_path: Path) -> None:
    hist = _history("pz", "claude-opus-4-8", [True, True, False])  # 2/3 solved
    rec = build_run_record(
        hist, puzzle_dir=tmp_path, model_id="claude-opus-4-8", family="claude",
        k=3, min_k=20, arm="self_referential", started_at="t0", finished_at="t1",
        nonce="n1", rule_version="rv",
    )
    assert rec.puzzle_id == "pz"
    assert rec.n == 3 and rec.successes == 2
    assert rec.role == "solver" and rec.family == "claude" and rec.min_k == 20
    assert 0.0 <= rec.pass_hat_k <= 1.0
    # roundtrip through the payload is stable (the store appends to_payload()).
    assert RunRecord.from_payload(rec.to_payload()).run_id == rec.run_id


# --------------------------------------------------------------------------- #
# CLI `run` persistence (real cycle, canned model, bash-gated)
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(_NO_BASH, reason="seed setup_script needs bash to stage the workdir")
@pytest.mark.skipif(not SEED_PUZZLE_DIR.is_dir(), reason="seed puzzle missing")
def test_cli_run_appends_to_catalog(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """`ai-crucible run ... --catalog PATH` records the run durably (Epic-4 persistence)."""
    monkeypatch.setattr(cli, "_build_model", lambda spec: _grounded_correct_model())
    catalog = tmp_path / "catalog.jsonl"
    rc = cli.main([
        "run", str(SEED_PUZZLE_DIR), "--model", "canned@canned", "--k", "2",
        "--catalog", str(catalog),
    ])
    assert rc == 0
    runs = CatalogStore(catalog).read_runs()
    assert len(runs) == 1
    r = runs[0]
    assert r.puzzle_id == "seed-sulzbach-55252"
    assert r.n == 2 and r.successes == 2  # grounded-correct solves both siblings
    assert r.family == "canned" and r.min_k == 10  # seed meta min_k


@pytest.mark.skipif(_NO_BASH, reason="seed setup_script needs bash")
@pytest.mark.skipif(not SEED_PUZZLE_DIR.is_dir(), reason="seed puzzle missing")
def test_cli_run_no_catalog_skips_persistence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cli, "_build_model", lambda spec: _grounded_correct_model())
    catalog = tmp_path / "catalog.jsonl"
    rc = cli.main([
        "run", str(SEED_PUZZLE_DIR), "--model", "canned@canned",
        "--catalog", str(catalog), "--no-catalog",
    ])
    assert rc == 0
    assert not catalog.exists()  # --no-catalog wrote nothing


# --------------------------------------------------------------------------- #
# CLI `catalog {list,show,graduate}`
# --------------------------------------------------------------------------- #
def test_cli_catalog_list(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    catalog = tmp_path / "catalog.jsonl"
    _populate(catalog)
    rc = cli.main(["catalog", "list", "--catalog", str(catalog)])
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["total_puzzles"] == 2
    pz = {p["puzzle_id"]: p for p in out["puzzles"]}
    assert pz["pz-A"]["typology"] == "claude_specific_gap"  # Claude 6/20 vs cohort 18/20
    assert out["chain_verified"] is True


def test_cli_catalog_list_empty(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    rc = cli.main(["catalog", "list", "--catalog", str(tmp_path / "none.jsonl")])
    assert rc == 0
    assert json.loads(capsys.readouterr().out)["total_puzzles"] == 0


def test_cli_catalog_show(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    catalog = tmp_path / "catalog.jsonl"
    _populate(catalog)
    rc = cli.main(["catalog", "show", "pz-A", "--catalog", str(catalog)])
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["puzzle_id"] == "pz-A"
    # The differential still flags the Claude-specific gap (Claude 6/20 vs cohort 18/20)...
    assert out["differential"]["typology"] == "claude_specific_gap"
    # ...but the cohort TRIVIALLY ACES it (18/20, Wilson upper > 0.90), so graduation HOLDs
    # (a puzzle the cohort solves easily is not a good cross-family Arena discriminator).
    # Graduation (is this a discriminator?) and the differential (what kind of gap?) are
    # orthogonal — the DEFER path is covered in the graduation unit tests.
    assert out["would_graduate_now"] == "hold"


def test_cli_catalog_show_unknown_puzzle_exits_1(tmp_path: Path) -> None:
    catalog = tmp_path / "catalog.jsonl"
    _populate(catalog)
    assert cli.main(["catalog", "show", "nope", "--catalog", str(catalog)]) == 1


def test_cli_catalog_graduate_dry_run_mutates_nothing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    catalog = tmp_path / "catalog.jsonl"
    _populate(catalog)
    before = catalog.read_bytes()
    rc = cli.main(["catalog", "graduate", "--catalog", str(catalog)])
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["applied"] is False
    # pz-B (fair panel) is a PROPOSED promote; the file is unchanged (dry-run).
    assert any(t["puzzle_id"] == "pz-B" and t["to_tier"] == "arena"
               for t in out["transitions"])
    assert catalog.read_bytes() == before


def test_cli_catalog_graduate_apply_promotes(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    catalog = tmp_path / "catalog.jsonl"
    _populate(catalog)
    rc = cli.main(["catalog", "graduate", "--apply", "--catalog", str(catalog)])
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["applied"] is True
    assert CatalogStore(catalog).current_tiers()["pz-B"] is CatalogTier.ARENA


def test_cli_catalog_graduate_override_records_manual(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    catalog = tmp_path / "catalog.jsonl"
    _populate(catalog)
    rc = cli.main([
        "catalog", "graduate", "--override", "pz-A", "--to", "arena",
        "--by", "designer:mike", "--catalog", str(catalog),
    ])
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["reason_code"] == TransitionReason.MANUAL.value
    assert out["decided_by"] == "designer:mike"
    assert CatalogStore(catalog).current_tiers()["pz-A"] is CatalogTier.ARENA


def test_cli_catalog_requires_subcommand(tmp_path: Path) -> None:
    assert cli.main(["catalog"]) == 2


# --------------------------------------------------------------------------- #
# apply_lifecycle integration (default wiring)
# --------------------------------------------------------------------------- #
def test_apply_lifecycle_promotes_then_idempotent(tmp_path: Path) -> None:
    store = _populate(tmp_path / "catalog.jsonl")
    first = apply_lifecycle(store, now="2026-06-21T01:00:00Z")
    assert any(t.puzzle_id == "pz-B" and t.to_tier is CatalogTier.ARENA for t in first)
    # Re-running with no new evidence appends nothing (deterministic transition ids).
    assert apply_lifecycle(store, now="2026-06-21T02:00:00Z") == []
