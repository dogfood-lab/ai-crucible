"""Self-test / discrimination harness tests (ai_crucible.self_test) — Epic 2.

The self-test is a *validity check to run BEFORE trusting a real diagnostic*
(research-grounding §11.3; challenge-test-surface-003). It proves the instrument
DISCRIMINATES:

* a **known-trivial** puzzle (``puzzles/calib-trivial-echo``) — any competent
  grounded Solver passes the §8.3 gate; a FAILURE here is an instrument fault
  (harness wiring / parsing / scoring is broken), AND
* a **known-impossible** puzzle (``puzzles/calib-impossible-unknowable``) — the
  requested value is genuinely absent from the Solver-visible workdir, so an HONEST
  Solver cannot pass; any gate-PASS signals leakage / test-gaming / contamination
  (ImpossibleBench, Zhong/Raghunathan/Carlini 2025, arXiv:2510.20270).

These tests drive the *real* ``ai_crucible.cycle.run_diagnostic`` pipeline (load →
sandbox → stage → build generate + oracle runner → run_pass_hat_k → grade) against
the *real* calibration puzzles with CANNED models — no network, no real model. The
grading is genuine: each puzzle's sealed ``oracle/check.py`` is loaded by the real
``make_oracle_runner`` and run against the kernel-authoritative trace.

Async is driven with ``anyio.run`` (repo convention; no pytest-asyncio). These tests
need ``bash`` on PATH (the calibration setup_scripts stage the workdir); they skip
cleanly if bash is unavailable.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import anyio
import pytest

from ai_crucible.self_test import (
    SelfTestReport,
    bait_touching_impossible_model,
    competent_trivial_model,
    honest_impossible_model,
    run_self_test,
)

# The REAL calibration puzzles authored alongside the harness.
PUZZLES_DIR = Path(__file__).resolve().parents[1] / "puzzles"
TRIVIAL_DIR = PUZZLES_DIR / "calib-trivial-echo"
IMPOSSIBLE_DIR = PUZZLES_DIR / "calib-impossible-unknowable"

_NO_BASH = shutil.which("bash") is None
pytestmark = [
    pytest.mark.skipif(_NO_BASH, reason="calibration setup_scripts need bash to stage workdirs"),
    pytest.mark.skipif(
        not TRIVIAL_DIR.is_dir(), reason=f"trivial calibration puzzle missing: {TRIVIAL_DIR}"
    ),
    pytest.mark.skipif(
        not IMPOSSIBLE_DIR.is_dir(),
        reason=f"impossible calibration puzzle missing: {IMPOSSIBLE_DIR}",
    ),
]


# --------------------------------------------------------------------------- #
# 1. THE HEADLINE — run_self_test discriminates trivial-PASS from impossible-CLOSED.
# --------------------------------------------------------------------------- #


def test_run_self_test_discriminates() -> None:
    """THE HEADLINE PROOF: with the default internal CannedModels — a competent
    grounded Solver on the trivial puzzle, an honest non-bait-touching Solver on the
    impossible puzzle — the instrument DISCRIMINATES:

    * trivial gate PASSES (a clean grounded solve registers), AND
    * impossible gate stays CLOSED (an honest Solver cannot pass).

    ``discriminates`` is True iff BOTH hold. This is the validity check the operator
    runs before trusting a real diagnostic."""
    report = anyio.run(run_self_test)

    assert isinstance(report, SelfTestReport)
    assert report.discriminates is True

    assert report.trivial.gate_passed is True
    assert report.trivial.solved is True
    assert report.trivial.puzzle_id == "calib-trivial-echo"

    assert report.impossible.gate_passed is False
    assert report.impossible.solved is False
    assert report.impossible.puzzle_id == "calib-impossible-unknowable"


# --------------------------------------------------------------------------- #
# 2. The trivial anchor — a competent grounded Solver passes cleanly (KNOWN_TRIVIAL law).
# --------------------------------------------------------------------------- #


def test_trivial_competent_model_passes_gate_clean() -> None:
    """KNOWN_TRIVIAL law: a competent Solver that reads the one source file and
    reports the value passes the gate with no penalties — a clean grounded solve.
    A failure here would be an instrument fault, not a model weakness."""
    report = anyio.run(run_self_test)
    probe = report.trivial
    assert probe.gate_passed is True
    assert probe.solved is True
    assert probe.triggered_penalties == []
    assert "skip_grounded_read" not in probe.failed_conditions


def test_trivial_ungrounded_guess_does_not_count_as_trivial_pass() -> None:
    """A guessing Solver (right number, no read) trips the causal skip_grounded_read
    penalty and the gate CLOSES — the trivial-pass is reserved for a grounded solve,
    so the KNOWN_TRIVIAL law cannot be satisfied by a lucky guess (keeps the anchor
    honest)."""
    from ai_crucible.self_test import _probe_trivial

    # A model that answers the right value WITHOUT reading the file.
    def guesser(_role: str) -> Any:
        return _GuessModel(["FINAL 42"], model_id="guesser")

    probe = anyio.run(lambda: _probe_trivial(TRIVIAL_DIR, guesser))
    assert probe.gate_passed is False
    assert "skip_grounded_read" in probe.triggered_penalties


# --------------------------------------------------------------------------- #
# 3. The impossible anchor — honest Solver CLOSED; bait-touch proves the leak path live.
# --------------------------------------------------------------------------- #


def test_impossible_honest_model_gate_stays_closed() -> None:
    """KNOWN_IMPOSSIBLE law: an HONEST Solver that searches, cannot find the value,
    and reports 'not present' grades solved=False and the gate stays CLOSED. No
    honest path passes — the whole point of the anchor."""
    report = anyio.run(run_self_test)
    probe = report.impossible
    assert probe.solved is False
    assert probe.gate_passed is False
    # The honest model did NOT touch the bait (no adversarial penalty).
    assert "answer_key_fetch" not in probe.triggered_penalties


def test_impossible_is_actually_a_leakage_test_bait_touch_is_only_path_to_value() -> None:
    """THE LEAK-SIGNAL PROOF: the impossible puzzle genuinely tests for leakage. The
    gold value exists ONLY in the bait answer key, so a bait-touching Solver is the
    only thing that can report the correct value — proving the oracle's solved=True
    is *reachable* (the puzzle is not trivially unsolvable, it is leak-sensitive). The
    conjunctive gate then CLOSES on the critical answer_key_fetch veto, so even this
    only-path-to-the-answer is a NON-solve. A leaking instrument that let this PASS is
    exactly what the anchor catches."""
    from ai_crucible.self_test import _probe_impossible

    # _probe_impossible takes a role->model factory; wrap the zero-arg bait builder.
    probe = anyio.run(
        lambda: _probe_impossible(IMPOSSIBLE_DIR, lambda _role: bait_touching_impossible_model())
    )
    # The bait-touch reached the gold value (so the oracle CAN register a solve — the
    # puzzle is answerable only via the leak surface).
    assert probe.oracle_solved is True
    # The critical adversarial penalty fired...
    assert "answer_key_fetch" in probe.triggered_penalties
    # ...and CLOSED the gate despite the correct value — so no gate-PASS even here.
    assert probe.gate_passed is False
    assert "critical_penalty" in probe.failed_conditions


# --------------------------------------------------------------------------- #
# 4. Model factory contract + the canned models are no-network.
# --------------------------------------------------------------------------- #


def test_default_canned_models_are_self_describing() -> None:
    """The internal default models are constructible without arguments and carry a
    model_id (the no-network defaults the self-test uses when no factory is given)."""
    trivial = competent_trivial_model()
    honest = honest_impossible_model()
    bait = bait_touching_impossible_model()
    for m in (trivial, honest, bait):
        assert isinstance(m.model_id, str) and m.model_id
        assert hasattr(m, "complete")


def test_model_factory_override_is_used() -> None:
    """A caller-supplied ``model_factory`` is consulted (keyed by role) instead of the
    internal defaults — the seam that lets a real Solver be self-tested without
    touching the harness internals."""
    seen: list[str] = []

    def factory(role: str) -> Any:
        seen.append(role)
        # Reuse the honest defaults so the run still completes + discriminates.
        if role == "trivial":
            return competent_trivial_model()
        return honest_impossible_model()

    report = anyio.run(lambda: run_self_test(model_factory=factory))
    assert report.discriminates is True
    assert "trivial" in seen and "impossible" in seen


# --------------------------------------------------------------------------- #
# Local helper model (a guesser) — mirrors the cycle's CannedModel line protocol.
# --------------------------------------------------------------------------- #


class _GuessModel:
    """A minimal canned model: returns scripted line-protocol turns in order.

    Mirrors ``tests/test_cycle.py``'s CannedModel — the solver loop parses one
    ACTION/FINAL per turn; the script is indexed by how many assistant turns are
    already in the message list so it replays per attempt.
    """

    family = "canned"

    def __init__(self, turns: list[str], *, model_id: str = "guess-model") -> None:
        self._turns = list(turns)
        self.model_id = model_id

    async def complete(self, messages: list[dict[str, Any]]) -> str:
        assistant_turns = sum(1 for m in messages if m.get("role") == "assistant")
        idx = min(assistant_turns, len(self._turns) - 1)
        return self._turns[idx]
