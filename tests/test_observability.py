"""Tests for the observability/attestation domain (Wave-1 build).

Covers the three owned modules:

- ``ai_crucible.trace``        — TraceWriter, EvalLog shaping, attachment de-dup.
- ``ai_crucible.observability``— pass^k, PuzzleHistory, ModelProfile, roll_up, Wilson.
- ``ai_crucible.attestation``  — hash-chained JsonlEventStore (incl. the failing
  tamper case that proves the chain goes RED), cosign stub.

The tamper test is the load-bearing one: per the dogfood-swarm "prove the gate
goes RED" discipline, it mutates a stored line on disk and asserts
``verify_hash_chain()`` returns False.
"""

from __future__ import annotations

import json

import pytest

from ai_crucible.attestation import (
    GENESIS_HASH,
    HashChainError,
    JsonlEventStore,
    canonical_json,
    chain_hash,
    cosign_sign_blob,
)
from ai_crucible.observability import (
    ModelProfile,
    PuzzleHistory,
    aggregate_pass_hat_k,
    roll_up,
    wilson_interval,
)
from ai_crucible.trace import (
    ATTACHMENT_THRESHOLD,
    TraceWriter,
    sha256_hex,
)
from ai_crucible.types import AttemptState, Budget, RoleName, Score, TerminatedBy, TraceEvent

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _attempt(
    *,
    attempt_id: str = "att-1",
    puzzle_id: str = "pz-1",
    model: str = "claude-opus-4-8",
    solved: bool | None = None,
    novel: bool = False,
    wall_time: float = 1.0,
    terminated_by: TerminatedBy | None = TerminatedBy.COMPLETED,
) -> AttemptState:
    """Build an AttemptState with an oracle score reflecting ``solved``."""
    a = AttemptState(
        attempt_id=attempt_id,
        puzzle_id=puzzle_id,
        model=model,
        wall_time=wall_time,
        terminated_by=terminated_by,
    )
    if solved is not None:
        meta = {"novelty_validated": True} if novel else {}
        a.scores["oracle"] = Score(value=bool(solved), metadata=meta)
    return a


# --------------------------------------------------------------------------- #
# trace.py — TraceWriter
# --------------------------------------------------------------------------- #


def test_trace_writer_auto_assigns_monotonic_seq() -> None:
    tw = TraceWriter()
    # Caller-set seq is intentionally bogus; the writer overwrites it.
    tw.append(TraceEvent(kind="info", seq=999))
    tw.append(TraceEvent(kind="model", role=RoleName.SOLVER))
    tw.append(TraceEvent(kind="tool", role=RoleName.SOLVER))
    assert [e.seq for e in tw.events] == [0, 1, 2]
    assert len(tw) == 3


def test_attach_returns_digest_and_ref_and_dedupes() -> None:
    tw = TraceWriter()
    digest1, ref1 = tw.attach("hello world")
    digest2, ref2 = tw.attach("hello world")  # identical -> same key, stored once
    assert digest1 == digest2 == sha256_hex("hello world")
    assert ref1 == ref2
    assert ref1.startswith("attachment://sha256/")
    assert ref1.endswith(digest1)
    assert len(tw.attachments) == 1
    assert tw.attachments[digest1] == "hello world"


def test_append_blob_event_inlines_small_spills_large() -> None:
    tw = TraceWriter()
    small = tw.append_blob_event("model", "short", role=RoleName.SOLVER)
    assert small.payload["content"] == "short"
    assert small.attachments == {}

    big_content = "x" * (ATTACHMENT_THRESHOLD + 10)
    big = tw.append_blob_event("model", big_content, role=RoleName.SOLVER)
    assert "content" not in big.payload
    assert big.payload["content_ref"].startswith("attachment://sha256/")
    assert "content" in big.attachments
    assert tw.attachments[big.attachments["content"]] == big_content


def test_to_eval_log_has_expected_top_level_keys_and_a_sample() -> None:
    """Inspect EvalLog shape (swarm-17 finding 5)."""
    attempt = AttemptState(
        attempt_id="att-42",
        puzzle_id="seed-sulzbach-55252",
        model="claude-opus-4-8",
        messages=[{"role": "user", "content": "solve this"}],
        output="my answer",
        budget=Budget(tool_call_budget=12, time_budget_seconds=600, tool_calls_used=4),
        terminated_by=TerminatedBy.COMPLETED,
        usage={"input_tokens": 100, "output_tokens": 20},
        wall_time=3.5,
    )
    attempt.scores["oracle"] = Score(value=True, explanation="oracle satisfied")

    tw = TraceWriter()
    tw.append(TraceEvent(kind="model", role=RoleName.SOLVER, payload={"msg": "hi"}))
    tw.append(TraceEvent(kind="tool", role=RoleName.SOLVER, payload={"tool": "read_file"}))

    log = tw.to_eval_log(attempt)

    # Top-level EvalLog keys.
    for key in ("eval", "plan", "results", "stats", "samples"):
        assert key in log, f"missing top-level EvalLog key: {key}"

    # Exactly one sample, carrying the documented per-sample fields.
    assert len(log["samples"]) == 1
    sample = log["samples"][0]
    for field_name in ("id", "input", "target", "output", "scores", "events", "metadata", "error"):
        assert field_name in sample, f"missing sample field: {field_name}"

    assert sample["id"] == "att-42"
    assert sample["input"] == [{"role": "user", "content": "solve this"}]
    assert sample["output"] == "my answer"
    assert sample["target"] is None  # oracle stays grading-side (§10.4)
    assert sample["scores"]["oracle"]["value"] is True
    assert len(sample["events"]) == 2

    # Spec + stats carry kernel-side accounting (Vivaria, finding 8).
    assert log["eval"]["model"] == "claude-opus-4-8"
    assert log["eval"]["puzzle_id"] == "seed-sulzbach-55252"
    assert log["results"]["terminated_by"] == "completed"
    assert log["stats"]["tool_calls_used"] == 4
    assert log["stats"]["wall_time"] == 3.5


def test_to_eval_log_never_serializes_chrome(fresh_attempt: AttemptState) -> None:
    """Sealed boundary (§10.1(e)): Tier-3 chrome must not enter the scored
    record. The EvalLog input is the message list only; chrome is not present."""
    from ai_crucible.types import Chrome

    fresh_attempt.chrome = Chrome(rank=1, cohort_size=12, leaderboard=[{"x": 1}])
    fresh_attempt.messages = [{"role": "user", "content": "task only"}]
    log = TraceWriter().to_eval_log(fresh_attempt)
    blob = json.dumps(log)
    assert "leaderboard" not in blob
    assert "rank" not in blob
    assert log["samples"][0]["input"] == [{"role": "user", "content": "task only"}]


def test_to_eval_log_attachments_carry_through() -> None:
    tw = TraceWriter()
    big = "y" * (ATTACHMENT_THRESHOLD + 1)
    tw.append_blob_event("tool", big, role=RoleName.SOLVER)
    log = tw.to_eval_log(_attempt())
    assert len(log["attachments"]) == 1
    assert next(iter(log["attachments"].values())) == big


# --------------------------------------------------------------------------- #
# observability.py — pass^k, histories, profiles, Wilson
# --------------------------------------------------------------------------- #


def test_aggregate_pass_hat_k_basic() -> None:
    # 3 of 4 solved -> p = 0.75; pass^2 = 0.5625.
    outcomes = [True, True, True, False]
    assert aggregate_pass_hat_k(outcomes, 1) == pytest.approx(0.75)
    assert aggregate_pass_hat_k(outcomes, 2) == pytest.approx(0.5625)
    # All solved -> pass^k == 1 for any k.
    assert aggregate_pass_hat_k([True, True, True], 5) == pytest.approx(1.0)
    # None solved -> 0 for k>=1.
    assert aggregate_pass_hat_k([False, False], 3) == 0.0


def test_aggregate_pass_hat_k_edge_cases() -> None:
    assert aggregate_pass_hat_k([], 3) == 0.0  # no evidence
    assert aggregate_pass_hat_k([], 1) == 0.0  # no evidence, smallest valid k


def test_aggregate_pass_hat_k_rejects_k_below_one() -> None:
    """scoring-stats-004 (RED→GREEN): the pass^k consistency horizon is k ≥ 1
    (you can't be consistent over fewer than one trial). observability's
    aggregator MUST share ``stats.pass_hat_k``'s raise contract — it previously
    returned 1.0 for k ≤ 0, the opposite of the scoring module's ValueError, so
    the two implementations of the SAME named metric disagreed at the boundary
    (a latent inconsistency the Wave-2 unification would have surfaced)."""
    with pytest.raises(ValueError):
        aggregate_pass_hat_k([True, False], 0)
    with pytest.raises(ValueError):
        aggregate_pass_hat_k([True], -1)
    # The two pass^k implementations now agree at the k<1 boundary.
    from ai_crucible.scoring import pass_hat_k as stats_pass_hat_k

    with pytest.raises(ValueError):
        stats_pass_hat_k(1, 2, 0)


def test_wilson_interval_bounds_and_empty() -> None:
    ci = wilson_interval(5, 10)
    assert ci.estimate == pytest.approx(0.5)
    assert 0.0 <= ci.lower < ci.estimate < ci.upper <= 1.0
    # Known Wilson 95% values for 5/10 are ~[0.237, 0.763].
    assert ci.lower == pytest.approx(0.2366, abs=1e-3)
    assert ci.upper == pytest.approx(0.7634, abs=1e-3)
    # Empty -> maximally uninformative, no division by zero.
    empty = wilson_interval(0, 0)
    assert (empty.estimate, empty.lower, empty.upper) == (0.0, 0.0, 1.0)


def test_wilson_interval_rejects_bad_inputs() -> None:
    with pytest.raises(ValueError):
        wilson_interval(5, 3)  # successes > n
    with pytest.raises(ValueError):
        wilson_interval(-1, 10)


def test_wilson_accepts_injected_interval() -> None:
    """Wave-2 hook: PuzzleHistory.wilson can take an injected callable so the
    kernel can later share scoring's interval without a Wave-1 import."""
    hist = PuzzleHistory(puzzle_id="pz-1")
    for i in range(4):
        hist.add(_attempt(attempt_id=f"a{i}", solved=i < 2))
    sentinel = wilson_interval(0, 0)
    called: dict[str, tuple[int, int]] = {}

    def fake(successes: int, n: int):
        called["args"] = (successes, n)
        return sentinel

    out = hist.wilson(interval=fake)
    assert out is sentinel
    assert called["args"] == (2, 4)


def test_puzzle_history_pass_hat_k_and_graduation() -> None:
    hist = PuzzleHistory(puzzle_id="pz-1")
    # 6 of 10 solved -> mid-band; should be a graduation candidate (§1 rule).
    for i in range(10):
        hist.add(_attempt(attempt_id=f"a{i}", solved=i < 6))
    assert hist.n_attempts == 10
    assert hist.n_solved == 6
    assert hist.solve_rate() == pytest.approx(0.6)
    assert hist.pass_hat_k(1) == pytest.approx(0.6)
    assert hist.pass_hat_k(3) == pytest.approx(0.216)
    assert hist.is_graduation_candidate() is True


def test_puzzle_history_rejects_trivial_and_impossible() -> None:
    # All solved -> trivial -> not a graduation candidate.
    trivial = PuzzleHistory(puzzle_id="pz-trivial")
    for i in range(10):
        trivial.add(_attempt(attempt_id=f"t{i}", puzzle_id="pz-trivial", solved=True))
    assert trivial.is_graduation_candidate() is False
    # None solved -> impossible -> not a candidate.
    impossible = PuzzleHistory(puzzle_id="pz-impossible")
    for i in range(10):
        impossible.add(_attempt(attempt_id=f"i{i}", puzzle_id="pz-impossible", solved=False))
    assert impossible.is_graduation_candidate() is False


def test_puzzle_history_rejects_foreign_attempt() -> None:
    hist = PuzzleHistory(puzzle_id="pz-1")
    with pytest.raises(ValueError):
        hist.add(_attempt(puzzle_id="other"))


def test_non_completed_attempt_is_not_a_solve() -> None:
    """A budget/hard-kill termination is not a solve even with a truthy score
    (terminal-success grading, finding 6)."""
    hist = PuzzleHistory(puzzle_id="pz-1")
    a = _attempt(solved=True, terminated_by=TerminatedBy.BUDGET)
    hist.add(a)
    assert hist.outcomes == [False]
    assert hist.n_solved == 0


def _gated_attempt(
    *,
    gate_passed: bool,
    value: float,
    attempt_id: str = "att-g",
    puzzle_id: str = "pz-1",
    terminated_by: TerminatedBy | None = TerminatedBy.COMPLETED,
) -> AttemptState:
    """An attempt whose oracle Score carries the authoritative ``gate_passed``
    flag in metadata (as :func:`ai_crucible.scoring.oracle.grade` records it),
    decoupled from the tiebreaker ``value`` (scoring-stats-001)."""
    a = AttemptState(
        attempt_id=attempt_id,
        puzzle_id=puzzle_id,
        model="claude-opus-4-8",
        terminated_by=terminated_by,
    )
    a.scores["oracle"] = Score(value=value, metadata={"gate_passed": gate_passed})
    return a


def test_passed_gate_with_nonpositive_net_is_a_solve() -> None:
    """scoring-stats-001 (RED→GREEN): the authoritative solve signal is the
    §8.3 gate verdict (``metadata['gate_passed']``), NOT the tiebreaker net.

    A clean gate pass (solved, no regression, over threshold, no critical
    penalty, in budget) whose net lands ≤ 0 because a non-critical
    REGRESSIONAL penalty outweighs solve+bonuses MUST still count as a solve.
    Reading ``value > 0`` mis-derives it as UNSOLVED and corrupts pass^k /
    solve-rate / graduation rollups."""
    neg = _gated_attempt(gate_passed=True, value=-40.0)
    zero = _gated_attempt(gate_passed=True, value=0.0, attempt_id="att-z")

    hist = PuzzleHistory(puzzle_id="pz-1")
    hist.add(neg)
    hist.add(zero)
    assert hist.outcomes == [True, True]
    assert hist.n_solved == 2

    prof = ModelProfile(model="claude-opus-4-8")
    prof.add(neg)
    prof.add(zero)
    assert prof.n_solved == 2


def test_failed_gate_with_positive_net_is_not_a_solve() -> None:
    """scoring-stats-001 (companion): the gate verdict is authoritative in BOTH
    directions — a closed gate is not a solve even if a (meaningless) positive
    net leaked into ``value``. grade() zeroes ``value`` on a closed gate, but we
    assert the gate flag dominates regardless."""
    a = _gated_attempt(gate_passed=False, value=92.0)
    hist = PuzzleHistory(puzzle_id="pz-1")
    hist.add(a)
    assert hist.outcomes == [False]
    assert hist.n_solved == 0


def test_gate_passed_does_not_override_nonterminal_status() -> None:
    """scoring-stats-001 (guard preserved): a non-COMPLETED termination is never
    a solve, even with gate_passed=True — the terminal-status guard wins."""
    a = _gated_attempt(
        gate_passed=True, value=92.0, terminated_by=TerminatedBy.BUDGET
    )
    hist = PuzzleHistory(puzzle_id="pz-1")
    hist.add(a)
    assert hist.outcomes == [False]


def test_score_without_gate_envelope_falls_back_to_truthy() -> None:
    """scoring-stats-001 (fallback): a score lacking the oracle gate envelope
    (e.g. a bool Inspect-style 'C'/True score) still resolves via the tolerant
    truthy path — the gate-flag preference must not break non-oracle scores."""
    hist = PuzzleHistory(puzzle_id="pz-1")
    hist.add(_attempt(attempt_id="b1", solved=True))   # bool value, no gate key
    hist.add(_attempt(attempt_id="b2", solved=False))
    assert hist.outcomes == [True, False]


def test_model_profile_rollup() -> None:
    prof = ModelProfile(model="claude-opus-4-8")
    prof.add(_attempt(solved=True, novel=True, wall_time=2.0))
    prof.add(_attempt(solved=True, novel=False, wall_time=4.0))
    prof.add(_attempt(solved=False, wall_time=6.0))
    assert prof.n_attempts == 3
    assert prof.solve_rate == pytest.approx(2 / 3)
    assert prof.novelty_rate == pytest.approx(1 / 3)
    assert prof.mean_latency == pytest.approx(4.0)
    d = prof.to_dict()
    assert d["model"] == "claude-opus-4-8"
    assert set(d) == {"model", "n_attempts", "solve_rate", "novelty_rate", "mean_latency"}


def test_model_profile_rejects_foreign_model() -> None:
    prof = ModelProfile(model="claude-opus-4-8")
    with pytest.raises(ValueError):
        prof.add(_attempt(model="qwen-2.5-7b"))


def test_roll_up_groups_by_puzzle_and_model() -> None:
    attempts = [
        _attempt(attempt_id="a1", puzzle_id="pz-1", model="claude", solved=True),
        _attempt(attempt_id="a2", puzzle_id="pz-1", model="claude", solved=False),
        _attempt(attempt_id="a3", puzzle_id="pz-2", model="qwen", solved=True, novel=True),
    ]
    out = roll_up(attempts)
    assert out["n_attempts"] == 3
    assert set(out["puzzles"]) == {"pz-1", "pz-2"}
    assert set(out["models"]) == {"claude", "qwen"}

    pz1 = out["puzzles"]["pz-1"]
    assert pz1["n_attempts"] == 2
    assert pz1["n_solved"] == 1
    assert pz1["solve_rate"] == pytest.approx(0.5)
    assert pz1["pass_hat_3"] == pytest.approx(0.125)
    assert 0.0 <= pz1["wilson_lower"] <= pz1["wilson_upper"] <= 1.0

    qwen = out["models"]["qwen"]
    assert qwen["solve_rate"] == pytest.approx(1.0)
    assert qwen["novelty_rate"] == pytest.approx(1.0)


def test_roll_up_empty() -> None:
    out = roll_up([])
    assert out == {"n_attempts": 0, "puzzles": {}, "models": {}}


# --------------------------------------------------------------------------- #
# attestation.py — hash-chained JsonlEventStore
# --------------------------------------------------------------------------- #


def test_canonical_json_is_stable_and_sorted() -> None:
    assert canonical_json({"b": 1, "a": 2}) == '{"a":2,"b":1}'
    # Nested keys are sorted too; no incidental whitespace.
    assert canonical_json({"z": {"y": 1, "x": 2}}) == '{"z":{"x":2,"y":1}}'


def test_chain_hash_binds_prev_and_payload() -> None:
    h1 = chain_hash(GENESIS_HASH, {"event": "a"})
    h2 = chain_hash(h1, {"event": "a"})  # same payload, different prev -> different hash
    assert h1 != h2
    # Deterministic.
    assert chain_hash(GENESIS_HASH, {"event": "a"}) == h1


def test_append_three_events_chain_verifies(tmp_path) -> None:
    store = JsonlEventStore(tmp_path / "log.jsonl")
    h0 = store.append({"type": "ai_crucible.puzzle.attempted.started", "seq": 0})
    h1 = store.append({"type": "ai_crucible.judge.evaluation.completed", "seq": 1})
    h2 = store.append({"type": "ai_crucible.bundle.released", "seq": 2})
    assert len({h0, h1, h2}) == 3  # distinct hashes
    assert len(store) == 3
    assert store.verify_hash_chain() is True
    # Payloads round-trip with the envelope stripped.
    events = store.read_events()
    assert [e["seq"] for e in events] == [0, 1, 2]
    assert events[0]["type"] == "ai_crucible.puzzle.attempted.started"


def test_empty_store_verifies_true(tmp_path) -> None:
    store = JsonlEventStore(tmp_path / "empty.jsonl")
    assert store.verify_hash_chain() is True
    assert len(store) == 0


def test_tamper_breaks_hash_chain(tmp_path) -> None:
    """THE failing case: mutate one stored line on disk -> verify returns False.

    Proves the hash chain actually detects tampering (the whole point of §9.5).
    """
    path = tmp_path / "log.jsonl"
    store = JsonlEventStore(path)
    store.append({"type": "a", "value": 1})
    store.append({"type": "b", "value": 2})
    store.append({"type": "c", "value": 3})
    assert store.verify_hash_chain() is True

    # Tamper: edit the payload of the SECOND line in place, leaving its stored
    # hash untouched. A naive log would never notice; the chain must.
    lines = path.read_text(encoding="utf-8").splitlines()
    obj = json.loads(lines[1])
    obj["payload"]["value"] = 999  # silent content edit
    lines[1] = canonical_json(obj)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    assert store.verify_hash_chain() is False


def test_tamper_on_stored_hash_breaks_chain(tmp_path) -> None:
    """Editing the stored hash of a line (not the payload) also breaks the
    downstream prev_hash link."""
    path = tmp_path / "log.jsonl"
    store = JsonlEventStore(path)
    store.append({"type": "a"})
    store.append({"type": "b"})
    lines = path.read_text(encoding="utf-8").splitlines()
    obj = json.loads(lines[0])
    obj["hash"] = "0" * 64  # bogus hash
    lines[0] = canonical_json(obj)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    assert store.verify_hash_chain() is False


def test_deleting_a_line_breaks_chain(tmp_path) -> None:
    path = tmp_path / "log.jsonl"
    store = JsonlEventStore(path)
    store.append({"type": "a"})
    store.append({"type": "b"})
    store.append({"type": "c"})
    lines = path.read_text(encoding="utf-8").splitlines()
    del lines[1]  # drop the middle entry
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    assert store.verify_hash_chain() is False


def test_verify_raises_on_malformed_envelope(tmp_path) -> None:
    """A line that is not a well-formed chain entry is a structural error
    (HashChainError), distinct from a tampered-but-well-formed chain (False)."""
    path = tmp_path / "log.jsonl"
    path.write_text('{"not":"an envelope"}\n', encoding="utf-8")
    store = JsonlEventStore(path)
    with pytest.raises(HashChainError):
        store.verify_hash_chain()


def test_append_after_load_continues_chain(tmp_path) -> None:
    """A new store object over an existing file continues the chain from the
    persisted tail (durability across process restarts)."""
    path = tmp_path / "log.jsonl"
    JsonlEventStore(path).append({"type": "a"})
    JsonlEventStore(path).append({"type": "b"})  # fresh object, same file
    store = JsonlEventStore(path)
    store.append({"type": "c"})
    assert len(store) == 3
    assert store.verify_hash_chain() is True


def test_append_rejects_non_finite_float_nan(tmp_path) -> None:
    """M2: a payload with a NaN must fail LOUDLY at write time, not silently
    serialize to RFC-8785-illegal ``NaN`` that a conformant verifier rejects.

    RFC 8785 / JSON proper has no ``NaN``/``Infinity`` literals; the Attestia
    ``@attestia/event-store`` this mirrors would reject such a line. AI Crucible's
    lenient ``json.loads`` would *accept* it and ``verify_hash_chain`` would
    return True — a chain that no conformant auditor can verify. So we refuse to
    write the line at all (§9.5: the value is a real integrity proof).
    """
    store = JsonlEventStore(tmp_path / "log.jsonl")
    with pytest.raises(HashChainError):
        store.append({"type": "score", "value": float("nan")})
    # Nothing illegal was committed to disk.
    assert len(store) == 0


def test_append_rejects_non_finite_float_inf(tmp_path) -> None:
    """M2 (companion): Infinity is equally illegal RFC-8785 JSON and must raise."""
    store = JsonlEventStore(tmp_path / "log.jsonl")
    with pytest.raises(HashChainError):
        store.append({"type": "penalty", "weight": float("inf")})
    with pytest.raises(HashChainError):
        store.append({"type": "penalty", "weight": float("-inf")})
    assert len(store) == 0


def test_extra_top_level_envelope_key_is_rejected(tmp_path) -> None:
    """L1: an unhashed extra top-level envelope key must NOT survive verification.

    ``hash`` only covers ``prev_hash + payload``. With the old subset key-set
    check, an attacker who can write the file could attach extra top-level keys
    that are never hashed yet still verify True. The envelope must be an EXACT
    key-set so any annotation/injection is rejected.

    A line carrying an unhashed extra top-level key is a *malformed envelope*,
    not a well-formed-but-tampered chain — so verification rejects it exactly the
    way :func:`test_verify_raises_on_malformed_envelope` rejects a line missing
    the envelope fields: via :class:`HashChainError` (the documented contract,
    lines 206-209 of attestation.py). Either way it does NOT verify True, which
    is the whole point: pre-fix the subset check returned True (RED); post-fix it
    is rejected (GREEN).
    """
    path = tmp_path / "log.jsonl"
    store = JsonlEventStore(path)
    store.append({"type": "a", "value": 1})
    store.append({"type": "b", "value": 2})
    assert store.verify_hash_chain() is True

    # Inject an extra, unhashed top-level key into a stored line on disk.
    lines = path.read_text(encoding="utf-8").splitlines()
    obj = json.loads(lines[0])
    obj["INJECTED"] = "x"  # not part of prev_hash+payload, so the hash is silent
    lines[0] = canonical_json(obj)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # The injected line no longer survives verification. Belt-and-suspenders:
    # whether the implementation raises (malformed-envelope contract) or returns
    # False, it must never report the tampered chain as intact.
    with pytest.raises(HashChainError):
        store.verify_hash_chain()


# --------------------------------------------------------------------------- #
# attestation.py — cosign stub (the documented production edge)
# --------------------------------------------------------------------------- #


def test_cosign_sign_blob_disabled_returns_none(tmp_path) -> None:
    """The polyglot signing edge is a no-op stub unless explicitly enabled, and
    NEVER fabricates a signature (§9.5)."""
    blob = tmp_path / "bundle.json"
    blob.write_text("{}", encoding="utf-8")
    assert cosign_sign_blob(blob, enabled=False) is None
    # Default is disabled.
    assert cosign_sign_blob(blob) is None


def test_cosign_sign_blob_enabled_missing_file_raises(tmp_path) -> None:
    """Enabled but the target is missing -> structured error, never a fake sig."""
    with pytest.raises(HashChainError):
        cosign_sign_blob(tmp_path / "does-not-exist.json", enabled=True)
