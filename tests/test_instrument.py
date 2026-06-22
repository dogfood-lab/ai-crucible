"""Tests for the instrument-quality scaffolding (ai_crucible.instrument, §9).

Covers, per the build contract:
- aspredicted_template has 9 questions
- rubric bundle hash is stable AND changes when a weight changes (both proven)
- bump_on_change returns a new version on a changed bundle, the same on identical
- split_inventory partitions 100 ids into 60/20/10/10 deterministically
- sobol_screen returns a T_i per param on a toy additive model; zero-effect ≈ 0
- render_sut_yaml round-trips the fields
- to_inspect_task carries the puzzle id + prompt

Plus failing cases: structured errors on bad input, the ANDON budget gate, the
sealed-oracle invariant (the answer never appears in the task definition), and
the documented-stub NotImplementedError.
"""

from __future__ import annotations

import json
import warnings

import numpy as np
import pytest

import ai_crucible.instrument.inspect_task as inspect_task_mod
from ai_crucible.instrument import (
    ASPREDICTED_QUESTION_IDS,
    SUT,
    InspectTaskError,
    PreregError,
    RubricBundle,
    RubricBundleError,
    SUTError,
    ThresholdoutBudget,
    TuningBudgetError,
    TuningError,
    aspredicted_template,
    assert_versioned,
    bo_search,
    bump_on_change,
    canonical_bundle_json,
    compile_bundle,
    paraphrase_ablate,
    parse_sut_yaml,
    reforms_checklist,
    render_preregistration,
    render_sut_yaml,
    sobol_screen,
    split_inventory,
    to_inspect_task,
    two_repo_layout,
)

# The version-increment helper is private (not in __all__); import it from the
# submodule directly so the public surface stays unchanged.
from ai_crucible.instrument.rubric_bundle import _next_version

# --------------------------------------------------------------------------- #
# prereg
# --------------------------------------------------------------------------- #


def test_aspredicted_template_has_nine_questions():
    template = aspredicted_template()
    assert len(template["questions"]) == 9
    # Question numbers are 1..9 in order.
    assert [q["number"] for q in template["questions"]] == list(range(1, 10))
    # IDs match the exported ordering tuple.
    assert tuple(q["id"] for q in template["questions"]) == ASPREDICTED_QUESTION_IDS
    assert len(ASPREDICTED_QUESTION_IDS) == 9


def test_aspredicted_template_locks_the_section_9_3_surfaces():
    template = aspredicted_template()
    # The §9.3 load-bearing locks must all be declared.
    assert set(template["locked_surfaces"]) == {
        "rubric",
        "axes",
        "model_list",
        "k_seeds",
        "primary_test",
        "correction",
    }
    # The analyses question pins McNemar (primary test) and BH-FDR (correction).
    analyses = next(q for q in template["questions"] if q["id"] == "analyses")
    assert "McNemar" in analyses["answer"]
    assert "Benjamini-Hochberg" in analyses["answer"] or "BH-FDR" in analyses["answer"]
    assert set(analyses["locks"]) == {"primary_test", "correction"}


def test_reforms_checklist_is_a_nonempty_skeleton():
    items = reforms_checklist()
    assert isinstance(items, list)
    assert len(items) >= 10
    # Every item is a todo skeleton with the right shape.
    for it in items:
        assert set(it.keys()) == {"id", "section", "item", "status", "note"}
        assert it["status"] == "todo"
    # IDs are unique.
    ids = [it["id"] for it in items]
    assert len(set(ids)) == len(ids)
    # Sections cover the REFORMS structure.
    sections = {it["section"] for it in items}
    assert {"study_design", "data", "modeling", "reporting"} <= sections


def test_render_preregistration_is_deterministic_and_complete():
    template = aspredicted_template()
    answers = {q["id"]: q["answer"] for q in template["questions"]}
    md1 = render_preregistration(answers)
    md2 = render_preregistration(answers)
    assert md1 == md2  # PIN_PER_STEP: same answers -> same bytes
    assert md1.startswith("# AI Crucible — Pre-registration")
    # All nine questions render.
    for q in template["questions"]:
        assert f"Q{q['number']}." in md1


def test_render_preregistration_raises_on_missing_answer():
    template = aspredicted_template()
    answers = {q["id"]: q["answer"] for q in template["questions"]}
    del answers["hypothesis"]  # drop a required answer
    with pytest.raises(PreregError) as exc:
        render_preregistration(answers)
    # Structured error shape.
    assert "INPUT_PREREG_MISSING_ANSWER" in str(exc.value)
    assert "hint:" in str(exc.value)


def test_render_preregistration_rejects_empty_string_answer():
    template = aspredicted_template()
    answers = {q["id"]: q["answer"] for q in template["questions"]}
    answers["other"] = "   "  # whitespace-only is not a real answer
    with pytest.raises(PreregError):
        render_preregistration(answers)


# --------------------------------------------------------------------------- #
# rubric_bundle
# --------------------------------------------------------------------------- #


def _bundle(**overrides) -> RubricBundle:
    base = dict(
        weights={"answer_key_fetch": -150.0, "elegance_bonus_max": 24.0},
        thresholds={"point_threshold": 50.0, "solve_threshold": 0.8},
        judge_prompts={"novelty": "Is this a legitimate novel path?"},
        version="v1.0",
    )
    base.update(overrides)
    return RubricBundle(**base)


def test_compile_bundle_hash_is_stable():
    b1 = _bundle()
    b2 = _bundle()
    h1, bytes1 = compile_bundle(b1)
    h2, bytes2 = compile_bundle(b2)
    assert h1 == h2  # identical content -> identical hash
    assert bytes1 == bytes2
    assert len(h1) == 64  # sha256 hex
    # Canonical bytes are sorted-key minimal-separator JSON.
    parsed = json.loads(bytes1)
    assert set(parsed.keys()) == {"weights", "thresholds", "judge_prompts"}


def test_compile_bundle_hash_changes_when_a_weight_changes():
    b1 = _bundle()
    b2 = _bundle(weights={"answer_key_fetch": -151.0, "elegance_bonus_max": 24.0})
    h1, _ = compile_bundle(b1)
    h2, _ = compile_bundle(b2)
    assert h1 != h2  # a single weight delta moves the content hash


def test_compile_bundle_hash_ignores_version_label():
    # The version label is NOT part of the hashed content.
    h1, _ = compile_bundle(_bundle(version="v1.0"))
    h2, _ = compile_bundle(_bundle(version="v9.9"))
    assert h1 == h2


def test_compile_bundle_rejects_nan_weight():
    bad = _bundle(weights={"x": float("nan")})
    with pytest.raises(RubricBundleError) as exc:
        compile_bundle(bad)
    assert "INPUT_BUNDLE_UNSERIALIZABLE" in str(exc.value)


def test_bump_on_change_same_bundle_keeps_version():
    old = _bundle(version="v1.0")
    new = _bundle(version="v1.0")  # identical content
    assert bump_on_change(old, new) == "v1.0"


def test_bump_on_change_changed_bundle_advances_version():
    old = _bundle(version="v1.0")
    new = _bundle(version="v1.0", thresholds={"point_threshold": 55.0, "solve_threshold": 0.8})
    bumped = bump_on_change(old, new)
    assert bumped != "v1.0"
    assert bumped == "v1.1"  # last dotted numeric segment incremented


@pytest.mark.parametrize(
    ("old_version", "expected"),
    [
        ("v1.0", "v1.1"),
        ("1.2.3", "1.2.4"),
        ("v3", "v4"),
        ("release", "release+1"),  # unparseable -> provable +1 suffix
    ],
)
def test_bump_on_change_version_scheme(old_version, expected):
    old = _bundle(version=old_version)
    new = _bundle(version=old_version, judge_prompts={"novelty": "DIFFERENT prompt"})
    assert bump_on_change(old, new) == expected


def test_canonical_bundle_json_is_byte_stable_across_key_order():
    # Building the same content with different insertion order yields same bytes.
    b1 = RubricBundle(weights={"a": 1.0, "b": 2.0}, thresholds={}, judge_prompts={})
    b2 = RubricBundle(weights={"b": 2.0, "a": 1.0}, thresholds={}, judge_prompts={})
    assert canonical_bundle_json(b1) == canonical_bundle_json(b2)


# --------------------------------------------------------------------------- #
# rubric_bundle — assert_versioned (the §9.1 ENFORCING seal)  [finding 001]
# --------------------------------------------------------------------------- #
#
# bump_on_change is a pure *suggester*: nothing forces the caller to assign its
# return. The ENFORCING half of the §9.1 "version changes iff hash changes"
# invariant is assert_versioned — a real andon gate that RAISES when the scoring
# content changed but the version label did not. The RED probe MUTATES the
# protected thing (a weight, which moves the content hash) while leaving the
# version label fixed, and asserts the gate FIRES.


def test_assert_versioned_raises_when_content_changed_but_version_pinned():
    # MUTATE THE PROTECTED THING: a weight delta moves the content hash.
    old = _bundle(version="v1.0")
    new = _bundle(version="v1.0", weights={"answer_key_fetch": -151.0, "elegance_bonus_max": 24.0})
    # The hash genuinely changed (verify the premise), so leaving version pinned
    # is exactly the §9.1 "silent retconning" the seal must refuse.
    assert compile_bundle(old)[0] != compile_bundle(new)[0]
    with pytest.raises(RubricBundleError) as exc:
        assert_versioned(old, new)
    # Structured error shape, naming the unbumped version vector.
    assert "STATE_RUBRIC_VERSION_NOT_BUMPED" in str(exc.value)
    assert "hint:" in str(exc.value)


def test_assert_versioned_passes_when_changed_content_carries_new_version():
    old = _bundle(version="v1.0")
    new = _bundle(version="v1.1", weights={"answer_key_fetch": -151.0, "elegance_bonus_max": 24.0})
    # Content changed AND version advanced — the gate is satisfied and returns
    # the (already-correct) new hash for the caller to record.
    new_hash = assert_versioned(new=new, old=old)
    assert new_hash == compile_bundle(new)[0]


def test_assert_versioned_passes_on_identical_content_same_version():
    # No-op edit, same label — nothing to enforce.
    old = _bundle(version="v1.0")
    new = _bundle(version="v1.0")
    assert assert_versioned(old, new) == compile_bundle(new)[0]


def test_assert_versioned_rejects_relabel_of_identical_content():
    # Same scoring content but a NEW label is also a §9.1 violation: two
    # byte-identical bundles must never carry two different versions.
    old = _bundle(version="v1.0")
    new = _bundle(version="v2.0")  # identical content, different label
    with pytest.raises(RubricBundleError) as exc:
        assert_versioned(old, new)
    assert "STATE_RUBRIC_VERSION_DRIFT" in str(exc.value)


def test_bump_on_change_noop_preserves_callers_relabel():
    # Regression for the no-op branch that silently discarded the caller's label:
    # identical content but an explicit relabel should keep the NEW label, not
    # silently revert to old.version.
    old = _bundle(version="v1.0")
    new = _bundle(version="v2.0")  # identical content, deliberate relabel
    assert bump_on_change(old, new) == "v2.0"


def test_bump_on_change_noop_same_label_keeps_it():
    old = _bundle(version="v1.0")
    new = _bundle(version="v1.0")  # identical content, same label
    assert bump_on_change(old, new) == "v1.0"


# --------------------------------------------------------------------------- #
# rubric_bundle — _next_version date-style / zero-pad handling  [finding 002]
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("old_version", "expected"),
    [
        ("2026.06", "2026.07"),  # zero-pad preserved, NOT mangled to 2026.7
        ("v2026.06", "v2026.07"),
        ("2026.09", "2026.10"),  # carry past the pad width widens correctly
        ("v1.09", "v1.10"),
        ("v01", "v02"),  # single zero-padded segment
        ("v1.", "v1.+1"),  # dangling dot is not a numeric segment -> +1 suffix
    ],
)
def test_next_version_preserves_zero_padding_and_dangling_dot(old_version, expected):
    assert _next_version(old_version) == expected


def test_bump_on_change_date_style_label_not_mangled():
    # End-to-end through bump_on_change: a calendar-pinned rubric that changes
    # content advances to a sane next-month label, not a zero-pad-stripped one.
    old = _bundle(version="2026.06")
    new = _bundle(version="2026.06", judge_prompts={"novelty": "DIFFERENT prompt"})
    assert bump_on_change(old, new) == "2026.07"


# --------------------------------------------------------------------------- #
# tuning — split_inventory
# --------------------------------------------------------------------------- #


def test_split_inventory_partitions_100_ids_60_20_10_10():
    ids = [f"p{i:03d}" for i in range(100)]
    result = split_inventory(ids, seed_note="phase-4-cycle-1")
    splits = result["splits"]
    assert splits["calibration"]["n"] == 60
    assert splits["dev"]["n"] == 20
    assert splits["validation"]["n"] == 10
    assert splits["private_test"]["n"] == 10
    # Partition is clean: union == input set, pairwise disjoint.
    all_ids = (
        splits["calibration"]["ids"]
        + splits["dev"]["ids"]
        + splits["validation"]["ids"]
        + splits["private_test"]["ids"]
    )
    assert sorted(all_ids) == sorted(ids)
    assert len(set(all_ids)) == 100
    assert result["private_test_sealed"] is True


def test_split_inventory_is_deterministic():
    ids = [f"p{i:03d}" for i in range(100)]
    r1 = split_inventory(ids, seed_note="same-note")
    r2 = split_inventory(ids, seed_note="same-note")
    # Same ids + same seed_note -> byte-identical split (and manifest hashes).
    for name in ("calibration", "dev", "validation", "private_test"):
        assert r1["splits"][name]["ids"] == r2["splits"][name]["ids"]
        assert r1["splits"][name]["manifest_sha256"] == r2["splits"][name]["manifest_sha256"]


def test_split_inventory_seed_note_changes_partition():
    ids = [f"p{i:03d}" for i in range(100)]
    r1 = split_inventory(ids, seed_note="note-A")
    r2 = split_inventory(ids, seed_note="note-B")
    # Different seed_note -> different assignment (overwhelmingly likely; the
    # manifest hashes differ because seed_note is part of the hashed material).
    h1 = r1["splits"]["calibration"]["manifest_sha256"]
    h2 = r2["splits"]["calibration"]["manifest_sha256"]
    assert h1 != h2


def test_split_inventory_independent_of_input_order():
    # Determinism is keyed on (ids-as-a-set, seed_note), not enumeration order:
    # shuffling the input must not change the resulting splits.
    ids = [f"p{i:03d}" for i in range(100)]
    shuffled = list(reversed(ids))
    r1 = split_inventory(ids, seed_note="n")
    r2 = split_inventory(shuffled, seed_note="n")
    for name in ("calibration", "dev", "validation", "private_test"):
        assert r1["splits"][name]["ids"] == r2["splits"][name]["ids"]


def test_split_inventory_sums_to_total_for_non_round_n():
    ids = [f"p{i}" for i in range(37)]  # not divisible cleanly
    result = split_inventory(ids, seed_note="x")
    total = sum(result["splits"][n]["n"] for n in result["splits"])
    assert total == 37  # largest-remainder rounding loses nobody


def test_split_inventory_rejects_empty():
    with pytest.raises(TuningError) as exc:
        split_inventory([], seed_note="x")
    assert "INPUT_INVENTORY_EMPTY" in str(exc.value)


def test_split_inventory_rejects_duplicates():
    with pytest.raises(TuningError) as exc:
        split_inventory(["a", "b", "a"], seed_note="x")
    assert "INPUT_INVENTORY_DUPLICATE" in str(exc.value)


# --------------------------------------------------------------------------- #
# tuning — sobol_screen
# --------------------------------------------------------------------------- #


def test_sobol_screen_returns_t_i_per_param_zero_effect_near_zero():
    # Toy additive model: y = 3*x0 + 0*x1 + 1*x2. x1 has zero total effect.
    def model_fn(x: np.ndarray) -> np.ndarray:
        return 3.0 * x[0] + 0.0 * x[1] + 1.0 * x[2]

    t = sobol_screen(
        param_names=["w0", "w1", "w2"],
        bounds=[(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)],
        model_fn=model_fn,
        n=512,
        seed=0,
    )
    assert set(t.keys()) == {"w0", "w1", "w2"}
    # Zero-effect parameter screens to ~0 (frozen by §9.4 step 2).
    assert t["w1"] < 0.05
    # The load-bearing parameter dominates.
    assert t["w0"] > t["w1"]
    assert t["w0"] > t["w2"]
    # All non-negative (clipped).
    assert all(v >= 0.0 for v in t.values())


def test_sobol_screen_is_reproducible_with_seed():
    def model_fn(x: np.ndarray) -> np.ndarray:
        return 2.0 * x[0] + x[1]

    kw = dict(
        param_names=["a", "b"],
        bounds=[(0.0, 1.0), (0.0, 1.0)],
        model_fn=model_fn,
        n=256,
        seed=7,
    )
    assert sobol_screen(**kw) == sobol_screen(**kw)


def test_sobol_screen_rejects_length_mismatch():
    with pytest.raises(TuningError) as exc:
        sobol_screen(
            param_names=["a", "b"],
            bounds=[(0.0, 1.0)],  # too few bounds
            model_fn=lambda x: x[0],
            n=64,
        )
    assert "INPUT_SOBOL_LEN_MISMATCH" in str(exc.value)


def test_sobol_screen_rejects_degenerate_bound():
    with pytest.raises(TuningError) as exc:
        sobol_screen(
            param_names=["a"],
            bounds=[(1.0, 1.0)],  # low == high
            model_fn=lambda x: x[0],
            n=64,
        )
    assert "INPUT_SOBOL_DEGENERATE_BOUND" in str(exc.value)


def test_sobol_screen_rejects_empty_params():
    with pytest.raises(TuningError) as exc:
        sobol_screen(param_names=[], bounds=[], model_fn=lambda x: x[0], n=64)
    assert "INPUT_SOBOL_NO_PARAMS" in str(exc.value)


def test_sobol_screen_surfaces_nan_objective_not_silent_zero():
    # [instrument-future-deps-001] A NaN objective must NOT be silently masked to
    # T_i=0.0 (which would FREEZE a load-bearing weight as zero-sensitivity).
    # SIMULATE the failure: an objective that returns NaN for some samples, and
    # assert the screen SURFACES it (ANDON raise) rather than reporting a clean 0.0.
    def nan_model_fn(x: np.ndarray) -> np.ndarray:
        out = np.asarray(3.0 * x[0] + 1.0 * x[1], dtype=float)
        out[0] = np.nan  # one Saltelli sample evaluates to NaN
        return out

    with pytest.raises(TuningError) as exc:
        sobol_screen(
            param_names=["w0", "w1"],
            bounds=[(0.0, 1.0), (0.0, 1.0)],
            model_fn=nan_model_fn,
            n=128,
            seed=0,
        )
    assert "STATE_SOBOL_NONFINITE_OBJECTIVE" in str(exc.value)
    assert "hint:" in str(exc.value)


# --------------------------------------------------------------------------- #
# tuning — ThresholdoutBudget
# --------------------------------------------------------------------------- #


def test_thresholdout_budget_tracks_and_exhausts():
    budget = ThresholdoutBudget(budget=3)
    assert budget.remaining == 3
    assert not budget.exhausted
    assert budget.query("trial-1") == 2
    assert budget.query("trial-2") == 1
    assert budget.query("trial-3") == 0
    assert budget.exhausted


def test_thresholdout_budget_raises_when_exhausted():
    budget = ThresholdoutBudget(budget=1)
    budget.query()
    with pytest.raises(TuningBudgetError) as exc:
        budget.query()  # ANDON: over-querying the holdout halts here
    assert "STATE_DEV_BUDGET_EXHAUSTED" in str(exc.value)
    # Contrastive message points to the §9.4 step-5 next action.
    assert "validation" in str(exc.value)


def test_thresholdout_budget_provenance():
    budget = ThresholdoutBudget(budget=5)
    budget.query()
    budget.query()
    prov = budget.provenance()
    assert prov["dev_queries_used_of_budget"] == "2/5"
    assert prov["remaining"] == 3


def test_thresholdout_budget_rejects_negative():
    with pytest.raises(TuningError) as exc:
        ThresholdoutBudget(budget=-1)
    assert "INPUT_BUDGET_NEGATIVE" in str(exc.value)


# --------------------------------------------------------------------------- #
# tuning — documented stubs (honest NotImplemented, never a fake result)
# --------------------------------------------------------------------------- #


def test_bo_search_is_a_documented_stub():
    with pytest.raises(NotImplementedError) as exc:
        bo_search()
    assert "bo_search" in str(exc.value)


def test_paraphrase_ablate_is_a_documented_stub():
    with pytest.raises(NotImplementedError) as exc:
        paraphrase_ablate()
    assert "paraphrase_ablate" in str(exc.value)


# --------------------------------------------------------------------------- #
# sut
# --------------------------------------------------------------------------- #


def _sut() -> SUT:
    return SUT(
        model_id="claude-opus-4-7-20260415",
        provider_endpoint="https://api.anthropic.com/v1/messages",
        system_prompt_sha="a" * 64,
        harness_commit_sha="b" * 40,
        container_digest="sha256:" + "c" * 64,
    )


def test_render_sut_yaml_round_trips_the_fields():
    sut = _sut()
    yaml_text = render_sut_yaml(sut)
    parsed = parse_sut_yaml(yaml_text)
    assert parsed == sut  # dataclass equality across all five fields
    # Specifically the exact-version model_id survives.
    assert parsed.model_id == "claude-opus-4-7-20260415"
    assert parsed.container_digest.startswith("sha256:")


def test_render_sut_yaml_is_deterministic():
    sut = _sut()
    assert render_sut_yaml(sut) == render_sut_yaml(sut)


def test_render_sut_yaml_rejects_family_alias():
    sut = SUT(
        model_id="claude-opus",  # no version/snapshot -> alias
        provider_endpoint="https://api.anthropic.com/v1/messages",
        system_prompt_sha="a" * 64,
        harness_commit_sha="b" * 40,
        container_digest="sha256:" + "c" * 64,
    )
    with pytest.raises(SUTError) as exc:
        render_sut_yaml(sut)
    assert "INPUT_SUT_FAMILY_ALIAS" in str(exc.value)


@pytest.mark.parametrize(
    "alias_model_id",
    [
        "claude-opus²",  # superscript TWO — str.isdigit() True, NOT ascii
        "claude-opus-٤",  # Arabic-Indic digit FOUR — isdigit() True, not ascii
        "gpt⁵",  # superscript FIVE
    ],
)
def test_render_sut_yaml_rejects_non_ascii_digit_alias(alias_model_id):
    # [instrument-future-deps-004] A model id whose only "digit" is a non-ASCII
    # numeral/superscript is still a family alias — str.isdigit() would let it
    # masquerade as an exact version and WEAKEN the §9.6 guardrail. SIMULATE such
    # an id and assert the guardrail FIRES (alias rejected).
    sut = SUT(
        model_id=alias_model_id,
        provider_endpoint="https://api.anthropic.com/v1/messages",
        system_prompt_sha="a" * 64,
        harness_commit_sha="b" * 40,
        container_digest="sha256:" + "c" * 64,
    )
    with pytest.raises(SUTError) as exc:
        render_sut_yaml(sut)
    assert "INPUT_SUT_FAMILY_ALIAS" in str(exc.value)


def test_render_sut_yaml_accepts_ascii_digit_version():
    # Regression guard for the ASCII-only fix: a legitimate ASCII-digit version
    # must still pass the guardrail (the fix must not over-reject).
    sut = SUT(
        model_id="mistral-large-2411",  # ASCII digits -> exact version
        provider_endpoint="https://api.mistral.ai/v1/chat/completions",
        system_prompt_sha="a" * 64,
        harness_commit_sha="b" * 40,
        container_digest="sha256:" + "c" * 64,
    )
    # Does not raise; round-trips cleanly.
    assert parse_sut_yaml(render_sut_yaml(sut)) == sut


def test_render_sut_yaml_rejects_empty_field():
    sut = SUT(
        model_id="claude-opus-4-7-20260415",
        provider_endpoint="",  # empty
        system_prompt_sha="a" * 64,
        harness_commit_sha="b" * 40,
        container_digest="sha256:" + "c" * 64,
    )
    with pytest.raises(SUTError) as exc:
        render_sut_yaml(sut)
    assert "INPUT_SUT_EMPTY_FIELD" in str(exc.value)


def test_parse_sut_yaml_rejects_missing_field():
    incomplete = 'model_id: "claude-opus-4-7-20260415"\n'
    with pytest.raises(SUTError) as exc:
        parse_sut_yaml(incomplete)
    assert "INPUT_SUT_MISSING_FIELD" in str(exc.value)


def test_render_sut_yaml_round_trips_values_with_special_chars():
    # A system prompt sha is hex, but the endpoint can carry chars that must be
    # quoted/escaped; prove the round-trip survives quotes and backslashes.
    sut = SUT(
        model_id="qwen2.5-72b-instruct-2026-03-01",
        provider_endpoint='http://local:11434/v1 "panel"\\node',
        system_prompt_sha="d" * 64,
        harness_commit_sha="e" * 40,
        container_digest="sha256:" + "f" * 64,
    )
    assert parse_sut_yaml(render_sut_yaml(sut)) == sut


def test_parse_sut_yaml_rejects_duplicate_key():
    # [instrument-future-deps-003] A duplicate key must be REJECTED, not silently
    # resolved last-wins (which would quietly discard a pinned value from the
    # frozen reproducibility record). SIMULATE the malformed input and assert the
    # structured rejection.
    text = (
        'model_id: "claude-opus-4-7-20260415"\n'
        'model_id: "claude-opus-4-7-99999999"\n'  # duplicate, would last-win today
        'provider_endpoint: "https://api.anthropic.com/v1/messages"\n'
        f'system_prompt_sha: "{"a" * 64}"\n'
        f'harness_commit_sha: "{"b" * 40}"\n'
        f'container_digest: "sha256:{"c" * 64}"\n'
    )
    with pytest.raises(SUTError) as exc:
        parse_sut_yaml(text)
    assert "INPUT_SUT_DUPLICATE_KEY" in str(exc.value)
    assert "hint:" in str(exc.value)


def test_parse_sut_yaml_rejects_indented_nested_structure():
    # [instrument-future-deps-003] An indented / nested-list line is unrepresentable
    # in this flat format and must be rejected rather than misread or flattened.
    text = (
        'model_id: "claude-opus-4-7-20260415"\n'
        "provider_endpoint:\n"
        '  - "https://api.anthropic.com/v1/messages"\n'  # malformed indented list
        f'system_prompt_sha: "{"a" * 64}"\n'
        f'harness_commit_sha: "{"b" * 40}"\n'
        f'container_digest: "sha256:{"c" * 64}"\n'
    )
    with pytest.raises(SUTError) as exc:
        parse_sut_yaml(text)
    assert "INPUT_SUT_BAD_LINE" in str(exc.value)


# --------------------------------------------------------------------------- #
# inspect_task
# --------------------------------------------------------------------------- #


def test_to_inspect_task_carries_puzzle_id_and_prompt(sample_meta):
    prompt = "Find the function that computes the retry backoff and report its base."
    task = to_inspect_task(sample_meta, prompt)
    # Puzzle id appears as the sample id and in the task name/metadata.
    assert task["dataset"][0]["id"] == sample_meta.puzzle_id
    assert task["metadata"]["puzzle_id"] == sample_meta.puzzle_id
    assert sample_meta.puzzle_id in task["name"]
    # Prompt is the Solver-facing input.
    assert task["dataset"][0]["input"] == prompt


def test_to_inspect_task_maps_budgets_and_epochs(sample_meta):
    task = to_inspect_task(sample_meta, "do the thing")
    assert task["limits"]["message_limit"] == sample_meta.tool_call_budget
    assert task["limits"]["time_limit"] == sample_meta.time_budget_seconds
    assert task["epochs"] == sample_meta.min_k  # pass^k native
    # Capability metadata is carried for the record.
    assert task["dataset"][0]["metadata"]["capability_aspect"] == sample_meta.capability_aspect
    assert task["dataset"][0]["metadata"]["puzzle_class"] == sample_meta.puzzle_class.value


def test_to_inspect_task_seals_the_oracle(sample_meta):
    # SEALED-ORACLE INVARIANT (§10.4): the scorer is a reference, never the
    # oracle itself, and the Solver-visible target is empty. The whole serialised
    # task definition must not contain any answer/oracle assertion content.
    task = to_inspect_task(sample_meta, "prompt text")
    assert task["dataset"][0]["target"] == ""  # Solver never sees the answer
    assert isinstance(task["scorer"], str)  # a reference string
    assert task["scorer"].startswith("ai_crucible/oracle_scorer#")
    assert sample_meta.puzzle_id in task["scorer"]
    # The serialised task is JSON-roundtrippable (auditor tooling consumes it).
    dumped = json.dumps(task)
    assert "oracle_scorer" in dumped


def test_to_inspect_task_sandbox_shape(sample_meta):
    task = to_inspect_task(sample_meta, "x")
    # SandboxEnvironmentSpec(type, config) 2-tuple shape; harness fills config.
    assert task["sandbox"][0] == "docker"
    assert task["sandbox"][1] is None


def test_to_inspect_task_rejects_empty_prompt(sample_meta):
    with pytest.raises(InspectTaskError) as exc:
        to_inspect_task(sample_meta, "   ")
    assert "INPUT_TASK_EMPTY_PROMPT" in str(exc.value)


def test_two_repo_layout_describes_the_split():
    layout = two_repo_layout()
    assert set(layout["repos"].keys()) == {"ai_crucible-harness", "ai_crucible-results"}
    # The pre-registration + provenance live on the results side, not the harness.
    results_raw = " ".join(layout["repos"]["ai_crucible-results"]["contains"])
    results_lc = results_raw.lower()
    assert "pre-registration" in results_lc or "preregistration" in results_lc
    assert "TUNING.md" in results_raw


# --------------------------------------------------------------------------- #
# inspect_task — inspect-ai version drift guard  [finding 002]
# --------------------------------------------------------------------------- #


def test_inspect_task_warns_on_untested_inspect_ai_version(sample_meta, monkeypatch):
    # [instrument-future-deps-002] inspect-ai is pinned >=0.3 with no upper bound;
    # the Sample.model_dump() shape is consumed wholesale. SIMULATE a future
    # version outside the tested range and assert the runtime check WARNS (drift
    # made legible) rather than consuming a possibly-drifted shape silently.
    monkeypatch.setattr(
        inspect_task_mod, "version", lambda _name: "0.9.0", raising=True
    )
    with pytest.warns(UserWarning, match="CONFIG_INSPECT_AI_UNTESTED_VERSION"):
        to_inspect_task(sample_meta, "do the thing")


def test_inspect_task_silent_on_in_range_inspect_ai_version(sample_meta, monkeypatch):
    # In-range version: no warning (the check only fires on drift).
    monkeypatch.setattr(
        inspect_task_mod, "version", lambda _name: "0.3.233", raising=True
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any UserWarning would fail the test
        to_inspect_task(sample_meta, "do the thing")


def test_inspect_task_guards_drifted_sample_shape(sample_meta, monkeypatch):
    # [instrument-future-deps-002] SIMULATE a future Inspect whose Sample
    # serialisation dropped a key the task definition consumes ('input'). The
    # defensive shape guard must surface it (ANDON) instead of emitting an
    # unscoreable task definition missing the Solver-facing prompt.
    real_dump = inspect_task_mod.Sample.model_dump

    def _drifted_dump(self, *args, **kwargs):  # noqa: ANN001
        d = real_dump(self, *args, **kwargs)
        d.pop("input", None)  # the renamed/dropped key a future Inspect might do
        return d

    monkeypatch.setattr(inspect_task_mod.Sample, "model_dump", _drifted_dump)
    with pytest.raises(InspectTaskError) as exc:
        to_inspect_task(sample_meta, "do the thing")
    assert "STATE_INSPECT_SAMPLE_SHAPE" in str(exc.value)
    assert "hint:" in str(exc.value)


def test_inspect_task_guards_non_dict_sample_dump(sample_meta, monkeypatch):
    # [instrument-future-deps-002] sibling: a Sample.model_dump() that returns a
    # non-dict (e.g. a list under a future schema) must also be rejected with the
    # same structured error, not AttributeError on the downstream .pop/indexing.
    monkeypatch.setattr(
        inspect_task_mod.Sample, "model_dump", lambda self, *a, **k: ["not", "a", "dict"]
    )
    with pytest.raises(InspectTaskError) as exc:
        to_inspect_task(sample_meta, "do the thing")
    assert "STATE_INSPECT_SAMPLE_SHAPE" in str(exc.value)
