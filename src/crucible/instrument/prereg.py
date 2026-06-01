"""Stage 1 — Pre-registration scaffolding (research-grounding §9.3).

Crucible is a measurement instrument, and the literature on keeping a tuned
instrument trustworthy is unambiguous: **lock the methodology and the scoring
formula before the production run** (Gelman & Loken 2014 "Garden of Forking
Paths"; Simmons, Nelson & Simonsohn 2011 "False-Positive Psychology" —
undisclosed flexibility inflates the false-positive rate from a nominal 5% to
~60%). This module produces the two artifacts §9.3 mandates *before* Phase 4
runs a single puzzle:

1. :func:`aspredicted_template` — the 9-question AsPredicted short form
   (https://aspredicted.org; Chambers & Tzavella 2022, *Nature Human Behaviour*
   6:29-42, doi:10.1038/s41562-021-01193-7). 30 minutes to fill, yields a
   timestamped citable URL.
2. :func:`reforms_checklist` — the REFORMS items skeleton (Kapoor et al. 2024,
   *Science Advances* 10(18):eadk3452; arXiv:2308.07832), the 19-author
   ML-science reporting consensus, published alongside results.
3. :func:`render_preregistration` — renders filled answers to a Markdown
   document suitable for committing to ``crucible-results`` and pasting into the
   AsPredicted form.

These are **scaffolding** (§9.9): they don't run a puzzle, they make Phase 4
audit-ready by construction. The locked content is fixed by §9.3 and not a
free-for-all: the **rubric, axes, model list, K seeds, primary statistical test
(McNemar's exact), and the multiple-comparison correction (BH-FDR)** are the
load-bearing locks. The template pre-fills crucible's defaults for exactly those
so the act of pre-registration cannot silently drift from the locked design.

Standards compliance (the six — workflow-standards.md):
- PIN_PER_STEP — 3: every function is a pure function of its arguments with no
  clock/RNG/IO; the same ``answers`` dict renders the same Markdown byte-for-byte,
  so a pre-registration artifact is replayable from its inputs.
- ANDON_AUTHORITY — 2: :func:`render_preregistration` *raises* (structured
  ``PreregError``) on a missing required answer rather than emitting a
  silently-incomplete pre-registration — a defect halts here instead of
  propagating a half-locked methodology downstream into the production run.
- NAMED_COMPENSATORS — n/a: no irreversible tool calls (pure, in-memory string
  building). The downstream irreversible act (filing the AsPredicted URL) is a
  human step outside this module.
- DECOMPOSE_BY_SECRETS — 3: the pre-registration surface (what is locked, the
  question text) changes together and lives here; it shares no state with the
  tuning or scoring surfaces. The *answers* (which change per study) are passed
  in, separate from the *form* (which is stable).
- UNCERTAINTY_GATED_HUMANS — 3: pre-registration *is* the uncertainty gate — it
  forces the human to commit the methodology before seeing data, the §9.3
  defense against the garden of forking paths. The template frames each lock
  contrastively (default value + the §-citation that justifies it).
- EXTERNAL_VERIFIER — 2: the artifacts this module emits are precisely what an
  *external* assessor checks (§9.6, §9.8 checklist items 1-2); the module's job
  is to make that external verification mechanical. It does not verify its own
  output beyond required-field presence.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "PreregError",
    "ASPREDICTED_QUESTION_IDS",
    "aspredicted_template",
    "reforms_checklist",
    "render_preregistration",
]


class PreregError(Exception):
    """Raised when a pre-registration cannot be rendered (e.g. a required answer
    is missing). Carries a structured ``[CODE] message (hint: ...)`` payload per
    the repo's Ship-Gate-B error shape."""


def _fail(code: str, message: str, hint: str) -> PreregError:
    return PreregError(f"[{code}] {message} (hint: {hint})")


# The nine AsPredicted short-form question keys, in form order. Stable IDs so a
# filled ``answers`` dict and a rendered document agree on ordering regardless of
# dict insertion order.
ASPREDICTED_QUESTION_IDS: tuple[str, ...] = (
    "data_collection",     # Q1
    "hypothesis",          # Q2
    "dependent_variable",  # Q3
    "conditions",          # Q4
    "analyses",            # Q5
    "outliers_exclusions", # Q6
    "sample_size",         # Q7
    "other",               # Q8
    "study_type",          # Q9
)


def aspredicted_template() -> dict[str, Any]:
    """Return the AsPredicted 9-question short form as a structured template.

    The return value has a ``questions`` list of exactly nine entries, each with
    ``id`` / ``number`` / ``prompt`` / ``answer`` (pre-filled with crucible's
    locked default where §9.3 fixes it, else an empty string to be completed) /
    ``locks`` (the §9.3 lock this question pins, or ``None``).

    The §9.3 locks — rubric, axes, model list, K seeds, primary test (McNemar's
    exact), correction (BH-FDR) — are seeded into the relevant questions so the
    pre-registration cannot silently disagree with the locked design.
    """
    questions: list[dict[str, Any]] = [
        {
            "id": "data_collection",
            "number": 1,
            "prompt": "Have any data been collected for this study already?",
            "answer": "No. Pre-registration is filed before the first Phase-4 "
            "production run (§9.3). Pilot/Lab-iteration data is excluded from "
            "the locked validation/private_test splits (§9.4 step 1).",
            "locks": None,
        },
        {
            "id": "hypothesis",
            "number": 2,
            "prompt": "What's the main question being asked or hypothesis being "
            "tested in this study?",
            # §9.10 supplies the canonical first-cycle hypothesis.
            "answer": "In an auditing game with an explicit elegance / novelty / "
            "answer-bypass distinction, Claude exhibits a higher novelty-bonus "
            "rate than cross-family peers (Llama, Qwen, Mistral) on seed-catalog "
            "puzzles (§5), as adjudicated by a cross-family judge panel (§9.10).",
            "locks": None,
        },
        {
            "id": "dependent_variable",
            "number": 3,
            "prompt": "Describe the key dependent variable(s) specifying how they "
            "will be measured.",
            "answer": "Primary: per-puzzle hard-gate clear (binary, §8.3 "
            "conjunctive gate) measured by the sealed out-of-band oracle (§10.4). "
            "Secondary: novelty-bonus rate (panel-validated, §8.7); pass^k "
            "consistency (k sibling attempts, §1). Net score (solve + elegance + "
            "novelty − penalties) is a within-passing-region tiebreaker only.",
            "locks": ["rubric", "axes"],
        },
        {
            "id": "conditions",
            "number": 4,
            "prompt": "How many and which conditions will participants be assigned "
            "to?",
            "answer": "Framing arms as the measured condition (§10.1(f)): "
            "neutral / self_referential / social_standings, self_referential "
            "the default. Models under test (the locked model list): "
            "[FILL: claude-opus-4-8-<snapshot>, + cross-family panel "
            "qwen/mistral/command-r exact version strings per SUT.yaml §9.6].",
            "locks": ["model_list", "axes"],
        },
        {
            "id": "analyses",
            "number": 5,
            "prompt": "Specify exactly which analyses you will conduct to examine "
            "the main question/hypothesis.",
            "answer": "Primary test: McNemar's exact test on paired binary "
            "gate-clear outcomes over the same puzzle set (Dror et al. 2018, "
            "ACL P18-1128, §9.3). Confidence intervals: Clopper-Pearson / "
            "Bayesian beta-binomial (Bowyer et al. 2025, arXiv:2503.01747; CLT "
            "inadmissible below ~300 items). Multiple-comparison correction: "
            "Benjamini-Hochberg FDR (1995, JRSS-B 57(1):289-300), Westfall-Young "
            "permutation for axis-correlated tests. Clustered SEs by puzzle "
            "family (Miller 2024, arXiv:2411.00640).",
            "locks": ["primary_test", "correction"],
        },
        {
            "id": "outliers_exclusions",
            "number": 6,
            "prompt": "Any secondary analyses, outlier handling, or data "
            "exclusion rules?",
            "answer": "Findings whose sign flips under judge-prompt paraphrase are "
            "below crucible's resolution and are NOT reported (§9.4 step 4, "
            "JudgeSense Bellibatlu et al. 2026). Attempts terminated by ERROR "
            "(kernel-side `terminated_by`) are excluded; BUDGET/TIME/HARD_KILL "
            "terminations count as non-solves, not exclusions.",
            "locks": None,
        },
        {
            "id": "sample_size",
            "number": 7,
            "prompt": "How many observations will be collected or what will "
            "determine sample size?",
            "answer": "K seeds per (model, puzzle, arm): N≥10 attempts, fixed "
            "temperature (§1 statistical floor; §9.7 publication floor of ≥10 "
            "seeds — Larsen 2025 arXiv:2512.12066 shows 18-28% decision-flip "
            "across seeds). Puzzle count per the catalog-size target locked at "
            "Phase-4 calibration (§6 #2): [FILL: N_puzzles].",
            "locks": ["k_seeds"],
        },
        {
            "id": "other",
            "number": 8,
            "prompt": "Anything else you would like to pre-register? (e.g., "
            "secondary analyses, variables collected for exploratory purposes, "
            "unusual analyses planned?)",
            "answer": "Rubric bundle is content-hashed (RULERS, Hong et al. 2026, "
            "arXiv:2601.08654); the leaderboard records (model_id, score, "
            "bundle_hash) so there is no silent retconning (§9.1). Access tier: "
            "black-box API — capability claims bounded accordingly (§9.6, Taylor "
            "et al. 2025 arXiv:2512.07810). Reproducibility window and tolerance "
            "band (±pp at 95% CI) declared per headline metric (§9.6).",
            "locks": ["rubric"],
        },
        {
            "id": "study_type",
            "number": 9,
            "prompt": "What is the main type of study you are pre-registering?",
            "answer": "Experiment (controlled measurement of frontier-LLM agent "
            "behavior under a sealed-oracle auditing game).",
            "locks": None,
        },
    ]
    return {
        "instrument": "AsPredicted short form (9 questions)",
        "source": "https://aspredicted.org",
        "citation": "Chambers & Tzavella 2022, Nature Human Behaviour 6:29-42, "
        "doi:10.1038/s41562-021-01193-7",
        "locked_surfaces": [
            "rubric",
            "axes",
            "model_list",
            "k_seeds",
            "primary_test",
            "correction",
        ],
        "questions": questions,
    }


def reforms_checklist() -> list[dict[str, Any]]:
    """Return the REFORMS checklist items skeleton (Kapoor et al. 2024).

    REFORMS (Reporting Standards for ML-based Science) is a 32-item consensus
    across study design, data, modeling, and reporting. This returns the item
    skeleton grouped by section; each item has ``id`` / ``section`` / ``item`` /
    ``status`` (default ``"todo"``) / ``note`` (empty) so a maintainer fills it
    in and publishes the completed checklist alongside results (§9.3).

    The items are the load-bearing subset most relevant to an eval instrument
    (leakage, splits, statistical validity, reproducibility) — the full 32 are
    enumerated by section in Kapoor et al. 2024; this skeleton is the practical
    working checklist crucible commits to ``crucible-results``.
    """
    raw: list[tuple[str, str, str]] = [
        # (section, id, item text)
        ("study_design", "SD1",
         "State the population or distribution the claim generalizes to."),
        ("study_design", "SD2",
         "State the concrete scientific claim and the capability aspect probed."),
        ("study_design", "SD3",
         "Pre-register the analysis (link the AsPredicted URL)."),
        ("data", "D1",
         "Describe each split (calibration/dev/validation/private_test) and its provenance."),
        ("data", "D2",
         "Publish a Datasheet per split (Gebru et al. 2018, arXiv:1803.09010)."),
        ("data", "D3",
         "State that train/dev/validation/private_test are disjoint by manifest hash."),
        ("data", "D4",
         "State the contamination/leakage controls (sealed oracle, locked test files, §10.4)."),
        ("modeling", "M1",
         "Report the exact model versions under test (SUT.yaml, §9.6)."),
        ("modeling", "M2",
         "Report the rubric bundle content hash and TUNING.md provenance."),
        ("modeling", "M3",
         "Report the Sobol sensitivity screen (which weights were load-bearing)."),
        ("modeling", "M4",
         "Report the dev-set query budget used of the Thresholdout budget (§9.4 step 3)."),
        ("reporting", "R1",
         "Report the primary test (McNemar exact) and the correction (BH-FDR)."),
        ("reporting", "R2",
         "Report confidence intervals admissible at small N (Clopper-Pearson / Bayesian)."),
        ("reporting", "R3",
         "Report clustered standard errors by puzzle family for cross-model deltas."),
        ("reporting", "R4",
         "Report ≥10 seeds per (model, scenario) as distributions, not point estimates."),
        ("reporting", "R5",
         "Declare the access tier and the resulting bound on capability claims."),
        ("reporting", "R6",
         "Declare the reproducibility window (calendar dates) and per-metric tolerance band."),
        ("reproducibility", "RP1",
         "Pin containers by SHA256 digest; lockfile all language deps."),
        ("reproducibility", "RP2",
         "Publish two repos: crucible-harness + crucible-results."),
        ("reproducibility", "RP3",
         "Provide Inspect-AI-compatible task definitions."),
    ]
    return [
        {
            "id": item_id,
            "section": section,
            "item": text,
            "status": "todo",
            "note": "",
        }
        for (section, item_id, text) in raw
    ]


def _require(answers: dict[str, Any], key: str) -> str:
    value = answers.get(key)
    if value is None or (isinstance(value, str) and not value.strip()):
        raise _fail(
            "INPUT_PREREG_MISSING_ANSWER",
            f"required pre-registration answer '{key}' is missing or empty",
            "pre-fill it from aspredicted_template()['questions'] before rendering; "
            "an incomplete pre-registration is not a lock",
        )
    return str(value)


def render_preregistration(answers: dict[str, Any]) -> str:
    """Render filled ``answers`` to a Markdown pre-registration document.

    ``answers`` maps each of the nine :data:`ASPREDICTED_QUESTION_IDS` to its
    completed text. Missing or empty required answers raise :class:`PreregError`
    (ANDON: a half-locked methodology must not render as if complete). The
    template's pre-filled defaults satisfy these by construction, so the common
    path — ``render_preregistration({q['id']: q['answer'] for q in
    aspredicted_template()['questions']})`` — succeeds and is the intended use.

    The output is deterministic: the same ``answers`` produce the same bytes
    (PIN_PER_STEP), suitable for committing to ``crucible-results`` and pasting
    into the AsPredicted form.
    """
    template = aspredicted_template()
    lines: list[str] = [
        "# Crucible — Pre-registration (AsPredicted short form)",
        "",
        f"_Instrument: {template['instrument']}_  ",
        f"_Source: {template['source']}_  ",
        f"_Citation: {template['citation']}_",
        "",
        "Locked surfaces (§9.3 — no silent change after this point): "
        + ", ".join(f"`{s}`" for s in template["locked_surfaces"])
        + ".",
        "",
    ]
    for q in template["questions"]:
        qid = q["id"]
        answer = _require(answers, qid)
        lines.append(f"## Q{q['number']}. {q['prompt']}")
        if q["locks"]:
            locks = ", ".join(f"`{lk}`" for lk in q["locks"])
            lines.append(f"_Locks: {locks}_")
        lines.append("")
        lines.append(answer.strip())
        lines.append("")
    # Trailing newline for a clean POSIX text file.
    return "\n".join(lines).rstrip() + "\n"
