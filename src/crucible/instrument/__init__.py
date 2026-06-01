"""Instrument-quality scaffolding — the audit chain (research-grounding §9).

Crucible is not just a game; it is a measurement instrument, and §9 specifies the
end-to-end audit chain that makes its results trustworthy to a third party. This
package is the Phase-1 *scaffolding* for that chain (§9.9): it doesn't run a
single puzzle, it makes Phase-4's first diagnostic cycle **audit-ready by
construction**.

The four §9.2 stages, mapped to modules:

* :mod:`crucible.instrument.prereg` — **Stage 1, Pre-registration** (§9.3): the
  AsPredicted 9-question short form, the REFORMS checklist skeleton, and a
  Markdown renderer. Locks the methodology before the production run (the §9.3
  garden-of-forking-paths defense).
* :mod:`crucible.instrument.rubric_bundle` — **Stage 2, the content-hashed
  bundle** (§9.1, §9.4): ``RubricBundle`` + ``compile_bundle`` (sha256) +
  ``bump_on_change`` (no silent retconning — the version changes iff the hash
  changes).
* :mod:`crucible.instrument.tuning` — **Stage 2, the 7-step tuning protocol**
  (§9.4): a real Sobol total-effect screen (``sobol_screen``), a real
  deterministic 60/20/10/10 split (``split_inventory``), and the dev-set query
  budget (``ThresholdoutBudget``). BO search + paraphrase ablation are documented
  stubs with the contract a full implementation must satisfy.
* :mod:`crucible.instrument.sut` — **Stage 4, the SUT record** (§9.6): ``SUT`` +
  ``render_sut_yaml`` — exact version strings, frozen submission→publication.
* :mod:`crucible.instrument.inspect_task` — **Stage 4, Inspect-AI task format**
  (§9.6, "the highest-leverage credibility move"): ``to_inspect_task`` maps a
  crucible puzzle onto the Inspect ``Task`` shape; ``two_repo_layout`` documents
  the ``crucible-harness`` + ``crucible-results`` split.

All cross-module contracts come from :mod:`crucible.types`; this package
implements against them and never redefines them.
"""

from __future__ import annotations

from crucible.instrument.inspect_task import (
    InspectTaskError,
    to_inspect_task,
    two_repo_layout,
)
from crucible.instrument.prereg import (
    ASPREDICTED_QUESTION_IDS,
    PreregError,
    aspredicted_template,
    reforms_checklist,
    render_preregistration,
)
from crucible.instrument.rubric_bundle import (
    RubricBundle,
    RubricBundleError,
    bump_on_change,
    canonical_bundle_json,
    compile_bundle,
)
from crucible.instrument.sut import (
    SUT,
    SUT_FIELDS,
    SUTError,
    parse_sut_yaml,
    render_sut_yaml,
)
from crucible.instrument.tuning import (
    SPLIT_FRACTIONS,
    ThresholdoutBudget,
    TuningBudgetError,
    TuningError,
    bo_search,
    paraphrase_ablate,
    sobol_screen,
    split_inventory,
)

__all__ = [
    # prereg (Stage 1)
    "PreregError",
    "ASPREDICTED_QUESTION_IDS",
    "aspredicted_template",
    "reforms_checklist",
    "render_preregistration",
    # rubric_bundle (Stage 2)
    "RubricBundleError",
    "RubricBundle",
    "canonical_bundle_json",
    "compile_bundle",
    "bump_on_change",
    # tuning (Stage 2)
    "TuningError",
    "TuningBudgetError",
    "SPLIT_FRACTIONS",
    "split_inventory",
    "sobol_screen",
    "ThresholdoutBudget",
    "bo_search",
    "paraphrase_ablate",
    # sut (Stage 4)
    "SUTError",
    "SUT",
    "SUT_FIELDS",
    "render_sut_yaml",
    "parse_sut_yaml",
    # inspect_task (Stage 4)
    "InspectTaskError",
    "to_inspect_task",
    "two_repo_layout",
]
