"""ai-crucible CLI — the unified entry point.

Backs both ``python -m ai_crucible`` and the ``ai-crucible`` console script /
PyInstaller binary that the npm launcher (``@dogfood-lab/ai-crucible``) distributes.
It is a thin dispatcher: the real work lives in the subcommand modules (today, the
judge-admission characterization in :mod:`ai_crucible.characterize.run`), and their
own argparse handles flags — this layer only routes ``argv[0]`` and forwards the rest
verbatim, so ``ai-crucible characterize --k 3`` is exactly ``python -m
ai_crucible.characterize.run --k 3``.
"""

from __future__ import annotations

import contextlib
import re
import sys
from datetime import UTC
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # Type-only imports — kept out of the runtime path so ``--help``/``--version`` stay
    # instant and free of the heavy kernel/model stack (the lazy imports inside the
    # ``run`` handler do the real loading, under main()'s structured-error guard).
    from ai_crucible.cycle import DiagnosticModel
    from ai_crucible.scoring.judge_panel import JudgeFn

# A structured Ship-Gate-B error string: ``[CODE] message (hint: ...)`` — the shape every
# loader/adapter in the repo emits via its ``_fail(code, message, hint)`` helper. We match on
# the rendered string (not the exception class) so the dispatcher stays free of the heavy
# scientific stack those classes live in, and so ANY future module using the same house shape
# is rendered cleanly without a code change here.
_STRUCTURED_ERROR_RE = re.compile(r"^\[[A-Z0-9_]+\] .+ \(hint: .+\)$", re.DOTALL)

#: Default durable-catalog log path (Epic 4). Relative to the CWD so the operator runs from
#: the repo/project root; ``run`` appends here, the ``catalog`` subcommand reads here, both
#: overridable with ``--catalog PATH``. A git-trackable JSONL artifact (the source of truth).
DEFAULT_CATALOG_PATH = "catalog/catalog.jsonl"


def _ensure_utf8_streams() -> None:
    """Make stdout/stderr UTF-8 so the operator-facing banner + run caveat (which carry
    ``ω`` / ``κ`` and other non-ASCII) don't crash on a legacy console.

    On a stock Windows console (cp1252) ``sys.stdout.write`` of a non-ASCII char raises
    ``UnicodeEncodeError`` — so ``ai-crucible --help`` (the banner has ``ω``) would crash
    before printing anything. Best-effort + guarded: a stream without ``reconfigure`` (a
    redirected pipe / a non-``TextIOWrapper``) or one already UTF-8 is left untouched, and
    a reconfigure failure never takes down the CLI. Called once at the top of :func:`main`.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        enc = (getattr(stream, "encoding", "") or "").lower().replace("-", "")
        if reconfigure is not None and enc != "utf8":
            # Best-effort: a reconfigure failure (locked stream, exotic wrapper) must never
            # take down the CLI — the worst case is the legacy-console encoding we started with.
            with contextlib.suppress(Exception):
                reconfigure(encoding="utf-8")


def _version() -> str:
    try:
        return version("ai-crucible")
    except PackageNotFoundError:  # running from a source tree without an install
        return "0.0.0+local"


def _usage() -> str:
    """The usage banner. The version is read from package metadata — the SAME single source
    as ``--version`` (models-cli-004), so the banner never drifts from the installed version
    on the next bump (the old banner hardcoded ``v0.2.0`` while ``--version`` read metadata).
    """
    return f"""\
ai-crucible — a diagnostic measurement instrument (research preview, v{_version()}).

Seats a cross-family panel of local LLM judges under a sealed measurement boundary and
scores attempts against a hidden oracle. NOTE: the judge panel's alt-test ω is still a
circular model-jury bootstrap until a human-labeling round runs; seats are provisional.

usage: ai-crucible <command> [options]

commands:
  characterize   run the judge-admission characterization on the local model panel
                 (needs Ollama + the local panel; forwards all flags — see
                 `ai-crucible characterize --help`)
  run            run one diagnostic cycle: a Solver attempts a puzzle in the sandbox,
                 graded out-of-band against the sealed oracle, emitting the pass^k /
                 Wilson rollup. usage:
                   ai-crucible run <puzzle-dir> --model <id>[@family]
                       [--k N] [--arm neutral|self_referential|social_standings]
                       [--panel <path>]
                 The model adapter is chosen by the optional @family tag: no tag or
                 @claude → Claude (Anthropic API; needs ANTHROPIC_API_KEY); any other
                 @family → an Ollama local model of that family (needs Ollama).
                 Human rollup chrome → STDERR; machine JSON summary → STDOUT.
                 Each run is APPENDED to the durable catalog (--catalog PATH, default
                 {DEFAULT_CATALOG_PATH}; --no-catalog to skip).
  catalog        read + curate the durable catalog (Epic 4 persistence/graduation). usage:
                   ai-crucible catalog list   [--catalog PATH]
                   ai-crucible catalog show   <puzzle-id> [--catalog PATH]
                   ai-crucible catalog graduate [--catalog PATH] [--apply]
                       [--override <puzzle-id> --to <tier> --by <actor>]
                 `list` shows tiers + per-puzzle differential typology + the defer
                 fraction; `graduate` previews (default) or applies (--apply) the
                 Lab→Arena→Regression transitions, or records an attested Designer
                 override out of a DEFER. Human chrome → STDERR; machine JSON → STDOUT.

options:
  --debug, -v    on error, print the full Python traceback (developer mode) instead of the
                 one-line structured error
  -V, --version  print the installed version and exit
  -h, --help     show this message and exit

exit codes:
  0  success (also: --help, --version; `run` completed a cycle and emitted a rollup)
  2  usage error — unknown command, or `run` invoked without a puzzle-dir / --model
  1  (characterize) ran but collected zero judgments — every model failed/unreachable;
     stderr carries a structured [CHARACTERIZE_NO_JUDGMENTS] {{code,message,hint}} JSON. CI
     gates should treat 1 (degraded/empty result) distinctly from 2 (bad invocation).
  1  (run) the puzzle failed to load or stage (a structured [CODE] msg (hint:) error on
     stderr) — distinct from 2 (a bad invocation that never reached the cycle).
"""


def _dispatch(command: str, rest: list[str]) -> int:
    """Route ``command`` to its subcommand. Raises the subcommand's exceptions unwrapped —
    :func:`main` owns the operator-vs-developer error contract around this call."""
    if command == "characterize":
        # Lazy import: keep `--version`/`--help` instant and free of the heavy
        # scientific/inspect-ai stack the characterization run pulls in. The import is
        # INSIDE main()'s guard so a packaging fault here (e.g. a missing scientific dep)
        # is rendered as a clean structured error, not a raw ModuleNotFoundError traceback.
        from ai_crucible.characterize.run import main as characterize_main

        return characterize_main(rest)

    if command == "run":
        return _run_diagnostic_command(rest)

    if command == "catalog":
        return _catalog_command(rest)

    sys.stderr.write(f"ai-crucible: unknown command {command!r}\n\n{_usage()}")
    return 2


def _build_model(model_spec: str) -> DiagnosticModel:
    """Construct a model adapter from a ``<id>[@family]`` spec (the ``--model`` value).

    The optional ``@family`` tag (split on the LAST ``@``, since a model id may itself
    contain ``@``-free ``:`` tags like ``mistral-small:24b``) chooses the adapter and
    feeds the panel's same-family exclusion (§10.2):

    * no ``@family`` tag, or ``@claude`` → :class:`~ai_crucible.models.claude_adapter.ClaudeModel`
      (the default Designer/Solver, Anthropic API — reads ``ANTHROPIC_API_KEY`` at call time);
    * any other ``@family`` → :class:`~ai_crucible.models.ollama_adapter.OllamaModel` of that
      family (a local model served by Ollama).

    Kept a module-level seam (not inlined) so the ``run`` tests inject a CANNED model via
    ``monkeypatch.setattr(cli, "_build_model", ...)`` and never construct a real adapter or
    hit a network/API. Imports the adapters lazily so ``--help``/``--version`` stay free of
    the model stack and a packaging fault is rendered as a structured error by ``main()``.
    """
    if "@" in model_spec:
        model_id, _, family = model_spec.rpartition("@")
    else:
        model_id, family = model_spec, ""

    fam = family.strip().lower()
    if fam and fam != "claude":
        from ai_crucible.models.ollama_adapter import OllamaModel

        return OllamaModel(model_id=model_id, family=fam)

    from ai_crucible.models.claude_adapter import ClaudeModel

    return ClaudeModel(model_id)


def _run_diagnostic_command(rest: list[str]) -> int:
    """Handle ``ai-crucible run <puzzle-dir> --model <id>[@family] [--k N] [--arm ...]
    [--panel <path>]``.

    Parses the args, builds the model adapter (:func:`_build_model`), runs one
    :func:`ai_crucible.cycle.run_diagnostic` cycle via ``asyncio.run``, writes the human
    rollup chrome to STDERR and the machine JSON summary to STDOUT, and returns 0 on a
    completed run. A bad invocation (missing puzzle-dir / ``--model``, unknown ``--arm``)
    returns 2 with a usage message; a load/stage failure raises a structured
    ``[CODE] msg (hint:)`` error that ``main()``'s top-level handler renders as one line
    (exit 1). Heavy imports (the kernel/cycle stack) are lazy so they live inside ``main``'s
    guard and a packaging fault renders cleanly.
    """
    import argparse
    import asyncio

    from ai_crucible.types import FramingArm

    parser = argparse.ArgumentParser(
        prog="ai-crucible run",
        description="Run one diagnostic cycle against a puzzle and emit the pass^k rollup.",
        add_help=True,
    )
    parser.add_argument("puzzle_dir", help="the puzzle directory (the one containing meta.json)")
    parser.add_argument(
        "--model",
        required=True,
        metavar="<id>[@family]",
        help="model spec; @family chooses the adapter (none/@claude → Claude, else Ollama)",
    )
    parser.add_argument("--k", type=int, default=1, help="sibling attempts for pass^k (>=1)")
    parser.add_argument(
        "--arm",
        choices=[a.value for a in FramingArm],
        default=FramingArm.SELF_REFERENTIAL.value,
        help="framing arm for the scored context (default: self_referential)",
    )
    parser.add_argument(
        "--panel",
        type=Path,
        default=None,
        help="optional composed seated-panel artifact (panel.json) for cross-family novelty",
    )
    parser.add_argument(
        "--catalog",
        type=Path,
        default=Path(DEFAULT_CATALOG_PATH),
        help=f"catalog log to append this run to (default: {DEFAULT_CATALOG_PATH})",
    )
    parser.add_argument(
        "--no-catalog",
        action="store_true",
        help="do not append this run to the durable catalog (Epic 4 persistence off)",
    )
    # argparse exits(2) on a parse error and prints usage to stderr — exactly the
    # bad-invocation contract (exit 2, distinct from a load/stage failure's exit 1). It
    # signals that exit by RAISING SystemExit; main()'s top-level handler only catches
    # Exception (not the BaseException SystemExit), so we translate it to a RETURN code here
    # — keeping `run` a plain `main(...) -> int` like every other subcommand (and -h → 0).
    try:
        args = parser.parse_args(rest)
    except SystemExit as exc:
        return int(exc.code) if isinstance(exc.code, int) else 2

    if args.k < 1:
        sys.stderr.write("ai-crucible run: --k must be >= 1\n")
        return 2

    # Lazy: keep the heavy kernel/cycle stack out of the --help/--version path.
    import uuid
    from datetime import datetime

    from ai_crucible.cycle import render_rollup, rollup_json, run_diagnostic

    # Resolve the panel ONCE: the JudgePanel drives the run's novelty adjudication, and the
    # SeatedPanel's quorum/escalate posture distils into the catalog PanelSignal (§A fairness).
    panel, panel_signal = _load_panel_and_signal(args.panel)
    model = _build_model(args.model)
    arm = FramingArm(args.arm)
    family = _family_of(args.model)

    started_at = datetime.now(UTC).isoformat()
    history = asyncio.run(
        run_diagnostic(Path(args.puzzle_dir), model, args.k, arm=arm, panel=panel)
    )
    finished_at = datetime.now(UTC).isoformat()

    # Stage-C honesty: human chrome → STDERR, machine JSON summary → STDOUT.
    sys.stderr.write(render_rollup(history, args.k) + "\n")

    # Epic-4 persistence: append this run to the durable catalog (the source-of-truth log
    # accumulating per-(puzzle,model) history across runs). Opt-out with --no-catalog.
    if not args.no_catalog:
        _append_run_to_catalog(
            args.catalog,
            history=history,
            puzzle_dir=Path(args.puzzle_dir),
            model=model,
            family=family,
            k=args.k,
            arm=args.arm,
            started_at=started_at,
            finished_at=finished_at,
            nonce=uuid.uuid4().hex[:12],
            panel_signal=panel_signal,
        )

    sys.stdout.write(rollup_json(history, args.k) + "\n")
    return 0


def _family_of(model_spec: str) -> str:
    """The model family of a ``<id>[@family]`` spec (the catalog's cross-family axis).

    No tag or ``@claude`` → ``"claude"`` (the generator family); any other ``@family`` →
    that family (e.g. ``qwen3-coder:480b@qwen`` → ``"qwen"``). Mirrors :func:`_build_model`'s
    split (on the LAST ``@`` so a model id's own ``:`` tags are untouched).
    """
    if "@" in model_spec:
        _, _, fam = model_spec.rpartition("@")
        fam = fam.strip().lower()
        return fam or "claude"
    return "claude"


def _append_run_to_catalog(
    catalog_path: Path,
    *,
    history: object,
    puzzle_dir: Path,
    model: DiagnosticModel,
    family: str,
    k: int,
    arm: str,
    started_at: str,
    finished_at: str,
    nonce: str,
    panel_signal: object,
) -> None:
    """Append one RunRecord to the durable catalog (Epic-4 persistence; CONTRACT §D).

    Best-effort + loud: a diagnostic run already SUCCEEDED by the time we persist, so a
    catalog write failure must not fail the run — it is reported to STDERR (operator
    chrome) but the rollup still stands. Reads the puzzle's ``min_k`` from its meta so the
    catalog stays self-contained for graduation. Imports are lazy to keep ``--help`` free
    of the catalog stack.
    """
    from ai_crucible.catalog import CatalogStore, RuleConfig, build_run_record
    from ai_crucible.puzzle import load_puzzle

    try:
        min_k = load_puzzle(puzzle_dir).meta.min_k
        record = build_run_record(
            history,  # type: ignore[arg-type]
            puzzle_dir=puzzle_dir,
            model_id=model.model_id,
            family=family,
            k=k,
            min_k=min_k,
            arm=arm,
            started_at=started_at,
            finished_at=finished_at,
            nonce=nonce,
            rule_version=RuleConfig().rule_version,
            role="solver" if family == "claude" else "cohort_solver",
            panel=panel_signal,  # type: ignore[arg-type]
        )
        store = CatalogStore(catalog_path)
        store.record_run(record)
        sys.stderr.write(
            f"catalog: recorded run {record.run_id} for {record.puzzle_id!r} "
            f"({record.successes}/{record.n}, family={family}) → {catalog_path}\n"
        )
    except Exception as exc:  # noqa: BLE001 — persistence must never fail a completed run.
        sys.stderr.write(
            f"catalog: WARNING — could not persist this run to {catalog_path}: {exc} "
            "(the diagnostic rollup above still stands; re-run or fix the catalog path)\n"
        )


def _load_panel_and_signal(path: Path | None):
    """Load a seated-panel artifact into BOTH a run :class:`JudgePanel` and the catalog
    :class:`~ai_crucible.catalog.types.PanelSignal` (``run --panel``).

    ``None`` (no ``--panel``) → ``(None, PanelSignal(present=False))`` — no cross-family
    fairness signal, so graduation will DEFER (the honest no-panel posture). Otherwise it
    reads the committed artifact ONCE (:func:`ai_crucible.characterize.panel_store.load_panel`),
    seats it via :meth:`JudgePanel.from_seated` for the run's novelty adjudication, and
    distils the SeatedPanel's quorum/escalate posture into the catalog PanelSignal
    (``fairness`` stays ``None`` — no confident cross-family fairness judge exists yet, ω
    on ice; CONTRACT §A). A missing/malformed artifact raises the panel store's structured
    ``[CODE] msg (hint:)`` error, rendered as one line by ``main()`` (exit 1).
    """
    from ai_crucible.catalog import panel_signal_from_seated

    if path is None:
        return (None, panel_signal_from_seated(None))

    from ai_crucible.characterize.panel_store import load_panel
    from ai_crucible.models.ollama_adapter import OllamaModel
    from ai_crucible.scoring.judge_panel import JudgePanel

    seated = load_panel(path)

    def judge_for(model_id: str) -> JudgeFn:
        # The seat carries the family; instantiate a local Ollama judge for it. The
        # family is re-bound by from_seated from the seat record, so a placeholder here
        # is fine — the panel reads the seat's family for exclusion.
        return OllamaModel(model_id=model_id, family="").as_judge()

    panel = JudgePanel.from_seated(seated, judge_for)
    return (panel, panel_signal_from_seated(seated))


def _catalog_command(rest: list[str]) -> int:
    """Handle ``ai-crucible catalog <list|show|graduate> [options]`` (Epic 4).

    The read + lifecycle surface over the durable catalog. ``list`` shows the tier
    distribution + per-puzzle typology + the DEFER-fraction health metric; ``show
    <puzzle_id>`` details one puzzle's runs/transitions/differential + what the lifecycle
    would decide now; ``graduate`` previews (default) or APPLIES (``--apply``) the
    Lab→Arena→Regression transitions, or records an attested Designer override
    (``--override``). Stage-C honesty throughout: human chrome → STDERR, machine JSON →
    STDOUT. Heavy imports are lazy (keep ``--help`` light); structured errors propagate to
    ``main()``'s operator/dev handler.
    """
    import argparse

    parser = argparse.ArgumentParser(prog="ai-crucible catalog", add_help=True)
    sub = parser.add_subparsers(dest="action")

    def _add_catalog_arg(p: argparse.ArgumentParser) -> None:
        p.add_argument(
            "--catalog", type=Path, default=Path(DEFAULT_CATALOG_PATH),
            help=f"catalog log path (default: {DEFAULT_CATALOG_PATH})",
        )

    p_list = sub.add_parser("list", help="tier distribution + per-puzzle typology + health")
    _add_catalog_arg(p_list)

    p_show = sub.add_parser("show", help="detail one puzzle (runs / transitions / differential)")
    p_show.add_argument("puzzle_id", help="the puzzle id to detail")
    _add_catalog_arg(p_show)

    p_grad = sub.add_parser("graduate", help="preview (default) or apply tier transitions")
    _add_catalog_arg(p_grad)
    p_grad.add_argument(
        "--apply", action="store_true",
        help="APPLY the transitions (default is a dry-run preview that mutates nothing)",
    )
    p_grad.add_argument(
        "--override", metavar="PUZZLE_ID",
        help="record an attested Designer override moving PUZZLE_ID out of a DEFER (manual)",
    )
    p_grad.add_argument(
        "--to", choices=["lab", "arena", "regression"],
        help="target tier for --override",
    )
    p_grad.add_argument(
        "--by", default="designer", help="attested actor for --override (e.g. designer:mike)",
    )

    try:
        args = parser.parse_args(rest)
    except SystemExit as exc:
        return int(exc.code) if isinstance(exc.code, int) else 2

    if args.action is None:
        sys.stderr.write("ai-crucible catalog: a subcommand is required "
                         "(list | show | graduate)\n")
        return 2

    if args.action == "list":
        return _catalog_list(args.catalog)
    if args.action == "show":
        return _catalog_show(args.catalog, args.puzzle_id)
    if args.action == "graduate":
        return _catalog_graduate(args.catalog, args)
    sys.stderr.write(f"ai-crucible catalog: unknown subcommand {args.action!r}\n")
    return 2


def _catalog_views(catalog_path: Path):
    """Load the store + fold the shared views the catalog subcommands render.

    Returns ``(store, tiers, aggregates, classifications, summary)``. Pure reads; the
    default frontier predicate + the catalog-recovered ``min_k`` drive the fold (so the
    saturation floor is computable from the log alone).
    """
    from ai_crucible.catalog import (
        CatalogStore,
        RuleConfig,
        classify_catalog,
        default_frontier_fn,
        min_k_map,
    )

    store = CatalogStore(catalog_path)
    rule = RuleConfig()
    tiers = store.current_tiers()
    aggregates = store.aggregate(is_frontier=default_frontier_fn, min_k_for=min_k_map(store))
    classifications = classify_catalog(aggregates, rule)
    summary = store.catalog_summary()
    return store, tiers, aggregates, classifications, summary


def _catalog_list(catalog_path: Path) -> int:
    """``catalog list`` — tier counts + per-puzzle typology + the DEFER-fraction health."""
    import json as _json

    if not catalog_path.exists():
        sys.stderr.write(f"catalog: no catalog at {catalog_path} yet "
                         "(run `ai-crucible run <puzzle> --model ...` to populate it)\n")
        sys.stdout.write('{"tiers": {}, "total_puzzles": 0, "puzzles": []}\n')
        return 0

    _store, tiers, aggregates, classifications, summary = _catalog_views(catalog_path)

    rows = []
    for pid in tiers:
        agg = aggregates.get(pid)
        cls = classifications.get(pid)
        rows.append({
            "puzzle_id": pid,
            "tier": tiers[pid].value,
            "claude": f"{agg.claude_successes}/{agg.claude_n}" if agg else "0/0",
            "cohort": f"{agg.cohort_successes}/{agg.cohort_n}" if agg else "0/0",
            "typology": cls.typology.value if cls else "n/a",
            "delta": round(cls.delta, 3) if cls else None,
        })

    # Human chrome → STDERR.
    lines = ["AI Crucible catalog"]
    t = summary["tiers"]
    lines.append(
        f"  tiers: lab={t.get('lab', 0)} arena={t.get('arena', 0)} "
        f"regression={t.get('regression', 0)}  ·  {summary['total_puzzles']} puzzles"
    )
    lines.append(
        f"  defer_fraction={summary['defer_fraction']:.2f} "
        "(rising ⇒ recruit a 3rd disjoint cross-family seat, NOT relax the threshold) "
        f"·  chain_verified={summary['chain_verified']}"
    )
    lines.append("")
    for r in rows:
        lines.append(
            f"  [{r['tier']:<10}] {r['puzzle_id']:<28} "
            f"claude {r['claude']:>6}  cohort {r['cohort']:>6}  → {r['typology']}"
        )
    sys.stderr.write("\n".join(lines) + "\n")

    sys.stdout.write(_json.dumps(
        {"tiers": summary["tiers"], "total_puzzles": summary["total_puzzles"],
         "defer_fraction": summary["defer_fraction"],
         "chain_verified": summary["chain_verified"], "puzzles": rows},
        default=str) + "\n")
    return 0


def _catalog_show(catalog_path: Path, puzzle_id: str) -> int:
    """``catalog show <puzzle_id>`` — runs + transition timeline + differential + verdict."""
    import json as _json

    if not catalog_path.exists():
        sys.stderr.write(f"catalog: no catalog at {catalog_path} yet\n")
        return 1

    store, tiers, aggregates, classifications, _summary = _catalog_views(catalog_path)
    if puzzle_id not in tiers:
        sys.stderr.write(f"catalog: no puzzle {puzzle_id!r} in {catalog_path}\n")
        return 1

    from ai_crucible.catalog import RuleConfig, promote_decision

    agg = aggregates[puzzle_id]
    cls = classifications.get(puzzle_id)
    runs = [r for r in store.read_runs() if r.puzzle_id == puzzle_id]
    transitions = [t for t in store.read_transitions() if t.puzzle_id == puzzle_id]
    verdict = promote_decision(agg, RuleConfig())

    lines = [f"AI Crucible catalog — puzzle {puzzle_id!r}  (tier: {tiers[puzzle_id].value})"]
    lines.append(f"  runs: {len(runs)}  ·  claude {agg.claude_successes}/{agg.claude_n}  "
                 f"·  cohort {agg.cohort_successes}/{agg.cohort_n}")
    if cls is not None:
        lo, hi = cls.delta_ci
        lines.append(f"  differential: {cls.typology.value}  "
                     f"(delta={cls.delta:+.3f}, 95% CI [{lo:+.3f}, {hi:+.3f}], "
                     f"p={cls.p_value:.3f}, bh_survived={cls.bh_survived})")
    lines.append(f"  would-graduate-now: {verdict.decision.value}  "
                 f"(claude_band={verdict.claude_band_ok}, cohort_nontrivial="
                 f"{verdict.cohort_nontrivial}, fairness_ok={verdict.fairness_ok}, "
                 f"deferred={verdict.deferred})")
    if transitions:
        lines.append("  timeline:")
        for tr in transitions:
            lines.append(f"    {tr.recorded_at}  {tr.from_tier.value}→{tr.to_tier.value}  "
                         f"{tr.reason_code.value}  (by {tr.decided_by})")
    sys.stderr.write("\n".join(lines) + "\n")

    sys.stdout.write(_json.dumps({
        "puzzle_id": puzzle_id,
        "tier": tiers[puzzle_id].value,
        "claude": {"successes": agg.claude_successes, "n": agg.claude_n},
        "cohort": {"successes": agg.cohort_successes, "n": agg.cohort_n},
        "differential": {
            "typology": cls.typology.value, "delta": cls.delta,
            "delta_ci": list(cls.delta_ci), "p_value": cls.p_value,
            "bh_survived": cls.bh_survived,
        } if cls else None,
        "would_graduate_now": verdict.decision.value,
        "transitions": [t.to_payload() for t in transitions],
        "n_runs": len(runs),
    }, default=str) + "\n")
    return 0


def _catalog_graduate(catalog_path: Path, args) -> int:
    """``catalog graduate`` — preview (default), ``--apply``, or ``--override``."""
    import json as _json
    import shutil
    import tempfile
    from datetime import datetime

    from ai_crucible.catalog import CatalogStore, RuleConfig, apply_lifecycle
    from ai_crucible.catalog.types import TransitionReason, TransitionRecord
    from ai_crucible.types import CatalogTier

    now = datetime.now(UTC).isoformat()

    # --- attested Designer override (a real mutation; promotes out of a DEFER) ------- #
    if args.override is not None:
        if args.to is None:
            sys.stderr.write("ai-crucible catalog graduate --override needs --to <tier>\n")
            return 2
        if not catalog_path.exists():
            sys.stderr.write(f"catalog: no catalog at {catalog_path} yet\n")
            return 1
        store, tiers, aggregates, _cls, _s = _catalog_views(catalog_path)
        if args.override not in tiers:
            sys.stderr.write(f"catalog: no puzzle {args.override!r} in {catalog_path}\n")
            return 1
        agg = aggregates[args.override]
        tr = TransitionRecord(
            puzzle_id=args.override,
            from_tier=tiers[args.override],
            to_tier=CatalogTier(args.to),
            reason_code=TransitionReason.MANUAL,
            recorded_at=now,
            decided_by=args.by,
            contributing_run_ids=list(agg.contributing_run_ids),
            evidence={"override": True, "actor": args.by},
            rule_version=RuleConfig().rule_version,
        )
        appended = store.record_transition(tr)
        verb = "recorded" if appended is not None else "already present (idempotent)"
        sys.stderr.write(
            f"catalog: override {verb} — {args.override} "
            f"{tr.from_tier.value}→{tr.to_tier.value} (by {args.by})\n"
        )
        sys.stdout.write(_json.dumps(tr.to_payload(), default=str) + "\n")
        return 0

    if not catalog_path.exists():
        sys.stderr.write(f"catalog: no catalog at {catalog_path} yet\n")
        sys.stdout.write('{"applied": false, "transitions": []}\n')
        return 0

    if args.apply:
        store = CatalogStore(catalog_path)
        transitions = apply_lifecycle(store, now=now)
        applied = True
    else:
        # Dry-run: preview against a TEMP COPY of the log so nothing is mutated.
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp) / "catalog.jsonl"
            shutil.copyfile(catalog_path, tmp_path)
            store = CatalogStore(tmp_path)
            transitions = apply_lifecycle(store, now=now)
        applied = False

    if transitions:
        head = "APPLIED" if applied else "PROPOSED (dry-run — nothing written; pass --apply)"
        lines = [f"catalog graduate — {head}:"]
        for t in transitions:
            lines.append(f"  {t.puzzle_id:<28} {t.from_tier.value}→{t.to_tier.value}  "
                         f"{t.reason_code.value}")
        sys.stderr.write("\n".join(lines) + "\n")
    else:
        sys.stderr.write(
            "catalog graduate: no transitions — every puzzle holds or DEFERs to the "
            "Designer (the honest posture while no confident cross-family fairness "
            "verdict exists; ω on ice). See `catalog list` for the defer_fraction.\n"
        )
    sys.stdout.write(_json.dumps(
        {"applied": applied, "transitions": [t.to_payload() for t in transitions]},
        default=str) + "\n")
    return 0


def main(argv: list[str] | None = None) -> int:
    """Dispatch ``argv`` to a subcommand. Returns a process exit code.

    Owns the operator-facing error contract (cli-operator-001 / error-hint-sweep-001): an
    exception escaping the dispatch is rendered as a SINGLE structured stderr line, never a raw
    multi-frame traceback. If the exception already carries the repo's ``[CODE] msg (hint: ...)``
    house shape (CalibrationLoadError, OllamaUnreachableError, PuzzleLoadError, …) that one line
    is written verbatim; otherwise it is wrapped in a generic ``[CLI_UNEXPECTED] …`` line that
    points to ``--debug``. The full traceback is opt-IN via ``--debug``/``-v`` (developer mode) —
    mirroring the kernel, where SealedBoundaryViolation/ChromeAccessError propagate unwrapped for
    the same operator-vs-developer reason. KeyboardInterrupt is a clean abort (exit 130).
    """
    _ensure_utf8_streams()  # banner/caveat carry non-ASCII (ω/κ) — survive a cp1252 console.
    argv = list(sys.argv[1:] if argv is None else argv)

    # Top-level --debug/-v is consumed HERE (before the rest is forwarded) so a subcommand's own
    # parser never sees it. It only changes how an error is rendered, not what runs.
    debug = False
    filtered: list[str] = []
    for tok in argv:
        if tok in ("--debug", "-v"):
            debug = True
        else:
            filtered.append(tok)
    argv = filtered

    if not argv or argv[0] in ("-h", "--help"):
        sys.stdout.write(_usage())
        return 0
    if argv[0] in ("-V", "--version"):
        sys.stdout.write(f"ai-crucible {_version()}\n")
        return 0

    command, rest = argv[0], argv[1:]
    try:
        return _dispatch(command, rest)
    except KeyboardInterrupt:
        # Ctrl-C is a deliberate operator abort, not a crash. 128 + SIGINT(2) = 130.
        sys.stderr.write("\nai-crucible: interrupted\n")
        return 130
    except Exception as exc:  # noqa: BLE001 — the operator-facing top-level handler.
        if debug:
            raise  # developer mode: let the interpreter print the full traceback.
        msg = str(exc)
        if _STRUCTURED_ERROR_RE.match(msg):
            # Already in the house shape — emit the one authored line, no stack chrome.
            sys.stderr.write(msg + "\n")
        else:
            # An unexpected/unstructured fault (packaging, import, programming bug): wrap it in
            # the house shape so the operator gets a code + an actionable next step.
            sys.stderr.write(
                f"[CLI_UNEXPECTED] {msg} (hint: re-run with --debug for the full traceback)\n"
            )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
