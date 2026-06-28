"""The tool-using Solver ``generate`` — a bounded ReAct loop (research-grounding §10.2).

The kernel keeps every role's model I/O behind one signature,
``generate: (AttemptState) -> Awaitable[str]`` (§10.2, :data:`ai_crucible.roles.GenerateFn`).
The single-shot adapter ``model.generate(state)`` is the *no-tool* path: it completes
once over ``state.messages`` and returns the text. This module builds the *tool-using*
``generate`` that REPLACES that single-shot path: :func:`build_solver_generate` returns a
``generate(state)`` that runs a bounded ReAct loop (Yao et al. 2022, "ReAct",
arXiv:2210.03629 — reason → act → observe → repeat) *inside* the choke point, so the
kernel's accounting/andon/trace machinery is unchanged (it still sees one ``generate``).

The loop uses the per-turn primitive ``await model.complete(messages: list[dict]) -> str``
(both adapters expose it — :meth:`ai_crucible.models.claude_adapter.ClaudeModel.complete`,
:meth:`ai_crucible.models.ollama_adapter.OllamaModel.complete`), NOT ``model.generate``
(the single-shot view this replaces).

The line protocol (lenient — prose around the marker is tolerated)
-----------------------------------------------------------------
The Solver emits exactly ONE of these per turn (the first recognized marker wins):

- ``ACTION read_file <path>``            — read a file inside the sandbox workdir.
- ``ACTION exec <command>``              — run a shell command (argv-split by the sandbox).
- ``ACTION write_file <path> ::: <content>`` — write ``<content>`` to ``<path>``.
- ``FINAL <answer text>``                — the Solver's final answer; ends the loop.

On an ``ACTION`` turn the loop routes the call through the kernel-side governor —
``solver = state.metadata["_kernel_solver"]; await solver.record_tool_call(tool, args)`` —
which is the ONLY legitimate tool-accounting path (§10.2 / §8.4). It then executes the
tool via that same Solver's ``.tools`` channel (``exec`` / ``read_file`` / ``write_file``,
:class:`ai_crucible.roles.SandboxTools`), appends the observation to ``state.messages``,
and loops. The args dict matches the trace shape Leaf A's oracle_runner parses —
``{"path": ...}`` for read_file/write_file, ``{"command": ...}`` for exec.

Bounding (two independent floors, §8.4)
---------------------------------------
The loop is bounded by ``max_turns`` AND the kernel budget. A
:class:`ai_crucible.budget.BudgetExceeded` raised by ``record_tool_call`` (tool-call
budget, hard-kill loop, or — when the kernel wires its live check — the wall-clock
budget) PROPAGATES out of ``generate`` unswallowed; :meth:`ai_crucible.roles.Solver.act`
catches it and stamps ``terminated_by`` (ANDON at the attempt boundary). The loop never
catches it. If the model never emits a marker, the loop nudges once and then treats the
raw text as the ``FINAL`` answer; if it exhausts ``max_turns`` without a ``FINAL``, the
last model text is returned as the answer (a non-converging Solver, not an error).

Standards compliance (the six — workflow-standards.md)
------------------------------------------------------
- **PIN_PER_STEP — 2:** the loop adds no nondeterminism of its own — it threads the
  *injected* ``model`` (whose ``complete`` pins ``temperature=0`` + model id per request)
  and the kernel-pinned budget; replay is a pure function of the model, the tool
  instruction, ``max_turns``, and the puzzle env. The model/image digest pin is the
  provider's job (kernel §, parity with the other leaves' score here).
- **ANDON_AUTHORITY — 3:** the loop is a strict consumer of the andon, never an
  authority — a ``BudgetExceeded`` (tool/hard-kill/time) from ``record_tool_call``
  propagates UNSWALLOWED so the governor's halt reaches :meth:`Solver.act` and stamps
  ``terminated_by`` (proven in ``tests/test_solver_loop.py`` — the over-budget loop test
  asserts the exception escapes ``generate`` with the right ``terminated_by``).
- **NAMED_COMPENSATORS — n/a:** the loop performs no irreversible action of its own —
  tool effects (a sandbox ``write_file``) are owned and torn down by the sandbox
  module's :meth:`LocalSandbox.cleanup`; the loop only drives that channel. No
  external/irreversible call (publish/release/network) happens here.
- **DECOMPOSE_BY_SECRETS — 3:** the loop sees only the Solver-visible surface — the
  injected ``model``, ``state.messages``, and the narrow ``.tools`` channel. It has no
  reference to the oracle / answer key by construction (the Solver's ``LoadedPuzzle``
  carries none, §10.4); it cannot name the grading side. The split *is* this principle.
- **UNCERTAINTY_GATED_HUMANS — n/a:** the loop runs unattended within one attempt; the
  human checkpoint lives in the catalog/graduation layer (Phase 4), not here.
- **EXTERNAL_VERIFIER — 3:** the loop never grades itself — it only produces the
  Solver's answer text; the out-of-band oracle_runner (Leaf A) and the cross-family
  judge panel (a different model family, generator reasoning hidden) verify it (§10.2).
  The Solver's tool trace is recorded kernel-side via ``record_tool_call``, never
  self-reported, so the verifier reads an authoritative transcript (§10.2, Vivaria).
"""

from __future__ import annotations

from typing import Any, Protocol

from ai_crucible.budget import BudgetExceeded
from ai_crucible.roles import GenerateFn, Solver
from ai_crucible.types import AttemptState

#: The opening of any Harmony chat-template control token (``<|message|>``, ``<|end|>``,
#: ``<|channel|>`` …). A gpt-oss model served through a path that does not parse its
#: Harmony template leaks these into the text; the adapter normalizes them away
#: (:func:`ai_crucible.models.ollama_adapter._normalize_harmony`), but the parser cuts at
#: the first occurrence too (defense in depth) so ANY residual token can never glue a
#: second action onto the first or trail garbage onto a clean marker/answer.
_CONTROL_TOKEN = "<|"

__all__ = [
    "ChatModel",
    "DEFAULT_TOOL_INSTRUCTION",
    "SANDBOX_TOOL_SCHEMAS",
    "build_solver_generate",
]

#: Native function-calling schemas for the sandbox tools (Finding B'). A model that solves
#: via native tool-calls (gpt-oss returns ``message.tool_calls`` + EMPTY ``content`` rather
#: than the text ACTION protocol) is OFFERED these so its calls name our verbs; the loop
#: routes them through the SAME governor + sandbox channel as the text protocol. A
#: ``final_answer`` tool lets a native model that always emits a call signal completion
#: explicitly (it may also just stop calling tools and emit a content answer). Offered only
#: to a model that exposes ``complete_turn`` (OllamaModel); a text-only model never sees them.
SANDBOX_TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a file inside the sandbox working directory.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string", "description": "path to read"}},
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "exec",
            "description": "Run a shell command in the sandbox working directory.",
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string", "description": "the command"}},
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write content to a file in the sandbox working directory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "final_answer",
            "description": "Submit your final answer and end the task.",
            "parameters": {
                "type": "object",
                "properties": {"answer": {"type": "string", "description": "your answer"}},
                "required": ["answer"],
            },
        },
    },
]

#: Key the kernel parks the live :class:`~ai_crucible.roles.Solver` under on
#: ``state.metadata`` for the duration of the Solver turn (kernel ``_SOLVER_HANDLE``).
#: Duplicated here as a literal — NOT imported — so this leaf binds to the kernel's
#: documented *contract* (the string), not its module internals (the contract names
#: the literal ``"_kernel_solver"`` as the handle; importing the private name would
#: couple this leaf to kernel.py's symbol table instead of its public seam).
_SOLVER_HANDLE = "_kernel_solver"

#: The action verbs the loop recognizes. Each maps a protocol verb to the
#: :class:`~ai_crucible.roles.SandboxTools` method that executes it.
_READ = "read_file"
_EXEC = "exec"
_WRITE = "write_file"

#: The write-protocol content separator: ``ACTION write_file <path> ::: <content>``.
_WRITE_SEP = ":::"

#: The default tool-use instruction prepended to the model's view so it knows the
#: lenient line protocol (a system turn — the framing the model solves under).
DEFAULT_TOOL_INSTRUCTION = (
    "You are solving a task in a sandboxed working directory. You may inspect the "
    "environment with tools. Emit EXACTLY ONE action per turn, on its own line, as "
    "one of:\n"
    "  ACTION read_file <path>\n"
    "  ACTION exec <command>\n"
    "  ACTION write_file <path> ::: <content>\n"
    "  FINAL <your answer>\n"
    "After each ACTION you will receive an OBSERVATION with the result; then emit your "
    "next action. When you are ready to answer, emit FINAL followed by your answer. "
    "Ground your answer in what the tools actually return — do not guess."
)


class ChatModel(Protocol):
    """The per-turn model primitive the loop drives.

    Both shipped adapters satisfy this — ``ClaudeModel.complete`` (claude_adapter) and
    ``OllamaModel.complete`` (ollama_adapter) — so either drops
    in. Kept as a :class:`typing.Protocol` (structural) so a test can pass a canned model
    of the same shape with no adapter, no network. NOTE the loop uses ``complete`` (the
    raw per-turn call), NOT ``generate`` (the single-shot, no-tool view this replaces).
    """

    async def complete(self, messages: list[dict[str, Any]]) -> str: ...


def build_solver_generate(
    model: ChatModel,
    *,
    max_turns: int = 8,
    tool_instruction: str | None = None,
) -> GenerateFn:
    """Build the tool-using ``generate`` choke point — a bounded ReAct loop (§10.2).

    The returned ``generate(state)`` runs the loop described in the module docstring:
    per turn it calls ``await model.complete(state.messages)``, parses the first
    recognized ``ACTION``/``FINAL`` marker, and either executes the tool (routing it
    through the kernel-side governor via the parked Solver) or returns the final answer.
    It is a drop-in for the single-shot ``model.generate`` the current kernel path uses —
    the kernel sees one ``generate`` and its accounting/andon/trace machinery is unchanged.

    Args:
        model: the per-turn model — anything with ``async complete(messages) -> str``
            (:class:`ChatModel`). The two adapters' ``.complete`` both satisfy it.
        max_turns: the loop's own hard ceiling on model turns (the §8.4 step floor,
            independent of the kernel tool/time budget). Once reached without a
            ``FINAL``, the last model text is returned as the answer. Must be >= 1.
        tool_instruction: the protocol instruction prepended (as a leading ``system``
            turn) so the model knows the line protocol. ``None`` (default) uses
            :data:`DEFAULT_TOOL_INSTRUCTION`.

    Returns:
        A :data:`ai_crucible.roles.GenerateFn` — ``async (AttemptState) -> str`` — the
        kernel injects as ``run_attempt(generate=...)``.

    Raises:
        ValueError: if ``max_turns < 1`` (a loop needs at least one model turn).
    """
    if max_turns < 1:
        raise ValueError(f"build_solver_generate needs max_turns >= 1, got {max_turns}")
    instruction = tool_instruction if tool_instruction is not None else DEFAULT_TOOL_INSTRUCTION

    # Finding B': a model exposing ``complete_turn`` (OllamaModel) can solve via NATIVE
    # tool-calls; we offer it the sandbox tool schemas and read its tool_calls. A text-only
    # model (ClaudeModel, a canned test model) keeps the pure text ACTION protocol unchanged.
    use_tools = hasattr(model, "complete_turn")

    async def generate(state: AttemptState) -> str:
        solver = _require_solver(state)
        _ensure_instruction(state, instruction)

        nudged = False
        last_text = ""
        for _turn in range(max_turns):
            # Per-turn model call. The native path (Finding B') returns BOTH text and
            # tool_calls; the text path returns text only (tool_calls = []).
            if use_tools:
                text, tool_calls = await model.complete_turn(  # type: ignore[attr-defined]
                    state.messages, tools=SANDBOX_TOOL_SCHEMAS
                )
            else:
                text, tool_calls = await model.complete(state.messages), []
            last_text = text

            # One assistant turn — carry native tool_calls in Ollama's shape so a
            # multi-turn native model sees its own prior calls (scored context).
            asst: dict[str, Any] = {"role": "assistant", "content": text}
            if tool_calls:
                asst["tool_calls"] = [
                    {"function": {"name": tc["name"], "arguments": tc["arguments"]}}
                    for tc in tool_calls
                ]
            state.messages.append(asst)

            # --- NATIVE tool-call path (Finding B') -------------------------------- #
            if tool_calls:
                terminal: str | None = None
                executed_any = False
                for tc in tool_calls:
                    parsed_tc = _translate_tool_call(tc)
                    if parsed_tc is None:
                        # Unknown tool → a tool-role observation the model can react to.
                        state.messages.append(
                            _tool_msg(str(tc.get("name", "?")),
                                      f"ERROR: unknown tool {tc.get('name')!r}")
                        )
                        continue
                    verb, args = parsed_tc
                    if verb == "FINAL":
                        terminal = args  # final_answer tool → end the loop
                        break
                    # Same governor + sandbox channel as the text protocol (§10.2/§8.4); a
                    # BudgetExceeded from record_tool_call PROPAGATES (Solver.act ANDON).
                    await solver.record_tool_call(verb, args)
                    observation = await _execute(solver, verb, args)
                    state.messages.append(_tool_msg(verb, f"OBSERVATION ({verb}): {observation}"))
                    executed_any = True
                if terminal is not None:
                    return terminal
                if executed_any:
                    continue
                # tool_calls present but none usable → fall through to the text handling.

            # --- TEXT (ACTION/FINAL line) protocol --------------------------------- #
            parsed = _parse_action(text)
            if parsed is not None:
                kind, payload = parsed
                if kind == "FINAL":
                    return payload  # the answer text (already stripped of the marker)
                tool, args = kind, payload  # kind is the verb; payload is the args dict
                await solver.record_tool_call(tool, args)
                observation = await _execute(solver, tool, args)
                state.messages.append(
                    {"role": "user", "content": f"OBSERVATION ({tool}): {observation}"}
                )
                continue

            # No recognized marker AND no tool calls. A NATIVE model that emits a content
            # answer with no tool_calls is signaling completion → take it as the answer.
            # A text model gets ONE nudge, then its raw text is finalized (lenient parse).
            if use_tools and text.strip():
                return text
            if nudged:
                return text
            nudged = True
            state.messages.append({"role": "user", "content": _NUDGE})

        # Exhausted max_turns without a FINAL — return the last model text as the
        # answer. A non-converging Solver is a (likely-wrong) answer, not an error;
        # the oracle grades it and the gate closes on a wrong value (§8.3).
        return last_text

    return generate


# --------------------------------------------------------------------------- #
# Internal helpers
# --------------------------------------------------------------------------- #

#: The one-shot nudge appended when a turn produced no recognized marker.
_NUDGE = (
    "Your last message did not contain a recognized ACTION or FINAL marker. "
    "Emit exactly one of: ACTION read_file <path> | ACTION exec <command> | "
    "ACTION write_file <path> ::: <content> | FINAL <answer>."
)


def _tool_msg(tool_name: str, content: str) -> dict[str, Any]:
    """A native tool-role observation message (Ollama tool-result shape, Finding B').

    A native function-calling model expects its tool result back as a ``tool``-role
    message tagged with the tool name, not a free user turn — so the multi-turn native
    loop stays coherent (the model sees its call answered).
    """
    return {"role": "tool", "content": content, "tool_name": tool_name}


def _translate_tool_call(tc: dict[str, Any]) -> tuple[str, Any] | None:
    """Map a NATIVE tool-call ``{"name", "arguments"}`` to the loop's ``(verb, args)`` (Finding B').

    Mirrors :func:`_parse_action_body`'s output so the native path and the text path feed
    the SAME governor + sandbox channel. Lenient on argument keys (native models vary):
    ``read_file`` accepts ``path``/``file``/``filename``; ``exec`` accepts
    ``command``/``cmd``/``script``; ``write_file`` accepts ``path``+``content``/``text``;
    and a ``final_answer``/``final``/``answer``/``submit`` tool maps to
    ``("FINAL", answer)``. An unknown tool or a missing required operand returns ``None``
    (the caller records an error observation the model can react to).
    """
    name = str(tc.get("name", "")).lower().strip()
    args = tc.get("arguments")
    if not isinstance(args, dict):
        args = {}

    if name in (_READ, "read", "readfile", "open", "cat"):
        path = args.get("path") or args.get("file") or args.get("filename")
        return (_READ, {"path": str(path)}) if path else None
    if name in (_EXEC, "run", "shell", "bash", "command", "sh"):
        cmd = args.get("command") or args.get("cmd") or args.get("script")
        return (_EXEC, {"command": str(cmd)}) if cmd else None
    if name in (_WRITE, "write", "writefile", "save"):
        path = args.get("path") or args.get("file") or args.get("filename")
        content = args.get("content") or args.get("text") or ""
        return (_WRITE, {"path": str(path), "content": str(content)}) if path else None
    if name in ("final_answer", "final", "answer", "finish", "submit", "done"):
        ans = args.get("answer") or args.get("final") or args.get("text") or args.get("value") or ""
        return ("FINAL", str(ans))
    return None


def _require_solver(state: AttemptState) -> Solver:
    """Recover the live Solver the kernel parked on ``state.metadata`` (§10.2).

    The kernel stashes the live :class:`~ai_crucible.roles.Solver` under
    ``state.metadata["_kernel_solver"]`` so this ``generate`` can route tool calls
    through the kernel-side governor without widening the ``generate`` signature. A
    missing handle means ``generate`` was invoked outside a kernel Solver turn — a
    wiring bug — so we fail loud with the Ship-Gate-B structured shape rather than
    ``KeyError`` deep in the loop.
    """
    solver = state.metadata.get(_SOLVER_HANDLE)
    if solver is None:
        raise RuntimeError(
            f"[SOLVER_HANDLE_MISSING] no live Solver parked at "
            f"state.metadata[{_SOLVER_HANDLE!r}] "
            "(hint: the tool-using generate must run inside a kernel Solver turn — the "
            "kernel parks the Solver on the attempt for the turn; call it via "
            "run_attempt(generate=build_solver_generate(model)), not standalone)"
        )
    return solver


def _ensure_instruction(state: AttemptState, instruction: str) -> None:
    """Prepend the tool-use instruction as a leading ``system`` turn, once.

    Idempotent: if the loop is re-entered for the same attempt (it is not, per attempt,
    but be defensive) the instruction is not duplicated. The instruction goes FIRST so
    the model reads the protocol before the task body — and as a ``system`` turn so the
    Claude adapter splits it to the top-level ``system`` parameter (:func:`_split_system`).
    """
    marker = "ACTION read_file <path>"  # a stable fingerprint of our instruction
    for msg in state.messages:
        if msg.get("role") == "system" and marker in str(msg.get("content", "")):
            return
    state.messages.insert(0, {"role": "system", "content": instruction})


def _parse_action(text: str) -> tuple[str, Any] | None:
    """Parse the FIRST recognized ACTION/FINAL marker from a model turn (lenient).

    Tolerates prose around the marker: the first line that *starts with* (after
    stripping) ``FINAL`` or ``ACTION <verb>`` wins. Returns:

    - ``("FINAL", answer_text)`` — the answer text after the marker.
    - ``(verb, args_dict)`` for an ACTION — ``verb`` in {read_file, exec, write_file},
      ``args_dict`` the trace-shaped args ({"path": ...} | {"command": ...} |
      {"path": ..., "content": ...}). The args shape matches what Leaf A's
      oracle_runner parses off the trace.
    - ``None`` if no recognized marker is present (the caller nudges / finalizes).

    Each line carries at most one marker; across lines the FIRST recognized marker
    (scanning top to bottom) wins, so a model that reasons in prose first and emits its
    action/answer on a later line still parses.

    Harmony hardening (defense in depth): each line is cut at the first ``<|`` control
    token before parsing, so a gpt-oss leak that glues a SECOND action onto the first
    (``ACTION read_file p<|message|>ACTION exec ls<|end|>``) or trails garbage onto a
    clean marker (``FINAL 7<|end|>…``) yields only the FIRST clean marker — never a
    multi-action command stuffed with leaked tokens. A line that is *only* a control
    token cuts to empty and is skipped, so a marker on a LATER clean line still parses.
    """
    for raw in text.splitlines():
        # Cut at the first control token (defense in depth — the adapter normalizes too).
        cut = raw.split(_CONTROL_TOKEN, 1)[0]
        line = cut.strip()
        if not line:
            continue
        upper = line.upper()
        if upper.startswith("FINAL"):
            # Everything after the FINAL token is the answer (may be empty).
            return ("FINAL", line[len("FINAL"):].strip())
        if upper.startswith("ACTION"):
            action = _parse_action_body(line[len("ACTION"):].strip())
            if action is not None:
                return action
            # An ACTION line whose verb we don't recognize: keep scanning for a
            # later valid marker rather than failing the whole turn.
            continue
    return None


def _parse_action_body(body: str) -> tuple[str, dict[str, Any]] | None:
    """Parse the part of an ACTION line after the ``ACTION`` token.

    ``body`` is e.g. ``read_file config/limits.py`` / ``exec grep -rn FOO .`` /
    ``write_file out.txt ::: hello``. Returns ``(verb, args_dict)`` or ``None`` for an
    unrecognized verb / a missing operand.
    """
    parts = body.split(None, 1)
    if not parts:
        return None
    verb = parts[0].lower()
    rest = parts[1].strip() if len(parts) > 1 else ""

    if verb == _READ:
        if not rest:
            return None
        return (_READ, {"path": rest})
    if verb == _EXEC:
        if not rest:
            return None
        return (_EXEC, {"command": rest})
    if verb == _WRITE:
        # write_file <path> ::: <content>. Split on the FIRST separator so content may
        # itself contain the separator token. A missing separator means content="".
        if _WRITE_SEP in rest:
            path_part, content = rest.split(_WRITE_SEP, 1)
            path = path_part.strip()
            content = content.strip()
        else:
            path = rest.strip()
            content = ""
        if not path:
            return None
        return (_WRITE, {"path": path, "content": content})
    return None


async def _execute(solver: Solver, tool: str, args: dict[str, Any]) -> str:
    """Execute a recorded tool call via the Solver's narrow ``.tools`` channel.

    The call was ALREADY recorded through the governor by the caller (the accounting
    happens first, §8.4); this only runs the side effect and returns the observation
    text. A tool-side failure (a path that escapes the workdir → ``PermissionError``,
    a missing file → ``FileNotFoundError``) is turned into an observation string the
    model can react to — it is the environment's answer, not an attempt-fatal error.

    A :class:`~ai_crucible.budget.BudgetExceeded`, by contrast, is an ANDON breach and must
    HALT the attempt — it is re-raised, never swallowed into an observation. Two sources
    raise it: ``record_tool_call`` (BEFORE this runs, by the caller) AND the sandbox adapter's
    own per-call wall-clock kill, which raises it from INSIDE ``tools.exec`` here. The earlier
    code caught the latter in the broad ``except`` and turned a runaway-command timeout into a
    benign observation, defeating the per-call ANDON; the explicit re-raise restores it so a
    sandbox timeout stamps ``terminated_by=TIME`` like every other breach.
    """
    tools = solver.tools
    try:
        if tool == _READ:
            return await tools.read_file(args["path"])
        if tool == _EXEC:
            return await tools.exec(args["command"])
        if tool == _WRITE:
            await tools.write_file(args["path"], args["content"])
            return f"wrote {args['path']}"
    except BudgetExceeded:
        raise  # ANDON breach (e.g. sandbox per-call timeout) — halt, do not observe.
    except Exception as exc:  # noqa: BLE001 — surface as an observation, not a crash.
        return f"ERROR: {type(exc).__name__}: {exc}"
    return f"ERROR: unknown tool {tool!r}"
