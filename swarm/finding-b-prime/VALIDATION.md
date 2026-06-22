# Finding B′ — multi-model Solver protocol (native tool_calls)

**Problem.** `solver_loop` read only the TEXT `ACTION`/`FINAL` line protocol. gpt-oss-cloud
solves via NATIVE function-calls — it returns `message.tool_calls` with EMPTY `content` —
so a text-only loop saw nothing and the model could not Solve (the prior swarm noted
"gpt-oss:120b flailed (Harmony + empty-content/native-tool_calls)").

**Fix.**
- `models/ollama_adapter.py`: `_extract_tool_calls` (normalizes `message.tool_calls` →
  `[{name, arguments}]`, JSON-decoding string-form arguments) + `complete_turn(messages,
  tools) -> (text, tool_calls)` that offers the function-calling schemas and returns both
  the text and the native calls (same single request path + provenance/retry guards).
- `solver_loop.py`: `SANDBOX_TOOL_SCHEMAS` (read_file/exec/write_file/final_answer) + a
  native branch in the loop (duck-typed on `complete_turn`): a turn's `tool_calls` route
  through the SAME governor + sandbox channel as the text protocol (`_translate_tool_call`
  maps tool name → verb, lenient on arg keys); a native content answer with no tool_calls
  terminates; `final_answer` ends the loop. A text-only model (Claude, canned) is
  UNCHANGED (the native branch is never taken).

**Tests.** 9 new (4 adapter in `test_models.py`, 5 loop in `test_solver_loop.py`):
native read→final_answer grounds + records through the governor; native content-answer
terminates; unknown tool → error observation (no crash); native budget breach propagates
the ANDON; the text protocol is unaffected; tool_calls extraction (dict + JSON-string args,
empty, malformed-skipped); complete_turn offers the schemas / omits them on the text path.

**Live validation (2026-06-21).** `ai-crucible run puzzles/seed-sulzbach-55252 --model
gpt-oss:120b-cloud@gpt-oss --k 1` — the daemon served `gpt-oss:120b-cloud`; the model
emitted native tool_calls that EXECUTED through the sandbox:

```
tool: exec      | args: {'command': 'ls -R'}
tool: exec      | args: {'command': 'grep -R "UPLOAD_MAX_ATTEMPTS" -n .'}
tool: read_file | args: {'path': 'config/limits.py'}
final output: '7'   →   gate_passed: True, no penalties (grounded, no bait touch)
```

gpt-oss now Solves via the native protocol (it grounded the read and answered 7, gate
PASS). Run-to-run variance is real (a prior sample answered differently and the gate
CLOSED) — that variance IS the pass^k diagnostic signal, not a bug. The instrument now
measures non-text-protocol models as Solvers.
