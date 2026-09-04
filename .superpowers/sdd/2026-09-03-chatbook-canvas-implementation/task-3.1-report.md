# Task 3.1 report — safe generic tool-record projection

## Implementation

- Added the immutable `ToolRecordProjection` contract and the four exact
  `ToolProjectionAudience` values: `display`, `log`, `cycle`, and
  `continuation`.
- Added the optional `ToolRecordProjectionProvider` protocol and
  `ToolCatalogRegistry.project_tool_record()` dispatch seam. Providers without
  the hook retain byte-compatible argument/result behavior.
- A throwing, malformed, or unavailable projector now fails closed to bounded
  tool name, call ID, success state, and exception-category metadata. No raw
  arguments, result, exception message, repr, or exception chain is used.
- Wired the catalog projector into `AgentService` and `LoopDeps`, then routed
  `AgentStep` display records, run-log tool calls/results, model records for
  sensitive tool-call turns, cycle keys, and durable continuation checkpoints
  through the appropriate audience. Immediate invocation and model tool-result
  history retain raw values as required.
- Added regression tests for default fallback behavior across builtin/local/
  skill/MCP catalog source shapes, all four audiences, failure closure, the
  runtime inventory, and continuation checkpoint projection.

## Raw-consumer inventory

| Consumer | Audience | Result |
| --- | --- | --- |
| `AgentStep.args` and `AgentStep.result`, consumed by live/resumed Console display | `display` | projected before step creation |
| `LoopDeps.on_record` tool-call and tool-result payloads | `log` | projected before durable run-log writer |
| model run-log entry for a sensitive tool-call turn | `log` | emits safe tool-call metadata rather than raw turn text |
| `recent_calls` cycle key | `cycle` | serializes projected arguments |
| `ToolBatchReady` / `FinalContinuation` checkpoints and terminal continuation results | `continuation` | persists projected calls/results; local model history stays raw |
| Console summaries and resumed markers | `display` | consume the already-projected `AgentStep` fields |

## RED

Command:

```sh
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q Tests/Agents/test_tool_record_projection.py
```

Relevant output before implementation:

```text
ImportError: cannot import name 'ToolProjectionAudience'
```

## GREEN

Focused runtime/catalog/run-store/continuation/cycle command:

```sh
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q Tests/Agents/test_agent_runtime.py Tests/Agents/test_tool_catalog.py Tests/Agents/test_provider_continuation_runtime.py Tests/Agents/test_run_log_writer.py Tests/Agents/test_run_log_on_record.py Tests/Agents/test_run_log_format.py
```

Relevant output:

```text
222 passed, 1 warning in 2.45s
```

Targeted static checks:

```sh
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m py_compile tldw_chatbook/Agents/agent_models.py tldw_chatbook/Agents/tool_catalog.py tldw_chatbook/Agents/agent_runtime.py tldw_chatbook/Agents/agent_service.py tldw_chatbook/Chat/console_agent_bridge.py
git diff --check
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m ruff check Tests/Agents/test_tool_record_projection.py --select F,I
```

Relevant output: `All checks passed!`

## Files changed

- `tldw_chatbook/Agents/agent_models.py`
- `tldw_chatbook/Agents/tool_catalog.py`
- `tldw_chatbook/Agents/agent_runtime.py`
- `tldw_chatbook/Agents/agent_service.py`
- `Tests/Agents/test_tool_record_projection.py`
- `Tests/Agents/test_agent_runtime.py`
- `Tests/Agents/test_provider_continuation_runtime.py`

## Self-review

- Confirmed every runtime `AgentStep` tool-call construction uses the display
  projection and no direct `args=dict(call.args)` remains.
- Confirmed run-record and cycle-key construction no longer serialize raw call
  arguments.
- Confirmed raw tool results remain only in the immediate model history path;
  durable continuation events receive projected checkpoints/results.
- Confirmed default catalog fallback keeps existing provider records unchanged.
- Confirmed projection exceptions and malformed metadata are fail-closed.

## Concerns

- The repository-wide Ruff invocation reports existing baseline findings in
  large Agent/Console modules, so it was not used as a pass/fail sweep. The
  changed projection test passes import/unused checks; all changed production
  modules compile and the diff is whitespace-clean.
- Canvas tool registration and its final metadata fields remain intentionally
  deferred to Tasks 3.2/3.3; this task supplies only the generic seam.
