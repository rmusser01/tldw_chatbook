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

## Fix round 1 (review follow-up)

### Implementation

- Added a registry-owned `has_tool_record_projection()` signal and wired it
  through `AgentService` into `LoopDeps`. The runtime now uses that
  authoritative opt-in distinction: compatibility providers take the default
  path without strict `allow_nan=False` validation, while opted-in projectors
  remain strictly validated and fail closed.
- Replaced the sensitive-tool detection based on projected-argument equality.
  Any tool whose owning provider opted into projection now gets the bounded
  `Tool call recorded: <name>` text for both the model run-log record and its
  persisted/displayed `STEP_MODEL` summary. Ordinary providers retain their
  original `turn.text[:200]` summary and model log content.
- Expanded the sentinel inventory test to inspect every emitted `AgentStep`,
  the durable step serialization, and the Console resumed-step summary. It
  verifies that a unique HTML sentinel present in both fence text and call
  arguments is absent outside volatile model history.
- Added default-provider NaN/Infinity/-Infinity coverage for every audience
  plus runtime display coverage, and malformed/non-finite opt-in projector
  tests that verify bounded fallback metadata without payload or exception
  leakage.

### RED

Command:

```sh
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q Tests/Agents/test_agent_runtime.py Tests/Agents/test_tool_record_projection.py
```

Relevant output before the fix:

```text
FAILED Tests/Agents/test_agent_runtime.py::test_tool_record_audience_inventory_keeps_raw_payload_only_in_model_history
FAILED Tests/Agents/test_agent_runtime.py::test_default_projection_keeps_nonfinite_tool_arguments_in_runtime_records
2 failed, 96 passed, 1 warning in 1.55s
```

The first failure showed the raw HTML sentinel in a resumed model-step
summary; the second showed the `ValueError` failed-projection metadata in
place of the legacy `NaN` value.

### GREEN

Focused runtime/catalog/run-store/continuation/cycle command:

```sh
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q Tests/Agents/test_agent_runtime.py Tests/Agents/test_tool_record_projection.py Tests/Agents/test_tool_catalog.py Tests/Agents/test_provider_continuation_runtime.py Tests/Agents/test_run_log_writer.py Tests/Agents/test_run_log_on_record.py Tests/Agents/test_run_log_format.py
```

Relevant output:

```text
267 passed, 1 warning in 2.58s
```

Targeted static checks:

```sh
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m py_compile tldw_chatbook/Agents/agent_models.py tldw_chatbook/Agents/tool_catalog.py tldw_chatbook/Agents/agent_runtime.py tldw_chatbook/Agents/agent_service.py
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/ruff check Tests/Agents/test_agent_runtime.py Tests/Agents/test_tool_record_projection.py Tests/Agents/test_provider_continuation_runtime.py --select F,I
git diff --check
```

Relevant output: `All checks passed!`

### Files changed in this round

- `tldw_chatbook/Agents/agent_runtime.py`
- `tldw_chatbook/Agents/agent_service.py`
- `tldw_chatbook/Agents/tool_catalog.py`
- `Tests/Agents/test_agent_runtime.py`
- `Tests/Agents/test_tool_record_projection.py`
- `Tests/Agents/test_provider_continuation_runtime.py`

### Self-review

- Confirmed the opt-in signal is computed only from the cached catalog owner;
  a projector cannot select the compatibility path through a returned marker.
- Confirmed a detector failure produces bounded failed-projection metadata and
  redacts the model summary as an unknown/sensitive mode.
- Confirmed only opted-in projectors use strict finite JSON validation; their
  serialization still cannot emit non-finite values after validation.
- Confirmed the sentinel test includes `STEP_MODEL` and its durable/resumed
  forms, rather than filtering to tool-call and tool-result steps.

### Concerns

- This follow-up intentionally leaves Canvas registration and turn staging to
  Tasks 3.2/3.3. No Canvas-specific branch was added to generic runtime calls.
