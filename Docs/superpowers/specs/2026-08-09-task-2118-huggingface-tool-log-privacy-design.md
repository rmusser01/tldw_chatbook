# TASK-2118 — HuggingFace tool-log privacy design

- Date: 2026-08-09
- Status: approved design, pending implementation plan
- Backlog: TASK-2118
- Existing decision: `backlog/decisions/029-local-private-data-boundary.md`
- ADR required: no
- ADR reason: ADR-029 already requires persistent logs to contain metadata only and explicitly permits tool names while excluding tool definitions, arguments, and request content. This task applies that decision without changing a boundary or contract.

## Outcome

`chat_with_huggingface` retains useful debug metadata without serializing tool descriptions, parameter schemas, enum values, or other unapproved request fields. Sensitive requests continue to omit the HuggingFace tools log entirely. Ordinary requests log only the tool names extracted by the existing allowlist helper.

## Verified current state

- `chat_with_huggingface` currently interpolates `payload["tools"]` directly into the `HuggingFace Tools` debug line when the request is not sensitive.
- The same ordinary-request block also builds `HuggingFace Final Payload` by excluding only `messages` and `tools`. That is denylist-shaped and can expose any newly added request field by default.
- `safe_llm_request_payload_summary()` already allowlists scalar request metadata and delegates tool handling to `safe_llm_tool_names()`, which returns recognizable tool names and drops definitions, schemas, and arguments.
- Sensitive HuggingFace requests currently log only model, streaming state, message count, and content byte count; the tools line is skipped.
- The focused baseline is green: `Tests/Chat/test_sensitive_llm_logging.py` has 68 passing tests, and the HuggingFace subset of `Tests/Chat/test_chat_functions.py` has 8 passing tests.
- No open pull request or competing remote branch for TASK-2118 was found before the task was claimed.

## Selected approach

Make two narrow call-site changes in `chat_with_huggingface`; do not add a sanitizer or abstraction.

1. Replace the ordinary `HuggingFace Final Payload (excluding messages, tools)` rendering with `safe_llm_request_payload_summary(payload)` and label it as safe fields only. This closes the remaining denylist-shaped request-payload log found by the acceptance-criterion sweep.
2. Preserve the separate `HuggingFace Tools` diagnostic, but pass a tools-only projection through `safe_llm_request_payload_summary(..., content_keys=())`. Its rendered data then contains only `tool_names`, without an unrelated message/system summary.
3. Preserve the existing sensitive branch exactly: metadata only, with no `HuggingFace Tools` line. The fix must not widen sensitive logging merely because tool names are permitted metadata.
4. Make no changes to `safe_llm_request_payload_summary()` or `safe_llm_tool_names()` unless a failing test proves their existing contract is insufficient.

The ordinary path will therefore emit allowed metadata twice when tools are present: once in the complete safe request summary and once under the tool-specific diagnostic label. That small duplication preserves the established diagnostic line and directly satisfies the task contract without adding helper flags or another logging abstraction.

## Alternatives considered

### Add a HuggingFace-specific redactor

Rejected. It would duplicate the existing allowlist contract and create another implementation that could drift when provider payloads evolve.

### Delete the `HuggingFace Tools` line

Rejected. Deletion is privacy-safe, but loses the task's explicitly required tool-name diagnostic and would make the acceptance criterion vacuous.

### Log safe tool names during sensitive requests too

Rejected. Tool names are allowed by ADR-029, but the current sensitive behavior is stricter. Widening it is unnecessary for this repair.

## Test design

Add a call-site regression test in `Tests/Chat/test_sensitive_llm_logging.py` that drives the real `chat_with_huggingface()` function with only its transport/config seams replaced. Do not construct a test application.

The test is parameterized over ordinary and `sensitive_llm_request()` contexts and captures Loguru through a temporary list sink, matching the repository's proven logging-test pattern. The supplied OpenAI-style tool contains:

- an allowed distinctive tool name;
- a distinctive description sentinel;
- a separate distinctive parameter-schema/enum sentinel.

Assertions:

- neither secret sentinel appears anywhere in captured logs on either path;
- the ordinary path includes the tool name and a `HuggingFace Tools` safe-summary line;
- the sensitive path contains no `HuggingFace Tools` line;
- existing message/system canaries remain absent under the sensitive path.

TDD evidence must include a red run against the current raw interpolation and a green run after the call-site change. Before completion, temporarily restore the raw tool interpolation and confirm the new test becomes red, then restore the fix. This mutation check proves the test detects the actual leak rather than only exercising the helper.

## Sweep boundary

Sweep every Python module under `tldw_chatbook/LLM_Calls/` for logger calls that directly format request-payload or tool-definition values. Use an AST/text-assisted inventory, then inspect each candidate in context because names such as `data` also represent response events.

- Any raw request/tool log is fixed in this task by routing through the existing allowlist helper.
- Response-status, bounded parser diagnostics, and values that are not request payloads/tool definitions are left unchanged and explicitly justified in TASK-2118 Implementation Notes.
- The sweep is evidence for acceptance criterion 4; a new permanent broad AST test is not required unless the sweep finds a recurring mechanically detectable contract that existing tests do not cover.

## Verification and scope

Required focused verification:

- `Tests/Chat/test_sensitive_llm_logging.py`
- HuggingFace tests in `Tests/Chat/test_chat_functions.py`
- Ruff lint/format checks for edited Python files
- `python -m py_compile` for edited Python files
- the documented LLM-call logging sweep
- `git diff --check`

The implementation scope is limited to the HuggingFace logging call sites, their function-level privacy regression test, TASK-2118 documentation, and this spec/plan. No application-state, provider-routing, transport, or UI behavior changes are included.
