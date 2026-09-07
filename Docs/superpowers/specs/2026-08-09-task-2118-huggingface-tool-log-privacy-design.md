# TASK-2118 — HuggingFace tool-log privacy design

- Date: 2026-08-09
- Status: implemented and verified
- Backlog: TASK-2118
- Existing decision: `backlog/decisions/029-local-private-data-boundary.md`
- ADR required: no
- ADR reason: ADR-029 already requires persistent logs to contain metadata only and explicitly permits tool names while excluding tool definitions, arguments, and request content. This task applies that decision without changing a boundary or contract.

## Outcome

`chat_with_huggingface` retains useful debug metadata without serializing tool descriptions, parameter schemas, enum values, or other unapproved request fields. Sensitive requests continue to omit the HuggingFace tools log entirely. Ordinary requests log only the tool names extracted by the existing allowlist helper.

## Verified pre-implementation state

- `chat_with_huggingface` interpolated `payload["tools"]` directly into the `HuggingFace Tools` debug line when the request was not sensitive.
- The same ordinary-request block also built `HuggingFace Final Payload` by excluding only `messages` and `tools`. That was denylist-shaped and could expose any newly added request field by default.
- `safe_llm_request_payload_summary()` already allowlisted scalar request metadata and delegated tool handling to `safe_llm_tool_names()`, which returned recognizable tool names and dropped definitions, schemas, and arguments.
- Sensitive HuggingFace requests logged only model, streaming state, message count, and content byte count; the tools line was skipped.
- The focused baseline was green: `Tests/Chat/test_sensitive_llm_logging.py` had 68 passing tests, and the HuggingFace subset of `Tests/Chat/test_chat_functions.py` had 8 passing tests.
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
- a separate distinctive parameter-schema/enum sentinel;
- and the request supplies a distinctive `user` sentinel, which the current denylist-shaped Final Payload log exposes but the allowlist helper intentionally drops.

Assertions:

- neither secret sentinel appears anywhere in captured logs on either path;
- the unallowlisted `user` sentinel appears nowhere in captured logs;
- the ordinary path has exactly one `HuggingFace Tools` record whose rendered value is the expected `{"tool_names": [<name>]}` projection and contains none of the keys `description`, `parameters`, or `enum`;
- the sensitive path contains no `HuggingFace Tools` line;
- existing message/system canaries remain absent under the sensitive path.

TDD evidence must include a red run against the current raw logs and a green run after both call-site changes. Before completion, perform two independent mutation checks: temporarily restore the raw tool interpolation and confirm the tool sentinel assertions become red; then restore the tool fix, temporarily restore the old denylist-shaped Final Payload expression, and confirm the `user` sentinel assertion becomes red. Restore both fixes afterward. These checks prove the test detects both leaks rather than only exercising the helper.

## Sweep boundary

Sweep every Python module under `tldw_chatbook/LLM_Calls/` for logger calls that directly format raw request-payload dictionaries or raw tool definitions, matching acceptance criterion 4. Use an AST/text-assisted inventory, then inspect each candidate in context because names such as `data` also represent response events.

- Any raw request-payload dictionary or tool-definition log is fixed in this task by routing through the existing allowlist helper.
- Response events, status metadata, bounded parser diagnostics, and individual content, prompt, response/output, credential-fragment, private endpoint/path, and exception/error-detail diagnostics are not AC 4 matches. The original identifier-filtered candidate subset was not a complete inventory of that broader privacy boundary. Final review therefore inspected all 523 logger calls in the two summarization modules and assigned 199 direct private diagnostics to the standalone follow-up: 100 local and 99 general, categorized as 21 input, 17 prompt, 21 credential-fragment, 11 private endpoint/path, 71 response/output, and 58 exception/error-detail sites. TASK-2118 records the complete categorized ownership without referencing the later task ID, preserving the repository's no-forward-reference rule and this repair's atomic scope.
- The sweep is evidence for acceptance criterion 4; a new permanent broad AST test is not required unless the sweep finds a recurring mechanically detectable contract that existing tests do not cover.

## Verification and scope

Required focused verification:

- `Tests/Chat/test_sensitive_llm_logging.py`
- HuggingFace tests in `Tests/Chat/test_chat_functions.py`
- `Tests/LLM_Calls/test_debug_log_fstring_hygiene.py`
- `Tests/Architecture/test_persistent_diagnostic_inventory.py`, with exact current-dev baseline comparison for unrelated stored-inventory drift
- Ruff lint/format checks for edited Python files
- `python -m py_compile` for edited Python files
- the documented LLM-call logging sweep
- `git diff --check`

The implementation scope is limited to the HuggingFace logging call sites, their function-level privacy regression test, the one branch-owned diagnostic-manifest digest, TASK-2118 documentation, the complete summarization follow-up inventory, the testing-evidence lesson, and this spec/plan. No application-state, provider-routing, transport, summarization runtime, or UI behavior changes are included.
