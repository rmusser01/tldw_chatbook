---
id: TASK-26005
title: 'Tool calls: coerce malformed model arguments before dispatch'
status: Done
assignee:
  - '@claude'
created_date: '2026-08-31 15:44'
updated_date: '2026-08-31 19:15'
labels:
  - agents
  - tools
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
A model that emits a stringified JSON array where an array is expected fails the call outright. Verified on origin/dev: Agents/agent_runtime.py:1149 does json.loads(call.arguments) straight into the provider path with no repair layer, so a common small-model failure mode costs a whole turn. Hermes runs coerce_tool_args, which is schema-aware and recursive: it converts JSON-string values to native arrays and objects where the schema calls for them and sanitizes fenced output.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A tool argument arriving as a JSON string where the schema declares an array or object is coerced to the native type before dispatch
- [x] #2 Coercion is schema-driven: a string is left alone where the schema declares a string
- [x] #3 Coercion is recursive through nested objects and arrays
- [x] #4 A value that cannot be coerced produces the existing validation error, not a silent wrong-type dispatch
- [x] #5 Every coercion is recorded in the run log so a systematically malformed model is visible rather than masked
- [x] #6 Tests include the adversarial cases: double-encoded JSON, fenced JSON, and a legitimately string-typed field containing bracket characters
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no. A best-effort repair at an existing choke point; no new dependency or seam.

1. Coerce from the SCHEMA, never from the value. Guessing from the value would corrupt a legitimately string-typed field containing brackets -- silent data corruption in place of a loud validation error, which is a worse failure than the one being fixed.
2. Repair only string-to-container, and only when the decoded value actually matches the declared type. Everything else falls through to existing validation (AC#4).
3. Put it in `invoke_by_name`, the one line every provider is reached through, so a provider added later inherits it. That method has two dispatch sites, so coerce once at the top rather than at either.
4. Keep it in its own module: `tool_catalog.py` is already 1644 lines and TASK-26007 also edits it.
5. Never raise and never block: a provider that cannot produce a schema gets its arguments unchanged.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Added `Agents/tool_arg_coercion.coerce_tool_args` and wired it into `ToolCatalogRegistry.invoke_by_name`, so a model that JSON-encodes an argument as a string no longer burns a turn on a validation failure.

**Schema-driven, deliberately.** Coercion never inspects the value to decide what it should be. A `{"type": "string"}` field containing `["not", "an", "array"]` is real prose and is left exactly alone; guessing would turn a loud validation error into silent data corruption, which is worse than the bug being fixed. A field with no declared type is also left alone, for the same reason.

**Only the safe direction.** String-to-array and string-to-object, and only when the decoded value actually matches the declared type. `"42"` under an array schema decodes fine but is not an array, so it is left for normal validation rather than substituted (AC#4). Decoding is bounded at two passes, which covers encoded-once and double-encoded without letting a pathological input drive a long loop. Markdown code fences are stripped first, since models wrap values in them.

**Placement.** `invoke_by_name` has two `provider.invoke` sites, so the repair runs once at the top rather than at either -- and that method is the single line every provider is reached through, which is why the ephemeral gate and the per-run call caps already live there. A provider added later inherits the repair without opting in. It is best-effort: a provider that cannot produce a schema gets its arguments unchanged, because failing to repair is strictly better than failing to dispatch.

The function lives in its own module rather than in `tool_catalog.py`, which is already 1644 lines and is also edited by TASK-26007 -- keeping them apart reduces the collision between two lanes of the burn-down plan.

**AC#5, stated precisely.** Coercions are reported through a `logger.warning` at the dispatch choke point naming the tool and the repaired field paths, so a systematically malformed model is visible rather than masked. This is the application log, NOT a `RunLogWriter` record: the run log is written explicitly by `agent_service`, and reaching it from the catalog would mean threading a writer through `ToolCatalogRegistry`. That is a real design question rather than a line of plumbing, so the cheaper visible-and-greppable form shipped and the difference is recorded here rather than the AC being quietly reinterpreted. Say so if literal run-log records are wanted and it becomes a follow-up.

**Verification.** 18 tests: 17 on the pure function plus one that drives a real provider through `invoke_by_name` and asserts the provider received `["a", "b"]` rather than the raw string -- the unit tests prove the function, that one proves it is wired in. Adversarial cases covered per AC#6: double-encoded JSON, fenced JSON, a string-typed field full of brackets, truncated JSON, empty and whitespace-only values, and a decode that yields the wrong type. Input immutability is asserted.

`Tests/Agents/` shows the same 15 baseline failures before and after (2197 passing, up 18); `Tests/MCP/` unchanged at 2 baseline failures / 915 passing.

**Files:** `tldw_chatbook/Agents/tool_arg_coercion.py` (new), `tldw_chatbook/Agents/tool_catalog.py`, `Tests/Agents/test_tool_arg_coercion.py` (new).

## Review round

**AC#5 had no test at all.** The integration test opened a `caplog.at_level(logging.WARNING)` block and never asserted on it — and it would not have worked anyway, since `tool_catalog` logs through loguru, which does not reach pytest's `caplog` without an explicit sink. Added a real test using a loguru sink that asserts exactly one report is emitted for a repaired call, naming the tool and the field, and none for an already-correct call.
<!-- SECTION:NOTES:END -->
