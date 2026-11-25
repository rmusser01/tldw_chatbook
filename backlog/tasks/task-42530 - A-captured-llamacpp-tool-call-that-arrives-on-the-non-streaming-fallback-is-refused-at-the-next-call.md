---
id: TASK-42530
title: "A captured llama.cpp tool call that arrives on the non-streaming fallback is refused at the next call"
status: Done
assignee:
- '@claude'
created_date: 2026-11-24 09:14
updated_date: 2026-11-25 01:22
labels:
- follow-up
- console
- agents
- trace
priority: medium
references:
- tldw_chatbook/Chat/console_trace_runtime.py
- tldw_chatbook/Chat/console_trace_continuation.py
- tldw_chatbook/Chat/console_provider_gateway.py
- Tests/Chat/test_console_project_instruction_traces.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by the review of TASK-42472 (2026-11-24) and reproduced on fu/r24-diag and on its parent d0a7e3badc. Paths are under `tldw_chatbook/` unless they start with `Tests/`.

**Why.** A llama.cpp stream that returns no content (task-19324) is sent once more without streaming, on its own `llama_fallback` call with its own run id. If the fallback's reply is a tool call, the agent runs the tool, and its next model call (a TOOL_LOOP call carrying the tool result) is refused at trace reservation with `trace_tool_chain_unavailable`. The factory's tool-loop check (`Chat/console_trace_runtime.py`, `ConsoleTraceBoundaryFactory.__call__`) requires the segment's latest call to be the run's previous logical call, but the latest call is the fallback. With Capture On (the default), the run ends `blocked` on 'Trace capture could not start' after the provider answered.

TASK-42472 AC#8's shared helper (`Chat/console_trace_continuation.py::failed_stream_fallback_parent`) sees past an errored fallback only. A fallback that bore a response is not covered, on purpose: a completed fallback must never authorize an unchanged retry (`Tests/Chat/test_console_failed_call_retry_ownership.py::test_a_retry_after_a_failed_llama_stream_fallback_is_checked_against_the_stream[fallback_completed]`).

**Evidence.** Probe `/private/tmp/claude-501/-Users-macbook-dev-Documents-GitHub-tldw-chatbook/3f117384-6f37-40d1-bdd4-c465d56c3f56/scratchpad/fu/r24fx-fallback-toolloop-probe.py`: `_project_console(response_mode="fallback", tools=2)`, Capture On. The run ended `blocked`, reservation errors `['trace_tool_chain_unavailable']`, lineage `agent_first=error` then `llama_fallback=response_started`. Same on d0a7e3badc and on fu/r24-diag.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A Capture-On llama.cpp agent run whose stream is empty and whose non-streaming fallback replies with a tool call runs the tool and completes, and the trace admits the next call; a test through the real llama.cpp harness is red before the fix
- [x] #2 The user's next message after such a run is sent
- [x] #3 After a completed fallback, a TOOL_LOOP call is admitted only with the fallback's reply and its tool results appended, never as an unchanged retry
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Reproduce with the task's probe shape (`_project_console(response_mode="fallback", tools=2)`, Capture On), and a variant where only the first stream is empty.
2. Extend the round-24 helper in `Chat/console_trace_continuation.py`: besides a failed resend (`is_failed_stream_fallback`), recognise a resend that bore a response for the failed streamed call (same owner, segment, turn, policy and surface, directly after it), and find it from the streamed call.
3. Factory tool-loop check (`Chat/console_trace_runtime.py`): when the segment's latest call is the answering resend of `previous`, admit the TOOL_LOOP call as the step after that response, not as a failed-call retry, and refuse it when it appends nothing (never an unchanged retry, AC#3).
3a. (Added after the first probe, 2026-11-24.) When every stream is empty, as in this task's own probe, the next tool-loop call's stream is empty too, and its resend was refused with `trace_turn_unavailable`: a resend of a tool-loop call ends on a tool result, not a saved user message. Admit that resend on the tool loop's turn, only directly after the failed tool-loop call it repeats and with nothing appended.
4. Next send (AC#2): the completed-turn witness counts an answering resend as this run's call (the foreign-call check), and the TASK-42534 recheck accepts tool history appended after a failed call whose resend answered.
5. Tests through the real llama.cpp harness (red before), gateway-level negatives (a TOOL_LOOP claiming an unchanged retry after a completed resend stays refused; another chain's answering resend is not this run's).
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
**Cause.** Two refusals, the second hidden behind the first.
1. As filed: the factory's tool-loop check (`ConsoleTraceBoundaryFactory.__call__`) requires the segment's latest call to be the run's previous call. After a llama.cpp stream that returned no content, the latest call is its own-run `llama_fallback` resend, which answered with the tool call, so the agent's next call was refused with `trace_tool_chain_unavailable`.
2. With 1 fixed, the task's own probe (`response_mode="fallback"`: every stream empty) still ended `blocked`: the next tool-loop call's stream was empty too, and its resend was refused with `trace_turn_unavailable`. A resend runs on its own `llama_fallback` run, and the factory takes a non-tool-loop call's turn from its last saved user message; a tool-loop request ends on a tool result. (A resend of an in-place retry with no tool history ends on the user's message, which is why TASK-42472's retry rounds worked.)

**Fix**, reusing the round-24 helper in `Chat/console_trace_continuation.py`:
- `_resends` holds the shared identity rule (a `llama_fallback` call directly after an errored call, same owner, segment, turn, policy and surface). `is_failed_stream_fallback` keeps its meaning; `is_answering_stream_fallback` is the response-bearing case; `answering_stream_fallback(streamed)` finds it from the streamed call. `failed_stream_fallback_parent(..., answered=True)` also accepts an answering resend.
- Factory, tool loop: when the latest call is the answering resend of `previous`, the call is admitted as that reply's next step, not as a failed-call retry, and refused if it appends nothing (AC#3: never an unchanged retry).
- Factory, resend of a tool-loop call: it takes the loop's turn (the request's last saved revision, as a tool-loop call does) and is admitted only directly after the settled failed tool-loop call it repeats (same owner, turn and policy, on the segment's surface tail), with nothing appended.
- Next message (AC#2): the completed-turn witness counts an answering resend as part of the run (the foreign-call check), and TASK-42534's tool-history recheck accepts history appended after a failed call whose resend answered.
- (Superseded in review round 1, below: `persist_request` now compares a resend with the failed call it repeats.)
- ADR-337 gained "Integration recorded 2026-11-24: stream resends inside a run"; the User Guide's semantic-trace-capture page says the run continues from a resend's reply with capture on.

**Tests** (`Tests/Chat/test_console_failed_call_retry_ownership.py`).
- `test_a_llamacpp_tool_call_on_the_stream_resend_runs_the_tool`, the real llama.cpp Console harness (`_project_console`, Capture On), `project`/`no_project`:
  - `first`: only the first stream is empty; lineage `agent_first` error, `llama_fallback` complete, `tool_loop` complete;
  - `every`: every stream is empty (the task's probe shape); lineage `agent_first` error, `llama_fallback` complete, `tool_loop` error, `llama_fallback` complete.
  Each asserts the tool's result went out, the reply is "Fixture reply", and the next message is sent and completes (AC#1, AC#2).
- `test_a_retry_after_a_failed_llama_stream_fallback_is_checked_against_the_stream` gained `fallback_completed_tool_step` (the resend answered; the next call carries its reply and a tool result: admitted) and `other_chain_answered` (another chain's resend answered in between: refused). The existing `fallback_completed` (an unchanged retry after an answering resend) is still refused (AC#3).
- `test_a_resend_of_a_tool_loop_call_repeats_its_failed_call`: through the gateway, a resend after a failed streamed tool-loop call is admitted (`failed_tool_loop`); a resend whose latest call answered (`answered`) or is a failed first call (`failed_first_call`) is refused with `trace_tool_chain_unavailable` (direct factory call).
- Red before (source at the TASK-42534 commit 4aab6c6570): 7 failed, the 4 harness cases (`assert ['trace_tool_..._unavailable'] == []`), `fallback_completed_tool_step`, and both resend negatives (`DID NOT RAISE`: an own-run resend was never checked against the call it repeats).
- Each part pinned by a temporary edit, reverted: without the "never an unchanged retry" check, `fallback_completed` is admitted; without the resend check, both resend negatives are admitted; without the tool-loop turn for a resend, the two `every` cases fail (`trace_turn_unavailable`); without the witness's answering-resend acceptance (foreign check), `first-project`, `first-no_project` and `every-project` fail at the next message (`completed_tool_turn_lineage`); without it in the tool-history recheck, `first-project`, `first-no_project` and `every-project` fail the same way.

**Files.** `tldw_chatbook/Chat/console_trace_continuation.py`, `tldw_chatbook/Chat/console_trace_runtime.py`; `Tests/Chat/test_console_failed_call_retry_ownership.py`; ADR-337 and the User Guide page above.

**Runs.** Same branch and runs as TASK-42534's notes.

**Review round 1 (2026-11-24).**
- *I3: nothing proved an answering resend repeated the failed call.* A `llama_fallback` call is `call_sequence` 0 of its own run, so `persist_request`'s retry header comparison never ran for it, and the factory's resend checks cover owner, turn, policy, surface and state only. `persist_request` now compares a `llama_fallback` call with the call whose boundary directly precedes its own, when that call has the same owner, segment, turn, policy and surface and is in ERROR: provider, model, endpoint, generation settings, response format, reasoning controls, system composition, rendered system parts, tool schemas and literal envelope. Running the existing comparison unchanged refused every llama.cpp resend: its literal envelope is `{"model": ..., "stream": false}` against the stream's `"stream": true` (temporary log, removed). `retry_header_components` (`Chat/console_trace_continuation.py`) therefore compares the envelope without its `stream` key, only for a pair whose generation settings flip True -> False, which `_retry_generation_matches` admits only with the `llama_fallback_retry` overlay. The earlier note that `persist_request` "was left alone" no longer holds. Test: `test_a_llamacpp_resend_is_compared_with_the_call_it_repeats`: a stand-in for the stored stream's header (another model) or its envelope (an extra `top_k`) refuses the resend before it is sent; unforged, it answers.
- *I4: the "a tool-loop resend appends nothing" guard was unpinned.* `test_a_resend_of_a_tool_loop_call_repeats_its_failed_call` gained `unchanged_after_failed_tool_loop` (a direct factory resend right after a failed, non-streamed tool-loop call: admitted) and `appended_after_failed_tool_loop` (the same request plus a tool result: refused with `trace_tool_chain_unavailable`).
- *I5: a tool-loop resend in a turn with a chat-dictionary transform was refused* (`trace_current_transform_owner`): the transform-owner check let only a tool-loop call carry the transformed user row before its tail. A tool-loop resend takes the loop's turn, so it is treated the same. Test: `every_transformed` in `test_a_llamacpp_tool_call_on_the_stream_resend_runs_the_tool` (project and no project), red before with `['trace_curre...nsform_owner']`.
- *Red evidence this round* (each temporary edit reverted; the touched file plus the project-fallback function, `-n 4`):
  - without INTERRUPTED in `bore_response`: both restart-and-discard cases fail (`completed_tool_turn_lineage`);
  - witness origin from the resend's own run: 7 fail, the four new llama.cpp tool runs, `llama_stream_exhausted-project`, `every-project`, `every_transformed-project`;
  - the validator's resend remap gated on project context again: 4 fail, the no-project llama.cpp tool runs and `every`/`every_transformed` without project;
  - the validator's old origin check (`origin != terminal`): `test_project_fallback_requires_the_exact_preceding_failed_stream[valid]` and `[fallback_origin]` fail;
  - no resend comparison: `[model]` and `[envelope]` forgeries are admitted;
  - the envelope compared with its `stream` key: every llama.cpp resend is refused (18 or more fail);
  - no append guard: `appended_after_failed_tool_loop` is admitted;
  - the transform-owner check as before: both `every_transformed` cases fail (`trace_current_transform_owner`);
  - (TASK-42532) no settings-equality check: both `forged_stream_settings` cases are admitted.
  At the round base 75bac413f5, the restart-and-discard cases pass (the send was blocked only on this branch), and the four new llama.cpp tool runs fail during the first run (`trace_turn_unavailable` or `trace_tool_chain_unavailable`).
- Runs for this round: TASK-42534's review notes.

**Re-review (2026-11-24).** `retry_header_components` compared each header on its own, so two missing envelope artifacts both read as `null` and matched, where the old content-id comparison failed closed; and it dropped `stream` from both sides of any resend pair. `retry_header_parts_match` replaces it and compares the pair: identical parts match; under the verified flip only the literal envelope may differ, both artifacts must load (a missing one refuses), and the stream's envelope must have `"stream": true` and equal the resend's with that turned false (canonical JSON, so `0` is not `false`). Tests in `test_a_llamacpp_resend_is_compared_with_the_call_it_repeats`: `missing_envelope` (both envelopes read as missing once the resend's header is written) and `unstreamed_envelope` (the stream's envelope reads `"stream": false`), both refused. Each was admitted at 7cfd270751 and under its matching temporary edit (a missing artifact treated as equal; the `"stream": true` requirement removed), both reverted. `console_trace_service.py`'s row goes 6,829 -> 6,826 (-3, exact).
<!-- SECTION:NOTES:END -->
