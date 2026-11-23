---
id: TASK-42472
title: "Decide whether an agent's follow-up provider call after a failed first call should be refused at trace reservation with surface_replacement_checkpoint_unavailable"
status: Done
assignee:
- '@claude'
created_date: 2026-11-23 06:38
updated_date: 2026-11-23 20:26
labels:
- follow-up
- console
- agents
- trace
- investigation
priority: low
references:
- tldw_chatbook/Chat/console_trace_service.py
- tldw_chatbook/Chat/console_trace_runtime.py
- tldw_chatbook/Agents/activation.py
- Tests/Performance/test_console_managed_received_acceptance.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found during round 22 of the follow-up burn-down (2026-11-23), by TASK-42412. Checked on the round-22 stack 336d15338b (dev 2f7790873a plus round 22's stacked branches); paths are under `tldw_chatbook/` unless they start with `Tests/`.

**Why.** In TASK-42412's local reproductions (a natural one in 1 of 18 standalone runs, and a forced one), the agent's first provider call (AGENT_FIRST) reserved its trace, and the TOOL_LOOP call that followed (`first_call=False`) was refused at reservation. The refusal was `ValueError("surface_replacement_checkpoint_unavailable")` from `prepare_current_surface_delta` (`Chat/console_trace_service.py:2927`, the branch `if replacement_range is not None and bootstrap`). It was reached through `ConsoleTraceBoundaryFactory.__call__` (`Chat/console_trace_runtime.py:816`), re-raised as `TraceReservationRefused` by `_rolled_back_refusal` (line 84), and left the reservation `not_established`. There the first call had failed because the test's own recorder (its stub provider path) raised, not a real provider. Whether a real provider error on an agent's first call leads to the same refusal on its next call, so that a user's agent run stops at trace reservation instead of reporting the provider error, is not known.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The notes record whether a real provider error on an agent's first call (not a test double) can be followed by a TOOL_LOOP call, and if so whether that call is refused with `surface_replacement_checkpoint_unavailable`
- [x] #2 If it can, the intended behaviour is recorded (refusal with a reason the user sees, or a reservation against the failed call's checkpoint) and the code matches it, with a test shown red first for any change
- [x] #3 If it cannot, the task closes with the code path that rules it out
<!-- AC:END -->

## Implementation Plan

1. Trace the code path from a failed AGENT_FIRST provider call to the next reservation: the agent loop's failure handling, the route choice, and the factory's failed-call retry rule.
2. Probe a real Console agent send with Capture On and the real `ConsoleTraceBoundaryFactory`, faking only the provider (HTTP or handler level), so the first call fails with a real provider error. Record whether a TOOL_LOOP call follows and whether its reservation is refused.
3. Re-run TASK-42412's forced reproduction with temporary instrumentation of the surface admission, to see what the refused TOOL_LOOP was compared against.
4. Change code only if a real path is refused; otherwise record why the behaviour is right and close.

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
No code change. A real provider error on an agent's first call can be followed by a TOOL_LOOP call, and that call is admitted, not refused. The refusal seen in TASK-42412 came from that test's `def reply(**kwargs)` double for `Chat_Functions.chat_api_call`, the trap TASK-41310 recorded in `lessons-testing-evidence.md`.

**Can a TOOL_LOOP follow a failed AGENT_FIRST (AC#1)? Yes, for providers dispatched through `chat_api_call`.**
- `Agents/agent_runtime.py::_recover_model_failure` retries a transient error in place and returns `NEXT_TURN`. `is_transient_model_error` counts `ChatRateLimitError`, and `ChatProviderError`/`ChatAPIError` with status 408/425/429/5xx/529, as transient. The Console default is `agent_max_model_retries = 2`.
- `Chat/console_streaming_request.py` routes every call after the run's first (`request_count >= 1`) as `TOOL_LOOP`.
- The factory (`console_trace_runtime.py`, `failed_call_retry`) then admits that call against the failed call's durable surface: "a failed call authorizes an unchanged retry". The retry must present the same surface; a changed one is refused.

**Is that call refused with `surface_replacement_checkpoint_unavailable`? No.**
- *The probe.* A temporary probe, deleted after use, ran a real `ConsoleChatController` + `ConsoleAgentBridge` + `ConsoleProviderGateway` agent send with Capture On and the real `ConsoleTraceBoundaryFactory`. It faked only the provider handler (`chat_api_call_fn`), so `Chat_Functions.chat_api_call` kept its real signature. The first call raised a real provider error type, and the second answered.
- *What it showed.* 14 runs, all passed: `ChatRateLimitError` (429), `ChatProviderError` (503), and an unexpected exception that the gateway surfaced as `ChatProviderError` (502); complete and streaming; with no context, with a session system prompt, with project instructions (`AGENTS.md`), and with both. Every run completed. The trace lineage was `agent_first=error` then `tool_loop=complete`, with no reservation error.
- *llama.cpp direct HTTP.* No TOOL_LOOP follows. That path raises raw `httpx` errors (`HTTPStatusError` 429/503, `ConnectError`, `ReadTimeout`), which `is_transient_model_error` does not count as transient. A complete-mode run ends with the provider error (8 runs). Streaming fails over inside the gateway on the `llama_fallback` route instead (8 runs, all completed). This is noted below as a possible follow-up; it is a retry-policy question, not a trace one.

**Why TASK-42412's TOOL_LOOP was refused.**
- *Forced reproduction, instrumented.* I re-ran TASK-42412's forced reproduction (a temporary 8 s sleep in `_read_default_stop`, the stop-path wait removed) with a temporary log of the surface admission. All three edits were reverted.
  - AGENT_FIRST was admitted against an empty surface. Its durable surface was then a single `('provider_overlay', 'omission')` node, not the user's saved revision.
  - The TOOL_LOOP retry's surface (that saved revision) therefore differed at index 0, a one-item replacement. No checkpoint existed, because `current_surface_checkpoint` matches only the route a checkpoint was published under, and AGENT_FIRST's is `agent_first`. So the replacement was refused with `surface_replacement_checkpoint_unavailable`.
- *Why the omission.* The trace verifier projects final values through `inspect.signature(chat_api_call)`. The test's bare `**kwargs` double binds nothing, so AGENT_FIRST's provider bundle was unavailable and the call was stored as an omission (TASK-41310).
- *Confirmation.* The same forced run with `@functools.wraps(Chat_Functions.chat_api_call)` on the double (temporary, reverted) stored AGENT_FIRST as `('message', 'revision')`. Both in-place TOOL_LOOP retries were then admitted (prefix 1, no replacement), and no `trace_reservation` failure was logged. The test still failed there, on `managed_received_prerequisite_timeout`, because the stop-path wait had been removed.

**Intended behaviour (AC#2).**
- A retry after a failed call is reserved against that call's durable surface, as long as the surface is unchanged. A changed surface is refused, by design: "ERROR proves a finished attempt, never authority to add unobserved tool results or replace its request surface". This is pinned by `Tests/Chat/test_console_failed_call_retry_ownership.py::test_failed_call_retry_requires_exact_durable_chain`: `valid`/`cold`/`system_valid`/`updated_system` are admitted, and the `changed_*` and unfinished cases are refused.
- The code matches this, so nothing changed and there is no red test.
- A user would see this refusal only if a first call's bundle were already unavailable (an alignment or sanitizer failure). After such a call, every later TOOL_LOOP is refused whether or not the call failed, so the failure is not the cause.

**AC#3** does not apply: AC#1 found that the path exists. Its condition (the path cannot happen) is false.

Files: this task file, and one sentence on TASK-42412's entry in `backlog/docs/lessons-testing-evidence.md` that names the refusal's real cause.
<!-- SECTION:NOTES:END -->
