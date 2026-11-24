---
id: TASK-42530
title: "A captured llama.cpp tool call that arrives on the non-streaming fallback is refused at the next call"
status: To Do
assignee: []
created_date: 2026-11-24 09:14
updated_date: 2026-11-24 09:14
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
- [ ] #1 A Capture-On llama.cpp agent run whose stream is empty and whose non-streaming fallback replies with a tool call runs the tool and completes, and the trace admits the next call; a test through the real llama.cpp harness is red before the fix
- [ ] #2 The user's next message after such a run is sent
- [ ] #3 After a completed fallback, a TOOL_LOOP call is admitted only with the fallback's reply and its tool results appended, never as an unchanged retry
<!-- AC:END -->
