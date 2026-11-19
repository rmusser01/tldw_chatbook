---
id: TASK-41508
title: "About 198 Console trace test nodes are red on dev and on no lane: PreparedProviderRequest rejects cache_boundary"
status: To Do
assignee: []
created_date: 2026-11-17 05:27
labels:
- follow-up
- tests
- console
- trace
priority: medium
references:
- tldw_chatbook/Chat/console_prepared_request.py
- Tests/Chat/test_console_project_instruction_traces.py
- Tests/Chat/test_console_trace_failed_tool_run.py
- Tests/Chat/test_console_trace_discarded_tool_run.py
- Tests/Chat/test_console_agent_call_thinking.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by the trace stream (round 15). On dev these fail with one cause, a `TypeError` for `PreparedProviderRequest(cache_boundary=...)`: `cache_boundary` is declared `field(default=None, init=False, repr=False)` (`console_prepared_request.py:708`, added by 7970fbcada), so test code constructing it with the keyword breaks.

**Counts.** `test_console_project_instruction_traces.py` 67, `test_console_trace_failed_tool_run.py` 51, `test_console_trace_discarded_tool_run.py` 31, `test_console_agent_call_thinking.py` 15 failed plus 8 errors. None of the four is on a PR lane.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The four files pass under a plain run, by changing the test builders to the current constructor
- [ ] #2 The four files are on a PR lane, with their time recorded
- [ ] #3 No `init=False` field is added or removed to make tests pass without a decision recorded in the notes
<!-- AC:END -->
