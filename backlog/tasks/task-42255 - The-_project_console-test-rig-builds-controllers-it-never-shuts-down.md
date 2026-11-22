---
id: TASK-42255
title: "The _project_console test rig builds controllers it never shuts down"
status: To Do
assignee: []
created_date: 2026-11-22 00:17
updated_date: 2026-11-22 00:17
labels:
- follow-up
- tests
- console
priority: low
references:
- Tests/Chat/test_console_project_instruction_traces.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by review of the consoleb stream (round 19). `_project_console` (`Tests/Chat/test_console_project_instruction_traces.py:46`) builds a `ConsoleChatController` and, in its `finally`, only disposes the scratch manager and closes the gateway; the comment there reads "the controller is never shut down". Seven test files use it (`test_console_project_instruction_traces.py`, `test_console_trace_failed_tool_run.py`, `test_console_trace_discarded_tool_run.py`, `test_console_trace_surface_identity.py`, `test_console_trace_model_retries.py`, `test_custom_endpoint_trace_surface.py`, `test_self_hosted_retry_accounting.py`). TASK-42155's controller-shutdown fixture (`fu/r19-consoleb`) does not cover it because the rig builds its own controller.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 `_project_console` shuts its controller down in its `finally` (or the fixture covers it), with its other resources closed after
- [ ] #2 The seven files pass through the scratch wrapper at `-n 4`
- [ ] #3 A run leaves no controller task or thread from the rig, shown by a before-and-after count
<!-- AC:END -->
