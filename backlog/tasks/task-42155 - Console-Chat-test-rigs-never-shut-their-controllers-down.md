---
id: TASK-42155
title: "Console Chat test rigs never shut their controllers down"
status: To Do
assignee: []
created_date: 2026-11-20 22:23
updated_date: 2026-11-20 22:23
labels:
- follow-up
- tests
- console
priority: low
references:
- Tests/Chat/test_console_chat_controller.py
- Tests/Chat/test_console_trace_discarded_tool_run.py
- Tests/Chat/test_console_prompt_queue_coordinator.py
- Tests/Chat/test_console_rewind_summarize.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
A consolerun follow-up (round 18), checked against the code:
- `Tests/Chat/test_console_chat_controller.py` constructs `ConsoleChatController(` 210 times and awaits `controller.shutdown()` 3 times.
- The rig helpers `_reload_console` (`test_console_trace_discarded_tool_run.py`) and `_durable_controller` (`test_console_prompt_queue_coordinator.py`, `test_console_rewind_summarize.py`) build controllers without shutting them down; a few tests shut theirs.

`ConsoleChatController.shutdown` exists (`Chat/console_chat_controller.py`). Since TASK-42112 the controllers' scratch directories are contained in pytest's basetemp. The controllers' threads, connections and other resources still outlive their tests. That likely feeds the end-of-run hang and fd growth filed separately.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The Chat rig helpers and the controller tests shut down every controller they create, through a fixture or the helper itself, not by hand in each test
- [ ] #2 A check fails when a test leaves a controller running (for example a count of live controllers or their worker threads at teardown), red before the change on at least one test
- [ ] #3 The touched Chat files pass at -n 4 and on their lanes
<!-- AC:END -->
