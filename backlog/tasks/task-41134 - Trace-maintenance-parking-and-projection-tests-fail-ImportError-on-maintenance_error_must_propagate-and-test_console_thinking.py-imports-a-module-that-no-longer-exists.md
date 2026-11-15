---
id: TASK-41134
title: "Trace maintenance parking and projection tests fail ImportError on maintenance_error_must_propagate, and test_console_thinking.py imports a module that no longer exists"
status: To Do
created_date: 2026-11-15 02:20
labels:
- follow-up
- console
- trace
- tests
- dev-red
priority: medium
references:
- Tests/Chat/test_console_trace_maintenance_parking.py
- Tests/Chat/test_console_trace_projection.py
- Tests/Chat/test_console_thinking.py
- tldw_chatbook/Chat/console_trace_maintenance_runtime.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Filed in round 14 of the 2026-10-23 burn-down (closeout, 2026-11-15). Checked against the round-14 stack top c7d4a41717 (on dev beab5c1e4e); line numbers are that tree's.

**Why.** The legacy trace maintenance loop's parking and projection runtime have no green tests, and a whole test file cannot be collected.

**Evidence.** Through run.sh at `-n 4`: `test_console_trace_maintenance_parking.py`, `test_console_trace_projection.py`, `test_trace_event_projection.py` and `test_console_thinking.py` give 11 failed, 74 passed, 1 error.
- The failures log `ImportError: cannot import name 'maintenance_error_must_propagate' from 'tldw_chatbook.Chat.console_trace_maintenance' (unknown location)` from `console_trace_maintenance_runtime.py:17`, inside `ConsoleRuntime._schedule_legacy_trace_maintenance`. The function exists (`console_trace_maintenance.py:2186`); "(unknown location)" suggests the tests replace that module with a stub that lacks it. TASK-40303's notes recorded the same 12 at base 26b2e677dd.
- The error is collection: `test_console_thinking.py:5` imports `tldw_chatbook.Chat.console_thinking`, which is not in the tree.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Each failing node passes or is retired with the commit that changed what it pinned named; the notes say whether the import failure can happen outside tests
- [ ] #2 test_console_thinking.py is retired or pointed at the module that now owns thinking display
- [ ] #3 The fixed files run on a PR lane
<!-- AC:END -->
