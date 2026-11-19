---
id: TASK-41652
title: "No test covers prompt mix-in binding for a spawned child run"
status: To Do
assignee: []
created_date: 2026-11-17 19:05
labels:
- follow-up
- agents
- tests
priority: low
references:
- Tests/Chat/test_console_spawn_subagent_trace_capture.py
dependencies:
- TASK-41310
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found during the TASK-41306..41320 follow-up burn-down (2026-11-17, merged as #3581..#3593). Checked on dev 97f8c58c0c; line numbers are that tree's, and paths are under `tldw_chatbook/` unless they start with `Tests/` or `backlog/`.

**Why.** ADR-286 binds operational prompt mix-ins to the conversation (schema v86). TASK-41310's child segments exercise the parent and child routes, but no test says whether a spawned child gets the parent conversation's mix-ins.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The intended behaviour (child inherits or does not) is stated in ADR-286 or this task's notes
- [ ] #2 A test spawns a child from a conversation with a mix-in and asserts that behaviour
<!-- AC:END -->
