---
id: TASK-41650
title: "Show a spawned child's trace calls under the parent's turn in the trace view"
status: To Do
assignee: []
created_date: 2026-11-17 19:05
labels:
- follow-up
- trace
- ui
priority: medium
references:
- tldw_chatbook/UI/Screens/trajectory_screen.py
- tldw_chatbook/Chat/console_trace_repository.py
dependencies:
- TASK-41310
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found during the TASK-41306..41320 follow-up burn-down (2026-11-17, merged as #3581..#3593). Checked on dev 97f8c58c0c; line numbers are that tree's, and paths are under `tldw_chatbook/` unless they start with `Tests/` or `backlog/`.

**Why.** TASK-41310 records each spawned child's requests in its own trace segment under the parent's turn (schema v87, `console_trace_child_segments`). The trace view does not show them yet; on 2026-11-17 the owner chose "later, as a follow-up".
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The trace view lists a child run's calls under the parent turn that spawned them
- [ ] #2 A test spawns a child with capture on and asserts that the parent's turn shows the child's calls
<!-- AC:END -->
