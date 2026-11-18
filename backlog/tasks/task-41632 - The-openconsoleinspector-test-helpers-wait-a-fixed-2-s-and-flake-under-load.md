---
id: TASK-41632
title: "The `_open_console_inspector` test helpers wait a fixed 2 s and flake under load"
status: To Do
assignee: []
created_date: 2026-11-17 19:05
labels:
- follow-up
- tests
- flaky
- console
priority: low
references:
- Tests/UI/test_console_internals_decomposition.py
- Tests/UI/test_console_scope_row.py
dependencies:
- TASK-41308
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found during the TASK-41306..41320 follow-up burn-down (2026-11-17, merged as #3581..#3593). Checked on dev 97f8c58c0c; line numbers are that tree's, and paths are under `tldw_chatbook/` unless they start with `Tests/` or `backlog/`.

**Why.** TASK-41308's MCP-row tests open the Inspector through `_open_console_inspector` (`Tests/UI/test_console_internals_decomposition.py:275`; a copy is at `Tests/UI/test_console_scope_row.py:141`). The helper gives the rail a fixed 2 s deadline, and it timed out in local runs at load 50-100 when the same tests passed alone.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Both helpers wait on the Inspector's visible condition with a deadline that tolerates a loaded runner (or the shared wait helper), not a fixed 2 s
- [ ] #2 The MCP-row tests pass in a `-n 4` run of their files
<!-- AC:END -->
