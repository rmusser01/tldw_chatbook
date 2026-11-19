---
id: TASK-41963
title: "Small Console test reds: change-tracking ordering margin, scope row on first mount, workspace-tree baseline digest"
status: To Do
assignee: []
created_date: '2026-11-19 07:30'
labels:
- follow-up
- tests
- console
priority: low
references:
- Tests/Chat/test_change_turn_tracking.py
- Tests/UI/test_console_scope_row.py
- Tests/UI/test_console_workspace_tree_performance.py
dependencies:
- TASK-41633
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found during the TASK-41630..41655 follow-up burn-down (2026-11-18/19). Checked on dev d0486978dd unless stated; paths are under `tldw_chatbook/` unless they start with `Tests/` or `backlog/`.

**Why.** From TASK-41633 (notes F7-F9): `test_review_runs_before_baseline_gate_and_tool_dispatch_waits` relies on a 0.5 s sleep margin and fails under load (wait on an event instead); `test_initial_mount_of_restored_persisted_scoped_session_warms_row_and_chip` shows "Scope: everything" on first mount (possible product defect); `test_old_projection_baseline_is_reproducible` fails on a fixture digest mismatch on dev.

**Failing nodes (dev d0486978dd, TASK-41633 runs).**
- `Tests/Chat/test_change_turn_tracking.py::test_review_runs_before_baseline_gate_and_tool_dispatch_waits`
- `Tests/UI/test_console_scope_row.py::test_initial_mount_of_restored_persisted_scoped_session_warms_row_and_chip`
- `Tests/UI/test_console_workspace_tree_performance.py::test_old_projection_baseline_is_reproducible`
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The ordering test waits on an event, not a time margin
- [ ] #2 The scope-row first mount is recorded as drift or fixed as a defect (red first)
- [ ] #3 The baseline digest test passes or its fixture is regenerated with a recorded reason
<!-- AC:END -->
