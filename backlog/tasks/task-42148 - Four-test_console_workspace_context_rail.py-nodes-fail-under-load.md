---
id: TASK-42148
title: "Four test_console_workspace_context_rail.py nodes fail under load"
status: To Do
assignee: []
created_date: 2026-11-20 20:06
updated_date: 2026-11-20 20:06
labels:
- follow-up
- tests
- console
- flaky
priority: low
references:
- Tests/UI/test_console_workspace_context_rail.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by the library stream (round 18), which saw four nodes fail under load at base with `KeyError: 'workspace-1'` and state-timing assertions. At d85e90bc9e (`-n 4`, through the test wrapper), a whole-file run at load 13-16 failed four nodes: `test_expanded_chats_section_capped_row_marker_surfaces_on_header`, `test_console_f1_tolerates_missing_tree_context_snapshot`, `test_collapsed_chats_section_header_shows_aggregate_unchanged` and `test_combined_menu_change_appearance_opens_picker_with_saved_identity`. A rerun of only those four at load 18-20 failed two: `assert 'Chats' == 'Chats ●'` (line 802) and `assert [] == [{'color': ..., 'icon': ...}]` (line 2440). The `KeyError` did not appear in these runs. TASK-41648 fixed a different `KeyError` in the workspace tree.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The four nodes are run repeatedly under load (at least 10 runs, load recorded), and each failure is attributed to a cause
- [ ] #2 Each cause is fixed with a wait on the product's own ready signal, not a longer pause, and the same repeated run passes
- [ ] #3 For each changed wait, a negative control that removes the behaviour it pins still fails the test
<!-- AC:END -->
