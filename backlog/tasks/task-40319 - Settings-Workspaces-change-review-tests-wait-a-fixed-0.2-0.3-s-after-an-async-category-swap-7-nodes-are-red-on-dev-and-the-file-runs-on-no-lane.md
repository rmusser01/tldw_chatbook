---
id: TASK-40319
title: "Settings Workspaces change-review tests wait a fixed 0.2-0.3 s after an async category swap: 7 nodes are red on dev and the file runs on no lane"
status: To Do
created_date: 2026-11-12 19:36
labels:
- follow-up
- tests
- settings
- workspaces
- flaky
priority: low
references:
- Tests/UI/test_settings_workspaces_category.py
- tldw_chatbook/UI/Screens/settings_screen.py
dependencies:
- TASK-39169
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in round 13 of the 2026-10-23 burn-down (cprod stream review r13p-rev1: red at base ae9f9e8d21 and at the cprod head). Checked by reading the code at the round-13 stack top 8b633d0101, 2026-11-12; line numbers are that tree's.

**Why.** Each change-review step presses a row that starts an async same-category swap (`_swap_category_panes` under `_category_swap_lock`, since 4fe45538f5), then waits a fixed `pause(0.2)` or `pause(0.3)` and asserts. Under load the swap has not finished. `test_change_review_unavailable_consent_has_no_toggle` was flaky at base already (TASK-39169 notes).

**Nodes** (`Tests/UI/test_settings_workspaces_category.py`): the six `test_change_review_*` tests and `test_folder_bindings_add_toggle_remove_and_inline_errors`. Fixed pauses at `:392-427`, `:548`, `:559`, `:563`, `:584`, `:591`, `:623`, `:667`, `:673`, `:709`, `:737`, `:765`. The file runs on no lane.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Each change-review and folder-binding step waits for the swap and the state it asserts within a deadline instead of a fixed pause
- [ ] #2 The file passes 3 of 3 whole-file runs at `-n 4`, and runs on a PR lane
<!-- AC:END -->
