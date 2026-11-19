---
id: TASK-31202
title: settings_screen.py needs a size-ratchet budget row
status: Done
assignee: []
created_date: '2026-09-02 19:25'
updated_date: '2026-10-06 06:59'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Spec 2026-09-01 non-goal follow-up: the ratchet that let library_screen triple also has no settings row; settings_screen.py was 15,922 lines at the 2026-08-02 doctrine baseline.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Budget row added at measured values
- [x] #2 Mutation-checked (dummy method -> fails)
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Closed by TASK-33007.8 (model-config Phase 7): Tests/Architecture/test_module_size_ratchet.py pins tldw_chatbook/UI/Screens/settings_screen.py at its measured 33,607 lines. Mutation-checked: a 4-line dummy method on SettingsScreen made test_module_does_not_grow_past_its_budget fail ("grew to 33611 lines (budget 33607, +4)"); restored. The rest of the split stays with task-1378.
<!-- SECTION:NOTES:END -->
