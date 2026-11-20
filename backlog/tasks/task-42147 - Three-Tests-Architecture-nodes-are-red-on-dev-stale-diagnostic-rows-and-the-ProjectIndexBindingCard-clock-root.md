---
id: TASK-42147
title: "Three Tests/Architecture nodes are red on dev: stale diagnostic rows and the ProjectIndexBindingCard clock root"
status: To Do
assignee: []
created_date: 2026-11-20 20:06
updated_date: 2026-11-20 20:33
labels:
- follow-up
- tests
- architecture
priority: low
references:
- Tests/Architecture/test_persistent_diagnostic_inventory.py
- Tests/Architecture/test_timer_path_static_update_inventory.py
- tldw_chatbook/Widgets/Settings_Widgets/project_index_binding_card.py
- tldw_chatbook/Agents/agent_service.py
- tldw_chatbook/Chat/console_store_first_persistence.py
- tldw_chatbook/Agents/agent_native_execution.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by the library stream (round 18). TASK-41307, TASK-41320, TASK-41637 and TASK-41640 recorded these as known base failures since at least fbf1c9ea74, and no task owns them. At d85e90bc9e (serial, through the test wrapper):
- `test_persistent_diagnostic_inventory.py::test_reviewed_diagnostic_changes_are_metadata_only` (line 620): two rows of `REVIEWED_METADATA_ONLY_DIAGNOSTICS` match nothing. `tldw_chatbook/Agents/agent_service.py` has no diagnostic containing "on_child_settled consumer raised" (6125fbc5b8 moved it to `Agents/agent_native_execution.py:1515`), and `Chat/console_chat_store.py` has none containing "Failed to flush Console roleplay context on first persist" (that message is now in `Chat/console_store_first_persistence.py`).
- `test_timer_path_static_update_inventory.py::test_clock_root_set_is_pinned` (line 2824): a new repeating-clock root, `set_interval` to `ProjectIndexBindingCard._poll` (`Widgets/Settings_Widgets/project_index_binding_card.py`).
- `test_timer_path_static_update_inventory.py::test_no_timer_path_update_defaults_to_layout_true` (line 2691): `.update(` calls reached from that root do not pass `layout=` (`_paint` line 167, `_feedback` line 194, `_launch` and `run` lines 356 and 365).

The card came with 2e9018b86d ("manage optional workspace keyword indexes").
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The two reviewed diagnostic rows name the diagnostics' current owners, or are removed with the reason, and `test_reviewed_diagnostic_changes_are_metadata_only` passes
- [ ] #2 `ProjectIndexBindingCard._poll` is in the pinned clock roots, and each `.update(` it reaches either passes `layout=` (proved as the file's message requires) or has a `CLASSIFIED_SITES` row saying why it need not
- [ ] #3 `test_persistent_diagnostic_inventory.py` and `test_timer_path_static_update_inventory.py` pass in a serial run
- [ ] #4 These three nodes run in a PR lane (no PR lane runs them today), so a later drift fails CI
<!-- AC:END -->
