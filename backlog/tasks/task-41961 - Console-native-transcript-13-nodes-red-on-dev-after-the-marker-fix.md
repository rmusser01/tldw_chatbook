---
id: TASK-41961
title: "Console native transcript: 13 nodes red on dev after the marker fix"
status: To Do
assignee: []
created_date: '2026-11-19 07:30'
labels:
- follow-up
- tests
- console
priority: low
references:
- Tests/UI/test_console_native_transcript.py
dependencies:
- TASK-41633
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found during the TASK-41630..41655 follow-up burn-down (2026-11-18/19). Checked on dev d0486978dd unless stated; paths are under `tldw_chatbook/` unless they start with `Tests/` or `backlog/`.

**Why.** TASK-41633's marker cleared most of `Tests/UI/test_console_native_transcript.py`; 13 nodes remain (More menu, tab traversal, action-row fit, click-to-clear, tool-status refresh, memory banner; TASK-41633 notes F5).

**Failing nodes (dev d0486978dd, TASK-41633 runs).**
- `Tests/UI/test_console_native_transcript.py::test_activity_header_click_and_o_share_expansion_state`
- `Tests/UI/test_console_native_transcript.py::test_assistant_turn_reconcile_preserves_container_and_answer_identity`
- `Tests/UI/test_console_native_transcript.py::test_console_more_choice_keeps_captured_target_after_selection_race`
- `Tests/UI/test_console_native_transcript.py::test_console_more_tab_traversal_stays_inside_menu_at_80_columns`
- `Tests/UI/test_console_native_transcript.py::test_console_selected_action_rows_fit_reference_terminals[message3-expected_actions3-size0]`
- `Tests/UI/test_console_native_transcript.py::test_console_selected_action_rows_fit_reference_terminals[message3-expected_actions3-size1]`
- `Tests/UI/test_console_native_transcript.py::test_console_selected_action_rows_fit_reference_terminals[message3-expected_actions3-size2]`
- `Tests/UI/test_console_native_transcript.py::test_console_tab_reaches_major_console_screen_regions`
- `Tests/UI/test_console_native_transcript.py::test_console_transcript_click_background_clears_selection`
- `Tests/UI/test_console_native_transcript.py::test_console_transcript_failed_action_row_includes_retry_without_exceeding_budget`
- `Tests/UI/test_console_native_transcript.py::test_console_transcript_more_menu_captures_message_and_closes_before_choice`
- `Tests/UI/test_console_native_transcript.py::test_mounted_console_refreshes_tool_status_after_assistant_preamble`
- `Tests/UI/test_console_native_transcript.py::test_mounted_console_sync_replaces_effective_memory_banner_without_stacking`
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Each node has a recorded cause and passes after repair or fix
<!-- AC:END -->
