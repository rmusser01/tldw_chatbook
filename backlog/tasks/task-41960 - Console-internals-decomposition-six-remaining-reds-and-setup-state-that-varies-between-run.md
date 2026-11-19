---
id: TASK-41960
title: "Console internals decomposition: six remaining reds and setup state that varies between runs"
status: To Do
assignee: []
created_date: '2026-11-19 07:30'
labels:
- follow-up
- tests
- console
- flaky
priority: medium
references:
- Tests/UI/test_console_internals_decomposition.py
dependencies:
- TASK-41633
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found during the TASK-41630..41655 follow-up burn-down (2026-11-18/19). Checked on dev d0486978dd unless stated; paths are under `tldw_chatbook/` unless they start with `Tests/` or `backlog/`.

**Why.** After TASK-41633's module marker, 6 nodes stay red (composer cursor blink/show/hide, staged-context room, pending-approval actions, setup tooltip; TASK-41633 notes F4), and 7 more fail intermittently (F12): e.g. `test_console_composer_empty_setup_blocked_state_shows_reason` sees "add an API key" where it expects "choose a model", so the setup state differs between runs, which suggests state leaking under the shared bootstrap profile.

**Failing nodes (dev d0486978dd, TASK-41633 runs).**
- `Tests/UI/test_console_internals_decomposition.py::test_console_composer_cursor_blink_toggles`
- `Tests/UI/test_console_internals_decomposition.py::test_console_composer_shows_cursor_when_focused`
- `Tests/UI/test_console_internals_decomposition.py::test_console_non_empty_staged_context_keeps_room_for_source_details`
- `Tests/UI/test_console_internals_decomposition.py::test_console_run_inspector_exposes_pending_approval_and_chatbook_artifact_actions`
- `Tests/UI/test_console_internals_decomposition.py::test_console_composer_hides_cursor_when_blurred`
- `Tests/UI/test_console_internals_decomposition.py::test_console_send_without_ready_runtime_keeps_setup_block_tooltip_only`
- `Tests/UI/test_console_internals_decomposition.py::test_console_composer_empty_setup_blocked_state_shows_reason`
- `Tests/UI/test_console_internals_decomposition.py::test_console_composer_active_run_blocker_keeps_generic_placeholder`
- `Tests/UI/test_console_internals_decomposition.py::test_console_large_paste_collapse_can_be_disabled_from_config[False]`
- `Tests/UI/test_console_internals_decomposition.py::test_console_composer_ranks_actions_by_current_availability`
- `Tests/UI/test_console_internals_decomposition.py::test_console_stop_button_tooltip_scopes_to_this_tab`
- `Tests/UI/test_console_internals_decomposition.py::test_console_stop_button_hidden_unless_streaming`
- `Tests/UI/test_console_internals_decomposition.py::test_console_control_labels_refresh_after_compact_control_sync`
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The run-to-run setup-state difference has a recorded cause and is fixed (isolation or product)
- [ ] #2 The six remaining nodes have recorded causes and pass
<!-- AC:END -->
