---
id: TASK-41505
title: "Single pilot.pause after a press races the handler under load in several Library, Media and navigation tests"
status: To Do
assignee: []
created_date: 2026-11-17 05:27
labels:
- follow-up
- tests
- flaky
priority: medium
references:
- Tests/UI/test_library_collections_actions.py
- Tests/UI/test_screen_navigation.py
- Tests/Library/test_library_ingest_runner.py
dependencies:
- TASK-41127
- TASK-41126
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Each of these waits one `pilot.pause()` after `Button.press` (or after a flag) and fails at load about 65-110 while passing at low load. Found in round 15:
- libcoll: `test_quick_capture_refresh_hard_delete_cancel_and_legacy_recovery_close` (characterization test), and the open-original check (the handler runs on a background thread);
- libtests review: `test_analyze_bulk_action_follows_the_selection_in_place` and `test_pressing_analyze_leaves_select_mode_and_paints_its_receipt` (render_fixes, off lane), `test_info_describes_the_same_stored_text_representation_sent_to_console` and other press-after-flag media nodes;
- libtests: existing census nodes flaky at load about 100: `test_navigation_keypress_during_splash_is_safely_ignored`, `test_boot_with_mcp_default_tab_lands_on_settings_tools_hub`;
- endpoint fix r1: `test_actual_handler_review_wake_waits_for_pinned_capacity_and_finishes` failed 2 of 2 at load 65-110 but 4 of 4 passed at load 8-40 (timing, unproven);
- `Tests/Workspaces/test_native_confirmed_undo.py` fails at base and head on different nodes each run ("timed out waiting for ..." and `'after\n' == 'before\n'`); TASK-40716 only notes it.
- closeout final checks (2026-11-17, load about 40): `Tests/Library/test_library_ingest_runner.py::test_explicit_parakeet_directory_is_snapshotted_before_dispatch` failed once in the serial recovered-suites step; it waits one `pilot.pause()` after `submit_library_ingest_job` and then asserts the executor was called. It passed alone and in two whole-file runs at load 32-46, and no round-15 change touches the ingest runner or its tests.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Each listed test waits on its observable condition (a state predicate with a bounded deadline), not on a fixed pause count
- [ ] #2 Each passes 10 of 10 at `-n 4` with the load recorded in the Implementation Notes
- [ ] #3 `test_actual_handler_review_wake_waits_for_pinned_capacity_and_finishes` has its cause identified, as timing or as a defect
- [ ] #4 `Tests/Workspaces/test_native_confirmed_undo.py` passes 10 of 10 at `-n 4`, with the cause of each of its two failure shapes named
<!-- AC:END -->
