---
id: TASK-39130
title: "Six Library UI tests are red on dev and run on no lane: an entry canvas sync, two Media side-by-side cases, the Notes export key and two persisted reader reads"
status: To Do
created_date: 2026-11-07 02:03
labels:
- follow-up
- tests
- library
- dev-red
priority: low
references:
- Tests/UI/test_library_canvas_sync_defects.py
- Tests/UI/test_library_media_side_by_side.py
- Tests/UI/test_library_notes_w5_kbd_focus.py
- Tests/UI/test_library_media_reader_shell.py
dependencies:
- TASK-39050
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in round 11 of the 2026-10-23 burn-down; checked on dev 2684e5453f, 2026-11-07. TASK-39050's notes (round-11 branch fu/r11-library) mention 13 Library UI reds but list no node ids.

**The run.** TASK-39050's file set on dev at `-n 4` used a `bootstrap_profile` marker plugin, so the RecoveryRequired trap does not hide the assertions. It gave 12 failed, 134 passed (load 32-47); a plain rerun of the 12 also failed.
- Six of the 12 belong to TASK-32185 (In Progress, AC#4): the Notes authority round-trip and switch nodes and `test_wide_files_task_return_restores_database_browse_receipt`.
- Two nodes the notes count are laned and pass: `test_high_stakes_file_notes_states_are_legible_in_shipped_themes` and `test_library_database_files_switch_retains_workspace_and_database_canvas`.

**The six reds no task owns.** None of them is in the census or any workflow.
- `test_library_canvas_sync_defects.py::test_entry_canvas_sync_does_not_focus_an_unrelated_replacement_row` (`:690`, `'chat-replacement' != 'chat-replacement'`).
- `test_library_media_side_by_side.py::test_compact_media_viewer_back_empty_page_focuses_recovery_control` (`NoMatches '#library-media-back'`). TASK-34000.26 noted it as inherited.
- `test_library_media_side_by_side.py::test_media_trash_collapse_choices_survive_filter_and_page_refresh` (`:831`, `46 > 46`).
- `test_library_notes_w5_kbd_focus.py::test_e_runs_export_selected_only_once_something_is_selected` (`:372`).
- `test_library_media_reader_shell.py::test_persisted_shared_library_read_honors_real_legacy_config` (`:1371`, `None is False`).
- `test_library_media_reader_shell.py::test_persisted_items_read_uses_default_when_key_is_absent` (`:1386`, `None is True`).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Each of the six nodes passes on dev, or is retired with the commit that changed the behaviour it pinned named in the notes; each red is shown first with the same command
- [ ] #2 Each node is classified as test drift or a product defect; a product defect gets a test that is red before its fix
- [ ] #3 The six nodes run on a PR lane (UI census tail entries of about 30 s each), with measured times recorded; never the fast PR contract
<!-- AC:END -->
