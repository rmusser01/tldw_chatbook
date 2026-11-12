---
id: TASK-39130
title: "Six Library UI tests are red on dev and run on no lane: an entry canvas sync, two Media side-by-side cases, the Notes export key and two persisted reader reads"
status: In Progress
assignee:
- '@claude'
updated_date: '2026-11-10 17:25'
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
- [x] #2 Each node is classified as test drift or a product defect; a product defect gets a test that is red before its fix
- [ ] #3 The six nodes run on a PR lane (UI census tail entries of about 30 s each), with measured times recorded; never the fast PR contract
<!-- AC:END -->

## Implementation Plan

1. Run the six nodes plain on base: four die with RecoveryRequired through `_build_test_app`; the two persisted reads swallow the same error into `None`.
2. Give the four mounted nodes a node-level `bootstrap_profile` marker; turn the two persisted reads into `@private_profile_test` cases that write their own child profile's config (TASK-39141's precedent).
3. Read each remaining failure, classify it (test drift with the commit that moved it, or product defect), re-pin drift, and fix a product defect behind a test shown red first.
4. Append the passing nodes at the UI census tail with measured times.

## Implementation Notes

**Status: 5 of 6 pass and are laned; the sixth is a product defect left for a follow-up (AC#1 and AC#3 unticked).**

**Red first, on base ae9f9e8d21** (run.sh, the six nodes plain at `-n 4`, load 69-79): 6 failed. Four died with `RecoveryRequired: raw_source_selection_changed` through `_build_test_app` -> `load_settings()`; the two persisted reads failed `None is False` / `None is True`. With node-level `bootstrap_profile` markers the four then failed on their own assertions, as the description lists.

**Per node** (each green alone afterwards, run.sh, one process):
- `test_library_canvas_sync_defects.py::test_entry_canvas_sync_does_not_focus_an_unrelated_replacement_row` -- **test drift.** 10711f19b9 (2026-09-10, task-32228, "arrive with the list focused, like every sibling") arms the Conversations list's entry focus on arrival: for its 2 s window every recompose re-focuses the first row. The test focused row 0 inside that window, so the arm (not the semantic restore it pins) put focus on the replacement. Re-pin: wait until `_library_pending_list_entry_focus` clears before the test's own focus. Negative control: `_restore_library_entry_focus` falling back to the widget id when the source row is gone (temporary, reverted) turns it red with the original `'chat-replacement' != 'chat-replacement'`.
- `test_library_media_side_by_side.py::test_media_trash_collapse_choices_survive_filter_and_page_refresh` -- **test drift.** 9187bc0307 (2026-09-07, task-31979, "widen Items list when the Reader has no item open") landed a day after task-31633's pin (b4497cdebe). The Trash Reader holds no item, so the Items list takes the freed width down to the Reader's floor: Items 66 -> 100, Reader 46 -> 46. Re-pin: Items grows, the Reader keeps its floor.
- `test_library_notes_w5_kbd_focus.py::test_e_runs_export_selected_only_once_something_is_selected` -- **product defect, fixed.** Select all and Clear had no portable Notes focus role, so their own press (which rebuilds the Notes canvas) restored the navigator fallback: focus jumped to the filter, where `e` types instead of exporting. That is the shape TASK-34000.28 fixed for Sort. Fix: `select-all` / `select-clear` roles in `_library_notes_semantic_role` and `_library_notes_role_target`. The test now follows the user's path (Select all focused, then pressed; then the same for Clear) and asserts focus stays on the rebuilt control. Red without the roles (focus `library-notes-filter`, both halves shown separately), green with them.
- `test_library_media_reader_shell.py::test_persisted_shared_library_read_honors_real_legacy_config` and `::test_persisted_items_read_uses_default_when_key_is_absent` -- **test drift.** They redirected `TLDW_CONFIG_PATH`, which the bound config source has refused since b5251e9a6e (2026-09-16, TASK-32628 raw-source admission): `read_cli_config_serialized()` raises `RecoveryRequired` (shown by a temporary direct call), and the reader's broad except returned `None`. Now `@private_profile_test` cases that write their own child profile's config, TASK-39141's precedent. Negative control: `library_open = true` in the legacy table (temporary) fails `True is False`.
- `test_library_media_side_by_side.py::test_compact_media_viewer_back_empty_page_focuses_recovery_control` -- **product defect, NOT fixed (proposed follow-up).** Since 80e4cd7056 (2026-09-08, task-32043 + task-32086) an exact empty page resets the Reader and returns the compact canvas to its list by itself, so `#library-media-back` no longer exists (the old `NoMatches`). The return's whole-screen recompose leaves focus at `None` (measured: focus `library-media-row-15` before the empty result, `None` after and 0.5 s later), so the keyboard has no landing. The test is re-pinned to the shipped flow (no Back to press) and stays red on the defect: "did not focus the Import recovery action; focus is None". A fix belongs in the task-32086 reset branch of the browse apply (arm the list entry focus when the recompose strands focus); it touches the entry-focus arm, so it is left to a follow-up rather than guessed here.

**Profiles.** The four mounted nodes carry node-level `@pytest.mark.bootstrap_profile` (their `_build_test_app` reads config); the two persisted reads are private-profile children.

**Covering runs.** Size ratchets: `Tests/Architecture/test_screen_size_ratchet.py`, `test_library_modules_size_ratchet.py`, `Tests/UI/test_library_recompose_ratchet.py` 61 passed after raising `library_screen.py` 38161 -> 38164 and `library_notes_controller.py` 7423 -> 7425 (notes in both ratchets). `Tests/Architecture/test_library_notes_wiring.py` 15 passed. The four Notes suites that pin focus roles plus the w5 file, under a scratch plugin stamping `bootstrap_profile` (deleted): 11 failed / 155 passed on head; the same 11 fail with the same messages with the role change reverted, so none is this change's.

**Lane (AC#3, five of six).** The five passing nodes are appended at the UI census tail (floor 1913 -> 1919 with TASK-39153's rail minors file). Serial times, three runs each (run.sh, the lane's flags, load 52-90): 23.9-42.8 s, 30.1-40.3 s, 22.3-22.5 s, 21.6-42.0 s, 23.3-44.5 s. `scripts/ui_lane_shard_replay.py --shards 36` over the 22 saved round-12 runs, each new entry charged at its slowest local time: max shard 12.3 -> 13.1 pytest min, under the 13.3 the header sets.

**Files.** `Tests/UI/test_library_canvas_sync_defects.py`, `Tests/UI/test_library_media_side_by_side.py`, `Tests/UI/test_library_notes_w5_kbd_focus.py`, `Tests/UI/test_library_media_reader_shell.py`, `tldw_chatbook/UI/Screens/library_screen.py`, `tldw_chatbook/UI/Library_Modules/library_notes_controller.py`, `Tests/Architecture/test_screen_size_ratchet.py`, `Tests/Architecture/test_library_modules_size_ratchet.py`, `scripts/ui_pr_gate_census.txt`, `scripts/check_ui_pr_gate_census.py`, `Docs/User_Guide/library/notes.md` (Select all and Clear keep focus).

