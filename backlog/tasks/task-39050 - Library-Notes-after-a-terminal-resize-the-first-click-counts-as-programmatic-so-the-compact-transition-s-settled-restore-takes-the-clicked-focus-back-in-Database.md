---
id: TASK-39050
title: "Library Notes: after a terminal resize the first click counts as programmatic, so the compact transition's settled restore takes the clicked focus back in Database Notes and Folder files"
status: Done
assignee:
- '@claude'
created_date: 2026-11-06 10:09
updated_date: 2026-11-06 16:13
labels:
- follow-up
- library
- notes
- focus
priority: low
references:
- tldw_chatbook/UI/Screens/library_screen.py
- tldw_chatbook/UI/Library_Modules/library_notes_controller.py
- Tests/UI/test_library_file_notes_workspace.py
- Tests/UI/test_library_shell.py
dependencies:
- TASK-38957
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in round 10 of the 2026-10-23 burn-down; checked on dev b2c618185f, 2026-11-06, and on the round-10 Library branch that lands TASK-38957 (not yet a pushed, final head). TASK-38957's notes record the click case as a follow-up. Line numbers are dev's; the branch's change is named by symbol.

**Why a click is read as programmatic.**
- `on_resize` sets `resize_settling = True` (`library_screen.py:8452`).
- Only real input clears it: `_mark_library_notes_user_interaction` (`library_notes_controller.py:3018-3020`). That runs from `on_mouse_down` (`library_screen.py:8560`), `on_key`, the wheel, and `action_focus_next`/`previous` (`:9422-9450`, TASK-32106's keyboard fix).
- `on_descendant_focus` counts `resize_settling` as programmatic (`:11203-11208`). The focus generation only advances when `relevant and not programmatic` (`:11251-11255`).
- Textual moves focus before it forwards the MouseDown. So for a click the screen sees DescendantFocus first, while `resize_settling` is still True, and the generation stays put.
- The settled restore's guard (`library_notes_controller.py:2768-2777`, `library_screen.py:5717-5735`) then still passes, and the restore moves focus back.
- TASK-38957 only widens `relevant` (`:11240-11248`) to Folder files, with `or self._file_notes_active()`. The `programmatic` term is shared by every Notes mode and is unchanged.

**Probe.** The probe held `_restore_library_notes_settled_focus`, resized across the compact breakpoint, acted, then ran the held restore. `resize_settling` was True before every action. The holds make it deterministic: 3 runs at `-n 4` gave the same result (load 18-20).

| Mode | Action | Dev | After TASK-38957 |
|---|---|---|---|
| Folder files (160x45 -> 79x30) | Tab | focus taken back (TASK-38957's bug) | kept; generation 1 -> 5 |
| Folder files | click the editor | taken back to `library-file-notes-items-grip` | taken back; generation 1 -> 1 |
| Database Notes (170x48 -> 100x30) | Shift+Tab | kept | kept |
| Database Notes | click the title | taken back to `library-note-body` | taken back; generation 1 -> 1 |

- No test covers a click. Tests that simulate a user focus call `_mark_library_notes_user_interaction()` before `.focus()` (for example `Tests/UI/test_library_shell.py:32410`), which skips this path.
- Media's resize focus code reads the same generation and settling flag (`library_screen.py:7719-7723`, `:7905-7915`). It was not probed.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 After a resize across the compact breakpoint, a `pilot.click` that moves focus between the transition's two restores advances the focus generation and keeps the clicked control, in Database Notes and in Folder files; one test per mode, each red before the fix (generation 1 -> 1, focus back on `library-note-body` / `library-file-notes-items-grip`)
- [x] #2 Programmatic landings during settling are still classified programmatic (the restore's own target, the restoring flag, Textual's focus reset after a prune): TASK-38957's Tab test and the Notes focus and canvas-sync pins stay green, named in the notes
- [x] #3 Media's and Conversations' resize-focus pins stay green; if Media shows the same click case, it is fixed and tested too, or recorded as out of scope with the probe result
- [x] #4 The new tests run on a PR lane (a census tail entry) and pass 20 of 20 at `-n 4`, with the load recorded
<!-- AC:END -->

## Implementation Plan

1. Write two mounted tests, Database Notes (170x48 -> 100x30, click the title) and Folder files (160x45 -> 79x30, click the editor): hold the settled restore, `pilot.click` between the two restores, run the held restore, and assert the generation advanced and the clicked control kept focus. Show both red on the base.
2. Fix the classification at its source, with no timer: Textual resolves a click's focus target through the public `Screen.get_focusable_widget_at`, which it calls only while forwarding a MouseDown, just before `set_focus`. `LibraryScreen` marks the user interaction there, so the click's `DescendantFocus` (which reaches the screen before the MouseDown does) no longer reads `resize_settling`.
3. AC#2: run TASK-38957's Tab test and the Notes focus and canvas-sync pins (under a temporary private-profile marker where `RecoveryRequired` blocks them locally).
4. AC#3: run Media's and Conversations' resize-focus pins, and probe Media's click case.
5. Append the two nodes at the census tail; 20 of 20 at `-n 4` with the load; re-pin the screen size ratchet under the owner waiver; bootbudget.

## Implementation Notes

- **Cause, confirmed.** Textual's `Screen._forward_event` focuses the clicked widget first, and only then forwards the MouseDown. The click's `DescendantFocus` therefore reaches the screen before `on_mouse_down` clears `resize_settling`. `on_descendant_focus` read the click as programmatic, so the focus generation stayed put and the settled restore's guard still passed.
- **Fix, at the source and without a timer.** `LibraryScreen` overrides the public `Screen.get_focusable_widget_at`. Textual 8.2.8 calls it only from that MouseDown branch, synchronously and just before `set_focus`. The override calls `_mark_library_notes_user_interaction()` there and returns the stock result. The click is now marked as real input before its focus exists, so the click's `DescendantFocus` is classified as the user's own move in every Notes mode.
  - `on_descendant_focus` and the restore guards are unchanged.
  - A programmatic landing during settling (the restore's own target, the restoring flag, Textual's reset after a prune) has no MouseDown, so it stays programmatic.
  - No Textual private API is overridden.
- **Tests (AC#1),** both in `Tests/UI/test_library_file_notes_workspace.py`, the file conftest already keeps on the bootstrap profile. Each holds `_restore_library_notes_settled_focus`, resizes across the breakpoint, asserts `resize_settling` is still set, `pilot.click`s a control, asserts the generation advanced, and then runs the held restore.
  - `test_database_notes_click_between_compact_restores_is_kept`: 170x48 -> 100x30, click the title.
  - `test_folder_files_click_between_compact_restores_is_kept`: 160x45 -> 79x30, click the editor.
  - **Red** on the base source: `assert 1 > 1` (generation 1 -> 1) in both. With that assertion temporarily off, the held restore took focus back to `library-note-body` and to `library-file-notes-items-grip`, the task's probe results. **Green** with the fix.
- **AC#2.** TASK-38957's `test_folder_files_user_focus_between_compact_restores_is_kept` passes. I ran the Notes focus and canvas-sync files, plus the Media and Conversations files below, under a scratch `-p` plugin that adds `bootstrap_profile`, because of the local `RecoveryRequired` trap. The files: canvas-sync defects, canvas-sync gaps, removed-widget focus and its extended file, w5 keyboard focus, and the whole File Notes workspace file. Results:
  - 328 passed and 13 failed, at load 47. TASK-38957's four named pins (`test_converted_notes_site_keeps_focus_on_a_real_key_press`, `test_notes_row_press_to_editor_keeps_focus_inside_the_canvas`, `test_automatic_notes_reconcile_restores_richer_control_focus`, `test_overlapping_reconciles_retain_original_semantic_focus`) are among the passes.
  - Rerun with the fix reverted, the same 12 of the 13 fail. The 13th, a Conversations filter case, was a load flake.
  - All 13 also fail on plain dev 2684e5453f (`fu-base-r11`), with the same command and no plugin.
  - None of the 13 is in any PR lane. They are pre-existing, not this change.
- **AC#3.** The Media and Conversations resize-focus pins pass: `test_library_media_side_by_side.py` (including `test_media_resize_focus_restore_yields_to_newer_user_focus` and `test_media_row_focus_moves_to_items_grip_when_resize_hides_items`), `test_library_conversations_continuity.py` (`test_conversation_find_and_new_focus_survive_resize`) and `test_library_media_reader_shell.py`. The only failures are the dev reds listed above.
  - **Media probe** (a temporary test, deleted): hold `_focus_library_media_grip_if_current`, resize 170x48 -> 80x24, click a visible control, then run the held transfer. At 80x24 the visible chain held only the top-bar buttons and the two grips, so the probe clicked `nav-home`.
  - The clicked control kept focus with and without the fix. The generation went 1 -> 2 with the fix and 1 -> 1 without.
  - Media's transfer moves focus only when focus is still on the hidden pre-resize control, or on nothing, or on an undisplayed widget. So a click on a visible control is never taken back. Media does not show this case, and no Media test was added (out of scope).
- **AC#4.** Both nodes are appended at the census tail (floor 1751). A temporary 20-repeat (a scratch plugin, since removed) gave 40 of 40, 20 of each, at `-n 4` in 113 s, at load 31.6-55.7 (mean 46.2) on 18 CPUs.
- **Ratchets (owner size waiver).**
  - `library_screen.py` screen row: 38149 -> 38161 lines, 1375 -> 1376 methods.
  - Pre-import LOC: `MAX_PASS_ADDED_LOC` 426,388 -> 426,400, with the snapshot regenerated by `scripts/update_boot_budget_snapshots.py --only preimport`. The +12 is all in `library_screen`, and the module count is unchanged.
- **User Guide.** `library/notes.md` and `library/file-notes.md` say that a click or key press after a resize keeps the chosen control.
- Files: `tldw_chatbook/UI/Screens/library_screen.py`, `Tests/UI/test_library_file_notes_workspace.py`, `Tests/Architecture/test_screen_size_ratchet.py`, `Tests/Performance/test_screen_preimport_payload_budget.py`, `Tests/Performance/boot_budget_snapshots/preimport_payload.json`, `scripts/ui_pr_gate_census.txt`, `scripts/check_ui_pr_gate_census.py`, `Docs/User_Guide/library/notes.md`, `Docs/User_Guide/library/file-notes.md`, `backlog/docs/lessons-textual.md`.
