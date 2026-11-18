---
id: TASK-38957
title: "Folder files: a user focus change does not supersede the compact transition's deferred focus restore"
status: Done
assignee:
- '@claude'
created_date: 2026-11-05 21:35
updated_date: 2026-11-06 06:57
labels:
- follow-up
- library
- focus
- file-notes
priority: low
references:
- tldw_chatbook/UI/Screens/library_screen.py
- tldw_chatbook/UI/Library_Modules/library_notes_controller.py
- Tests/UI/test_library_file_notes_workspace.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in round 8 of the 2026-10-23 burn-down; checked on dev 29592ecc9b, 2026-11-05. TASK-38695 (Done) first raised this as a "Proposed follow-up (product, not fixed here)".

**Code path.**
- `_transition_library_notes_presentation` (`tldw_chatbook/UI/Screens/library_screen.py:8180-8257`) takes a snapshot of `focus_intent_generation` for its guard (`:8220-8226`).
- It then queues `_restore_library_notes_focus_identity` (`:8248`), plus a settled restore two turns later (`tldw_chatbook/UI/Library_Modules/library_notes_controller.py:2764-2774`).
- A user's focus change advances the generation only when `relevant` holds (`library_screen.py:11240-11255`). `relevant` requires the Database Notes workflow (`_library_notes_workflow_active`, `:5128-5133`), the rail, Media or Conversations.
- In Folder files mode none of those applies, so the generation stays put and the guard (`:5717-5734`) still passes. The deferred restore then moves focus back to where it was before the user's newer choice.

**Observed.** `Tests/UI/test_library_file_notes_workspace.py:772` (`test_folder_files_automatic_items_priority_does_not_steal_focus_or_persist`) failed 2 of 72 runs from exactly this sequence. TASK-38695 fixed only the test.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 In Folder files mode, a user focus change made after a compact/wide transition but before its settled restore keeps the user's focus; a test focuses the editor between the two restores without waiting for the settled one and is red on dev
- [x] #2 Programmatic focus and the Database Notes, rail, Media and Conversations focus behaviour are unchanged, and the existing Notes focus and canvas-sync pins stay green
- [x] #3 The new test passes 20 of 20 runs under -n 4, with the load recorded
<!-- AC:END -->

## Implementation Plan

1. A test on the real Library screen, in Folder files: open a file, close the Files pane, hold the compact transition's settled restore, resize to 79, move focus to the editor as a user would, then let the restore run. Show it red on base.
2. Count Folder files in the `relevant` predicate of `LibraryScreen.on_descendant_focus`, so a user's focus change there advances `focus_intent_generation` and vetoes the deferred restore. Database Notes, the rail, Media and Conversations are unchanged by construction: the added term is true only while Folder files owns the canvas.
3. Show the existing Notes focus and canvas-sync pins unchanged, and run the new test 20 times under `-n 4`.

## Implementation Notes

- **Fix.** `relevant` in `LibraryScreen.on_descendant_focus` gains `or self._file_notes_active()`. A user's focus change in Folder files now advances the generation, so `_restore_library_notes_settled_focus` finds its guard stale and leaves the user's focus alone. Programmatic landings (the restore's own target, the restoring flag, `resize_settling`) are still classified programmatic, so the change only adds the mode the task names.
- **Test.** `test_folder_files_user_focus_between_compact_restores_is_kept` (real Library screen, 160x45 then 79x30). It holds `_restore_library_notes_settled_focus`, Tabs to the editor after the first restore, and runs the held restore; the editor must keep focus.
  - It uses Tab, not a click. A click's `DescendantFocus` reaches the screen before its `MouseDown`, and the `MouseDown` is what ends `resize_settling`. So after any resize, the first click's focus counts as programmatic, in every Notes mode. My first version clicked and stayed red after the fix. That is outside this task's criteria (AC#2 keeps the other modes unchanged), so it is listed as a follow-up.
- **Red and green.** Red with the base `library_screen.py` at the second assertion (focus back on `library-file-notes-items-grip`); green with the fix. AC#3: 20 of 20 under `-n 4` (a temporary `repeat` parametrize, since removed), 45 s, load 21-25 on 18 CPUs.
- **AC#2.**
  - `test_folder_files_automatic_items_priority_does_not_steal_focus_or_persist` passes.
  - Four Database Notes focus and canvas-sync pins fail locally with the `RecoveryRequired` trap. Under a temporary `@private_profile_test` marker (since removed) they pass 4 of 4 with and without the fix: `test_converted_notes_site_keeps_focus_on_a_real_key_press`, `test_notes_row_press_to_editor_keeps_focus_inside_the_canvas`, `test_automatic_notes_reconcile_restores_richer_control_focus`, `test_overlapping_reconciles_retain_original_semantic_focus`.
  - The canvas-sync defects, gaps, removed-widget focus and w5 keyboard-focus files: 28 failed and 24 passed, both with and without the fix, the same 28 (`RecoveryRequired`).
- The `library_screen.py` screen-size row goes 38177 -> 38180 under the owner's waiver. The node is appended at the census tail.
- Files: `tldw_chatbook/UI/Screens/library_screen.py`, `Tests/UI/test_library_file_notes_workspace.py`, `Tests/Architecture/test_screen_size_ratchet.py`, `scripts/ui_pr_gate_census.txt`, `scripts/check_ui_pr_gate_census.py`.

