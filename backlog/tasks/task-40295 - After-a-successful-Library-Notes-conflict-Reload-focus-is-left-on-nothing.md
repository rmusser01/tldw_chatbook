---
id: TASK-40295
title: "After a successful Library Notes conflict Reload, focus is left on nothing"
status: Done
assignee:
- '@claude'
updated_date: '2026-11-13 05:40'
created_date: 2026-11-12 19:36
labels:
- follow-up
- library
- notes
- focus
priority: low
references:
- tldw_chatbook/UI/Library_Modules/library_notes_controller.py
- Tests/UI/test_library_shell.py
- scripts/library_shell_push_deselect.txt
dependencies:
- TASK-39152
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in round 13 of the 2026-10-23 burn-down: TASK-39152 re-pinned this test's drifted steps and left its last step red as a product defect. Checked at the round-13 stack top 8b633d0101, 2026-11-12: the nodes below fail when run together (run.sh, `-n 4`, load ~13), with the failure quoted.

**What the user sees.** After a save conflict, choosing Reload replaces the editor with the file's version. When the conflict region unmounts, focus goes nowhere, so the keyboard does nothing until the user clicks. The cause: `_resolve_library_note_conflict` (`UI/Library_Modules/library_notes_controller.py:4915`) focuses nothing after a successful Reload.

**Node.** `Tests/UI/test_library_shell.py::test_library_note_footer_tracks_editor_states_and_ancillary_contents`: "Footer did not settle to 'esc notes | ctrl+end end'; got 'esc notes'." The footer reflects the empty focus.

The `test_library_shell.py` nodes are deselected from the `library-shell-dev` push job (`scripts/library_shell_push_deselect.txt`, TASK-39153); remove each line when its fix lands.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 After a successful conflict Reload, focus lands on the reloaded editor (or another named control), and the footer shows that control's tier
- [x] #2 The node passes 3 of 3 at `-n 4`, shown red before the fix, and its line leaves `scripts/library_shell_push_deselect.txt`
<!-- AC:END -->

## Implementation Plan

1. In `_resolve_library_note_conflict`, after a resolution that leaves no conflict and no validation veto, focus the note body (the same after-refresh helper the validation path uses).
2. Re-pin the node's last footer step to the body's tier (a focused text field carries the N-31 typing chip) and assert the body holds focus.
3. Show the node red with the fix reverted; remove its deselect line.

## Implementation Notes

Premise held on base 26b2e677dd: `_resolve_library_note_conflict` focused nothing after a resolution that cleared the conflict.

- After a resolution that leaves no conflict and no validation veto (Reload, Overwrite, a non-conflict failure), it now focuses `#library-note-body` through the existing after-refresh helper the validation path uses (`_focus_library_note_validation_field("body")`).
- The node's last pin, "esc notes | ctrl+end end", could not match any focus: `ctrl+end` is shown only with the body focused, and since TASK-34000.40 (N-31) a focused text field adds "typing in field" (the node's own first editor step pins "esc notes | typing in field | ctrl+end end"). Re-pinned to that tier, plus an assertion that the body holds focus.
- Red/green: with the new branch removed, the node fails on that step with "got 'esc notes'" (load ~64); with it, it passes (3 of 3 at `-n 4`, load ~41: the eleven stream nodes passed 11/11 in each of three runs).
- Files: `UI/Library_Modules/library_notes_controller.py` (+2, ratchet row 7425 -> 7427), `Tests/UI/test_library_shell.py`, `Tests/Architecture/test_library_modules_size_ratchet.py`, `scripts/library_shell_push_deselect.txt`, `Docs/User_Guide/library/notes.md`.

Covering run (shared by the stream's five tasks, final tree): 58 Library UI files that reference the changed seams (`test_library_shell.py` whole, side-by-side, canvas-sync defects, choice strips, multiselect, honesty, workbench focus, screen navigation and 50 more), `-n 4`, load 42-66: 2023 passed, 863 failed. Every failing function was re-run on the head tree and on the base source (26b2e677dd, same tests): of those 871 nodes 850 fail on head and 845 on base, almost all the local RecoveryRequired wall for plain Tests/UI files plus the deselected nodes owned by TASK-32503/34000.60/40296/40299. Head-only reds re-checked: `test_focus_library_list_entry_falls_back_to_first_row_outside_active_selection` (fake rows without `ancestors`; fixed in TASK-40293), `test_any_non_markdown_item_with_content_says_why_rendered_is_absent[video]` (3 of 3 green on head when re-run alone), `test_literal_history_restore_keeps_kind_definition_and_guards` (red 1-3 of 3 params on base too), and four nodes that passed on re-run. `Tests/Architecture` Library wiring files: 49 passed (serial).
