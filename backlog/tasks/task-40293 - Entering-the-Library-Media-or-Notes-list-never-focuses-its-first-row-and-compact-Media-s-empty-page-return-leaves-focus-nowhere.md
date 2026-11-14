---
id: TASK-40293
title: "Entering the Library Media or Notes list never focuses its first row, and compact Media's empty-page return leaves focus nowhere"
status: To Do
created_date: 2026-11-12 19:36
labels:
- follow-up
- library
- focus
- keyboard
priority: medium
references:
- tldw_chatbook/UI/Screens/library_screen.py
- Tests/UI/test_library_shell.py
- Tests/UI/test_library_media_side_by_side.py
- scripts/library_shell_push_deselect.txt
dependencies:
- TASK-39130
- TASK-39152
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in round 13 of the 2026-10-23 burn-down: TASK-39152 and TASK-39130 left these tests red, unchanged, as product defects. Checked at the round-13 stack top 8b633d0101, 2026-11-12: the nodes below fail when run together (run.sh, `-n 4`, load ~13), with the failure quoted.

**What the user sees.**
- Pressing Enter on the Media or Notes rail row should put focus on the list's first row, so the arrow keys move through the list at once. Focus stays where it was instead: `library-ingest-top-button` for Media, `#library-browse-library-grip` for Notes at 170x48.
- On compact Media, a filter or delete that leaves an exactly empty page returns the canvas to its list by itself (80e4cd7056, task-32043 / task-32086). Focus then ends at `None`, so the keyboard has nowhere to land.

**Cause, from TASK-39152's and TASK-39130's notes.**
- Since de4d38b59a, `_library_focus_channel_owns_this_window()` (`UI/Screens/library_screen.py:8950`) reads "no rows mounted yet" as a settled empty page. The whole-screen recompose therefore restores the pre-press focus and disarms the entry arm. Making the predicate own the window while the arm is set made the first Media node pass whole. Notes is probably the same cause.
- The compact empty-page return's whole-screen recompose strands focus. TASK-39130 places the fix in the task-32086 reset branch of the browse apply: arm the list entry focus when the recompose strands focus.

**Nodes** (failures at 8b633d0101):
- `Tests/UI/test_library_shell.py::test_library_media_list_focuses_first_row_and_arrow_keys_move_it`: "entering the list must focus its first row" (`library-media-row-0` not focused).
- `::test_library_media_tab_away_survives_a_background_recompose_within_the_settle_window` and `::test_library_media_entry_focus_survives_three_chained_recomposes`: `library-media-row-0` not focused.
- `::test_library_notes_list_focuses_first_row_and_arrow_keys_move_it`: "entering the list must focus its first row" (`library-notes-tree-note-1` not focused).
- `Tests/UI/test_library_media_side_by_side.py::test_compact_media_viewer_back_empty_page_focuses_recovery_control`: "Empty viewer return did not focus the Import recovery action; focus is None."

The `test_library_shell.py` nodes are deselected from the `library-shell-dev` push job (`scripts/library_shell_push_deselect.txt`, TASK-39153); remove each line when its fix lands. The side-by-side node runs on no lane; TASK-39130 laned the file's other passing nodes.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Entering the Media or Notes list from its rail row focuses the first row and the arrow keys move through the list, also when a background recompose lands inside the entry window
- [ ] #2 On compact Media, the automatic return from an exactly empty page lands focus on a visible recovery control or list control, never on nothing
- [ ] #3 The five nodes pass 3 of 3 at `-n 4`, each shown red before the fix, and their lines leave `scripts/library_shell_push_deselect.txt`
- [ ] #4 The side-by-side node runs on a PR lane
<!-- AC:END -->
