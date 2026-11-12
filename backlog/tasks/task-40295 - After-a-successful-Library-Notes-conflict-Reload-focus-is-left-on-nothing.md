---
id: TASK-40295
title: "After a successful Library Notes conflict Reload, focus is left on nothing"
status: To Do
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
- [ ] #1 After a successful conflict Reload, focus lands on the reloaded editor (or another named control), and the footer shows that control's tier
- [ ] #2 The node passes 3 of 3 at `-n 4`, shown red before the fix, and its line leaves `scripts/library_shell_push_deselect.txt`
<!-- AC:END -->
