---
id: TASK-40294
title: "The compact Library Notes footer keeps the rail tier for 2 s or more after Notes entry has put focus on a note row"
status: To Do
created_date: 2026-11-12 19:36
labels:
- follow-up
- library
- notes
- footer
priority: low
references:
- tldw_chatbook/UI/Library_Modules/library_notes_controller.py
- tldw_chatbook/UI/Screens/library_screen.py
- Tests/UI/test_library_shell.py
- scripts/library_shell_push_deselect.txt
dependencies:
- TASK-39152
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in round 13 of the 2026-10-23 burn-down: TASK-39152 classified the node as a product defect and left it red. Checked at the round-13 stack top 8b633d0101, 2026-11-12: the nodes below fail when run together (run.sh, `-n 4`, load ~13), with the failure quoted.

**What the user sees.** On compact Notes entry the system's entry focus lands on a note row, but the footer still shows the rail tier for 2 s or more. The navigator keys `n new` and `g folder` already work, but the footer does not advertise them.

**Node.** `Tests/UI/test_library_shell.py::test_library_note_footer_covers_navigator_create_sync_and_exit`: "Footer did not settle to 'n new | / find | g folder | esc rail'; got '/ focus search | F6 next pane | esc focus rail'."

The `test_library_shell.py` nodes are deselected from the `library-shell-dev` push job (`scripts/library_shell_push_deselect.txt`, TASK-39153); remove each line when its fix lands.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 After compact Notes entry puts focus on a note row, the footer shows the navigator tier within the test's settle window
- [ ] #2 The node passes 3 of 3 at `-n 4`, shown red before the fix, and its line leaves `scripts/library_shell_push_deselect.txt`
<!-- AC:END -->
