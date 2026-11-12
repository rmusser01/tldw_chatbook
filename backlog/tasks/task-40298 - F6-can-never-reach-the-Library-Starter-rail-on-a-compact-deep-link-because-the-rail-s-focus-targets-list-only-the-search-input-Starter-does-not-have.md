---
id: TASK-40298
title: "F6 can never reach the Library Starter rail on a compact deep link, because the rail's focus targets list only the search input Starter does not have"
status: To Do
created_date: 2026-11-12 19:36
labels:
- follow-up
- library
- focus
- keyboard
priority: low
references:
- tldw_chatbook/UI/Screens/library_screen.py
- Tests/UI/test_library_shell.py
- scripts/library_shell_push_deselect.txt
dependencies:
- TASK-39152
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in round 13 of the 2026-10-23 burn-down: TASK-39152 classified the node as a product defect and left it red. Checked at the round-13 stack top 8b633d0101, 2026-11-12: the nodes below fail when run together (run.sh, `-n 4`, load ~13), with the failure quoted.

**What the user sees.** On a compact deep link that hides the rail's search (the Starter route), F6 never lands in the rail, so a keyboard user cannot reach the Starter's rows.

**Cause, from TASK-39152's notes.** 5dd1077df6's rail pane target (`WorkbenchPaneTarget("library-rail", ("library-search-input",))`, `UI/Screens/library_screen.py:1481-1543`) lists only `library-search-input`, which the Starter rail does not have.

**Node.** `Tests/UI/test_library_shell.py::test_library_starter_hidden_route_focuses_compact_rail_without_search`: `LibraryRailRowButton(id='library-row-ingest-import-media').has_focus` is False.

The `test_library_shell.py` nodes are deselected from the `library-shell-dev` push job (`scripts/library_shell_push_deselect.txt`, TASK-39153); remove each line when its fix lands.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 On a compact route whose rail has no search input, F6 reaches the rail's first row
- [ ] #2 The node passes 3 of 3 at `-n 4`, shown red before the fix, and its line leaves `scripts/library_shell_push_deselect.txt`
<!-- AC:END -->
