---
id: TASK-40298
title: "F6 can never reach the Library Starter rail on a compact deep link, because the rail's focus targets list only the search input Starter does not have"
status: Done
assignee:
- '@claude'
updated_date: '2026-11-13 05:40'
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
- [x] #1 On a compact route whose rail has no search input, F6 reaches the rail's first row
- [x] #2 The node passes 3 of 3 at `-n 4`, shown red before the fix, and its line leaves `scripts/library_shell_push_deselect.txt`
<!-- AC:END -->

## Implementation Plan

1. Define one rail `WorkbenchPaneTarget` (search, then Import, Explore, collapse) and use it in every route's F6 target tuple, so no route lists only the search box.
2. Check whether the node also races the rail's row mount; if so, wait for the row before pressing F6.
3. Show the node red with the fix reverted; remove its deselect line.

## Implementation Notes

Premise held on base 26b2e677dd: every route-specific F6 tuple (Media, Notes, Prompts, Skills, Conversations, Collections) named only `library-search-input` for the rail.

- One class constant `_LIBRARY_RAIL_FOCUS_TARGET` (search, then the compact Import row, Explore all tools, Collapse; the default tuple's existing list) now serves all seven tuples, so no route can list a rail target the Starter rail does not have. Net +3 lines in `library_screen.py` (ratchet row raised 38164 -> 38167 in this commit).
- The node also raced the rail: `_wait_for_evidence_round` returns before the Starter rail mounts its rows (a probe saw only `library-rail-heading` at that point and the three rows 1 s later), and F6 picks from what is mounted when pressed. The test now waits for `#library-row-ingest-import-media` before the first F6; the assertions are unchanged.
- Red/green: with the route tuples restored to search-only (the wait kept), the node fails on `import_button.has_focus` (load ~70); with the constant it passes (3 of 3 at `-n 4`, load ~41: the eleven stream nodes passed 11/11 in each of three runs).
- Files: `UI/Screens/library_screen.py`, `Tests/UI/test_library_shell.py`, `Tests/Architecture/test_screen_size_ratchet.py`, `scripts/library_shell_push_deselect.txt`, `Docs/User_Guide/library.md`.

Covering run (shared by the stream's five tasks, final tree): 58 Library UI files that reference the changed seams (`test_library_shell.py` whole, side-by-side, canvas-sync defects, choice strips, multiselect, honesty, workbench focus, screen navigation and 50 more), `-n 4`, load 42-66: 2023 passed, 863 failed. Every failing function was re-run on the head tree and on the base source (26b2e677dd, same tests): of those 871 nodes 850 fail on head and 845 on base, almost all the local RecoveryRequired wall for plain Tests/UI files plus the deselected nodes owned by TASK-32503/34000.60/40296/40299. Head-only reds re-checked: `test_focus_library_list_entry_falls_back_to_first_row_outside_active_selection` (fake rows without `ancestors`; fixed in TASK-40293), `test_any_non_markdown_item_with_content_says_why_rendered_is_absent[video]` (3 of 3 green on head when re-run alone), `test_literal_history_restore_keeps_kind_definition_and_guards` (red 1-3 of 3 params on base too), and four nodes that passed on re-run. `Tests/Architecture` Library wiring files: 49 passed (serial).
