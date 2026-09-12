---
id: TASK-32100
title: >-
  Library Notes: opening a note supersedes the folder-tree locator, so 'Locating
  note…' / row reveal is abandoned on every open
status: Done
assignee:
  - '@claude'
created_date: '2026-09-08 22:42'
updated_date: '2026-09-10 16:41'
labels:
  - library
  - notes
  - ux
  - critique-8
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
After task-32050 the detail load no longer depends on the tree tokens, but the row click still bumps the notes navigation generation and cancels `_locate_library_notes_tree_target`, so the tree never reveals the opened note's row. Found by the task-32050 review (PR #2519). Rider from the critique-8 fix wave reviews (plan Docs/superpowers/plans/2026-09-08-library-crit8-wave.md; wave PRs #2519 #2523 #2524 #2525 #2528 #2531).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Opening a note from the list reveals and marks its row in the folder tree
- [x] #2 A second click while the locator runs supersedes only the older locator, not the detail load
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Re-verify the premise on current dev (peer Notes wave 1 landed): instrument a mounted real-port click and observe whether the locator started by _begin_library_note_load survives. VERIFIED RED: 'supersede -> 3 / locator start gen=3 / supersede -> 4 / locator end -> False'.
2. Diagnose: on_descendant_focus treats the row click's OWN focus event as 'the user took control', bumping focus_intent_generation and calling _supersede_library_notes_navigation while navigation_status shows. Both fences kill a locator that was started with focus=False and therefore has no focus stake at all.
3. Failing test first (Tests/UI/test_library_crit8_notes_loader.py, real DB + real NotesScopeService): filter for a note nested in a folder, click its row, assert the folder is revealed in tree_expanded_ids and the placement is marked.
4. Minimal fix: record whether the running locator intends to take focus (navigation_focus_intent on the notes state); fence focus_intent_generation only for a focus=True locator; supersede on foreign focus only for a focus=True locator.
5. Re-run the new test plus the folder-navigator + crit8 loader suites (they pin the focus=True supersede behaviour).
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Premise re-verified on current dev (e6cb464239, after the peer's Notes wave 1): still broken. A traced mounted click reads `supersede -> 3 / locator start focus=False gen=3 / supersede -> 4 / locator end -> False` -- the locator `_begin_library_note_load` starts is abandoned on EVERY open.

Root cause: `on_descendant_focus` treats the row click's own DescendantFocus (which lands after the handler, while "Locating note…" shows) as "the user took control": it bumps `focus_intent_generation` and calls `_supersede_library_notes_navigation`. Both fences kill the locator. But that locator runs with `focus=False` -- it only reveals and marks a row and never touches focus, so it has no stake in the focus intent at all.

Fix (3 edits): `LibraryNotesState.navigation_focus_intent` records whether the locator behind `navigation_status` will take focus; `_locate_library_notes_tree_target` captures `focus_generation` only when `focus=True` (its `current()` fence skips it otherwise); `on_descendant_focus` supersedes a running locator only when that locator wants focus. The focus=True supersede paths -- the four folder-navigator tests that pin `/`, Back, and an explicit supersede abandoning a blocked locator -- are untouched and stay green.

Tests (Tests/UI/test_library_crit8_notes_loader.py, real CharactersRAGDB + real NotesScopeService + real clicks, because a direct locator call posts no focus event): `test_opening_a_filtered_note_reveals_its_folder_in_the_tree` (red first: "expanded is set()") and `test_a_second_open_supersedes_the_older_locator_not_its_own_load` (blocks the locator service, clicks a second note mid-flight, asserts the second note opens, its row is marked and the older reveal wrote nothing). 6 passed.

Files: tldw_chatbook/UI/Screens/library_screen.py, tldw_chatbook/UI/Library_Modules/library_notes_state.py, Tests/UI/test_library_crit8_notes_loader.py, Tests/Architecture/test_library_notes_wiring.py (field-count ratchet 100 -> 101), Docs/User_Guide/library/notes.md.
<!-- SECTION:NOTES:END -->
