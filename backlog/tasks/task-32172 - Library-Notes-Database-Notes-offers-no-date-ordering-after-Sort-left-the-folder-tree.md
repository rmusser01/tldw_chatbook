---
id: TASK-32172
title: >-
  Library Notes: Database Notes offers no date ordering after Sort left the
  folder tree
status: Done
assignee: []
created_date: '2026-09-09 09:10'
updated_date: '2026-09-09 17:55'
labels:
  - library
  - notes
  - critique-notes-2026-09
  - rider
  - list
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Rider from the Notes critique fix wave (plan
Docs/superpowers/plans/2026-09-09-library-notes-critique-wave.md); raised in
the task review of task-32128. task-32128 correctly gated the flat list's
Sort control off the folder tree — the tree's order is the repository's
paging contract, and a Sort control there would lie about what it controls.
That left Database Notes with no date-ordering control anywhere once any
folder exists (which is always true in practice, because Agent_Lessons is
seeded): the repository's folder paging and the deep-link locator both hard-
code `ORDER BY title COLLATE NOCASE`, with no ORDER BY parameter through
`page_note_placements` or the locator to plumb Newest/Oldest through. Re-
offering Sort in the tree requires that plumbing to exist first, or the
control would be decorative again.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Choosing Newest or Oldest reorders folder placements by the note's
  modified date, consistently across paged results within a folder
- [x] #2 The deep-link locator honours the same order parameter, not a fixed
  title order
- [x] #3 A Sort control is re-added to the folder tree only once both above
  hold true
- [x] #4 The behavior is pinned in a test
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add an `order` parameter (title|newest|oldest) to `LocalNoteFolderRepository.page_note_placements`, `locate_note_tree_placement` and `search_note_tree_placements`, sharing one ORDER-BY/rank-predicate definition so the locator's containing-offset is computed against the same order the pager returns.
2. Thread `order` through `NotesScopeService` and the three LibraryScreen call sites, reading `_notes_state.sort`.
3. Re-enable the Sort control on the folder tree (`sort_available`) and reload the tree (root + expanded folders) when the sort value changes.
4. Reconcile the task-32128 absence pins with a recorded reason; add repository + UI tests (paged newest order across a page boundary, locator agreement, tree re-sort across pages).
5. Live-verify on the power profile, update Docs/User_Guide/library/notes.md, commit.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Made the placement order a repository PARAMETER of both halves task-32128 found hard-coded, then gave the folder tree its Sort control back.

Approach. `page_note_placements` paged `ORDER BY title COLLATE NOCASE` and `locate_note_tree_placement` counted its `placement_offset` as a rank against that same fixed order -- two halves, one order, neither a parameter, so a Sort control could only reorder the loaded window. Both now take `order` (`PLACEMENT_ORDERS` = title|newest|oldest, validated closed because it reaches an ORDER BY by interpolation), sharing `_placement_order_term` and `_placement_rank_clauses` so pager and locator cannot diverge. Threaded through `NotesScopeService` (default "title", so no existing caller changes) to three LibraryScreen sites via one `_library_notes_placement_order()` read of `_notes_state.sort`; folder-children paging deliberately does NOT get it (folders have no date).

Choosing a value re-pages rather than repainting -- only a reload moves a note across a page boundary -- and `_request_library_notes_tree_initial_load` now reloads the EXPANDED folders too, since a folder's placements only exist while it is open (this also fixes the editor-return and post-import refreshes, which left open folders childless). The save-time in-place re-sort (`patch_notes_tree_branches_title`) takes the order as well: a save moves `last_modified`, so re-sorting that window by title under Newest would contradict the next page load.

Trade-off. The FTS filter window keeps the search seam's own order (grouped by folder path); re-running it on a sort change would need a new injected controller dependency, so Sort renders disabled there with 'Filter results keep their own order. Clear the filter to sort.' Verified this costs nothing live: `test_library_note_keyboard_capability_matrix[filter_sort-*]` is already baseline-red at d0ff40842f.

Behaviour change worth knowing: the tree now honours the persisted sort, whose default is "newest", so Database Notes' default browse order moves from title to most-recently-changed. Forced -- the button has always read 'Sort: Newest'.

One real bug found in RED: `last_modified` is a DATETIME column, so an anchor value read into Python returns a datetime and re-binds in a different textual shape than the column holds, mis-ranking the note against itself. The rank clauses now read the anchor in SQL via `(SELECT ... FROM notes WHERE id = ?)`.

Files: Notes/note_folder_repository.py, Notes/notes_scope_service.py, UI/Screens/library_screen.py, UI/Library_Modules/library_notes_controller.py, Widgets/Library/library_notes_canvas.py, Library/library_notes_tree_paging.py, Library/library_shell_state.py; Tests/UI/test_library_notes_riders_r_list.py (new) plus reconciled pins in test_note_folder_repository.py, test_notes_scope_service_folders.py, test_library_notes_rename_propagation_t31796.py, test_library_notes_wave_list.py, test_library_notes_folder_navigator.py, test_library_shell.py, test_library_note_import_flow.py; Docs/User_Guide/library/notes.md.
<!-- SECTION:NOTES:END -->
