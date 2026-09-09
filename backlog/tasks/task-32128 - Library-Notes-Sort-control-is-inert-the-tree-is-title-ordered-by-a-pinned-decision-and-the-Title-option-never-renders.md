---
id: TASK-32128
title: >-
  Library Notes Sort control is inert: the tree is title-ordered by a pinned
  decision and the Title option never renders
status: Done
assignee: []
created_date: '2026-09-08 21:39'
updated_date: '2026-09-09 07:15'
labels:
  - library
  - notes
  - ux
  - critique-notes-2026-09
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Observed by the evidence assessor: Newest → Oldest changes the label and nothing else; both modes show strict alphabetical order, which is neither of the DB orders. `test_placement_title_sort_key_matches_repository_tiebreakers` pins title ordering for tree placements, so the control contradicts a decision. 'Title' does not fit the 38-column pane and never renders. Evidence: Library ▸ Notes critique snapshot `.impeccable/critique/2026-09-09T04-24-50Z__w-chatbook-widgets-library-library-notes-canvas-py.md` (dev c4a7b1911f, 2026-09-08/09, dual-agent live review at 235x52/100x30/60x24 on a fresh and a seeded profile with a 71-file Obsidian-style vault; parent re-tested every disagreement on a third profile).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Either the Sort control is removed from the tree view, or Newest/Oldest visibly reorder the rows within each folder
- [x] #2 The decision is recorded in the task and the pinning test is reconciled with it
- [x] #3 Every remaining sort option renders at the narrowest pane width
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Read the pinned title-order test and the repository ORDER BY.\n2. Decide: remove Sort from the tree view (repository contract) or reorder placements.\n3. Pin the decision with a test and record it in the notes.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
DECISION: the Sort control is REMOVED from the folder tree; it stays on the flat list.

Why, having read the pin and the repository: `page_note_placements` pages with `ORDER BY title COLLATE NOCASE, n.id` (root) and `ORDER BY title COLLATE NOCASE, id, membership_id` (folder), and every offset the tree browses with is computed against that order -- including `locate_note_tree_placement`'s `containing_offset`, which is how a deep link finds a note's page. A Newest/Oldest control could therefore only re-sort the twenty placements already loaded: on any folder with more than one page it would show "the alphabetically first twenty, ordered by date" while the pager copy said "Notes 1-20 of N". A control that lies on every paged folder is worse than no control, and ordering by date properly is a repository change (an ORDER BY parameter through paging AND the locator), not a display one. The critique's preference for a working control is noted and declined on that ground.

`test_placement_title_sort_key_matches_repository_tiebreakers` keeps its assertion and gains a docstring recording this decision, so the next reader finds the reasoning at the pin.

AC#3: the flat list's strip now renders all three options inside a 38-column pane -- before this, "Title" was painted at x=32..48 and never seen. The pin asserts over every action in that frame, not only the three options.

COMPLETE CENSUS of the removed control's consumers (`grep -rn "library-notes-sort" Tests/ tldw_chatbook/ Docs/`), and what each needed:

Production (3 sites, all KEPT -- the control still ships for the flat list): the canvas's Button + choice strip; `handle_library_notes_sort` / `handle_library_notes_sort_choice` in the notes controller (+ their screen delegators); the compact-stage CSS for `#library-notes-sort-choices`.

Tests that MOUNT it and needed reconciling (7):
- test_library_canvas_sync_defects.py:271 -- dropped the `sort_open` trigger (its two siblings pin the same focus contract).
- test_library_canvas_scoped_sync.py:476 -- latency probe now waits on "New".
- test_library_note_import_flow.py:445,482 -- the fence checks the three controls that still exist.
- test_library_shell.py::test_library_shell_notes_sort_opens_direct_choices_and_applies_one_value -- body rewritten to pin the ABSENCE plus the untouched sort key, with the reason at the pin.
- test_library_shell.py::test_library_shell_notes_navigator_has_named_action_groups_and_filter_label -- id dropped from the expected browse-actions set.
- test_library_shell.py::test_library_shell_restored_notes_sort_and_filter_render_on_first_paint -- the restored key is pinned on the state instead of a button label.
- test_library_shell.py::test_library_note_footer_covers_navigator_create_sync_and_exit -- the sort-strip footer segment removed with its reason.

Tests that mention it but were ALREADY FAILING on dev before this branch (verified by running each node id at f054f35ae1 in a detached worktree), so they were left alone: test_library_shell.py::test_library_shell_notes_row_opens_notes_list_canvas, ::test_library_shell_notes_list_actions_use_two_named_horizontal_rows, ::test_library_shell_notes_list_renders_bracketed_titles_verbatim, ::test_library_note_editor_back_restores_exact_wide_browse_context, and test_library_file_notes_workspace.py::test_wide_files_task_return_restores_database_browse_receipt. All five wait for `#library-notes-row-0/-1` or press the control after such a wait, and Database Notes composes a folder tree. Two of them carry a recorded note anyway, since they are the ones a reader would consult.

Not tests: Docs/superpowers plan and review archives (historical), and `Tests/UI/test_css_build_integrity.py` / `test_library_notes_characterization.py` / `test_library_notes_wiring.py`, which pin the CSS block, the handler-name spelling and the handler wiring -- all still true.

Files: tldw_chatbook/Widgets/Library/library_notes_canvas.py, Tests/UI/test_library_notes_rename_propagation_t31796.py, Tests/UI/test_library_canvas_sync_defects.py, Tests/UI/test_library_canvas_scoped_sync.py, Tests/UI/test_library_note_import_flow.py, Tests/UI/test_library_shell.py, Tests/UI/test_library_notes_wave_list.py, Docs/User_Guide/library/notes.md.
FINAL WHOLE-BRANCH REVIEW (2026-09-09), three follow-ups:

C1 (merge blocker, fixed): the census above reconciled `test_library_canvas_scoped_sync.py:476` but MISSED the second reference in the SAME file -- `test_notes_per_click_updates_keep_screen_and_canvas_identity` also waited on `#library-notes-sort` and spent 30 s failing `#library-notes-sort never mounted`. It now waits on `#library-notes-new` and drops the Sort round trip, with the reason at the line. Fixing that exposed the next line, `#library-notes-row-0`, which was ALREADY red at the wave base f054f35ae1 (a flat-list id; the tree indexes note rows across its folder rows) -- the base run simply never reached it. It now presses the first `.library-notes-tree-note-row`, so the file goes from 2 failed/7 passed at the base to 1 failed/8 passed here. The surviving failure, `test_import_status_lines_patch_the_mounted_static_without_recompose`, reproduces unchanged at the base. The generalisable half is recorded in backlog/docs/lessons-testing-evidence.md.

C2 (verified, no change): `test_library_shell.py::test_library_note_keyboard_capability_matrix[filter_sort-terminal_size0|1]` -- the task-10 path that activates `#library-notes-sort` then `#library-notes-sort-oldest`. Both parameters fail IDENTICALLY at this branch head and at the base f054f35ae1, with `AssertionError: Keyboard filter submit never reached search_notes.` -- i.e. they never reach the sort activation at all. Baseline failure, unrelated to this task.

I4 (deleted coverage restored): removing the shell test's press -> apply round trip left NOTHING asserting that pressing a sort option applies it (the wave test only asserted the options compose). `Tests/UI/test_library_notes_wave_list.py::test_pressing_a_sort_option_applies_that_sort` now joins the two real halves -- the option Button this canvas composes (asserting it carries the `.library-notes-sort-choice` class the screen's `@on` selector matches) and the real `LibraryNotesController.handle_library_notes_sort_choice` -- and asserts the sort becomes "oldest", the strip closes, select mode ends and the canvas is synced. The reconciliation comment in test_library_shell.py now names it.
REVIEW ROUND 2 (PR #2544, Qodo finding 6): hiding the chooser when a tree projection exists left `sort_choices_visible` set, so a chooser opened during the initial flat-list window kept the footer offering "choose sort" and spent the first Escape on a mode nothing was painting. `_library_notes_canvas_kwargs` now clears the flag in the same pass that decides the tree owns the list (it already built the projection there; it is now built once and reused). Pinned by `test_the_tree_taking_over_closes_the_flat_sort_chooser` and `test_the_flat_list_keeps_its_open_sort_chooser`.
<!-- SECTION:NOTES:END -->
