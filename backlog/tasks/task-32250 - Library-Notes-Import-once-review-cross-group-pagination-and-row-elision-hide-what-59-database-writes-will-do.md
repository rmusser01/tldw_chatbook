---
id: TASK-32250
title: >-
  Library Notes Import once review: cross-group pagination and row elision hide
  what 59 database writes will do
status: Done
assignee:
  - '@claude'
created_date: '2026-09-10 18:05'
updated_date: '2026-09-11 16:13'
labels:
  - library
  - notes
  - ux
  - critique-notes-2026-09
  - import
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found independently by both assessors, and it survives #2540's row-geometry fixes -- those did land and are pinned by name (`test_both_relationships_are_buttons_under_their_own_descriptions`, one-line rows with adjacent controls, >=15 rows at 235x52, `Skip all`/`Create all` per group). This is the residual, and it is the half that matters at the moment of approval.

Page 1 of 3 is 23 `vault/Archive/Archived note NNN.md` rows. Groups are split across page boundaries, so the counts change meaning per page (`New (23)`, `New (22)`, `New (13)`) and **the total never appears before you approve 59 database writes**. Every New row ends `. ke...`, hiding the keywords and link count the guide says the row exists to convey; the 120-character-filename row loses its entire outcome clause. Deciding what to skip requires holding page 1's 23 Archive rows in mind while reading page 3. Meanwhile the review -- the only thing being done -- gets about 120 of 235 columns while the Notes list it is not about keeps its share.

Partially fixed prior P2: task-32125 and task-32134 covered the chooser and the source-change half. Cause INFERRED from behaviour; not re-traced by the parent. Captures: C 30/33/34, D 25/26/27.

Fix direction: page within groups, collapse a long uniform run to one summary row with a disclosure, order Unsupported/Empty first, and give the review the full pane width while it owns the pane.

Evidence: Library ▸ Notes critique snapshot `.impeccable/critique/2026-09-10T17-32-16Z__w-chatbook-widgets-library-library-notes-canvas-py.md` (dev e6cb464239, 2026-09-10, dual-assessor live review at 235x52 and 100x30/60x24 on two fresh profiles and one seeded profile with a git-backed 71-file Obsidian vault fixture; the reconciling parent re-tested every disagreement on a third pair of profiles and traced each surviving cause to code). Captures under the session scratchpad `notes-crit2/{C,D,R}/caps/`.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The total item count is visible before the approve action, not only per page
- [x] #2 A group is never split across pages: the count shown on a page means the same thing as the count in its header
- [x] #3 Every review row states the resulting title, its keywords and its link count without elision at 235x52
- [x] #4 A long uniform run collapses to one summary row with a disclosure rather than 23 near-identical rows
- [x] #5 'Next page' on the last page is visibly distinct from an active pager
- [x] #6 The review owns the full pane width while it is the task in hand
- [x] #7 Covered by a test for the un-elided row at 235x52 and a test for group-contained pagination
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Page over classification-ordered items so groups are contiguous; the heading states what its count means when a group spans pages.\n2. Collapse a long uniform run into one summary row with a disclosure.\n3. Middle-elide the path so the decision-bearing outcome survives at 235x52.\n4. Disabled pager buttons carry their state as text.\n5. The review takes the pane while it is the task in hand.\n6. RED/GREEN tests for the un-elided row and group-contained pagination.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Reproduced on the real 71-file vault through the production chain: page 1 of 3, 23 `vault/Archive/Archived note NNN.md` rows, "New (23)" then "New (22)" then "New (13)", every New row ending "· ke…".

The fix is one change of unit plus three small ones.

PAGING BY RENDERED ROWS. `_page` sliced `plan.items` in path order, so a page held arbitrary slices of three groups. It now slices a classification-ordered view (`REVIEW_CLASSIFICATION_ORDER`, defined once and shared with the canvas so a heading cannot sit over the wrong rows) and fills pages by what they RENDER, never cutting through a run, under a hard 200-source mount ceiling. Consequence on the review vault: one page, 66 sources, "New (58)". A group too big for one page still splits, and its heading then reads "New (25 of 58 on this page)" so a page count always says which number it means.

UNIFORM-RUN COLLAPSE. Eight or more consecutive rows sharing a folder, a classification, an action and a destination collapse to one summary row with a disclosure that still holds every individual Skip/Create. First cut of this was render-only, which introduced the defects own shape one level down -- the 45-note Archive run rendered as "25 files" on page 1 and "20 files" on page 2 -- so the run key moved into the pager (`review_run_key`) and the canvas takes it from there; a test pins that the two find the same boundaries.

THE ROW SPENDS ITS PATH BUDGET LAST. `_review_row_summary` put the full path first, so a 120-character filename pushed the outcome clause off the row and every other row lost its keywords and link count. The path middle-elides to 56 cells; the resulting title, keywords and link count survive at 235 columns.

PAGER AND PANE. Previous/Next carry their state as text through the same helper task-32257 fixed. And `_library_notes_work_first_preferences` hands the whole pane to the review while it is open (phase review or importing), with the transition driven from the one seam that sees it, `_publish_library_note_import_snapshot`.

AC#1 was already satisfied before this change -- the status line has always stated the total above the rows -- and is now pinned; it reads "Review 66 sources before import." after task-32258.

One deliberate simplification, marked in code: the collapsed disclosures chrome is set with inline styles, because the app-tier `Collapsible` rule (a round border, a 3-row title, a 3-row floor, a bottom margin) outranks this widgets own BUNDLED_CSS whatever its specificity -- five lines of chrome for a summary the pager budgeted one line for.

Verified at 235x52 and 100x30 through the production canvas on the real vault (`wave3-caps/import-review/10-review-page1-235x52.txt`, `10-review-page1-100x30.txt`).

Files: `tldw_chatbook/Library/library_note_import_state.py`, `tldw_chatbook/Notes/note_import_plan_models.py`, `tldw_chatbook/Widgets/Library/library_note_import_canvas.py`, `tldw_chatbook/UI/Library_Modules/library_notes_controller.py`, `Tests/UI/test_library_notes_wave_import_ux.py`, `Tests/Library/test_library_note_import_state.py`, `Tests/UI/Library_Modules/test_library_note_import_controller.py`, `Tests/Widgets/Library/test_library_note_import_canvas.py`, `Docs/User_Guide/library/notes.md`.
**Fix round 1 (review findings 8, 9).** The canvas sorted its groups by
`tuple(_CLASSIFICATION_LABELS)` while the pager used
`REVIEW_CLASSIFICATION_ORDER` -- two hand-kept orders that agreed by luck, and a
divergence would have put groups on a page in an order the pager did not budget
for. The canvas now derives its order from the shared sequence, pinned in
`test_the_pager_and_the_canvas_find_the_same_runs`. And a run past the
200-source mount ceiling read "200 files" then "50 files" on the next page --
the defect's own shape, one level down. `NoteImportPage.run_totals` carries each
run's whole size, so both halves read "200 of 250 files".

**Interaction with task-32259 (fixed after both merged to dev).** Was: choosing a source left the list open (`test_the_review_takes_the_pane_while_it_is_the_task_in_hand`'s second assertion) -- superseded by task-32259 AC#2 (every phase of Import once closes the list); the pin now asserts `items_open is False` for the select phase, and the review's own `(library_open, items_open) == (False, False)` assertion is unchanged.
<!-- SECTION:NOTES:END -->
