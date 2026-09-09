---
id: TASK-32135
title: >-
  Library Notes Import once review: 71 five-line rows with no bulk action and
  the Skip/Create controls 130 columns from the path
status: Done
assignee: []
created_date: '2026-09-08 21:39'
updated_date: '2026-09-09 06:36'
labels:
  - library
  - notes
  - ux
  - critique-notes-2026-09
  - import
  - layout
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Both assessors: each review row is five lines ('Ready to import as a new note.' / 'Content: create 1 new note.' / 'Create in vault / Archive.' plus the path and right-aligned Skip/Create); about six rows fit per screen and 25 wheel notches moved from item 1 to item 8. Reviewing 71 items honestly is dozens of identical screens, which undermines the reviewed-mutation principle. The repeat-import review (Unchanged repeat, folder collision) is strong and should keep its grammar. Evidence: Library ▸ Notes critique snapshot `.impeccable/critique/2026-09-09T04-24-50Z__w-chatbook-widgets-library-library-notes-canvas-py.md` (dev c4a7b1911f, 2026-09-08/09, dual-agent live review at 235x52/100x30/60x24 on a fresh and a seeded profile with a 71-file Obsidian-style vault; parent re-tested every disagreement on a third profile).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Rows render on one line (path · action · destination) with their controls adjacent to the path
- [x] #2 Rows are grouped under a header per action class with a per-group Skip
- [x] #3 At 235x52 at least 15 rows are visible per screen
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. One-line review rows (path · action · destination) with Skip/Create adjacent.
2. Per-group Skip all / Create all on the group header row.
3. Empty #notes-import-review-options slot above the groups for the Obsidian toggle.
4. Row-count test at 235x52.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Each review item is now one `Horizontal.note-import-row` (height 1): a nowrap/ellipsis summary 'path · effect · destination' (or 'path · reason' for a non-importable item) followed by that item's own Skip/Create new/Update existing controls. Only a matched item's target label and diff still render below the row, so the bulk of a review is exactly one line per source. Group headings became a `note-import-group-row` carrying 'Skip all' and, where the group can create notes, 'Create all'; a new GroupActionRequested message reaches `LibraryNoteImportController.set_group_action`, which applies the action to the rendered page's items of that class only -- exactly the count the header shows.

An empty `#notes-import-review-options` container sits above the groups as the slot for task-32129's Obsidian toggle.

Two existing tests pinned the old five-line layout and were updated: the classification-group test queries the row text (and now covers the two new classifications), and the 60-column scroll test needed more items to still overflow. `_ProductionCssCanvasApp` became a `ConsolidatedCSSApp` because a bare App does not register the consolidated widget defaults the real app loads through `_get_default_css`, so row heights differed from production.

Live at 235x52 the first review page paints 25 rows (19 New + 1 Unsupported + 5 Skipped) on one screen, against the AC's 15.

Files: Widgets/Library/library_note_import_canvas.py (+ regenerated css/widget_defaults_{self,scoped}.tcss), UI/Library_Modules/library_note_import_controller.py, Tests/Widgets/Library/test_library_note_import_canvas.py, Tests/UI/test_library_notes_wave_import_ux.py, Docs/User_Guide/library/notes.md. Caps 05/06.

Review addendum (Qodo findings 3, 9): `set_group_action` now converts the classification to its enum at the seam, so an unknown group is refused with the shipped failure notice instead of silently changing nothing. Measured at 60 and 80 columns, an uncertain or update-existing row pushed its trailing buttons entirely outside the body; **Confirm this match**, **Replace note content** and **Add folder placement** moved onto their own line under the row, and a test asserts every review button lies inside the body's content region at 60x24. A New source is still one line.
<!-- SECTION:NOTES:END -->
