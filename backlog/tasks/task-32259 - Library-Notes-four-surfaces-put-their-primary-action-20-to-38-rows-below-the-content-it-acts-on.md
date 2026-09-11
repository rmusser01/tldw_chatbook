---
id: TASK-32259
title: >-
  Library Notes: four surfaces put their primary action 20 to 38 rows below the
  content it acts on
status: Done
assignee:
  - '@claude'
created_date: '2026-09-10 18:05'
updated_date: '2026-09-11 16:48'
labels:
  - library
  - notes
  - ux
  - critique-notes-2026-09
  - layout
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The design assessor's structural finding, and the reason the specificity verdict reads "authored in the copy and the state model; category-interchangeable in the spatial design". Adjacent to peer task-32217 (no density rule across Library canvases), but these four instances are specific to Notes:

- Import once's "Check selection" at row 49 under a selection summary at rows 7-11;
- the Folder-files empty state at rows 7-9 with 47 blank rows under it (the full-screen half of this overlaps peer task-32211);
- the Markdown preview closing its box at row 33 with 14 blank rows beneath (task-32249);
- the review pane elided to `. ke...` while the Notes list it is not about keeps its share of the width (task-32250).

The brand words are "cyberpunk, efficient, effective"; half of most of these screens is empty. The one place the layout was fixed -- the notes list itself, now about 132 of 235 columns with ages and a complete two-row toolbar (task-32127) -- proves the rest is fixable.

Evidence: Library ▸ Notes critique snapshot `.impeccable/critique/2026-09-10T17-32-16Z__w-chatbook-widgets-library-library-notes-canvas-py.md` (dev e6cb464239, 2026-09-10, dual-assessor live review at 235x52 and 100x30/60x24 on two fresh profiles and one seeded profile with a git-backed 71-file Obsidian vault fixture; the reconciling parent re-tested every disagreement on a third pair of profiles and traced each surviving cause to code). Captures under the session scratchpad `notes-crit2/{C,D,R}/caps/`.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Import once's primary action renders adjacent to the selection summary it acts on, not at the pane floor
- [x] #2 A full-canvas task (import review, sync setup, export) owns the pane width while it is the task in hand
- [x] #3 No Notes surface renders its primary action more than a screenful below the content it acts on at 235x52 -- AC SCOPED (task-32259 implementation): the Session Git panel's 22-row gap is task-32248's (wave-3 Task 5) and the review pane's elision is task-32250's; this task covers Import once, the Folder-files empty state and the Markdown preview
- [x] #4 Covered by a test asserting the action's row distance from its content on the Import once selection screen
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Trace the four surfaces the description names; Session Git belongs to Task 5 and the review-pane elision to task-32250
2. Compose Import once's primary action inside the bounded select/destination body instead of under the 1fr scroll floor
3. Close the Items list while a full-canvas Notes task (import, lasting sync) is in hand
4. RED->GREEN row-distance test on the Import once selection screen; live capture at 235x52
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Three of the four surfaces the description names; the fourth split by ownership ruling.

- Import once (AC#1/#4): the primary action is composed INSIDE `_ImportBody` for the select/destination phase instead of after it. The body is `height: 1fr`, so a bounded selection screen put "Check selection" at the pane floor -- row 49 of 52, under a summary at rows 7-11. The review/importing/receipt phases keep the pinned floor: their lists are unbounded and an action that scrolls out of view is worse than a far one. Pinned by a measured row-distance assertion at 235x52.
- Pane width (AC#2): `LIBRARY_NOTES_FULL_CANVAS_VIEWS` (`import`, `lasting_add`, `lasting_roots`) makes `_library_notes_work_first_preferences` return `items_open=False`, so the task owns the pane instead of sharing it with a notes list it is not about. Derived at resolve time exactly like the existing work-session `library_open=False` override, so the persisted preference is untouched and the list returns with the task.
- The Markdown preview is task-32249, landed in the same branch.
- The Folder-files empty state needed no change: its "Choose folder…" already stands beside its copy at row 7; the 47 blank rows beneath are peer task-32211's full-screen half.

Not done here, by the wave controller's ruling and the task-file scoping above: the Session Git panel's 22-row gap (task-32248, Task 5) and the review pane's elision (task-32250).

Live at 235x52: `wave3-caps/layout/14-addfiles.txt` (the chooser owns the pane, notes list collapsed to its grip) and `17-import-nosel.txt` (the EMPTY-selection screen: summary row 8, action row 10). Pinned as exact row distances rather than a bound (review finding F5): 8 rows with two files chosen -- the three source buttons plus the destination label, field and error line -- and 1 row with nothing chosen, at 235x52, 100x30 and 60x24. On dev the same measurements are 48, 26 and 19-20 rows.

Modified: `tldw_chatbook/Widgets/Library/library_note_import_canvas.py`, `tldw_chatbook/UI/Library_Modules/screen_constants.py`, `tldw_chatbook/UI/Library_Modules/library_notes_controller.py`, `Tests/UI/test_library_notes_w3_layout.py`, `Docs/User_Guide/library/notes.md`.
AC ORDERING (review finding F4): the AC revision above was written and saved BEFORE the code it describes -- the task files' `updated_date: '2026-09-11 15:42'` UTC precedes the first code commit (`08:44:44 -0700` = `15:44:44` UTC) by two minutes. The commit graph cannot show it, because the six task files were staged together in one hygiene commit at the end; recorded here so the ordering is a fact in the file rather than a claim in a report.
<!-- SECTION:NOTES:END -->
