---
id: TASK-32389
title: >-
  Library Notes: below 64 columns the reader keeps an empty work pane while the
  list is cut
status: Done
assignee:
  - '@claude'
created_date: '2026-09-11 10:30'
updated_date: '2026-09-11 16:48'
labels:
  - library
  - notes
  - layout
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
task-32217 settled that an empty work pane hands its columns to the list. `_sync_library_notes_reader_layout_from_shell` (`library_screen.py:6248-6258`) passes `priority="items"` unconditionally, so at 60 columns the notes reader lays out 32/18 with nothing in the work pane -- 18 columns held by an empty stage while the list is cut to 32. The reviewer of task-32360 ruled this a separate defect from that task's narrow-stage return, and it bypasses an already-shipped rule rather than raising an open design question. See also task-32304 (the same empty-work-pane symptom below 64 columns on Collections, Conversations and Skills) and task-32065, which introduced the `list_first_when_empty` rule both want -- whoever takes one should take the other rather than re-deriving it.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 At 60 columns with nothing open, the notes list takes the columns the empty work pane was holding, as task-32217's rule requires
- [x] #2 Opening a note restores the reading split at that width
- [x] #3 The 60-column case is pinned in the notes layout test file
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Reproduce the 32/18 split with the layout resolver at 60 columns
2. Opt the Notes profile into task-32065's list_first_when_empty rule
3. Drop an items priority below the single-stage floor when the work pane is empty, so the one rule owns the case
4. Pin 60 columns both ways in Tests/UI/test_library_notes_wave_list.py
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Reproduced at the resolver: `resolve_adaptive_reader_layout(60, ..., priority="items")` returned `items_width=32, reader_width=18` -- the 32/18 the task describes.

The Notes list view asks for `priority="items"` whenever the list owns the workflow, and that request takes the width-starved early return, which keeps the list at its floor and gives the remainder to the work pane. task-32065's `list_first_when_empty` rule -- "with nothing to read, the list wins the stage" -- sits BELOW that return and never saw the width. Two changes, both small:

1. `LIBRARY_NOTES_READER_PROFILE` opts into `list_first_when_empty` (Media was the only profile using it).
2. `resolve_adaptive_reader_layout` drops an `items` priority when `not reader_has_item and profile.list_first_when_empty and width < LIBRARY_EMERGENCY_WIDTH`, so the existing rule owns the case for every caller rather than being restated in a second branch. With something open, `reader_has_item` is True and nothing changes -- no other profile sets the flag, so no other destination's geometry moves.

RED on dev: "18 columns are still held by an empty work pane while the list is cut to 32." GREEN on branch, pinned from both sides at 60 columns in `Tests/UI/test_library_notes_wave_list.py`.

Live at 60x24: `wave3-caps/layout/31-60col-notes.txt` (the list is the whole stage, ~50 of 60 columns) and `32-60col-note-open.txt` (opening a note hands it back).

Modified: `tldw_chatbook/Utils/adaptive_reader_state.py`, `tldw_chatbook/UI/Library_Modules/screen_constants.py`, `Tests/UI/test_library_notes_wave_list.py`, `Docs/User_Guide/library/notes.md`.
<!-- SECTION:NOTES:END -->
