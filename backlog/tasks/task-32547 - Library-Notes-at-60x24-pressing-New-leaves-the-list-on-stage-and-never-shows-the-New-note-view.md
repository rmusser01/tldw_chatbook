---
id: TASK-32547
title: >-
  Library Notes: at 60x24 pressing "New" leaves the list on stage and never
  shows the New note view
status: In Progress
assignee: []
created_date: '2026-09-13 06:47'
updated_date: '2026-09-14 16:58'
labels:
  - library
  - notes
  - critique-3
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Critique #3 (dev 5fd502dbac), assessor B, persona Jordan on a compact terminal. D10 (workaround: Ctrl+N).

**What happened.** 60x24, Notes list → "New": the footer changes to "enter create | esc notes" but the stage still shows the list and the work pane stays collapsed to its `Notes` grip — Blank note / From a template… are never on screen (B 54). Captures: B 54.

**Cause.** INFERRED: single-stage routing does not promote the New note view. Not covered by 32202 / 32455 (compact surplus rows, ctrl+n region scoping) or 32391 (Ctrl+N transitional frame).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 At 60x24 pressing New promotes the New note view (Blank note / From a template…) to the stage with Blank note focused and the footer reading enter create note
- [x] #2 Escape from that view returns to the list at the same size
- [x] #3 A test at 60x24 pins the promotion
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Reproduce at 60x24: New leaves the list on stage, footer says 'enter create | esc notes'.
2. Trace: _sync_library_notes_reader_layout_from_shell passes reader_has_item=(view != 'list'), but the create view never sets _notes_state.view -- it is carried by _library_selected_row_id == LIBRARY_ROW_CREATE_NOTE. With reader_has_item False the list_first_when_empty rule keeps the list and collapses the work pane.
3. Fix reader_has_item to include the create view.
4. Pin at 60x24 through the real shell: Blank note displayed+focused, footer 'enter create', Escape returns to the list.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Same root cause as task-32544 AC#2, and one clause fixes both.

The New note view is carried by `_library_selected_row_id == LIBRARY_ROW_CREATE_NOTE`, never by `_notes_state.view` (which stays 'list' throughout create mode). `_sync_library_notes_reader_layout_from_shell` derived `reader_has_item` from `view` alone, so at 60 columns the resolver took task-32065's `list_first_when_empty` branch -- the rule that keeps the LIST rather than an empty work pane below the single-stage floor -- and returned items_open=True / items_width=50 / reader_width=0. The compact stage machinery was never at fault: `_compact_library_notes_stage` already answered 'notes' for the create region, and the reader shell inside that stage was the half making the decision. Proven by reading the resolver's return before touching anything, then pinned at the resolver AND through the real screen.

Live at 60x24: pressing New now promotes the view -- 'New note', '█Blank note' focused, 'From a template…' on screen, the list collapsed to its Notes grip, footer 'enter create | esc notes'; Escape returns to the list unchanged (wave4-caps/layout/layout-17-60x24-new-after, layout-18-60x24-escape-back; before: layout-02-60x24-new).

Files: UI/Screens/library_screen.py, Tests/UI/test_library_notes_w4_layout.py.
<!-- SECTION:NOTES:END -->
