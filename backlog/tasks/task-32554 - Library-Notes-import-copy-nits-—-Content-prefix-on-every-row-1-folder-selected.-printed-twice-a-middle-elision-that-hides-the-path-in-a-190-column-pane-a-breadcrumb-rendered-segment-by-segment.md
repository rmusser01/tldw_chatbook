---
id: TASK-32554
title: >-
  Library Notes: import copy nits — "Content:" prefix on every row, "1 folder
  selected." printed twice, a middle-elision that hides the path in a 190-column
  pane, a breadcrumb rendered segment-by-segment
status: Done
assignee: []
created_date: '2026-09-13 06:48'
updated_date: '2026-09-14 19:50'
labels:
  - library
  - notes
  - critique-3
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Critique #3 (dev 5fd502dbac), assessor A (B's captures show the same strings), persona Jordan, Import once workflow. Minor observations grouped by surface.

1. Every New review row starts "Content: create 1 new note:" — "Content:" carries nothing (A 31; B 29).
2. The setup pane says "1 folder selected." then "1 folder selected: /Users/…/crit3/…vault" — the same fact twice, and the middle-elision hides `A/fresh/` in a 190-column pane that had room for the whole path (A 30; B 28).
3. The picker breadcrumb renders a 100+ character path segment-by-segment (A 27).

**Cause.** PROVEN copy.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Review rows drop the "Content:" prefix
- [x] #2 The setup pane states the selection once, with the full path when it fits and a middle-elision only when it does not
- [x] #3 The picker breadcrumb elides from the middle to one line
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Reproduce live (done: every New row reads 'Content: create 1 new note: ...'; the pane prints '1 folder selected.' then '1 folder selected: /Users/.../w4-imp…vault'; the picker breadcrumb renders every segment and runs off the dialog).
2. RED pins in Tests/UI/test_library_notes_wave_import_ux.py.
3. Fixes: drop the 'Content: ' prefix in _effect_summary (library_note_import_state.py) and capitalise the clause; make the select-phase status_line state the next step instead of repeating the count; measure the mounted pane and widen the folder-path budget so a path that fits renders whole; port enhanced_file_picker's middle-collapsing breadcrumb into the base dialog.
4. GREEN + live captures + guide stamps.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Three copy defects on the Import once path, each fixed where the string is built.

**AC#1 — "Content:".** `_effect_summary` opened all four of its branches with that label. The row already reads `<path> · <effect> · <placement>`, so the column's subject is the one thing it never needed to name — and those nine columns came off the clause that was already being clipped (live, the Library-review row ended '· Cre…' before and '· Create in va…' after). The clauses are capitalised so they still read as sentences beside their siblings.

**AC#2 — the selection stated twice, and elided in a pane that had room.** Two separate causes. The status line restated the count; it now states the next step instead ('Check the selection to see what will be imported.', or 'Choose a Notes destination, then check the selection.' while one is still required), leaving the summary as the only line that names the selection. And `_bounded_source_name`'s budget was the fixed compact floor of 48: the canvas now composes at that floor and widens the line from the width it actually got.

Measuring that width is the part that needed a second round. The mounted `Static` is a NEW widget on every recompose and is still unmeasured inside the `call_after_refresh` the recompose schedules, so reading it alone left the line at the floor in the real app until the terminal was resized — while a bare widget-host pin at 190 columns was green. The canvas keeps its width across its children's recomposes, so it is the ruler when the child reads 0. Both pins are kept: the host for the copy rule, the screen route for the measurement. Lesson recorded in `backlog/docs/lessons-textual.md`.

**AC#3 — the breadcrumb.** `FileSystemPickerScreen._update_breadcrumbs` rendered every segment, so a 100+ character location ran off the dialog and the crumbs naming where you ARE were the ones clipped. It now keeps root + '…' + the tail on one line — the shape `EnhancedFileDialog` already used — with each crumb's absolute path still on its tooltip and the full location still spelled out above. The enhanced picker keeps its own override: it also carries `btn.data` for its own click handler, and merging the two is a bigger change than this task.

**Files.** `Library/library_note_import_state.py`, `Widgets/Library/library_note_import_canvas.py`, `Third_Party/textual_fspicker/base_dialog.py`, `Tests/UI/test_library_notes_w4_import_keyboard.py`, `Tests/Library/test_library_note_import_state.py`, `Tests/Notes/test_note_import_planner.py`, `Docs/User_Guide/library/notes.md`.
<!-- SECTION:NOTES:END -->
