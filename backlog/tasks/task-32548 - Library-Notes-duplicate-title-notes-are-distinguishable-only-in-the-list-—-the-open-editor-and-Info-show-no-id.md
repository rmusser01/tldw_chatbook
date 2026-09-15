---
id: TASK-32548
title: >-
  Library Notes: duplicate-title notes are distinguishable only in the list —
  the open editor and Info show no id
status: Done
assignee: []
created_date: '2026-09-13 06:47'
updated_date: '2026-09-14 19:25'
labels:
  - library
  - notes
  - critique-3
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Critique #3 (dev 5fd502dbac), both assessors, personas Alex and Riley. Residual of task-32254 (list-row tie-break shipped in #2611).

**What happened.** Two "Reading list" rows are tie-broken in the list ("· #8d61" / "· #0a89", B 29; "#0dc8 / #c4e8", A 41). Opening one: the editor header reads "Reading list", Info reads "reading, study · v1 · 12 words", and nothing identifies which of the two is open (A 43; B 37, 38). Captures: A 41, 43; B 29, 37, 38.

**Cause.** INFERRED: `note_row_tiebreak_labels()` (task-32254) is a list-row projection; the editor header and Info do not consult it.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 When the open note's title collides with another visible note, the editor header or Info Properties carries the same tie-break suffix the list row shows
- [x] #2 A test opens the second of two same-titled notes and asserts the suffix is rendered in the editor
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Reproduce (done: opening the second 'Reading list' shows a bare 'Reading list' header while the list row shows '· #8a41').
2. RED pin on the editor title Static.
3. Add LibraryNotePresentationState.title_suffix, computed once per canvas sync from the already-built tree projection via note_row_tiebreak_labels; render it after ellipsizing the title.
4. GREEN + live.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Reproduced live at dev 2f97a42c9a (235x52): the list showed "Reading list ·
Unfiled · 2m · #7f65" and "· #8a41"; opening the second one gave a heading of
"Reading list" and nothing else identifying it
(`editor-00-32548-editor-header-no-tiebreak-235x52`).

**Fix.** `LibraryNotePresentationState.title_suffix` carries the row's own
tie-break into the editor. It is computed by the one function the list row
asks (`note_row_tiebreak_labels`, via a new
`note_title_tiebreak_suffix(projection, note_id)`), so the two cannot
disagree; the heading strip's three Statics — editor, Preview and Info —
render `ellipsize(title) · suffix`, appended AFTER the ellipsis so a long
title cannot eat the identity. The Preview BODY's document heading keeps the
plain title: that is the note's own title as written, not a list identity.

Resolved in `_library_notes_canvas_kwargs`'s editor branch, where the
projection has just been built and `_selected_note_id` is already read —
NOT inside `_library_note_presentation_state`, which runs on every keystroke
in the body. (First attempt put it at the top of that method and broke
`test_the_flat_list_keeps_its_open_sort_chooser`, whose fake has no
`_selected_note_id`; the editor branch is both cheaper and the only place
the value is used.)

**Known limit, stated rather than papered over.** At 100x30 the heading strip
is about 46 columns and clips the title to "Reading …", suffix included —
exactly as it clipped before this change
(`editor-10-32548-editor-header-compact-clipped-100x30`). AC#1's "editor
header or Info Properties" is met at the review width; a compact-width
identity is a separate question and is named as a rider in the group report.

**Tests.** `::test_opening_the_second_of_two_same_titled_notes_shows_its_
tiebreak_in_the_editor` reads the suffix off the LIST ROW production rendered
and requires the editor heading (and Info's) to match it — it supplies
neither string. RED on detached origin/dev, GREEN here, plus a negative
control that an unrepeated title keeps a plain header. Live at 235x52.

Modified: `tldw_chatbook/Widgets/Library/library_notes_canvas.py`,
`tldw_chatbook/UI/Library_Modules/library_notes_controller.py`,
`tldw_chatbook/UI/Library_Modules/library_notes_state.py`, the pin file, the
guide.
<!-- SECTION:NOTES:END -->
