---
id: TASK-32137
title: >-
  Library Notes list rows show no age, duplicate titles are indistinguishable,
  and long titles clip
status: Done
assignee: []
created_date: '2026-09-08 21:39'
updated_date: '2026-09-09 07:15'
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
PROVEN: tree rows (`_compose_tree_rows`) carry no age label; the two-line title-plus-age code path lives in `_compose_list`, which is dead once a tree projection exists (always, because Agent_Lessons is seeded). Two notes titled 'Reading list' render as identical rows in the tree and in filtered results. The guide says rows show 'title and age'. Depends on task-32127 for width. Evidence: Library ▸ Notes critique snapshot `.impeccable/critique/2026-09-09T04-24-50Z__w-chatbook-widgets-library-library-notes-canvas-py.md` (dev c4a7b1911f, 2026-09-08/09, dual-agent live review at 235x52/100x30/60x24 on a fresh and a seeded profile with a 71-file Obsidian-style vault; parent re-tested every disagreement on a third profile).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Tree rows show a relative age
- [x] #2 Visible duplicate titles get a folder then modified-date suffix at render time
- [x] #3 The unreachable flat-list age code is used or removed
- [x] #4 The guide matches
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Failing tests: tree rows carry an age; duplicate titles get folder + age.\n2. Add age_label to the tree row model and one shared row-label renderer for both list paths.\n3. Update the guide.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
`LibraryNotesTreeRow` gains `age_label`, filled by `_note_row` (the one function that builds a note row, for both the browse and filter projections) from the placement's own note record. Both list paths now render through one `compose_note_row_label`: "title · age", with the folder inserted when a filter has scattered the rows or when the title repeats under the same parent -- "Reading list · Unfiled · 2h".

AC#3: the flat list's two-line title/age branch is GONE rather than kept -- both renderers call the shared helper, so there is one row-label spelling. The flat renderer itself stays: it is the fallback whenever no tree projection exists (before the first branch load, and in every harness without a paged notes service), not dead code.

The one-line label is recorded at the pin it changes, `test_library_shell_notes_list_renders_bracketed_titles_verbatim`: the title is now the label's first " · " segment rather than its first line, and the assertion splits on that. (That test fails on dev before this branch too -- it waits for a flat `#library-notes-row-1` that a folder tree replaces -- so the edit is a record for whoever repairs it, not a green step.)

The duplicate suffix is deliberately scoped to same-parent siblings, which is also what the AC asks for. Two rows with one title in two different folders already sit under their own folder rows, and spending the width there ellipsized the semantic sync status at 60 columns (caught by test_live_host_renders_duplicate_placements_and_preserves_focus_at_60x20, which passes again with the narrower rule).

Not changed: `.library-notes-row` stays `height: 2`, since a stylesheet-parity test pins that height across the notes and prompts rows; the flat fallback row therefore has an unused second line. The tree rows, which are what a real profile renders, are height 1.

Live at 235x52: every row carries an age ("· 40m"), and both seeded "Reading list" notes render "Reading list · Unfiled · 40m" (caps/01). They remain hard to tell apart in that capture only because the seed gave them the same minute; the folder and age are both rendered.

Files: tldw_chatbook/Library/library_notes_tree_state.py, tldw_chatbook/Widgets/Library/library_notes_canvas.py, Tests/UI/test_library_shell.py, Tests/UI/test_library_notes_wave_list.py, Docs/User_Guide/library/notes.md.
<!-- SECTION:NOTES:END -->
