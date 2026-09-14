---
id: TASK-32550
title: >-
  Library Notes: list-toolbar keyboard nits — "/" re-focuses the filter with
  stale text and the caret at the start, and the Tab count to a toolbar button
  depends on filter state
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
Critique #3 (dev 5fd502dbac), both assessors, persona Alex. Two small keyboard defects on the same toolbar.

1. `/` on an already-filtered list re-focuses the field with the old text and the caret at the start, so typing "scaling" produced "scalingReading"; End + Ctrl+U are needed first (A 41).
2. Disabled Sort is skipped in the Tab order while a filter shows, so the same recipe `/` + Tab×4 lands on Add from files… on an unfiltered list and on Export on a filtered one — B's power run opened the Export bundle canvas by accident (B D15). Captures: A 41; B §3.

**Cause.** INFERRED.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 "/" on a filtered list selects the existing filter text (or clears it) so typing replaces it
- [x] #2 Tab counts to a toolbar button do not change with filter state, or notes.md's keyboard recipes name the state they assume
- [x] #3 Tests pin both
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Re-derive both halves live.
2. AC#1 does NOT reproduce at this dev: the notes filter is a SelectAllOnFocusingClickInput and Textual's select_on_focus is on, so '/' selects the existing text and typing replaces it (verified: filter 'Reading', focus out, '/', type 'SCAL' -> 'SCAL'). Pin it so it cannot regress.
3. AC#2 reproduces: '/'+Tab x4 lands on 'Add from files…' unfiltered and on 'Export' filtered (disabled Sort leaves the focus chain). Doc branch - pin both counts and name the filter state in notes.md's keyboard recipes.
4. No production code.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
**No production code. Both halves re-derived live at dev 2f97a42c9a first,
and the task's first defect does not reproduce.**

AC#1: the notes filter is a `LibraryRailSearchInput`, i.e. a
`SelectAllOnFocusingClickInput`, and Textual's `Input(select_on_focus=True)`
default applies to a programmatic `.focus()`. Walked it: filter submitted as
"Reading", focus moved away, `/`, then typed "SCAL" — the box read "SCAL",
not "SCALReading" (`editor-10-32550-slash-replaces-filter-235x52`). The same
code is present at the critique's own dev (5fd502dbac), so A's "scalingReading"
was some other gesture. Pinned instead of changed:
`::test_slash_on_a_filtered_list_selects_the_existing_text` asserts the
selection range AND that one keystroke replaces the value; it passes on
detached origin/dev, which is the honest outcome for a defect that is not
there.

AC#2 reproduces exactly as filed: `/` then Tab ×4 lands on **Add from files…**
unfiltered and on **Export** filtered, because a filter disables **Sort** and
a disabled Button leaves the focus chain
(`editor-10-32550-tab4-unfiltered-add-from-files-235x52`,
`editor-10-32550-tab4-filtered-export-235x52`). Taking the AC's second branch:
Sort's disabled state is load-bearing (task-32128's "filter results keep their
own order"), so the counts are documented rather than equalised — notes.md's
Notes-list section now carries a "Keyboard on this toolbar" block naming both
states, and `::test_tab_counts_to_toolbar_buttons_are_pinned_per_filter_state`
pins them so the documented recipe cannot silently go stale.

Modified: `Tests/UI/test_library_notes_w4_editor.py`,
`Docs/User_Guide/library/notes.md`.
<!-- SECTION:NOTES:END -->
