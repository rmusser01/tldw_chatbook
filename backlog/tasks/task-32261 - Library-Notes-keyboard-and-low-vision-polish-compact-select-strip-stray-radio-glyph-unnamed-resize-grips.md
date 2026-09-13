---
id: TASK-32261
title: >-
  Library Notes keyboard and low-vision polish: compact select strip, stray
  radio glyph, unnamed resize grips
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
  - a11y
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Four small findings from the same persona pass. Adjacent to peer task-32227 (select-strip spacing on Media) and peer task-32235 (one glyph, three meanings across Library); these are the Notes instances:

- at 100x30 the select strip renders `0 selected  Done  All 10  Clear` -- "Export selected" is absent although the guide lists it;
- `0 selected` is printed twice in the strip, with no gap before the focused `[Done]`;
- a stray `o` glyph renders on primary actions and reads as an unselected radio;
- the `--->` resize grips carry no accessible name.

The rest of this persona's pass was strong -- focus is legible by shape on every control that has it, and it survives a monochrome capture -- which is what makes these four stand out.

Evidence: Library ▸ Notes critique snapshot `.impeccable/critique/2026-09-10T17-32-16Z__w-chatbook-widgets-library-library-notes-canvas-py.md` (dev e6cb464239, 2026-09-10, dual-assessor live review at 235x52 and 100x30/60x24 on two fresh profiles and one seeded profile with a git-backed 71-file Obsidian vault fixture; the reconciling parent re-tested every disagreement on a third pair of profiles and traced each surviving cause to code). Captures under the session scratchpad `notes-crit2/{C,D,R}/caps/`.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Export selected is present at 100x30, or the guide stops claiming it
- [x] #2 The selection count is printed once on the compact strip the critique measured, on its own line rather than jammed against the focused Done -- AC SCOPED (task-32261 implementation): the wide layout still mounts both counters, and "one source of truth" for them is peer task-32272 (wave-3 Task 6), whose brief names the same select-mode counters in the same file
- [x] #3 The Notes canvas follows the Library glyph legend: the circle marks a DISABLED action (beside its reason), `☐/☑` mark selection, and neither is used for the other -- AC REVISED (task-32261 implementation): the circle is `LIBRARY_DISABLED_ACTION_MARKER`, a Library-wide decision task-32235 shipped, documented in `Docs/User_Guide/library.md`'s legend table and pinned by `test_one_meaning_per_library_glyph`. Dropping it from disabled actions (which task-32235's own AC#1 text also asks for) is a Library-wide change across every canvas, its guide pages and ~20 test files -- it belongs to the peer that owns that legend, not to the Notes instance
- [x] #4 The resize grips carry an accessible name
- [x] #5 Covered by a test asserting the compact select-strip contents
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Measure the compact select strip at the 42-column Items pane a 100x30 terminal resolves
2. Hide the in-strip duplicate counter in compact so Export selected fits the pane (the wide duplicate is peer task-32272's)
3. Confirm the pane grips already carry an accessible name (task-32355) and pin it on the Notes route
4. Re-scope AC#3: the circle marker is the Library-wide disabled-action legend task-32235 shipped
5. RED->GREEN compact strip test; guide + stamp
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Measured first: at 100x30 the layout resolver gives the Notes list pane 42 columns, and the select strip composed `0 selected`(11) + `Done`(9) + `All N`(10) + `Clear`(10) + `○ Export`(12) = 52. Reproduced in a canvas harness at that exact width -- Export landed at x=41 of 42, one cell showing.

AC#1/#2/#5: the in-strip `#library-notes-selected-count` is hidden (`display = not compact`, flipped in place by `apply_compact_presentation` on a breakpoint crossing) because the same count is already printed on its own line directly below. Measured after: Done 1-9, All 10-19, Clear 20-29, Export 30-42. It stays MOUNTED rather than dropped: `_apply_library_row_toggle` is the shared in-place counter patcher for Conversations/Media/Notes and queries it by id; a missing widget there falls back to a full recompose, which `test_notes_per_click_updates_keep_screen_and_canvas_identity` forbids.

Scoped deliberately, and recorded in AC#2: the WIDE layout still mounts both counters. "One source of truth" for them is peer task-32272, whose wave-3 Task 6 brief names the same select-mode counters in the same file. Sequencing note for whoever lands that: if 32272 removes `#library-notes-selection-status`, this compact hide must go with it or compact loses its counter entirely.

AC#3 REVISED (see the AC text): the glyph is `LIBRARY_DISABLED_ACTION_MARKER`, the Library-wide disabled-action marker task-32235 shipped -- documented in `Docs/User_Guide/library.md`'s legend table and pinned by `test_one_meaning_per_library_glyph`. Dropping it from disabled actions (which task-32235's own AC#1 text also asks for) is a change across every Library canvas, its guide pages and ~20 test files, and belongs to the peer that owns that legend. Pinned on this strip instead: the marker appears only on a disabled control that carries a reason, and never a `☐`/`☑`.

AC#4 was already delivered by task-32355 (`LibraryAdaptiveReaderPaneGrip` sets `name`/`tooltip` and paints the name down its column); pinned on the Notes route, which the crit-10 test only covered on Conversations.

Live at 100x30: `wave3-caps/layout/22-compact-select-strip.txt`.

Modified: `tldw_chatbook/Widgets/Library/library_notes_canvas.py`, `Tests/UI/test_library_notes_w3_layout.py`, `Docs/User_Guide/library/notes.md`.
AC ORDERING (review finding F4): the AC revision above was written and saved BEFORE the code it describes -- the task files' `updated_date: '2026-09-11 15:42'` UTC precedes the first code commit (`08:44:44 -0700` = `15:44:44` UTC) by two minutes. The commit graph cannot show it, because the six task files were staged together in one hygiene commit at the end; recorded here so the ordering is a fact in the file rather than a claim in a report.
<!-- SECTION:NOTES:END -->
