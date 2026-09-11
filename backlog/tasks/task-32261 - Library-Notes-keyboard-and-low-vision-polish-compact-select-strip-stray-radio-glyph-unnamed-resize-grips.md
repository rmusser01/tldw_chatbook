---
id: TASK-32261
title: >-
  Library Notes keyboard and low-vision polish: compact select strip, stray
  radio glyph, unnamed resize grips
status: In Progress
assignee:
  - '@claude'
created_date: '2026-09-10 18:05'
updated_date: '2026-09-11 15:42'
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
- [ ] #1 Export selected is present at 100x30, or the guide stops claiming it
- [ ] #2 The selection count is printed once on the compact strip the critique measured, on its own line rather than jammed against the focused Done -- AC SCOPED (task-32261 implementation): the wide layout still mounts both counters, and "one source of truth" for them is peer task-32272 (wave-3 Task 6), whose brief names the same select-mode counters in the same file
- [ ] #3 The Notes canvas follows the Library glyph legend: the circle marks a DISABLED action (beside its reason), `☐/☑` mark selection, and neither is used for the other -- AC REVISED (task-32261 implementation): the circle is `LIBRARY_DISABLED_ACTION_MARKER`, a Library-wide decision task-32235 shipped, documented in `Docs/User_Guide/library.md`'s legend table and pinned by `test_one_meaning_per_library_glyph`. Dropping it from disabled actions (which task-32235's own AC#1 text also asks for) is a Library-wide change across every canvas, its guide pages and ~20 test files -- it belongs to the peer that owns that legend, not to the Notes instance
- [ ] #4 The resize grips carry an accessible name
- [ ] #5 Covered by a test asserting the compact select-strip contents
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Measure the compact select strip at the 42-column Items pane a 100x30 terminal resolves
2. Hide the in-strip duplicate counter in compact so Export selected fits the pane (the wide duplicate is peer task-32272's)
3. Confirm the pane grips already carry an accessible name (task-32355) and pin it on the Notes route
4. Re-scope AC#3: the circle marker is the Library-wide disabled-action legend task-32235 shipped
5. RED->GREEN compact strip test; guide + stamp
<!-- SECTION:PLAN:END -->
