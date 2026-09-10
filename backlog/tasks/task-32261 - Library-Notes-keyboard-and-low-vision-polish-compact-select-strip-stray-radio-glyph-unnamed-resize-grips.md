---
id: TASK-32261
title: >-
  Library Notes keyboard and low-vision polish: compact select strip, stray
  radio glyph, unnamed resize grips
status: To Do
assignee: []
created_date: '2026-09-10 18:05'
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
- [ ] #2 The selection count is printed once, and is separated from the adjacent focused button
- [ ] #3 No radio-like glyph renders on a control that is not a radio or a checkbox
- [ ] #4 The resize grips carry an accessible name
- [ ] #5 Covered by a test asserting the compact select-strip contents
<!-- AC:END -->
