---
id: TASK-31976
title: 'Library media: keyboard focus on a list row is invisible against selection'
status: To Do
assignee: []
created_date: '2026-09-07 22:48'
labels:
  - library
  - media
  - ux
  - css
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Critique #6 P1 (lead), both assessors. `.library-media-row:focus` and `.library-media-row-selected` share the identical treatment in `_agentic_terminal.tcss` (`background: $ds-focus-bg; color: $ds-focus-fg; text-style: bold underline`), so the row the keyboard is ON is visually indistinguishable from the row that is selected/open. Assessment B measured the focused-row background against the open-item row and found them separated by three units in a single colour channel (rgb(28,70,102) vs rgb(25,68,102)) with bold+underline shared and no glyph, so arrow-key navigation reads as a dead key. In a keyboard-first product this is the sharpest contradiction of the brand. Applies to the sibling row canvases (conversations/notes/prompts) too, which share the same idiom.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A focused list row is visually distinct from a selected/open row at 235x52 and 100x30, with a cue that is not colour-only (a glyph or a clearly separated background, never colour-alone)
- [ ] #2 Arrow-key movement between rows is visible without a selection change (a painted pin over the real screen asserts the focused row's cue moves on Down/Up)
- [ ] #3 The compact 2-cell row keeps its label intact (no outline: heavy regression)
- [ ] #4 The same distinction holds for the conversations, notes and prompts row canvases
<!-- AC:END -->
