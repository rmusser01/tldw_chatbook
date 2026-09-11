---
id: TASK-32143
title: >-
  Idea: Library Notes editor chrome strip with word count, cursor line, save
  state and the main actions without tabbing
status: In Progress
assignee:
  - '@claude'
created_date: '2026-09-08 21:39'
updated_date: '2026-09-11 16:34'
labels:
  - library
  - notes
  - ux
  - critique-notes-2026-09
  - idea
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Improvement pitched by the design assessor for the power user: a status strip at the bottom of the body (where terminal users look) carrying words, cursor line and save state, plus Save / Preview / Keywords / Delete reachable without Tab. Today 'Saved' floats above the mode tabs and the word count lives under Info. Size S. Evidence: Library ▸ Notes critique snapshot `.impeccable/critique/2026-09-09T04-24-50Z__w-chatbook-widgets-library-library-notes-canvas-py.md` (dev c4a7b1911f, 2026-09-08/09, dual-agent live review at 235x52/100x30/60x24 on a fresh and a seeded profile with a 71-file Obsidian-style vault; parent re-tested every disagreement on a third profile).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Design agreed with the user before implementation
- [ ] #2 The strip replaces the floating meta line rather than adding to it
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Live-reproduce the editor chrome at 235x52: confirm which line 'floats' (AC#2's target).
2. Read every pin on the candidate widgets before moving anything.
3. RED test on the real editor route: the strip's facts cell is absent.
4. Add the facts cell to the existing save-state row (the floating line becomes the strip), fed by the word count the controller already computes plus the body TextArea's cursor.
5. Hide it below 80 columns; never focusable.
6. GREEN; live captures at 235x52 and 100x30 and below 80; guide + stamp; CSS bundle sync + boot-CSS budget.
<!-- SECTION:PLAN:END -->
