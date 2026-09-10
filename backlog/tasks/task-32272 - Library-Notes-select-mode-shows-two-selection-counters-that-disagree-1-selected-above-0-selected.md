---
id: TASK-32272
title: >-
  Library Notes select mode shows two selection counters that disagree: 1
  selected above 0 selected
status: To Do
assignee: []
created_date: '2026-09-10 18:05'
labels:
  - library
  - notes
  - ux
  - critique-notes-2026-09
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Observed by the peer session at 235x52 on a seeded profile at dev `3315241674`, and not filed anywhere: in Notes select mode the toolbar reads "1 selected" while the line directly below it reads "0 selected". Two counters for one selection, on the same pane, at the same moment, disagreeing -- and the count is the only feedback the mode gives about what a bulk action will touch.

Cause INFERRED, not traced: two independently maintained counts, one updated on the row toggle and one on the selection-set change.

Related but distinct: task-32261 covers the same strip printing `0 selected` twice and losing "Export selected" at 100x30. This task is the two counters holding **different** numbers.

Evidence: Library ▸ Notes critique snapshot `.impeccable/critique/2026-09-10T17-32-16Z__w-chatbook-widgets-library-library-notes-canvas-py.md` (dev e6cb464239, 2026-09-10, dual-assessor live review at 235x52 and 100x30/60x24 on two fresh profiles and one seeded profile with a git-backed 71-file Obsidian vault fixture; the reconciling parent re-tested every disagreement on a third pair of profiles and traced each surviving cause to code). Captures under the session scratchpad `notes-crit2/{C,D,R}/caps/`.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Every visible selection count on the Notes canvas reads the same number at all times
- [ ] #2 There is one source of truth for the count, pinned by a test that toggles a single row and asserts both labels
- [ ] #3 Verified live at 235x52 with a capture
<!-- AC:END -->
