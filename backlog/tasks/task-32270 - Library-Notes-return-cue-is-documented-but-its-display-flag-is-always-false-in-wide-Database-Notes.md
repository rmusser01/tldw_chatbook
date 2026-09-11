---
id: TASK-32270
title: >-
  Library Notes return cue is documented but its display flag is always false in
  wide Database Notes
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
  - docs
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The guide claims: "When Library navigation is closed, one stable cue names the return destination: `< Library / Notes`". PROVEN absent at 235 columns -- `library_browse_route_swap.py:133` sets `task_return.display = wide_focused_task`, and `wide_focused_task` is False whenever `adaptive_database_notes` is true, which is always, in wide Database Notes. So the control the guide describes can never render in the state the guide describes.

Either the cue should render there, or the guide should stop promising it; both are small, and the choice belongs with whoever owns the back-cue grammar (task-32139 shipped the compact half).

Evidence: Library ▸ Notes critique snapshot `.impeccable/critique/2026-09-10T17-32-16Z__w-chatbook-widgets-library-library-notes-canvas-py.md` (dev e6cb464239, 2026-09-10, dual-assessor live review at 235x52 and 100x30/60x24 on two fresh profiles and one seeded profile with a git-backed 71-file Obsidian vault fixture; the reconciling parent re-tested every disagreement on a third pair of profiles and traced each surviving cause to code). Captures under the session scratchpad `notes-crit2/{C,D,R}/caps/`.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The cue renders in wide Database Notes with navigation closed, or the guide no longer claims it does
- [ ] #2 Covered by a test asserting the cue's presence or absence in the state the guide describes
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Prove the display flag: library_browse_route_swap's wide_focused_task requires not adaptive_database_notes
2. Take the AC's documentation branch -- the cue is a compact control (task-32136/32139)
3. Correct the three guide sentences that promise it on a wide terminal
4. Pin both the absence in wide Database Notes and the guide's own wording
<!-- SECTION:PLAN:END -->
