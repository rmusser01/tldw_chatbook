---
id: TASK-32270
title: >-
  Library Notes return cue is documented but its display flag is always false
  in wide Database Notes
status: To Do
assignee: []
created_date: '2026-09-10 18:05'
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
