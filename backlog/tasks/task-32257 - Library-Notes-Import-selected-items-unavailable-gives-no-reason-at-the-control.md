---
id: TASK-32257
title: >-
  Library Notes: Import selected items unavailable gives no reason at the
  control
status: To Do
assignee: []
created_date: '2026-09-10 18:05'
labels:
  - library
  - notes
  - ux
  - critique-notes-2026-09
  - import
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The disabled control states that it is unavailable and nothing else, so there is no way to learn what would make it available.

What makes this a defect rather than a nit is that the correct grammar already exists four panels away on the same screen: the sync setup's disabled radio reads "Unavailable - server sync-folder capability not installed", and Session Git's disabled Commit reads "Stage at least one session note to commit". Disabled controls carrying their reason as text is one of this screen's genuine strengths; this control is the exception.

Evidence: Library ▸ Notes critique snapshot `.impeccable/critique/2026-09-10T17-32-16Z__w-chatbook-widgets-library-library-notes-canvas-py.md` (dev e6cb464239, 2026-09-10, dual-assessor live review at 235x52 and 100x30/60x24 on two fresh profiles and one seeded profile with a git-backed 71-file Obsidian vault fixture; the reconciling parent re-tested every disagreement on a third pair of profiles and traced each surviving cause to code). Captures under the session scratchpad `notes-crit2/{C,D,R}/caps/`.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The disabled Import control carries its reason as text at the control
- [ ] #2 The reason names what would make it available
- [ ] #3 Covered by a test asserting the reason string in the disabled state
<!-- AC:END -->
