---
id: TASK-32260
title: >-
  Library opens in 12.6 s on a seeded profile of 27 items versus 2.7 s on a
  fresh one
status: To Do
assignee: []
created_date: '2026-09-10 18:05'
labels:
  - library
  - notes
  - ux
  - critique-notes-2026-09
  - performance
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Measured by the evidence assessor across the two profiles used for this review: a fresh profile opens Library in 2.7 s; the seeded profile (10 notes, 11 media, 6 conversations) takes 12.6 s. Twenty-seven items is not a data-volume effect at any plausible per-item cost, so something on the seeded path is doing per-open work the empty path skips. Cause untraced.

Filed because a 12.6 s open on a trivially small corpus predicts a considerably worse number on a real one, and because everything else measured on this screen was fast (35 KB note in 0.7 s, 179-path vault tree in 0.39 s, 59-file import under 2 s to review).

Evidence: Library ▸ Notes critique snapshot `.impeccable/critique/2026-09-10T17-32-16Z__w-chatbook-widgets-library-library-notes-canvas-py.md` (dev e6cb464239, 2026-09-10, dual-assessor live review at 235x52 and 100x30/60x24 on two fresh profiles and one seeded profile with a git-backed 71-file Obsidian vault fixture; the reconciling parent re-tested every disagreement on a third pair of profiles and traced each surviving cause to code). Captures under the session scratchpad `notes-crit2/{C,D,R}/caps/`.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The cause of the seeded-versus-fresh open-time gap is identified and named
- [ ] #2 Library opens on a 27-item profile within a documented budget
- [ ] #3 A regression test or a recorded measurement pins the open time
<!-- AC:END -->
