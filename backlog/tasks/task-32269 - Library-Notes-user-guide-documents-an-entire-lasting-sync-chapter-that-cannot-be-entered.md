---
id: TASK-32269
title: >-
  Library Notes user guide documents an entire lasting-sync chapter that cannot
  be entered
status: In Progress
assignee:
  - '@claude'
created_date: '2026-09-10 18:05'
updated_date: '2026-09-11 15:07'
labels:
  - library
  - notes
  - ux
  - critique-notes-2026-09
  - docs
  - sync
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
`Docs/User_Guide/library/notes.md` documents the lasting-sync chapter in full: Check -> review -> Activate -> conflicts -> Manage sync folders -> Pause/Resume. Live, `Check changes` failed on every folder tried by all three assessors, no root can exist, and Manage sync folders never appears (task-32243). The guide gives the most space of any chapter to the one of the three worlds no reader in this configuration can enter.

Until task-32243 lands, the chapter needs a stated precondition or a known-limitation note. After it lands, the chapter needs an actual walk against a real admitted root before its stamp is refreshed -- the whole chapter is currently written from the design, not from the surface.

Evidence: Library ▸ Notes critique snapshot `.impeccable/critique/2026-09-10T17-32-16Z__w-chatbook-widgets-library-library-notes-canvas-py.md` (dev e6cb464239, 2026-09-10, dual-assessor live review at 235x52 and 100x30/60x24 on two fresh profiles and one seeded profile with a git-backed 71-file Obsidian vault fixture; the reconciling parent re-tested every disagreement on a third pair of profiles and traced each surviving cause to code). Captures under the session scratchpad `notes-crit2/{C,D,R}/caps/`.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The chapter states, in the user's terms, what is required for a folder to be admitted
- [ ] #2 Every step in the chapter has been walked against a real admitted root before its 'Verified against' stamp is refreshed
- [ ] #3 Nothing in the chapter describes a surface that cannot be reached from the shipped build
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Land the task-32243 fix first.
2. Walk the lasting-sync chapter live against a real admitted $HOME vault.
3. Rewrite the chapter: state the admission precondition in the user's terms, list the refusal reasons and their copy, remove or supersede claims that cannot run; refresh the stamp.
<!-- SECTION:PLAN:END -->
