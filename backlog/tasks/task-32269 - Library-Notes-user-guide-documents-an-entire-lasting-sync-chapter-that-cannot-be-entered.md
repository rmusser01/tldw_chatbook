---
id: TASK-32269
title: >-
  Library Notes user guide documents an entire lasting-sync chapter that cannot
  be entered
status: Done
assignee:
  - '@claude'
created_date: '2026-09-10 18:05'
updated_date: '2026-09-11 15:32'
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
- [x] #1 The chapter states, in the user's terms, what is required for a folder to be admitted
- [x] #2 Every step in the chapter has been walked against a real admitted root before its 'Verified against' stamp is refreshed
- [x] #3 Nothing in the chapter describes a surface that cannot be reached from the shipped build
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Land the task-32243 fix first.
2. Walk the lasting-sync chapter live against a real admitted $HOME vault.
3. Rewrite the chapter: state the admission precondition in the user's terms, list the refusal reasons and their copy, remove or supersede claims that cannot run; refresh the stamp.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
The chapter could not be walked before task-32243 landed, so it was written from the design. Walked end to end on this branch at 235x52 against a 179-file git-backed vault under $HOME: refusal copy on an in-profile folder, Choose folder again, Check (60 safe / 0 attention), Activate ("60 applied · durable receipt recorded"), the notes appearing under a '⇄ Sync managed' folder with '⇄ Synced placement' badges, Manage sync folders (which only exists once a root is active), and a second Check on the persisted root ("Manual check finished").

Added to 'Add from files and lasting sync': what a folder has to be before it can be checked, and the named refusals with their copy. Added to 'Set up lasting folder sync': what a refusal looks like at step 4 and where Manage sync folders appears at step 5. Stamped with the walk.

One gap found and left unfixed, recorded in the stamp: the root row in Manage sync folders reads 'Sync folder (name unavailable before cutover)' instead of the display name the user typed.

Files: Docs/User_Guide/library/notes.md.
<!-- SECTION:NOTES:END -->
