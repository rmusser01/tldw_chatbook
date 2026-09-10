---
id: TASK-32244
title: >-
  Library Notes lasting sync admission rejects every file whose primary group
  is not the process egid
status: To Do
assignee: []
created_date: '2026-09-10 18:05'
labels:
  - library
  - notes
  - ux
  - critique-notes-2026-09
  - sync
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
PROVEN while bisecting task-32243. `notes_sync_filesystem.py:191-192` admits a file only when `owner_group == os.getegid()`. Any vault whose files carry another group -- `wheel` under `/private/tmp`, a shared project group, a copied or restored tree -- reports `unsupported_metadata` for every file, and the root then fails `root_discovery_incomplete`. In this harness all 60 fixture files failed exactly this way, which is what made the whole lasting-sync journey untestable for two runs.

The check is standing in for writability, and primary-group equality is not that test: a file the user *owns*, or one whose group is in the user's supplementary group list, is writable and is rejected anyway. The failure is also silent in aggregate -- `root_discovery_incomplete` says discovery did not finish, not that 60 of 60 files were refused for one reason.

Evidence: Library ▸ Notes critique snapshot `.impeccable/critique/2026-09-10T17-32-16Z__w-chatbook-widgets-library-library-notes-canvas-py.md` (dev e6cb464239, 2026-09-10, dual-assessor live review at 235x52 and 100x30/60x24 on two fresh profiles and one seeded profile with a git-backed 71-file Obsidian vault fixture; the reconciling parent re-tested every disagreement on a third pair of profiles and traced each surviving cause to code). Captures under the session scratchpad `notes-crit2/{C,D,R}/caps/`.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A file the caller can write is admitted regardless of whether its primary group equals the process egid (owner match and supplementary-group membership both count)
- [ ] #2 A file that genuinely cannot be written still reports `unsupported_metadata`, with the reason naming what is wrong
- [ ] #3 `root_discovery_incomplete` names how many files were rejected and the dominant reason, not only that discovery was incomplete
- [ ] #4 Covered by a test over a file whose group is not the caller's egid but which the caller can write
<!-- AC:END -->
