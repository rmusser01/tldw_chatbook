---
id: TASK-32244
title: >-
  Library Notes lasting sync admission rejects every file whose primary group is
  not the process egid
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
- [x] #1 A file the caller can write is admitted regardless of whether its primary group equals the process egid (owner match and supplementary-group membership both count)
- [x] #2 A file that genuinely cannot be written still reports `unsupported_metadata`, with the reason naming what is wrong
- [x] #3 `root_discovery_incomplete` names how many files were rejected and the dominant reason, not only that discovery was incomplete
- [x] #4 Covered by a test over a file whose group is not the caller's egid but which the caller can write
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Confirm the group check rejects a writable file (own uid, group wheel) under /private/tmp.
2. RED test over a snapshot whose owner_group is not the caller egid but whose owner/mode make it writable.
3. Replace the egid equality with an honest writability test from the snapshot's own owner/group/mode bits (owner match, supplementary-group membership, other bits).
4. Aggregate the per-file refusals so root_discovery_incomplete names the count and the dominant reason instead of raising on the first refusal.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
`_metadata_issue` demanded `owner_user == os.geteuid() and owner_group == os.getegid()`. The second half is not a writability test: a file you own whose group is any other group you belong to -- or, as in this harness, `wheel` under /private/tmp -- was refused, and the root then failed `root_discovery_incomplete` with no count and no reason. The check now asks the permission bits the snapshot already carries: owner match uses S_IWUSR, supplementary-group membership uses S_IWGRP, otherwise S_IWOTH. A file that genuinely cannot be written still reports `unsupported_metadata` (it previously did NOT -- a 0444 file you own was admitted and would have failed at replace time).

`observe_root` no longer raises on the first refused file. It counts refusals across the walk (`collections.Counter`) and raises `NotesSyncRootRefused("root_discovery_incomplete", detail="N of M files refused; K for <reason>")`; the discovery-level failure names its count too. Worst case is one full walk, which is what a good root already costs.

Proof: before, a vault under /private/tmp (group wheel) refused all files and failed the root; after, the same vault is admitted (scratchpad repro2.py, wave3-caps/sync/repro-after-reasons.txt).

Files: tldw_chatbook/Notes/notes_sync_filesystem.py, tldw_chatbook/Notes/notes_sync_runtime.py, Tests/Notes/test_notes_sync_filesystem.py, Tests/Notes/test_notes_sync_runtime.py.
<!-- SECTION:NOTES:END -->
