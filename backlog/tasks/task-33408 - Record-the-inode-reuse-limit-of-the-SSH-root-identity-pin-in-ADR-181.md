---
id: TASK-33408
title: Record the inode-reuse limit of the SSH root identity pin in ADR-181
status: Done
assignee:
  - '@claude'
created_date: '2026-09-28 20:30'
updated_date: '2026-09-28 16:52'
labels:
  - console
  - workspaces
dependencies:
  - TASK-33202
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The root pin identifies a root by (st_dev, st_ino). The live UAT for PR #2879 showed ext4 handing a deleted folder's inode straight back to a folder recreated at the same path, so the pin cannot tell them apart and serves the new folder without a STALE_IDENTITY stop. ADR-181 does not state this limit. Deferred from PR #2879's final review (ruling R14).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 ADR-181 states the inode-reuse limit, what the pin still guarantees, and why it is accepted
<!-- AC:END -->

## Implementation Notes

Inode-reuse limit documented in ADR-181, Decision section: `(st_dev, st_ino)` names a directory, not its history. Filesystems that hand freed inodes straight back (observed on ext4, 2026-09-27 live host) make recreated directories indistinguishable from originals. Stronger identity (ext4's `i_generation`) is filesystem-specific and not stdlib-portable. The pin defeats symlink swaps and escapes from root — same limit as local bindings. See `backlog/decisions/181-ssh-remote-workspace-bindings.md`.
