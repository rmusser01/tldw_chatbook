---
id: TASK-32586
title: 'Library Notes: lasting sync needs a folder-creation door of its own'
status: To Do
assignee: []
created_date: '2026-09-14 22:48'
labels:
  - library
  - notes
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Proven blocked twice against a live database by wave-4 group 2, which is why task-32535's AC#4 was amended rather than attempted. A synced note cannot keep the vault's folder tree today: create_folder refuses a manual child of a subtree that holds a managed placement (_require_manual_folder_subtree raises FolderCapabilityError sync_managed_folder), and a folder created some other way then reads has_managed_folder_ownership, so NotesScopeSyncAuthority._verified_folder rejects it as folder_authority_changed on the next run — a nested tree would collapse to flat on the following sync. So every synced note sits directly in the sync-managed root folder and the review says so. Giving lasting sync its own folder-creation path, authorised as the sync owner rather than as a manual edit, is what unblocks the vault hierarchy.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Lasting sync can create folders inside its own managed subtree without going through the manual-folder refusal
- [ ] #2 A folder it creates passes _verified_folder on the next run rather than reading as folder_authority_changed
- [ ] #3 A vault with nested directories syncs to a matching Library folder tree and stays nested across a second check
- [ ] #4 notes.md's 'Synced notes do not keep the vault folder tree' paragraph is updated to what ships
<!-- AC:END -->
