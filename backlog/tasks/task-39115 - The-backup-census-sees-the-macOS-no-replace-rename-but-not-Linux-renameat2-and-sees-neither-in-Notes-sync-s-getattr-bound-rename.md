---
id: TASK-39115
title: "The backup census sees the macOS no-replace rename but not Linux renameat2, and sees neither in Notes sync's getattr-bound rename"
status: To Do
created_date: 2026-11-07 02:03
labels:
- follow-up
- backup
- architecture
- tests
priority: low
references:
- Tests/Architecture/test_backup_owner_inventory.py
- tldw_chatbook/Backup_Recovery/native_platform.py
- tldw_chatbook/TTS/profile_migration_namespace.py
- tldw_chatbook/Notes/sync_paths.py
dependencies:
- TASK-39053
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in round 11 of the 2026-10-23 burn-down; checked on dev 2684e5453f, 2026-11-07, with the census from round-11 branch fu/r11-backupinv (TASK-39053). Line numbers are dev's.

- The census vocabulary has `renameatx_np` (macOS) but not Linux `renameat2`. On the branch its alias resolver (`native_candidate`) recognises only `renameatx_np`.
- With `renameat2` added by name, a probe found one new row, `TTS/profile_migration_namespace._rename_noreplace` (`:163`).
- `Backup_Recovery/native_platform.py:32` (`fn, flags = libc.renameat2, 1`) stays invisible through its tuple alias.
- `Notes/sync_paths._rename_with_flags` (`:237-258`) looks its function up with `getattr(libc, function_name)` and has no row on either platform.
- So the Linux no-replace publication path, and the Notes sync rename on both platforms, are outside the owner review.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Linux `renameat2` calls, direct and through a tuple alias, are counted the way `renameatx_np` is; a unit test is red before and green after
- [ ] #2 `_rename_with_flags` has a reviewed row, or is listed as an owner-accepted dynamic-binding limit; the decision is recorded
<!-- AC:END -->
