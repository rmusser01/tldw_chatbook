---
id: TASK-39115
title: "The backup census sees the macOS no-replace rename but not Linux renameat2, and sees neither in Notes sync's getattr-bound rename"
status: Done
assignee: ['@claude']
created_date: 2026-11-07 02:03
updated_date: 2026-11-10 18:23
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
- [x] #1 Linux `renameat2` calls, direct and through a tuple alias, are counted the way `renameatx_np` is; a unit test is red before and green after
- [x] #2 `_rename_with_flags` has a reviewed row, or is listed as an owner-accepted dynamic-binding limit; the decision is recorded
<!-- AC:END -->

## Implementation Plan

1. Unit tests first: direct `libc.renameat2(...)`, the tuple alias `fn, flags = libc.renameat2, 1`, and `function = getattr(libc, name)` where `name` is a string constant (or a conditional of two) are counted like `renameatx_np`.
2. Census: add `renameat2` to the vocabulary and the native alias resolver; resolve `getattr(obj, <string constant or local string alias>)` to the producer name.
3. Recompute; give `_rename_noreplace`, `rename_noreplace` and `_rename_with_flags` reviewed rows.

## Implementation Notes

The census now counts Linux `renameat2` the way it counts `renameatx_np`: called directly, through a tuple alias (`fn, flags = libc.renameat2, 1`), and through `getattr(obj, name)` where `name` is a string constant, a conditional of constants, or a local name bound to them. The `getattr` form resolves as `obj.<name>` for each string the name can hold, in the assigning function's lexical scope.

- New rows: `TTS/profile_migration_namespace._rename_noreplace` renameat2 1 and `Backup_Recovery/native_platform.rename_noreplace` renameat2 1 (both take their symbol's reviewed disposition), plus `Notes/sync_paths._rename_with_flags` renameatx_np 1 and renameat2 1.
- Decision (AC#2): `_rename_with_flags` has a reviewed row; no dynamic-binding limit is listed. It is `external_input` / `external-notes-workspace-or-export`, because every caller is a `PinnedSyncRoot` method renaming inside the pinned Notes sync root, as the module's other rows are. `guarded_rename_available` (`getattr(...) is not None`, never called) gains no row.
- Red/green (AC#1): `test_census_counts_linux_renameat2_like_renameatx_np` failed on the base scanner with `assert 0 == 1` at the direct call and passes now; its `probe` function pins that a getattr existence check is not a producer.
- Files: `Tests/Architecture/test_backup_owner_inventory.py`, `backlog/docs/backup-recovery-owner-inventory.md`.
