---
id: TASK-39110
title: "Personal Context's databases have no backup owner, so a profile that has used them previews as incomplete with an unknown owner, although the interview-drafts policy says that store is excluded from backup"
status: In Progress
assignee: ['@claude']
created_date: 2026-11-07 02:03
updated_date: 2026-11-07 06:25
labels:
- follow-up
- backup
- recovery
- personal-context
- needs-owner-decision
priority: medium
references:
- tldw_chatbook/DB/private_sqlite.py
- tldw_chatbook/Personal_Context/interview_launch.py
- tldw_chatbook/Backup_Recovery/owner_registry.py
- tldw_chatbook/Backup_Recovery/inventory.py
- tldw_chatbook/Notes/recovery.py
- tldw_chatbook/Workspaces/project_index/recovery.py
- backlog/docs/backup-recovery-owner-inventory.md
dependencies:
- TASK-39053
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in round 11 of the 2026-10-23 burn-down; checked on dev 2684e5453f, 2026-11-07, and on round-11 branch fu/r11-backupinv, which lands TASK-39053. Line numbers are dev's.

**The ruling.** Owner decision (2026-11-06), TASK-39053 row 13: "Keep unsupported + follow-up." The follow-up names `db.workspace_project_index` and `personal_context.interview_drafts`.

**`db.workspace_project_index` is already handled.** Its exclusion is installed and tested:
- `ProjectIndexCacheDeclaration` (`Workspaces/project_index/recovery.py:9`) marks the tree `intentionally_excluded`, installed through the Workspaces factory (`Backup_Recovery/owner_registry.py:72`);
- `Tests/Backup_Recovery/test_workspace_project_index_owner.py` tests it.
- Only the inventory doc and TASK-39053's follow-up miss it.

**Personal Context is not handled.**
- Its interview-drafts policy says "Encrypted interview drafts are local-only and excluded from centralized backup." (`DB/private_sqlite.py:485`).
- But no recovery owner declares the file (`interview_launch.py:105`), and `Personal_Context` is not in `owner_registry.py`'s factory list.
- `personal_context.repository`'s database is in the same position. TASK-24400 (Done) says that store "remains outside centralized backup".

**Probe.** Run on dev (scratch profile, real `discover()` after `install_adapters()`):
- `db.workspace_project_index` and `notes.sync_state` (`_DeviceStateExclusion`, the precedent) come out `intentionally_excluded`;
- `tldw_chatbook_personal_context.db` and `tldw_chatbook_personal_context_interviews.db` come out owner `unknown`, `unsupported`;
- the preview is incomplete, with `unsupported_owner`.

So any profile that has used Personal Context gets an incomplete backup preview with an unknown owner, for files the code says are excluded.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A backup preview of a profile holding the interview-drafts database (and its `-wal`/`-shm`/`-journal`) lists those files under `personal_context.interview_drafts` as intentionally excluded, not `unknown`, and capture of them is refused; a test is red before the change, showing the `unknown` row
- [ ] #2 The owner decides whether `personal_context.repository`'s database is excluded the same way, kept as an explicit `unsupported` row, or captured; the decision is recorded and the preview follows it
- [ ] #3 The inventory doc names the installed exclusions for `db.workspace_project_index`, `notes.sync_state` and the Personal Context owners
- [ ] #4 The new tests run on a PR lane (never the fast PR contract), with three timings of that step recorded
<!-- AC:END -->

## Implementation Plan

1. Record the owner decision (2026-11-06) for `personal_context.repository`: capture it, in its own design task; nothing is captured this round.
2. Red first: a real `discover()` after `install_adapters()` of a profile holding the interview-drafts database and its `-wal`/`-shm`/`-journal`, and the repository database, shows them as owner `unknown` today.
3. Generalize the `notes.sync_state` exclusion (`Notes/recovery.py` `_DeviceStateExclusion`) into one Backup_Recovery-local declaration of an exact SQLite store plus sidecars under the user data dir, and install it for `notes.sync_state` (unchanged behaviour) and `personal_context.interview_drafts` (intentionally excluded, capture/validate/relocate refused). Declare `personal_context.repository` with the same class as an explicit `unsupported` row (present files block, absent files are unused). The declarations live in `Backup_Recovery/config_adapter.py`'s factory, so installing them imports nothing from the Personal Context package.
4. Name the installed exclusions (`db.workspace_project_index`, `notes.sync_state`, the Personal Context owners) in the inventory doc.
5. Put the tests in `Tests/Architecture/test_backup_owner_inventory.py` (already on two lanes) and time its lane step three times.
