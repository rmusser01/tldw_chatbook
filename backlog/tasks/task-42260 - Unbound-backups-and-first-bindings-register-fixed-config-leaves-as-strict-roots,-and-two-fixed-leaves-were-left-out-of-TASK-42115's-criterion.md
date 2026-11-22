---
id: TASK-42260
title: "Unbound backups and first bindings register fixed config leaves as strict roots, and two fixed leaves were left out of TASK-42115's criterion"
status: To Do
assignee: []
created_date: 2026-11-22 00:17
updated_date: 2026-11-22 00:17
labels:
- follow-up
- backup
priority: low
references:
- tldw_chatbook/Backup_Recovery/profile_paths.py
- tldw_chatbook/Backup_Recovery/effective_roots.py
- backlog/tasks/task-42115 - A-default-location-Replace-cannot-create-or-start-the-pet-make-pet-files-config-companions-with-a-fixed-file-set.md
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by the backupdef stream (TASK-42115, round 19). TASK-42115's AC#3 was rescoped to bound profiles, so an unbound backup and a first binding still register fixed config leaves (the fixed file set) as strict roots rather than as file roots that may be deleted. The `notes.templates` and `rag.definitions` fixed leaves were also left out of AC#5. Read from the TASK-42115 file on `fu/r19-backupdef`; the registration code was not re-read for this filing.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 An unbound backup and a first binding register each fixed config leaf as the kind of root a bound profile registers, or the difference is recorded with its reason
- [ ] #2 `notes.templates` and `rag.definitions` are covered the same way as the other fixed leaves
- [ ] #3 A test per path (unbound backup, first binding) is red before the fix
<!-- AC:END -->
