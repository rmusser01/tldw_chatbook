---
id: TASK-39113
title: "Two backup-inventory SQLite rows name the wrong module or target kinds, and the guard compares only owner names"
status: To Do
created_date: 2026-11-07 02:03
labels:
- follow-up
- backup
- architecture
- tests
priority: low
references:
- backlog/docs/backup-recovery-owner-inventory.md
- Tests/Architecture/test_backup_owner_inventory.py
- tldw_chatbook/DB/private_sqlite.py
dependencies:
- TASK-39053
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in round 11 of the 2026-10-23 burn-down; checked on dev 2684e5453f, 2026-11-07, and on round-11 branch fu/r11-backupinv (TASK-39053), where the doc rows are unchanged. Line numbers are dev's.

A probe on round-11 branch fu/r11-backupinv, where TASK-39053 reconciles the names, compared the doc's 93 `sqlite:` rows with the registry's 93 members. On dev 2684e5453f the counts still differ (89 rows, 94 members). On the branch the names match, but two rows' columns do not, and the rows below are unchanged there.

| Owner | Doc row (`backup-recovery-owner-inventory.md`) | Registry (`DB/private_sqlite.py`) | Changed by |
|---|---|---|---|
| `notes.sync_state` | `:171`: `Notes/note_import_receipts`, `_PRIVATE_FILE` | `:453`: `Notes/notes_device_state_store`, `_PRIVATE_AND_READ_ONLY` | aa579990df |
| `notifications.event_state` | `:173`: `_MEMORY` | `_PRIVATE_OR_MEMORY` | 8c0567168d, TASK-21131 |

- `test_sqlite_registry_members_have_explicit_census_rows` compares the set of names only, so the review doc can drift from what each owner actually opens.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Every `sqlite:` row's module and target-kind columns match the registry
- [ ] #2 The guard fails when a row's module or target kinds differ from the registry; the red is shown on the two stale rows before they are corrected
- [ ] #3 The test stays on its PR lane, with its time recorded
<!-- AC:END -->
