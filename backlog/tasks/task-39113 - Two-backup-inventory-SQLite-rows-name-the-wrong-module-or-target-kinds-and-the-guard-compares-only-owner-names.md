---
id: TASK-39113
title: "Two backup-inventory SQLite rows name the wrong module or target kinds, and the guard compares only owner names"
status: Done
assignee: ['@claude']
created_date: 2026-11-07 02:03
updated_date: 2026-11-07 05:56
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
- [x] #1 Every `sqlite:` row's module and target-kind columns match the registry
- [x] #2 The guard fails when a row's module or target kinds differ from the registry; the red is shown on the two stale rows before they are corrected
- [x] #3 The test stays on its PR lane, with its time recorded
<!-- AC:END -->

## Implementation Plan

1. Extend `test_sqlite_registry_members_have_explicit_census_rows` to compare each `sqlite:` row's module and target-kind columns with `_SQLITE_OWNER_POLICIES` (parsed from the AST), not only the owner names.
2. Run it against the unchanged doc to show the red on `notes.sync_state` and `notifications.event_state`.
3. Correct the two doc rows; rerun the file serially and time it on its lane step.

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
**Approach.** `test_sqlite_registry_members_have_explicit_census_rows` (`Tests/Architecture/test_backup_owner_inventory.py`) still checks the name sets first, then compares every `_SQLITE_OWNER_POLICIES` entry's `(production_module, kinds constant)` with its doc row's module and target-kind columns. Both sides are read without importing the registry: the policies from the AST (positional args 0 and 1), the rows by regex. The doc's fourth column (cohort/status) is review prose and is not compared.

**Red first.** Before the doc fix the extended test failed on exactly the two rows the task names: `notes.sync_state` (`('tldw_chatbook/Notes/note_import_receipts', '_PRIVATE_FILE')` vs `('tldw_chatbook/Notes/notes_device_state_store', '_PRIVATE_AND_READ_ONLY')`) and `notifications.event_state` (`_MEMORY` vs `_PRIVATE_OR_MEMORY`). Green after correcting both rows in `backlog/docs/backup-recovery-owner-inventory.md`.

**Lane and time (AC#3).** The file stays on its two lanes (dreams-boundary "Run Dreams admission-sensitive contracts", and the backup-recovery qualification workflow); no lane pin changed. The node alone: 0.96 s serial (load 35-41). The whole file serially: 18 passed in 49.3 s at load 35-44. Step timings are in TASK-39110's notes (same step, measured once for this round's additions).

**Files.** `Tests/Architecture/test_backup_owner_inventory.py`, `backlog/docs/backup-recovery-owner-inventory.md`.
<!-- SECTION:NOTES:END -->
