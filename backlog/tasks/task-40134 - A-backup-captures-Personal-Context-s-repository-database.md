---
id: TASK-40134
title: "A backup captures Personal Context's repository database"
status: To Do
created_date: 2026-11-10 17:13
labels:
- follow-up
- backup
- recovery
- personal-context
priority: medium
references:
- tldw_chatbook/Backup_Recovery/config_adapter.py
- tldw_chatbook/Backup_Recovery/credential_policies.py
- tldw_chatbook/Backup_Recovery/credentials.py
- tldw_chatbook/Backup_Recovery/data_groups.py
- tldw_chatbook/Sync_Interop/recovery.py
- tldw_chatbook/DB/private_sqlite.py
- backlog/docs/backup-recovery-owner-inventory.md
- Docs/superpowers/specs/2026-11-10-personal-context-backup-key-custody-design.md
dependencies:
- TASK-40133
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Filed in round 13 of the 2026-10-23 burn-down from TASK-39148's owner-approved design, `Docs/superpowers/specs/2026-11-10-personal-context-backup-key-custody-design.md`. Owner decision 2026-11-10 (AskUserQuestion): "Approve all 11". Checked on dev ae9f9e8d21, 2026-11-10; line numbers are dev's.

**Why.** Today the repository is `_LocalSQLite(..., "personal_context_repository_capture_unsupported", excluded=False)` (`Backup_Recovery/config_adapter.py:1163-1176`).
- A present file is an `unsupported` row, which blocks every backup preview of a profile that has used Personal Context (`config_adapter.py:705-712`, `inventory.py:57`).
- Owner ruling, 2026-11-06: "Capture it: back up the database and design key custody so a restore can read it."

**Approved design (decisions 1, 3, 7, 8, 9).**
- **Capture the file as it is.** It is ordinary SQLite with ciphertext columns, so it can be copied and schema-validated without the key (`Personal_Context/crypto.py:39-71`, `repository.py:246-261`).
  - Copy it lock-safely like `Sync_Interop/recovery.py`'s `_SyncAdapter` (`:49-90`).
  - Its private-SQLite policy lacks `centralized_backup_allowed` / `recovery_capture_allowed` (`DB/private_sqlite.py:470-475`).
  - Its schema version lives in `personal_context_schema`, not `user_version` (`repository.py:229-234`; `Backup_Recovery/sqlite_validation.py:1150-1188`).
  - The adapter replaces the `_LocalSQLite` row. Registering both under one owner id raises `conflicting_installed_owner` (`owner_registry.py:84-86`).
- **Group.** The database goes in a new selectable "Personal Context" group (`data_groups.py`). `Tests/Backup_Recovery/test_data_groups.py:353-354` lists the owner as non-selectable today.
- **Key.** The profile key travels as a `personal_context` credential record, referenced by `(personal_context.repository, profile_meta, key_account)` from TASK-40133.
  - It is captured only in `include` and `rollback` modes, both of which force encryption (`capture.py:301-302`).
  - A backup without credentials carries the database alone, and its preview says "database only: readable on this device".
  - The key is never in a readable archive.
- **First link in flight.** Capture is refused while a `first_link_freeze` or `first_link_rebaseline_commit` row exists (`repository.py:214-226`, `:311-320`).
- **Device-only records** are included, as **Export recovery copy** already does (decision 8).
- **Detached imports.** A `_SQLiteDeclaration` owner is picked up by the detached layout automatically (`detached_layout.py:636-642`, `:1080-1085`). Without a guard, that would import the repository into an archive without its key. Decision 10 refuses it there by name.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A backup preview of a profile holding the repository database (with its `-wal`) lists it as captured in a selectable "Personal Context" group, and the preview is no longer incomplete because of it (red first)
- [ ] #2 The archive holds a consistent copy of the database that passes the installed exact-schema validation for each v9 layout from TASK-40133
- [ ] #3 A backup that includes credentials carries the profile key as a `personal_context` credential record, and capture checks that the key opens the captured copy (its manifest authenticates) before writing the record
- [ ] #4 A backup without credentials carries no key material, and its preview says the database is readable only on this device
- [ ] #5 Inert extraction cannot write the key to disk from any archive; a test pins it
- [ ] #6 Capture is refused with a named reason, shown in the preview, while a first-link freeze or rebaseline marker exists
- [ ] #7 A detached source holding the repository is refused by name and never imported without its key
- [ ] #8 The owner inventory doc's Personal Context rows and the Personal Context User Guide page's backup paragraph describe the new behaviour
<!-- AC:END -->
