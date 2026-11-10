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
  - The key is never in a readable archive.
  - A database whose reference is still empty carries no key record.
- **The reference is not a secret.** Exclude mode is the default, and it rewrites every `SQLITE_CREDENTIAL_COLUMNS` owner's database in staging (`credentials.py:1100-1101`). It turns a citation column into `recovery:setup_required` (`credentials.py:704-706`).
  - The Personal Context reference must be kept unchanged there, never sanitized. Otherwise a later same-device restore of a key-less backup would lock for good, contrary to decision 3.
  - A backup without credentials carries the database alone, and its preview says its key is not in the backup.
- **Restore before readable restores ship.** This task does not install keys, so no restore publishes the repository yet.
  - A replace restore, including one with the default "Everything" selection, keeps the live Personal Context repository out of publication. The live database is left byte-for-byte unchanged, and the archive keeps the captured copy.
  - The restore plan carries one line saying Personal Context will not be restored.
  - Today the row is `unsupported`, so restores never touch the live database. This keeps that property.
    - Stamping a marker instead would lock a readable profile with no way back: a later rollback is itself a replace restore (`later_rollback.py`), and a locked service refuses Remove and Start fresh (`Personal_Context/service.py:184-187`).
  - Isolated restores likewise leave the repository out.
  - This keeps the task safe on its own.
- **Link keys.** The device wrapping key, staged integrity keys and dataset staging keys are never captured (decision 5). A backup that includes credentials discloses that they are left out, as it already does for skill-trust keys (`credentials.py:1052-1065`).
- **Interim sync guard.** A restored link must not resume with stale cursors.
  - The repository's adapter is `activation_required`, and every restore binds all activation-required owners in `finalize_candidate` (`publication.py:592-604`).
  - So Personal Context sync (`app.py:2625-2662`) can stay off until the restored generation is reviewed, through `activation.py:224-227`.
  - This guard holds on its own.
  - Its review approves Personal Context sync only after a restore that published neither `personal_context.repository` nor `runtime.sync_state`. After one that published either, sync stays off, with a named reason shown to the user, until a review that clears the link state exists.
- **First link in flight.** Capture is refused while a `first_link_freeze` or `first_link_rebaseline_commit` row exists (`repository.py:214-226`, `:311-320`).
- **Device-only records** are included, as **Export recovery copy** already does (decision 8).
- **Detached imports.** A `_SQLiteDeclaration` owner is picked up by the detached layout automatically (`detached_layout.py:636-642`, `:1080-1085`). Without a guard, that would import the repository into an archive without its key. Decision 10 refuses it there by name.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A backup preview of a profile holding the repository database (with its `-wal`) lists it as captured in a selectable "Personal Context" group, and the preview is no longer incomplete because of it (red first)
- [ ] #2 The archive holds a consistent copy of the database, including device-only records, that passes the installed exact-schema validation for each v9 layout from TASK-40133
- [ ] #3 A backup that includes credentials carries the profile key as a `personal_context` credential record, and capture checks that the key opens the captured copy (its manifest authenticates) before writing the record
- [ ] #4 A backup without credentials carries no key material and keeps the database's key reference exactly as it was (it is not sanitized); its preview says the key is not in the backup
- [ ] #5 After a replace restore with the default "Everything" selection, the live Personal Context repository is byte-for-byte unchanged, the restore plan said beforehand that Personal Context would not be restored, and the archive still holds the captured copy; an isolated restore likewise leaves the repository out, and its plan says so (red first)
- [ ] #6 Inert extraction cannot write the key to disk from any archive; a test pins it
- [ ] #7 Capture is refused with a named reason, shown in the preview, while a first-link freeze or rebaseline marker exists
- [ ] #8 The review of a backup that includes credentials says the device wrapping key, staged integrity keys and dataset staging keys are left out, as it does for skill-trust keys
- [ ] #9 After any restore, Personal Context sync does not resume until the restored generation is reviewed; this guard holds on its own (red first)
- [ ] #10 That review approves Personal Context sync only after a restore that published neither `personal_context.repository` nor `runtime.sync_state`; after one that published either, sync stays off and the user sees a named reason saying so
- [ ] #11 A detached source holding the repository is refused by name and never imported without its key
- [ ] #12 The owner inventory doc's Personal Context rows and the Personal Context User Guide page's backup paragraph describe the new behaviour
<!-- AC:END -->
