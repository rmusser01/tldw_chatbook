---
id: TASK-40135
title: "A replace restore makes restored Personal Context records readable"
status: To Do
created_date: 2026-11-10 17:13
labels:
- follow-up
- backup
- recovery
- personal-context
priority: medium
references:
- tldw_chatbook/Backup_Recovery/credentials.py
- tldw_chatbook/Backup_Recovery/staging.py
- tldw_chatbook/Backup_Recovery/replacement.py
- tldw_chatbook/runtime_policy/server_credentials.py
- Docs/superpowers/specs/2026-11-10-personal-context-backup-key-custody-design.md
dependencies:
- TASK-40133
- TASK-40134
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Filed in round 13 of the 2026-10-23 burn-down from TASK-39148's owner-approved design, `Docs/superpowers/specs/2026-11-10-personal-context-backup-key-custody-design.md`. Owner decision 2026-11-10 (AskUserQuestion): "Approve all 11". Checked on dev ae9f9e8d21, 2026-11-10; line numbers are dev's.

**Why.** Restore never writes a non-server credential back to the keyring today.
- Records are retained encrypted in `<control>/replacement-<op>/credentials.age` after an acknowledgement (`Backup_Recovery/staging.py:697-768`, `replacement.py:1604-1620`).
- So a restored Personal Context database can be read only where the original keyring entry still exists for its path.

**Approved design (decisions 2 and 4).** Follow the server-credential remap. Remapped server credentials go to a fresh `recovery_<hex>_<purpose>` scope with set-if-absent (`runtime_policy/server_credentials.py:688-695`), and the staged reference is rewritten (`credentials.py:1231-1302`).

For Personal Context, a replace restore **always** installs key material into a fresh account, `personal-context:recovery_<hex>`, with set-if-absent, and rewrites the staged `key_account`. That includes the key-less same-device case.
- **Where the material comes from.** From the archive's credential record. For a key-less archive, it is copied from the account the archive names on this device.
- **Authenticate first.** Either way, the material is used only after it authenticates the restored database's manifest head. An account named by an archive is never trusted unauthenticated.
- **Nothing authenticates.** The staged reference becomes `recovery:setup_required`, so the database names no account and stays locked.
- **Why always a fresh account.** A restore never leaves two databases, live or retained, naming one account. Linking or removing the restored profile (TASK-40133) therefore cannot break a rollback of the retained original.
- **No overwrites.** It never overwrites or deletes an entry it did not create. This is ADR-126 decision 8 (see its 2026-11-10 note).
- **Check after publishing.** It opens the published database and authenticates the manifest head again. It journals the key step, so `recover --finish` and `recover --rollback` are idempotent.
- **Rollback copy.** The replace-mode rollback copy already captures the live key, because it is a credential record in `rollback` mode (`replacement.py:445-469`).
- **No re-encryption.** Records are not re-encrypted (decision 4; re-encryption is a separate follow-up).

**Landing.** This lands in the same PR as TASK-40134. Until it does, a replace restore leaves the restored and the retained databases naming one account.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 After a replace restore from an encrypted backup that includes credentials, onto the same data path and onto a different one, Personal Context opens and lists the restored records (red first)
- [ ] #2 Every replace restore, including one from a backup without credentials on the same device, installs authenticated key material into a newly created account and stamps it. A keyring entry that existed before the restore is byte-for-byte unchanged, and afterwards no two databases (live or retained) name one account
- [ ] #3 Key material from an account the archive names is used only after it authenticates the restored database; when nothing authenticates, the database is published naming no account (`recovery:setup_required`) and Personal Context is locked
- [ ] #4 Restore checks that the restored database opens with the installed key before it reports success; on failure it removes only the account it created, leaves the database in place, and Personal Context is locked
- [ ] #5 A later rollback of that restore opens the original records with their original key
- [ ] #6 A rollback after the restored profile has linked, or after Remove local profile on it, still opens the original records
- [ ] #7 A restore interrupted after the key step and finished with `recover --finish` or `--rollback` leaves the database's key reference naming an account that exists and holds its key; a test covers both
- [ ] #8 The restore plan says, before the restore runs, that Personal Context will be restored with its key
<!-- AC:END -->
