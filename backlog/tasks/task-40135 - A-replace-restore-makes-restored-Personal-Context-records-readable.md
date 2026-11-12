---
id: TASK-40135
title: "A replace restore makes restored Personal Context records readable"
status: To Do
created_date: 2026-11-10 17:13
updated_date: '2026-11-12 19:38'
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

**What changes.** TASK-40134 keeps the repository out of publication. From this task, a replace restore publishes it.
- Isolated restores still leave it out.
- A replace restore **always** installs key material into a fresh account, `personal-context:recovery_<hex>`, with set-if-absent, and rewrites the staged `key_account`. That includes the key-less same-device case.
- **Where the material comes from.** From the archive's credential record. For a key-less archive, it is copied from the account the archive names on this device. If the archived reference is empty, it is copied from the target path's legacy account.
- **Authenticate first.** Either way, the material is used only after it authenticates the restored database's manifest head. An account named by an archive is never trusted unauthenticated.
- **Nothing authenticates, or there is no secure keyring.** The staged reference becomes the marker `recovery:setup_required:<reason>` (spec Q2).
  - The reason is `key_not_in_backup` or `key_install_unavailable`.
  - Every table except `profile_meta.key_account` is unchanged from the archived copy. The database names no account and stays locked.
  - Such a restore is the first one that ends locked, so its plan must say so before it runs (constraint 4).
- **Why always a fresh account.** A restore never leaves two databases, live or retained, naming one account. Linking or removing the restored profile (TASK-40133) therefore cannot break a rollback of the retained original.
- **No overwrites.** It never overwrites or deletes an entry it did not create. This is ADR-126 decision 8 (see its 2026-11-10 note).
- **Check after publishing.** It opens the published database and authenticates the manifest head again. It journals the key step, so `recover --finish` and `recover --rollback` are idempotent.
- **Rollback copy.** The replace-mode rollback copy already captures the live key, because it is a credential record in `rollback` mode (`replacement.py:445-469`).
- **No re-encryption.** Records are not re-encrypted (decision 4; re-encryption is a separate follow-up).
- **Preview copy for a backup without credentials.** It now says "Database only: readable only on this device, and only until Personal Context is linked or removed."
  - The first link overwrites the named account in place (`repository.py:2051-2055`, `:2613`), and Remove deletes it. After either, every earlier key-less backup fails to authenticate, even on this device.
  - `exclude` is the default capture mode (`capture.py:296`).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 After a replace restore from an encrypted backup that includes credentials, onto the same data path and onto a different one, Personal Context opens and lists the restored records (red first)
- [ ] #2 When a key authenticates the restored database, the restore installs it into a newly created account and stamps it: the restored database names an account that no other database, live or retained, names, and a keyring entry that existed before the restore is byte-for-byte unchanged
- [ ] #3 Key material from an account the archive names (or, for an empty reference, the target path's legacy account) is used only after it authenticates the restored database. When nothing authenticates, or there is no secure keyring, the database is published with `profile_meta.key_account` set to `recovery:setup_required:<reason>`, every table except `profile_meta.key_account` is unchanged from the archived copy, and Personal Context is locked
- [ ] #4 A same-device replace restore from a backup without credentials opens and lists the records
- [ ] #5 The preview of a backup without credentials says it is readable only on this device, and only until Personal Context is linked or removed
- [ ] #6 Restore checks that the restored database opens with the installed key before it reports success; on failure it removes only the account it created, leaves the database in place, and Personal Context is locked
- [ ] #7 A later rollback of that restore opens the original records with their original key
- [ ] #8 A rollback after the restored profile has linked, or after Remove local profile on it, still opens the original records
- [ ] #9 A restore interrupted after the key step and finished with `recover --finish` or `--rollback` leaves the database's key reference naming an account that exists and holds its key; a test covers both
- [ ] #10 The restore plan says, before the restore runs, whether Personal Context will be restored readable with its key or will end locked, and why
<!-- AC:END -->

## Implementation Notes

**2026-11-12 (round-13 closeout): ADR-203.** ADR-203's model-disclosure grants have not shipped: the ADR is "Accepted (design only)", and `tldw_chatbook/` has no grant code. Grants are "encrypted, peer-local and never synchronize/export" (ADR-203 :30), and "Recovery never imports local grant authority" (:85). When they ship, their storage must stay out of the captured repository database, or a restore must drop them, with a test that pins it.
