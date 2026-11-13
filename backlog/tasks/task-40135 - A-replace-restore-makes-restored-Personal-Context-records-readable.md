---
id: TASK-40135
title: "A replace restore makes restored Personal Context records readable"
status: Done
created_date: 2026-11-10 17:13
updated_date: '2026-11-13 20:30'
assignee:
- '@claude'
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
- [x] #1 After a replace restore from an encrypted backup that includes credentials, onto the same data path and onto a different one, Personal Context opens and lists the restored records (red first)
- [x] #2 When a key authenticates the restored database, the restore installs it into a newly created account and stamps it: the restored database names an account that no other database, live or retained, names, and a keyring entry that existed before the restore is byte-for-byte unchanged
- [x] #3 Key material from an account the archive names (or, for an empty reference, the target path's legacy account) is used only after it authenticates the restored database. When nothing authenticates, or there is no secure keyring, the database is published with `profile_meta.key_account` set to `recovery:setup_required:<reason>`, every table except `profile_meta.key_account` is unchanged from the archived copy, and Personal Context is locked
- [x] #4 A same-device replace restore from a backup without credentials opens and lists the records
- [x] #5 The preview of a backup without credentials says it is readable only on this device, and only until Personal Context is linked or removed
- [x] #6 Restore checks that the restored database opens with the installed key before it reports success; on failure it removes only the account it created, leaves the database in place, and Personal Context is locked
- [x] #7 A later rollback of that restore opens the original records with their original key
- [x] #8 A rollback after the restored profile has linked, or after Remove local profile on it, still opens the original records
- [x] #9 A restore interrupted after the key step and finished with `recover --finish` or `--rollback` leaves the database's key reference naming an account that exists and holds its key; a test covers both
- [x] #10 The restore plan says, before the restore runs, whether Personal Context will be restored readable with its key or will end locked, and why
- [x] #11 The portable relocation advice — make an encrypted backup with credentials on the original device, then restore it with Replace — works end to end for a profile holding Personal Context; a test pins it (coordinator ruling, round 14, carried over from the TASK-40138 review)
<!-- AC:END -->

## Implementation Plan

1. `KeyringProfileKeyProtector.install(account, material)`: set-if-absent with given material (`create` delegates to it).
2. `Personal_Context/recovery.py`: one outcome function used at plan time and at staging. Candidates are the archive's `personal_context` record, then the account the archived database names on this device (or, for an empty reference, the target path's legacy account); a value is used only after it authenticates the archived copy's manifest head. Outcome: readable (secure keyring present), or locked with `personal_context_key_install_unavailable` (no secure keyring) / `personal_context_key_not_in_backup` (nothing authenticates).
3. `plan_restore`: replace restores publish the repository and carry `personal_context_restore_readable` or `personal_context_restore_locked:<reason>`; isolated restores still hold it back (TASK-40136). The group becomes restorable.
4. `stage_restore` (replace): recompute the outcome; refuse if it differs from the plan. Readable: install into a fresh `personal-context:recovery_<hex>` account (set-if-absent, read back) and stamp the staged `key_account`; locked: stamp `recovery:setup_required:<reason>`. Every other table is untouched. A failed staging deletes only the account it created. The key step precedes the journal, so `recover --finish` publishes a candidate naming an account that exists and `--rollback` puts back an original naming its untouched account.
5. After publication (restore and `recover --finish`): reopen the published database and authenticate with the stamped account; on failure delete only that `recovery_` account, keep the database, report `personal_context_key_unverified`.
6. Copy for the plan lines and the result; tests: unit tests for the outcome/stamp, end-to-end child runs through `RecoveryService` with an in-process keyring (same path, other path + other keyring, no credentials on the same device, rollback after link and after Remove, interrupted finish/rollback, verify failure).

Deviation (see notes): staging only names the account; the keyring entry is written just before publication, not during staging.

## Implementation Notes

**2026-11-12 (round-13 closeout): ADR-203.** ADR-203's model-disclosure grants have not shipped: the ADR is "Accepted (design only)", and `tldw_chatbook/` has no grant code. Grants are "encrypted, peer-local and never synchronize/export" (ADR-203 :30), and "Recovery never imports local grant authority" (:85). When they ship, their storage must stay out of the captured repository database, or a restore must drop them, with a test that pins it.

**2026-11-13 (round 14, stream pcc, implementation).** Replace restores publish the repository readable, or locked with the reason named before they run.

- **Which key.** `Personal_Context/recovery.restore_key`: the archive's `personal_context` record, else the account the archived copy names on this device, else (empty reference) the target path's legacy account; a value counts only after `profile_key_opens` authenticates the copy's manifest head. Result: readable (secure keyring present), `personal_context_key_install_unavailable` (no secure keyring), or `personal_context_key_not_in_backup` (nothing authenticates). A removed profile holds no key.
- **Plan (AC#10).** `plan_restore` (replace only; isolated still holds back until TASK-40136) adds `personal_context_restore_readable:<logical id>` or `personal_context_restore_locked:<reason>:<logical id>`, computed from the sealed archive's own copy (an in-memory image through the `recovery.personal_context.repository` memory target, now `_PRIVATE_MEMORY_AND_READ_ONLY`) and record. An unreadable payload refuses the plan (`personal_context_validation_unavailable`). The group is offered for restore again. The review shows the copy.
- **Staging.** `staging._stamp_personal_context` -> `stamp_restored_key` recomputes the outcome on the staged copy and refuses anything but the plan's line (`personal_context_restore_changed`), then rewrites only `profile_meta.key_account`: a fresh `personal-context:recovery_<32 hex>` account (derived from the candidate path and logical id, so unique per staging and recomputable; refused if occupied), the marker, or `''` for a removed row so a restored secure-removal record can never delete an entry on this device.
- **Install.** Deviation from the first plan: no keyring write while staging. A real include-mode restore goes through two abort-and-retry reviews (`credential_isolated_retention_required`, then the safety copy's own credential review), and installing at staging left one orphan entry holding the key per attempt. `replacement._install_personal_context_keys` re-derives the key from the archive, writes it set-if-absent (`KeyringProfileKeyProtector.install`, new; `create` delegates) and reads it back, after credential application and just before `publish_candidate`; `recover --finish` repeats it before publishing (an existing entry with the same value is accepted). The journal's recorded candidate names the account, which is what makes finish and rollback idempotent; there is no separate journal event.
- **Check (AC#6).** After `replace` and after `recover --finish`, `RecoveryService` runs `verify_restored_keys`: the published copy must open with its `recovery_` account; otherwise only that account is deleted, the data stays and the result carries `personal_context_key_unverified`.
- **Partial archives.** TASK-40134's `credential_personal_context_link_keys_excluded:<id>` disclosure made every include-mode archive with Personal Context `partial`, so `plan_restore` refused it (`partial_replacement`) and AC#1 could not hold. `credential_policies.MANUAL_CREDENTIAL_ISSUES` now lists it beside `credential_manual_recovery_required:` for `capture` and `archive_writer` (still disclosed and still acknowledged).
- **Not changed.** A Personal Context record still triggers `credential_isolated_retention_required` (publication recomputes that review from the retained scopes, and the record is retained in `credentials.age`); its copy ("need manual recovery") is broader than this case. An interrupted restore rolled back after the key step leaves its unpublished `recovery_` entry in the keyring (never named by any database).

**Evidence.** Red on the restack base 944380a54d (source files restored, tests kept): the six end-to-end scenarios fail on the child's assertions (no readable plan line; the repository was not restored), the three UI copy cases on the missing copy, the narrowed isolated test on `'personal_context' in ('settings',)`, and the nine unit nodes on the missing API. Negative controls on the new code, each red and reverted: accepting a key without authentication (`other_key` marker case raised `personal_context_restore_changed`), dropping the occupied-account check (`DID NOT RAISE`), not deleting on failed verification (keyring dict mismatch).

**Tests.** `Tests/Backup_Recovery/test_restore_data_groups.py::test_replace_restore_makes_personal_context_readable_and_rolls_back[link-include|remove-include|other_device-include|link-exclude|finish-include|rollback-include]`: a scratch child drives `RecoveryService` (backup, inspection, plan, restore, rollback) with two in-process `SecureKeyring` "devices" and the null global keyring; `other_device` is AC#11 (encrypted backup with credentials, another keyring and data path, then Replace). Each runs 60-90 s serially (load 7-30). `Tests/Backup_Recovery/test_credentials.py` (unit: names a new account for each key source, refuses an occupied account, marker cases, plan mismatch, failed verification); `Tests/UI/test_backup_data_groups.py::test_restore_review_says_how_personal_context_will_end` and an AC#5 assertion added to TASK-40134's preview test. TASK-40134's replace-mode plan test was narrowed to isolated mode. None of these files is in a PR lane (follow-up). Covering run after the rebase onto ab4bdf751d: `test_credentials.py`, `test_restore_data_groups.py`, `Tests/UI/test_backup_data_groups.py`, `Tests/Personal_Context/test_repository.py`, `test_key_protector.py` 194 passed (`-n 4`); `Tests/DB/test_private_sqlite_inventory.py` 60 passed plus the base-red `test_backup_inventory_matches_current_sqlite_and_settings_operations`; the persistence census row test passes with the four new rows.

Files: `tldw_chatbook/Personal_Context/{recovery.py,key_protector.py}`, `tldw_chatbook/Backup_Recovery/{restore_plan.py,restore_groups.py,staging.py,replacement.py,recovery_service.py,capture.py,archive_writer.py,credential_policies.py}`, `tldw_chatbook/DB/private_sqlite.py`, `tldw_chatbook/UI/Screens/backup_restore_screen.py`, the tests above, `backlog/docs/{backup-recovery-owner-inventory.md,sqlite-private-owner-inventory.md}`, `Docs/User_Guide/settings/personal-context-profile.md`.

**2026-11-13 (after rebasing onto pca 76af2a47a6).** A keyring that does not answer is never "no key": `recovery._stored_value` now refuses (`personal_context_keyring_read_failed`) instead of returning nothing, so the plan, staging and the pre-publication install refuse rather than plan a lock or treat an account as free; after publication a failed read reports `personal_context_key_check_unavailable` and keeps the new entry instead of deleting it. Pinned by `test_credentials.py::test_a_keyring_that_does_not_answer_is_never_no_key` (red under a temporary revert to the old `return None`: the stamp raised `personal_context_restore_changed`). The rebase conflict in `key_protector.py` kept pca's `exists()` and made `install` (set-if-absent with given material) use it; `create` delegates to `install`.

**Census follow-up (same day).** The DB inventory pins C-row ids and admits each `open_recovery_validation` caller by shape: C101 (`_archived_key`, memory) and C102 (`verify_restored_key`) are the documented `connect_private_sqlite` sites, the id range in `test_inventory_has_stable_unique_connection_and_backup_ids` extends to C102, and `_qualified_fixed_validation_dispatch` admits exactly `open_recovery_validation(OWNER, candidate, writable=True, with_restrictions=True)` in `stamp_restored_key`. `Tests/DB/test_private_sqlite_inventory.py`: 60 passed plus the base-red `test_backup_inventory_matches_current_sqlite_and_settings_operations`.

**2026-11-13, review r14pccA fixes.**
- **Rollback deletes the restore's own account (spec Q4).** `recovery.forget_restored_keys(plan, candidate)` deletes `restore_account(candidate, key)` for each `personal_context_restore_readable:<key>` plan line, and nothing else. It runs after `_rollback_replacement` at all three sites: a finalize failure inside `replace`, `recover --rollback`, and a finalize failure inside `recover --finish`. A keyring failure there leaves an unnamed entry; it never fails the rollback. The `[rollback-include]` child now asserts the entry is gone.
- **"Could not check" is not "wrong key".** The new `repository_keys.profile_key_authenticates` returns True, False or None. False is definite: not key material, or AEAD/HMAC failure on the manifest head. `profile_key_opens` is `authenticates(...) is True`. `verify_restored_key` deletes only on a definite failure or a missing entry. It returns `personal_context_key_check_unavailable` and keeps the entry when there is no secure keyring, the keyring does not answer, a read or SQLite error occurs, or the result is undecided. Test: `test_credentials.py::test_a_key_check_that_cannot_tell_keeps_the_entry[no_secure_keyring|read_error]`.
- **Named refusals.** `personal_context_restore_changed`, `personal_context_key_account_in_use`, `personal_context_key_install_failed` and `personal_context_validation_unavailable` have messages, and are also listed in `known`. `restore_changed` tells the user to review the restore again. Test: `test_personal_context_restore_refusals_are_named`.
- **Red on the previous head:** the four codes read `backup_operation_failed`, and both cannot-tell cases deleted or misreported.

