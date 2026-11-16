---
id: TASK-41120
title: "Old Personal Context keyring accounts kept for recovery copies are never deleted"
status: Done
assignee:
- '@claude'
updated_date: '2026-11-16 21:23'
created_date: 2026-11-15 02:20
labels:
- follow-up
- personal-context
- backup
- security
priority: low
references:
- tldw_chatbook/Personal_Context/repository_keys.py
- tldw_chatbook/Personal_Context/recovery.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Filed in round 14 of the 2026-10-23 burn-down (closeout, 2026-11-15). Checked against the round-14 stack top c7d4a41717 (on dev beab5c1e4e); line numbers are that tree's.

**Why.** Each restore and re-encryption can leave an old encryption key in the keyring for good. It reads no row of the re-encrypted profile, but it does read the recovery copies, and nothing ever removes it once those copies are gone.

**Evidence.** `repository_keys.py:336-337`: re-encryption deletes the old account only `if not keep_old_account`, and `recovery.reencrypt_profile` (`recovery.py:667-685`) passes `keep_old_account=retained` while any recovery copy exists. TASK-40140's notes (review r14pccB, minor): "the old account is kept while any recovery copy exists, and is never deleted later. … Proposed follow-up: delete kept accounts when the last recovery copy is deleted."
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Deleting the last recovery copy that names a kept Personal Context account deletes that account, and only when no live database names it
- [x] #2 A test re-encrypts with a recovery copy kept, deletes the copy and checks the account is gone, red before the change
<!-- AC:END -->

## Implementation Plan

1. `recovery.reencrypt_profile`: count as retained only copies whose ciphertext still exists and that may hold a Personal Context repository (coverage names one, or the copy's evidence is unreadable); a deleted copy (status `missing`) no longer keeps the account.
2. When the old account is kept, record it in the recovery control root (one exclusive-create record per kept account: account, database path, the copies that may name it). `RepositoryKeyCustody.reencrypt_profile` reports the kept account through a callback.
3. `recovery.forget_kept_keys(control_root)`, called by `delete_recovery_copy` after the unlink: for each record whose copies are all gone, delete the account only when the recorded database does not name it, then drop the record. Never fails the deletion.
4. Tests: re-encrypt with a copy kept, delete the copy, the account is gone (red first); a live database naming the account keeps it.

## Implementation Notes

**2026-11-15 (round 15, stream pc).** A Personal Context account that re-encryption keeps for recovery copies is now recorded, and deleting the last copy that may name it deletes it, unless a live database names it.

- **Which copies may name it.** Copies are encrypted, so which account one names cannot be read. `recovery._copies_naming(control)` counts a listed copy only while its ciphertext exists (status not `missing`; deleting a copy keeps its journal, so before this a deleted copy kept the account for good) and its coverage includes a `personal_context.repository` logical id; a copy whose evidence cannot be read (`recovery_required`) counts. A failed listing keeps the account without a record, as before.
- **Record.** `RepositoryKeyCustody.reencrypt_profile(on_kept=...)` passes the kept account; `recovery.reencrypt_profile` writes `personal-context-kept-<random>.json` (account, absolute database path, the copy ids) into the recovery control root with `control_records._write` (exclusive, private, durable). A failed write only leaves the entry, as before.
- **Deletion.** `delete_recovery_copy` calls `recovery.forget_kept_keys(control_root)` after its unlink. For each record whose copies are all gone, it deletes the account (it must match `KEY_ACCOUNT`) only when the recorded database does not name it (`profile_meta.key_account`, read-only; a read error counts as naming it), then removes the record. It never raises: the copy is already deleted, and a leftover entry holds only an old key that reads no live row.
- **Tests.** `Tests/Personal_Context/test_repository.py::test_a_kept_account_goes_only_when_no_copy_or_live_database_names_it[deleted|no_personal_context|names_live_database]` (a deleted copy or one without Personal Context keeps nothing; a kept account outlives its last copy while the database names it, then goes with its record); `test_reencryption_keeps_the_old_account_while_a_recovery_copy_may_name_it` now also checks the record. AC#2: `Tests/Backup_Recovery/test_restore_data_groups.py::...[reviewed-include]`, a real replace restore, the review's re-encryption (kept: the restore's copy captured the Personal Context repository), then `RecoveryService.start_delete_copy`: the kept account is the only entry gone and the records read.
- **Red/green.** Red on the previous code: `reviewed` failed at `assert set(kept) - set(current['device'].values) == {(SERVICE, restored_account)}` (the kept account stayed), 162 s; the unit cases `deleted`/`no_personal_context` failed on `assert … not in backend.values`, `names_live_database` on `AttributeError: … forget_kept_keys`. Green: `reviewed` 1 passed (192 s, load 80), unit tests 16 passed.
- **Green at low load.** `reviewed` 103 s under the default budget (load 7-45).
- **Covering runs (2026-11-15/16, `-n 4` unless noted).** All 12 `test_replace_restore_makes_personal_context_readable_and_rolls_back` scenarios: 12 passed under the default 300 s child budget at load 7-45 (46-173 s each). `Tests/Personal_Context` + `Tests/UI/test_settings_personal_context.py` + `Tests/UI/test_personal_context_proposal_review.py`: 803 passed, 6 failed, 1 error at load 100-117; on rerun all pass except `test_production_settings_inspects_private_and_deleted_metadata[size0]`, a 5 s provenance wait that also fails on base with the same command at load 85. `test_recovery_copies.py`, `test_later_rollback.py`, `test_recovery_service.py`, `test_credentials.py`, `test_later_rollback_credential_ui.py`, `Tests/DB/test_private_sqlite_inventory.py`, the rest of `test_restore_data_groups.py`: 235 passed, 1 setup error (`test_f9_later_rollback_requires_explicit_credential_review`: a pip-install timeout under load; serially it fails "real exec expected", identically on base). `test_restore_plan.py`: 119 passed. `Tests/Architecture/test_module_size_ratchet.py` 207 passed; ruff gate, markup census and preflight pass.
- **Lane.** The new tests sit in the files their TASK-40140 neighbours use (`test_repository.py`, `test_restore_data_groups.py`, `test_settings_personal_context.py`), none of which a PR lane runs whole; no census entry added (round-15 rule: the shard count is the coordinator's decision).
- **Files.** `tldw_chatbook/Personal_Context/{recovery.py,repository_keys.py}`, `tldw_chatbook/Backup_Recovery/recovery_copies.py`, `Tests/Personal_Context/test_repository.py`, `Tests/Backup_Recovery/test_restore_data_groups.py`, `Docs/User_Guide/settings/personal-context-profile.md`.
