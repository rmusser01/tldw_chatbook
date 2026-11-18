---
id: TASK-41122
title: "A later rollback does not reinstall the Personal Context key the rollback copy escrowed"
status: Done
assignee:
- '@claude'
updated_date: '2026-11-16 21:23'
created_date: 2026-11-15 02:20
labels:
- follow-up
- personal-context
- backup
priority: medium
references:
- tldw_chatbook/Backup_Recovery/later_rollback.py
- Docs/superpowers/specs/2026-11-10-personal-context-backup-key-custody-design.md
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Filed in round 14 of the 2026-10-23 burn-down (closeout, 2026-11-15). Checked against the round-14 stack top c7d4a41717 (on dev beab5c1e4e); line numbers are that tree's.

**Why.** The approved design (spec Q4, option A) says the rollback copy escrows the live key "That covers a keyring that is lost between restore and rollback." If the original account is gone when the user rolls back later, the rolled-back database cannot be opened.

**Evidence.** Spec `:268-282`. `later_rollback.py` has no Personal Context handling at all (no reference to the owner, its key or `credentials.age`). The restore-time rollback paths delete only the restore's own account (TASK-40135, `recovery.forget_restored_keys`); nothing reinstalls the original key from the copy.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A later rollback whose original Personal Context account is missing reinstalls the escrowed key into a fresh account (set-if-absent) and points the rolled-back database at it
- [x] #2 A rollback whose original account is still present leaves the keyring unchanged
- [x] #3 Tests cover both, the first red before the change
<!-- AC:END -->

## Implementation Plan

1. `Personal_Context.recovery.restore_key(..., rollback=True)`: a later rollback keeps the account its copy names; only when that account holds nothing on this device and the escrowed record authenticates the copy does it pick the record.
2. Plan: `plan_restore` adds the Personal Context issue for local snapshots too (rollback semantics), so a missing account plans `personal_context_restore_readable`.
3. Staging stamps the fresh account for that case only (an unchanged verdict writes nothing); replacement installs it set-if-absent before publication (existing path); the later rollback verifies the published key like a restore.
4. Tests: the end-to-end Personal Context restore harness gets a lost-keyring later rollback (red first) and asserts the keyring is unchanged by a later rollback whose original account is present.

## Implementation Notes

**2026-11-15 (round 15, stream pc).** A later rollback now reinstalls the escrowed Personal Context key when the account its copy names is gone, and changes nothing otherwise.

- **Why it never happened.** A later rollback is a replace restore from the rollback copy (`later_rollback.execute_rollback` -> `stage_restore` -> `replacement.replace`), but `plan_restore` skipped the Personal Context plan issue and `stage_restore` skipped the key stamping whenever `plan.local_snapshot` was set, so the rolled-back database always named its original account.
- **Rollback semantics (`Personal_Context.recovery.restore_key(..., rollback=True)`).** The copy keeps the account it names, `(None, None)`, unless the protector exists, that account (the named one, or the legacy path account for an empty reference) holds nothing on this device, and the rollback copy's escrowed record authenticates the copy; then it returns the record. A present account (any value), a marker or a removed profile, or no secure keyring, leaves everything as before.
- **Wiring.** `plan_restore` now computes the issue for local snapshots too (`rollback=local_snapshot is not None`), so a lost account plans `personal_context_restore_readable:<id>`. `stage_restore` always calls `_stamp_personal_context`; `stamp_restored_key(rollback=True)` returns without writing when the verdict is "keep", and otherwise stamps the fresh `recovery_` account and refuses on a changed verdict, as a restore does. `install_restored_keys` re-derives with the same semantics, so the existing set-if-absent install before publication, `forget_restored_keys` on a failed replacement, and `recover --finish` all apply unchanged. `RecoveryService.start_rollback` now runs `_verify_personal_context` after the rollback, as `start_restore` does.
- **Tests.** `Tests/Backup_Recovery/test_restore_data_groups.py::test_replace_restore_makes_personal_context_readable_and_rolls_back`: new `lost_keyring-include` scenario deletes the original account after the restore, then rolls back: the rolled-back database names a new `recovery_` account, that is the only new keyring entry, every other entry is unchanged and both records read (AC#1). The `link`, `remove` (and `lost_keyring` before its deletion) rollbacks now also assert the whole keyring is unchanged across the rollback and the original account is named (AC#2).
- **Red/green.** Red with the source reverted to base (patch saved, files checked out, re-applied): `lost_keyring` failed `AssertionError: personal-context:profile_aaa163…` (the rolled-back database still named the missing original account), 221 s at load 55. Green: `lost_keyring`, `link`, `remove` 3 passed (about 420 s each, `-n 3`, load 80-120; the child's 300 s budget was raised to 900 for that run only and restored, since at that load the unchanged `link`/`remove` scenarios also passed 300 s; at load 54 `link` took 165 s serially on base).
- **Green at low load.** `lost_keyring` 168 s, `link` 127 s, `remove` 173 s, under the default budget (load 7-45).
- **Covering runs (2026-11-15/16, `-n 4` unless noted).** All 12 `test_replace_restore_makes_personal_context_readable_and_rolls_back` scenarios: 12 passed under the default 300 s child budget at load 7-45 (46-173 s each). `Tests/Personal_Context` + `Tests/UI/test_settings_personal_context.py` + `Tests/UI/test_personal_context_proposal_review.py`: 803 passed, 6 failed, 1 error at load 100-117; on rerun all pass except `test_production_settings_inspects_private_and_deleted_metadata[size0]`, a 5 s provenance wait that also fails on base with the same command at load 85. `test_recovery_copies.py`, `test_later_rollback.py`, `test_recovery_service.py`, `test_credentials.py`, `test_later_rollback_credential_ui.py`, `Tests/DB/test_private_sqlite_inventory.py`, the rest of `test_restore_data_groups.py`: 235 passed, 1 setup error (`test_f9_later_rollback_requires_explicit_credential_review`: a pip-install timeout under load; serially it fails "real exec expected", identically on base). `test_restore_plan.py`: 119 passed. `Tests/Architecture/test_module_size_ratchet.py` 207 passed; ruff gate, markup census and preflight pass.
- **Lane.** The new tests sit in the files their TASK-40140 neighbours use (`test_repository.py`, `test_restore_data_groups.py`, `test_settings_personal_context.py`), none of which a PR lane runs whole; no census entry added (round-15 rule: the shard count is the coordinator's decision).
- **Files.** `tldw_chatbook/Personal_Context/recovery.py`, `tldw_chatbook/Backup_Recovery/{restore_plan.py,staging.py,recovery_service.py}`, `Tests/Backup_Recovery/test_restore_data_groups.py`, `Docs/User_Guide/settings/personal-context-profile.md`.
