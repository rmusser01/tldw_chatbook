---
id: TASK-41124
title: "Restore review says credentials 'need manual recovery' for a backup whose Personal Context key a replace restore installs itself"
status: Done
assignee:
- '@claude'
created_date: 2026-11-15 02:20
updated_date: 2026-11-15 19:12
labels:
- follow-up
- backup
- personal-context
- ux
priority: low
references:
- tldw_chatbook/UI/Screens/backup_restore_screen.py
- tldw_chatbook/Backup_Recovery/replacement.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Filed in round 14 of the 2026-10-23 burn-down (closeout, 2026-11-15). Checked against the round-14 stack top c7d4a41717 (on dev beab5c1e4e); line numbers are that tree's.

**Why.** Since TASK-40135 a replace restore installs a readable Personal Context key into a fresh account. The review still warns that credentials need manual recovery, which tells the user to do work the restore does for them.

**Evidence.** A Personal Context credential record still triggers `credential_isolated_retention_required` (`replacement.py:1497-1499`; TASK-40135's notes, "Not changed"). Its copy, `backup_restore_screen.py:2039` and `:2095`, reads "Some (incoming) credentials need manual recovery and will remain in the encrypted archive." (The other half of this follow-up, a rolled-back restore leaving a `recovery_` keyring entry, was fixed in TASK-40135 review r14pccA: rollback deletes the restore's own account at all three sites.)
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 When the only retained credential is a Personal Context key the restore installs, the review does not say it needs manual recovery
- [x] #2 Other retained credentials keep the current copy
- [x] #3 Tests cover both, red before the change
<!-- AC:END -->

## Implementation Plan

1. A pure classifier: the retained credential scopes of a staged replace candidate are all Personal Context keys, and the plan has no locked Personal Context line (so this restore installs each one).
2. `RecoveryService.start_restore`: when `replace()` asks for the retention review, record that classification in the operation result.
3. The restore credential review uses it: the installed-key case says the restore installs the key and a copy stays in the encrypted archive, without "manual recovery"; every other retention keeps the current copy.
4. Tests: classifier unit cases, a mounted screen test for both copies (red before), and the Personal Context end-to-end restore asserting the flag on its real retention review.

## Implementation Notes

**Summary.** When a replace restore stops for the retention review (`credential_isolated_retention_required`) and every credential it keeps in the encrypted archive is a Personal Context key that the same restore installs, the review now reads "This restore installs Personal Context's key itself; a copy also stays in the encrypted archive. No replacement was published. Accept encrypted retention, enter a new rollback password, and review restore again.", and its checkbox reads **Accept encrypted retention**. Any other retention keeps "Some incoming credentials need manual recovery..." and **Accept encrypted manual retention**. The review step itself, its code and its acknowledgement are unchanged.

**Approach.**
- `recovery_service.retains_only_installed_keys(scopes, records, issues)` (pure): the staged descriptor's `credential_scopes` with action `retain` all belong to `personal_context` records, and the plan has no `personal_context_restore_locked:` line (a locked key is not installed; one plan line exists for every restored repository, and a captured key record exists only for a live profile).
- `RecoveryService.start_restore`: when `replace()` raises exactly that retention review, `_retention_installs_keys(candidate, plan)` reads the candidate's descriptor and staged `credential-recovery.json` and records `personal_context_keys_installed: True` in the operation result before re-raising. Any failure to classify reads as manual, and never replaces the review.
- The screen passes that flag to `_show_restore_credential_review`.
- The staged material is read the way `publication._descriptor` reads `candidate.json` (pinned directory, private-file JSON read). The first version used `credentials._material`, which goes through storage admission: the end-to-end run showed `RecoveryRequired: storage_scope_not_enrolled`, which replaced the review with `backup_operation_failed`. That is why the classifier now catches every exception.
- Scope: the restore review only. A later rollback restores from a local snapshot, adds no Personal Context plan lines and installs no key, so its retention copy (the evidence's second site) keeps the manual wording; that is AC#2's "other retained credentials keep the current copy".

**Tests and red/green evidence** (load 53-120 on 18 CPUs):
- `Tests/UI/test_backup_restore_setup_entry.py::test_restore_retention_review_says_when_the_restore_installs_the_key[True|False]` (whole-file UI census entry): a service operation records the flag (or not) and raises the retention review; the test reads the copy and the checkbox. `[True]` red before the change ("'manual' is contained here: Some incoming credentials need manual recovery..."). `[False]` pins the unchanged copy; it was red with the screen temporarily ignoring the flag (`installs_keys = retention`). Both green after.
- `Tests/Architecture/test_backup_owner_inventory.py::test_retention_review_names_only_keys_the_restore_installs` (6 cases: key only, key plus a remapped server credential, key plus a generation key, a non-remappable server credential, a locked key, nothing retained). Red with the classifier temporarily reduced to "a key is retained": 3 failed.
- The real path: `Tests/Backup_Recovery/test_restore_data_groups.py::test_replace_restore_makes_personal_context_readable_and_rolls_back` records the flag of every restore retention review. `[no_keyring-include]` passed with `RETENTION == [False, True]` (the locked restore keeps the manual copy, the readable one gets the new copy). Red at the base service code (the glue temporarily reverted): `AssertionError: [False, False]`. The main path asserts `[True]` for include and `[]` for exclude.

**Docs.** `Docs/Backup-and-Recovery.md`, beside the credential-omission paragraph.

**Files.** `tldw_chatbook/Backup_Recovery/recovery_service.py`, `tldw_chatbook/UI/Screens/backup_restore_screen.py`, `Tests/UI/test_backup_restore_setup_entry.py`, `Tests/Architecture/test_backup_owner_inventory.py`, `Tests/Backup_Recovery/test_restore_data_groups.py`, `Docs/Backup-and-Recovery.md`.
