---
id: TASK-40136
title: "A restore without a usable Personal Context key keeps the data and says what to do"
status: Done
assignee:
- '@claude'
updated_date: '2026-11-13 17:45'
created_date: 2026-11-10 17:13
labels:
- follow-up
- backup
- recovery
- personal-context
- ux
priority: medium
references:
- tldw_chatbook/Personal_Context/bootstrap.py
- tldw_chatbook/Personal_Context/repository.py
- tldw_chatbook/Backup_Recovery/isolated_restore.py
- tldw_chatbook/Widgets/Settings_Widgets/personal_context_panel.py
- Docs/User_Guide/settings/personal-context-profile.md
- Docs/superpowers/specs/2026-11-10-personal-context-backup-key-custody-design.md
dependencies:
- TASK-40134
- TASK-40135
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Filed in round 13 of the 2026-10-23 burn-down from TASK-39148's owner-approved design, `Docs/superpowers/specs/2026-11-10-personal-context-backup-key-custody-design.md`. Owner decision 2026-11-10 (AskUserQuestion): "Approve all 11". Checked on dev ae9f9e8d21, 2026-11-10; line numbers are dev's.

**Why.** A database whose key is missing already opens locked and is never given a new key (`Personal_Context/repository.py:380-387`, `bootstrap.py:59-73`). But the user is not told why, or what to do.
- The User Guide's **Profile locked** row says only "Unlock the key protector" (`Docs/User_Guide/settings/personal-context-profile.md:326`).
- A locked service refuses Remove local profile and Start fresh (`Personal_Context/service.py:184-186`). So the copy names only what the user can do while locked: restore or roll back through Backup & Restore, then choose **Try again**.

The restore cases in the approved design (spec Q2 table; decisions 3 and 6):
- **No key in the backup.** A backup without credentials, restored onto a device that lacks the key.
- **Isolated restored profile.** It runs with `keyring.backends.null.Keyring` (`Backup_Recovery/isolated_restore.py:393`, `:423-426`), which the key protector refuses (`key_protector.py:138-141`).
  - Isolated restores have so far left the repository out (TASK-40134, TASK-40135).
  - From this task they publish it with the marker reason `keyring_unavailable`, the key retained in the encrypted restore copy (decision 6), and a plan line saying it will be locked.
  - Precedence: an isolated restore uses `key_not_in_backup` and its copy only when the spec's row-1 condition holds: the archive had no key, and no account on this device authenticates the restored database. Otherwise it uses `keyring_unavailable`, whose "Restore with Replace" is then true.
  - `restore_isolated` (`isolated_restore.py:49`) runs in the live process with the device keyring, so this can be checked there. Only the launched copy uses the null backend (`:393`, `:423-426`).
- **No secure keyring on a replace restore.** The key stays in the retained encrypted restore copy (`replacement.py:1604-1620`).
- **Key mismatch.** The stored account's key does not authenticate the database's manifest.
  - This is detected by authentication (TASK-40133), never by version. `key_version` starts at 1 (`key_protector.py:48`) and only a link bumps it (`repository.py:2051-2055`), so two unrelated keys usually share a version.
  - Today a wrong key surfaces as `ProfileIntegrityError`, which `bootstrap.py:69-73` reports as the tamper-like `profile_integrity_invalid`. A tampered manifest raises the same error (`repository.py:1341-1352`).
  - So a manifest-authentication failure at open is reported as a key mismatch, and integrity failures beyond the manifest stay `profile_integrity_invalid`.
  - A restore never stamps a non-authenticating account (TASK-40135), so this case comes from outside restores.

**Where the reason is kept.** The three restore cases all leave the marker `recovery:setup_required:<reason>` in `profile_meta.key_account`.
- TASK-40135 writes the replace cases, and this task writes the isolated case.
- So a later start still names the right reason. For example, a no-keyring restore of a backup that did include credentials must not later read as "key not in backup".

The spec's Q2 table gives each case a named reason and the copy.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 In each restore case (key not in the backup, isolated restored profile, no secure keyring on a replace restore), no keyring entry is created, changed or deleted. Every table except `profile_meta.key_account` is unchanged from the archived copy, `key_account` carries the marker `recovery:setup_required:<reason>`, and Personal Context is locked with that case's reason: `personal_context_key_not_in_backup`, `personal_context_keyring_unavailable` or `personal_context_key_install_unavailable` (red first)
- [x] #2 The case's reason is recorded with the marker, so after a restart Personal Context still shows that case's reason and copy; a no-keyring restore of a backup that did include credentials never reads as `personal_context_key_not_in_backup`
- [x] #3 A database whose stored account holds a different key, including a different key with the same `key_version`, fails manifest authentication at open and is locked as `personal_context_key_mismatch`, not `profile_integrity_invalid`; integrity failures beyond the manifest stay `profile_integrity_invalid`; the data is byte-identical and no keyring entry changes (red first)
- [x] #4 The restore plan lists the first three cases as issues the user must acknowledge before the restore runs
- [x] #5 The Settings panel's locked state shows the spec's copy for each reason, naming only actions available while locked (restore or roll back through Backup & Restore, then Try again); an isolated restore shows the key-not-in-backup reason and copy only when the archive had no key and no account on this device authenticates the restored database, and the keyring-unavailable reason and copy otherwise; `scripts/check_markup_interpolation.py` passes
- [x] #6 After a replace restore with no secure keyring, a second replace restore from the same archive once a keyring is available makes the records readable
- [x] #7 The Personal Context User Guide troubleshooting table describes these states
<!-- AC:END -->

## Implementation Plan

1. Repository: a `recovery:setup_required:<reason>` marker locks with that reason (`ProfileLockedError.reason_code`), so it survives a restart; a stored key that does not authenticate the manifest head at open locks as `personal_context_key_mismatch`. Integrity failures past the manifest are unchanged.
2. Isolated restores publish the repository with the marker: `personal_context_key_not_in_backup` only when the archive had no key and no account on this device authenticates the copy (checked in the live process), else `personal_context_keyring_unavailable`. Nothing is written to the keyring.
3. Plan: each locked case (`personal_context_restore_locked:<reason>:<id>`, both modes) needs the acknowledgement `credential_personal_context_restore_locked:<reason>:<id>` before staging writes anything; the restore review shows the cases with their copy (isolated restores included), and the safety copy's own credential review ignores these codes.
4. Settings panel: per-reason locked copy from the spec, naming only restore/roll back through Backup & Restore, then Try again.
5. Tests: repository reasons and restart, key mismatch, isolated plan/stage, acknowledgement refusal, end-to-end no-keyring replace then a second replace once a keyring exists, isolated restore end to end, panel copy; User Guide troubleshooting rows.

## Implementation Notes

**2026-11-13 (round 14, stream pcc).** A locked restore names its case before it runs, keeps the data and keyring as they were, and Settings says why after every start.

- **Reasons at open (AC#1-#3).** `repository_keys._named_key_account` turns a `recovery:setup_required:<reason>` marker with one of the three restore reasons into a `ProfileLockedError` whose `reason_code` is that reason; bootstrap already reports `reason_code`, so the reason comes back on every start. `_open_keys` reports a stored key that fails AEAD/HMAC on the manifest head (the `ProfileIntegrityError` behind `_manifest_on_connection`'s lock) as `personal_context_key_mismatch`; any other lock stays as it was, and record-level integrity failures stay quarantined (`integrity_failure`). Since TASK-40133 a wrong key surfaced as `profile_locked` at open, not `profile_integrity_invalid` as the filing said; the AC's outcome (mismatch, not integrity) holds either way.
- **Isolated restores (AC#1, #5).** They now publish the repository. `recovery._isolated` maps the live-process check to `personal_context_key_not_in_backup` only when the archive had no key record and no account on this device opens the copy, else `personal_context_keyring_unavailable`; nothing is written to the keyring. Staging reads the record presence from the archived credential payload, because isolated staging sets the stage copy aside. Stamping now goes through `open_recovery_validation` (the restricted candidate connection) with a new `stamping_key_account` authorizer flag admitting only `UPDATE profile_meta.key_account`; isolated staging has no maintenance session, so the private-SQLite seam refused a writable open there (`discovery_read_only_required`).
- **Acknowledgement (AC#4).** Each `personal_context_restore_locked:<reason>:<id>` plan line needs `credential_personal_context_restore_locked:<reason>:<id>` in `acknowledged_credential_issues`; `stage_restore` raises `CaptureReviewRequired` with the missing codes before it writes anything. `plan_restore` accepts these (and only these) acknowledgements in isolated plans; the safety copy's exact credential-issue match ignores them. The restore review shows "Personal Context will be restored but stay locked. Nothing has been written yet." with one checkbox per case carrying its copy, for isolated restores too.
- **Settings (AC#5).** `personal_context_panel._LOCKED_REASON_COPY` holds the spec's four texts; the locked state still offers only **Try again**.
- **Dead code.** `personal_context_not_restored` (constant, message and its UI test) is gone: no restore holds the repository back any more.
- **Docs (AC#7).** User Guide troubleshooting rows for the four locked states and the restore paragraph; owner inventory restore paragraph; the SQLite inventory's restricted-validation prose, and `Tests/DB/test_private_sqlite_inventory.py` admits exactly the `stamp_restored_key` call (C101/C102 are the two `connect_private_sqlite` sites).

**Evidence.** Red on the TASK-40135 commit's source (0533473632, tests kept): marker and mismatch tests read `(True, 'profile_locked')`; the isolated stamping cases raised `TypeError` (no `isolated`); the acknowledgement tests did not raise; the five panel cases had no `#personal-context-locked-reason` copy; the UI acknowledgement test lacked the copy; the `no_keyring` and both isolated end-to-end children failed their assertions. `test_an_integrity_failure_past_the_manifest_is_not_a_key_mismatch` passes there too (it pins an invariant base already keeps); no negative control was run for it.

**Tests.** `Tests/Personal_Context/test_repository.py` (marker reasons across two starts, key mismatch with the same and another `key_version`, integrity past the manifest); `Tests/Backup_Recovery/test_credentials.py` (isolated stamping per spec row, acknowledgement refusal in both modes); `Tests/Backup_Recovery/test_restore_data_groups.py::...[no_keyring-include]` (no secure keyring: locked with `key_install_unavailable`, keyring and every row unchanged, the reason read by a fresh process, then a second Replace from the same archive once the keyring is back is readable: AC#6), `[isolated-include]`, `[isolated_elsewhere-exclude]`; `Tests/UI/test_settings_personal_context.py::test_locked_profile_says_why_and_offers_only_what_a_locked_profile_allows`; `Tests/UI/test_backup_data_groups.py` (acknowledgement copy; keyring-unavailable plan line). `scripts/check_markup_interpolation.py` OK. Covering run (`Tests/Personal_Context`, `test_credentials.py`, `test_restore_data_groups.py`, `test_restore_plan.py`, the two UI files, `-n 4`, load 26-36): 1,000 passed, 2 failed; the two (`test_restore_plan.py::test_shipped76_archive_stages_installed_schema_without_changing_source[False|True]`, a ChaChaNotes dispatch-checkpoint row diff) fail identically on the pca base ab4bdf751d. The end-to-end children take 50-90 s each serially.

Files: `tldw_chatbook/Personal_Context/{repository_keys.py,recovery.py}`, `tldw_chatbook/Backup_Recovery/{restore_plan.py,staging.py,replacement.py,recovery_service.py,sqlite_validation.py}`, `tldw_chatbook/UI/Screens/backup_restore_screen.py`, `tldw_chatbook/Widgets/Settings_Widgets/personal_context_panel.py`, the tests above, `Docs/User_Guide/settings/personal-context-profile.md`, `backlog/docs/{backup-recovery-owner-inventory.md,sqlite-private-owner-inventory.md}`.

