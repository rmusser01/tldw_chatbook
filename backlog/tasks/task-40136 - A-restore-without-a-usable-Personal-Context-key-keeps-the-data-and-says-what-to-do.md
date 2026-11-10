---
id: TASK-40136
title: "A restore without a usable Personal Context key keeps the data and says what to do"
status: To Do
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
- The obvious-looking way out, **Remove local profile**, deletes the data and the key (`repository.py:4972-4997`).

The restore cases in the approved design (spec Q2 table; decisions 3 and 6):
- **No key in the backup.** A backup without credentials, restored onto a device that lacks the key.
- **Isolated restored profile.** It runs with `keyring.backends.null.Keyring` (`Backup_Recovery/isolated_restore.py:393`, `:423-426`), which the key protector refuses (`key_protector.py:138-141`).
- **No secure keyring on a replace restore.** The key stays in the retained encrypted restore copy (`replacement.py:1604-1620`).
- **Key mismatch.** The stored account's key does not authenticate the database's manifest.
  - This is detected by authentication (TASK-40133), never by version. `key_version` starts at 1 (`key_protector.py:48`) and only a link bumps it (`repository.py:2051-2055`). So Remove local profile followed by a new profile, or an unlinked profile from another machine, gives a different key with the same version.
  - Today that surfaces as `ProfileIntegrityError`, which `bootstrap.py:69-73` reports as the tamper-like `profile_integrity_invalid`.
  - A restore never stamps a non-authenticating account (TASK-40135), so this case comes from outside restores.

The spec's Q2 table gives each case a named reason and the copy.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 In each restore case (key not in the backup, isolated restored profile, no secure keyring on a replace restore), the restored database is byte-identical afterwards, no keyring entry is created, changed or deleted, and Personal Context is locked with that case's reason: `personal_context_key_not_in_backup`, `personal_context_keyring_unavailable` or `personal_context_key_install_unavailable` (red first)
- [ ] #2 A database whose stored account holds a different key, including a different key with the same `key_version`, is detected by authentication and locked as `personal_context_key_mismatch`, not `profile_integrity_invalid`; its data is byte-identical and no keyring entry changes (red first)
- [ ] #3 The restore plan lists the first three cases as issues the user must acknowledge before the restore runs
- [ ] #4 The Settings panel's locked state shows the spec's copy for each reason, including that Remove local profile deletes the data, and `scripts/check_markup_interpolation.py` passes
- [ ] #5 After a replace restore with no secure keyring, a second replace restore from the same archive once a keyring is available makes the records readable
- [ ] #6 The Personal Context User Guide troubleshooting table describes these states
<!-- AC:END -->
