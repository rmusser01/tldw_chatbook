---
id: TASK-42151
title: "A Replace at the default config location without the Settings group is refused at its first binding"
status: To Do
assignee: []
created_date: 2026-11-20 20:06
updated_date: 2026-11-20 20:45
labels:
- follow-up
- backup
- recovery
priority: medium
references:
- tldw_chatbook/Backup_Recovery/replacement.py
- Tests/Backup_Recovery/test_created_owner_rollback.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by the petsdef stream (TASK-42113, round 18); it predates the pet work. At the default config location (`~/.config/tldw_cli/config.toml`, with `RecoveryService(default_control_root())` as the F9 default-profile workflow uses), a Replace whose data groups leave out the Settings group fails. TASK-42113's notes (round-18 petsdef branch) describe the reproduction: a temporary `location` parameter in `Tests/Backup_Recovery/test_created_owner_rollback.py` ran every case at the default location, and temporary wrappers named each failure. For the `library` and `projects` cases it is `replacement_config_container_unverified`, the third refusal in `_first_profile_container` (`tldw_chatbook/Backup_Recovery/replacement.py:871`; the raise at `:971`), at the Replace's first binding, with no config-folder file involved. On the round-18 petsdef branch (fu/r18-petsdef, not yet on dev), TASK-42114 gave the scenario a `LOCATION` input (`separate` or `default`), but `test_later_rollback_retires_the_files_a_restore_created` hard-codes `LOCATION='separate'` (line 674 there). So these cases can run at the default location with a test that passes `LOCATION='default'`.

Rechecked at d85e90bc9e with TASK-42113's temporary harness change applied and then reverted (`-n 2`, through the test wrapper): `[library-default]` and `[projects-default]` both fail, the restore ending `state: failed, phase: replacing, issues: ('backup_operation_failed',)`. The harness's own cases keep the config in a separate folder, so CI does not exercise this; real profiles use the default location.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The cause is named: which check in `_first_profile_container` refuses at the default location, and why it holds there
- [ ] #2 A Replace with the `library` case's groups, and one with the `projects` case's groups (neither includes Settings), at the default config location completes, or is refused before any work with a reason the user can act on, as the cause supports
- [ ] #3 A test runs those two cases at the default location (scratch HOME, `default_control_root()`), red before the change
<!-- AC:END -->
