---
id: TASK-42113
title: "A Replace at the default config location fails for pet cases, and the pet cannot start after one"
status: To Do
assignee: []
created_date: 2026-11-20 17:26
updated_date: 2026-11-20 17:26
labels:
- follow-up
- backup
- pets
priority: high
references:
- tldw_chatbook/Widgets/Tamagotchi/tamagotchi_storage.py
- tldw_chatbook/Backup_Recovery/control_records.py
- Tests/Backup_Recovery/test_created_owner_rollback.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by TASK-42022 (round 18), from temporary probes with a scratch HOME. TASK-42000 (round 17) made pet settings a per-profile file in the config folder and added it to backups. Its tests used only a separate config folder.

At the default config location (`~/.config/tldw_cli`) the config folder holds `recovery-bootstrap`, so it is never a root; each config-folder file is bound as its own root. The probes showed:

- A Replace from a backup with no pet file, while the profile has one (`[pets-none]`), fails at activation with `activation_profile_outside_maintenance` (`control_records.py` ~601) and rolls back (`replacement_rolled_back`).
- `[pets-missing]` and `[pets-dated-only]` fail with `replacement_destination_parent_required`.
- After a successful default-location Replace, the footer pet fails with `RecoveryRequired: storage_scope_not_enrolled`: a pet file created after the binding falls under no root.

So a restore at the default location, where most users keep their profile, can fail for common backups since TASK-42000.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Tests run the pet Replace cases at the default config location (under a scratch HOME) and reproduce each failure above before the fix
- [ ] #2 A Replace at the default location completes for every pet case the separate-folder suite covers, with the same outcome as on a separate config folder
- [ ] #3 The footer pet starts and saves after a default-location Replace, and on a default-location profile that was never restored; no RecoveryRequired
- [ ] #4 The notes say, with evidence, whether other config-folder file owners hit the same failures at the default location; any that do are fixed here or have their own task
- [ ] #5 The default-location cases run on a PR or dev-push lane, with their time recorded
<!-- AC:END -->
