---
id: TASK-42115
title: "A default-location Replace cannot create or start the pet: make pet files config companions with a fixed-file set"
status: To Do
assignee: []
created_date: 2026-11-20 18:48
updated_date: 2026-11-20 22:34
labels:
- follow-up
- backup
- pets
priority: high
references:
- tldw_chatbook/Backup_Recovery/config_participants.py
- tldw_chatbook/Backup_Recovery/settings_file_participants.py
- tldw_chatbook/Backup_Recovery/storage_admission.py
- tldw_chatbook/Backup_Recovery/publication.py
- tldw_chatbook/Backup_Recovery/later_rollback.py
- Tests/Backup_Recovery/test_created_owner_rollback.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Split from TASK-42113 (round 18, coordinator decision 2026-11-20). TASK-42113 keeps the default-location reproduction (its refusal tests) and the notes on other owners. The changes below extend the guarded publication and rollback proofs, so they get their own stream and review.

Until then, the failure is fail-safe, and `test_created_owner_rollback.py::test_a_default_location_pet_replace_refuses_and_keeps_the_pets` pins it. At the default location:
- a Replace that would create a pet file refuses before publication (`replacement_destination_parent_required`);
- a Replace that retires the profile's pet file refuses at activation (`activation_profile_outside_maintenance`) and rolls back.

In both cases nothing is created, the profile keeps its pets, and the app starts.

**Still exposed:**
- **dev builds since TASK-42000:** an install whose registry already holds a dated pet copy beside `config.toml` as a root of its own stays exposed. That means a backup taken since TASK-42000 and before TASK-42114. Once the pet store rotates the copy away, every admission refuses and the app does not start. TASK-42114 stops new registrations but cannot remove old ones, so such an install needs AC#6 or the workaround.
- **main builds** (by reading main's code, TASK-42114): backups register the dated copies in the shared `~/.config/tldw_chatbook`, and main's pet store rotates them there, so a main install stays exposed for as long as it runs a main build. No TASK-42000 build rotates the shared folder, so upgrading ends the trigger. Only an install that already stopped on main needs AC#6 or the workaround.

The workaround is untested: recreate an empty file, mode 0600, at the missing path.

**2026-11-20, coordinator decision after review rev2:** deleted-file-root tolerance needs one design across all root consumers, the recovery journal included. The attempt is parked on `fu/r18-deleted-roots-wip` (tip `ee6d0ac966`, not pushed) and moves here, with TASK-42113's step A (`_bind_activation`, `bootstrap._binding`). It recorded each regular-file root with its folder's inode and let a deleted file root hold nothing, then scope its own exact path.

Three reviews in a row each found another root consumer that the per-site approach missed:
- **rev1, finalization:** `publication._finalization_session` required coverage of a root the publication roots had dropped, so a later rollback, its abort and a second Replace stuck in `recovery_required`.
- **rev1, exact-path ownership:** the abort (`replacement._abort_prepublication`) and containment could not cover the deleted root's own path.
- **rev2, crash recovery:** under a recovery journal, `Admission._groups` (and `_publication_roots`, which passes `deleted=()`) still refuses any missing root the journal did not move (`replacement._recovery_admission_aliases`: `recovery_root_unverified`). After a `[pets-none]` Replace at the default location and a second Replace killed at `finalize_candidate`, both `finish` and `rollback` failed, and startup was refused.

At the default config location (`~/.config/tldw_cli`), the config folder holds `recovery-bootstrap`, so it is never a root. Only `config.toml`'s own members and the three fixed leaves in `_CONFIG_CAPTURE_LEAVES` (`ui_state.toml`, `recent_emojis.json`, `runtime_policy.json`) have a path there through admission (`config_participants.sibling_selector`/`companion_guard`), capture (`_config_capture_sources`), publication (`_config_publication_scopes`) and later rollback (`_config_file_relations`). Pet files fall back to roots of their own. Evidence from TASK-42113's probes (scratch HOME, `RecoveryService(default_control_root())`):

- A Replace that would create `tamagotchi_pets.json` (`[pets-missing]`, `[pets-present]`, `[pets-ungrouped]`) refuses with `replacement_destination_parent_required` for the destination `~/.config/tldw_cli` (`replacement._register_publication_parents`).
- After a successful default-location Replace, the footer pet fails with `RecoveryRequired: storage_scope_not_enrolled`. The `pet` route admits the config folder itself (`raw_participants.scope`, `admission_paths` starting with the parent) and leases each dated copy on its own (`settings_file_participants.preflight`). At the default location, neither falls under a root.

Since TASK-42114, the pet store's dated copies are excluded from backups. Its review also made a restore skip the dated rows that older archives hold (`rotating_copy`, coordinator decision 2026-11-20), so neither capture nor restore handles a dated copy any more. A backup holds only two fixed pet files: `tamagotchi_pets.json` (local id `""`) and `tamagotchi_pets.seeded` (local id `seeded`). That fits the fixed-leaf mechanism once it allows an owner more than one fixed leaf.

Plan (TASK-42113's steps B to D):
- **B.** The `pet` route becomes a config companion: `sibling_selector` returns the selector for `tamagotchi_pets.json`. The preflight checks dated-copy members under the companion guard instead of leasing each one, and the config folder is no longer leased on its own.
- **C.** `_CONFIG_CAPTURE_LEAVES` generalizes from owner → leaf to owner → {local id: leaf}, with the pet owner's two leaves. Affected: `storage_admission._config_capture_*`, `publication._config_scope_candidates`/`_config_publication_scopes`, and `later_rollback._config_file_relations`, `_config_file_retirement_scopes` and `_created_config_file_targets` (whose owner map must include the pet adapter).
- **D.** The pet cases of `test_created_owner_rollback.py` at the default location: the scenario already takes `LOCATION`. They need laning in the owner suites.

Related, from TASK-42113:
- Every other config-folder owner hits the same creation refusal at the default location: `config.history` (`config.toml.bak`, `config_backup_<stamp>.toml`), `rag.definitions` (`rag_pipelines.toml`) and `notes.templates` (`note_templates.json`). This predates TASK-42000. `config.history`'s dated snapshots have the variable-name problem pets had before TASK-42114, so those owners are left to a separate task. Retiring them, and creating them later from the app, were not probed.
- `bootstrap._binding` (a bound profile's own binding check) is part of the same design. Changing it means updating the `bootstrap.py` digest that `Tests/Backup_Recovery/test_config_companion_missing_record_evidence.py` pins.
- Until the design lands, a registered single file that the user deletes still stops every admission (TASK-42114's AC#4 notes).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 At the default config location, a Replace completes for every pet case the separate-folder suite covers ([pets-missing], [pets-present], [pets-dated-only], [pets-none], [pets-ungrouped]), with the same outcome as on a separate config folder, including the later rollback; each case is red before the change
- [ ] #2 The footer pet starts and saves after a default-location Replace, with no RecoveryRequired; a test is red before the change
- [ ] #3 The fixed config leaves keep their current behaviour at both locations (the config sibling and companion suites stay green), and pet files are never registered as roots of their own at either location
- [ ] #4 The default-location pet cases run on a PR or dev-push lane, with their times recorded
- [ ] #5 The notes say whether config.history, rag.definitions and notes.templates should join the fixed-leaf set, and a task is filed for them if they should
- [ ] #6 One design for a deleted (absent) file root covers every root consumer: admission grouping, capture naming, maintenance sessions, publication and finalization coverage, containment, Replace activation (step A), a bound profile's binding check, the abort, and the recovery journal (finish and rollback after a crash). It has tests, each red first, for the next Replace, a later rollback, an abort, finish and rollback after a crash, and startup, at the default location and on a separate folder. An absent folder root (an unmounted drive) still refuses
<!-- AC:END -->
