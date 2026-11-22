---
id: TASK-42151
title: "A Replace at the default config location without the Settings group is refused at its first binding"
status: Done
assignee: ['@claude']
created_date: 2026-11-20 20:06
updated_date: 2026-11-21 21:52
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
- [x] #1 The cause is named: which check in `_first_profile_container` refuses at the default location, and why it holds there
- [x] #2 A Replace with the `library` case's groups, and one with the `projects` case's groups (neither includes Settings), at the default config location completes, or is refused before any work with a reason the user can act on, as the cause supports
- [x] #3 A test runs those two cases at the default location (scratch HOME, `default_control_root()`), red before the change
<!-- AC:END -->

## Implementation Plan

1. Reproduce the `library` and `projects` cases at `LOCATION='default'` with a temporary probe that names which clause of the third `replacement_config_container_unverified` refusal in `_first_profile_container` holds, and why it holds only at the default location.
2. Decide whether that refusal protects user data. Keep it if it does (and make the refusal actionable if the cause supports that); fix it if it is wrong.
3. Pin the outcome with a default-location test for both cases (scratch HOME, `default_control_root()`) in `Tests/Backup_Recovery/test_created_owner_rollback.py`, red before the change; then prove the next startup and a rollback still work.

## Implementation Notes

**Cause (AC#1).** The refusal is the third `replacement_config_container_unverified` in `_first_profile_container`, the compound check after `actual` is built. A test-only wrapper (no source edit) recorded it for both cases at the default location. `controls` was empty, and two of the check's clauses held:
- `protected_overlap`: the bootstrap root (`~/.config/tldw_cli/recovery-bootstrap`), the service control root (`recovery/control`), the staged candidate and the acquired archive (both under `recovery/control-work`) all sit inside the config folder;
- the registry clause: the `bootstrap.unbound` root `recovery-bootstrap/unbound-owner` overlaps the folder, is not one of the profile's names and is not a profile file. Its exception applies only when the bootstrap root is one of `controls`.

`controls` comes from `_preserved_default_controls`, which recognizes those fixed controls only through `service_storage.is_preserved_service_container`. That function requires the folder to be a synthetic restore destination. With the Settings group, config files restore into `~/.config/tldw_cli`, so it is one, and the binding proves the folder with the controls set aside and never enrolls it. The `library` and `projects` plans map nothing there: no destination, restore, retire or container path is in the folder; they only preserve its files, controls included. So nothing recognized the controls. On a separate config folder none of those control paths are inside the folder, which is why the harness never saw it.

**The refusal was wrong (AC#2: completes).** It guards against enrolling a folder that holds the recovery controls as a profile root, and that guard stays. But the default folder is never enrolled anyway: with Settings the binding skips it. A plan that places nothing in the folder is the safer of the two cases, yet it was refused only because the controls were recognized through the destination test. The new `_untouched_service_work` recognizes the same fixed service store and private work. It uses `is_preserved_service_container`'s item conditions: a preserved `recovery.control:service` item and `is_private_service_work`, which verifies the default storage. It applies only when no destination, restore, retire or container path overlaps the folder, and the folder is a real directory. `_preserved_default_controls` and `protected_overlap` accept it beside the existing synthetic test. The bootstrap-root conditions are unchanged. Every other refusal still holds, pinned below.

**Tests (AC#3).** `test_created_owner_rollback.py::test_a_default_location_replace_without_settings_completes[library|projects]` runs the existing scenario at `LOCATION='default'`: config at `~/.config/tldw_cli/config.toml`, `RecoveryService(default_control_root())`, scratch HOME. It checks the Replace, then (new `admit_after_restore` key) a start with startup permission `(True, 'startup_allowed')` plus `acquire_storage()`, then the later rollback retiring every created file, then a start again (`then='later_rollback'`). Red before the fix: both `state: failed, phase: replacing, issues: ('backup_operation_failed',)`, the reported symptom. Green after: `2 passed in 181.25s` (175 s each at `-n 4`, load 44-56). The file is already a whole-file target of the dev-push Backup_Recovery owner-suite job.

Unit pins in `test_default_service_container.py`. `test_first_default_config_binding_preserves_exact_service_only` and `test_first_default_binding_never_enrolls_control_or_config_parent` now also run with the archive root mapped `elsewhere`, outside the config folder. `elsewhere-valid` proves the folder, `elsewhere-service_replaced` sees the service change, and the binding enrolls only `config.toml` and the profile file, never the folder or a control. These three were red without the fix (`replacement_config_container_unverified`). The refusals `unknown_protected`, `nondefault_protected`, `unknown_sibling`, `unpreserved_bootstrap`, `bootstrap_mutation` (a planned retire inside the bootstrap root) and `foreign_bootstrap_registry` still refuse for `elsewhere`, before and after (40 passed). This file is in no CI lane (see follow-ups).

**Other runs** (through the wrapper). The whole dev-push owner-suite lane step, run locally with its deselects at `-n 4 --timeout=1500`, gave 1074 passed in 56:50 at load 17-46. It includes TASK-42113's `test_a_default_location_pet_replace_refuses_and_keeps_the_pets`, whose plans do touch the config folder, so they still refuse as pinned. `test_first_replacement_binding.py`, `test_first_config_lock.py` and `test_first_binding_unused_scaffolds.py` all passed (45 passed together with the next file). `test_default_service_replacement.py` (in no lane) failed under load in three runs with the fix: its F9 child hit its fixed 150 s deadline twice, and the later-rollback child's `asyncio.timeout` once, after the F9 replacement had passed. At base dc371f0842 with this file reverted it passed once (428 s, close to its 600 s cap) and failed once the same way (the F9 child's 150 s deadline). Its default-profile plan restores Settings into the folder, so the new branch never applies to it.

**Review rev1.** The safety clause in `_untouched_service_work` (the folder must not overlap any destination, restore, retire or container path) is now pinned by `test_default_service_container.py::test_untouched_default_folder_refuses_any_planned_path_in_it`. Each case adds one planned path to the valid `elsewhere` plan and expects `replacement_config_container_unverified` from `_first_config_container`:
- `nonsynthetic_root`: a second, non-synthetic root mapped onto the folder;
- `restore_child`: a restore of a file in it;
- `retire_child`: a retire of its `note.txt`;
- `container_child`: a new container in it;
- `ancestor_destination`: a destination that is the folder's parent.

With the clause deleted, all five failed `DID NOT RAISE <class 'ValueError'>` (5 failed, 40 passed); with it, the file passes (45 passed). The helper now runs its `any(item ...)` check for a preserved default service item before the overlap scan.

**Files.** `tldw_chatbook/Backup_Recovery/replacement.py`: new `_untouched_service_work`, a call to it in `_preserved_default_controls`, and one in `_first_profile_container`'s `protected_overlap`; +36 lines. Also `Tests/Backup_Recovery/test_created_owner_rollback.py` and `Tests/Backup_Recovery/test_default_service_container.py`.
