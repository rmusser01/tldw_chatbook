---
id: TASK-41719
title: "A settings restore that includes pet settings refuses with replacement_destination_parent_required"
status: Done
assignee: ['@claude']
created_date: 2026-11-18 06:57
updated_date: 2026-11-19 15:45
labels:
- follow-up
- backup
priority: medium
references:
- tldw_chatbook/Backup_Recovery
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by TASK-41521 (round 16). A restore that includes the settings group with pet settings refuses with `replacement_destination_parent_required` whenever there is a pet file to restore, whether `~/.config/tldw_chatbook` is missing or holds a dated pet copy. `tamagotchi.config` lives at that global path, outside every profile and shared by all of them.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A settings restore with pet settings either restores them or excludes them with a named, user-visible reason; it never fails the whole restore
- [x] #2 Tests cover the folder missing and present, red before the fix
<!-- AC:END -->

## Implementation Plan

1. Reproduce: a cross-profile replace restore of the Settings group from a source with a pet file, into a target whose `~/.config/tldw_chatbook` is missing, and one holding a dated copy.
2. Choose exclusion, not restore: restoring would publish into a folder shared by every profile, which is a product decision about what a restore may overwrite; excluding overwrites nothing.
3. In `plan_restore` (every replace plan goes through it), leave `tamagotchi.config` rows out of a replace restore's selection, with their root when it holds nothing else, and add a named plan issue with a review message.
4. Also never map a target pet file as an absence to retire (`selected_absent_target_ids`), so a backup whose pet folder lacks the main file cannot delete the device's pet settings.
5. Tests red before the fix: the folder missing, present with a dated copy, and present with the main file; the target's pet files unchanged after the restore.

## Implementation Notes

**Summary.** A Replace restore now leaves pet settings out instead of failing. `tamagotchi.config` lives in `~/.config/tldw_chatbook` (`%APPDATA%\tldw_chatbook` on Windows), one folder shared by every profile and outside all of them. Publication fences only a profile's registered folders, so it refused to publish a new pet file there (`replacement_destination_parent_required`, at `_register_publication_parents`' first check) after the review had shown the restore as fine. Exclusion was chosen over restoring because restoring would mean letting a Replace publish into a device-wide folder, which is a product decision about what a restore may overwrite; excluding overwrites nothing. An isolated restore is unchanged and still restores pet files.

- `restore_plan.plan_restore` (every replace plan, from the Backup screen, the CLI and `resolve_destinations`, goes through it): in replace mode without a local snapshot, `tamagotchi.config` files and their synthetic roots are dropped from the destinations and the group selection, and the plan gets the issue `pet_settings_left_in_place`. Capture gives each single file its own synthetic root, so the root goes with its file. A later rollback (`local_snapshot`) is unchanged.
- `restore_groups.selected_absent_target_ids` never maps a target pet file as an absence to retire. Without this, a backup whose pet folder held only a dated copy records the main file as absent, and the Replace planned to retire the device's `tamagotchi_pets.json`; the publication then rolled back (`replacement_rolled_back`), failing the whole restore. The owner set and issue code live in one place (`restore_groups.LEFT_IN_PLACE_OWNERS`, `LEFT_IN_PLACE_ISSUE`).
- User-visible reason: `recovery_service._ISSUE_MESSAGES` explains it ("Pet settings are not restored: this device keeps them in one folder shared by every profile, so restoring into an existing profile leaves that folder as it is. Manual extraction can recover them from this backup."); the Backup screen's restore review shows that message beside the issue, as it does for Personal Context; the CLI lists the issue code in its plan. `Docs/Backup-and-Recovery.md` (Replace section) says so.

**Behaviour change to know.** At base, a Replace restore did overwrite the device's pet file when the target already held every pet file in the backup at the same path (probe: target main file `{"target"...}` became the backup's `{"source"...}`, restore succeeded). It failed in every other shape: pet folder missing, a dated copy only, or any dated copy in the backup the target lacked. With this change a Replace never writes or removes pet files in any shape.

**Tests (AC#2).** `Tests/Backup_Recovery/test_restore_data_groups.py::test_replace_restore_leaves_pet_settings_in_place_and_says_why`, two children with their own HOME (backup on one, Replace of Conversations, Settings and Workspaces on the other): `[missing]` (no pet folder at the target), `[present]` (the folder holds the main file and a dated copy), `[dated-source]` (the backup holds only a dated copy, the target holds the main file). Each asserts the restore succeeds, the plan names `pet_settings_left_in_place`, restores the config, plans no pet row to restore or retire, and the pet folder is exactly as before (still absent for `[missing]`).
- Red at base (`restore_plan.py`, `restore_groups.py`, `recovery_service.py`, `backup_restore_screen.py` and `rag_indexing.py` from c176f553b3, tests unchanged): `[missing]` and `[dated-source]` failed with the restore `failed` in `replacing` (`replacement_destination_parent_required` underneath, traced in a probe); `[present]` failed on the issue assertion because base overwrote the main file instead.
- Red with only the absence skip removed: `[dated-source]` failed, restore `failed` in `replacing` (`replacement_rolled_back`, probe: the plan retired the target's main pet file).
- Green: 3 passed, about 100 s each (`-n 4`, load 30-60).
- `Tests/UI/test_backup_data_groups.py::test_restore_review_says_why_pet_settings_stay_in_place`: the review shows the issue with its message. Red at base (the issue line had no message); green. The file: 21 passed (`-n 4`). It already has a whole-file entry in the UI census, so the census is unchanged.

**Lane.** `test_restore_data_groups.py` already runs whole in the Backup_Recovery owner suites job (push to dev); the three cases add about 5 min of child time there (noted in the job's comment).

**Covering runs** (`-n 4`, load 26-61): the Backup_Recovery plan, destination, data-group, empty-group, absence, later-rollback, group-scope, isolated-restore, pet roundtrip, RAG discovery/indexing/validation, group CLI and RAG retained roundtrip files, with `Tests/UI/test_backup_restore_screen.py`, `test_backup_data_groups.py` and `test_backup_restore_destinations.py`: 562 passed, 2 failed (18 min). The two (`test_shared_home_pet_roundtrip.py::test_native_shared_home_pets_and_backups_restore_for_both_profiles`, `test_rag_retained_roundtrip.py::test_native_rag_retained_roundtrip_with_fresh_passive_reads`) fail the same way with all five changed source files at base c176f553b3: their capture preview is incomplete (`missing_required`) before anything this task touches runs. Serially, `Tests/Architecture/test_backup_owner_inventory.py` and `test_module_size_ratchet.py`: 284 passed. `./scripts/preflight.sh`: passed. Ruff: no new findings on the changed lines (the five it reports in the touched files are on untouched lines and exist at base).

**Files.** `tldw_chatbook/Backup_Recovery/restore_plan.py`, `tldw_chatbook/Backup_Recovery/restore_groups.py`, `tldw_chatbook/Backup_Recovery/recovery_service.py`, `tldw_chatbook/UI/Screens/backup_restore_screen.py`, `Tests/Backup_Recovery/test_restore_data_groups.py`, `Tests/UI/test_backup_data_groups.py`, `Docs/Backup-and-Recovery.md`, `.github/workflows/derived-artifacts.yml`.
