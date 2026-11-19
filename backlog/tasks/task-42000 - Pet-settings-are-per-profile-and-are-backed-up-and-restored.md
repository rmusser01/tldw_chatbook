---
id: TASK-42000
title: "Pet settings are per-profile and are backed up and restored"
status: Done
assignee: ['@claude']
created_date: 2026-11-19 15:58
updated_date: 2026-11-19 17:59
labels:
- backup
- settings
priority: medium
references:
- tldw_chatbook/Widgets/Tamagotchi
- tldw_chatbook/Backup_Recovery/settings_file_participants.py
- tldw_chatbook/Backup_Recovery/restore_groups.py
- Docs/Backup-and-Recovery.md
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Pet settings (`tamagotchi.config`, the pet files) live in one folder, `~/.config/tldw_chatbook`, which sits outside every profile and is shared by all of them. Because of that, a Replace restore that includes pet settings fails the whole restore with `replacement_destination_parent_required` (TASK-41719): publication will not create files in a folder outside the profile.

Owner decision, 2026-11-19, on TASK-41719's options: "Lets change that so pets are per-user/profile and are backed up". Pets move into each profile, so they are captured and restored like the profile's other settings files and no longer need a shared folder.

Coordinator's call on the upgrade path (engineering, stated to the owner): a profile that has no pet settings of its own starts from a copy of the old shared files the first time it needs them. That way no profile loses the pets it shows today. The shared files are never modified or deleted.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Each profile keeps its own pet settings: a change made under one profile is not seen by another profile on the same device
- [x] #2 A backup captures the profile's pet settings, and a Replace restore puts them back into that profile; pet settings never make a restore fail
- [x] #3 Isolated restore, Manual extraction and a later rollback treat the pet files like the profile's other settings files
- [x] #4 On upgrade, a profile with no pet settings of its own starts from a copy of the old shared files; the shared files are never changed or deleted
- [x] #5 TASK-41719's restore cases (pet folder missing, present, backup with only a dated copy) succeed with the pets restored, each red before the change
- [x] #6 The User Guide and Docs/Backup-and-Recovery.md describe per-profile pets
<!-- AC:END -->

## Implementation Plan

**What a profile is here.** A profile is a config selector (`TLDW_CONFIG_PATH`, else `~/.config/tldw_cli/config.toml`) plus its data folder (`<data_dir>/<users_name>`). Its settings files sit beside the selector: `ui_state.toml`, `recent_emojis.json`, `runtime_policy.json`, `note_templates.json`, `rag_pipelines.toml`, `config.toml.bak`/`config_backup_<date>.toml` and `themes/`. Each resolves `config._get_effective_config_path().parent` at call time (writers), or `discovery_context(config).config_path.parent` (backup discovery). Capture gives each single file its own synthetic root (`capture.py`, `root:sha256(logical_id)`), and restore derives the file's destination from the target profile's config (`staging._config_targets`). `config.history` is the model for an owner with one current file plus dated copies.

**1. Where pets live.** `<profile config folder>/tamagotchi_pets.json`, plus its dated copies `tamagotchi_pets.backup_<YYYYmmdd_HHMMSS>.json` in the same folder. The owner ID stays `tamagotchi.config`, and it stays in the Settings group.

**2. Readers and writers that change.**
- `Widgets/Tamagotchi/tamagotchi_storage.py`. `ConfigFileStorage()` takes its path from `config._get_effective_config_path().parent` and loses the unused `app_name` argument. `JSONStorage.__init__` seeds a missing file from `self._initial_data(operation)`, which is `{}` by default, and raises when that first write fails, so a half-made store is never used. `ConfigFileStorage._initial_data` holds the upgrade copy (step 5).
- `Backup_Recovery/settings_file_participants.binding`. A pet store counts as installed only when it is the exact `ConfigFileStorage` type and its path is the profile's config folder plus `tamagotchi_pets.json`, as `notes.templates` already does. The pet route's dated-copy preflight and `pet_operation` are unchanged; they already work relative to the selected file.
- `Widgets/AppFooterStatus.py`: comment only. The single `ConfigFileStorage()` call site stays, so `test_dormant_tamagotchi_path_constructors_have_no_installed_call_sites` keeps passing.

**3. Backup participant, archive and inventory.**
- `Widgets/Tamagotchi/recovery.py::_ConfiguredPets`:
  - `discover` returns the current file (`unused` when absent, with its config dependency, like every config-folder file) and every exact dated copy in the config folder. It no longer marks the folder's other files `unsupported`: those belong to other owners now. A dated copy keeps its leaf-keyed ID.
  - `_restore_path(profile, payload, root, config_target)` keeps every structural check (owner, exact IDs, synthetic top-level root, `parent_id == root_id`) and returns `config_target.parent / leaf`. `_parent()` goes away.
- `staging._config_targets` passes `config_target`.
- **Archive format: unchanged.** The owner ID, the logical IDs (`profile:<id>:tamagotchi.config` and `…:<dated leaf>`), the own synthetic root and the leaf `relative_path` do not depend on the source folder. Restore never reads a historical source path; it derives each destination from the target profile.
- So an older backup's pet rows are the same rows a new capture writes, and they restore into the per-profile folder, grouped or ungrouped (format 1 with no `data_groups`; nothing here is group-specific).
- **Inventory:** only the discovered paths change. Pet rows from two profiles stop being aliases unless the two profiles' config files share one folder.
- **Old multi-profile backups.** Every older backup with two or more profiles holds the pet files as alias rows with one `shared_group`. After this change, each alias's destination is its own profile's folder, and `restore_plan` refuses that with `shared_target_split`, or `shared_scope_expansion_required` when only one profile is selected. Plan: those two checks in `restore_plan` skip `tamagotchi.config` rows, through a named owner set with a comment. Each profile then gets its own copy of the old shared bytes, which is the upgrade rule applied to a restore. Staging's alias checks already allow that: equal bytes, and aliases are keyed by group and destination. (Decision point D5.)
- **Co-rooted files (crafted format-1 archives).** No row is ever dropped. Each payload in a root is checked against its own owner's expected destination, and one root has one destination. So a pet file that shares a root with other files restores only when that root's destination is the profile's config folder and every co-rooted file belongs there (for example `config.toml`). Otherwise the restore refuses with `owner_relocation_unverified:<owner>`: in derive mode through the root-consistency check, and otherwise through the per-file check. Today the same archive refuses with `owner_relocation_unverified:tamagotchi.config`.
- **Later rollback:** `later_rollback._CREATED_FILE_OWNERS` and `_CREATED_OWNER_CLASSES` gain `tamagotchi.config` → `("Widgets.Tamagotchi.recovery", "_ConfiguredPets")`. A cross-profile restore that creates pet files is then retired by TASK-41521's created-file proof, which re-reads the owner's own `discover` filtered by path, so dated copies keep their IDs. The shared-config-folder refusal (`local_snapshot_created_config_folder_shared`) applies to pet files as it does to other config-folder files. A same-profile restore is already covered by `_known_absences`.

**4. Isolated restore, Manual extraction, portable and detached import.**
- Isolated restore: destinations are derived from the isolated profile's config, so pets land in its config folder through `_config_targets`; nothing pet-specific is left.
- Manual extraction is owner-agnostic: payload files plus a report of logical ID, owner and relative path. With the archive unchanged, pet rows extract like any other Settings file; this is a test, not a code change.
- `portable_relocation`:
  - `_pet_destination`, and its exception in the stage-plan check, are deleted: pet targets are now under `attempt/profiles`.
  - The pet-only `AppData/Roaming` creation in `host_environment` is also deleted. The `APPDATA` variable stays.
- `detached_layout` (ADR-256 portable import):
  - It declares both the profile's pet files (`selected.parent`) and the old roaming files (`home/AppData/Roaming/tldw_chatbook`).
  - It captures the profile's own files when any exist, and otherwise the old roaming ones: the upgrade rule again, so an older portable copy keeps its pets.
  - Old roaming files that are not captured get one review line ("Old shared pet files were not imported; this profile has its own.").
  - Unknown files in the roaming folder still refuse with `detached_unknown_content`.

**5. The upgrade copy.**
- **When it runs:** in `ConfigFileStorage()` (the footer pet, only while `[tamagotchi] enabled`), inside the existing writing `pet` raw scope, only when the profile has no pet files of its own: no `tamagotchi_pets.json` and no dated copy. The scope's preflight has already listed the dated copies (`observed_files`).
- **What it copies:** the old shared current file, `~/.config/tldw_chatbook/tamagotchi_pets.json`, or `%APPDATA%\tldw_chatbook\…` on Windows, which is the old `_parent()` expression kept in the storage module. It is read without following links, capped at the owner's 16 MiB limit, and must hold a JSON object. The old dated copies are not copied (decision point D3).
- **Atomicity:** the copy is one write of the new current file through `_write_data`: a temporary file, then `raw.replace`, under the profile's admission. Before it there is no current file, and after it the profile holds the old pets.
- **Failure half-way:**
  - A read error other than "absent" (permissions, a directory, an oversized file) raises before anything is written. The footer already falls back to an in-memory pet and logs a warning, and the next start tries again.
  - A failed write raises: the temporary file is removed and no current file exists, so the next start tries again.
  - A link, a non-object or invalid JSON at the old path counts as "nothing to copy" and logs a warning, as `_read_data` treats bad JSON today.
- **The old shared folder** is only ever read by this copy. No backup, restore, rollback or writer touches it again: the owner's discovery, restore path and writers all point at the profile.

**6. Docs (AC#6).** `Docs/Backup-and-Recovery.md` (Settings group: per-profile pets, the upgrade copy, older backups), a User Guide paragraph (`Docs/User_Guide/settings.md`, beside the config-only settings), `Config_Files/default_config.toml`'s `[tamagotchi]` comment, `Docs/Development/Textual-Tamagotchis.md`'s storage bullet and `Docs/Architecture/backup-recovery.md`'s portable pet paragraph.

**7. Tests.** Each new or changed test is shown red at base (c176f553b3 + this task file) before the change.
- `Tests/Backup_Recovery/test_created_owner_rollback.py` (lane: Backup_Recovery owner suites):
  - New cases. `pets-missing`: the target's old shared folder is absent. `pets-present`: it holds a dated pet copy. `pets-dated-only`: the backup holds only a dated copy and the target has its own current file. `pets-ungrouped`: an ungrouped (format 1, no `data_groups`) backup, Replace-restored, with an old shared folder present on the target.
  - Each case writes the source's pet files where the installed owner discovers them (a new `pets` folder kind, taken from the owner's discovered path, so at base it is the old shared folder). It Replace-restores cross-profile, asserts the pets are in the target profile's config folder, then later-rolls back and asserts the created pet files are gone (for `pets-dated-only`: the dated copy retired and the target's own file back). It also asserts that the target's old shared folder is byte-identical throughout.
  - At base all four fail: `replacement_destination_parent_required` for the missing, present and ungrouped cases, `replacement_rolled_back` for dated-only. (AC#2, AC#3 later rollback, AC#4 "never changed", AC#5, the coordinator's ungrouped point.)
  - A Manual-extraction check (`preview_inert_extraction` plus `extract_inert` of the Settings group) on the `pets-missing` backup asserts the pet rows and their bytes (AC#3).
- `Tests/Backup_Recovery/test_shared_home_pet_roundtrip.py` is renamed `test_profile_pet_roundtrip.py` and rewritten:
  - Fixture: the base red (`missing_required` for `research.local` in both profiles, because the fixture points `research_db_path` at a database nothing creates) is fixed in the seed with `_populate_required_dependencies`, as the created-owner harness does.
  - Body: an old shared pet file is present in HOME, so both profiles start from a copy of it (AC#4). Alpha's later save is not seen by beta (AC#1), and the shared file stays byte-identical. Capture holds each profile's own pet rows with no aliases. An isolated restore puts each profile's pets into its own restored config folder, and fresh reads by `ConfigFileStorage` under each restored profile return that profile's pets (AC#3 isolated).
  - Its unit tests move to the new `_restore_path` signature. They add co-rooted cases: a pet file sharing a root with `config.toml` maps into the config folder; one sharing a root with a data-folder file (`personas`) refuses `owner_relocation_unverified`, in both the derive and the reviewed check (coordinator point 2).
  - The file is on no lane today (decision point D7).
- `Tests/Backup_Recovery/test_restore_plan.py` (lane): a crafted two-profile format-1 manifest whose pet rows share a `shared_group` (the shape every older multi-profile backup has) plans each alias into its own profile's config folder, where base raises.
- `Tests/Backup_Recovery/test_operational_owners.py` (lane):
  - Upgrade-copy tests (AC#4), one profile per test, with `TLDW_CONFIG_PATH` and HOME set: the old pets are copied once and the old files keep their bytes, inode and mtime; a profile with its own current file, or only a dated copy, is not seeded; an unreadable old file raises with no profile file written, and a retry after the fix seeds; a failing `raw.replace` leaves no current file and no temporary, and a retry seeds; an old symlink is not followed.
  - `test_optional_pet_default_and_exact_backups_block_unknown_children` becomes a config-folder discovery test: unrelated files beside the config are not claimed.
  - `test_real_pet_and_execution_history_writers_are_excluded` sets the config path.
- `Tests/Backup_Recovery/test_inventory_unused_owners.py`: the `pets_*` routes are rewritten for the config folder (absent → `unused`; linked file → not included).
- `Tests/Backup_Recovery/test_settings_file_participant_lifetimes.py`: the `pet` family patches `config._get_effective_config_path`, as `templates` does.
- `Tests/Backup_Recovery/test_detached_layout.py`: the two Windows-skipped pet tests become OS-independent (files written directly). They cover only-profile pets captured, only-old-roaming pets captured, both present (profile's captured, roaming declared and reported), and an unknown roaming file refused. The portable relocation test expects pets under the attempt's profile config folder; if it still needs Windows, it is updated and reported as not run.
- Covering runs: the files above plus `Tests/Widgets/test_tamagotchi*.py`, `test_later_rollback.py`, `test_later_created_destinations.py`, `test_config_file_retirement.py`, `test_portable_relocation.py` and `test_isolated_restore.py`; then `Tests/Architecture/test_backup_owner_inventory.py` and `test_module_size_ratchet.py` serially, and preflight. The production diagnostic inventory changes for `tamagotchi_storage.py` (new warning calls); the rows are read before `--write`.

**Decision points for the coordinator (defaults in brackets):**
- D1. Re-seeding. A profile whose pet files are later all removed (by a Replace from a backup with no pets, or by hand) has none of its own again, so the next start copies the old shared file again. [Literal rule as written. The alternative, "seed once per profile", needs a durable marker.]
- D2. Two profiles whose config files share one folder share pets, as they already share `ui_state.toml`, themes, recent emoji and note templates. [Accept: per-profile means per config folder.]
- D3. The upgrade copies the current pet file only, not the old dated copies, which the app never reads. [Current file only.]
- D4. A Settings Replace from a backup whose profile had no pet files retires the target profile's pet files, like other settings files. Today it retires nothing when the source had no pet folder. [Like other settings files.]
- D5. Old multi-profile backups: each profile's alias restores into its own folder (the `restore_plan` carve-out). [Carve-out. The alternative is to refuse those pet rows by name.]
- D6. Crafted archives whose pet file shares a root with files from another folder are refused by name (`owner_relocation_unverified:<owner>`). This is the one place pets can still fail a restore, and the writer never produces that shape. [Refuse.]
- D7. `test_profile_pet_roundtrip.py` (isolated restore, AC#1, AC#4) runs on no lane. Adding it to the Backup_Recovery owner suites step needs a `derived-artifacts.yml` edit. [Coordinator's call.]

**Coordinator's decisions (2026-11-19):** D2–D7 accepted as planned. D1 was changed to "seed once": the copy from the shared folder happens at most once per profile, and a durable per-profile marker records it.

## Implementation Notes

**Summary.** Pets live beside the profile's config, in `tamagotchi_pets.json`, its dated copies and a `tamagotchi_pets.seeded` marker, so backup, Replace, isolated restore, Manual extraction and a later rollback treat them like the profile's other settings files. The archive rows are unchanged, so older backups restore into the profile. The first time the pet runs in a profile with no pet files and no marker, it copies the old shared file once; the shared folder is only ever read. Two restore-path bugs that pets exposed are fixed where every caller goes through them.

**Approach (files).**
- `Widgets/Tamagotchi/tamagotchi_storage.py`:
  - `ConfigFileStorage()` takes its path from `config._get_effective_config_path().parent`. `path_validation.validate_path` refuses a pet file that resolves outside that folder.
  - `JSONStorage.__init__` calls a `_prepare(operation)` hook inside its existing writing scope.
  - `ConfigFileStorage._prepare` seeds once. When the profile has no current file, no dated copy (the scope's `observed_files`) and no marker, it copies the old shared current file (D3). The read follows no link, is capped at 16 MiB and needs a JSON object; anything else is "nothing to copy" with a warning. The copy is one atomic `_write_data`; a failed write raises. The marker is then published from the scope's temporary with `raw.replace`.
  - A read error raises before anything is written, so the footer falls back to an in-memory pet and the next start tries again.
- `Backup_Recovery/raw_participants.py`: the pet write scope also admits `tamagotchi_pets.seeded`.
- `Backup_Recovery/settings_file_participants.py`: the installed pet path is the profile's config folder.
- `Widgets/Tamagotchi/recovery.py`:
  - `discover` returns the current file (`unused` when absent, with its config dependency), each dated copy and the marker (`…:seeded`, only when present), and nothing else from the config folder.
  - `_restore_path(..., config_target)` keeps every structural check and maps to `config_target.parent`.
- `Backup_Recovery/staging.py` passes the target config.
- `Backup_Recovery/later_rollback.py`: `tamagotchi.config` joins `_CREATED_FILE_OWNERS` and `_CREATED_OWNER_CLASSES`.
- `Backup_Recovery/restore_plan.py`: `_PER_PROFILE_ALIAS_OWNERS` exempts only `tamagotchi.config` alias rows from `shared_target_split` (D5; narrowed in fix round 1, below).
- `Backup_Recovery/portable_relocation.py`: `_pet_destination` and the pet-only `AppData/Roaming` creation are deleted.
- `Backup_Recovery/detached_layout.py`: the profile's own pet files are imported; old roaming ones only when the profile has none, otherwise declared and reported ("Old shared pet files were not imported; this profile has its own.").
- **Retired-root fixes.** Capture enrolls each config-folder file outside `_CONFIG_CAPTURE_LEAVES` as its own registered root. After a Replace retired such a file (pets under D4), the next start failed: first `FileNotFoundError` in `config_participants.verified_user_data_directory`, then `storage_admission_unavailable` from `storage_admission._contains_owned_path` (`resolve(strict=True)`). Both now treat a missing registered root as containing nothing.
- Comments: `AppFooterStatus.py`, `Config_Files/default_config.toml`.

**D1: the seed-once marker, and why.** The marker is `tamagotchi_pets.seeded` beside the pet file. The pet owner owns it: it is discovered, captured and restored as `profile:<id>:tamagotchi.config:seeded`, the raw pet scope writes it, and the inventory never shows it as unowned or unsupported.
- Existing per-profile places were rejected:
  - `config.toml` and `ui_state.toml` are replaced by a Settings Replace, so a backup without the flag would clear it, and the next start would copy the shared pets again.
  - Device-local stores and the recovery control records are another domain, and a restore does not carry them.
- Discovery lists the marker only when it exists, so a backup never records it as absent. A Replace therefore never retires a target's marker, and a backup that holds one restores it.
- So a profile that has taken its copy keeps the marker through any Replace, including one that removes all its pets (`pets-none` below), and is never seeded again. A profile that never ran the pet takes its one copy on first run.

**Acceptance criteria (evidence).**
- **AC#1:** `test_profile_pet_roundtrip.py::test_each_profile_keeps_its_own_pets_through_backup_and_isolated_restore`. Beta never sees the pet alpha saved, and each restored profile reads only its own pets.
- **AC#2:** `test_created_owner_rollback.py` cases `[pets-missing]`, `[pets-present]`, `[pets-dated-only]`, `[pets-none]` and `[pets-ungrouped]`: a cross-profile Replace, then a later rollback. The pets are in the target profile's config folder after the restore, with the source's text. "Pets never fail a restore" holds except for crafted co-rooted archives, which are refused by name (D6).
- **AC#3:**
  - Isolated restore: the roundtrip test; each profile's pets sit beside its restored config, and fresh `ConfigFileStorage` reads return them.
  - Manual extraction: `[pets-missing]` extracts the Settings group, and the report and outputs hold both pet rows with the source bytes.
  - Later rollback: every created pet file is retired, and `[pets-dated-only]` and `[pets-none]` bring the target's own file back.
  - Portable: `test_detached_layout.py` pet tests.
- **AC#4:**
  - Upgrade copy: `test_operational_owners.py::test_pet_store_copies_the_old_shared_pets_once`, `test_pet_store_never_copies_into_a_profile_with_pet_files[current|dated|marker]`, `test_a_failed_pet_copy_writes_nothing_and_the_next_start_retries[unreadable|publication]` and `test_pet_store_follows_no_link_out_of_place`, plus the roundtrip seeds.
  - Shared files never change: every test that has an old shared folder checks its bytes, and the unit and roundtrip tests also check its inode and mtime. The harness checks after the restore, after the pet start and after the rollback.
- **AC#5:** `[pets-missing]`, `[pets-present]` and `[pets-dated-only]`, each red at base (below). The ungrouped coordinator point is `[pets-ungrouped]`.
- **AC#6:** `Docs/User_Guide/settings.md` (new "The footer pet" section under Advanced Config, the only place the pet is turned on) and `Docs/Backup-and-Recovery.md` (Replace and isolated-restore sections). Also updated: `Docs/Development/Textual-Tamagotchis.md` and `Docs/Architecture/backup-recovery.md`.

**Red at base.** Base is c176f553b3 plus this task's test changes, with `tldw_chatbook/` checked out to base (`git diff > patch`, `git checkout`, run, `git apply` in one command; `-n 4`, load 19-30).
- `test_operational_owners`:
  - The discovery, copy-once, `never_copies[*]` and `follows_no_link` tests fail on their path assertions: the store and the owner still point at `~/.config/tldw_chatbook`.
  - `failed_copy[*]` fails with `DID NOT RAISE OSError`.
  - The retired-root test fails with `assert FileNotFoundError(...) is False`.
- `test_restore_plan::test_only_pet_aliases_…[one|both-tamagotchi.config]` fails with `assert 'shared_scope_expansion_required' is None` and `'shared_target_split' is None`.
- `test_profile_pet_roundtrip.py`:
  - The staging and co-rooted-with-config tests fail on their assertions (base refuses with `owner_relocation_unverified:tamagotchi.config` or `:config`, or derives `~/.config/tldw_chatbook`).
  - The integration test's seed child fails `assert store.filepath==selector.parent/'tamagotchi_pets.json'`.
  - The changed `rejects_unrecognized_payload_mapping[*]` fails with `TypeError` (the new `_restore_path` argument).
- `test_detached_layout`:
  - `[profile]` and `[both]` fail `AssertionError: detached_unknown_content`.
  - The portable relocation child fails with `isolated_destination_parent_overlaps_control`.
- `test_inventory_unused_owners[pets_absent|pets_link]` and `test_settings_file_participant_lifetimes[pet]` fail on their path assertions.
- Harness cases, with a temporary `recovery_service.issue_code` wrapper to name the code behind `backup_operation_failed`:
  - `[pets-missing]`, `[pets-present]`, `[pets-dated-only]` and `[pets-ungrouped]` fail with `replacement_destination_parent_required`.
  - `[pets-none]` fails with `preserved_group_inventory_invalid`: base's pet owner refuses the marker as an unknown file in its folder.
- On the new pet code, without the two retired-root fixes, `[pets-none]` failed at the pet start after the restore, first with `FileNotFoundError` in `verified_user_data_directory` and, with that fixed, with `storage_admission_unavailable`. A temporary probe showed the `resolve(strict=True)` frame on the retired `tamagotchi_pets.json` root.

**Invariants, red under temporary source edits** (all reverted; `-n 4`):
- `ui.state` added to `_PER_PROFILE_ALIAS_OWNERS`: `test_only_pet_aliases_…[one|both-ui.state]` (`assert None == 'shared_scope_expansion_required'` and `'shared_target_split'`).
- The marker's logical-ID check dropped: `rejects_…[marker_id]` (`DID NOT RAISE`).
- Pet rows skipped in `_config_targets` (the withdrawn exclusion's shape): the wrong-folder staging cases, and `test_co_rooted_pet_and_data_file_refuse_by_name[data-*]` and `[config-True]`. `[config-False]` stays green under that edit, because the co-rooted personas file refuses by its own check.
- The roaming fallback removed from `_pet_members`: `test_portable_pets_…[roaming]` (`assert {} == {...}`).

**Green (final code).**
- In-process unit and alias runs: 46 passed. Detached pet tests: 5 passed (75 s).
- All ten `test_created_owner_rollback.py` cases are green, the five pet cases among them: see batch 2 below.
- `test_profile_pet_roundtrip.py`, serial with the owner-suites step's flags (`-p no:xdist --timeout=300 --tb=short`): 30 passed in 90.1 s; the integration test took 88.6 s at load 15-18. Charge about 1.5 min to the lane.

**Covering runs** (`-n 4` unless noted).
- Batch 1, at load 14-19: 652 passed, 2 failed, 1 skipped (410 s). It covered `test_operational_owners`, `test_restore_plan`, `test_inventory_unused_owners`, `test_settings_file_participant_lifetimes`, `test_later_rollback`, `test_later_created_destinations`, `test_config_file_retirement`, `test_retirement_only_rollback`, `test_portable_relocation`, `test_config_sibling_capture`, `test_capture_root_order`, `test_bound_config_companions`, six `test_config_companion_*` files, `test_admission_evidence_reuse`, `test_effective_admission_roots`, `test_admission_path_resolution`, `test_guarded_raw_evidence_reuse`, `test_sqlite_inside_config_scope` and `Tests/Widgets/test_tamagotchi*.py`.
  - The two failures, `test_settings_file_participant_lifetimes.py::test_runtime_serialization_is_inside_admitted_source` and `::test_writeback_failure_does_not_wedge_the_operation_file_list`, fail the same way at base with the same command (3 failed at base, the third being this task's `[pet]` red). They are not this change.
- Batch 2 (load 11-16, 852 s): 174 passed and 2 failed, over the whole of `test_created_owner_rollback.py` (10 cases), `test_detached_layout.py`, `test_profile_pet_roundtrip.py`, `test_isolated_restore.py`, `test_capture_service.py`, `test_inert_extraction.py` and `test_restore_data_groups.py`.
  - The two failures were `[databases]` and `[tools]`. This task's new "restored with the source's text" check also ran on their `config.history` files, whose restored text legitimately differs. The check now covers only the pet files.
  - After that: `[databases]`, `[tools]`, `[pets-missing]` and `[pets-present]`, 4 passed (152 s, load 22).
- `Tests/CI`: 1811 passed, 2 skipped.
- Serially: `Tests/Architecture/test_backup_owner_inventory.py`, `test_module_size_ratchet.py`, `test_screen_size_ratchet.py` and `test_recovery_entrypoints.py`. The census test failed until `backlog/docs/backup-recovery-owner-inventory.md` gained the two new producer rows (`ConfigFileStorage._prepare` write, `_read_shared_pets` open) and the updated `tamagotchi.config` row. After that: 75 passed in that file, and the others passed, 340 in the first run minus that one.
- No size-ratchet row lists a touched module.
- `ruff check`: no new findings in any changed file (per-file counts equal at base and now).
- Preflight: the diagnostic inventory drift is the one added `logger.warning("Old shared pet file not copied: not a readable pet file")`, which interpolates nothing. It was reviewed with `--statements` and written with `--write`.

**Lane (D7).** `Tests/Backup_Recovery/test_profile_pet_roundtrip.py` is added to the Backup_Recovery owner-suites step (push to dev) in `.github/workflows/derived-artifacts.yml`, after the created-owner rollback file. It is also in the `BACKUP_OWNER_SUITE_TARGETS` pin in `Tests/CI/test_ci_queue_pressure_contract.py`. The file was renamed from `test_shared_home_pet_roundtrip.py`. Its old base red (`missing_required` for `research.local`, from a custom research path nothing created) is fixed in the seed.

**Not verified here:**
- Windows: the `%APPDATA%` old shared path, and `host_environment` without the AppData creation. The portable relocation test now runs on macOS and passed.
- A real multi-profile older backup end to end: the alias carve-out is pinned at planner level with a crafted manifest.

### Review r17pt-rev1, fix round 1

- **Important 1: a dot-named config folder.**
  - The pet store refused a dot-named config folder: `validate_path` rejects a base directory whose name starts with a dot, so with `TLDW_CONFIG_PATH=~/.tldw/config.toml` the footer fell back to an in-memory pet on every start.
  - The check now passes `allow_hidden=True`. It is there only to refuse a link out of the folder, and `raw.file` already opens with `O_NOFOLLOW` under a pinned parent.
  - Test: `test_operational_owners.py::test_pet_store_persists_in_a_dot_named_config_folder`, in a `.tldw` folder, across two stores. Red with the fix reverted: `[ValueError('Access to hidden files/directories is not allowed'), …]`.
- **Important 2: direct tests for the retired-root guard.** Three tests in `Tests/Backup_Recovery/test_config_participant_lifetimes.py`, which runs on the PR lane "recovered suites" (32 passed, 12.6 s under that step's flags `--storage-lease-guard --timeout=180 --tb=short`, run with `-p no:xdist`, load 12-16):
  - `test_an_absent_registered_root_grants_nothing_beneath_it` checks two things for a bound profile whose enrolled directory is then removed: `_contains_capture_path((absent,), absent/'x') is False`, and `acquire_storage(absent/'x')` refuses (`RecoveryRequired('storage_admission_unavailable')`, from `bootstrap._binding`'s strict resolve).
  - `test_a_retired_settings_root_does_not_break_the_data_directory_check`: a retired `note_templates.json` root enrolled before the data directory's root, and `verified_user_data_directory` still returns the data directory.
  - `test_a_data_directory_check_with_every_root_absent_refuses`: the check raises `config_directory_selection_changed`.
  - Red with `storage_admission.py` and `config_participants.py` taken from c176f553b3: `assert FileNotFoundError(...) is False`, `assert FileNotFoundError(...) == PosixPath(...)`, and `AssertionError: FileNotFoundError(...)`.
- **Minor 3: a FIFO at the old shared path.** The copy's open adds `O_NONBLOCK`, so a FIFO counts as nothing to copy. Test: `test_pet_store_copy_treats_a_fifo_as_nothing_to_copy` (skipped without `os.mkfifo`). Red with the flag removed: `Failed: Timeout (>60.0s)` at the open.
- **Minor 4: a missing config folder.** `_ConfiguredPets.discover` on a missing folder returned a blocking `unavailable` row. The dead `except FileNotFoundError` could not catch it, because `pinned_directory` raises `PrivatePathError('missing_parent')`. It now checks `os.path.lexists(parent)` first and returns the same `unused` current-file row, with its config dependency, that an absent pet file beside an existing config gets. Test: `test_pet_owner_reports_a_missing_config_folder_as_no_pets`. Red at 124c2d9bc7: `'unavailable' != 'unused'`.
- **Minor 5: comments.** The comments in `storage_admission._contains_owned_path` and `config_participants.verified_user_data_directory` now say the guard covers every absent root. The `effective_roots.py` comment now names `bootstrap._binding` as the strict resolver that still refuses an unproved root.
- **Minor 6: D5 narrowed (coordinator's call).**
  - `_PER_PROFILE_ALIAS_OWNERS` now exempts pet aliases only from `shared_target_split`, so every alias selected may restore into differing folders.
  - A partial selection refuses `shared_scope_expansion_required` again, as for every owner. A Replace of one profile therefore cannot overwrite a pet file that a co-foldered profile reads.
  - Aliases sent into one shared folder refuse as `ui.state`'s do (`destination_collision` in the crafted manifest).
  - Consequence: an older multi-profile backup restores its pets when all its profiles are restored together. `Docs/Backup-and-Recovery.md` says so.
  - Test: `test_restore_plan.py::test_only_pet_aliases_restore_into_each_profiles_own_folder[differ|shared|one-selected × tamagotchi.config|ui.state]`. Red at 124c2d9bc7: `[one-selected-tamagotchi.config]` with `assert None == 'shared_scope_expansion_required'`.
- **Minor 7: User Guide.** `settings.md` now says the marker is written on the first run whether or not anything was copied.
- **Runs (fix round 1, load 11-17):**
  - Changed fast files with `-n 4` (`test_operational_owners`, `test_restore_plan`, `test_config_participant_lifetimes`, `test_inventory_unused_owners`, `test_settings_file_participant_lifetimes`, `test_detached_layout`, `test_config_sibling_capture`, `test_effective_admission_roots`, `test_admission_path_resolution`, `Tests/Widgets/test_tamagotchi*.py`): 497 passed and 2 failed. The two are the `test_settings_file_participant_lifetimes` reds already present at base.
  - The five pet cases of `test_created_owner_rollback.py` and `test_profile_pet_roundtrip.py`, with the owner-suites flags (`-n 4 --timeout=300 --tb=short`): 35 passed (219 s; cases 73-143 s).
  - `Tests/CI`: 1811 passed, 2 skipped.
  - `Tests/Architecture/test_backup_owner_inventory.py` (serial): 75 passed.
  - Preflight passed.
