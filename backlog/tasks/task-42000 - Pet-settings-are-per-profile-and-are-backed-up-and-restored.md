---
id: TASK-42000
title: "Pet settings are per-profile and are backed up and restored"
status: In Progress
assignee: ['@claude']
created_date: 2026-11-19 15:58
updated_date: 2026-11-19 16:13
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
- [ ] #1 Each profile keeps its own pet settings: a change made under one profile is not seen by another profile on the same device
- [ ] #2 A backup captures the profile's pet settings, and a Replace restore puts them back into that profile; pet settings never make a restore fail
- [ ] #3 Isolated restore, Manual extraction and a later rollback treat the pet files like the profile's other settings files
- [ ] #4 On upgrade, a profile with no pet settings of its own starts from a copy of the old shared files; the shared files are never changed or deleted
- [ ] #5 TASK-41719's restore cases (pet folder missing, present, backup with only a dated copy) succeed with the pets restored, each red before the change
- [ ] #6 The User Guide and Docs/Backup-and-Recovery.md describe per-profile pets
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
