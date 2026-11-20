---
id: TASK-42114
title: "A backup that captured a dated pet copy stops the app starting once the pet store rotates it"
status: Done
assignee: ['@claude']
created_date: 2026-11-20 17:50
updated_date: 2026-11-20 21:37
labels:
- follow-up
- backup
- pets
priority: high
references:
- tldw_chatbook/Backup_Recovery/capture_service.py
- tldw_chatbook/Widgets/Tamagotchi/recovery.py
- tldw_chatbook/Backup_Recovery/restore_plan.py
- tldw_chatbook/Backup_Recovery/detached_layout.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by the review of TASK-42022 (round 18). This startup blocker is already on dev.

A plain backup's `_capture_names` (`capture_service.py:243`) registers each captured config-folder file as its own `backup.source.<sha>` root, dated pet copies included. The pet store's rotation (`_cleanup_old_backups`, `max_backups=3`) later deletes that copy. From then on, every admission raises `FileNotFoundError` in `Admission._groups` → `_tokens` → `root.resolve(strict=True)`. `config.py` runs `admit_startup()` at import, so the app refuses to start.

The reviewer reproduced it at the default config location and on a separate config folder. After the backup, startup admission was ok. Three pet saves later, the fourth save raised `FileNotFoundError`, and the next start gave `RecoveryRequired: storage_admission_unavailable` (SystemExit).

TASK-42022's note that `_groups`/`_tokens` needs no change holds only for Replace.

Design constraint (coordinator): keep the refusal for an absent folder root, which may be an unmounted drive. Any loosening is limited to a file root whose parent folder is present.

**2026-11-20, coordinator decision after review rev2:** deleted-file-root tolerance needs one design across all root consumers, the recovery journal included. That work is parked on `fu/r18-deleted-roots-wip` and moves to TASK-42115. AC#3's admission half moves with it. This task is met by the trigger fix (dated copies are never captured or registered) plus the plan-time skip of older archives' dated rows.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A test reproduces the failure (backup, then pet rotation, then the next start), red before the fix, at both the default location and a separate config folder
- [x] #2 After the fix, the app starts and pets save after that sequence
- [x] #3 The root cause is named, and the fix covers the trigger: the pet store's own rotating dated copies are neither captured nor registered as roots (admission's handling of a deleted file root moved to TASK-42115, coordinator decision 2026-11-20)
- [x] #4 The notes say, with evidence, whether other config-folder files that the app deletes on its own reach the same state
- [x] #5 The test runs on a PR or dev-push lane
- [x] #6 A restore from an archive made before the fix (holding dated rows) never recreates dated copies, keeps a target's own same-named copy, still restores the current pet file, and its later rollback completes; portable import matches
<!-- AC:END -->

## Implementation Plan

1. Reproduce with the reviewer's probe at both locations, and list the file roots a plain backup registers to scope AC#4.
2. Exclude the pet store's dated copies from capture (`intentionally_excluded`), with the rotation test at both locations, red at base.
3. After review: skip older archives' dated rows at plan time, make portable import match, and decide portable import's "own pets" on the pet file or marker only.

## Implementation Notes

**Root cause.**
- `capture_service._capture_names` registers every captured file that no registered root covers as an admission root of its own (`backup.source.<sha>`). On a plain backup this covers each config-folder file at any location, since the config folder itself is never captured as a folder, and the data folder's single files too.
- The pet store deletes its own dated copies as it rotates (`_cleanup_old_backups`, three kept). The absent root then makes `Admission._groups` → `_tokens` → `resolve(strict=True)` raise `FileNotFoundError` on every admission, so `admit_startup()` refuses the app.

**Trigger fix.** `Widgets/Tamagotchi/recovery.py` lists the dated copies as `intentionally_excluded`. They are the store's own rotating backups, which the app never reads (TASK-42000 D3 already skipped them on upgrade), so a backup neither captures them nor registers them as roots, and Replace preserves the profile's own copies.
- Registering them would also have added about three registry entries (each with three lock files) per backup, because the names change on every save.
- Behaviour change for TASK-42000: a backup holding only a dated copy now holds no pets. `[pets-dated-only]` and `[pets-missing]` gained an `uncaptured` key.

**Older archives (review).** Every pet archive since TASK-32628 holds the store's dated copies.
- Restoring one created the copy, and the later rollback then refused it (`local_snapshot_created_members_changed`). A target holding a same-named copy refused with `preserved_target_collision`.
- `restore_plan.plan_restore` now skips an owner's rotating copies, through the pet adapter's `rotating_copy`. A restore never recreates them, keeps a target's own copy, and still restores the current pet file, and the later rollback has nothing of them to retire.
- Portable import (`detached_layout`) lists dated copies as excluded too. It now decides whether the profile has pets of its own on the pet file or the marker only (review rev2 Minor 2); a dated copy alone stays behind.

**Tests.**
- `test_created_owner_rollback.py::test_a_backup_then_pet_rotation_keeps_the_app_starting[separate|default]`. The scenario gained a `LOCATION` input; the default location uses `~/.config/tldw_cli/config.toml` under the scratch HOME and `RecoveryService(default_control_root())`.
- `[pets-legacy-dated]` and `[pets-legacy-dated-collision]` build an archive the old way, by stubbing discovery in the source process to include dated copies. They run the restore, the later rollback and a fresh start (a new `admit` producer role).
- `test_detached_layout.py`: the portable pet cases, plus `[profile-dated-only]`.
- Updated: `test_operational_owners.py` (dated copies discovered as excluded) and `test_profile_pet_roundtrip.py`.

**Red/green** (rebuilt branch, base d669810711).
- Rotation test with the discovery change reverted: both locations failed with `FileNotFoundError` on the rotated copy. With it: passed, 52 passed across the pet, roundtrip and operational files.
- Old-archive and portable cases with the plan-time and portable changes reverted: 6 failed ("restore created pets.dated", `preserved_target_collision`, and four portable cases). With them: 14 passed.
- `[profile-dated-only]` failed with `assert {} == {...}` before the portable decision change. With it, the detached file passed: 47 passed.

**AC#4.**
- A probe listed the regular-file roots registered after a plain backup of the harness cases at both locations:
  - config folder: `config.toml`, `config.toml.bak`, `config_backup_<stamp>.toml`, `rag_pipelines.toml`, `note_templates.json`, the pet files;
  - data folder: the databases, the MCP files, the chatbooks registry.
- Searching the app's deletions found only one routine deletion among config-folder files: the pet rotation.
  - `config.py` removes `config.toml` only when the same failed call created it.
  - Nothing prunes config snapshots or `.bak`.
  - The MCP log rotation rewrites both generations atomically, so neither is ever absent.
- A registered file the user deletes still stops every admission; tolerating that is TASK-42115's.
- Possible workaround for an install already stopped (not tested): recreate an empty private file (mode 0600) at the missing path.

**Lane.** `test_created_owner_rollback.py` and `test_detached_layout.py` are in the Backup_Recovery owner-suites dev-push lane. Serial times with that lane's flags:
- rotation test: about 19 s per location;
- legacy cases: about 76 s and 79 s (load 4-7).

**Files.** `tldw_chatbook/Widgets/Tamagotchi/recovery.py`, `tldw_chatbook/Backup_Recovery/restore_plan.py`, `tldw_chatbook/Backup_Recovery/detached_layout.py`; the tests above; `Docs/User_Guide/settings.md`, `Docs/Backup-and-Recovery.md`, `Docs/Architecture/backup-recovery.md`, `backlog/docs/backup-recovery-owner-inventory.md`.
