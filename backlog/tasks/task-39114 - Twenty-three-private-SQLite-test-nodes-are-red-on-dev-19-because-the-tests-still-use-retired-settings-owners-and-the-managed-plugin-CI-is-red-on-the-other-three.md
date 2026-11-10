---
id: TASK-39114
title: "Twenty-three private-SQLite test nodes are red on dev, 19 because the tests still use retired settings owners, and the managed-plugin CI is red on the other three"
status: Done
assignee:
- '@claude'
updated_date: '2026-11-10 20:35'
created_date: 2026-11-07 02:03
labels:
- follow-up
- tests
- sqlite
- backup
- ci
priority: low
references:
- Tests/DB/test_private_sqlite.py
- Tests/DB/test_private_sqlite_inventory.py
- tldw_chatbook/Backup_Recovery/detached_layout.py
- tldw_chatbook/DB/private_sqlite.py
- .github/workflows/plugin-qualification.yml
dependencies:
- TASK-39053
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in round 11 of the 2026-10-23 burn-down; checked on dev 2684e5453f, 2026-11-07. TASK-39053's notes (round-11 branch fu/r11-backupinv) list them as pre-existing. Line numbers are dev's.

**The run.** `Tests/DB/test_private_sqlite.py` and `Tests/DB/test_private_sqlite_inventory.py` at `-n 4` on dev: 23 failed, 385 passed, 2 skipped (61.6 s, load 34-47). A rerun of the 23 nodes gave the same 23. None is the RecoveryRequired trap.

| Cause | Nodes | What fails |
|---|---|---|
| Retired settings owners: `settings.integrity`, `settings.bulk_backup` and `settings.vacuum` left the registry in bd1a4258ea (TASK-33244). `test_private_sqlite.py` still uses them as fixture owners (`:76` `COPY_BACKUP_OWNER_IDS`, and from `:1392` on). | 19 | "Unknown SQLite owner" or a KeyError. Includes `test_retired_settings_owner_policies_are_absent`, the `read_only_*` and `copy_private_sqlite_*` nodes, and `copy_backup_owners…[settings.bulk_backup]`. |
| Descriptor-close count drift, not yet explained | 1 | `test_migration_immutable_failure_retains_complete_destination[True]`: `assert 4 == 6` at `:938`. |
| A raw connection added by e350634a8a (portable Windows bundle) is not in the private-SQLite inventory: `Backup_Recovery/detached_layout._normalize_sqlite` | 3 | `test_raw_connection_census_is_qualified_and_transition_aware`, `test_private_sqlite_seam_calls_use_literal_module_owned_ids`, `test_backup_inventory_matches_current_sqlite_and_settings_operations`, all in the inventory file. |

**Lanes.**
- No required lane runs either whole file.
- `plugin-qualification.yml:114` runs the inventory file on PRs that touch Plugins paths. Its latest run (42321038264) is red on ubuntu, macOS and Windows on exactly the three inventory nodes.
- `derived-artifacts.yml:981` runs one inventory node, which is green.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 All 23 nodes pass at `-n 4`, each shown red first with the same command
- [x] #2 No node is deleted or weakened to go green: every retired-owner node still exercises the same seam with a current registry owner
- [x] #3 The descriptor-close count is explained (which closes went missing, and why), and the expectation follows production; if it is a production regression, that is fixed instead
- [x] #4 `_normalize_sqlite`'s connection is qualified in the inventory, and the plugin-qualification job's three nodes pass
- [x] #5 Both files run on a required PR lane (never the fast PR contract), with three timings of that step's command recorded
<!-- AC:END -->

## Implementation Plan

1. Re-run both files at `-n 4` on base ae9f9e8d21 and record the 23 reds.
2. Retired owners: re-point each `settings.*` fixture owner at a current registry owner with the same target kinds and backup flags (read-only seam -> a current READ_ONLY_URI owner; copy-backup list -> the current backup-enabled owners the list misses), and re-pin `test_retired_settings_owner_policies_are_absent` to bd1a4258ea's retirement.
3. Descriptor-close count: read which closes the test counts, diff against the commit that changed them, and fix production if it is a regression, else re-pin with the commit named.
4. Qualify `_normalize_sqlite`'s raw connection in the private-SQLite inventory (census row, seam owner) so the three inventory nodes pass.
5. Lane both files on a required PR lane (not pr-fast-lane), update the CI contract pin, and time the step three times.

## Implementation Notes

**Base.** ae9f9e8d21. Both files at `-n 4` through run.sh: 23 failed, 383 passed, 2 skipped (49 s, load ~23). These are the same 23 nodes as the description, and none is `RecoveryRequired`. After the change, the same command gives 408 passed, 2 skipped (204 s, load ~70). The two extra nodes are the Dreams and Guardian copy cases and the new owner's target-kind case.

**Retired owners (19 nodes, test drift; AC#1, AC#2).** bd1a4258ea (TASK-33244) removed `settings.integrity`, `settings.bulk_backup` and `settings.vacuum` together with the legacy settings window, which was their only caller. Every node still exercises its seam, now with a current owner of the same shape:
- `settings.integrity` (read_only_uri, no flags) became `db.media.integrity`, which has the identical policy shape. This covers the read-only, sticky-parent, special-filename and kind-mismatch nodes.
- `settings.bulk_backup` (private_file + read_only_uri, backup allowed, no preserve or capture flags) became `tts.profile_migration_backup`, which has the identical flags, in the `copy_private_sqlite_*` nodes.
- In `COPY_BACKUP_OWNER_IDS`, its slot now holds the two backup-enabled owners the list had missed, `recovery.operations.dreams` and `recovery.operations.guardian`. That is why `test_every_backup_enabled_owner_has_a_behavioral_operation` was red twice over.
- `test_retired_settings_owner_policies_are_absent` now pins all seven retired Settings owners as absent, and no `settings.*` owner at all.

**Descriptor-close count (1 node, test drift; AC#3).**
- **The commit.** A first-parent bisect over b5251e9a6e..ae9f9e8d21 (905 commits, each a `git archive` tree run through run.sh) put the first red at 5a915a43e5 (TASK-33370/33373). That commit is test-only: it moved `Tests/DB` from the per-test profile to the bootstrap profile.
- **Which closes went missing.** A temporary stack-recording spy (reverted) showed the same 4 production closes under both profiles: the validation owner's file and parent pins (`profile_errors.close`), and the destination's two pins (`close_profile_migration_destination`). The 2 extra closes under the per-test profile came from the `chatbook-storage-admission` hold thread, `admission._lock` (registry lock) and `native_files.pinned_directory`. That thread ended its hold inside the window the process-wide `os.close` spy recorded.
- **Why.** The "Admission also owns two directory pins for a file-backed view" comment described another thread, not the cleanup, so this is not a production regression.
- **The change.** The spy now counts only closes made on the cleanup's own thread, and the node expects 4 for the boundary path and 2 otherwise, each closed once.
  - Green under the bootstrap profile, and 3 of 3 under a temporarily restored per-test profile (conftest edit, reverted).
  - Control: dropping the retained `parent_fd` close in `_ProfileMigrationValidationOwner.close` (reverted) fails the node with `assert 3 == 4`.

**`_normalize_sqlite` (3 inventory nodes, a real seam gap; AC#4).** e350634a8a (portable Windows bundle) opened the normalized destination with a raw `sqlite3.connect`, outside the private seam. Changes:
- It now opens through `connect_private_sqlite("recovery.detached_layout", destination, must_exist=True)`.
- The new owner is private_file only, with no backup or capture authority. It is registered in `DB/private_sqlite.py` and documented as C99 in `backlog/docs/sqlite-private-owner-inventory.md`; the id list now runs to C99.
- Its `open_recovery_validation(installed.owner_id, ...)` call joins the qualified-dispatch list, as the `sqlite_validation` and `credentials` precedents did.
- Its one direct `connection.backup(output, ...)` (restricted candidate into the seam-opened destination, under the cancel and deadline guard) is named in the direct-backup census, and the census mutation test's baseline carries it.

Results:
- `Tests/Backup_Recovery/test_detached_layout.py` and `test_portable_relocation.py`: 44 passed, 3 skipped.
- The three nodes the plugin-qualification job runs pass with `-p Tests.Plugins.platform_support` (3 passed, 327 s, load ~70).
- `.github/workflows/plugin-qualification.yml` needed no edit.

**Lane (AC#5).** Both files are appended to `pr-fast-lane-arch` ("PR Fast Lane (architecture-review tests)", its own required job, not the fast PR contract), last because `Tests/DB` keeps the bootstrap profile. `ARCH_REVIEW_TARGETS` is updated to match.
- recovered-suites and architecture-guards both forbid a file that another step lists, and console-p0-gate runs the inventory's `test_response_rule_import_dispatch_is_exact_and_validated` node.
- Admission lane 3 would also fit, but one local run of its whole step took over 30 min at load 60-80 before it was stopped.
- Step timings (that step's whole pytest command, serial, through run.sh): 484 s, 338 s and 312 s wall (pytest 477/333/307 s), each 541 passed, 2 skipped, 1 xfailed. The 1-minute load fell from 52 to 32 across the runs. The job ran 2.1-2.9 min in CI before (four PR runs on 2026-11-10) against its 20-min cap.

**Files.** `Tests/DB/test_private_sqlite.py`, `Tests/DB/test_private_sqlite_inventory.py`, `tldw_chatbook/DB/private_sqlite.py`, `tldw_chatbook/Backup_Recovery/detached_layout.py`, `backlog/docs/sqlite-private-owner-inventory.md`, `.github/workflows/derived-artifacts.yml`, `Tests/CI/test_ci_queue_pressure_contract.py`, and a lessons entry in `backlog/docs/lessons-testing-evidence.md` (process-wide `os.close` spy).

**Review r13c-rev1.**
- **C1, missing census row.** `backlog/docs/backup-recovery-owner-inventory.md` lacked the new owner, which turned two nodes in the required architecture-guards job red: `test_sqlite_registry_members_have_explicit_census_rows` ("SQLite registry members without a row: ['recovery.detached_layout']") and `test_every_persistence_source_symbol_has_reviewed_owner_row`.
  - Fix: a `sqlite:recovery.detached_layout` row now sits beside `recovery.credentials`, and the `_normalize_sqlite` row's call column reads `connect_private_sqlite`.
  - Before the fix both nodes failed with those messages; after it the file gives 19 passed, serial.
- **Storage admission (coordinator ruling (a)).** `recovery.detached_layout` still takes storage admission: it is not on `_with_storage_admission`'s bypass list. `recovery.credentials` bypasses admission; adding the new owner to that list was considered and declined as the less conservative choice. The Windows path, which uses `_prepare_artifact` instead of the POSIX helper, is not exercised here.
