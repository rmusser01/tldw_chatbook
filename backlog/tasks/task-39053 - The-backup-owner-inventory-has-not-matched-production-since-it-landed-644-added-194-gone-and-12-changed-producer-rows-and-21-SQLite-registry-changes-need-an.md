---
id: TASK-39053
title: "The backup owner inventory has not matched production since it landed: 644 added, 194 gone and 12 changed producer rows and 21 SQLite registry changes need an owner and cohort review"
status: Done
assignee:
- '@claude'
created_date: 2026-11-06 10:09
updated_date: 2026-11-06 18:02
labels:
- follow-up
- backup
- recovery
- architecture
- tests
- needs-owner-decision
priority: high
references:
- backlog/docs/backup-recovery-owner-inventory.md
- backlog/docs/sqlite-private-owner-inventory.md
- Tests/Architecture/test_backup_owner_inventory.py
- tldw_chatbook/DB/private_sqlite.py
- Tests/DB/test_store_kernel.py
- .github/workflows/derived-artifacts.yml
dependencies:
- TASK-38989
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in round 10 of the 2026-10-23 burn-down; checked on dev b2c618185f, 2026-11-06, and on Architecture PR #3436 (head b7c8e809b3), which lands TASK-38989. Line numbers are dev's and are the same on that head. TASK-38989 carved these two nodes out of its scope and left them red for an owner review. #3436 adds an `architecture-guards` PR job and keeps this file out of it.

**The drift.** It was recomputed with the test's own `census()` / `production_census()`. The result is the same on dev, on every round-10 branch and on e8eeae7db9:
- The doc has 1059 producer rows and production has 1509: **644 added, 194 gone, 12 changed**.
- The SQLite registry (`_SQLITE_OWNER_POLICIES`, `tldw_chatbook/DB/private_sqlite.py`) has **13 new members and 8 gone**.
- `Tests/Architecture/test_backup_owner_inventory.py` (serial, 23.6 s on dev): 2 failed, 9 passed. The two are `test_every_persistence_source_symbol_has_reviewed_owner_row` (`:146`) and `test_sqlite_registry_members_have_explicit_census_rows` (`:184`). No PR lane runs the file.
- It has been red since it landed:
  - at b5251e9a6e (2026-09-16, the squash that created it, TASK-32628): 407 / 86 / 9, SQLite 8 / 5;
  - at the doc's last edit, 81184d04f5 (2026-10-11): 536 / 170 / 10, SQLite 9 / 8.
  - Done tasks that recorded "11 passed" ran it on branch snapshots.

**What needs a decision.**
1. **13 new SQLite members:** chat.launch_wake, chat.trace_maintenance, db.chunking_lab, db.workspace_project_index, library.legacy_recovery, personal_context.interview_drafts, personal_context.repository, plugins.registry, recovery.operations.dreams, recovery.operations.guardian, tests.store_kernel, utils.db_upgrade_notice, workflows.local. dreams and guardian (`private_sqlite.py:216-227`, f927f98add) are `centralized_backup_allowed` and `recovery_capture_allowed`.
2. **A test owner in the production registry.** `"tests.store_kernel"` (`private_sqlite.py:420-424`, owner `Tests/DB/test_store_kernel`) was added by cbb39dd767 (TASK-33132). The SQLite private-owner census already accepts it (`backlog/docs/sqlite-private-owner-inventory.md:377`, row C97).
3. **8 rows that exist only in the doc.**
   - `sqlite:recovery.operations.receipts` (doc `:1349`) was never in the registry (`git log -S` finds no registry commit) and was stale when the test landed.
   - Seven `sqlite:settings.*` rows (doc `:178-184`) lost their registry entries in bd1a4258ea (TASK-33244, which removed the Tools settings window). That commit updated the other inventory, not this one.
4. **644 added producer rows.**
   - 535 are in 194 modules the doc never lists. The largest packages: Plugins 63 (`Plugins/authority_store.py` alone 25), Tools 43, Agents 40, Audio 40, Chat 35, Actor_Packs 33, Backup_Recovery 31.
   - 109 are in modules the doc already lists.
5. **194 gone rows.** 58 are in 15 deleted files (for example `UI/Tools_Settings_Window.py`, 25), 44 look like renames within a module, and the rest are removed or moved code.
6. **12 changed counts**, for example `ChatbookCreator` x5, `_connect_registered_sqlite` connect 1 -> 2, `BaseDB._get_connection` 1 -> 2, `KeyringServerCredentialStore._save_index` set_password 1 -> 3.

Classifying rows into owners and backup cohorts is a product decision about what a backup captures. The criteria leave the classifications to the owner.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Each of the 13 new SQLite members has a reviewed row with a classification and a cohort; where a member's `SQLiteOwnerPolicy` flags do not settle its cohort, the owner decides it and the decision is recorded
- [x] #2 The owner decides whether a test-owned policy (`tests.store_kernel`) may stay in the production registry; the decision is recorded and the registry and census follow it
- [x] #3 The 8 rows that exist only in the doc are removed, each with its cause recorded
- [x] #4 Every added and changed producer row has an owner-reviewed classification (the owner signs off the classifications, or the cohorts they fall in, in this task's notes), and every gone row is removed
- [x] #5 Both nodes pass; they are red before (2 of 2 on dev b2c618185f: 644 / 194 / 12, SQLite 13 / 8)
- [x] #6 Negative control: a synthetic unreviewed producer turns the test red, and the failure message lists every added, gone and changed row
- [x] #7 The file runs on a PR lane (the Architecture guards job, or another the owner picks; never the fast PR contract), with three timings of that lane's command recorded
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Recompute the drift at dev 2684e5453f with the test's own `census()` / `production_census()`; check whether any gone row is a scanner blind spot rather than removed code, and extend the scanner (test only) if so.
2. Classify every added and changed producer row and every new SQLite member by the doc's existing rules and each owner's `SQLiteOwnerPolicy` flags; remove gone rows, recording the cause of the 8 doc-only SQLite rows.
3. AC#2: implement the recommended option for `tests.store_kernel` as a small, reversible change; record the alternative.
4. Make the failure message list every added, gone and changed row (AC#6); show it with a synthetic unreviewed producer.
5. Lane the file on the `architecture-guards` job, pin it in `ARCHITECTURE_GUARD_TARGETS`, time the step three times.
6. Write the cohort sign-off table in Implementation Notes; leave AC#1, #2 and #4 unticked and the task In Progress pending owner sign-off.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
**Status: signed off.** The owner signed off the cohort table on 2026-11-06, answering in the coordinator's session; the answers are recorded next to the table.

Owner decision (2026-11-06): "Agent proposes, you sign off: a stream classifies every row by the existing doc's rules and each module's policy flags, groups them by cohort and flags only the ambiguous ones; you sign off the cohort table in the PR, and nothing merges until you do."

**What changed.** Branch `fu/r11-backupinv`, base dev 2684e5453f.
- The census had blind spots that hid live producers, so some rows looked gone. All are fixed in the test only, each with a unit test that was red first:
  - `from <module> import os` never bound `os`. 74 files import `os` from `Utils/platform_files` (73 with the absolute import, 1 with `from .platform_files`; 68 at module scope), and 24 "gone" rows were live `os.replace` / `os.unlink` calls.
  - A tuple assignment, `fn, flags = libc.renameatx_np, 0x4`, hid the native no-replace publication in `Backup_Recovery/native_platform.rename_noreplace`. Assigned aliases are now lexically scoped (review r11b-rev1): the module-wide binding had also counted `fn = libc.fstatfs; fn(...)` in `linux_identity` as `renameatx_np`.
  - The `Utils/atomic_file_ops` writers (`atomic_write_text`, `_bytes`, `_json`, `atomic_copy`) were not in the vocabulary. 19 modules call them: 4 had rows that looked gone (`trajectory_export`, `model_discovery_disk_cache`, `config_profiles`, `file_operation_tools`), and the other 15 were never counted.
  - (review r11b-rev1) A producer passed by reference was never counted. This covers `asyncio.to_thread(path.write_text, ...)`, `drain_thread(archive.write_bytes)` and `run_in_executor(None, atomic_write_text, ...)`. The four Personas export rows (`_dictionary_export_worker`, `_export_expression_set`: mkdir, write_text, write_bytes) are restored. `x.open` / `x.connect` values passed as arguments are not counted: here they are widget state (`header.open`, `event.open`) or `webbrowser.open`.
  - (review r11b-rev1) `yaml.safe_dump` / `safe_dump_all` / `dump_all` were not in the vocabulary. `EvalConfigLoader.save` writes the qualified `eval.definitions` source with `safe_dump` into a `raw.file(...)` stream, so its row carries onto `safe_dump`. `yaml.dump`, `json.dump`, `pickle.dump` and `toml.dump` were already counted as `dump`.
  - (review r11b-rev1) `os.rename` and `os.link`, which publish like `os.replace`: `snapshot_store._move_verified`, `ModelArtifactService._promote`, the backup `crypto.transform` and others.
  - (review r11b-rev2) Arguments to `inspect.signature`, `isinstance`, `issubclass` and `callable` are not counted, because they run nothing. This removed two false rows, `inspect.signature(os.replace)` in `ConsoleVideoController._require_external_video_pinned_capabilities` and `inspect.signature(os.link)` in `Tool_Packs/publication._link_supports_directory_descriptors`.
- With these forms counted, the drift at dev 2684e5453f is 714 added, 166 gone and 13 changed producer rows, plus 13 new and 8 gone SQLite members. The original scanner gave 644 / 194 / 12.
- Known limits, recorded in the doc: these are outside the vocabulary.
  - `tempfile` creation: 86 rows if counted, 22 of them in symbols that have no other row. Follow-up.
  - Third-party writers such as pydub `AudioSegment.export`, subprocess outputs, and dynamic bindings such as `local_tool_impls`' `file_os`.
- `backlog/docs/backup-recovery-owner-inventory.md`:
  - Moved or renamed producers replace their old rows in place.
  - The 13 changed counts are updated.
  - The 166 gone rows are removed: 61 moved or renamed (62 rows replace them), 53 in 13 deleted files, 17 in symbols that no longer exist, and 35 in symbols that still exist. A table in the doc says where each of those 35 symbols' producer went.
  - Only one of the 35 still writes outside the vocabulary: `AudioService.create_m4b_with_chapters`, which writes a temporary `.m4a` through tempfile and pydub, and the M4B through an ffmpeg subprocess. The doc says so instead of calling it removed.
  - A new section, "Census reconciliation (TASK-39053)", holds the rules, the 12 new SQLite rows and the 652 new producer rows, grouped by cohort.
  - The 8 doc-only SQLite rows are removed, each with its cause. `recovery.operations.receipts` was never a registry member: it was written for the doc in b5251e9a6e, and `git log -S` finds no registry commit. Four `settings.*` owners (`pre_restore_backup`, `restore`, `schema`, `single_backup`) left in c9f243f4a2 on 2026-08-29. The other three (`bulk_backup`, `integrity`, `vacuum`) left in bd1a4258ea (TASK-33244) on 2026-10-10. The task description's "all seven in bd1a4258ea" is corrected here.
- AC#2, the recommended option: `tests.store_kernel` is removed from `_SQLITE_OWNER_POLICIES`. `Tests/DB/test_store_kernel.py` now registers the same policy with `monkeypatch.setitem` for its own tests. The SQLite census retires C97, in the `.md` row and in the id-sequence test. All of this is one commit (faeffb9c0e); reverting it restores the old arrangement.
- AC#6: both nodes now raise with the full listing. Before, a bare `assert ==` let pytest truncate the diff ("851 lines hidden").
- AC#7: the file is laned on `architecture-guards` ("Run Architecture guard contracts"), appended after the last target, and pinned in `ARCHITECTURE_GUARD_TARGETS`.
- No production behaviour changed except the AC#2 registry entry. What a backup captures is unchanged.

**Cohort sign-off table** (714 added producer rows, 12 new SQLite members, and AC#2):

| # | Owner / cohort | Rule | Rows | Examples | Owner |
| --- | --- | --- | --- | --- | --- |
| 1 | Moved or renamed producers (any cohort) | Same producer under a new symbol or module: it keeps its reviewed row, in place | 62 (replace 61) | `RunLogWriter._bind_under_scope` (qualified, agents.history); `NotesDeviceStateStore._connect` (qualified, notes.sync_state); `EvalConfigLoader.save` safe_dump (generic_boundary, eval.definitions) | signed off 2026-11-06 |
| 2 | Changed counts | Classification kept. 11 counts went up with calls of the same mechanism. 2 went down because calls moved out: `ChatbookCreator._collect_notes` write 8→3 and `PinnedSyncRoot._create_temporary` os.unlink 2→1 | 13 | `ChatbookCreator._create_zip_archive` open 1→3; `BaseDB._get_connection` 1→2; `KeyringServerCredentialStore._save_index` set_password 1→3 | signed off 2026-11-06 |
| 3 | Same mechanism in a reviewed module or symbol | A new call in a reviewed symbol, or a new symbol that does what the module's reviewed rows do (read in source). File tools follow their reviewed split. Symbols that write into the user-selected root (write, mkdir, write-target opens, like `WriteFileTool.execute`) are `external_input`. Read, lock and verify helpers stay `unsupported`, like the old `read_file` / `edit_file` opens | 64 | `ConsoleChatStore._persist_generation_variant` (memory); `ChatbookImporter._preflight_archive` (generic_boundary); `local_tool_impls._mkdir_relative_directory` (external_input); `models_dev_catalog._write_cache_file` (disposable, diagnostics); `briefing_audio.generate_script_audio` (qualified, subscriptions.assets: it writes the WAV into the qualified `briefing_audio_dir`) | signed off 2026-11-06 |
| 4 | generic | Private path, SQLite or file utility boundary | 22 | `atomic_file_ops._publish_no_clobber`; `SQLiteStoreKernel._open_connection`; `private_sqlite_helper.run` | signed off 2026-11-06 |
| 5 | memory, new modules | `value.to_json()` into a column or payload; `yaml.safe_dump(value)` with no stream; `Image.open(BytesIO(...))`; exact `sqlite3.connect(":memory:")` | 42 | `ChatPersistenceService.commit_completed_voice_pair`; `note_frontmatter._dump`; `portrait_performance.normalize_performance_frame` | signed off 2026-11-06 |
| 6 | process-transport, new modules | Writes to a child's stdin or the module's own stdio, a pipe, a pty or a socket | 40 | `diarizer_worker.serve`; `PosixTerminalBackend.write`; `age_worker._encrypt` (pipe-only worker); `Petdex/network.fetch_bytes` (connect) | signed off 2026-11-06 |
| 7 | Registered SQLite owners, unsupported | A row in an owner's production module takes the owner id as cohort; with no capture flag it stays `unsupported` | 26 | `PluginRegistry.__init__` (plugins.registry); `PersonalContextRepository._connect`; `LegacyCollectionsRecovery._publish_export` (os.link) | signed off 2026-11-06 |
| 8 | Dreams and Guardian, qualified | `centralized_backup_allowed` and `recovery_capture_allowed`, with installed adapters and owner tests (f927f98add, c49d71439e), the same as the other `recovery.operations.*` owners | 6, plus 2 SQLite rows | `DreamsDB` inherits:BaseDB (qualified, db.dreams); `_GuardianAdapter.capture` (generic_boundary, native-operational-recovery) | signed off 2026-11-06 |
| 9 | Named cohorts, unsupported | The module's reviewed cohort, else the cohort table's package cohort | 125 | agents 36, tts 23, models 15, mcp 13, workspaces 9, assets 8, config 6, skills 5, files 4, other 6 | signed off 2026-11-06 |
| 10 | miscellaneous, unsupported | No named cohort, so the doc's conservative default applies. This includes pipes, sockets and widget writes that row 6 does not cover | 307 | Plugins 61, Actor_Packs 30, UI 29, LLM_Management 24, Audio 23, Web_Server 17, Tool_Packs 16, Personal_Context 15, Tools 13, other 79 | signed off 2026-11-06 |
| 11 | Backup_Recovery's own new pipeline | Classified as proposed and signed off (listed below); was `unsupported` pending review | 20 | `produce_detached_archive`; `relocation_lock`; `acquire_source` | signed off 2026-11-06: proposals applied |
| 12 | New SQLite members, settled by flags | No capture flag, so `sqlite/unsupported` | 8 | chat.launch_wake, plugins.registry, workflows.local | signed off 2026-11-06 |
| 13 | New SQLite members whose policy reason says "excluded" | `db.workspace_project_index` ("excluded from backup and sync") and `personal_context.interview_drafts` ("excluded from centralized backup") are `sqlite/unsupported`. A False flag does not make an exclusion; `db.chachanotes.primary` has the same flags. Alternative: an installed exclusion adapter, like `notes.sync_state`'s `_DeviceStateExclusion` | 2 | — | signed off 2026-11-06: keep unsupported + follow-up |
| 14 | AC#2: `tests.store_kernel` | Implemented: removed from the production registry, and the kernel tests register it themselves. Alternative: keep it in the registry, with a `tests/excluded` row in both inventories | 1 member | — | signed off 2026-11-06: accepted |
| 15 | Scanner vocabulary | The blind spots above, fixed in the test only. Against the original scanner they account for 28 "gone" rows, 1 changed count and 70 added rows. Known limits: tempfile, third-party writers, subprocesses, dynamic bindings, `x.open`/`x.connect` passed by reference, and introspection arguments (`inspect.signature`, `isinstance`, `issubclass`, `callable`). The doc says vocabulary extensions are part of owner review | — | `from …platform_files import os`; `to_thread(path.write_bytes, …)`; `yaml.safe_dump`; `os.rename` | signed off 2026-11-06: accepted |

Owner sign-off (2026-11-06), quoted answers to the table:
- Rows 1-10 and 12: "Sign off."
- Row 11: "Apply the proposals." The 20 Backup_Recovery pipeline rows are now classified exactly as proposed below.
  - Detached-layout staging is `disposable` (private output staging).
  - The detached-source reads (`acquire_source` open, `review_source.walk` open) are `external_input` (verified acquired archive read only). Its private copy (`acquire_source` create_private_file, `_hash` write) is `disposable` (private archive staging).
  - `portable_relocation` and `config_binding` are `generic_boundary` / backup_control.
  - `destinations._configs` and `effective_roots` are `generic_boundary` / backup_control_reader. The proposal said "reader"; this is the doc's existing backup reader cohort.
  - `launcher._open_tty` is `process_artifact`.
  - `agent_worktree_storage` stays `unsupported` (workspaces).
- Row 13: "Keep unsupported + follow-up." `db.workspace_project_index` and `personal_context.interview_drafts` stay `sqlite/unsupported`; see Proposed follow-ups.
- Rows 14 and 15: "Accept both." faeffb9c0e (`tests.store_kernel` registered by its test, not production) and the scanner vocabulary extension stay.

Proposed follow-ups (the owner files them; no task file created here):
- Install real exclusion adapters for `db.workspace_project_index` and `personal_context.interview_drafts`, like the installed `notes.sync_state` exclusion (`_DeviceStateExclusion` in `Notes/recovery.py`). Their policy reasons already say they are excluded from backup.

The 20 row-11 rows, classified as proposed (owner sign-off 2026-11-06):
- `detached_layout`: `_normalize_sqlite` connect, create_private_file ×2 and write; `produce_detached_archive` create_private_file ×2 and write ×2; `_restored_cache_paths` open. This is the confined import of acquired portable bytes. Proposed: `disposable`, private output staging.
- `detached_source`: `_hash` write; `acquire_source` create_private_file and open; `review_source.walk` open. This is the read-only acquisition of detached files. Proposed: reads `external_input` ("verified acquired archive read only"); the private copy `disposable`.
- `portable_relocation`: `_write` create_private_file and write; `relocation_lock` create_private_file and open. Proposed: `generic_boundary`, backup_control.
- `config_binding.preserve_owned_binding`: os.replace and os.unlink. Proposed: `generic_boundary`, backup_control.
- `destinations._configs` ZipFile and `effective_roots.effective_roots` open. Proposed: `generic_boundary`, reader.
- `launcher._open_tty` open (`/dev/tty`). Proposed: `process_artifact`.
- `agent_worktree_storage.prepare_agent_worktrees_base` secure_private_directory. Proposed: stays `unsupported`, cohort workspaces (durable agent worktrees).

Observations for later cohort work (no decision needed now):
- The Personal Context database is encrypted, and its keys are in the keyring (`key_protector`, `link_key_custody`, both rows `unsupported`). A backup of the database without its key cannot be read.
- `Tools/local_tool_impls` reaches `os.replace` through a dynamic `file_os` binding, which the static census cannot see.

**Evidence.**
- Red before, recomputed with the test's own `production_census()` at dev 2684e5453f:
  - Doc 1059 rows against production 1509: 644 added, 194 gone, 12 changed.
  - SQLite: 13 new, 8 gone.
  - At the base worktree, serial: 2 failed, 9 passed in 24.6 s: `test_every_persistence_source_symbol_has_reviewed_owner_row` and `test_sqlite_registry_members_have_explicit_census_rows`.
- Each scanner extension's unit test was red before its fix:
  - `test_census_tracks_os_bound_by_from_import_shim`: `assert 0 == 1`.
  - `test_census_tracks_native_callable_bound_by_tuple_assignment`: `assert 0 == 1`.
  - `test_census_native_alias_assignment_is_lexically_scoped`: `assert 1 == 0` (the `identity` case).
  - `test_census_counts_producer_callables_passed_by_reference`: `assert 0 == 1`.
  - `test_census_counts_yaml_safe_dump_serializers`: `assert 0 == 1`.
  - `test_census_tracks_native_rename_and_link_publication`: `assert 0 == 2`.
  - `test_census_ignores_arguments_to_introspection_calls` (review r11b-rev2): it counted 4 rows where it should count none (`assert Counter({...}) == Counter()`).
- Green after: `Tests/Architecture/test_backup_owner_inventory.py` (serial), 18 passed in 35.6 s (load average 28-45). Both inventory nodes pass, and the doc's 1607 rows equal production's 1607.
- After the owner sign-off (row-11 classifications applied; still 1607 rows): the inventory file (serial): 18 passed in 74.3 s (load average 48-56). `Tests/CI` at `-n 4`: 1215 passed, 1 skipped. preflight: all derived-artifact checks passed.
- AC#6 negative control, rerun after review r11b-rev1 (temporary source edits, then reverted):
  - The edits: a synthetic `_synthetic_unreviewed_producer` (open + write) in `Utils/file_durability.py`; a removed `mkdir` in `tls_trust._merged_bundle_path`; an extra `open` in `tls_trust._file_digest`.
  - Result: "2 added, 1 gone, 1 changed", with all four rows listed.
- AC#2 covering tests (round 1, no DB change since): `Tests/DB/test_store_kernel.py`, `test_private_sqlite_inventory.py` and `test_private_sqlite.py` at `-n 4`.
  - Result: 23 failed, 399 passed. The same 23 fail on plain dev 2684e5453f with the same command (23 failed, 401 passed). The 2 fewer passes are `OWNER_KIND_CASES` parameters of the removed owner.
  - These three files are in no PR lane, and their 23 reds are pre-existing.
- `Tests/CI` at `-n 4`: 1215 passed, 1 skipped.
- AC#7 timings:
  - CI before, read-only: on #3436 (fu/r10-arch, run 42218508176) the step took 402 s and the job 7.6 min, against a 20 min cap.
  - Local, `bash $FU/jobstep.sh fu-r11-backupinv "Run Architecture guard contracts"`, serial. Round 1, three runs: 318.6 s, 348.9 s and 370.8 s of pytest time (345, 375 and 398 s wall), 173 passed each time, with load average 27-65 and other agents sharing the 18 CPUs. After review r11b-rev1, one run: 591.9 s of pytest time (656 s wall), 177 passed, with load average 39-65. The file alone stays about 28 s, so the extra time is load.
  - The file alone: 27.7 s serial (24.6 s at base).
- `bash $FU/bootbudget.sh fu-r11-backupinv` (review r11b-rev1 round): 32 passed. `./scripts/preflight.sh` through run.sh (after review r11b-rev2): all derived-artifact checks passed.

**Files.**
- `Tests/Architecture/test_backup_owner_inventory.py`
- `backlog/docs/backup-recovery-owner-inventory.md`
- `tldw_chatbook/DB/private_sqlite.py`
- `Tests/DB/test_store_kernel.py`
- `Tests/DB/test_private_sqlite_inventory.py`
- `backlog/docs/sqlite-private-owner-inventory.md`
- `.github/workflows/derived-artifacts.yml`
- `Tests/CI/test_ci_queue_pressure_contract.py`
<!-- SECTION:NOTES:END -->
