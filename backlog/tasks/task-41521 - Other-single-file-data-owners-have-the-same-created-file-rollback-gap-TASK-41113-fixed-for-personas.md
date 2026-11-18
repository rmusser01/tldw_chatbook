---
id: TASK-41521
title: "Other single-file data owners have the same created-file rollback gap TASK-41113 fixed for personas"
status: Done
assignee: ['@claude']
created_date: 2026-11-17 05:27
updated_date: 2026-11-18 06:11
labels:
- follow-up
- backup
- data-loss
priority: high
references:
- tldw_chatbook/Backup_Recovery
- tldw_chatbook/Personas
dependencies:
- TASK-41113
- TASK-41114
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by the actor stream (round 15). TASK-41113 fixed the later-rollback refusal after a restore created the persona store file, for personas only. Other single-file data owners (dictionary history, RAG context, and any other owner that a restore can create) have the same gap, so a later rollback may refuse or leave the created file behind. Also noted: mid-session quarantine (`_compensate_failed_creation`) updates only `_blocked_intent_ids`, so the cached summary stays unblocked until a clear; this is pre-existing and harmless.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Every backup-enabled single-file owner is listed in the notes with whether a restore can create its file
- [x] #2 Each owner that can is covered by a rollback test that fails before the fix, except the Personal Context repository, archives added to an existing chatbooks folder, and MCP store fresh copies
- [x] #3 The mid-session quarantine summary updates immediately, or the notes record why it stays
<!-- AC:END -->

## Implementation Plan

1. Enumerate the backup-enabled owners from `owner_registry.install_adapters()` and `data_groups.group_for_owner`, and classify which capture a single file.
2. Run a cross-profile restore end to end that creates each candidate file, record which files it creates, and where the later rollback refuses.
3. Admit the owners that use TASK-41113's created-file proof unchanged by adding them to `_CREATED_FILE_OWNERS`; list the others with the reason.
4. Extend TASK-41113's created-target scenario (and its nested-target counterpart, whose own copies must come back) to cover each admitted owner, red at base.
5. AC#3: read `_compensate_failed_creation` and decide whether the cached summary should refresh.

## Implementation Notes

**Summary.** A later rollback now retires what a cross-profile restore created for every backup-enabled single-file owner a restore can create, except the shapes tracked in TASK-41705 (the Personal Context repository), TASK-41706 (Saved chatbook archives added to an existing `chatbooks/` folder) and TASK-41707 (MCP `fresh` copies, never run); AC#2 was narrowed to that in review r16rs-rev2. Round 16 first admitted eight `_Definition`-style files beside the persona store (TASK-41113's proof unchanged). Phase 2 (review r16rs-rev1) generalized the proof: an owner-to-adapter-class map replaces the chained type check; the archive root must be synthetic; a raw file is re-read through its owner's own `discover` filtered by path (so `.bak`, `.1` and dated copies keep their installed IDs); a database is never opened by the proof, takes the reviewed inventory's own row and is observed by device and inode only; the TTS profile store retires with its reference-audio sharer; and SQLite sidecars retire with their database, after the new safety copy has saved what the WAL held. The coordinator's cached recovery summary now names an intent quarantined mid-session (AC#3).

**AC#1: the backup-enabled single-file owners.** Enumerated from `owner_registry.install_adapters()` with `data_groups.group_for_owner` (owners with no group are not backup-enabled), keeping the owners whose capture is one regular file (with, at most, rotation, backup or SQLite sidecars beside it). "Creatable" means a cross-profile replace restore into a profile without the file publishes it with no previous file (`previous is None`), which sends it through `later_rollback._created_destination_target`; a same-profile restore keeps the target's own logical id, and `_known_absences` already retires it (read from the code, not run here). "Run" means a probe or test restored a source holding the file into a populated target without it and printed the created artifacts.

| Owner | Adapter | File | Restore can create it | Later rollback now | Test |
|---|---|---|---|---|---|
| `personas` | `_Definition` | data `tldw_chatbook_personas.json` | yes, run | retired (TASK-41113) | persona `created-target` |
| `audio.history`, `chat.dictionary_history`, `chat.grammars`, `chat.prompt_history`, `chat.rag_context`, `feedback` | `_Definition` | data JSON/JSONL files | yes, run | retired | persona `created-target`, `nested-target` |
| `notes.templates` | `_Definition` | config folder `note_templates.json` | yes, run | retired; refused by name (`local_snapshot_created_config_folder_shared`) when another profile's config shares the folder | persona routes; `[shared-config-folder]` |
| `chatbooks.registry` | `_ChatbookRegistry` | data `tldw_chatbook_chatbooks.json` | yes, run | retired, with the archives it references | persona routes; `[databases]` |
| `chatbooks.archives` (tree) | `_ChatbookArchives` | data `chatbooks/` | a created `chatbooks/` root: yes, run; archives added to an existing folder: yes (read from the restore's replace semantics, not run) | created root: retired (`_CREATED_TREE_ROOT_OWNERS`); added archives: **not covered** | `[databases]` |
| `config.history` | `_History` | `config.toml.bak` and `config_backup_<date>.toml` beside the config | yes, run | retired; each copy keeps its path-keyed ID | `[databases]`, `[tools]` |
| `rag.definitions` (pipelines file) | `_Definitions` | config folder `rag_pipelines.toml` | yes, run | retired; config-folder refusal applies | `[tools]` |
| `mcp.local`, `mcp.targets`, `mcp.context`, `mcp.permissions` (+`.bak`), `mcp.history` (+`.1`) | `MCP/recovery._Store` | data folder JSON/JSONL | yes, run (valid store content passes capture's credential step; no product check was changed) | retired; siblings keep their IDs | `[tools]` |
| `ui.state`, `ui.emoji_recents` | `_Definition` | config folder | yes | retired by `_created_config_file_targets` (unchanged) | `test_config_file_retirement.py` |
| `runtime.source_state` | `_SourceState` | config folder `runtime_policy.json` | yes | as above | `test_config_file_retirement.py` |
| `config` | `_Config` | the profile's `config.toml` | no: a replace restore writes over the target's own selector | n/a | - |
| `tamagotchi.config` | `_ConfiguredPets` | `~/.config/tldw_chatbook/tamagotchi_pets.json`, outside every profile | no, run: the restore itself refuses to publish into that folder (`replacement_destination_parent_required`), whether the folder is absent or holds a dated pet copy and was backed up first | refused by the generic rule (`local_snapshot_absence_unclassified`; the named refusal added in phase 2 was dropped in review r16rs-rev2 as unreachable) | none possible end to end |
| `db.chachanotes.primary`, `db.workspaces` | SQLite | data folder | no, run: a target without them is refused before the restore (`target_unverified`; their dependents' rows go unavailable) | n/a | - |
| `db.prompts.primary`, `db.agent_runs`, `notes.file_notes`, `tts.profile_store` (+ sharer `tts.references`) | SQLite | data folder | yes, run (deleted from the populated target first) | retired, sidecars and sharer with them | `[databases]` |
| `db.dreams`, `db.guardian`, `kanban.local`, `db.subscriptions`, `db.scheduled_tasks`, `notifications.client`, `runtime.event_state`, `runtime.sync_state` | SQLite | data folder | yes, run | retired | `[databases]` |
| `db.media.primary`, `db.library_collections`, `db.library_ingest_jobs` | SQLite | data folder | yes, run | retired | `[library]` (`db.library_collections` also `[tools]`) |
| `db.evals`, `research.local`, `writing.local` | SQLite | data folder | yes, run | retired | `[projects]` |
| `db.rag_indexing` | SQLite (`_Indexing`) | data `rag_indexing.db` | yes, run, from a database built with the owner's installed schema (today's `RAGIndexingDB` adds an `embedding_cache` table that capture refuses as `unsupported_schema`; follow-up) | retired | `[tools]` |
| `personal_context.repository` | SQLite (encrypted) | data `tldw_chatbook_personal_context.db` | yes (read from the code, not run) | **not covered** | - |

Owners that share another owner's file: `chat.attachments`, `notes.sync_bindings`, `quiz.local`, `study.local` (ChaChaNotes, which a restore cannot create) and `tts.references` (retires with `tts.profile_store`). Owners with no backup group (`hooks.permissions`, `cache.model_catalog`, `db.workspace_project_index`, `runtime.*` locks and logs, `notes.sync_state`, `personal_context.interview_drafts`) are not backup-enabled (review r16rs-rev1, Minor 3).

**Tests (AC#2).** `Tests/Backup_Recovery/test_created_persona_subtree_rollback.py`'s TASK-40305/41113 scenario script, in the Backup_Recovery owner suites lane:
- `created-source` also writes each of the eight files (the registry as one metadata-only entry) and restores with the Audio, Saved chatbooks, Conversations, Prompts and Workspaces groups (`changed` keeps its two).
- `test_later_rollback_of_a_restore_that_created_actor_pack_trees` (`created-target`) asserts each file is absent before the restore, created by it with the source's text, and gone after the later rollback.
- `test_later_rollback_of_a_restore_that_created_nested_actor_pack_folders` (`nested-target`) writes its own copy of each file first, asserts the restore replaced each, and that the rollback brought each back. Only a created file (`previous is None`) may be retired; the post-restore check makes the post-rollback one non-vacuous (own and incoming text differ for every file).

Red/green (load 55-80, `-n 2` unless noted):
- Red at base: with the set reduced to `{"personas"}` (temporary environment hook in the source, since removed), the created test failed `ValueError: local_snapshot_absence_unclassified` in `_created_destination_target`'s first check; the nested test passed, as it should (the restore replaces those files).
- Each owner on its own: with the full set less that one owner (same hook), the created test failed the same way, `local_snapshot_absence_unclassified` at `_created_destination_target`'s first check, for each of the eight (serial, 115-257 s each, four at a time, load 40-100).
- Green: 2 passed (332 s).

**AC#3: the mid-session quarantine summary now updates.** The stale summary was not harmless: `app_service_wiring.ensure_actor_pack_recovery` reads the cached `ensure_recovered()` summary on every Personas mount, and only when it names no blocked intent does it run the Buddy library setup (`BuddyLibrary.ensure_builtin`), which start-up skips while an intent is quarantined. After a mid-session quarantine the cached summary still named none, so every later Personas mount still ran that setup, which start-up withholds while an intent is quarantined; with the summary updated, the next mount sets `actor_pack_recovery_blocked` and skips it, as start-up does. `_compensate_failed_creation` now calls `_sync_recovery_summary()`, the summary update `clear_blocked_intents` already made (moved into that helper). Test: `Tests/Actor_Packs/test_persona_actor_pack_coordinator.py::test_mid_session_quarantine_updates_the_cached_recovery_summary`, red at base (`assert () == ('a20fc797...',)`), green after; the file: 19 passed.

**Lane.** `Tests/Actor_Packs/test_persona_actor_pack_coordinator.py` now runs whole in the PR lane `recovered-suites` (sandboxed block, after the Actor Pack export file): 19 tests, 13-15 s under the step's flags (`-p no:xdist --storage-lease-guard --timeout=180`) at load 12-16 (review r16rs-rev1; re-measured in fix round 2).

**Covering runs** (`-n 4`, load 35-100): `test_later_created_destinations.py`, `test_later_rollback.py`, `test_retirement_only_rollback.py`, `test_later_rollback_handoff.py`: 45 passed. `test_rollback_group_scope.py`, `test_config_file_retirement.py`: 57 passed. `Tests/Actor_Packs/test_actor_pack_creation.py`, `Tests/UI/test_actor_pack_recovery_seam.py`: 22 passed. Serially, `Tests/Architecture/test_backup_owner_inventory.py`'s interrupted-import, clear, blocked, approver-census and review-label nodes: 17 passed. The whole of `test_created_persona_subtree_rollback.py` (the changed, created and nested routes, the native nested pack and its nine negatives): 23 passed (753 s, `-n 4`, load 35-60).

**Files.** `tldw_chatbook/Backup_Recovery/later_rollback.py`, `tldw_chatbook/Backup_Recovery/publication.py`, `tldw_chatbook/Actor_Packs/persona_coordinator.py`, `Tests/Backup_Recovery/test_created_persona_subtree_rollback.py`, `Tests/Backup_Recovery/test_created_owner_rollback.py` (new), `Tests/Actor_Packs/test_persona_actor_pack_coordinator.py`, `.github/workflows/derived-artifacts.yml`, `Docs/Backup-and-Recovery.md`.

**AC#2 split (review r16rs-rev2).** The shapes not covered here are tracked:
- TASK-41705: `personal_context.repository`. A restore can create it, but a source fixture needs the encrypted profile and its key material (keyring), which this harness does not build; not attempted.
- TASK-41706: Saved chatbook archives added to an existing `chatbooks/` folder. A replace restore publishes the incoming archive files inside the target's existing root (and retires the target's own). Retiring them needs a nested-file proof of the folders they were published into, like `_created_persona_subtree` does for nested pack folders, and that proof is persona-specific (it checks the ChaChaNotes core and the persona config). A created `chatbooks/` root is covered.
- TASK-41707: MCP `fresh` copies, which a restore writes only when the target already holds the store; never run.
`tamagotchi.config` has no end-to-end test: today's restore refuses to write its folder (see the table).

### Review r16rs-rev1, fix round 1

- **Important 1: the recreated-file guard is tested.** New route `recreated-target` and test `test_later_rollback_refuses_a_created_file_recreated_after_the_restore`: after the restore, the created `feedback` file is replaced by a copy of itself through a temporary file and `os.replace` (same bytes, new inode). The later rollback's preview refuses with `local_snapshot_created_root_changed`, and the file is left as it was. Green: 1 passed (92 s serial, load 52-39). Red with the device/inode comparison in `_created_destination_target` temporarily disabled: `AssertionError: later rollback accepted a created file recreated after the restore` (79 s). Added test time: one more created-source backup and one target restore, about 1.5 min serial.
- **Minor 3.** `rag.definitions`' `rag_pipelines.toml` is in the table and covered (`[tools]`), and `db.workspace_project_index` is in the "no backup group" sentence.
- **Minor 4: a created file beside the config refuses when the folder is shared.** In `_created_destination_target`, a created file whose parent is the profile's config folder is refused with `local_snapshot_created_config_folder_shared` when another bound profile's config lives in the same folder, as `_created_config_file_targets` and publication already refuse. Test: `test_created_owner_rollback.py::...[shared-config-folder]`: the restore creates `note_templates.json`, a second profile's config is then bound in that folder (`control_records.bind_profile`; publication refuses a restore into such a folder, so only a later binding can create the state), and the later rollback must refuse by name and leave the file.

### Phase 2 (review r16rs-rev1): the remaining owners

**Code (`later_rollback.py`).**
- `_CREATED_OWNER_CLASSES` (owner to `(module, class)` under `tldw_chatbook`, loaded by `_created_owner_class`) replaces the chained type check for created roots, nested Persona folders and created files alike; `type(owner) is` the mapped class, so a substituted declaration is still refused.
- A created file must sit under a synthetic archive root (`document.directories[...].synthetic`), so a file of a declared tree never takes the single-file path.
- The re-read of a raw file is the owner's own `discover` (config loaded from the profile's config file, under `_preview_reads` outside a held session), filtered by path: `mcp_permissions.json.bak`, `mcp_execution_log.jsonl.1`, `config.toml.bak` and `config_backup_<date>.toml` keep their installed IDs, and `rag_pipelines.toml` keeps `rag.definitions:pipelines`. A database is not opened by the proof: opening it rewrites its `-shm`, and the reviewed plan's fingerprint hashes the `-shm` it retires, so execution refused with `target_changed` once sidecars existed. Its row is the reviewed inventory's own (discovered and validated when the target was previewed), and it is observed by device and inode, since its WAL and writers may still change its bytes; the safety copy keeps them. `_stage_read_sources` adds the profile's config to the held capture scope for the raw re-read, and accepts a database row, which carries no file metadata.
- Raw files keep their digest, capped at the owner's `_RAW_RECOVERY_LIMITS` (the history file's 16 MiB limit refused the shared cap).
- `_CREATED_FILE_SHARERS` (`tts.profile_store` -> `tts.references`): the sharer's row on the same file, in the inventory's shared group, is kept, and `_known_absences` retires it with the database; the re-read row takes the inventory's group. `publication.py` now lets two owners' rows on one shared database file in the same shared group retire as one (it refused that pair with `duplicate_retirement_target`; the existing sidecar alias rule already did this for `-wal`/`-shm`).
- `_CREATED_SHARED_FILE_OWNERS` (`tamagotchi.config`): named refusal `local_snapshot_created_file_shared_outside_profile` (the coordinator's call). No test reaches it: the restore refuses to write the pet folder first (table). Dropped in fix round 2.
- `chatbooks.archives` joins `_CREATED_TREE_ROOT_OWNERS`.

**Docs.** `Docs/Backup-and-Recovery.md`'s later-rollback section says files and databases the restore created are removed again, and when the review refuses instead (a file replaced after the restore, a file beside a now-shared config; since fix round 2 also the shapes tracked in TASK-41705 and TASK-41706).

**Tests.** New `Tests/Backup_Recovery/test_created_owner_rollback.py`, one orchestrating child per profile like the persona scenario, five cases: `[databases]`, `[library]`, `[projects]`, `[tools]` (each restore creates its files in a populated target, some after deleting a database from it; the later rollback must leave none, and no `-wal`/`-shm`), and `[shared-config-folder]` (refusal). In `[databases]` a writer opens the created kanban database in WAL mode after the restore and exits without a checkpoint, so its `-wal` and `-shm` files exist before the rollback and must go with it. Added to the Backup_Recovery owner suites lane after the persona rollback file.

**Red/green.**
- Green, `test_created_owner_rollback.py`: 5 passed with the final code (`-n 4`, load 11-23): `[tools]` 171 s, `[databases]` 168 s, `[projects]` 113 s, `[library]` 102 s, `[shared-config-folder]` 76 s.
- Red at base (`later_rollback.py` and `publication.py` from b52e74ae26, test unchanged): 5 failed (428 s, `-n 4`, load 52-91): `[databases]`, `[library]`, `[projects]` and `[tools]` with `ValueError: local_snapshot_absence_unclassified`, `[shared-config-folder]` with `AssertionError: later rollback accepted shared-config-folder`.
- Per owner at base, from the probes (a source holding the file, restored into a populated target, artifacts printed in `_created_destination_target`): each of `db.dreams`, `db.guardian`, `db.prompts.primary`, `db.agent_runs`, `notes.file_notes`, `tts.profile_store`, `db.media.primary`, `db.library_collections`, `db.evals`, `rag.definitions` (pipelines), `mcp.context` and `chatbooks.registry` was refused at the first check (`local_snapshot_absence_unclassified`) while not admitted; the others were refused at the same check by construction (not in the admitted set), and the case runs above go red together. Intermediate reds that shaped the code: the history file's 16 MiB limit (`invalid_capture_byte_limit` with the shared cap), the config outside the held scope at execution (`capture_source_outside_scope`), the TTS sharer (`local_snapshot_created_owner_conflict`, then `publication_preserved_overlap`, then `duplicate_retirement_target`), and `[databases]` with the kanban sidecars present (`target_changed` at execution while the proof opened the database, also with an identity-only observation; green once the proof stopped opening it).
- Persona scenario after the generalization: the whole of `test_created_persona_subtree_rollback.py`, 24 passed (844 s, `-n 4`, load 13-83, before the database re-read change, which touches only created databases); after it, the created, nested and recreated routes, 3 passed (78-146 s).
- Covering runs on the final code (`-n 4`, load 11-23): `test_later_created_destinations.py`, `test_later_rollback.py`, `test_retirement_only_rollback.py`, `test_later_rollback_handoff.py`, `test_rollback_group_scope.py`, `test_config_file_retirement.py`, `test_held_sqlite_rollback.py`: 134 passed (302 s); with `test_persona_actor_pack_coordinator.py` before the last change, 153 passed. Serially: `Tests/Architecture/test_module_size_ratchet.py` and the whole `Tests/Architecture/test_backup_owner_inventory.py`, 282 passed (85 s). Publication: `test_publication_crashes.py`, `test_publication_finalization.py` and `test_replacement_publication_scopes.py`, 84 passed and 3 failed (`test_actual_mapped_publication_adds_only_its_receipt_bound_selector_scope[split_pair]`, `test_complete_cli_replacement_admits_mapped_sources[False|True]`); the same three fail with `later_rollback.py` and `publication.py` from base 45b1949ca3 (same command, load 49), so they are not this change.
- Added test time: the new file's five cases are 76-171 s each at `-n 4` (about 10 min of child time); the persona file gains the recreated route (79 s).

### Review r16rs-rev2, fix round 2

- **Critical 1: the coordinator file leaked a storage lease under the recovered-suites flags.** `test_ensure_recovered_is_single_flight_across_threads` ran `ensure_recovered` on six bare threads; each opened its own ChaChaNotes handle, which outlived the test (`left 1 new live storage lease(s) open: SQLite .../actors.db on thread Thread-1`, 1 error in 3 of 3 lane-flag runs, and once more here before the fix). The product caller already scopes that handle: `app_service_wiring.ensure_actor_pack_recovery` runs recovery inside `operation_owned_connection(self.chachanotes_db)`. The test's threads now do the same, so the leak was the test's, not the product's. Under `-p no:xdist --storage-lease-guard --timeout=180`: 19 passed, 3 of 3 runs (13-15 s, load 12-16).
- **Important 1: a retired created database's post-restore writes reach the new safety copy.** In `[databases]`, the writer now opens the created kanban database in WAL mode with `wal_autocheckpoint=0`, adds a board, commits and exits without a checkpoint, so the board is only in the `-wal` (asserted non-empty). After the later rollback the test opens the new safety copy (`list_recovery_copies`, `archive_reader.acquire` with the new password, `verify_sealed`), reads the captured kanban database and finds the board, as `test_later_created_destinations.py` does for created edits. Green: 1 passed (92 s serial, load 12-16). Red with `_KanbanAdapter.capture` temporarily copying only the main file: `AssertionError: ('kanban.local', [])`.
- **Minor 1.** The Summary no longer says a database is observed by identity, size and time: it is never opened by the proof and is observed by device and inode.
- **Minor 2.** `Docs/Backup-and-Recovery.md` names the shapes a later rollback still refuses: a Personal Context database, and Saved chatbook archives added to an existing chatbooks folder. The "outside every profile" clause went with Minor 3.
- **Minor 3 (coordinator's call).** The `tamagotchi.config` special case is gone; the generic rule refuses it (`local_snapshot_absence_unclassified`), and the restore already refuses to write its folder. The owner table says so.
- **AC#2 split.** AC#2 now excludes TASK-41705, TASK-41706 and TASK-41707, filed here (status To Do, labels follow-up and backup, depending on this task); with that, every creatable owner is covered and AC#2 is ticked.

### Review r16rs-rev3, fix round 3

- AC#2 names the excluded shapes instead of the higher-id tasks (CLAUDE.md section 2); TASK-41705, TASK-41706 and TASK-41707 track them and depend on this task.
- The `[databases]` case now shows by itself that the board lived only in the `-wal`: after the writer, it copies the main database file alone and finds no board there. Control: with the writer running a passive checkpoint before it exits, the case failed at that assertion (`AssertionError: kanban.local`, 42 s). Green: 1 passed (102 s serial, load 7); preflight passed. (Fix round 4 corrects this note: this check runs before the inventory and the rollback preview, so it could not catch a checkpoint made by them.)

### Review r16rs-rev4, fix round 4

- The main-file-only check now also runs right after `preview_rollback` (whose target is a fresh `preview_backup`), so the test guards against the inventory or the rollback preview checkpointing the `-wal` before execution: the board must still be missing from the main file there, and the new safety copy must then hold it. A checkpoint made inside execution, before the safety capture, cannot be seen from this test: execution runs in one call, and its capture reads the main file and the `-wal` together. Control: with `_created_destination_target` temporarily running a passive checkpoint on a created database, the case failed at the new check (`AssertionError: ('kanban.local', 'previewed')`, 50 s). Green: 1 passed (96 s serial, load 7-9); preflight passed.

