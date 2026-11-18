---
id: TASK-41521
title: "Other single-file data owners have the same created-file rollback gap TASK-41113 fixed for personas"
status: In Progress
assignee: ['@claude']
created_date: 2026-11-17 05:27
updated_date: 2026-11-18 03:14
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
- [ ] #2 Each owner that can is covered by a rollback test that fails before the fix
- [x] #3 The mid-session quarantine summary updates immediately, or the notes record why it stays
<!-- AC:END -->

## Implementation Plan

1. Enumerate the backup-enabled owners from `owner_registry.install_adapters()` and `data_groups.group_for_owner`, and classify which capture a single file.
2. Run a cross-profile restore end to end that creates each candidate file, record which files it creates, and where the later rollback refuses.
3. Admit the owners that use TASK-41113's created-file proof unchanged by adding them to `_CREATED_FILE_OWNERS`; list the others with the reason.
4. Extend TASK-41113's created-target scenario (and its nested-target counterpart, whose own copies must come back) to cover each admitted owner, red at base.
5. AC#3: read `_compensate_failed_creation` and decide whether the cached summary should refresh.

## Implementation Notes

**Summary.** A later rollback now retires eight more single files a cross-profile restore created, through TASK-41113's created-file proof unchanged: `audio.history`, `chat.dictionary_history`, `chat.grammars`, `chat.prompt_history`, `chat.rag_context`, `chatbooks.registry`, `feedback` and `notes.templates` join `personas` in `later_rollback._CREATED_FILE_OWNERS`. The only code change in the proof is its declaration-type check, which now accepts the chatbook registry's own `_Definition` subclass, `_ChatbookRegistry`, for that owner alone (the same per-owner form the created-root check uses for `eval.definitions`). The SQLite databases, the MCP stores, `config.history` and `tamagotchi.config` can also be created by a restore and are still refused; they need their own proofs (see the table and the follow-ups). The coordinator's cached recovery summary now names an intent quarantined mid-session (AC#3).

**AC#1: the backup-enabled single-file owners.** Enumerated from `owner_registry.install_adapters()` with `data_groups.group_for_owner` (owners with no group are not backup-enabled), keeping the owners whose capture is one regular file (with, at most, rotation, backup or SQLite sidecars beside it). "Creatable" means a cross-profile replace restore into a profile without the file publishes it with no previous file (`previous is None`), which sends it through `later_rollback._created_destination_target`; a same-profile restore keeps the target's own logical id, and `_known_absences` already retires it (read from the code, not run here). Probe: a created-source profile holding the file, restored into a populated profile without it (the `created-target` route below), with the created artifacts printed from `_created_destination_target`.

| Owner | Adapter | File | Restore can create it | Later rollback after this task |
|---|---|---|---|---|
| `personas` | `_Definition` | data `tldw_chatbook_personas.json` | yes (TASK-41113) | retired (TASK-41113) |
| `chat.dictionary_history` | `_Definition` | data `tldw_chatbook_chat_dictionary_history.json` | yes, probe | retired (this task) |
| `chat.rag_context` | `_Definition` | data `tldw_chatbook_chat_rag_context.json` | yes, probe | retired (this task) |
| `feedback` | `_Definition` | data `tldw_chatbook_feedback.json` | yes, probe | retired (this task) |
| `chat.grammars` | `_Definition` | data `tldw_chatbook_chat_grammars.json` | yes, probe | retired (this task) |
| `chat.prompt_history` | `_Definition` | data `prompt_history.jsonl` | yes, probe | retired (this task) |
| `audio.history` | `_Definition` | data `tldw_chatbook_audio_history.json` | yes, probe | retired (this task) |
| `notes.templates` | `_Definition` | config folder `note_templates.json` | yes, probe | retired (this task) |
| `chatbooks.registry` | `_ChatbookRegistry` (a `_Definition` subclass) | data `tldw_chatbook_chatbooks.json` | yes, probe (refused at base: first check of `_created_destination_target`) | retired (this task) |
| `ui.state` | `_Definition` | config folder `ui_state.toml` | yes | already retired by `_created_config_file_targets`; `test_config_file_retirement.py::test_actual_config_file_retirement_preserves_edits_then_restores_absence[ui.state]` |
| `ui.emoji_recents` | `_Definition` | config folder `recent_emojis.json` | yes | as `ui.state` (`[ui.emoji_recents]`) |
| `runtime.source_state` | `_SourceState` | config folder `runtime_policy.json` | yes | as `ui.state` (`[runtime.source_state]`) |
| `config` | `_Config` | the profile's `config.toml` | no: a replace restore writes over the target's own selector, which exists | n/a |
| `config.history` | `_History` | `config.toml.bak` and dated siblings | yes, from `staging.py` (restored beside the target config); not run | not covered: refused like any non-admitted owner; follow-up |
| `tamagotchi.config` | `_ConfiguredPets` | `~/.config/tldw_chatbook/tamagotchi_pets.json` (+ dated backups), outside the profile | yes, from `_restore_path`; not run | not covered; follow-up |
| `mcp.local`, `mcp.targets`, `mcp.context`, `mcp.permissions` (+`.bak`), `mcp.history` (+`.1`) | `MCP/recovery._Store` | data folder JSON/JSONL | yes, from `_Store.discover` (canonical data-folder path); not run: this harness's Tools-group backup stops at `credential_processing_unavailable` | not covered; follow-up |
| SQLite owners: `db.agent_runs`, `db.chachanotes.primary`, `db.dreams`, `db.evals`, `db.guardian`, `db.library_collections`, `db.library_ingest_jobs`, `db.media.primary`, `db.prompts.primary`, `db.rag_indexing`, `db.scheduled_tasks`, `db.subscriptions`, `db.workspaces`, `kanban.local`, `notes.file_notes`, `notifications.client`, `personal_context.repository`, `research.local`, `runtime.event_state`, `runtime.sync_state`, `tts.profile_store`, `writing.local` | SQLite declarations | one database file each, with `-wal`/`-shm` sidecars | yes when the target lacks the database; probe: `notes.file_notes` deleted from the target, created by the restore, and refused (`REFUSED` at the first check, `local_snapshot_absence_unclassified`) | not covered; follow-up |

Owners that share another owner's file (`chat.attachments`, `notes.sync_bindings`, `quiz.local`, `study.local` in ChaChaNotes; `tts.references` in the TTS profile database) have no file of their own. Owners with no backup group (`hooks.permissions`, `cache.model_catalog`, `runtime.*` locks and logs, `notes.sync_state`, `personal_context.interview_drafts`) are not backup-enabled.

**Tests (AC#2).** `Tests/Backup_Recovery/test_created_persona_subtree_rollback.py`'s TASK-40305/41113 scenario script, in the Backup_Recovery owner suites lane:
- `created-source` also writes each of the eight files (the registry as one metadata-only entry) and restores with the Audio, Saved chatbooks, Conversations, Prompts and Workspaces groups (`changed` keeps its two).
- `test_later_rollback_of_a_restore_that_created_actor_pack_trees` (`created-target`) asserts each file is absent before the restore, created by it with the source's text, and gone after the later rollback.
- `test_later_rollback_of_a_restore_that_created_nested_actor_pack_folders` (`nested-target`) writes its own copy of each file first, asserts the restore replaced each, and that the rollback brought each back. Only a created file (`previous is None`) may be retired; the post-restore check makes the post-rollback one non-vacuous (own and incoming text differ for every file).

Red/green (load 55-80, `-n 2` unless noted):
- Red at base: with the set reduced to `{"personas"}` (temporary environment hook in the source, since removed), the created test failed `ValueError: local_snapshot_absence_unclassified` in `_created_destination_target`'s first check; the nested test passed, as it should (the restore replaces those files).
- Each owner on its own: with the full set less that one owner (same hook), the created test failed the same way, `local_snapshot_absence_unclassified` at `_created_destination_target`'s first check, for each of the eight (serial, 115-257 s each, four at a time, load 40-100).
- Green: 2 passed (332 s).

**AC#3: the mid-session quarantine summary now updates.** The stale summary was not harmless: `app_service_wiring.ensure_actor_pack_recovery` reads the cached `ensure_recovered()` summary on every Personas mount, and only when it names no blocked intent does it run the Buddy library setup (`BuddyLibrary.ensure_builtin`), which start-up skips while an intent is quarantined. After a mid-session quarantine the cached summary still named none, so every later Personas mount still ran that setup, which start-up withholds while an intent is quarantined; with the summary updated, the next mount sets `actor_pack_recovery_blocked` and skips it, as start-up does. `_compensate_failed_creation` now calls `_sync_recovery_summary()`, the summary update `clear_blocked_intents` already made (moved into that helper). Test: `Tests/Actor_Packs/test_persona_actor_pack_coordinator.py::test_mid_session_quarantine_updates_the_cached_recovery_summary`, red at base (`assert () == ('a20fc797...',)`), green after; the file: 19 passed.

**Lane.** `Tests/Actor_Packs/test_persona_actor_pack_coordinator.py` runs on no PR lane (nightly-deep only); proposed to the coordinator for a lane rather than edited into `derived-artifacts.yml`, which this brief did not cover.

**Covering runs** (`-n 4`, load 35-100): `test_later_created_destinations.py`, `test_later_rollback.py`, `test_retirement_only_rollback.py`, `test_later_rollback_handoff.py`: 45 passed. `test_rollback_group_scope.py`, `test_config_file_retirement.py`: 57 passed. `Tests/Actor_Packs/test_actor_pack_creation.py`, `Tests/UI/test_actor_pack_recovery_seam.py`: 22 passed. Serially, `Tests/Architecture/test_backup_owner_inventory.py`'s interrupted-import, clear, blocked, approver-census and review-label nodes: 17 passed. The whole of `test_created_persona_subtree_rollback.py` (the changed, created and nested routes, the native nested pack and its nine negatives): 23 passed (753 s, `-n 4`, load 35-60).

**Files.** `tldw_chatbook/Backup_Recovery/later_rollback.py`, `tldw_chatbook/Actor_Packs/persona_coordinator.py`, `Tests/Backup_Recovery/test_created_persona_subtree_rollback.py`, `Tests/Actor_Packs/test_persona_actor_pack_coordinator.py`.

**Not met: AC#2 for the owners the table marks "not covered".** Each needs more than joining the set: the SQLite owners a proof for the database and its `-wal`/`-shm` sidecars; the MCP stores their credential review and retained `fresh`/`history` destinations; `config.history` its path-keyed family of files beside the config, which the proof's owner-keyed rediscovery does not model; `tamagotchi.config` a path outside the profile. They are proposed as follow-ups, so this task stays In Progress.

Also read from the code, not run: a restore of saved chatbooks *with* archives into a profile without a `chatbooks/` folder creates that tree root, and `chatbooks.archives` is not in `_CREATED_TREE_ROOT_OWNERS`, so that later rollback would still refuse even though the registry file now retires. The test's registry holds one metadata-only entry for that reason.
