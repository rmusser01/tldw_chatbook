---
id: TASK-38714
title: "Audit wrong-thread closes of thread-affine database handles"
status: Done
assignee:
- '@claude'
created_date: 2026-11-04 00:59
updated_date: 2026-11-05 07:46
labels:
- follow-up
- database
- threading
- bug
priority: medium
references:
- tldw_chatbook/Backup_Recovery/participants.py
- tldw_chatbook/Notes/file_notes_replica.py
- tldw_chatbook/Workflows/authoring.py
- tldw_chatbook/DB/Workflows_DB.py
- tldw_chatbook/DB/store_kernel.py
dependencies:
- TASK-36531
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in round 6 of the 2026-10-23 follow-up burn-down (TASK-36531's fix). Checked on dev c2cdb6b82e, 2026-11-03.

**The trap.** `FileNotesReplica.close()` (`Notes/file_notes_replica.py:130-139`) closes through `_core_closing` (`Backup_Recovery/participants.py:464`), which b5251e9a6e (TASK-32628) added. For a registered repository type, `_core_closing` refuses a close made from any thread other than the connection's resource thread (:477-493). The close then returns with the connection still open. It raises nothing and logs nothing. TASK-36531 traced a replica left open at quit to this: the replica was opened on one default-executor thread (`asyncio.to_thread`) and closed on another. It fixed that one seam by giving the workspace a one-thread executor for both. Nobody has checked the other closes.

**Thread-affine handles.** Besides the replica, `_core_closing` guards the closes in `ChaChaNotes_DB.py`, `Client_Media_DB_v2.py`, `Prompts_DB.py`, `Library_Ingest_Jobs_DB.py`, `Evals_DB.py`, `Dreams_DB.py`, `Subscriptions_DB.py`, `store_kernel.py`, `app_collections_wiring.py` and `Chat/console_change_review_read.py`. Thread-local handles have a related problem: `store_kernel.py`'s `close()` (:287-298) closes only the calling thread's connection, so a close made from a pool thread leaves the opener's connection open.

**Closes routed through a pool thread.** The audit starts from these sites:
- `asyncio.to_thread(<x>.close)` in `app_lifecycle.py` (:213, :400, :427, :1112), `Chat/console_runtime.py` (:6846), `Chat/console_voice_attempts.py` (:858), `Chat/console_send_diagnostics.py` (:235) and `Audio/voice_transcription.py` (:190, :270);
- `Workflows/authoring.py`, which is one candidate. It opens `WorkflowsDB` with `asyncio.to_thread(WorkflowsDB, ...)` (:57) and closes it with `asyncio.to_thread(self._db.close)` (:168). On a read, `WorkflowsDB.close()` (`DB/Workflows_DB.py:76`) is a plain close under an `RLock` on a `check_same_thread=False` connection, and `WorkflowsDB` is not among `_repository_types()`. So it may turn out not to be thread-affine; the audit should record that either way.

`Dreams_DB.close_all()` (:410) is the explicit all-threads close for a handle its caller owns outright, so it is out of scope.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Every close of a thread-affine database handle that can run on a thread other than the one that opened it now runs on the opening thread, or the task notes show why that handle is not thread-affine; the notes list every call site checked
- [x] #2 Each close that is fixed has a test that fails on the base build when the open and the close land on different threads (for example with a thread-per-call executor, as TASK-36531's test does)
<!-- AC:END -->

## Implementation Plan

ADR required: no
ADR path: N/A
Reason: Four call-site fixes using existing retirement helpers; no new lifecycle contract.

1. Classify every `_core_closing`-guarded close: single connection (refused off its opening thread), thread-local (closes only the caller's connection), or acquired and closed in one frame.
2. Check every close dispatched to another thread (an AST scan for `to_thread`/`run_in_executor`/`submit` taking a close, shutdown or dispose), including the listed sites and `Workflows/authoring.py:168`.
3. Check every database close call site outside `DB/` for the thread that opened the handle it retires against the thread the close runs on.
4. Fix each close that runs off the opener's thread with the existing helper for that store, and give each fix a thread-per-call test, red before.

## Implementation Notes

**Handle kinds.**
- **Single connection, refused off its opening thread by `_core_closing`.** `FileNotesReplica._connection` and `LibraryIngestJobsDB._conn`, each opened lazily by whichever thread uses it first.
- **Thread-local; `close()` retires only the calling thread's connection.** `CharactersRAGDB`, `MediaDatabase`, `PromptsDatabase`, `EvalsDB`, `DreamsDB.close`, `SubscriptionsDB.close`, and the `SQLiteStoreKernel` family (`AgentRunsDB`, `WorkspaceDB`, `LibraryCollectionsDB`, `GuardianDB`, `ClientNotificationsDB`, `RAGIndexingDB`, `WorkspaceProjectIndexDB`).
- **Acquired and closed in one frame.** `app_collections_wiring.py:376` (`_build_deferred_collections_capture_parts`) and `Chat/console_change_review_read.py:152` (`read()`).
- **Not thread-affine.**
  - `WorkflowsDB`, confirming the filer: one `check_same_thread=False` connection, a plain close under an `RLock`, and not in `repository_types()`. `AdmittedConnection.close` has no thread check. A probe through the wrapper opened it on one thread and closed it on another: the native handle raised "Cannot operate on a closed database" and `_closed` was True.
  - `LocalWritingService` and `LocalResearchService` (`app_lifecycle.py:400`, `:427`): connections in a dict keyed by thread id, `check_same_thread=False`, and `close()` closes every idle one directly, not through `_core_closing`.
  - `EventStateRepository.close` and `NotesDeviceStateStore.close` close from any thread. `DreamsDB.close_all`, `SubscriptionsDB.close_all_connections` and `CharactersRAGDB.quiesce_connections` are the all-thread closers.
  - `WorkspaceProjectIndexDB.close_native` raises `ProjectIndexRetirementUnproven` unless retirement is proven, so a wrong-thread close there is loud by design.

**Closes dispatched to another thread (all 40 the scan found).** None of these is a wrong-thread close of a thread-affine database handle.
- **Not a database handle.**
  - `app_lifecycle.py:213` `RecoveryService.close`: executor shutdown and workspace discard.
  - `:1112` and `Chat/console_send_diagnostics.py:235`: `UIResponsivenessMonitor.close` joins threads.
  - `:599` `RawCliRuntime.shutdown`, `:760` `ChangeReviewConsentService.shutdown`, `:775` `MeetingSessionOwner.shutdown`.
  - `app_shutdown.py:174-175`: SSH sessions and masters.
  - `Chat/console_runtime.py:6847`: provider gateway `aclose`, an HTTP client.
  - `console_runtime.py:6832` and `console_chat_controller.py:18435/18495`: scratch-space dispose, files.
  - `Chat/console_voice_attempts.py:858`: provider transport force-close. `console_voice_worker.py:255`: the voice worker.
  - `Audio/voice_transcription.py:190` (Parakeet subprocess), `:247` (executor), `:268/270` (STT stream). `voice_process_entry.py:110`, the STT. `TTS/backends/chatterbox.py:1329`. `UI/Navigation/audio_cpp_model_handoff.py:323`.
  - `Chat/console_agent_bridge.py:9510/9546` and `console_chat_store.py:5403`: in-memory message inboxes (`Agents/fleet_messages.py`).
  - `console_chat_controller.py:3440`: remote sessions. `console_provider_gateway.py:7024`: an HTTP response.
  - `TTS/profile_repository.py:6107`: executor shutdown. That repository opens and closes per operation on its own one-thread executor.
  - `Agents/run_hooks.py:767/1097`, `run_webhooks.py:210`, `runtime_policy/server_context.py:1048`.
  - `Widgets/Library/library_file_notes_workspace.py:2766`: the session owner, Git with no SQLite.
- **Opened and closed on the same pool thread.** `Chat/chat_conversation_scope_service.py:195` (`list_and_close`) and `Subscriptions/db_offload.py:76` (`run_and_close`).
- **Already correct.** `library_file_notes_workspace.py:2774`, TASK-36531's one-thread executor.
- **Out of scope.** `Dreams/track_reconcile.py:774`, `close_all`.
- **Thread-local close on a pool thread that is not a defect.** `console_runtime.py:6812`: when the Change Review coordinator never started, `_close_publisher_once` runs `runs_db.close` on a pool thread. No publisher thread ever opened a connection, so that close reaches only the pool thread's own, if any.
- **Not thread-affine.** `Workflows/authoring.py:168`, `app_lifecycle.py:400` and `:427` (above).

**Database close sites outside `DB/` (63; a read-only sweep classified each, and I re-checked the cross-thread ones).**
- **The same frame or thread as the open (52).**
  - `app_speech.py:707`; `app_ingest_queue.py:956/963/980/1009` (restore worker) and `:5013`; `config.py:8555`; `app_feature_glue.py:240`.
  - `app.py:1833/2094/2126/2176/5009/5072`; `app_lifecycle.py:1576`.
  - `UI/Screens/home_screen.py:731`; `personas_screen.py:9582/12267`; `UI/Console_Modules/workspace.py:4012` and `message.py:2261`; `Home/active_work_adapter.py:752`.
  - `Chat/console_runtime.py:4984/5141`; `console_worktree_recovery.py:255/334`; `chat_conversation_scope_service.py:193`.
  - `Agents/activation.py:254`; `agent_service.py:4197`; `Workflows/local_steps.py:129`; `MCP/local_server_tools.py:149`; `Workspaces/change_retention.py:230`.
  - `Subscriptions/db_offload.py:74`; `watchlist_preview_service.py:66`; `Local_Ingestion/local_file_ingestion.py:2651`.
  - `Chatbooks/chatbook_creator.py:507/693/1818/1869`; `chatbook_importer.py:1070/1080/1404/1851/1946/3207`.
  - `Notes/notes_scope_service.py:2024/2199`; `Backup_Recovery/dictionary_source_job.py:180`; `Dreams/track_reconcile.py:397`.
  - `Notes/file_notes_service.py:371` and `library_file_notes_workspace.py:2790`, both on the replica thread since TASK-36531.
- **Use and close both on the loop thread (4).** `console_runtime.py:6824` (worker handles are retired by their own owners); `Web_Server/mobile_service.py:1864/1904` (`:1904` after `quiesce_connections`).
- **Not relevant.** `runtime_policy/server_parity_state.py:50-51` (an all-thread close, and an in-memory-only close; neither has callers). `RAG_Search/simplified/rag_service.py:4461` (a Chroma client). `Chat/console_agent_bridge.py:9564` (the in-memory inbox store). `Dreams/track_reconcile.py:478/484` (`close_all`, out of scope).
- **Cross-thread, fixed below (4).** The four fixes that follow.

**Fixes.** Each close now retires its handle on the thread that opened it.
1. **`app_lifecycle.py:1292`, the Library ingest job store at app shutdown.** `_reconcile_research_source_held_jobs` read the store with `asyncio.to_thread`. If that read opened the store's single connection, the shutdown close on the UI thread was refused silently, and the connection stayed open. The read now runs on the UI thread. The module documents the store as UI-thread-only, and every registry write to it already runs there.
2. **`Widgets/settings_agents_panel.py:420`, `on_unmount`.** The panel's own `AgentRunsDB` read definitions through `to_thread` and reconciled orphans on a thread worker. Each left a connection open on its thread, and the unmount close reaches only the UI thread's. Both now go through the existing `run_owned_db_call`, which retires the handle each call opened, on that call's thread.
3. **`RAG_Search/simplified/db_connection_pool.py:61`, `close_all_pools`.** Keyword searches ran on executor threads, each caching a pooled `MediaDatabase` connection, and `close_all_pools` closes only the closing thread's. `_perform_fts5_search` now runs its query through `run_retiring_connection` (new, in `db_connection_pool.py`), which wraps the existing `operation_owned_connection`. The pooled database instance is still reused. Trade-off: each search opens a connection (marked `ponytail:` with the upgrade path).
4. **`Evals/eval_orchestrator.py:1164/1173`, `aclose`/`close`.** Each off-loop `store_result` left an `EvalsDB` connection on its pool thread, and `aclose()` closes only the loop's. The write now closes the connection it opened, on its own thread, and leaves one its thread already held. Nothing in the app calls `aclose`/`close` today (the app never closes its orchestrator), but `DEVELOPER_GUIDE.md` documents `await orchestrator.aclose()`.

**Tests (AC#2).** Each installs a thread-per-call default executor, as TASK-36531's test does, captures the native handles the store opens and the thread that opened each, and asserts each worker handle raises "closed database" after the owner's close. The ingest test also asserts `_conn is None`. A thread-local slot on an exited thread cannot be read, so the native handle is the evidence there. Each was red with only its source fix reverted:
- `Tests/App/test_submit_library_ingest_job.py::test_held_job_scan_leaves_the_ingest_store_closable_at_shutdown`: `assert <AdmittedConnection ...> is None`.
- `Tests/UI/test_settings_agents_offloop_read.py::test_owned_runs_db_retires_every_worker_connection_at_unmount`: `DID NOT RAISE`. Appended to the UI PR-gate census tail, with the floor raised to the entry count.
- `Tests/RAG_Search/test_keyword_leg_db_resolution.py::test_keyword_search_leaves_no_executor_connection_after_close`: `DID NOT RAISE`. Verified under a temporary `@private_profile_test` marker, then removed: this file's plain tests all fail locally with `RecoveryRequired: raw_source_selection_changed` (38 of 38 at base, TASK-38111).
- `Tests/Evals/test_orchestrator_offloop_writes.py::test_aclose_leaves_no_worker_thread_connection_open`: `DID NOT RAISE`. It also checks both results are still durable.

**Covering runs (`-n 4`, load 20-26).**
- Settings panel, Evals, ingest-restore and worker-failure files: 120 passed, 9 failed. The 9 are `RecoveryRequired` default-path tests in `test_eval_orchestrator*.py`, the same set at base.
- `test_submit_library_ingest_job.py`: 120 passed, 33 failed, the same 33 `RecoveryRequired` failures at base.
- The RAG files: the base failure set, plus the new test, which fails locally only through that error.
- `Tests/CI`: 1177 passed. Module size ratchet: 201 passed.

**Files.** `tldw_chatbook/app_ingest_queue.py`, `tldw_chatbook/Widgets/settings_agents_panel.py`, `tldw_chatbook/RAG_Search/simplified/db_connection_pool.py`, `tldw_chatbook/RAG_Search/simplified/rag_service.py`, `tldw_chatbook/Evals/eval_orchestrator.py`, the four test files above, `scripts/ui_pr_gate_census.txt`, `scripts/check_ui_pr_gate_census.py`.

