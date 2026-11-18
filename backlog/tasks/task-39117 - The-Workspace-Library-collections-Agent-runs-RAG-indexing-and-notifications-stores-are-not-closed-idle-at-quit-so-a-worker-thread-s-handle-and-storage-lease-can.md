---
id: TASK-39117
title: "The Workspace, Library collections, Agent runs, RAG indexing and notifications stores are not closed idle at quit, so a worker thread's handle and storage lease can outlive the app"
status: Done
assignee: ['@claude']
created_date: 2026-11-07 02:03
updated_date: 2026-11-08 00:26
labels:
- follow-up
- database
- lifecycle
priority: low
references:
- tldw_chatbook/app_lifecycle.py
- tldw_chatbook/DB/store_kernel.py
- tldw_chatbook/DB/Workspace_DB.py
- tldw_chatbook/DB/Library_Collections_DB.py
- tldw_chatbook/DB/AgentRuns_DB.py
- tldw_chatbook/DB/RAG_Indexing_DB.py
- tldw_chatbook/Notifications/client_notifications_db.py
- Tests/DB/test_sync_log_retention.py
dependencies:
- TASK-39047
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in round 11 of the 2026-10-23 burn-down; checked on dev 2684e5453f, 2026-11-07, and on round-11 branch fu/r11-storage, which lands TASK-39047. TASK-39047 deliberately covers media, Dreams and Guardian only. Line numbers are dev's.

**What the quit close covers on fu/r11-storage.**
- `_close_main_databases` loops over exactly four stores: ChaChaNotes, media, Dreams and Guardian.
- Among the kernel stores, only `DreamsDB` and `GuardianDB` set `SQLiteStoreKernel._connection_quiescence`. For any other kernel store, `close_idle_connections` is a no-op; TASK-39047's review notes pin that on a `WorkspaceDB`.

**Kernel stores with no idle close at quit.**
- `WorkspaceDB` (`DB/Workspace_DB.py:18`), held as `local_workspace_db` (`app_service_wiring.py:2180`).
- `LibraryCollectionsDB` (`DB/Library_Collections_DB.py:25`), held as `local_library_collections_db` (`app_collections_wiring.py:695`).
- `AgentRunsDB` (`DB/AgentRuns_DB.py:307`), a per-runtime instance closed only on the calling thread.
- `RAGIndexingDB` (`DB/RAG_Indexing_DB.py:107`), opened per ingestion.
- `ClientNotificationsDB` (`Notifications/client_notifications_db.py:38`, `app_service_wiring.py:2612`).
- `WorkspaceProjectIndexDB` (`DB/Workspace_Project_Index_DB.py:194`).

`app_lifecycle.py` on dev references none of the first five. The project-index store is drained another way: at quit, `_shutdown_app_owned_lifecycles` calls `close_project_index_service` (`app_lifecycle.py:752-756`), which runs `service.aclose()` and `_factory.assert_retired()` (`Workspaces/project_index/runtime.py:299-314`). AC#1's per-store probe decides whether that drain also retires the store's worker-thread handles.

**Why it matters, and what is not yet known.**
- The kernel's `close` (`DB/store_kernel.py:287`) closes only the calling thread's handle.
- Guardian needed `check_same_thread=False` before a cross-thread close could work (TASK-39047's notes). Workspace, AgentRuns and Library Collections do not pass it.
- Whether these stores actually leave a live lease on a worker thread has not been measured.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 For each named store, a probe measures whether a pool-thread handle leaves a live lease after a real `on_unmount`; the result is recorded per store
- [x] #2 Each store that leaks has no live lease on any thread after `on_unmount`, with a test per store that is red on dev
- [x] #3 A straggler's write is never aborted and the quit close stays inside its time bound; the TASK-38955 and TASK-39047 quit tests stay green by name
<!-- AC:END -->

## Implementation Plan

1. Probe (AC#1): per store, open a handle on a kept pool thread, run the real `on_unmount` (`_unmount_owner`), and list the threads still holding a storage lease on the file.
2. For the stores the app holds (`local_workspace_db`, `local_library_collections_db`, `client_notifications_db`): reuse TASK-39047's mechanism. Each joins its file's `SQLiteConnectionQuiescenceRegistry` (one kernel helper) and opens with `check_same_thread=False`, and `_close_main_databases` closes them idle after the four main stores.
3. Tests in `Tests/DB/test_sync_log_retention.py` (already laned): a per-store pool-thread lease test (red on dev) and a per-store straggler-write test (red under the TASK-39047 control: both `close_if_idle` in-use checks removed). TASK-38955 and TASK-39047 quit tests stay green by name.
4. Record what the probe found for AgentRuns, RAG indexing and the project index, which the quit does not hold.

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
**Probe (AC#1).** For each store a handle was opened on a kept pool thread (a one-thread executor that stays alive across the quit). Then the real `on_unmount` ran (`_unmount_owner`), and the test listed the threads still holding a storage lease on the file (`_lease_threads`). On dev (pre-fix source, this branch's tests):

| Store | Owner that closes it at quit | Left on dev | After this change |
|---|---|---|---|
| `WorkspaceDB` | `_close_main_databases` (`local_workspace_db`) | `['MainThread', 'kept_0']` | `[]` |
| `LibraryCollectionsDB` | `_close_main_databases` (`local_library_collections_db`) | `['MainThread', 'kept_0']` | `[]` |
| `ClientNotificationsDB` | `_close_main_databases` (`client_notifications_db`) | `['kept_0']` (schema is deferred, so the UI thread opened none) | `[]` |
| `AgentRunsDB` | the console runtime's dispose, which the quit runs (`dispose_console_runtime`) | `['kept_0']` (a `get_run` on the pool; the dispose closed only its own thread's handle) | `[]` |
| `RAGIndexingDB` | the RAG ingest worker after each batch, and the backfill for a DB it opened | `['MainThread', 'kept_0']` after one ingest batch whose embedding path read the attached cache store on a pool thread; the worker had closed only its own thread's handle | `[]` |
| `WorkspaceProjectIndexDB` | `close_project_index_service` | `_unmount_owner` stubs `_shutdown_app_owned_lifecycles`, so the probe called `close_project_index_service` directly. Real service, registry and scanner; `enable`, a `status` on a pool thread, then a `refresh` held at publish. Leases before the close: `[('project-index-writer', 'workspace_project_index.db')]`. After the close: `[]`. | no leak, unchanged |

The project-index probe was a temporary test file, removed after the run.

**Approach (AC#2).** One mechanism (TASK-39047's), through one kernel helper:
- `SQLiteStoreKernel._join_connection_registry(db_path)` sets the file's `SQLiteConnectionQuiescenceRegistry` for a file-backed store; `:memory:` joins nothing. The kernel's own opener now passes `check_same_thread=False` once a store has joined (RAG indexing uses it). Stores with their own opener pass it themselves: Workspace, Library Collections, notifications, AgentRuns.
- `_close_main_databases` (`app_lifecycle.py`) runs the idle close for the three stores the app holds, after ChaChaNotes, media, Dreams and Guardian.
- `ConsoleRuntime.dispose` runs the AgentRunsDB idle close off the loop (`asyncio.to_thread`) before its existing own-thread `close()`, so agent-worker and receipt-read handles go too.
- `ingestion_indexing._retire_indexing_db` (idle close, then own-thread close, only for a real `RAGIndexingDB`) replaces the worker's per-batch own-thread close. `backfill_semantic_index` now retires a DB it opened itself in a `finally`. Its body moved unchanged into `_backfill_with`; before, that DB was never closed.
- A handle in use (transaction, core operation, acquisition) is never closed: the close is skipped and the handle closes with the process.
- The per-batch idle close also closes other threads' idle handles on the indexing file, for example a concurrent backfill's between statements. Its next call reopens through the getter.

**Red first.**
- `test_app_unmount_retires_a_worker_thread_handle_of_each_kernel_store[workspace|library_collections|notifications]`: red on dev with the leases above.
- `test_app_unmount_retires_every_agent_runs_handle_through_the_console_runtime`: `['kept_0'] == []` on dev.
- `test_rag_ingest_batch_leaves_no_indexing_handle_for_the_quit`: `['MainThread', 'kept_0'] == []` on dev.
- All green after.

**AC#3.**
- `test_app_unmount_kernel_store_close_never_aborts_a_straggler_write[...]` and `test_owner_idle_close_never_aborts_a_straggler_write[agent_runs|rag_indexing]`: a worker pauses inside a transaction across the close; both rows commit and the close returns in under 1 s. Green before and after; red 3 of 3 and 2 of 2 under TASK-39047's control (both in-use checks removed from `close_if_idle`): `ProgrammingError: Cannot operate on a closed database.`, 0 rows.
- TASK-38955 and TASK-39047 quit tests pass by name in the same file run: `test_app_pass_retires_the_handles_it_opens_on_its_worker_thread`, `test_app_unmount_retires_both_stores_on_every_thread`, `test_app_unmount_store_close_never_aborts_a_straggler_write`, `test_app_unmount_store_close_skips_a_busy_store_without_blocking`, `test_app_unmount_ui_close_bounds_its_checkpoint_wait`, `test_app_unmount_retires_a_worker_thread_handle_of_each_store[...]`, `test_app_unmount_store_close_never_aborts_a_straggler_write_in_each_store[...]`, `test_idle_close_refuses_while_a_getter_is_opening_a_handle[...]`, `test_kernel_idle_close_is_a_no_op_for_a_store_without_a_registry`, `test_app_unmount_store_close_skips_a_store_with_a_core_operation_running[...]`.
- The whole file: 49 passed serially.

**Lane.** `Tests/DB/test_sync_log_retention.py` is the last target of dreams-boundary "Run Dreams and Guardian boundary contracts" (`--storage-lease-guard`); no pin changed. That step through `jobstep.sh`, three times, with the round's first nodes: 1017 passed each, in 884.0 s, 607.7 s and 422.8 s at 1-minute load 48, 44 and 13. Its last CI run took 7.0 min (dev run 42296831798, job 11.1 min). Re-run once with this task's later nodes (AgentRuns, RAG indexing, the owner straggler cases): 1021 passed in 861.8 s at load 39, with no lease left under the guard.

**Covering run.** 78 files (RAG ingestion/backfill/indexing, AgentRuns and Agents, kernel-store, Console runtime ownership and the DB owner files) at `-n 4`: 1901 passed, 173 failed, 4 errors in 27.6 min (load 40-90). Every failing node was rerun per file on base `fu-base-r12`: 170 of 175 fail there too. On a rerun on this branch, 4 of the other 5 pass. The fifth, `test_forced_reindex_replaces_stale_vector_chunks_and_clears_query_cache`, fails on base too when run alone (twice on each tree). The earlier 54-file backup/DB/media run is in TASK-39110's and TASK-39118's notes.

**Review r12b-rev2.**
- **The registry grew with dead connections.** `console_change_review_read.read` (AgentRuns) and `app_collections_wiring`'s capture setup read (Library Collections) close the handle they acquired natively, without unregistering it, so the registry kept one dead connection per read for the session. `SQLiteConnectionQuiescenceRegistry.register` now forgets handles whose native close already happened (the `in_transaction` getter raises), which covers both callers and any later one.
  - Test `test_registry_stays_flat_when_a_finite_read_closes_its_own_handle[agent_runs|library_collections]` reproduces those reads: before the fix the counts were `[2, 3, 4]`; now they stay flat.
- **The backfill's `finally` had no test.** `test_backfill_retires_only_the_indexing_db_it_opened[self_opened|caller]`: a DB the backfill opened leaves no lease, and a caller's DB keeps its handle.
  - Red under two temporary edits: with no retire, `self_opened` failed; retiring the caller's DB as well, `caller` failed.

**Pins and size.**
- `Tests/DB/test_core_sqlite_owner_privacy.py`'s rag_indexing case expects `check_same_thread=False` now. It is red on dev for another reason: `RAG_Indexing_DB` has no `connect_private_sqlite`.
- Owner size waiver, dated rows: `app_lifecycle.py` 2203 -> 2207 and `console_runtime.py` 7069 -> 7072.

**Files.** `tldw_chatbook/DB/store_kernel.py`, `tldw_chatbook/DB/Workspace_DB.py`, `tldw_chatbook/DB/Library_Collections_DB.py`, `tldw_chatbook/Notifications/client_notifications_db.py`, `tldw_chatbook/DB/AgentRuns_DB.py`, `tldw_chatbook/DB/RAG_Indexing_DB.py`, `tldw_chatbook/Chat/console_runtime.py`, `tldw_chatbook/RAG_Search/ingestion_indexing.py`, `tldw_chatbook/app_lifecycle.py`, `Tests/DB/test_sync_log_retention.py`, `Tests/DB/test_core_sqlite_owner_privacy.py`, `Tests/Architecture/test_module_size_ratchet.py`.
<!-- SECTION:NOTES:END -->
