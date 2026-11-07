---
id: TASK-39117
title: "The Workspace, Library collections, Agent runs, RAG indexing and notifications stores are not closed idle at quit, so a worker thread's handle and storage lease can outlive the app"
status: In Progress
assignee: ['@claude']
created_date: 2026-11-07 02:03
updated_date: 2026-11-07 22:47
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
- [ ] #1 For each named store, a probe measures whether a pool-thread handle leaves a live lease after a real `on_unmount`; the result is recorded per store
- [ ] #2 Each store that leaks has no live lease on any thread after `on_unmount`, with a test per store that is red on dev
- [x] #3 A straggler's write is never aborted and the quit close stays inside its time bound; the TASK-38955 and TASK-39047 quit tests stay green by name
<!-- AC:END -->

## Implementation Plan

1. Probe (AC#1): per store, open a handle on a kept pool thread, run the real `on_unmount` (`_unmount_owner`), and list the threads still holding a storage lease on the file.
2. For the stores the app holds (`local_workspace_db`, `local_library_collections_db`, `client_notifications_db`): reuse TASK-39047's mechanism. Each joins its file's `SQLiteConnectionQuiescenceRegistry` (one kernel helper) and opens with `check_same_thread=False`, and `_close_main_databases` closes them idle after the four main stores.
3. Tests in `Tests/DB/test_sync_log_retention.py` (already laned): a per-store pool-thread lease test (red on dev) and a per-store straggler-write test (red under the TASK-39047 control: both `close_if_idle` in-use checks removed). TASK-38955 and TASK-39047 quit tests stay green by name.
4. Record what the probe found for AgentRuns, RAG indexing and the project index, which the quit does not hold.

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
**Probe (AC#1).** Each store got a handle on a kept pool thread (`asyncio.to_thread` with a one-thread default executor that stays alive across the quit), then the real `on_unmount` ran through `_unmount_owner` with the store under its app attribute, and the threads still holding a storage lease on the file were listed (`_lease_threads`). On dev (pre-fix source, this branch's tests):

| Store | App holds it as | Left after `on_unmount` on dev | After this change |
|---|---|---|---|
| `WorkspaceDB` | `local_workspace_db` | `['MainThread', 'kept_0']` | `[]` |
| `LibraryCollectionsDB` | `local_library_collections_db` | `['MainThread', 'kept_0']` | `[]` |
| `ClientNotificationsDB` | `client_notifications_db` | `['kept_0']` (schema is deferred, so the UI thread opened none) | `[]` |
| `AgentRunsDB` | not held by the app (console runtime `_agent_runs_db` / bridge `runs_db`, plus per-call instances) | `['MainThread', 'kept_0']` (probe instance under an attribute the quit does not know) | unchanged: still leaks |
| `RAGIndexingDB` | not held by the app (opened per ingestion batch / backfill) | `['MainThread', 'kept_0']` (same probe shape) | unchanged: still leaks |
| `WorkspaceProjectIndexDB` | project-index service, drained by `close_project_index_service` | not measured: `_unmount_owner` stubs `_shutdown_app_owned_lifecycles`, and a real project-index runtime was out of reach this round | not measured |

The AgentRuns and RAG indexing rows come from a temporary parametrization of the new test, removed afterwards. They show only that the quit cannot reach an instance it does not hold; whether the production owners leave pool-thread handles was not measured.

**Approach (AC#2, partly).** TASK-39047's mechanism, reused for the three stores the app holds:
- `SQLiteStoreKernel._join_connection_registry(db_path)` (new, one helper) sets the file's `SQLiteConnectionQuiescenceRegistry` for a file-backed store; `:memory:` joins nothing.
- `WorkspaceDB`, `LibraryCollectionsDB` and `ClientNotificationsDB` call it in `__init__` before their first connection and open with `check_same_thread=False` (the idle close is the one cross-thread call, as for Guardian).
- `_close_main_databases` (`app_lifecycle.py`) runs the idle close for them after ChaChaNotes, media, Dreams and Guardian; a busy store is skipped and logged as before. An in-memory notifications fallback is skipped (`is_memory_db`).

**Red first.** `test_app_unmount_retires_a_worker_thread_handle_of_each_kernel_store[workspace|library_collections|notifications]` (`Tests/DB/test_sync_log_retention.py`): red on the pre-fix source with the leases in the table above; green after. `test_app_unmount_kernel_store_close_never_aborts_a_straggler_write[...]` (AC#3): green before (nothing was closed) and after; red 3 of 3 under TASK-39047's control (both in-use checks removed from `close_if_idle`): `ProgrammingError: Cannot operate on a closed database.`, 0 rows.

**AC#3.** The close returned in under 1 s in every straggler case (4-36 ms under the control). TASK-38955 and TASK-39047 quit tests pass by name in the same file run (45 passed serially): `test_app_pass_retires_the_handles_it_opens_on_its_worker_thread`, `test_app_unmount_retires_both_stores_on_every_thread`, `test_app_unmount_store_close_never_aborts_a_straggler_write`, `test_app_unmount_store_close_skips_a_busy_store_without_blocking`, `test_app_unmount_ui_close_bounds_its_checkpoint_wait`, `test_app_unmount_retires_a_worker_thread_handle_of_each_store[...]`, `test_app_unmount_store_close_never_aborts_a_straggler_write_in_each_store[...]`, `test_idle_close_refuses_while_a_getter_is_opening_a_handle[...]`, `test_kernel_idle_close_is_a_no_op_for_a_store_without_a_registry`, `test_app_unmount_store_close_skips_a_store_with_a_core_operation_running[...]`.

**Not done (AC#1 project index, AC#2 AgentRuns and RAG indexing).** The quit holds neither an `AgentRunsDB` nor a `RAGIndexingDB`; their owners (the console runtime's dispose, which closes only its own thread's handle; the RAG ingest worker, which closes its handle after each batch on its own thread) would have to run the idle close themselves once those stores join their registries. Proposed as a follow-up with the project-index probe. The task stays In Progress.

**Size.** `app_lifecycle.py` 2203 -> 2207 under the owner size waiver (dated row).

**Lane.** `Tests/DB/test_sync_log_retention.py` is the last target of dreams-boundary "Run Dreams and Guardian boundary contracts" (`--storage-lease-guard`); no pin changed. That step through `jobstep.sh`, three times, with this round's nodes: 1017 passed each, in 884.0 s, 607.7 s and 422.8 s at 1-minute load 48, 44 and 13. Its last CI run took 7.0 min (dev run 42296831798, job 11.1 min). No new node leaked a lease under the guard.

**Covering run.** 54 files (Backup_Recovery inventory, owner and data-group files, `Tests/UI/test_backup_data_groups.py`, the Workspace / Library Collections / notifications / store-kernel DB tests, `Tests/Media_DB`, the media and quiescence DB tests) at `-n 4` on this branch: 1379 passed, 27 failed, 3 skipped in 13 min (load ~40). Rerun per file on base `fu-base-r12`: 25 of the 27 fail there too. The other two were `test_core_owner_uses_literal_seam_and_preserves_connection_contract[media-first-create|media-harden]`, which pin media's exact connect kwargs; they now include the `factory`, as the ChaChaNotes case already does (TASK-39118). `test_every_durable_installed_adapter_has_a_named_selectable_group` was red on dev on `db.workspace_project_index` (missing from its non-selectable set), so it never reached the Personal Context owners; that row is added with them and the test passes.

**Files.** `tldw_chatbook/DB/store_kernel.py`, `tldw_chatbook/DB/Workspace_DB.py`, `tldw_chatbook/DB/Library_Collections_DB.py`, `tldw_chatbook/Notifications/client_notifications_db.py`, `tldw_chatbook/app_lifecycle.py`, `Tests/DB/test_sync_log_retention.py`, `Tests/Architecture/test_module_size_ratchet.py`.
<!-- SECTION:NOTES:END -->
