---
id: TASK-39117
title: "The Workspace, Library collections, Agent runs, RAG indexing and notifications stores are not closed idle at quit, so a worker thread's handle and storage lease can outlive the app"
status: In Progress
assignee: ['@claude']
created_date: 2026-11-07 02:03
updated_date: 2026-11-07 06:43
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
- [ ] #3 A straggler's write is never aborted and the quit close stays inside its time bound; the TASK-38955 and TASK-39047 quit tests stay green by name
<!-- AC:END -->

## Implementation Plan

1. Probe (AC#1): per store, open a handle on a kept pool thread, run the real `on_unmount` (`_unmount_owner`), and list the threads still holding a storage lease on the file.
2. For the stores the app holds (`local_workspace_db`, `local_library_collections_db`, `client_notifications_db`): reuse TASK-39047's mechanism. Each joins its file's `SQLiteConnectionQuiescenceRegistry` (one kernel helper) and opens with `check_same_thread=False`, and `_close_main_databases` closes them idle after the four main stores.
3. Tests in `Tests/DB/test_sync_log_retention.py` (already laned): a per-store pool-thread lease test (red on dev) and a per-store straggler-write test (red under the TASK-39047 control: both `close_if_idle` in-use checks removed). TASK-38955 and TASK-39047 quit tests stay green by name.
4. Record what the probe found for AgentRuns, RAG indexing and the project index, which the quit does not hold.
