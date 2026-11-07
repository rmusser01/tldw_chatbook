---
id: TASK-39117
title: "The Workspace, Library collections, Agent runs, RAG indexing and notifications stores are not closed idle at quit, so a worker thread's handle and storage lease can outlive the app"
status: To Do
created_date: 2026-11-07 02:03
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

**Kernel stores with no idle close at quit.** `app_lifecycle.py` on dev never references any of them.
- `WorkspaceDB` (`DB/Workspace_DB.py:18`), held as `local_workspace_db` (`app_service_wiring.py:2180`).
- `LibraryCollectionsDB` (`DB/Library_Collections_DB.py:25`), held as `local_library_collections_db` (`app_collections_wiring.py:695`).
- `AgentRunsDB` (`DB/AgentRuns_DB.py:307`), a per-runtime instance closed only on the calling thread.
- `RAGIndexingDB` (`DB/RAG_Indexing_DB.py:107`), opened per ingestion.
- `ClientNotificationsDB` (`Notifications/client_notifications_db.py:38`, `app_service_wiring.py:2612`).
- `WorkspaceProjectIndexDB` (`DB/Workspace_Project_Index_DB.py:194`).

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
