---
id: TASK-39047
title: "Worker-thread handles of the media, Dreams and Guardian stores are never retired, so their storage leases outlive the worker and the app's quit"
status: In Progress
assignee: ['@claude']
created_date: 2026-11-06 10:09
updated_date: 2026-11-06 14:55
labels:
- follow-up
- database
- lifecycle
- needs-owner-decision
priority: medium
references:
- tldw_chatbook/app_lifecycle.py
- tldw_chatbook/DB/Client_Media_DB_v2.py
- tldw_chatbook/DB/Dreams_DB.py
- tldw_chatbook/DB/Guardian_DB.py
- tldw_chatbook/DB/base_db.py
- tldw_chatbook/DB/ChaChaNotes_DB.py
- tldw_chatbook/Backup_Recovery/participants.py
- tldw_chatbook/Backup_Recovery/runtime_maintenance.py
- tldw_chatbook/Chat/scope_picker_listers.py
- tldw_chatbook/Utils/text_selection_crash_guard.py
- Tests/DB/test_sync_log_retention.py
dependencies:
- TASK-38955
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in round 10 of the 2026-10-23 burn-down; checked on dev b2c618185f, 2026-11-06, and on storage PR #3435 (head 3cc391fadc), which lands TASK-38955. Line numbers are dev's; ones marked #3435 are code that PR adds. TASK-38955 narrowed its AC#2 to ChaChaNotes and says "A follow-up will track this" and "The other `to_thread` sites were not audited"; TASK-38986 says the same of the other offload sites.

**What TASK-38955 does for ChaChaNotes.**
- ChaChaNotes has a per-file `SQLiteConnectionQuiescenceRegistry` (`DB/base_db.py:180`), created at `ChaChaNotes_DB.py:2919`.
- `close_idle_connections` (#3435 `ChaChaNotes_DB.py:3258`) closes every thread's handle through `close_if_idle` (#3435 `base_db.py:355-384`).
- That runs from `_close_main_databases` (#3435 `app_lifecycle.py:437`), at the end of `on_unmount`.

**Media has no such path.**
- Media keeps one connection per thread (`self._local = threading.local()`, `Client_Media_DB_v2.py:1110`).
- `close_connection` closes only the calling thread's handle; `core_closing` refuses a close from another thread (`participants.py:482`). At quit #3435 closes media on the UI thread only (#3435 `app_lifecycle.py:447`, `:462-472`); dev closes neither store's worker handles.
- `participant.connections[connection] = lease` (`participants.py:563`) is a plain dict, so a pool thread's handle and its lease outlive the thread.
- A probe ran the real scope-picker tag lister (`build_keyword_tag_lister`, `asyncio.to_thread(media_db.get_keyword_usage_stats)` at `scope_picker_listers.py:519`) on a pool thread, then the real `_close_main_databases`. Media's pool-thread lease survived the quit close and the thread's exit: `[('media','pool_0')]`. A 1 s local-pause drain then returned False.

**The same holds for other stores.** An AST census of #3435's tree found 223 (dev has the same 223) `asyncio.to_thread` offloads that reach a DB: 139 call `<db>.<method>` directly and 84 pass a DB to a helper. A probe made one call per class on a pool thread and counted leases:

| Store | Offload sites | Worker lease left behind | Retired at quit on #3435 |
|---|---|---|---|
| DreamsDB | 65 | yes, outlives the thread | no; the app-lifetime `dreams_db` (`app_service_wiring.py:3346`) is never closed, and `close_all` (`Dreams_DB.py:410`) is only called for the reconcile's temporary instance |
| MediaDatabase | 8, plus thread workers | yes, outlives the thread | UI thread only |
| GuardianDB | 0 unwrapped today | yes, if a call is unwrapped | no (`store_kernel.close` is calling-thread only) |
| CharactersRAGDB | 48 | yes, outlives the thread | yes (TASK-38955) |
| SubscriptionsDB | 52 | only while the thread lives | implicit: `_ThreadExitCleanup` (`Subscriptions_DB.py:198`) |
| ScheduledTasksDB | 34 | no: one connection per operation | not needed |

- Media handles are also opened by thread workers that never retire them, for example:
  - `Widgets/Library/library_media_canvas.py:654` and `library_media_viewer.py:1039` (`run_worker(thread=True)`);
  - `library_screen.py:20535` and `:20946` (`@work(thread=True)`);
  - `RAG_Search/pipeline_functions_simple.py:203/218/231` and `Dreams/cycle_service.py:583/1218` (`asyncio.to_thread`).

**Design constraints.**
- Every `asyncio.to_thread` call and every thread worker runs on one default executor, `FreshContextExecutor` (`Utils/text_selection_crash_guard.py:172`, installed at `:284`). That seam is shared, but retiring after every job measured 74 ms per job for ChaChaNotes and 60 ms for media, against about 0.1 ms with the handle kept (100 jobs, load 26). Per-job retirement there is not viable for hot sites.
- `run_owned_db_call` does not include `MediaDatabase` (`base_db.py:161`), so routing a media call through it retires nothing; `operation_owned_connection` does handle media (`base_db.py:86`).
- Live backup: `monitor_app` (`runtime_maintenance.py:698-733`) retires only the loop thread's caches. TASK-31993.18 AC#2 deliberately keeps other threads' handles blocking the pause. So a leftover worker handle should make a live backup's local pause fail to settle. This is inferred from the code and the 1 s drain probe; no backup was run in a mounted app.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The owner picks the mechanism: a cross-thread registry with an idle close at quit (as ChaChaNotes has), a generic idle closer over the participant's connections, or retirement at each site. The decision and the measured per-job cost of retiring are recorded in this task's notes
- [ ] #2 After a real `on_unmount`, media holds no live lease on any thread; the test first runs the real scope-picker tag lister on a pool thread kept alive across the quit, and is red before the fix (`[('media','pool_0')]` left)
- [ ] #3 The same holds for a DreamsDB and a GuardianDB handle opened on a pool thread through `asyncio.to_thread`; each case is red before the fix
- [ ] #4 The quit close never aborts a straggler's write and stays within its time bound: the TASK-38955 quit tests in `Tests/DB/test_sync_log_retention.py` stay green by name, and an equivalent straggler test exists for each newly retired store
- [ ] #5 The owner decides whether worker handles that block a live backup's local pause are in scope here (TASK-31993.18 AC#2 keeps them blocking on purpose); the decision is recorded
<!-- AC:END -->

## Implementation Plan

1. Red tests in `Tests/DB/test_sync_log_retention.py`: the real scope-picker tag lister (media), and `asyncio.to_thread` reads of a DreamsDB and a GuardianDB, each on a pool thread kept alive across a real `on_unmount`; assert no live lease is left on the store's file on any thread.
2. Give MediaDatabase, DreamsDB and GuardianDB the file's `SQLiteConnectionQuiescenceRegistry` (`sqlite_connection_quiescence_registry`, as ChaChaNotes): register each held connection when it is published, unregister it on every close, and hold a registry acquisition across each getter so the idle close never closes a handle a getter is returning. The kernel (`store_kernel.py`) does this for any kernel store that keeps a registry; only Dreams and Guardian opt in.
3. `close_if_idle` also refuses while a counted core operation is running on the file (every media, Dreams and Guardian transaction is one), checked under the registry lock; each store gets `close_idle_connections()`.
4. `_close_main_databases` runs the idle close for ChaChaNotes, media, Dreams and Guardian, and leaves a busy store to close with the process.
5. Straggler tests (a worker writing transactions across the quit close) for media, Dreams and Guardian; the TASK-38955 quit tests stay green by name.
