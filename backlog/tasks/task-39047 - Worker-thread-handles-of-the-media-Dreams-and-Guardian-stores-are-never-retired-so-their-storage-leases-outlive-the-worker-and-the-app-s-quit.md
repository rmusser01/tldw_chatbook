---
id: TASK-39047
title: "Worker-thread handles of the media, Dreams and Guardian stores are never retired, so their storage leases outlive the worker and the app's quit"
status: Done
assignee: ['@claude']
created_date: 2026-11-06 10:09
updated_date: 2026-11-06 18:27
labels:
- follow-up
- database
- lifecycle
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
- [x] #1 The owner picks the mechanism: a cross-thread registry with an idle close at quit (as ChaChaNotes has), a generic idle closer over the participant's connections, or retirement at each site. The decision and the measured per-job cost of retiring are recorded in this task's notes
- [x] #2 After a real `on_unmount`, media holds no live lease on any thread; the test first runs the real scope-picker tag lister on a pool thread kept alive across the quit, and is red before the fix (`[('media','pool_0')]` left)
- [x] #3 The same holds for a DreamsDB and a GuardianDB handle opened on a pool thread through `asyncio.to_thread`; each case is red before the fix
- [x] #4 The quit close never aborts a straggler's write and stays within its time bound: the TASK-38955 quit tests in `Tests/DB/test_sync_log_retention.py` stay green by name, and an equivalent straggler test exists for each newly retired store
- [x] #5 The owner decides whether worker handles that block a live backup's local pause are in scope here (TASK-31993.18 AC#2 keeps them blocking on purpose); the decision is recorded
<!-- AC:END -->

## Implementation Plan

1. Red tests in `Tests/DB/test_sync_log_retention.py`: the real scope-picker tag lister (media), and `asyncio.to_thread` reads of a DreamsDB and a GuardianDB, each on a pool thread kept alive across a real `on_unmount`; assert no live lease is left on the store's file on any thread.
2. Give MediaDatabase, DreamsDB and GuardianDB the file's `SQLiteConnectionQuiescenceRegistry` (`sqlite_connection_quiescence_registry`, as ChaChaNotes): register each held connection when it is published, unregister it on every close, and hold a registry acquisition across each getter so the idle close never closes a handle a getter is returning. The kernel (`store_kernel.py`) does this for any kernel store that keeps a registry; only Dreams and Guardian opt in.
3. `close_if_idle` also refuses while a counted core operation is running on the file (every media, Dreams and Guardian transaction is one), checked under the registry lock; each store gets `close_idle_connections()`.
4. `_close_main_databases` runs the idle close for ChaChaNotes, media, Dreams and Guardian, and leaves a busy store to close with the process.
5. Straggler tests (a worker writing transactions across the quit close) for media, Dreams and Guardian; the TASK-38955 quit tests stay green by name.

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
**Owner decision (2026-11-06), AC#1:** "Registry + idle quit close: give each store the same cross-thread SQLiteConnectionQuiescenceRegistry ChaChaNotes has, closed idle at quit. Proven pattern, no per-job cost, never aborts a straggler's write."

**Owner decision (2026-11-06), AC#5:** "Out of scope: keep TASK-31993.18's deliberate blocking; this task fixes quit-time and thread-exit leaks only; file a follow-up if a mounted live backup actually stalls on one." `monitor_app` is unchanged: a worker handle still blocks a live backup's local pause until it is retired. No mounted backup was run, so no stall was seen and no follow-up is filed.

**Measured per-job cost of retiring (AC#1).** These are the filing's measurements, on #3435 at load 26, over 100 jobs on the shared default executor. Retiring after every job cost 74 ms per job for ChaChaNotes and 60 ms for media. Keeping the handle cost about 0.1 ms. The chosen mechanism adds no per-job retirement: each held handle stays open until the quit's idle close.

**Approach.** This reuses TASK-38955's mechanism; there is no new close path.
- MediaDatabase, DreamsDB and GuardianDB now keep the file's `SQLiteConnectionQuiescenceRegistry` (`sqlite_connection_quiescence_registry`) as `_connection_quiescence`, as ChaChaNotes does.
  - Media registers each handle when its getter publishes it and unregisters it on every close (`Client_Media_DB_v2.py`).
  - The store kernel (`store_kernel.py`) does the same for any kernel store whose `_connection_quiescence` is set. Its class default is None, so other kernel stores are unchanged; Dreams and Guardian set it in `__init__`. Dreams' own `_close_connection`, which `close_all` also uses, unregisters too.
- `base_db.registry_acquisition` is a new decorator on the media and kernel getters. It holds a registry acquisition for the getter's whole call, as ChaChaNotes' getter does, so the idle close can never close a handle that a getter is about to return.
- `close_if_idle(repository)` now also refuses while a counted core operation runs on the file (`participants.core_operation_active`, read under the registry lock). Every media, Dreams and Guardian transaction is a core operation, and so is every kernel `connection()` block, so their writes are covered from before their getter runs. Each store gets `close_idle_connections()`.
- `_close_main_databases` runs the idle close for ChaChaNotes, media, Dreams and Guardian. An idle store then gets the bounded UI-thread close. A busy store is skipped and logged ("<Store> still in use at quit; its handles close with the process").
- Guardian now opens with `check_same_thread=False`, as Dreams already did. Without it the cross-thread close raised `ProgrammingError: SQLite objects created in a thread can only be used in that same thread`. Each thread still uses only its own handle.
- Because media now has `_connection_quiescence`, `operation_owned_connection` takes its registry branch for media, the same branch ChaChaNotes uses. The TASK-38955 retention-pass test covers that path and is green.

**Tests (`Tests/DB/test_sync_log_retention.py`), with red evidence.**
- `test_app_unmount_retires_a_worker_thread_handle_of_each_store[media|dreams|guardian]` (AC#2, AC#3). Each store opens a handle on a pool thread that stays alive across the quit: media through the real scope-picker tag lister, Dreams and Guardian through `asyncio.to_thread` reads. The quit then runs the real `on_unmount`, with its own hops on another executor. The test asserts that no live lease is left on the store's file on any thread. Red at base: media `['kept_0'] == []`; Dreams and Guardian `['MainThread', 'kept_0'] == []`.
- `test_app_unmount_store_close_never_aborts_a_straggler_write_in_each_store[...]` (AC#4). A writer pauses inside an open transaction across the quit close, then writes again and commits. The test asserts that both rows landed and the close took under 1 s. It is green at base, which closed none of these stores' worker handles. It was red 3 of 3 with a temporary edit that removed both the transaction check and the core-operation check from `close_if_idle`. I first tried a writer looping transactions, the shape of TASK-38955's test. It could not be made red: with both checks removed, it stayed green in 2 of 2 runs. The writer releases the GIL in COMMIT, so the close lands between its transactions, and its next getter then waits on the acquisition. TASK-38955's ChaChaNotes loop test also stayed green under that control, but for another reason. A ChaChaNotes transaction holds a registry use token for its whole length (`TransactionContextManager.__enter__` -> `begin_use`), so `_active_uses` refuses the close, and the control left that check alone.
- `test_app_unmount_store_close_skips_a_store_with_a_core_operation_running[media|dreams|guardian]`. A worker holds a raw handle inside a core operation between two reads, with no transaction open. The test asserts that both reads ran and that the logger got the "still in use" line. Red 3 of 3 with the core-operation check removed: `ProgrammingError: Cannot operate on a closed database`.
- `test_idle_close_refuses_while_a_getter_is_opening_a_handle[...]`. A raw getter call on a worker blocks in its connection open, and the test checks the idle close's verdict, first while the open is blocked, then after it finishes. Red 3 of 3 with `registry_acquisition` disabled: `(True, True) == (False, True)`.
- TASK-38955's quit tests stay green by name: `test_app_pass_retires_the_handles_it_opens_on_its_worker_thread`, `test_app_unmount_retires_both_stores_on_every_thread` (its owner stub is now the shared `_unmount_owner` helper, with its assertions unchanged), `test_app_unmount_store_close_never_aborts_a_straggler_write`, `test_app_unmount_store_close_skips_a_busy_store_without_blocking` and `test_app_unmount_ui_close_bounds_its_checkpoint_wait`.
- The whole file passes serially with `--storage-lease-guard` (36 passed).

**Review r11s-rev1.** `SQLiteStoreKernel.close_idle_connections` raised `AttributeError` on a kernel store that keeps no registry (any but Dreams and Guardian). It now returns True and closes nothing, since such a store tracks no other thread's handle. `test_kernel_idle_close_is_a_no_op_for_a_store_without_a_registry` (a WorkspaceDB) was red before with `AttributeError: 'NoneType' object has no attribute 'close_if_idle'`. The `needs-owner-decision` label is dropped, because both decisions are recorded above.

**Covering run.** I ran `Tests/DB`, `Tests/Dreams`, `Tests/Guardian`, 25 participant and owner files in `Tests/Backup_Recovery`, and five more covering files, at `-n 4` (load 26-37). Worktree: 273 failed, 4524 passed. Base `fu-base-r11`: 274 failed, 4511 passed. The failure sets differ only in nodes that flake on both trees: rerun twice on each tree, 3 or 8 of the same 20 nodes failed. The base failures are red on dev outside any lane, for example `LocalResearchService` has no `_connection`.

**Diagnostic inventory.** `app_lifecycle.py`'s row changed (68 -> 68 calls): the two ChaChaNotes-only quit-close statements became `f"Error closing {label} at quit: {type(exc).__name__}"` and `f"{label} still in use at quit; its handles close with the process"`. `label` is one of four fixed store names and the other value is an exception type name, so no user content, secret, path or URL reaches the sink. `Docs/security/production-diagnostic-inventory.json` was regenerated with `--write` after that review.

**Size waiver** (owner, 2026-11-06): `app_lifecycle.py` 2200 -> 2203 and `Client_Media_DB_v2.py` 9436 -> 9453 in `Tests/Architecture/test_module_size_ratchet.py`, each with a dated comment.

**Files.** `tldw_chatbook/DB/base_db.py`, `tldw_chatbook/DB/store_kernel.py`, `tldw_chatbook/DB/Client_Media_DB_v2.py`, `tldw_chatbook/DB/Dreams_DB.py`, `tldw_chatbook/DB/Guardian_DB.py`, `tldw_chatbook/Backup_Recovery/participants.py`, `tldw_chatbook/app_lifecycle.py`, `Tests/DB/test_sync_log_retention.py`, `Tests/Architecture/test_module_size_ratchet.py`.
<!-- SECTION:NOTES:END -->
