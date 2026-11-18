---
id: TASK-38955
title: "App quit leaves ChaChaNotes and media connections open, including the sync-log retention pass's pool-thread handles"
status: Done
assignee: ['@claude']
created_date: 2026-11-05 21:35
updated_date: 2026-11-06 07:12
labels:
- follow-up
- database
- lifecycle
priority: medium
references:
- tldw_chatbook/app_lifecycle.py
- tldw_chatbook/app_shutdown.py
- tldw_chatbook/DB/base_db.py
- Tests/DB/test_sync_log_retention.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in round 8 of the 2026-10-23 burn-down; checked on dev 29592ecc9b, 2026-11-05. TASK-34389 (retention) and TASK-38714 (wrong-thread closes) are both Done, and neither covers a connection that is opened and never closed, or the missing close at shutdown.

**Retention pass.**
- `_perform_sync_log_retention` (`tldw_chatbook/app_lifecycle.py:1510-1555`) runs `asyncio.to_thread(db.delete_sync_log_entries_older_than, cutoff)` for both DBs (`:1543-1545`) and never retires the connection that call opens on the pool thread.
- `perform_media_cleanup` (`:1567-1576`) does it properly: it calls `close_connection()` in a `finally` on the same thread. The repo's helpers for this are `run_owned_db_call` (`tldw_chatbook/DB/base_db.py:140`) and `operation_owned_connection` (`:45`).
- The pass runs at startup and then every 24 h (`app_lifecycle.py:1433-1439`).
- A probe through the scratch-profile wrapper: one pass raised the ChaChaNotes `registered_connection_count()` from 1 to 2, and `private_sqlite._ordinary_connections` from 2 to 4 leases, all held on the default-executor thread.

**Quit.**
- Of the DB owners, `on_unmount` (`app_lifecycle.py:1022-1306`), `_shutdown_app_owned_lifecycles` and `tldw_chatbook/app_shutdown.py` close only the ingest-jobs store, the writing service and the research service. They also close the chunking lab, workflow authoring, MCP saves and SSH masters, none of which is a ChaChaNotes or media DB.
- Nothing calls `quiesce_connections` or `close_connection` on `chachanotes_db`, or `close_connection` on `media_db`. A grep outside `DB/` finds no caller at shutdown.
- `Subscriptions_DB.py:266-295` records a measurement that a clean exit leaves no `-wal` file. So the criteria are about connections and leases being retired at a defined moment, not about WAL loss.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 After a sync-log retention pass, neither DB keeps a connection or lease that the pass opened; a test in Tests/DB/test_sync_log_retention.py compares registered connections and leases before and after and is red on dev (count 1 to 2, leases +2)
- [x] #2 After app unmount, ChaChaNotes leaves no registered connection or ordinary lease on any thread, and media leaves none on the UI thread; a test is red on dev. Media pool-thread handles depend on each worker retiring its own (follow-up)
- [x] #3 The quit path stays bounded: the closes run after workers settle and never hold the exit past the shutdown watchdog
<!-- AC:END -->

## Implementation Plan

1. Red test in `Tests/DB/test_sync_log_retention.py`: one retention pass on a kept-alive executor thread, comparing registered connections and live leases before and after.
2. Run each prune under `operation_owned_connection` on its worker thread.
3. Red test that drives the real `LifecycleMixin.on_unmount` with real stores holding handles on the UI thread and a live pool thread; then close both stores at the end of unmount, after workers and services settle: ChaChaNotes quiescence (all threads, on a thread, bounded) and each store's UI-thread close.
4. Show the close stays bounded and keeps the loop live when a ChaChaNotes user never finishes.

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
2026-11-05, evidence added at filing (round 8, commit 3). Priority raised from low to medium because the leftover retention-pass lease also blocks a backup pause.

**Probe.** Run with the `Tests/DB/test_sync_log_retention.py` fixtures, one case per process, through the scratch-profile wrapper:
- One retention pass, then closes on the main thread: 2 storage leases were left on asyncio_0, and `_begin_local_pause().drain()` with a 1 s budget returned False after 1.00 s.
- Control, the same pass plus `close_connection` on the worker thread: 0 leases were left, and the drain returned True in 0.00 s.

**Effect during a real backup (inferred from code, not run in a mounted app).**
- `monitor_app` (`runtime_maintenance.py:698-731`) calls `retire_current_thread_caches` (`participants.py:113`), which closes only the loop thread's caches.
- It then retries `pause.drain` until its 30 s deadline and raises `RecoveryRequired("runtime_native_resources_not_settled")`.
- So a retention pass that ran since startup could make the next backup fail to settle.

**Implementation (2026-11-06, @claude).**
- **Retention (AC#1).** `_perform_sync_log_retention` (`tldw_chatbook/app_lifecycle.py`) now runs each store's prune inside `operation_owned_connection(db)` on its worker thread, so the handle the prune opens is retired on that same thread.
- **Quit (AC#2, AC#3).** The new `_close_main_databases` is the last step of `on_unmount`. It runs after `_drain_app_unmount_services` (worker cancel and settle), the ingest pool shutdown, the ingest store close, and the writing and research service closes.
  - ChaChaNotes: `close_idle_connections()` runs through `asyncio.to_thread`. It closes the registered handles on every thread, but only when none is in use; a busy store is skipped and its handles close with the process. Superseded the first version's waiting quiescence, see "Review round 1" below.
  - Each store not skipped then runs `close_connection()` on the UI thread, with that handle's checkpoint wait bounded by `_QUIT_CLOSE_BUSY_SECONDS` (0.5 s; the exit watchdog defaults to 120 s).
  - In-memory stores are skipped: they hold no lease, and closing them discards their data.
  - Failures are logged by type and never abort unmount.
- **Media scope.** Media has no cross-thread registry, and the participants layer deliberately never closes a foreign thread's handle (`retire_current_thread_caches`). So media handles on pool threads still depend on each worker retiring its own, as the retention pass now does. The other `to_thread(<db>.<method>)` sites were not audited.
- **TASK-38972 trap.** The UI-thread ChaChaNotes close after quiescence hits the dead cached handle and logs a ProgrammingError. The TASK-38972 commit fixes this at `close_connection`.

**Tests (`Tests/DB/test_sync_log_retention.py`).**
- `test_app_pass_retires_the_handles_it_opens_on_its_worker_thread` (AC#1). Red on dev: `(1, 2, 2) == (1, 1, 0)` (registered count 1 to 2, +2 leases). Green after.
- `test_app_unmount_retires_both_stores_on_every_thread` (AC#2). It drives the real `LifecycleMixin.on_unmount` with a stub owner around real stores: ChaChaNotes handles on the UI thread and on a live pool thread, and a media handle on the UI thread. It records the handle count when the workers settle (still 2), so it also pins that the stores close after them. Red against base's `app_lifecycle.py` (e8eeae7db9): `((2, 3), [2]) == ((0, 0), [2])` (registered handles, live leases; the count when workers settled). Green after. Correction: the first version of this test wrapped `owner._close_main_databases`, so on base it errored with AttributeError instead of failing an assertion. The `(2, 3) == (0, 0)` this note used to cite came from an earlier draft.
- The first version's AC#3 test (`..._is_bounded_and_keeps_the_loop_live`) was replaced in review round 1 by the three tests below.

**Covering comparison, run against the base worktree.** Targets: `Tests/DB`, `Tests/ChaChaNotesDB`, `Tests/App/test_app_shutdown.py`, `Tests/TTS/test_tts_app_ownership.py`, `Tests/Backup_Recovery/test_speech_initialization_maintenance.py`, `Tests/Research/test_app_research_wiring.py`, `Tests/Web_Server`, `Tests/Chat/test_console_trace_compaction_admission.py`, with `-n 4`.
- Base: 298 failed, 3588 passed. Here: 301 failed, 3589 passed; the run includes 4 new tests.
- The three failures only here were `test_physical_drain_callback_persists_only_after_delayed_worker_finishes[False]`, `test_actual_pin_death_refuses_backup_and_preserves_sibling_lock[before_backup]` and `test_normal_and_early_failure_cycles_do_not_grow_children_or_fds`. Isolated, they pass 6 of 6 on both trees, so they were load flakes; this run was at load about 50.

**UI and app-boot comparison.** The quit close runs on every real app exit, so these suites boot the app and assert native handle counts: `Tests/UI/test_console_fixture_ownership.py`, `test_console_finite_read_ownership.py`, `test_library_inspection_admission.py`, `test_console_received_intent_feedback.py`, `test_personas_workbench.py`, `test_roleplay_character_conversation_browse.py`, `test_console_selection_notes.py`, `test_console_archive_recovery_flow.py`, `test_personas_pending_save_drafts.py`, `Tests/App/test_worker_failure_event.py`, and `Tests/Backup_Recovery/test_recovery_restart.py` plus `test_console_config_sync_lifetime.py`.
- Base: 16 failed, 611 passed, 49 errors (local RecoveryRequired reds).
- Here: 15 failed, 612 passed, 49 errors.
- No failure appears only here.

**Size.** `app_lifecycle.py` went from 2147 to 2189 lines, raised under the owner waiver; review round 1 took it to 2200.

**Review round 1 (r10st-rev1, 2026-11-06).** The first version's quit close used `quiesce_connections(timeout_seconds=5)`. While a barrier waits, every new acquisition and cursor use raises `database_maintenance_in_progress`, and so does each statement inside an already-open transaction. Workers can outlive the unmount's 3 s cancel grace, and the barrier aborted their writes: data loss. Base has no barrier. The fixes:
- **No barrier at quit.** New `SQLiteConnectionQuiescenceRegistry.close_if_idle()` (`DB/base_db.py`), exposed as `CharactersRAGDB.close_idle_connections()`. It closes every registered handle while holding the registry lock, and only if no barrier, acquisition, managed use or native transaction is active. Otherwise it closes nothing and returns False. It never sets the barrier token, so work arriving meanwhile waits for the lock and then reopens, instead of raising. A busy store is skipped entirely, with an info log, and its handles close with the process.
  - Why not the ruled `timeout_seconds=0`: `begin_quiescence(0)` does fail at once when work is active. But when it succeeds, the token stays set through `close_registered`, and a straggler idling between transactions then meets the barrier on its next statement.
  - Under a temporary edit, that variant failed the new looping-writer test in 13 of 15 runs (`database_maintenance_in_progress`). The committed version passed 15 of 15.
- **Bounded UI-thread close.** Before each store's UI-thread `close_connection()`, the handle's `busy_timeout` is set to `_QUIT_CLOSE_BUSY_SECONDS`, so its `wal_checkpoint(TRUNCATE)` waits at most 0.5 s on another writer's lock. This is the same idea as `operation_owned_connection(..., operation_deadline=)`. Before the fix, the reviewer measured a 15.63 s loop gap.
- **AC#2 narrowed by the coordinator (filer's ruling).** Media has no cross-thread registry, and the codebase deliberately never closes another thread's handle. That needs design, so AC#2 now requires ChaChaNotes clean on every thread and media clean on the UI thread. Media pool-thread handles depend on each worker retiring its own; for example, `Chat/scope_picker_listers.py:519` calls `to_thread(media_db.get_keyword_usage_stats)`. A follow-up will track this.
- **New tests.**
  - `test_app_unmount_store_close_never_aborts_a_straggler_write`: a straggler loops two-statement transactions across the close, and every statement must commit. Red at f9674b9293: `CharactersRAGDBError: database_maintenance_in_progress`.
  - `test_app_unmount_store_close_skips_a_busy_store_without_blocking`: a straggler mid-transaction holds the WAL write lock. The close must take under 1 s with a loop gap under 1 s, and the writer must commit. Red at f9674b9293: close 19.94 s, loop gap 14.95 s.
  - `test_app_unmount_ui_close_bounds_its_checkpoint_wait`: a foreign connection holds the write lock, and the idle close is stood in by `lambda: True`, as for a UI handle opened during that await. The close and the loop gap must stay within the bound + 1 s. Red with the busy bound removed (temporary edit): 15.51 s, 15.53 s.
  - All four quit tests passed together 15 of 15 runs at load 25-31.
- **Reviewer probes, after.**
  - `r10st-rev1-quitprobe.py` with 0 and 2000 rows: the close took 15 ms and 7.6 ms; ChaChaNotes ended at 0 registered and 0 leases; one media lease was left on the pool thread, which the narrowed AC#2 excludes.
  - `stragglerprobe.py` and `-tb.py`: `{'a': 'committed', 'b': 'saved'}`, no errors.
  - `writerprobe.py 20`: close 0.00 s, max loop gap 0.02 s.

**Diagnostic inventory (reviewed before `--write`).** Two error calls were added at quit. Each interpolates only `type(exc).__name__`. In review round 1, re-pinned after a second review: "Error quiescing ChaChaNotes at quit" was replaced by "Error closing ChaChaNotes at quit" (type name only), and a fixed-text info log was added: "ChaChaNotes still in use at quit; its handles close with the process".
<!-- SECTION:NOTES:END -->
