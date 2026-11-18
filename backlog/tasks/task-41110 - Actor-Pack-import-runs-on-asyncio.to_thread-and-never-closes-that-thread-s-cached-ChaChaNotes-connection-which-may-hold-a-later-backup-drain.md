---
id: TASK-41110
title: "Actor Pack import runs on asyncio.to_thread and never closes that thread's cached ChaChaNotes connection, which may hold a later backup drain"
status: Done
assignee: ['@claude']
created_date: 2026-11-15 02:20
updated_date: 2026-11-15 18:25
labels:
- follow-up
- actor-packs
- backup
priority: medium
references:
- tldw_chatbook/Actor_Packs/import_controller.py
- tldw_chatbook/Backup_Recovery/participants.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Filed in round 14 of the 2026-10-23 burn-down (closeout, 2026-11-15). Checked against the round-14 stack top c7d4a41717 (on dev beab5c1e4e); line numbers are that tree's.

**Why.** A capture or restore pause waits for every native ChaChaNotes connection lease. If an idle executor thread keeps the connection an import opened, a backup started later can wait on it.

**Evidence.**
- `import_controller.py:246-248` runs inspect and activate through `asyncio.to_thread(execute)`, and `:350` runs cleanup the same way. Neither closes the thread's cached connection after the work.
- TASK-40304's notes: "the worker thread's own ChaChaNotes connection lease keeps a drain waiting"; its in-flight test closes that connection by hand in its publish hook.
- Other services close theirs after the thread work: `Writing_Interop/writing_scope_service.py:183-192` and `Research_Interop/research_scope_service.py:235-244` run `close_settled_core_cache` (`Backup_Recovery/participants.py`) after `to_thread`.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A test completes an Actor Pack import, then begins a capture pause and records whether the drain completes
- [x] #2 If it waits, the import releases its thread's cached connection when the work ends, as the interop services do, and the test is red before the fix
<!-- AC:END -->

## Implementation Plan

1. Add a test next to TASK-40304's activation tests (`Tests/Architecture/test_backup_owner_inventory.py`) that drives the real `ActorPackImportController` through inspect and activate on its own asyncio loop, then begins a local pause on the test thread, retires that thread's own caches (runtime maintenance's order) and records whether `pause.drain` completes.
2. Run it at base; if the drain waits, name the lease holder (thread and owner) in the notes.
3. If it waits: run each controller worker body (inspect/activate in `_start`, cleanup in `_shutdown_runner`) through `run_finite_local_worker`, the shared helper the other `asyncio.to_thread` finite workers use, so the thread's own cached ChaChaNotes connection closes when the work ends. Green after.
4. Negative control: revert the source, the test is red again. Covering runs: the Architecture file serially, `Tests/Actor_Packs` with `-n 4`.

## Implementation Notes

The drain waited. `test_actor_pack_import_leaves_no_worker_connection_for_a_later_drain` (`Tests/Architecture/test_backup_owner_inventory.py`) drives the real `ActorPackImportController` (inspect, then activate `create_new`, then `shutdown`) on its own loop, then begins a local pause on the test thread, retires that thread's own caches (runtime maintenance's order) and requires `pause.drain` within 5 s. At base (cc6006fcf5) it was red: `AssertionError: ChaChaNotes leases still held by threads ['asyncio_0']`, the default-executor thread that ran the import, after `asyncio.run` had already shut that executor down (a core handle outlives its thread).

Fix: `_start` now runs each operation body through `run_finite_local_worker` (`Backup_Recovery/participants.py`), as the other `asyncio.to_thread` finite workers do (Library share, Settings builders, scheduler handlers), and `_shutdown_runner` runs its cleanup the same way. It closes only the store caches the call itself opened on that thread, and leaves a transaction or a running operation alone. The interop services' `close_settled_core_cache` was not used: it is a drain-time hook for long-lived runtime producers, not a per-call close. Green after; the test also closes every thread's handle in its `finally` (`close_idle_connections`) so a red run leaks nothing into later Architecture tests.

Runs (load 34-53):
- Architecture file, `-k actor_pack`, serial: 9 passed. `test_actor_pack_boundary.py` + `test_application_services_bundle.py`, serial: 23 passed.
- `Tests/Actor_Packs -n 4`: 239 passed, 2 failed: `test_actor_pack_export.py::test_archive_matches_independent_golden_bytes[character|persona]`, also red with the base `import_controller.py` (same command; another round-15 stream owns that file).
- UI consumers `test_actor_pack_import_workflow.py`, `test_actor_pack_import_review.py`, `test_roleplay_import_app.py` with `-n 4`: 55 passed, 6 failed; all 6 are `test_actor_pack_import_review.py` (`ValueError: height and width must be > 0`, a dialog that never queries the controller), and the same 6 fail with the base source.
- `ruff_gate.py`: exit 0.

Files: `tldw_chatbook/Actor_Packs/import_controller.py`, `Tests/Architecture/test_backup_owner_inventory.py`.
