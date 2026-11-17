---
id: TASK-41115
title: "The persona later-rollback test's reopen child fails 'database is locked' on add_note about once in 50 fixture builds at load 36 or more"
status: In Progress
assignee: ['@claude']
created_date: 2026-11-15 02:20
updated_date: 2026-11-17 00:18
labels:
- follow-up
- backup
- tests
- flaky
priority: low
references:
- Tests/Backup_Recovery/test_created_persona_subtree_rollback.py
- Tests/Backup_Recovery/thread_diagnostics.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Filed in round 14 of the 2026-10-23 burn-down (closeout, 2026-11-15). Checked against the round-14 stack top c7d4a41717 (on dev beab5c1e4e); line numbers are that tree's.

**Why.** The file runs whole in the push-to-dev Backup_Recovery owner job (TASK-40307), where an intermittent red hides real ones.

**Evidence.** TASK-40306's notes: one reopen child in one `-n 4` run at load 36-57 failed `sqlite3.OperationalError: database is locked` on `add_note`, about 9 s after mount, under ChaChaNotes' 15 s busy timeout, so another connection held the write lock that long. The child now records `observe_notes_write_failure` (`test_created_persona_subtree_rollback.py:178-185`), which names the holding threads on the next failure.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The holder of the write lock is identified from a captured failure (local under load or CI) and recorded
- [ ] #2 The product or fixture cause is fixed, and the file passes 20 runs at -n 4 under load 40 or more
<!-- AC:END -->

## Implementation Plan

1. Capture: repeat the file's fixture-building nodes (`-k refuses_unproved_graph -n 4`, each worker builds the module fixture) in parallel lanes until a reopen child fails `database is locked`. The fixture's failure note keeps only the record's last 6000 characters, which cut the error code; replace it with a compact summary (wait time, SQLite error name, transaction entry, each thread's name and top frames).
2. Name the holder from the captured thread record, confirm the SQLite mechanism with a standalone probe, and record both.
3. Fix the product or fixture cause with a test red first where it can be made deterministic.
4. Run the whole file 20 times at `-n 4` with the load average at or above 40, and record each run's load and result.

## Implementation Notes

**AC1 (met): the holder.** Capture campaign (`-k refuses_unproved_graph -n 4`, each worker builds the module fixture; `-p no:timeout` so slow builds reach the reopen child): 64 fixture builds at load 64-130. One reopen child failed `sqlite3.OperationalError: database is locked` at the INSERT in `_add_note_with_cursor` (2026-11-15, lane B run 1, load 75-85). Its thread record (the old 6000-character tail, which cut the error code) showed no thread in a ChaChaNotes write transaction. The one ChaChaNotes thread busy at that moment was a finite worker in `base_db.run_owned_db_call` -> `invoke` -> `operation_owned_connection.__exit__` -> `close_connection` -> `_close_connection_handle` executing `PRAGMA wal_checkpoint(TRUNCATE)`. The others were Actor Pack recovery reading bundled Buddy assets (`ensure_builtin`, file reads), `automatic_work.recover` opening the AgentRuns DB, and idle pool threads. A TRUNCATE (like FULL/RESTART) checkpoint holds the WAL writer lock while it waits for readers, and `run_owned_db_call` gives it the connection's 15 s busy timeout, so a write that starts just after the closer begins loses the lock race at the end of its own 15 s. Confirmed without the app: a standalone probe had the writer started 0.4 s after the closer wait 15.2 s, and the new test below fails `database is locked` on `add_note` every time at base. The 48 builds after the diagnostics were improved produced no second failure.

**Fix (product).** `CharactersRAGDB._close_connection_handle` sets `busy_timeout = 0` before its close-time `wal_checkpoint(TRUNCATE)`: the checkpoint truncates when it can and otherwise returns at once, leaving a busy WAL to a later close (the existing `operation_deadline` path already bounded this wait for deadline callers). All ChaChaNotes closes route through that function. Line count unchanged (module size ratchet).

**Test.** `Tests/DB/test_operation_owned_connection_retirement.py::test_closing_worker_does_not_hold_other_writers_behind_a_live_reader` (a lane file): a raw reader holds a snapshot, a pool thread runs a finite read under `operation_owned_connection` (no deadline) and closes, and the test thread writes a note as soon as the worker enters `_close_connection_handle`. Red at base: `sqlite3.OperationalError: database is locked` from `add_note` (31 s). Green after; 5 of 5 separate runs at load 78-102 (9-16 s each); the whole file 20 passed.

**Fixture (diagnostics and budgets).** `_REOPEN` records the INSERT's wait and the thread names, and `_prepare_native` now attaches a compact summary (`_note_write_summary`: wait, error class and SQLite error name, transaction entry, every thread's top 12 frames) instead of the record's raw tail. The seed, capture and reopen children fail only once their process tree is idle for 60 s (cap 600 s), as TASK-40306 did for the replacement: at load 83-106 the capture child hit its fixed 120 s in every build. The two negative tests' pytest timeouts (600 s, which also cover the module fixture) are now 3600 s, above the fixture's summed child caps.

**AC2 (not met).** The whole file at `-n 4`, two lanes at once: 2 runs, load 70-119 (1-minute samples every 30 s), both `1 failed, 4 passed, 18 errors`. Every fixture-dependent node errored on the `persona-replacement` child reaching its 600 s cap while still busy, and the main test failed on its fixed 860 s budget; no run got as far as a lock. The ambient load from other sessions stayed at 70-110 all afternoon, so the 20 runs were stopped. See the report for the options.

Other runs:
- ChaChaNotes close covering run (`Tests/ChaChaNotesDB`, `Tests/DB`, four Backup_Recovery owner files, two pinning files; `-n 4`, load ~100): 4000 passed, 297 failed. The same 54 files at base (load ~80): 281 failed. Of the 20 nodes that failed only at head, the 4 that failed again in a second head run (`test_archive_recovery_query.py` cold-locked budgets, `test_private_sqlite.py::test_restore_fails_promptly...`, `test_sqlite_source_pin_lifetime.py::test_optional_sidecar...`) fail at base too at the same load (11 of 12 nodes), and none touches the TRUNCATE close (DELETE-mode or non-ChaChaNotes stores).
- `Tests/Architecture/test_module_size_ratchet.py`: 207 passed. `ruff_gate.py`: exit 0. Preflight: passed.

Files: `tldw_chatbook/DB/ChaChaNotes_DB.py`, `Tests/DB/test_operation_owned_connection_retirement.py`, `Tests/Backup_Recovery/test_created_persona_subtree_rollback.py`, `backlog/docs/lessons-testing-evidence.md`.
