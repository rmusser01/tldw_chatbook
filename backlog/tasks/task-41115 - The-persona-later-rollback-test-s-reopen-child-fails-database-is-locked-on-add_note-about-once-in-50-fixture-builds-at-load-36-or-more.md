---
id: TASK-41115
title: "The persona later-rollback test's reopen child fails 'database is locked' on add_note about once in 50 fixture builds at load 36 or more"
status: In Progress
assignee: ['@claude']
created_date: 2026-11-15 02:20
updated_date: 2026-11-17 04:28
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

### Review r15ac-rev1, fix round 1

- **Finding 2 (Minor):** `base_db._operation_owned_connection` no longer lowers `busy_timeout` to the operation deadline before closing, and its docstring drops "A monotonic deadline also bounds the departing handle's checkpoint wait". The ChaChaNotes close checkpoint never waits now, and no other owned store checkpoints at close. `operation_deadline` was still accepted then (two archive-search call sites passed it; round 2 removed it). `test_owned_cleanup_deadline_bounds_checkpoint_with_live_reader` is removed; `test_closing_worker_does_not_hold_other_writers_behind_a_live_reader` covers a close behind a live reader. Runs: the DB retirement file, `test_store_kernel.py`, `test_archive_recovery_query.py`, `test_private_sqlite_lock_preservation.py`, `Tests/Dreams/test_connection_deadline.py` (`-n 4`): 78 passed, 1 failed. The failure is `test_copy_borrows_one_two_slot_envelope_for_sequential_preparations` (`ValueError: Unknown SQLite owner: settings.bulk_backup`), which fails the same way in the base worktree.
- **Finding 4 (fold-in):** `Tests/Backup_Recovery/test_persona_observation_budget.py`'s fake `_child` takes `idle=` and records it, and its pins name the current budgets: seed, capture, replacement and reopen at 600 s with the 60 s idle check; the later child idle-checked; the final reopen and capture fixed. The darwin case expected 150, but the main test applies the non-Windows rule there, so it is now 600. 12 failed at base and head (`TypeError: ... unexpected keyword argument 'idle'`); 12 passed now.
- AC#2 (20 runs at load 40 or more) stays open; the coordinator will schedule it at the end of the round.

### Review r15ac-rev2, fix round 2

- **Finding 3 (Minor):** `operation_owned_connection` and `_operation_owned_connection` no longer take `operation_deadline`. The two archive-search call sites stop passing it (`Chat/archive_recovery/search_service.py`). The workspace read still bounds its own connection with its existing `connection(operation_deadline=...)`, busy timeout and progress handler. The two tests that passed `operation_deadline=0` now test what remains: `test_cleanup_preserves_borrowed_timeout_and_transaction` and `test_cleanup_tolerates_already_retired_native_handle`. Round 1's note said "three callers"; there were two.
- Runs (`-n 4`, load 35-43): the archive-recovery coordinator, operations, search, dispatch and query files, the DB retirement file, `test_console_archive_recovery_flow.py`, `test_console_async_mcp_snapshot.py` and `test_ui_ready_module_census.py`: 59 failed and 254 passed after the removal, 58 failed and 255 passed with round 1's `base_db.py`, `search_service.py` and test file (same command). The one extra, `test_ui_ready_module_census_stays_at_the_pinned_size`, passed alone and twice more at `-n 4` (load 12-15). The other 58 are the same nodes in both runs: 23 in `test_console_archive_recovery_dispatch.py` and 32 in `test_console_async_mcp_snapshot.py` (`RuntimeError: original agent storage source changed`, a missing `ConsoleChatController` attribute, MCP catalog ownership asserts), plus 3 archive-recovery flow nodes.

### AC#2 attempt 2 (2026-11-16 20:12-20:27, coordinator-scheduled at load ~35-60)

Two lanes of `twenty.sh` (the whole file, `-n 4`, 1-minute load sampled every 30 s):

| run | secs | load min-max (mean) | result |
| --- | --- | --- | --- |
| A-1 | 911 | 38.4-67.5 (57.4) | 1 failed, 22 passed |
| B-1 | 918 | 38.3-67.5 (56.9) | 1 failed, 22 passed |

In both runs the one failure was the main test, `test_native_nested_persona_pack_survives_reopen_and_later_rollback`: its `persona-later` child reached the cap the test computes from its fixed 860 s budget (`timed out after 299 seconds`) while still busy. Setup had used the rest of the 860 s. No reopen child failed: the 5 fixture builds in each run (the module fixture on 4 workers, plus the main test's own) all passed `add_note`. The runs were stopped after the first pair because the budget fails the file at this load whatever the lock does. AC#2 is not met: it needs the main test's 860 s budget (and its 900 s pytest timeout) made load-tolerant, as TASK-40306 did for the child budgets. `test_persona_observation_budget.py` pins that arithmetic and would change with it.
