---
id: TASK-36531
title: File Notes owned replica stays open after the workspace shuts down at quit in the database/files switch test
status: Done
assignee:
- '@claude'
created_date: 2026-10-24 00:44
updated_date: 2026-11-03 21:44
labels:
- follow-up
- library
- tests
priority: medium
references:
- Tests/UI/test_library_file_notes_workspace.py
dependencies:
- TASK-36383
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by TASK-36383 (PR #3232, 2026-10-23), re-checked on dev 5b5920b8e2. Split out of TASK-36528, which keeps that file's contrast and 40-column reds. `Tests/UI/test_library_file_notes_workspace.py::test_library_database_files_switch_retains_workspace_and_database_canvas` fails at :7392-7393 with "DID NOT RAISE sqlite3.ProgrammingError".

Once the app has exited, the test expects the workspace's owned replica to be closed, so that `owned_replica.list_deleted(...)` raises. The workspace closes it only in `shutdown()` (`_close_owned_replica`), and the app's exit has already run that, so the test's own `await workspace.shutdown()` returns at once. It does not raise. Either the File Notes workspace leaves the replica it owns open after it shuts down at quit, which would leak one database connection per workspace lifetime, or the test's ownership expectation is stale. Nobody has diagnosed it. The file runs on no PR lane, so this red reaches no PR run.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The failure is traced to a cause, recorded in the task notes
- [x] #2 If the workspace leaves a replica it owns open, its shutdown closes it; if the expectation is stale, the test is corrected and the notes record why
- [x] #3 test_library_database_files_switch_retains_workspace_and_database_canvas passes on dev
<!-- AC:END -->

## Implementation Plan

ADR required: no
ADR path: N/A
Reason: Thread affinity of one owned resource inside the workspace; no architectural change.

1. Traced on eba3ad14b5 with logging in `shutdown` and `_close_owned_replica`: the shutdown at quit runs to completion and calls `_close_owned_replica`. The replica's connection was opened on the default-executor thread that ran `_build_runtime` (`asyncio_2`), and the close ran on another (`asyncio_8`). `FileNotesReplica.close()` (since b5251e9a6e) retires a file-backed connection only on its creating thread and silently refuses a foreign-thread close, so the connection stays open: one leaked connection per workspace lifetime.
2. The test's probe is stale as well: after a successful close a file-backed replica sets `_connection = None` and reopens on next use, so `owned_replica.list_deleted(...)` can never raise `ProgrammingError`.
3. Fix: open the owned replica on a one-thread executor the workspace keeps, and run `_close_owned_replica` on that executor at shutdown, then retire it.
4. Test: assert the retired native handle is closed (`_connection is None`, and the handle captured before quit raises `ProgrammingError`) instead of reusing the replica; show it red without the fix.
5. 8+ repeat runs of the test; diagnose the earlier-reported `_library_note_detail is None` failure (about :7240) separately and report it rather than widening the task.

## Implementation Notes

**Cause (AC#1).** Traced on eba3ad14b5 with temporary logging in `shutdown()` and `_close_owned_replica()`. The quit-time shutdown, from `LibraryScreen.on_unmount`, runs to completion and calls `_close_owned_replica()`. The owned replica's connection had been opened on the default-executor thread that ran `_build_runtime` (`asyncio_2`). The close ran on a different pool thread (`asyncio_8`), and `FileNotesReplica.close()` refuses that silently: since b5251e9a6e it retires a file-backed connection only on its creating thread (`_core_closing`: "foreign-thread close is refused"). After shutdown the connection still answered `SELECT 1`. So the workspace leaked one open connection per lifetime, and both halves of AC#2 apply:
- The workspace left a replica it owns open.
- The test's probe was stale too. After a successful close, a file-backed replica sets `_connection = None` and reopens on its next use, so `owned_replica.list_deleted(...)` can never raise. Negative control with the fix in place and the old probe kept: the new asserts pass, then the old line fails "DID NOT RAISE".

**Fix (AC#2).** An owned replica is now created on a one-thread executor the workspace keeps (`_replica_thread`, created lazily in `_build_runtime`). `shutdown()` runs `_close_owned_replica` on that executor and then retires it. Replicas passed in by callers are untouched.

**Test.** `test_library_database_files_switch_retains_workspace_and_database_canvas` captures the native handle after init. After quit and `shutdown()` it asserts `owned_replica._connection is None` and that the captured handle raises `ProgrammingError`, instead of reusing the replica. Red without the fix: `assert <AdmittedConnection ...> is None`. Before the change the test failed 8 of 8 runs at the filed spot ("DID NOT RAISE").

**AC#3 evidence and a separate cause.** With the fix the test passed 4 of 4 serial runs (load 29-34) and 4 of 4 runs four at a time (load ~37). It failed 8 of 8 four-at-a-time runs at load 37-42, but at the earlier line `assert screen._library_note_detail is not None` (now :7351), the "about :7240" failure in this task's triage. That failure has a different cause, and this task does not widen to cover it:
- The test holds the Database detail fetch (`release_detail`) across the whole Files mount and initial scan. `_refresh_library_note_detail` bounds that load with `LIBRARY_NOTE_LOAD_DEADLINE_SECONDS = 3.0` (task-32050, 15b7293976).
- Instrumented under that load, every failing run held the fetch 3.95-4.90 s and logged the deadline timeout, which leaves no session (`_library_note_detail` None).
- The same four-at-a-time runs with the deadline patched to 30 s passed 4 of 4.
- 4c7d47e45f (TASK-34000.28) touches neither that path nor the deadline, so this is a test load race, not that regression. It is reported as a proposed follow-up.

**Also.** Backup_Recovery workspace tests (`test_file_notes_maintenance.py`, `test_unsaved_editors.py`, `test_notes_recovery_controls.py`): 116 passed. Module size ratchet re-pinned 9568 -> 9582 with a dated comment.

**Files.** `tldw_chatbook/Widgets/Library/library_file_notes_workspace.py`, `Tests/UI/test_library_file_notes_workspace.py`, `Tests/Architecture/test_module_size_ratchet.py`.
