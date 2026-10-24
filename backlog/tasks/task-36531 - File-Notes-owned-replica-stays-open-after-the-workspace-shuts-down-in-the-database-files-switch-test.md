---
id: TASK-36531
title: File Notes owned replica stays open after the workspace shuts down in the database/files switch test
status: To Do
created_date: 2026-10-24 00:44
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

After `await workspace.shutdown()`, the test expects the workspace's owned replica to be closed, so that `owned_replica.list_deleted(...)` raises. It does not raise. Either the File Notes workspace leaves the replica it owns open after shutdown, which would leak a database connection on every Library database/files switch, or the test's ownership expectation is stale. Nobody has diagnosed it. The file runs on no PR lane, so this red reaches no PR run.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The failure is traced to a cause, recorded in the task notes
- [ ] #2 If the workspace leaves a replica it owns open, its shutdown closes it; if the expectation is stale, the test is corrected and the notes record why
- [ ] #3 test_library_database_files_switch_retains_workspace_and_database_canvas passes on dev
<!-- AC:END -->
