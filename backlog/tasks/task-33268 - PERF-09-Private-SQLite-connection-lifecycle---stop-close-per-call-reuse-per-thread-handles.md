---
id: TASK-33268
title: 'PERF-09: Private-SQLite connection lifecycle - stop close-per-call, reuse
  per-thread handles'
status: To Do
created_date: 2026-09-28 18:02
dependencies:
- TASK-33260
labels:
- performance
- database
- perf-audit-2026-09
priority: high
references:
- qa/perf-structural-audit-2026-09-27/report.md
- qa/perf-structural-audit-2026-09-27/appendix-issues-by-pr.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Since ADR-125 (09-07), every file-backed connect_private_sqlite spawns a python -I -S helper process, about 45-75 ms per open. The off-loop wrappers close the handle after every call: run_owned_db_call, operation_owned_connection, run_finite_local_worker, the scope services' list_and_close, the Media seam, NotesScopeService, run_db_off_loop and SyncStateRepository. ScheduledTasksDB opens a connection per operation. Library, Notes, media reader, Watchlists, Schedules and agent trace writes therefore pay a helper spawn per operation. Each close also runs a blocking wal_checkpoint(TRUNCATE). Thread-local handles registered in the quiescence and participants registries leak when their thread exits. ADR-125's helper stays as designed; the fix is to open far fewer connections. Source: the 2026-09-27 structural efficiency audit at dev 840ed2ca58 (qa/perf-structural-audit-2026-09-27/report.md, section 4, PERF-09; every issue with file:line is listed under PERF-09 in qa/perf-structural-audit-2026-09-27/appendix-issues-by-pr.md).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Library, Notes, media-reader, Watchlists and Schedules operations reuse long-lived per-thread connections; repeated operations spawn no new helper (census-pinned)
- [ ] #2 Closing a connection no longer runs a blocking wal_checkpoint(TRUNCATE) on hot paths
- [ ] #3 Handles owned by exited threads do not stay registered (leak test)
- [ ] #4 Helper spawns during an idle minute on Console are 0, and during a warm Library visit at most 1
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
