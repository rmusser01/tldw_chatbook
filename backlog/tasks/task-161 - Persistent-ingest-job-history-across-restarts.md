---
id: TASK-161
title: Persistent ingest job history across restarts
status: Done
assignee: []
created_date: '2026-07-11 22:02'
labels:
  - follow-up
  - ingest
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The Library ingest job registry is in-memory only; queued/failed jobs are lost on quit. Persist job history so users can review and retry across restarts.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Ingest job history survives an app restart
- [x] #2 Failed/queued jobs can be retried after restart
<!-- AC:END -->

## Implementation Notes

Delivered across three commits (SQLite store, registry restore/plan_restore + retry_count carry, app wiring):

- `DB/Library_Ingest_Jobs_DB.py`: a small `BaseDB` subclass persisting each `LibraryIngestJob` row (including `retry_count`), single reused WAL connection.
- `Library/library_ingest_jobs.py`: `LibraryIngestJobRegistry.attach_store` write-through hook (upsert/delete on every mutation), `restore()` to seed an in-memory registry from rows, and the pure `plan_restore()` helper that normalizes non-terminal (`PARSING`/`WRITING`) jobs left over from a hard quit into `FAILED` (`"Interrupted by app restart"`, AC1+AC2: retryable via `requeue`) and prunes the oldest jobs past `max_persisted`.
- `config.py`: `get_library_ingest_jobs_db_path()` (same custom-path/default pattern as the other `get_*_db_path` helpers) → `~/.local/share/tldw_cli/.../tldw_chatbook_library_ingest_jobs.db`.
- `app.py`: `LibraryIngestQueueMixin._restore_ingest_jobs()` opens the store, attaches it to the already-constructed `self.library_ingest_jobs`, runs `plan_restore`/`restore`, and writes back the plan's normalize/prune diffs — wrapped in try/except so a corrupt store logs a warning and starts empty rather than crashing `on_mount`. Called once, early in `on_mount`; the store is closed in `on_unmount` after the ingest parse pool shuts down. `_MAX_PERSISTED_INGEST_JOBS = 500`.
- `Home/active_work_adapter.py` + `Library/library_ingest_state.py`: a `· retry {n}` suffix (helper duplicated in each module — Library must stay Home-import-free) appended to a FAILED job's status line once `retry_count > 0`, so a job restored after a restart and retried again still shows its retry count on both the Home card and the Library ingest queue row.

Tests: `Tests/Library/test_library_ingest_jobs_restore.py` (new — end-to-end store↔registry round-trip covering interrupted-job normalization and post-restart retry), plus retry-indicator regression tests added to `Tests/Home/test_active_work_adapter.py` and `Tests/Library/test_library_ingest_state.py`. Full `Tests/DB/ Tests/Library/ Tests/Home/` suite (608 tests) passes; `import tldw_chatbook.app` smoke-tested clean.
