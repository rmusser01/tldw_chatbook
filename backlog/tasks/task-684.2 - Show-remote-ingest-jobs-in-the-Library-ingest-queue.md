---
id: TASK-684.2
title: Show remote ingest jobs in the Library ingest queue
status: To Do
assignee: []
created_date: '2026-07-26 04:33'
updated_date: '2026-07-26 05:45'
labels:
  - ingest
  - consolidation
dependencies: []
parent_task_id: TASK-684
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Remote ingest jobs are monitored in a separate window from local ones, so a user running both has to look in two places to answer one question: what is happening to my imports.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Remote jobs appear in the same queue as local ones
- [ ] #2 A queue row makes clear whether the job is running locally or on a server
- [ ] #3 Remote job state changes are reflected without leaving the screen
- [ ] #4 Queue actions behave sensibly for remote jobs or are hidden for them
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Slices A (schema v3), B (row origin) and C (cancelled lifecycle) are DONE. Remaining: the polling worker.

1. Map server status -> IngestJobState in a pure function first (MediaIngestJobStatus.status plus progress_percent/progress_message/error_message/media id), so the mapping is unit-testable without a server. Unknown statuses must not silently become 'done'.
2. Poll on @work(exclusive=True, thread=True, group=...) -- its OWN group, never exclusive without one.
3. MARSHALLING HAZARD, already solved once in this file for the parse pool (see _ingest_pool_callback's docstring): Textual's call_from_thread BLOCKS the calling thread on the marshalled call and only guards against the loop being None, not against it shutting down. A poll result arriving as the user quits can park this thread while the quit path waits on it -- mutual deadlock. Mitigation is two-layer: check the shutdown flag on the worker thread BEFORE marshalling, and have the marshalled body no-op on the same flag.
4. Drive from batch_id via list_media_ingest_jobs, falling back to get_media_ingest_job for a single remote_job_id; both are already on ServerMediaReadingService.
5. Stop polling once every remote job in the batch is terminal, so a finished batch does not poll forever.
6. Wire the cancel action to cancel_media_ingest_jobs_batch -> mark_cancelled (the state it produces now exists).
7. Tests with a fake service: status mapping, terminal stop condition, and that a shutdown flag suppresses marshalling.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Slices A and B landed; polling and cancel remain.

SLICE A (schema v3, commit 95700cf93) -- the thing 684.1's routing was blocked on. ingest_jobs gains origin ('local'/'server'), remote_job_id and batch_id; LibraryIngestJob carries them; submit() takes origin; new attach_remote() records the ids the server only issues after accepting a submission (deliberately not a state transition -- attaching ids does not move a job through its lifecycle). The state CHECK now admits 'cancelled', which SQLite can only do by rebuilding the table, so it is done once now rather than forcing a second rebuild when cancel arrives. The rebuild copies by explicit column list so a future column cannot silently shift position.

Verified beyond synthetic fixtures: seeded a copy of the REAL live v2 database (~/.local/share/tldw_cli/default_user) with a realistic spread -- a done job with JSON ingest_options and a content_hash, a permanent failure with error_detail, a superseded queued job with progress -- then migrated it with the production class. Version went to 3, all 12 compared columns across every row were preserved exactly, new columns defaulted to local/None, and the relaxed CHECK accepted a cancelled/server row. Original untouched.

DELIBERATE RESTRAINT: the IngestJobState.CANCELLED member, its queue-row rendering and the cancel action are NOT in slice A. Only the CHECK is forward-looking, because relaxing it later would need another table rebuild. The enum and its consumers land together in the slice that can exercise them.

SLICE B (queue row origin) -- once both backends share one queue, 'done - notes.txt' cannot say which machine did the work. Rows carry origin and server ones get a ' - on server' suffix; local stays unannotated as the common case. The row builder returns from five state branches, so it was renamed and wrapped: the marker is stamped in exactly one place and a new state cannot ship without it. A test asserts the marker in all five states, and both new behaviours were mutation-checked.

REMAINING: poll remote state on a worker via list_media_ingest_jobs/get_media_ingest_job and marshal onto the UI thread the way the parse-pool coordinator does; then the cancel action (cancel_media_ingest_jobs_batch) together with the CANCELLED enum member, its row rendering, the terminal-state sets in clear_finished, and _COUNTS_LINE_ORDER.
<!-- SECTION:NOTES:END -->
