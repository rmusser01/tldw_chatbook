---
id: TASK-684.2
title: Show remote ingest jobs in the Library ingest queue
status: To Do
assignee: []
created_date: '2026-07-26 04:33'
updated_date: '2026-07-26 05:31'
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
SEQUENCING: this task now comes BEFORE 684.1's routing slice. 684.1's mapping layer landed, but routing a submission has nowhere to record it -- see the registry constraints below -- so the registry work has to precede it.

1. Migrate the ingest_jobs table (DB/Library_Ingest_Jobs_DB.py, currently schema v2) to v3:
   - add origin ('local'/'server'), remote_job_id and batch_id
   - relax the state CHECK, which today allows only queued/parsing/writing/done/failed; the server also reports cancelled and carries a cancellation_reason
   - media_id stays local-only and must become nullable in practice: LibraryIngestJobRegistry.mark_done currently *requires* media_id: int, which a server submission never produces, so it needs a server-shaped completion path
2. Give LibraryIngestJob the matching fields and surface origin on the queue row.
3. Poll remote state with the existing client APIs (list_media_ingest_jobs(batch_id), get_media_ingest_job(job_id)) on a worker, marshalling updates onto the UI thread the way the parse-pool coordinator already does. MediaIngestJobStatus already carries status, progress_percent, progress_message, error_message, source and batch_id.
4. Map queue actions: cancel -> cancel_media_ingest_jobs_batch; hide the ones with no remote equivalent.
5. Tests for row rendering, origin labelling, state transitions and the migration.
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
