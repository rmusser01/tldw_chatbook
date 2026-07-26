---
id: TASK-684.2
title: Show remote ingest jobs in the Library ingest queue
status: To Do
assignee: []
created_date: '2026-07-26 04:33'
updated_date: '2026-07-26 14:01'
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
- [x] #1 Remote jobs appear in the same queue as local ones
- [x] #2 A queue row makes clear whether the job is running locally or on a server
- [x] #3 Remote job state changes are reflected without leaving the screen
- [x] #4 Queue actions behave sensibly for remote jobs or are hidden for them
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
Server-submitted jobs now live in the same ingest queue as local ones, labelled by origin, updated by a poller that folds server statuses back into the local registry.

Shape: schema v3 adds origin/remote_job_id/batch_id and admits the server's 'cancelled' state (table rebuild -- a CHECK constraint change). The decision layer (Library/server_ingest_reconcile.py) is Textual-free and tested against a real registry, so the whole fold-back is exercised without an app, a worker or a server; the worker only fetches and hands over. Two properties it owns: a settled job is never rewritten (polling repeats, so reconciling a terminal status twice must not re-stamp the finish time), and an unrecognised status moves nothing.

FOUR defects that unit tests structurally could not catch, all found by contact with a live server:

1. Pagination was dead in production. The poller asked for successive offsets, but neither the client nor ServerMediaReadingService accepted an 'offset' -- so every real call raised TypeError and fell into a one-page fallback. The endpoint documents offset (max 10000) and returns has_more/next_offset, so paging was always supported; only our client failed to expose it. My fake declared offset because I wrote it to match my own call site, which is exactly how the keyword-only cancel bug hid too: a fake can agree with a wrong assumption, a real signature cannot. Guarded now by a contract test asserting the REAL class accepts what the poller sends.

2. The broad 'except TypeError' around that call is what made it invisible -- it cannot distinguish 'no such parameter' from 'the callable raised TypeError internally', so a genuine bug read as a missing feature and degraded silently. Replaced with an explicit signature check (_accepts_keyword), which also treats an unreadable signature and **kwargs as support rather than downgrading real services.

3. WORST: MediaIngestJobStatus.result was typed ReadingImportResponse -- the *reading-list* import result (source/imported/updated/skipped), a different domain reused by mistake. A media ingest reports {status, media_id, media_uuid, error, warnings, db_message}, so validation failed with four missing fields on every COMPLETED job. Only a finished job triggers it, and the poller treats a raised list call as transient: it logged at debug and retried forever, so jobs would have sat 'queued' in the UI indefinitely while the poller churned. Now typed as the free-form object the endpoint documents, pinned by a regression test holding a verbatim live payload.

4. The docstring on mark_remote_done claimed the server's response 'carries only counts, no media id' -- false, and derived from that same mistyping. Corrected. The behaviour is unchanged and still right for a better reason: the id addresses a row in the SERVER's library while media_id locally means a row in this machine's, so storing it would point 'Open in Library' at a wrong or absent local row. Opening a server-ingested item needs a server-aware affordance (filed as task-688).

Live-verified end to end after the fixes: submit -> queued -> completed -> DONE. The status vocabulary can only be learned this way, since the endpoint types status as a bare string with no enum; 'queued', 'completed' and 'cancelled' are now confirmed on the wire, 'running' and 'failed' are still only mapped, not observed.

Files: DB/Library_Ingest_Jobs_DB.py (v3), Library/library_ingest_jobs.py, Library/server_ingest_status.py + server_ingest_reconcile.py (new), app.py (poller/_reconcile_remote_batch/_accepts_keyword), tldw_api/client.py + media_reading_schemas.py, Media/server_media_reading_service.py.
<!-- SECTION:NOTES:END -->
