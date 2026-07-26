---
id: TASK-684.2
title: Show remote ingest jobs in the Library ingest queue
status: To Do
assignee: []
created_date: '2026-07-26 04:33'
updated_date: '2026-07-26 05:16'
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
