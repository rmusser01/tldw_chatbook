---
id: TASK-684.2
title: Show remote ingest jobs in the Library ingest queue
status: To Do
assignee: []
created_date: '2026-07-26 04:33'
updated_date: '2026-07-26 04:45'
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
1. Use the existing client APIs: list_media_ingest_jobs(batch_id), get_media_ingest_job(job_id) and cancel_media_ingest_jobs_batch already exist on TLDWAPIClient and are surfaced by ServerMediaReadingService.
2. Mirror remote jobs into LibraryIngestJobRegistry rather than merging two sources at render time, so queue rows, counts and Home's active-work adapter keep one source of truth.
3. Give LibraryIngestJob an origin (local/server) and surface it on the row.
4. Poll remote state on a worker and marshal updates onto the UI thread the way the parse-pool coordinator already does.
5. Map queue actions: cancel maps to cancel_media_ingest_jobs_batch; hide the ones with no remote equivalent.
6. Tests for row rendering, origin labelling and state transitions.
<!-- SECTION:PLAN:END -->
