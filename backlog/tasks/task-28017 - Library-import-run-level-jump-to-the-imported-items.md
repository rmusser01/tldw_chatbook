---
id: TASK-28017
title: Library import - run-level jump to the imported items
status: To Do
assignee: []
created_date: '2026-09-02 04:11'
updated_date: '2026-09-02 21:07'
labels:
  - library
  - media-ux
dependencies: []
references:
  - >-
    .impeccable/critique/2026-09-02T04-00-36Z__tldw-chatbook-ui-screens-library-screen-py.md
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Completing an import leaves the user on the Import canvas; new items are reachable only via per-row Open in Library links or the rail. After a multi-file run there is no single view-what-I-just-imported affordance.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A completed run offers one action that lands the user in the media list showing the run's items
- [ ] #2 Per-row links keep working
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
RECON (not started — bigger than polish): filtering the media list to a run's exact items needs an id-set filter added to MediaBrowseScope (Library/library_media_state.py) threaded through LibraryMediaBrowseController._search -> service.search_media, which has no id-list param today. The per-row 'Open in Library' (handle_library_ingest_open -> _open_job_in_library -> _navigate_to_media) opens the SINGLE-item viewer, not a scoped list. Run->media-ids exists only per-job (ingest_jobs.batch_id + media_id in Library_Ingest_Jobs_DB) with no query that consumes it. Natural anchor for the action: the run-group header in library_ingest_canvas.py (IngestQueueGroup.batch_id/job_ids). Lazy alt: switch to the media list on default last_modified_desc so the fresh run floats to top (skips new filter code; fails if user re-sorted or concurrent edits interleave).
<!-- SECTION:NOTES:END -->
