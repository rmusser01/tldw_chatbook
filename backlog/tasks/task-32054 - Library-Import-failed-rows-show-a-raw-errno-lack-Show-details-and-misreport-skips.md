---
id: TASK-32054
title: >-
  Library Import: failed rows show a raw errno, lack 'Show details', and
  misreport skips
status: Done
assignee: []
created_date: '2026-09-08 18:23'
updated_date: '2026-09-08 19:28'
labels:
  - library
  - import
  - ux
  - critique-8
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
A failed job reads '✗ failed · reading-notes.md · Parse pool could not start: [Errno 28] No space left on device' on a disk with 80 GB free (the real cause was POSIX semaphore exhaustion); the documented 'Show details' row action is absent; files the forecast said 'will skip' become 'failed' with the pool reason; a 6-file batch stacks several 'Import finished — 1 failed' toasts; Retry appends '· attempt 2' where the guide says '· retry 1'. Evidence: critique #8 snapshot `.impeccable/critique/2026-09-08T16-05-41Z__tldw-chatbook-ui-screens-library-screen-py.md` (dev 1c022378cb, 2026-09-08, dual-agent live review at 235x52/100x30/60x24 on a fresh and a seeded profile). Register row 5.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Pool-start and OS-level failures map to a plain-language reason with a next step (for example a system resource limit with 'Restart the app, then Retry')
- [x] #2 Every failed row offers 'Show details' with the underlying error available on demand
- [x] #3 Unsupported files keep their skip reason and are not offered Retry
- [x] #4 A batch produces one completion toast
- [x] #5 The retry suffix matches the user guide (docs or copy updated, one of the two)
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add map_ingest_failure/IngestFailureCopy to library_ingest_state (errno-28 pool start -> plain summary + next step; unsupported stays non-retryable).
2. Wire the FAILED queue row to the mapped summary+next step; make Show details available on every failed row (raw text under it).
3. Pool-start failure keeps an unsupported file SKIPPED instead of FAILED.
4. Collapse a batch's per-file settle toasts into one Import finished toast.
5. Retry suffix -> ' retry N' (matches the guide and Home).
6. Docs + live verify.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
map_ingest_failure(exc_or_text, context=) -> IngestFailureCopy(summary, detail, next_step, retryable) now sits in library_ingest_state.py and is the queue row's single source of failure copy. An ENOSPC raised while starting the parse pool becomes 'The import worker couldn't start on this machine (system resource limit) · Restart the app, then Retry'; the raw text stays on .detail and only 'Show details' renders it. Show details is offered on every failed row with any error text (the gate was error_detail, which the pool failure never carries) -- the canvas test that pinned the old gate is REVERSED with its reason, not deleted. A pool that cannot start now records an unsupported file as SKIPPED with the classifier's own reason instead of a retryable failure carrying the pool error (app._unsupported_ingest_source_error). The batch-settle toast defers one turn of the event loop, so a folder import's per-file dispatch (which really does cross active->0 between files) reports once. Retry suffix reads ' · retry N', matching the guide and Home.

Live: fresh profile, 6-file inbox import, all four ACs observed at 235x52 -- rows read the mapped reason with no errno, Show details revealed 'Parse pool could not start: [Errno 28] No space left on device', export.json/weird.xyz stayed '○ skipped' with Dismiss only, and one toast 'Import finished — 4 failed · 2 skipped'.

Files: tldw_chatbook/Library/library_ingest_state.py, tldw_chatbook/Widgets/Library/library_ingest_canvas.py, tldw_chatbook/UI/Library_Modules/library_ingest_state.py, tldw_chatbook/UI/Library_Modules/library_ingest_controller.py, tldw_chatbook/app.py, Tests/UI/test_library_crit8_recovery_copy.py (new), Tests/UI/test_library_ingest_canvas.py, Tests/Library/test_library_ingest_state.py, Tests/Library/test_library_ingest_runner.py, Docs/User_Guide/library/import-and-export.md
<!-- SECTION:NOTES:END -->
