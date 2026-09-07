---
id: TASK-31942
title: >-
  Library media - analysis saves never commit (Reader Save and bulk Analyze lost
  on exit)
status: Done
assignee:
  - '@claude'
created_date: '2026-09-07 07:28'
updated_date: '2026-09-07 07:50'
labels:
  - library
  - media-ux
  - bug
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found during wave-5 PR J Task 3 and reproduced with a two-connection probe: LocalMediaReadingService.save_analysis_version delegates to MediaDatabase.create_document_version, whose docstring says it assumes an existing transaction context, but the service calls it bare. The INSERT lands on the app's thread-local connection, which is left in_transaction; a second connection (another process, the next app launch) sees only the ingest version. The Reader's Generate/Save analysis and the bulk Analyze run therefore persist only when some later write on the same connection happens to commit, and are silently lost on exit otherwise. overwrite_analysis_version / delete_analysis_version and the highlight writers need the same audit.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A saved analysis is visible to a second SQLite connection immediately after save_analysis_version returns (pinned with a real second connection, not the app's)
- [x] #2 overwrite_analysis_version, delete_analysis_version and any other service-layer DocumentVersions or highlight write commit the same way, or are shown to already run inside a transaction
- [x] #3 A failed write leaves no partial version row and surfaces as an error the Reader shows
- [x] #4 The Reader's analysis section and the list row's analysed marker reflect the persisted state after an app restart (live-verified once)
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Reproduce with a two-connection probe (a file-backed MediaDatabase, save through the service, read through a plain sqlite3 connection). 2. Wrap the service-layer analysis write in the DB's transaction() (it nests: joins an open transaction, commits only at the outermost level); leave create_document_version's in-transaction contract untouched for the DB's own callers. 3. Pin save/overwrite/delete through a real second connection plus a mid-write failure that leaves no partial row and raises. 4. Audit every other DocumentVersions/highlight writer. 5. Live: scratch profile, save in the Reader, quit, relaunch.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Root cause: LocalMediaReadingService.save_analysis_version called MediaDatabase.create_document_version bare; that method assumes an open transaction and never commits, so the INSERT sat in an open transaction on the app's thread-local connection (invisible to any other connection, rolled back on close) and poisoned the next write on that connection — delete_analysis_version's own transaction() merely joined it. Fix: save_analysis_version wraps the call in db.transaction() (overwrite delegates to save; delete already opened its own transaction and commits now that nothing leaks). Audit: scope-service async wrappers inherit (worker-thread connection); MediaWindow_v2 callers inherit and already notify on exception; meeting_speaker_rename's two calls sit inside its own transaction; the server service is HTTP; every other local-service mutator opens its own transaction. AC#3 needed no UI change: _save_library_media_analysis already catches and warns, and the failure now reaches it. Live (scratch profile, one instance): a second process saw version 2 committed while the app ran; after quit and relaunch the Reader's Analysis tab and the row's analysed marker showed the saved text. Riders: create_document_version's contract is docstring-only (a future bare caller can reintroduce this); the list row's analysed marker refreshes only on the next fetch until PR J's item 19 lands. Files: tldw_chatbook/Media/local_media_reading_service.py, Tests/Media/test_local_media_reading_service.py.
<!-- SECTION:NOTES:END -->
