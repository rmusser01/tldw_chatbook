---
id: TASK-31942
title: >-
  Library media - analysis saves never commit (Reader Save and bulk Analyze lost
  on exit)
status: To Do
assignee: []
created_date: '2026-09-07 07:28'
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
- [ ] #1 A saved analysis is visible to a second SQLite connection immediately after save_analysis_version returns (pinned with a real second connection, not the app's)
- [ ] #2 overwrite_analysis_version, delete_analysis_version and any other service-layer DocumentVersions or highlight write commit the same way, or are shown to already run inside a transaction
- [ ] #3 A failed write leaves no partial version row and surfaces as an error the Reader shows
- [ ] #4 The Reader's analysis section and the list row's analysed marker reflect the persisted state after an app restart (live-verified once)
<!-- AC:END -->
