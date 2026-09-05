---
id: TASK-31590
title: >-
  Library Import - Analyze-skipped outcomes leak across jobs that reuse a media
  id
status: To Do
assignee: []
created_date: '2026-09-05 05:15'
labels:
  - library
  - media-ux
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Wave 4 PR D (#2400) scoped the Analyze-skipped outcome records to survive Clear finished, but a later ingest job that reuses a media id without an intervening Clear finished still inherits the earlier job's outcome: its row can paint as analyzed and drop out of the Analyze N skipped count although the new version has no analysis (residual from the Qodo-fix re-review of #2400). Outcomes should be keyed by ingest job, not by media id alone.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A second ingest job that reuses a media id, started without Clear finished, is offered Analyze N skipped and its row is not painted as analyzed by the earlier job's outcome
- [ ] #2 Analyze N skipped counts only the current job's unresolved rows
- [ ] #3 A test pins both behaviours
<!-- AC:END -->
