---
id: TASK-31580
title: >-
  Library media - replace the remaining raw view literals with the named
  constants
status: To Do
assignee: []
created_date: '2026-09-05 03:24'
labels:
  - library
  - media-ux
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Wave 4 PR B introduced _MEDIA_VIEW_LIST and _MEDIA_VIEW_VIEWER and used them in new code; 55 raw list and viewer literals remain in the media surface helpers (Qodo rule on PR #2386, file-wide sweep declined for scope).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 No raw list or viewer view literals remain in the media shell helpers
- [ ] #2 Existing tests pass unchanged
<!-- AC:END -->
