---
id: TASK-165
title: Richer Media screen state restore (row highlight and viewer detail)
status: To Do
assignee: []
created_date: '2026-07-11 22:02'
labels:
  - follow-up
  - media
  - navigation
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
MediaScreen restore is scalar-only for the selected id: the list re-highlight and viewer detail are not re-populated after a tab round-trip. Restore the row highlight and, when a viewer was open, re-fetch its detail (mirror the Library media-viewer on_mount re-kick pattern).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Restored Media screen re-highlights the previously selected row,A previously open media viewer re-fetches its detail on restore
<!-- AC:END -->
