---
id: TASK-165
title: Richer Media screen state restore (row highlight and viewer detail)
status: Done
assignee: []
created_date: '2026-07-11 22:02'
updated_date: '2026-07-12 02:36'
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
- [x] #1 Restored Media screen re-highlights the previously selected row
- [x] #2 A previously open media viewer re-fetches its detail on restore
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Fixed in the persistence batch (branch claude/followups-persistence). See Docs/superpowers/plans/2026-07-11-followups-persistence.md.
<!-- SECTION:NOTES:END -->
