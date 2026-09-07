---
id: TASK-158
title: Export success message carries the creator's item counts
status: Done
assignee: []
created_date: '2026-07-11 22:01'
updated_date: '2026-07-11 23:52'
labels:
  - follow-up
  - export
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
On a successful Library export the completion notification reports only the path; the creator's own outcome message (item counts, deleted-mid-export skips) is discarded. Surface those counts so a partial export (records deleted between count and export) is visible to the user.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Success notification includes exported item counts
- [ ] #2 Deleted-mid-export skips are reflected in the message
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Fixed in the quick-wins batch (branch claude/followups-quickwins). See Docs/superpowers/plans/2026-07-11-followups-quickwins.md.
<!-- SECTION:NOTES:END -->
