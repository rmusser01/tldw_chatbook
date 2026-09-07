---
id: TASK-164
title: Persist Library per-pane filters and sort across tab switches
status: Done
assignee: []
created_date: '2026-07-11 22:02'
updated_date: '2026-07-12 02:36'
labels:
  - follow-up
  - library
  - navigation
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
PR #595 added cross-visit state persistence for Library/Media/Search selection and view, but per-pane filter state is not persisted: Library notes sort/filter, conversations query, and the media type filter all reset on a tab round-trip. Persist them via the existing save_state/restore_state.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Notes sort/filter survives a tab switch away and back
- [x] #2 Conversations query survives
- [x] #3 Media type filter survives
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Fixed in the persistence batch (branch claude/followups-persistence). See Docs/superpowers/plans/2026-07-11-followups-persistence.md.
<!-- SECTION:NOTES:END -->
