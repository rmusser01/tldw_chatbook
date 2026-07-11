---
id: TASK-164
title: Persist Library per-pane filters and sort across tab switches
status: To Do
assignee: []
created_date: '2026-07-11 22:02'
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
- [ ] #1 Notes sort/filter survives a tab switch away and back
- [ ] #2 Conversations query survives
- [ ] #3 Media type filter survives
<!-- AC:END -->
