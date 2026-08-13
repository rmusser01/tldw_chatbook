---
id: TASK-15703
title: Make Library conversations list scrollable and paginated
status: In Progress
assignee: []
created_date: '2026-08-12'
labels:
  - library
  - conversations
  - ux
priority: medium
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Let users browse every saved conversation from Library without the result set
being clipped by terminal height or silently capped at the first fetched page.
Filtering must search the complete saved-conversation collection so older
conversations remain discoverable.

Design: `Docs/superpowers/specs/2026-08-12-library-conversations-pagination-design.md`
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [ ] #1 The Library conversation rows render in a vertically scrollable viewport, and every row on the current page is reachable by mouse and keyboard regardless of terminal height
- [ ] #2 The view shows at most 20 conversations per page, exposes Previous and Next controls plus the current page and result range, and allows every saved conversation to be reached
- [ ] #3 Submitting a conversation filter searches the complete saved-conversation collection, resets to page 1, and reports the filtered total
- [ ] #4 Paging and filtering keep the last successful page visible while loading, reject stale responses, and present a recoverable error without misreporting an empty library
- [ ] #5 Automated state, service-call, and Textual Pilot tests cover scrolling, first/middle/last pages, full-dataset filtering, empty results, and failures
<!-- AC:END -->
