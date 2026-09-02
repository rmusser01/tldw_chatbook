---
id: TASK-28242
title: 'Review sets - Phase 3: entry points (Review these / Review selected)'
status: To Do
assignee: []
created_date: '2026-09-02 22:29'
labels:
  - library
  - media-ux
dependencies:
  - TASK-28240
  - TASK-28241
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Create review sets from the media browse result and from a Select-mode selection (design: backlog/docs/design-library-review-sets.md).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A 'Review these' action on the media list pins the WHOLE filtered browse result (page through to last_page in a worker, de-dupe by id, cap 500 with pin-first-500 + warn on overflow) and opens it active at cursor 0
- [ ] #2 A 'Review selected' third Select-mode bulk action (next to Export/Delete selected) pins the selected ids ordered by a deterministic sort-order query (NOT the mounted rows, since RowSelection is unordered and can span pages)
- [ ] #3 Both paths land the user in the Reader walking the new set; the filter-query surface is covered by 'Review these' (no separate RAG-search integration)
<!-- AC:END -->
