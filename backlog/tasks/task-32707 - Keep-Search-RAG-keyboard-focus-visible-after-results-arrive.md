---
id: TASK-32707
title: Keep Search RAG keyboard focus visible after results arrive
status: To Do
assignee: []
created_date: '2026-09-17 05:00'
labels:
  - library
  - search-rag
  - keyboard
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
During TASK-2377 native verification, submitting a real local keyword query from the focused Search/RAG query field scrolls Evidence into view while focus remains on library-rag-query-input. At 80x24 the retained input region is (29, -19, 50, 3), completely outside the viewport; the footer still says typing in field. The same condition occurs in both themes; at 170x48 the query is clipped above the panel viewport. Evidence: Docs/superpowers/qa/2026-09-17-rag-scope-recovery/result.json and the ready SVG captures. This differs from TASK-32053's card focus cue and TASK-4023's intended Evidence reveal: users now have keyboard focus in an invisible input after a successful query. Review the result-arrival reveal/focus contract, including a newer user focus decision while retrieval is in flight.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 After a query completes, the focused control remains visibly painted in wide and compact layouts in both themes.
- [ ] #2 Evidence remains discoverable and query text is retained; a newer user focus choice during retrieval is not overwritten.
- [ ] #3 Mounted timing and native keyboard checks cover query submission and result arrival without provider calls.
<!-- AC:END -->
