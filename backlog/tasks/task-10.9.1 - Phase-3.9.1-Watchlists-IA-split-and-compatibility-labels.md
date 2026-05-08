---
id: TASK-10.9.1
title: 'Phase 3.9.1: Watchlists IA split and compatibility labels'
status: To Do
assignee: []
created_date: '2026-05-08 03:36'
labels:
  - product-maturity
  - phase-3-9-library-collections
  - watchlists
dependencies:
  - TASK-10.8
references:
  - Docs/superpowers/specs/2026-05-08-library-collections-ia-split-design.md
parent_task_id: TASK-10.9
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Rename the user-facing combined W+C destination to Watchlists while preserving compatibility route IDs and monitored-source workflows.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Shell destination metadata command palette navigation and help copy show Watchlists instead of W+C or Watchlists+Collections
- [ ] #2 The Watchlists destination body no longer renders Collections sections summaries or collection management copy
- [ ] #3 Home and Console visible active-work labels say Watchlists while preserving existing route and payload compatibility where needed
- [ ] #4 Focused regressions cover navigation command palette Watchlists body copy and active-run follow-through
<!-- AC:END -->
