---
id: TASK-16324
title: Add iterative gap-driven replanning to local research runs
status: To Do
assignee:
  - '@robert'
created_date: '2026-08-15 05:15'
labels:
  - research
dependencies:
  - TASK-16322
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Phase 1 generates sub-queries once and never revises them as evidence arrives. Add a bounded iteration loop modeled on tldw_server stop_criteria: after an initial synthesis pass, identify thin or unanswered sub-questions, generate follow-up sub-queries, and loop within budget and max_iterations.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 After an initial synthesis pass a gap-analysis step identifies unanswered or thin sub-questions,Follow-up sub-queries are generated and executed bounded by max_iterations and the budget ledger,The final report reflects evidence gathered across iterations and names remaining gaps,Iteration transitions are visible in the run event stream,Tests pin the iteration bound and gap reporting
<!-- AC:END -->
