---
id: TASK-3999
title: >-
  Eval-harness: persist the run report to a gitignored results dir (spec
  deviation)
status: To Do
assignee: []
created_date: '2026-08-09 14:48'
labels:
  - rag
  - eval
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The P1 spec promised the run report would be written to a gitignored results directory in addition to the printed summary; only the printed summary shipped -- a plan-time silent drop noted by the final review (TASK-3894).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A gated run writes report.to_dict() as JSON to a gitignored results path and prints where it was written
- [ ] #2 README documents the results-dir location and how to find a run's report
<!-- AC:END -->
