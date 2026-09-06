---
id: TASK-31822
title: Repair Console Stop clipping after Redirect action was added
status: To Do
assignee:
  - '@codex'
created_date: '2026-09-06 06:18'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The real 160x48 Console action row budgets 37 cells while active controls require 47 after TASK28227 introduced Redirect. Stop is clipped by hidden overflow. Forced test scrolling conceals the production defect; preserve the original regression until a bounded runtime layout fix is approved.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 At the existing 160x48 viewport the active Stop button is physically visible and clickable without forced scrolling or programmatic Stop focus.
- [ ] #2 The action-width calculation accounts for applicable controls without changing Redirect semantics or widening test deadlines.
- [ ] #3 Real click/cancellation and relevant composer layout tests pass, with ordinary composer focus established for synthetic Send setup.
<!-- AC:END -->
