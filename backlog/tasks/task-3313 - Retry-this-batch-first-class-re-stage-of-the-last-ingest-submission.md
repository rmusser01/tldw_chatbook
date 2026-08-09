---
id: TASK-3313
title: >-
  Retry this batch: first-class re-stage of the last ingest submission
status: In Progress
assignee:
  - '@claude'
created_date: '2026-08-08 20:30'
labels:
  - library
  - ingest
  - ux
priority: medium
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Approved by the owner via task-3310 (ruling 3). After Start, the ingest form auto-clears to invite the next source — but the likeliest next action after a failure (or after installing the dependency a warning just named) is the SAME source again, and today that means re-typing/re-browsing; per-row Retry is buried in the queue. Alex-persona flag from the 2026-08-07 critique, re-confirmed by the arc's live verification.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [ ] #1 After a submission reaches a terminal state, a single visible action re-stages that submission's source (path/URL) with its options and metadata restored to the form
- [ ] #2 The action is keyboard-reachable and advertised in the ingest shortcut set
- [ ] #3 Re-staging runs a fresh preflight (tooling installed since the last run is picked up; the old forecast is not reused)
- [ ] #4 The affordance survives the in-place update discipline (object-identity test across queue ticks) and appears only when a last submission exists
<!-- AC:END -->
