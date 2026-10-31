---
id: TASK-38441
title: >-
  Owner decision: whether to commit the projection-attribution probe kit so
  TASK-36571's measurements can be rerun from the repo
status: To Do
assignee: []
created_date: '2026-10-31 07:16'
labels:
  - console
  - performance
  - tooling
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
TASK-36571's component attribution and its paired live rounds come from a probe kit (the pf2 probe extended with sys.monitoring code-object timers, a run driver, and analyzers that read the raw per-window records) kept outside the repository with the session's evidence. A reader of the task file cannot rerun the analysis or check a figure against the raw data, and the next projection task (TASK-36571.1) will need the same kit. The owner has to decide whether the kit belongs in the repo and where, since it drives a live app against a real provider.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The owner has recorded whether the kit is committed, and where
- [ ] #2 If committed: the analyzers run on a committed sample of raw records and reproduce TASK-36571's base attribution table
- [ ] #3 If committed: the kit never imports tldw_chatbook into a process that touches a real profile, and its README states the isolation it relies on
<!-- AC:END -->
