---
id: TASK-227
title: >-
  Stop task-cancel can race the agent worker finally-reset and finish done
  instead of cancelled
status: To Do
assignee: []
created_date: '2026-07-14 03:33'
labels:
  - console
  - agents
  - reliability
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found while testing the stop-before-first-token fix: stop_active_run()'s task cancellation can race _run_agent_reply's finally block resetting _stop_requested, letting a still-running background bridge thread complete its run as 'done' although the user stopped it. Documented in .superpowers/sdd/task-8-report.md (gate fix wave). Narrow window; the store-level no-op fix covers the common case.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A stop that lands during an in-flight agent bridge thread always persists the run as cancelled,A regression test covers the cancel-vs-finally race deterministically
<!-- AC:END -->
