---
id: TASK-26028
title: 'Scheduling: per-handler preflight validation before dispatch'
status: To Do
assignee: []
created_date: '2026-08-31 15:46'
labels:
  - scheduling
  - reliability
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Nothing checks whether a task can succeed before firing it. Verified on origin/dev: validation happens at creation, and a grep for preflight across tldw_chatbook/Scheduling returns zero - at dispatch the loop calls the handler directly (Scheduling/scheduler/loop.py:342-363), so a watchlist whose source was deleted or a briefing whose provider key was removed fails at run time, repeatedly, on schedule. Hermes validates provider key, delivery target and skill availability before firing and alerts once per condition rather than once per occurrence.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A handler may declare a preflight check that runs immediately before dispatch
- [ ] #2 A failed preflight records a distinct outcome from a handler failure, so the cause is legible
- [ ] #3 A failed preflight does not consume the occurrence in a way that hides the problem - the task remains visible as needing attention
- [ ] #4 The user is told once per condition, not once per occurrence, composing with the incident grouping
- [ ] #5 Handlers without a preflight check dispatch exactly as today
- [ ] #6 Preflight is bounded in time and cannot itself wedge the loop
<!-- AC:END -->
