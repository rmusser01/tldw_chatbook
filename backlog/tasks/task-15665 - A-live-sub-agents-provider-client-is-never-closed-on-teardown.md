---
id: TASK-15665
title: 'A live sub-agent''s provider client is never closed when the app tears down mid-flight'
status: To Do
assignee: []
created_date: '2026-08-11 21:30'
labels:
  - console
  - agents
  - resources
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
PR 3a-1 Task 6b (audit F5) made `ConsoleProviderGateway.aclose()` SKIP running loops and RETAIN their entries, because the previous behaviour closed a live child's connection pool on the child's own loop mid-request. Retaining is the right trade, but nothing closes those clients early: if the app tears down while children are in flight, one pool per live child is left to its finalizer instead of being closed deliberately. Bounded by `max_live_subagents` per conversation, so small, but it is an accepted leak rather than a handled shutdown.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A teardown with live children closes each child's pool once that child's own loop stops, deliberately rather than by finalizer
- [ ] #2 No pool belonging to a still-running child is closed mid-request (the existing F5 regression still passes)
- [ ] #3 A test asserts no retained client entry outlives its owning lifeline
<!-- AC:END -->
