---
id: TASK-154
title: Home Pause control has no ingest semantics for ingest items
status: To Do
assignee: []
created_date: '2026-07-11 22:01'
labels:
  - follow-up
  - home
  - ingest
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The generic Home Pause control renders for local Library ingest items (they enter the Running feed) but has no ingest-pause behavior wired. It should either be suppressed for ingest-kind items or wired to a real queue-pause action. Found in F3 live QA.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Pause is not shown for ingest items OR is wired to a real pause action,Behavior covered by a Home pilot
<!-- AC:END -->
