---
id: TASK-33920
title: Live-verify Databricks through AI Gateway in Console
status: To Do
assignee: []
created_date: '2026-10-03 01:38'
labels:
  - providers
  - live
  - engine
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
TASK-33010 shipped the Databricks preset through the hosted engine, but its live check (streaming and non-streaming Console turns against a real workspace through AI Gateway) was never run: no workspace credentials were available. Split out of TASK-33010 so that task can close on what it delivered.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A streaming and a non-streaming Console turn succeed against a real Databricks workspace through AI Gateway
- [ ] #2 The capture tool's Databricks capture replays under its record, or the record is amended citing the fixture
<!-- AC:END -->
