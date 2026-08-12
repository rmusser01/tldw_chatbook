---
id: TASK-15669
title: 'AgentRunsDB._CURRENT_SCHEMA_VERSION says 3 while the database records 6'
status: To Do
assignee: []
created_date: '2026-08-11 21:30'
labels:
  - db
  - agents
  - tech-debt
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
`AgentRunsDB._CURRENT_SCHEMA_VERSION` is 3, but a live `agent_runs.db` created by current code holds schema_version rows 4, 5 and 6 (verified by querying a freshly created database during PR 3a-1 Task 7 verification). The drift predates PR 3a-1 - rows 4 and 5 were already being inserted against a constant of 3 - and each subsequent migration has followed the existing pattern rather than fixing it blind. The constant is what CLAUDE.md points every schema change at, so a constant that disagrees with the database is a trap for the next person who reads it.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The constant and the highest recorded schema_version agree for a freshly created database
- [ ] #2 An existing database created by an older build still opens and reports the same effective schema
- [ ] #3 A test fails if the two diverge again
- [ ] #4 Whatever the resolution, it is recorded in the class docstring so the next migration author is not left guessing
<!-- AC:END -->
