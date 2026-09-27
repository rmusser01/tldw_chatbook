---
id: TASK-33088
title: One schema migration engine across stores
status: To Do
assignee: []
created_date: '2026-09-27 19:46'
labels: [refactor, db]
dependencies:
  - TASK-33087
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Roughly 14 ad-hoc migration runners are split across two version-storage mechanisms (PRAGMA user_version ladders versus schema_version tables), each with its own step ladder, runner loop, and column-exists guard idioms. The scaffolding is duplicated per store while the ladders keep growing (agent_runs just added v18 through v21). A single runner supporting SQL-file and Python steps with per-step verification removes the per-store scaffolding; ChaChaNotes' guarded-trigger and idempotent step machinery must remain expressible as step types.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A single migration runner supports versioned steps from SQL files or Python with per-step verification.
- [ ] #2 ChaChaNotes guarded-trigger and idempotent migration steps remain expressible as step types.
- [ ] #3 Every store migrated to the engine reaches an identical resulting schema version, verified against pre-migration baselines.
- [ ] #4 The migration scaffolding LOC removed is recorded in Implementation Notes.
<!-- AC:END -->
