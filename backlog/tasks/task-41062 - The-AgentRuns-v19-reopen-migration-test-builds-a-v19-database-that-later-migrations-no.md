---
id: TASK-41062
title: "The AgentRuns v19 reopen-migration test builds a v19 database that later migrations no longer accept"
status: To Do
created_date: 2026-11-15 02:03
labels:
- follow-up
- architecture-review-2026-10-25
- tests
- database
- agents
priority: low
references:
- Tests/DB/test_agent_worktree_recovery.py
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
`Tests/DB/test_agent_worktree_recovery.py::test_runtime_reopen_migrates_v19_and_preserves_definition_cap` fails on dev ecad404bd4 (2026-11-15) with `sqlite3.OperationalError: table workflow_task_claims already exists` (`DB/AgentRuns_DB.py:3890`).

The test makes its "v19" database by opening a current one and dropping only `agent_worktrees`, a few columns, and the schema_version rows from 20 on. Tables that later migrations create, such as `workflow_task_claims`, stay behind, so the replayed migration collides with them. The test no longer proves that a real v19 database upgrades. The architecture-review runs on 2026-11-13 saw it red too.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The test starts from a database that matches the real v19 schema (a fixture or a schema built by the v19 migration chain)
- [ ] #2 The test passes on dev, and still checks that the definition cap survives the upgrade
<!-- AC:END -->
