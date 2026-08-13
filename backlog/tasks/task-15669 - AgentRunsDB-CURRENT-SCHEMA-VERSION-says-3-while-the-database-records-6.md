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

**State update, 2026-08-13 (fleet PR 3a-2, `feat/fleet-autowake`):** the drift has widened by one. Commit `f3bc3c19f` (the durable per-run wake-delivery ledger) added a fourth idempotent-ALTER migration — `agent_runs.wake_delivered_at` plus `INSERT OR IGNORE INTO schema_version (version) VALUES (7)` (`AgentRuns_DB.py:344`) — again following the established pattern rather than fixing the constant blind. A freshly created database now records rows 4, 5, 6 AND 7 while `_CURRENT_SCHEMA_VERSION` (line 38) still says 3. The task title's "records 6" is therefore stale; treat this paragraph as the current truth when fixing.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The constant and the highest recorded schema_version agree for a freshly created database
- [ ] #2 An existing database created by an older build still opens and reports the same effective schema
- [ ] #3 A test fails if the two diverge again
- [ ] #4 Whatever the resolution, it is recorded in the class docstring so the next migration author is not left guessing
<!-- AC:END -->
