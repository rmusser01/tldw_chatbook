---
id: TASK-15669
title: AgentRunsDB._CURRENT_SCHEMA_VERSION says 3 while the database records 6
status: Done
assignee: []
created_date: '2026-08-11 21:30'
updated_date: '2026-08-18 04:42'
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
- [x] #1 The constant and the highest recorded schema_version agree for a freshly created database
- [x] #2 An existing database created by an older build still opens and reports the same effective schema
- [x] #3 A test fails if the two diverge again
- [x] #4 Whatever the resolution, it is recorded in the class docstring so the next migration author is not left guessing
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Fixed as the fleet PR3b Task 4 fold (coordinator ruling #3 in Docs/superpowers/plans/2026-08-17-fleet-pr3b-steering.md), branch feat/fleet-3b-continuation. The drift had widened again since this task's last update: a fresh DB recorded version rows 4..10 (TASK-16800's change_notes migrations consumed 8-10) while the constant still said 3 -- so the fold landed as v11, not the plan's v8. Resolution: _CURRENT_SCHEMA_VERSION = 11 == MAX(schema_version) on a fresh DB; the contract (bump BOTH the constant and the INSERT OR IGNORE row on every migration) is recorded in the AgentRunsDB class docstring. Evidence per AC: (1) test_schema_version_constant_agrees_with_the_version_table asserts constant == MAX(version) on a fresh DB -- failed 3 != 10 before the fix, passes at 11 == 11; (2) test_pre_v11_db_gains_resumed_from_run_id_and_opens_twice hand-builds the pre-v11 shape (version rows to 10, no resumed_from_run_id) and opens it twice through AgentRunsDB -- same effective schema, guarded-ALTER idempotent; (3) the agreement test is the diverge-again tripwire (any future append without a constant bump goes red); (4) the class docstring records the resolution and the from-now-on rule for the next migration author. All in Tests/DB/test_agent_runs_db.py (50 -> 53 passed).
<!-- SECTION:NOTES:END -->
