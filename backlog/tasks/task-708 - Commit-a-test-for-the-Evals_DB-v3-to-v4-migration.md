---
id: TASK-708
title: Commit a test for the Evals_DB v3 to v4 migration
status: Done
assignee:
  - '@claude'
created_date: '2026-07-26 14:30'
updated_date: '2026-07-27 05:52'
labels:
  - evals
  - tests
  - db
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by the whole-branch review of PR 2 of the Evals rebuild (the word bench engine). Not a defect introduced by that PR unless stated; each is a seam the engine leaves for the screen that consumes it.

PR 2 took `Evals_DB.SCHEMA_VERSION` from 3 to 4, adding `eval_runs.run_group_id` plus its index. The upgrade path was verified during implementation by a manual heredoc script, not by a committed test. Every automated test builds a fresh v4 database and therefore exercises `_create_schema`, never the `ALTER TABLE`.

Existing users take the migration path. This repo's own spec lists schema-version collisions at merge as a standing risk, and a manual check does not survive the renumber it is warning about. Sibling databases in this repo do carry migration tests.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A committed test builds a database at version 3, reopens it at the current version, and asserts `user_version` and the `run_group_id` column and index
- [x] #2 The test asserts reopening again is idempotent
- [x] #3 The test lives with the other Evals tests and runs in the normal suite
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Find the sibling migration-test pattern already used in this repo (Tests/DB/test_agent_runs_db.py's hand-built legacy-schema fixture) and follow it rather than inventing a new shape.
2. Build the exact v3 Evals_DB shape by hand with raw sqlite3 (copied from _create_schema, minus run_group_id and its index), seed it with rows in every affected table.
3. Open it through the real EvalsDB class and assert user_version == 4, the run_group_id column and its index exist, and the seeded rows survived intact.
4. Assert a second EvalsDB open against the same file is idempotent (no raise, no duplicate index, data intact).
5. Revert-check: remove the migration step from Evals_DB.py, confirm the new tests fail, restore.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
New file `Tests/Evals/test_evals_db_v3_to_v4_migration.py`, following the pattern of `Tests/DB/test_agent_runs_db.py`'s `_LEGACY_V1_AGENT_RUNS_DDL` fixture (a hand-built raw-sqlite3 legacy schema, opened through the real DB class) rather than inventing a new shape.

**Approach**: `_V3_SCHEMA_DDL` is `Evals_DB._create_schema`'s exact v3 shape (all tables, FKs, CHECK constraints, indexes, FTS5 tables/triggers) with only `eval_runs.run_group_id` and `idx_eval_runs_group` removed, ending in `PRAGMA user_version = 3`. `_build_v3_database` writes this by hand with raw `sqlite3`, then seeds one row each in `eval_tasks`, `eval_models`, `eval_runs`, and `eval_results` that must survive the upgrade untouched.

Four tests: (1) a sanity check that the hand-built fixture genuinely lacks the v4 column before it is ever opened through EvalsDB, so the migration test below proves something; (2) opening it through `EvalsDB` reaches `user_version == 4`, adds `run_group_id` and `idx_eval_runs_group` (asserting the index's column list, not just its name), the four seeded rows are byte-identical, and the migrated DB is still fully functional end-to-end (`create_run` + `update_run(run_group_id=...)` + `list_runs(run_group_id=...)` round-trips); (3) reopening the same file a second time does not raise, does not duplicate the index, and the seeded row is still intact.

**Revert-check performed**: removed the `current_version < 4` migration block from `Evals_DB._migrate_schema` (leaving `PRAGMA user_version = SCHEMA_VERSION` in place, so a v3 file silently claims v4 without actually gaining the column -- the exact bug this task guards against). Re-ran the new test file: the two migration-assertion tests failed as expected (`run_group_id` missing from `PRAGMA table_info`); the fixture sanity-check test was unaffected (it doesn't touch migration code). Restored the migration block and confirmed all 3 tests plus the rest of `Tests/Evals/test_evals_db.py` (38 total) pass again.

**Files**: `Tests/Evals/test_evals_db_v3_to_v4_migration.py` (new).
<!-- SECTION:NOTES:END -->
