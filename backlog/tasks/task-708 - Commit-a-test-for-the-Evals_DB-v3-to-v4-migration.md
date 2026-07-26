---
id: TASK-708
title: >-
  Commit a test for the Evals_DB v3 to v4 migration
status: To Do
assignee: []
created_date: '2026-07-26 14:30'
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
- [ ] A committed test builds a database at version 3, reopens it at the current version, and asserts `user_version` and the `run_group_id` column and index
- [ ] The test asserts reopening again is idempotent
- [ ] The test lives with the other Evals tests and runs in the normal suite
<!-- AC:END -->
