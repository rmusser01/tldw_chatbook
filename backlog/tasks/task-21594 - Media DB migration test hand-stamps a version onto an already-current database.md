---
id: TASK-21594
title: >-
  Media DB migration test hand stamps a version onto an already current database
status: To Do
assignee: []
created_date: '2026-08-23'
labels:
  - testing
  - dev-red
  - media
priority: low
---
## Description

`Tests/Media_DB/test_media_db_v2.py::TestDatabaseCRUDAndSync::test_reading_progress_reopens_through_versioned_migration`
is red on dev. It hand-stamps `schema_version = 2` onto a database that is already current, so
reopening replays v5→v6 against columns that exist and dies on `duplicate column name:
chunk_engine_version`. The test will break again on every future media migration.

## Acceptance Criteria

- [ ] The test builds its historical database with a real historical fixture rather than stamping a version number onto a current one
- [ ] It is green, and remains green when a new media migration is added on top
- [ ] The repair is verified not to have made the test vacuous — it must still fail if the reading-progress migration is broken

## Evidence (verified first-hand on dev 33ff5b754, 2026-08-23)

```
pytest Tests/Media_DB/test_media_db_v2.py -k reading_progress_reopens
  -> 1 failed, 55 deselected
  DatabaseError: Schema initialization failed: Migration v5->v6 failed:
  duplicate column name: chunk_engine_version
```

`Tests/DB/historical_bootstrap_v6.py` already exists and is the right tool. Same family as
TASK-21441: the shipped writer can no longer construct historical schemas, so fixtures have to
come from a bootstrap module rather than from production code.
