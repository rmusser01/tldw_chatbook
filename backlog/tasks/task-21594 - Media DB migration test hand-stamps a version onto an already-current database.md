---
id: TASK-21594
title: >-
  Media DB migration test hand stamps a version onto an already current database
status: Done
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

- [x] The test builds its historical database with a real historical fixture rather than stamping a version number onto a current one
- [x] It is green, and remains green when a new media migration is added on top
- [x] The repair is verified not to have made the test vacuous — it must still fail if the reading-progress migration is broken

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

## Implementation Plan

1. Reproduce the red on the working base before changing anything.
2. Replace the hand-degraded fixture with `media_db_at_version(path, 2)` so the
   real chain builds a genuinely v2-shaped database.
3. Assert the historical preconditions the migration under test depends on.
4. Mutate each migration the test claims to cover and prove it goes red.
5. Prove durability by adding a throwaway v8→v9 migration and re-running.

## Implementation Notes

Reproduced on `a71e62e4b`: `Migration v5->v6 failed: duplicate column name:
chunk_engine_version`. The old fixture built a current (v8) database, then
hand-degraded it — `DROP TABLE ReadingProgress`, `ALTER TABLE Media DROP COLUMN
transcription_provenance_json`, `UPDATE schema_version SET version = 2` — so it
had to be taught about every artifact each new migration added, and broke on the
first one nobody remembered to remove (v5→v6's `chunk_engine_version` on
`UnvectorizedMediaChunks`).

Rebuilt on `Tests/DB/historical_bootstrap_v6.media_db_at_version(path, 2)`,
which patches `_CURRENT_SCHEMA_VERSION` and lets the production chain build a
real v2 database. The test now asserts its historical preconditions explicitly
(`schema_version == 2`; neither `ReadingProgress` nor `MediaReadItLaterState`
exists yet) before reopening unpatched and replaying v2→v8.

**One thing the bootstrap module could not supply.** Seeding had to be a raw
historical `INSERT` into `Media` with the v1 column set:
`add_media_with_keywords()` targets the current schema and fails against a real
v2 database with `no such column: transcription_provenance_json` (added at
v4→v5). This is the same "the shipped writer can no longer construct historical
schemas" family the filing names — it applies to row writers, not only to schema
builders.

**Mutation results (each applied to production, then reverted).**

| Mutation | Result |
| --- | --- |
| `_apply_migration_v2_to_v3` bumps the version without creating `ReadingProgress` | 1 failed (`no such table: ReadingProgress`) |
| `_apply_migration_v3_to_v4` bumps the version without creating `MediaReadItLaterState` | 1 failed (`no such table: MediaReadItLaterState`) |

Both prove the assertions still reach the migration under test, and that no
open-time "ensure tables exist" path silently supplies either table on a
migrating database.

**Durability.** With a throwaway v8→v9 migration registered
(`ALTER TABLE Media ADD COLUMN probe_future_column`) and
`_CURRENT_SCHEMA_VERSION = 9`, the repaired test still passed — the failure mode
the old fixture had is structurally gone, because nothing stamps a version any
more. Reverted after the probe.

**Counts.** `Tests/Media_DB/test_media_db_v2.py` 55 passed / 1 failed → 56
passed. `Tests/Media_DB` + `Tests/DB` together: 1382 passed / 13 failed, all 13
pre-existing ChaChaNotes sync-log/FTS reds unrelated to this change.

**Files.** `Tests/Media_DB/test_media_db_v2.py`. No production changes.
