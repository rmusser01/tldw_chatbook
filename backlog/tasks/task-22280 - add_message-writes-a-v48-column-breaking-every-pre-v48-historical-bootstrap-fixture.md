---
id: TASK-22280
title: >-
  add_message writes a v48 column, breaking every pre-v48 historical-bootstrap
  fixture (12 dev-red migration tests)
status: To Do
assignee: []
created_date: '2026-08-24'
labels:
  - testing
  - database
priority: high
dependencies: []
---

## Description

Found during task-22200 (backfill pacing), baselined against pristine dev
`983aa5878` in a clean worktree: **12 migration tests are red on dev itself**,
all with `sqlite3.OperationalError: table messages has no column named
assistant_generation_state`:

- `Tests/DB/test_chachanotes_v47_messages_fts_backfill.py` — 8 of 14 red
  (including the SIGKILL resumability witness and both snapshot-upgrade
  interleave tests for the hot-writer IMMEDIATE contract)
- `Tests/DB/test_chachanotes_sync_log_retention_migration.py` — 4 of 7 red

Mechanism: `CharactersRAGDB.add_message` now unconditionally INSERTs
`assistant_generation_state` (the sync/continuation work around `a2e056a8d` /
`dc10bda0a` / `b89052dc4`), but that column is only created by the v47->v48
migration (`chachanotes_v47_to_v48_console_library_policy.sql:22`). Every
fixture that builds a genuinely historical DB via
`Tests/ChaChaNotesDB/historical_bootstrap.chachanotes_db_at_version(path, 44/45)`
and then seeds through `add_message` (`_seed_v45` and friends) fails at the
seed, so the migration guarantees those files existed to pin (deferred FTS
rebuild, kill-safety, rewind-on-failure, the v47 trigger guards) currently
have no working witnesses.

Note the coverage shape when fixing: task-22200's pacing tests deliberately
sidestep this by seeding at CURRENT schema + the migration's own
`'delete-all'` reset (see `Tests/DB/test_chachanotes_fts_backfill_pacing.py`'s
module docstring), which reproduces the backfill WINDOW but cannot replace the
v45-replay coverage these red tests provided. The likely fix directions:
either `add_message` probes/branches on column presence (ugly, production code
serving tests), or the historical seeding helpers write pre-v48 rows with
version-appropriate SQL instead of calling today's `add_message` — the
historical-bootstrap module's own philosophy ("knowledge about the single
migration the test pins, owned by that test") points at the latter.

## Acceptance Criteria

- [ ] The 12 enumerated tests pass on dev without weakening what they pin (the deferred-rebuild, kill-safety, rewind, and trigger-guard assertions stay intact)
- [ ] Seeding a `chachanotes_db_at_version(..., 44/45)` fixture with messages works again, via a mechanism that does not require production `add_message` to know about test schemas
- [ ] A guard or lesson entry records how a production column addition silently invalidated historical-bootstrap seeding, so the next `messages` column addition does not repeat this
