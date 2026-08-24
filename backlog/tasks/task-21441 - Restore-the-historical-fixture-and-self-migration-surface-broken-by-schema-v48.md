---
id: TASK-21441
title: Restore the historical-fixture and self-migration surface broken by schema v48
status: To Do
assignee: []
created_date: '2026-08-24 04:32'
labels:
  - perf-review-2026-08-22
  - database
  - migrations
  - regression
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Schema v48 made two changes that, together, mean the shipped ChaChaNotes writer can no longer construct or upgrade a pre-v48 database. Fifteen tests are red on pristine dev as a result, including the packaging test that simulates a real installed distribution upgrading a v35 database. The shipped TUI is insulated because its two boot-path opens hand the migration a seed, but every other consumer of CharactersRAGDB -- scripts, external callers, and the migration test surface itself -- is not. Found while landing TASK-21128 during the 2026-08-22 performance burn-down; the defect belongs to v48, not to that task.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A pre-v47 database can be upgraded by opening CharactersRAGDB without the caller supplying a ConsoleLibraryMigrationSeed, or the requirement is documented as an intentional API contract with a migration note and a defaulting path for non-TUI callers
- [ ] #2 The shipped message writer can populate a pre-v48 messages table, so historical fixtures can once again be built with production code rather than hand-rolled SQL
- [ ] #3 Tests/Packaging/test_installed_distribution.py::test_installed_distribution_migrates_v35_database_to_current passes for both source and sdist
- [ ] #4 The 14 reds in Tests/DB and the actor-pack migration red in Tests/ChaChaNotesDB are green, with each fix addressing the mechanism rather than re-pinning the assertion around it
- [ ] #5 test_schema_version_is_47's stale pin is updated in a way that does not require editing on every future schema bump
- [ ] #6 A regression test proves a bare CharactersRAGDB open of an existing older database still migrates, so the next migration step cannot silently reintroduce a caller-supplied-data requirement
<!-- AC:END -->

## Evidence (measured first-hand on pristine dev `b3a7cf97b`, 2026-08-23)

Two independent mechanisms, both introduced by v48:

**1. The migration now demands data the caller must supply.** `_upgrade_v47_to_v48`
(`DB/ChaChaNotes_DB.py:6225-6237`) raises `SchemaError: Console library migration seed
is required for v47 upgrade.` unless the constructor was handed a
`ConsoleLibraryMigrationSeed`. A *fresh* database is exempt via `fresh_without_seed`
(initial version 0), so this only bites databases that already exist — i.e. exactly the
upgrade case. The DB class can no longer self-migrate.

**2. The shipped writer cannot write a pre-v48 schema.** `add_message`'s INSERT names
`assistant_generation_state` unconditionally (`DB/ChaChaNotes_DB.py:10547`), so building
an old-schema fixture with production code fails with `table messages has no column named
assistant_generation_state`. This is the pattern the migration tests are built on.

### What is red

| surface | count | mechanism |
|---|---|---|
| `Tests/DB/test_chachanotes_v47_messages_fts_backfill.py` | 9 | both (one is a stale `test_schema_version_is_47` pin) |
| `Tests/DB/test_chachanotes_sync_log_retention_migration.py` | 4 | seed required |
| `Tests/DB/test_chachanotes_sync_log_retention.py` | 1 | writer/schema mismatch |
| `Tests/ChaChaNotesDB/test_actor_pack_migration.py` | 1 | seed required |
| `Tests/Packaging/test_installed_distribution.py` (`[source]`, `[sdist]`) | 2 | seed required |

Run commands: `pytest Tests/DB/` → 14 failed, 1270 passed, 1 skipped.
`pytest Tests/ChaChaNotesDB/ Tests/DB/test_chachanotes_console_library_policy_migration.py`
→ 1 failed, 403 passed. `pytest Tests/Packaging/test_installed_distribution.py -k migrates`
→ 2 failed.

### Why this is not a user-facing outage today

All 12 production files that construct `CharactersRAGDB` do thread the seed through,
including both boot-path opens (`config.py:7216` eager, `config.py:7268` lazy). Verified
by reading each site, not by grep alone. So the shipped TUI upgrades correctly.

The exposure is everything *else*: the packaging test is red precisely because it opens
the database the way an installed distribution's non-TUI consumer would. A migration step
that requires caller-supplied data turns a self-contained upgrade into one that only works
from inside one application. That is the part worth fixing, beyond the red tests.

### Attribution

Surfaced while renumbering TASK-21128 from v48 to v49 after a schema-version collision.
Every red listed here reproduces on pristine dev with no burn-down changes applied; the
21128 branch actually *fixes* one of them. Filed against v48, not 21128.
