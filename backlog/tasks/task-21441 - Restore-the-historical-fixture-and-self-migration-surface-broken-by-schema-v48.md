---
id: TASK-21441
title: Restore the historical-fixture and self-migration surface broken by schema v48
status: Done
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
- [x] #1 A pre-v47 database can be upgraded by opening CharactersRAGDB without the caller supplying a ConsoleLibraryMigrationSeed, or the requirement is documented as an intentional API contract with a migration note and a defaulting path for non-TUI callers
- [x] #2 The shipped message writer can populate a pre-v48 messages table, so historical fixtures can once again be built with production code rather than hand-rolled SQL
- [x] #3 Tests/Packaging/test_installed_distribution.py::test_installed_distribution_migrates_v35_database_to_current passes for both source and sdist
- [x] #4 The 14 reds in Tests/DB and the actor-pack migration red in Tests/ChaChaNotesDB are green, with each fix addressing the mechanism rather than re-pinning the assertion around it
- [x] #5 test_schema_version_is_47's stale pin is updated in a way that does not require editing on every future schema bump
- [x] #6 A regression test proves a bare CharactersRAGDB open of an existing older database still migrates, so the next migration step cannot silently reintroduce a caller-supplied-data requirement
<!-- AC:END -->

## Implementation Plan

1. Re-measure every red on the implementation base (`a71e62e4b`), not on the
   filing's base, and record the per-test failing MECHANISM rather than the count.
2. Direction: **restore self-migration**. Make `console_library_migration_seed`
   genuinely optional in `_migrate_from_v47_to_v48`, defaulting to the same
   `auto_retrieve_on_send = 0` the fresh-database path already uses; keep a
   wrong-typed seed a hard error; warn when defaulting on an existing database.
   Delete the now-dead duplicate pre-flight check in `_initialize_schema`.
3. Make `add_message` write the column set the OPEN database actually has, so a
   historical fixture is built by production code; raise rather than silently
   drop when a caller supplies a value for a column the schema lacks.
4. Prove the migration is still atomic, re-enterable, interrupt-safe, and that
   the unseeded result is byte-identical to the seeded-False result (content
   hash, not row count) with `PRAGMA integrity_check` clean.
5. Repair the one stale test double (`sync_log` retention's hand-built envelope
   hash predates v48's `assistant_generation_state` envelope key).
6. Document the defaulting contract in `DB/migrations/README.md` and the
   constructor docstring; mutation-test every new assertion.

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

## Implementation Notes

Restored self-migration rather than documenting the requirement as a contract.

**Direction, and why.** The seed carries one boolean, and
`config.load_console_library_migration_seed` already yields `False` for a
missing or non-boolean `chat_defaults.rag_auto_retrieve_on_send` -- the same
value the step's own `fresh_without_seed` branch has always written. So the
requirement was not protecting an invariant; it was protecting nothing, at the
cost of the DB class being able to upgrade itself. Defaulting is also fail-safe
in the direction `console_library_policy` itself defines (absent authority is
Never/Blocked, never permission): the worst case for an unseeded upgrade is a
user re-enabling automatic retrieval, against a current worst case of the
database refusing to open at all. `_migrate_from_v47_to_v48` now defaults an
absent seed, logs a warning when it does so for an existing database, and still
raises for a wrong-typed one -- absent is a legitimate state, malformed is a
caller defect. The duplicate pre-flight check in `_initialize_schema` is gone;
being keyed on entering at exactly v47, it never even fired for the v35 case it
was meant to guard. The packaging canary passes untouched.

**Writer.** `add_message` built its INSERT from the newest schema's column list
-- an assertion about a schema it never checked -- so v48's
`messages.assistant_generation_state` made the shipped writer unable to populate
a pre-v48 `messages` table, which is exactly how `Tests/ChaChaNotesDB/
historical_bootstrap.py` builds migration fixtures. It now derives the column
list from `PRAGMA table_info(messages)` (read once per instance) and drops an
absent column **only when its value is None** -- the `NULL` it would have
received anyway -- raising otherwise, so the adaptive path cannot mask an
incompletely migrated database. This is per-schema, not per-column: the next
nullable `messages` column needs no fixture work. `create_assistant_continuation`
deliberately keeps its fixed list; that data has nowhere to go in an old schema.

**Trade-off accepted:** one `PRAGMA table_info` per `CharactersRAGDB` instance.

**Stale doubles repaired, not re-pinned.** Three tests pinned the old contract
and were changed deliberately: the `None` halves of
`test_v47_missing_or_invalid_seed_leaves_schema_unchanged` and
`test_v47_upgrade_rejects_missing_or_invalid_seed_before_ddl` INVERT (absent
seed now migrates, with the resulting policy asserted equal to an explicit
`False` seed); their wrong-type halves survive. `test_retention_does_not_break_
the_committed_intent_readers` hand-rolled `{"content", "role"}` and went red
when v48 added `assistant_generation_state` to the envelope contract -- it now
builds the payload through one helper that states the contract once.
`test_every_production_characters_rag_db_opener_passes_explicit_seed` stays and
matters more, since a production opener that stopped passing a seed would now
default silently instead of raising.

**Evidence.** Baseline on `a71e62e4b`: 13 failed / 1303 passed / 1 skipped in
`Tests/DB`, 1 failed / 404 passed in `Tests/ChaChaNotesDB` (+ the policy
migration file), 2 failed in the packaging migrate test. After: 0 failed / 1710
passed / 1 skipped across `Tests/DB` + `Tests/ChaChaNotesDB`, and 2 passed
`[source]` + `[sdist]`. Migration correctness is proven by a real SIGKILL inside
the v48 transaction (child stalls in the guarded version bump; the parent first
witnesses the open write lock, then proves nothing is visible, kills, and shows
the file still at v47 with an unchanged content hash and a clean
`PRAGMA integrity_check`), by a deterministic mid-step failure that rewinds the
whole chain, and by content HASHES over every table -- unseeded vs explicit
`False`, and post-kill vs uninterrupted -- rather than row counts. Twelve
mutations of the implementation were each confirmed to turn the new assertions
red. AC #5 needed no work: task-21128 already replaced that literal with
`_CURRENT_SCHEMA_VERSION` and moved the one exact pin to the newest step's own
file; verified on this base.

**Modified or added files.** `tldw_chatbook/DB/ChaChaNotes_DB.py`,
`tldw_chatbook/DB/migrations/README.md`,
`Docs/security/production-diagnostic-inventory.json` (one row, the new warning),
`Tests/ChaChaNotesDB/historical_bootstrap.py`,
`Tests/DB/test_chachanotes_console_library_policy_migration.py`,
`Tests/DB/test_chachanotes_console_library_migration_seed_openers.py`,
`Tests/DB/test_chachanotes_sync_log_retention.py`, and the new
`Tests/DB/test_chachanotes_bare_open_self_migration.py`.
