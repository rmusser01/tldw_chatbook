---
id: TASK-896
title: >-
  site_configs is lazily created, by the wrong DB class, in the subscriptions
  database file
status: Done
assignee:
  - '@claude'
created_date: '2026-07-27 15:00'
updated_date: '2026-07-27 15:55'
labels:
  - watchlists
  - tech-debt
  - db
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
`SiteConfigManager._create_tables` (`tldw_chatbook/Subscriptions/site_config_manager.py:258-272`) issues `CREATE TABLE IF NOT EXISTS site_configs` at runtime, from a service, on a `CharactersRAGDB` instance that is pointed at the subscriptions database path.

Two problems stack here:

- **It is lazily created from a service.** That is the pattern task-690 just finished removing from this package — a lazily-created table cannot be safely `ALTER`ed by a migration, because on a fresh database the migration runs before the table exists. 690 relocated `local_watchlist_alert_rules` for exactly this reason and `local_watchlist_runs` before it. This is the last one, and it was found while verifying 690's own acceptance criterion that no table in `tldw_chatbook/Subscriptions/` is created lazily from a service.
- **It is created by the wrong class.** The table lives in the subscriptions database file but is defined and owned by `CharactersRAGDB`, so neither class's schema is the truth about that file's contents. `SubscriptionsDB._initialize_schema` does not know the table exists.

This is live code, not dead: `UI/SiteConfigSettings.py` and `Subscriptions/web_scraping_pipelines.py` both reach it through `get_site_config_manager()`.

Deliberately left out of task-690 rather than fixed silently, because deciding which database should own `site_configs` is a judgement call that task did not carry.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A decision is recorded on which database owns `site_configs`, with the reasoning stated
- [x] #2 The table is created by that database's `_initialize_schema`, not by a service at runtime
- [x] #3 A migration covers databases that already have the table from the lazy path, and existing site configs survive it
- [x] #4 The migration is idempotent across fresh, already-migrated, and legacy databases, each verified by a test
- [x] #5 `SiteConfigSettings` and `web_scraping_pipelines` keep working unchanged
- [x] #6 No table under `tldw_chatbook/Subscriptions/` is created lazily from a service, verified across the whole package
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Confirm where site_configs physically lives: read SiteConfigManager.__init__ (db_path defaults to get_subscriptions_db_path(); opens CharactersRAGDB against that path) -- confirmed it is the subscriptions database file, just created by the wrong class.
2. Read local_watchlist_alert_rules relocation (commit f45e0b721, TASK-690) as precedent for shape/comment style.
3. Add site_configs CREATE TABLE + index to SubscriptionsDB._initialize_schema, verbatim column shape, with a comment recording the ownership decision.
4. Remove SiteConfigManager._create_tables and its lazy CREATE TABLE call; instead construct-and-close a SubscriptionsDB against the same db_path in __init__ so that schema (declared in step 3) is guaranteed to exist before SiteConfigManager's own CharactersRAGDB connection is used -- CharactersRAGDB continues to own the actual read/write connection, unchanged.
5. Add DB-level tests (fresh / idempotent reopen / legacy-with-rows) mirroring the alert_rules precedent, plus manager-level tests through the public SiteConfigManager API.
6. Sweep tldw_chatbook/Subscriptions/ for any remaining CREATE/ALTER TABLE issued by a service at runtime (AC #6).
7. Run the required regression suites one file at a time.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Ownership decision (AC #1): `SiteConfigManager.__init__` always builds `CharactersRAGDB(db_path, CLI_APP_CLIENT_ID)` with `db_path = get_subscriptions_db_path()` when no explicit path is given -- confirmed by reading the constructor, not assumed. site_configs has therefore always physically lived in the subscriptions database file; it was only ever declared by the wrong class (CharactersRAGDB, lazily, at runtime). SubscriptionsDB is the class that already owns every other table in that file (subscriptions, watchlists, local_watchlist_runs, local_watchlist_alert_rules), so site_configs now joins them in SubscriptionsDB._initialize_schema. No data moves -- same file, same column shape, just declared in the right place.

Mechanics: SiteConfigManager still reads/writes site_configs through its own CharactersRAGDB connection (unchanged, so SiteConfigSettings/web_scraping_pipelines need no changes -- AC #5). Since that connection doesn't know about SubscriptionsDB's schema, __init__ now does `SubscriptionsDB(db_path, CLI_APP_CLIENT_ID).close()` first, purely to run _initialize_schema (which has no versioned migrations and runs unconditionally on every open) against the same file before any query touches site_configs. `_create_tables` and its lazy CREATE TABLE/INDEX are deleted.

Idempotency (AC #3/#4), three cases each covered by a test:
- Fresh: Tests/DB/test_subscriptions_db_site_configs.py::test_site_configs_table_owned_by_db and test_site_configs_index_created -- constructing a bare SubscriptionsDB against a new file creates the table with no SiteConfigManager involved.
- Already-migrated: test_site_configs_schema_creation_is_idempotent_across_reopen -- opening the same file twice is a silent no-op (CREATE TABLE/INDEX IF NOT EXISTS).
- Legacy (table + rows created the old way): test_site_configs_table_and_rows_survive_legacy_lazy_creation builds a raw sqlite file with exactly the shape the old SiteConfigManager._create_tables used to create, inserts a row, then opens it via SubscriptionsDB and asserts the row survives untouched (CREATE TABLE IF NOT EXISTS never rebuilds, so there is no copy step that could drop or corrupt data).
Manager-level coverage in Tests/Subscriptions/test_site_config_manager.py exercises the same three cases through the public SiteConfigManager API (get_config/save_config/list_configs/apply_preset/delete_config), including a legacy DB with an existing row and repeated SiteConfigManager construction against the same file.

AC #6 sweep: `grep -rn "CREATE TABLE\|ALTER TABLE\|CREATE INDEX\|CREATE VIRTUAL TABLE" tldw_chatbook/Subscriptions/ --include="*.py"` now returns nothing. site_configs was the last such case (per TASK-690's own closing note); no table under tldw_chatbook/Subscriptions/ is created lazily from a service any more.

Files changed:
- tldw_chatbook/DB/Subscriptions_DB.py -- declares site_configs + its index in _initialize_schema, with a comment recording the ownership decision.
- tldw_chatbook/Subscriptions/site_config_manager.py -- removed _create_tables/lazy DDL; __init__ now ensures schema via a throwaway SubscriptionsDB open+close before building its own CharactersRAGDB connection.
- Tests/DB/test_subscriptions_db_site_configs.py (new) -- DB-level fresh/idempotent/legacy coverage plus a check that SiteConfigManager._create_tables is gone.
- Tests/Subscriptions/test_site_config_manager.py (new) -- manager-level fresh/idempotent/legacy coverage plus a preset/delete smoke test.

Verified green (run one file at a time): Tests/DB/test_subscriptions_db_site_configs.py (5 passed), Tests/Subscriptions/test_site_config_manager.py (4 passed), Tests/DB/test_subscriptions_db.py (11 passed), Tests/DB/test_subscriptions_db_watchlists.py (28 passed), Tests/Subscriptions/ (107 passed), Tests/Watchlists/ (145 passed). ruff check clean on all changed files.
<!-- SECTION:NOTES:END -->
