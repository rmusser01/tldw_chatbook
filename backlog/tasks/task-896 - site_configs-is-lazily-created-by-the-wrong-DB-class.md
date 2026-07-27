---
id: TASK-896
title: >-
  site_configs is lazily created, by the wrong DB class, in the subscriptions
  database file
status: To Do
assignee: []
created_date: '2026-07-27 15:00'
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
- [ ] #1 A decision is recorded on which database owns `site_configs`, with the reasoning stated
- [ ] #2 The table is created by that database's `_initialize_schema`, not by a service at runtime
- [ ] #3 A migration covers databases that already have the table from the lazy path, and existing site configs survive it
- [ ] #4 The migration is idempotent across fresh, already-migrated, and legacy databases, each verified by a test
- [ ] #5 `SiteConfigSettings` and `web_scraping_pipelines` keep working unchanged
- [ ] #6 No table under `tldw_chatbook/Subscriptions/` is created lazily from a service, verified across the whole package
<!-- AC:END -->
