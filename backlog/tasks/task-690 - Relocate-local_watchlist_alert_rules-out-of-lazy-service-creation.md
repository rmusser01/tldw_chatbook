---
id: TASK-690
title: >-
  Relocate local_watchlist_alert_rules out of lazy service-side creation
status: Done
assignee: []
created_date: '2026-07-25 22:05'
labels:
  - watchlists
  - followup
  - tech-debt
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
local_watchlist_alert_rules is still created lazily by LocalWatchlistsService._ensure_alert_rule_schema (local_watchlists_service.py:958-976), called from six places. It declares a foreign key to subscriptions.

Phase A (PR #917) established the constraint that all schema lives in SubscriptionsDB — table creation in _initialize_schema, additive migration in _ensure_watchlists_schema — and no table is created lazily from a service. Task 2 of that plan relocated local_watchlist_runs for exactly this reason: a lazily-created table cannot be safely ALTERed by a migration, because on a fresh database the migration runs before the table exists.

local_watchlist_alert_rules was deliberately left out of scope to keep that task bounded. This is the same treatment, applied to the last remaining lazily-created table in this module.

Do it before anything needs to add a column to that table, not after — that ordering is what forced the workaround-then-refactor sequence the first time.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 local_watchlist_alert_rules is created by SubscriptionsDB._initialize_schema, not by a service
- [x] #2 LocalWatchlistsService._ensure_alert_rule_schema and all six of its call sites are removed
- [x] #3 A migration adds the table to databases that predate the relocation, and is idempotent across fresh, already-migrated, and legacy databases
- [x] #4 Existing alert-rule behaviour is unchanged — the Subscriptions and Watchlists suites pass with no new failures
- [x] #5 No table in tldw_chatbook/Subscriptions/ is created lazily from a service any more (with one caveat noted below, out of this task's scope)
<!-- AC:END -->

## Implementation Notes

**Approach.** Moved the `CREATE TABLE IF NOT EXISTS local_watchlist_alert_rules (...)` DDL (unchanged, byte-for-byte column set, including the `FOREIGN KEY (job_id) REFERENCES subscriptions(id) ON DELETE CASCADE`) from `LocalWatchlistsService._ensure_alert_rule_schema` into `SubscriptionsDB._initialize_schema` (`tldw_chatbook/DB/Subscriptions_DB.py`), right after `local_watchlist_runs`, matching that table's precedent and comment style. Removed `_ensure_alert_rule_schema` and all six call sites (`list_alert_rules`, `get_alert_rule`, `create_alert_rule`, `update_alert_rule`, `delete_alert_rule`, `_evaluate_alert_rules_for_run`) from `local_watchlists_service.py` — grep-verified the count both before and after (six call sites plus the definition; zero references anywhere in the repo afterward).

No `_ensure_watchlists_schema` (additive-migration) changes were needed: unlike `local_watchlist_runs`, which needed a genuine `ALTER TABLE ... ADD COLUMN batch_id` after relocation, `local_watchlist_alert_rules`'s shape isn't changing at all — only *where* it gets created. `CREATE TABLE IF NOT EXISTS` inside `_initialize_schema` (which `BaseDB.__init__` always runs, unconditionally, on every open) is therefore sufficient on its own to cover all three required cases, since it never touches an existing table:
- **Fresh DB**: table doesn't exist yet → created alongside `subscriptions` in the same `executescript`.
- **Already migrated**: table exists from a prior run of this same code → `IF NOT EXISTS` no-ops.
- **Legacy DB** (table exists from the old lazy path, possibly with rows): `IF NOT EXISTS` no-ops and the existing table + rows are left completely untouched.

**On the orphan-row risk flagged in the brief:** it does not apply here. That risk (Phase A's CHECK-widening incident) came from a table *rebuild* — drop, recreate with new constraints, copy rows through DML that FK enforcement checks. This relocation never rebuilds or copies `local_watchlist_alert_rules`; `CREATE TABLE IF NOT EXISTS` is a pure no-op against an existing table, and SQLite does not validate FK targets at `CREATE TABLE` time (only on DML with enforcement on). Verified this directly with a test that opens a legacy DB containing an alert-rule row whose `job_id` points at a subscription that no longer exists — it opens cleanly and the orphaned row survives untouched.

**Three idempotency cases, tested** in `Tests/DB/test_subscriptions_db_watchlists.py`:
1. `test_alert_rules_table_owned_by_db` — fresh DB, table + expected columns present without any service call.
2. `test_alert_rules_schema_creation_is_idempotent_across_reopen` — same file opened twice via `SubscriptionsDB(...)`, second open is a no-op.
3. `test_alert_rules_table_and_rows_survive_legacy_lazy_creation` — hand-built legacy DB (raw `sqlite3`, table shape matching the old lazy path, a `subscriptions` row and a referencing alert-rule row) reopened via `SubscriptionsDB`; row survives.
4. `test_alert_rules_orphaned_row_survives_reopen_with_fk_enforcement_on` — same, but with an orphaned `job_id`; opens without `IntegrityError`, row survives.
5. `test_lazy_alert_rule_schema_helper_is_gone` — asserts the method is gone from the class.

**Something that contradicted the brief (AC #5):** the alert-rules table was **not** the last lazily-created table in the package. `tldw_chatbook/Subscriptions/site_config_manager.py:265` (`SiteConfigManager._create_tables`, called from `__init__`) still does `CREATE TABLE IF NOT EXISTS site_configs (...)` on demand — and does so against a `CharactersRAGDB` instance (not `SubscriptionsDB`) pointed at the subscriptions DB path (`get_subscriptions_db_path()`). This is live code (wired through `get_site_config_manager()` into `UI/SiteConfigSettings.py` and `web_scraping_pipelines.py`), not dead code. It's a different table in a different DB class than what this task's ACs describe, so fixing it would silently widen scope; reporting it here per instructions rather than fixing it. AC #5 is checked off for the `local_watchlist_alert_rules`/`SubscriptionsDB` scope this task actually covers, with this caveat noted.

**Tests run** (`.venv/bin/pytest`, one file at a time, in the foreground): `Tests/DB/test_subscriptions_db_watchlists.py` (28 passed, incl. 5 new), `Tests/DB/test_subscriptions_db.py` (11 passed), every file under `Tests/Subscriptions/` (all green), every file under `Tests/Watchlists/` (all green). No regressions.

**Files touched:**
- `tldw_chatbook/DB/Subscriptions_DB.py` — added `local_watchlist_alert_rules` table to `_initialize_schema`.
- `tldw_chatbook/Subscriptions/local_watchlists_service.py` — removed `_ensure_alert_rule_schema` and its six call sites.
- `Tests/DB/test_subscriptions_db_watchlists.py` — added the five tests described above.
