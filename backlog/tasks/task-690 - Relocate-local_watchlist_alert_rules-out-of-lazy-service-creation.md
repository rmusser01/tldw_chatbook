---
id: TASK-690
title: >-
  Relocate local_watchlist_alert_rules out of lazy service-side creation
status: To Do
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
- [ ] #1 local_watchlist_alert_rules is created by SubscriptionsDB._initialize_schema, not by a service
- [ ] #2 LocalWatchlistsService._ensure_alert_rule_schema and all six of its call sites are removed
- [ ] #3 A migration adds the table to databases that predate the relocation, and is idempotent across fresh, already-migrated, and legacy databases
- [ ] #4 Existing alert-rule behaviour is unchanged — the Subscriptions and Watchlists suites pass with no new failures
- [ ] #5 No table in tldw_chatbook/Subscriptions/ is created lazily from a service any more
<!-- AC:END -->
