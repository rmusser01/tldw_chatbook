---
id: TASK-963
title: Fix the failing raw-connection census test on dev
status: To Do
assignee: []
created_date: '2026-07-27 18:06'
labels:
  - db
  - tests
  - dev-baseline
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Tests/DB/test_private_sqlite_inventory.py::test_raw_connection_census_is_qualified_and_transition_aware fails on pristine origin/dev. The census asserts every raw sqlite connection site is qualified and transition-aware; it currently reports an unaccounted site, {('tldw_chatbook/DB/Subscriptions_DB', 'ensure_site_configs_schema'): 1}. Either that call site needs qualifying the way its siblings are, or the census needs updating to account for it deliberately. Confirmed pre-existing and unrelated to the path-naming audit branches (#996, #999) by running the file on a pristine origin/dev worktree.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The census test passes on a clean checkout,Subscriptions_DB.ensure_site_configs_schema is either qualified like its siblings or explicitly accounted for with a stated reason,No other raw-connection site is silently unaccounted for
<!-- AC:END -->
