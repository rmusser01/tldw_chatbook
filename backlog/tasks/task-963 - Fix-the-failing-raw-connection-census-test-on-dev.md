---
id: TASK-963
title: Fix the failing raw-connection census test on dev
status: Done
assignee:
  - '@zcode'
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
- [x] #1 The census test passes on a clean checkout,Subscriptions_DB.ensure_site_configs_schema is either qualified like its siblings or explicitly accounted for with a stated reason,No other raw-connection site is silently unaccounted for
<!-- AC:END -->

## Implementation Plan

1. Reproduce the census failure on a clean checkout of current origin/dev.
2. If it still fails, qualify `ensure_site_configs_schema` like its siblings or account for it in the census; if it passes, verify why and close as already-fixed with evidence.

ADR required: no
ADR path: N/A
Reason: Test-baseline verification only; no application behavior or architecture change.

## Implementation Notes

Closed as already-fixed on dev, verified rather than assumed.

Evidence (worktree at `934b28f39a` = origin/dev, 2026-09-11):
`Tests/DB/test_private_sqlite_inventory.py` — **37 passed**, including
`test_raw_connection_census_is_qualified_and_transition_aware`. The unaccounted
site named in the description no longer exists: `ensure_site_configs_schema`
(`tldw_chatbook/DB/Subscriptions_DB.py:378`) now opens its caller-supplied path
through `connect_private_sqlite` instead of a raw connect, with an in-code
comment stating the private-path guards rationale — exactly the "qualified like
its siblings" branch of AC #1. The green census itself asserts no other
raw-connection site is unaccounted for (AC #1's third clause). No code was
changed by this task.
