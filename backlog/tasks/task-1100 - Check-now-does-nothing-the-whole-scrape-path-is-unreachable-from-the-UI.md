---
id: TASK-1100
title: >-
  Check now does nothing — the scrape path is unreachable from the UI
status: To Do
assignee: []
created_date: '2026-07-28 06:00'
labels:
  - watchlists
  - bug
  - critical
  - uat
priority: critical
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Pressing **Check now** on a Watchlists source produces nothing at all: no run, no items, `subscriptions.last_checked` still NULL, `last_error` still NULL, and no error the user can see. Verified against two real feeds (`https://summitroute.com/blog/feed.xml`, `https://feeds.simplecast.com/qm_9xx0g`) on a clean profile, `origin/dev` `b72c1deeb`.

**Watchlists cannot fetch anything. The feature does not work.**

## The backend is fine — this is a wiring problem

Driven directly, the scrape path works perfectly:

```
RUN STATUS: completed
stats: {'items_found': 10, 'items_ingested': 10, 'new_items_found': 10, 'response_time_ms': 268}
ITEMS FETCHED: 10
FIRST TITLE: Lightsail object storage concerns - Part 2
```

## One break found and fixed; at least one more remains

`LocalWatchlistsService` returns rows carrying **both** `"id": "local:subscription:1"` (namespaced, what the UI passes everywhere) and `"source_id": 1`. `local.launch_run` does `int(source_id)`, so the namespaced form raised `ValueError: invalid literal for int() with base 10: 'local:subscription:1'`, which `_check_now_source` swallowed into a debug log.

Fixed on branch `fix/watchlists-check-now-source-id` by resolving namespaced source ids in `WatchlistScopeService`, mirroring the `_run_id_from_item_id` / `_rule_id_from_item_id` it already has for the other two entity types. Covered by `Tests/Subscriptions/test_watchlist_check_now_source_id.py`, red before and green after.

**That fix alone does not make the live app work.** With it applied, clicking Check now in the running app still produced 0 runs and 0 items. So there is at least one more break between the button and the service. The next thing to establish is whether the source row is actually being selected — `handle_check_now_requested` returns silently when `event.entity is None`, and `Preview`/`Check now` are disabled when nothing is selected, so a selection that never registers would look exactly like this.

## Why this went unnoticed

Every prior test and UAT used placeholder URLs and never asserted that anything was fetched. The three earlier UAT runs walked create/scope/add/rename/delete and stopped short of `Check now`, so the product's central function — go and get the content — had never once been exercised.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Pressing Check now on a selected source fetches it: a run is recorded and items land in `subscription_items`
- [ ] #2 Verified live against a real feed from a clean profile, with the item count shown
- [ ] #3 Pressing Check now with nothing selected tells the user so, instead of silently doing nothing
- [ ] #4 A failure to fetch surfaces to the user and sets `last_error`, rather than being swallowed into a debug log
- [ ] #5 A test drives the button and asserts items are ingested, proven to fail against current code
- [ ] #6 `Preview` and `Re-run source` are checked for the same id mismatch
<!-- AC:END -->
