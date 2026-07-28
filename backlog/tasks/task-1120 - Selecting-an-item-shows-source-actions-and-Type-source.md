---
id: TASK-1120
title: >-
  Selecting an item shows "Type: source" and offers source actions
status: Done
assignee: []
created_date: '2026-07-28 10:30'
labels:
  - watchlists
  - bug
  - ui
  - uat
priority: high
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Clicking a row in the **Items** table selects it, and the Inspector names it correctly — but classifies it as a source:

```
Selected: Lightsail object storage concerns - Part 2
Type: source
           Preview
          Check now
```

`Preview` and `Check now` are *source* actions. The item actions the Inspector is built to offer — `Mark reviewed`, `Ingest`, `Ignore` — never appear, so an item cannot be acted on at all.

Observed with real scraped content on `origin/dev` `79152bbb6`: 10 items fetched from `https://summitroute.com/blog/feed.xml`, clean profile.

`InspectorPane._entity_type` decides this, and the entity reaching it evidently carries the shape of a source rather than an item. Worth checking what `ItemSelected` puts on the wire versus what `SourceSelected` does, and whether the Items table's rows are being routed through the sources selection path — the Sources table's own selection defect (task-1105) suggests these tables share more wiring than is obvious.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Selecting an item reports `Type: item`
- [x] #2 The Inspector offers `Mark reviewed`, `Ingest` and `Ignore` for a selected item
- [x] #3 Those actions change the item's status, verified against the database
- [x] #4 Source, run, rule and notification selections still report their own types
- [x] #5 A test selects an item and asserts the reported type and offered actions, proven to fail against current code
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
**Cause.** `InspectorPane._entity_type` guessed from shape, and its first test was `"source_type" in entity or "url" in entity`. Every item `normalize_watchlist_item` produces carries **both**: `source_type` is the type of the feed the item came from, and `url` is the article's own link. The item branch (`item_id`, `source_name`) sat two tests below and was never reached, so every fetched item typed as `source`.

The same guessing hid a second one this task's AC#4 test caught: `normalize_watchlist_run` puts its counts under `stats`, not as `found_count`/`processed_count` keys, so a run matched no branch at all and typed as **`unknown`** — the Inspector's `else` arm, which offers only `Delete`.

**Fix.** Every normalizer stamps an explicit `entity_kind`; that is the backend's own answer and now decides, via a `_ENTITY_KINDS` map (`subscription`/`watchlist_source` → source, `watchlist_run` → run, `watchlist_item` → item, `watchlist_alert_rule` → rule, `client_notification` → notification). The shape heuristics stay as a fallback for hand-built dicts (tree scopes, fixtures), reordered so `item_id` outranks the source keys and `run_id` is recognised.

**AC#3 needed two more things that did not exist.** With the typing fixed, the three item buttons appeared and did nothing: `WatchlistsBackendController.update_item_status` probes for `update_item`/`update_item_status`/`mark_item_status` on the scope service and **none of them existed**, so it raised `NotImplementedError` into `_update_item_status`'s `except Exception: ... .debug(...)` — the swallow TASK-1090 covers. Behind that, `watchlists.items` was registered in the runtime-policy registry with `(LIST, DETAIL)` only, so `watchlists.items.update.local` was an unregistered action id and the enforcer denied it. Added:

- `LocalWatchlistsService.update_item(item_id=, status=)` over the long-unused `SubscriptionsDB.mark_item_status`, validating status against `ITEM_STATUSES`;
- `WatchlistScopeService.update_item(...)`, which resolves the namespaced `local:watchlist_item:2` id and refuses the server backend explicitly (the server API carries no item-status route, exactly as `list_items` already refuses it);
- `UPDATE` on the `watchlists.items` policy resource, with the two new action ids added to the audited parity matrix in `Tests/RuntimePolicy/test_runtime_policy_core.py`.

**Verified in the running app** (clean scratch profile, 10 real items from `https://summitroute.com/blog/feed.xml`) — clicking the third item row:

```
Selected: S3 backups and other strategies for ensuring data
durability through ransomware attacks
Type: item
         Mark reviewed
             Ingest
             Ignore
```

and pressing `Mark reviewed` removed it from the `new` list (10 → 9), which is the status write landing in the database.

**Files:** `UI/Watchlists_Modules/inspector_pane.py`, `Subscriptions/local_watchlists_service.py`, `Subscriptions/watchlist_scope_service.py`, `runtime_policy/registry.py`, `Tests/UI/test_watchlists_item_actions.py` (new), `Tests/RuntimePolicy/test_runtime_policy_core.py`.
<!-- SECTION:NOTES:END -->
