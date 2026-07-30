# Task 1 report: DB — tables, watchlist fields, the queue flag's write and read paths

## Status: Done

## What was built

- `tldw_chatbook/DB/Subscriptions_DB.py` (`_ensure_watchlists_schema`):
  - `briefings(id, watchlist_id, status, error, covers_through_item_id, covers_from_ts,
    selection_mode, preset_id, model_used, body_markdown, item_count, featured_count,
    overflow_count, created_at, updated_at)` and `briefing_items(briefing_id, item_id,
    featured, PRIMARY KEY(briefing_id, item_id))` — additive `CREATE TABLE IF NOT
    EXISTS`, added right after the `update_watchlists_timestamp` trigger. No data
    migration, no `BEGIN IMMEDIATE` (nothing to migrate — TASK-1362's machinery not
    cargo-culted in, per the brief and the spec).
  - `watchlists.briefing_selection_mode TEXT DEFAULT 'auto_featured'` and
    `watchlists.default_briefing_preset_id INTEGER` via the column-presence idiom
    (same shape as the `content_kind` pattern), added right after the `watchlists`
    CREATE TABLE block.
  - Indexes: `idx_briefings_watchlist_status(watchlist_id, status)`,
    `idx_briefing_items_item(item_id)`.
  - New methods (added after `mark_item_status`, before `find_duplicate_items`):
    `set_item_briefing_queued(item_id, queued)`, `insert_briefing(watchlist_id,
    status='generating') -> int`, `update_briefing(briefing_id, **fields)`,
    `get_briefing(briefing_id) -> dict|None`, `list_briefings(watchlist_id) ->
    list[dict]` (newest first), `latest_completed_watermark(watchlist_id) ->
    int|None` (`MAX(covers_through_item_id)` where `status IN ('complete',
    'empty')` — `'failed'` deliberately excluded).
- `tldw_chatbook/Subscriptions/watchlist_normalizers.py` (`normalize_watchlist_item`):
  added `"queued_for_briefing": bool(row.get("queued_for_briefing"))`, coercing
  SQLite's 0/1 to a real bool (`queued_for_briefing` itself already existed on
  `subscription_items` per ADR-018 — only the write method and the normalizer
  carry-through were missing).
- `Tests/Subscriptions/test_briefing_selection.py` (new, `pytestmark =
  pytest.mark.unit`): the brief's three tests, with a full body written for the
  third (see below).

## The real watchlist-creation API

`SubscriptionsDB` has **no** `create_watchlist` method. Watchlists are created
through `tldw_chatbook.Subscriptions.watchlist_bundle_service.WatchlistBundleService.create(name, description=None, tags=None) -> dict`,
which does the `INSERT INTO watchlists` and auto-suffixes on name collision. Its
return dict's `"id"` key is the watchlist id. The stub test was rewritten to
`WatchlistBundleService(db).create(name="w")["id"]`.

The third test's full body: insert a subscription via `add_subscription`, insert
one `subscription_items` row directly (matching the existing
`test_fts_backfill.py` pattern), fetch it through `get_new_items(subscription_id=...,
status="new")`, and assert `normalize_watchlist_item(...)["queued_for_briefing"]`
toggles `False -> True -> False` around calls to `set_item_briefing_queued`.

## Test results

```
Tests/Subscriptions/test_briefing_selection.py::test_briefings_tables_exist_with_watermark_column PASSED
Tests/Subscriptions/test_briefing_selection.py::test_latest_completed_watermark_ignores_failed_and_interrupted PASSED
Tests/Subscriptions/test_briefing_selection.py::test_queue_flag_round_trips_through_the_normalizer PASSED
3 passed

Tests/Subscriptions/ (full directory, includes the above)
193 passed in 38.34s
```

## Mutation checks

1. **Include `'failed'` in `latest_completed_watermark`'s status set.** Changed the
   `WHERE` clause to `status IN ('complete', 'empty', 'failed')`. Result: RED —
   `test_latest_completed_watermark_ignores_failed_and_interrupted` failed with
   `assert 99 == 55` (the `failed` briefing's `covers_through_item_id=99` wrongly
   won the MAX). Restored via Edit to `('complete', 'empty')`.
2. **Drop the normalizer's `queued_for_briefing` line.** Removed the dict entry
   from `normalize_watchlist_item`. Result: RED —
   `test_queue_flag_round_trips_through_the_normalizer` failed with
   `KeyError: 'queued_for_briefing'`. Restored via Edit.

Full `Tests/Subscriptions/` suite re-run after restoring: 193 passed.

## Files touched

- `tldw_chatbook/DB/Subscriptions_DB.py`
- `tldw_chatbook/Subscriptions/watchlist_normalizers.py`
- `Tests/Subscriptions/test_briefing_selection.py` (new)
