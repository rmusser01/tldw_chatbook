---
id: TASK-15464
title: 'Watchlists items feed: narrow projection and indexed effective-date ordering'
status: Done
assignee:
  - '@claude'
created_date: '2026-08-11 12:05'
updated_date: '2026-08-11 16:30'
labels:
  - perf
  - watchlists
  - db
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
From the audit: the items list queries (`DB/Subscriptions_DB.py:1833-1844/:1862-1872` and `get_new_items`' main path) select `i.*` — pulling `content` (full scraped article text) for up to 100 list rows — and order the whole table by `COALESCE(datetime(published_date), datetime(created_at)) DESC`: per-row datetime parsing, unindexable, LIMIT applied post-sort. O(table) work per Items-pane refresh, multiplied by the fresh-construction cost until task-15463 lands.

Fix direction: a list projection without `content` (fetch on select), plus a stored normalized effective-date column with an index. The schema change goes through `_ensure_watchlists_schema` as a proper migration with legacy-row backfill — identical sort semantics for rows with and without `published_date`. Evidence and method: Docs/Design/2026-08-11-input-latency-audit.md (audit of dev 82b595049; all file:line cites verified there).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Items list queries no longer select the content column for list rows (evidence); item detail still loads full content
- [x] #2 Ordering uses an indexed column with sort order identical to today for both legacy and new rows (migration + tests)
- [x] #3 Items-pane refresh latency before/after on a large corpus recorded
<!-- AC:END -->

## Implementation Plan

1. Probe real SQLite (3.49.1, this repo's bundled version) BEFORE writing any
   migration: confirm `COALESCE(datetime(published_date), datetime(created_at))`
   behavior for NULL/`''`/unparseable `published_date`, confirm tie-break order
   (ties resolve by ascending `id`/rowid today, with or without an index), and
   confirm whether `ALTER TABLE ADD COLUMN ... GENERATED ALWAYS AS (...)` can add
   a `STORED` vs `VIRTUAL` generated column to an EXISTING table.
2. Trace every consumer of `get_new_items`'s rows (`watchlist_normalizers.
   normalize_watchlist_item`, `article_list._render_row`, `content_pane.py`,
   `briefing_selection.py`'s OWN separate query) to determine which columns the
   list path actually reads, before deciding the narrowed projection.
3. Migration in `_ensure_watchlists_schema`: add a `effective_date` generated
   column (`VIRTUAL`, since `STORED` cannot be added via `ALTER TABLE ADD
   COLUMN`) computed as the exact COALESCE expression above, guarded by
   `PRAGMA table_xinfo` (not `table_info` -- probe found `table_info` never
   lists a virtual generated column at all), plus an index on it.
4. Narrow `get_new_items` (both the FTS and LIKE-fallback branches of
   `_search_items_rows`, and the main path) to an explicit column list
   excluding `content` and `extracted_data`; rewrite the ORDER BY and `since`
   predicate to use the new column + index.
5. Add a DETAIL fetch (`SubscriptionsDB.get_item_content`, threaded through
   `LocalWatchlistsService`/`WatchlistScopeService`/`WatchlistsBackendController`)
   and wire `WatchlistsCollectionsScreen.handle_item_selected` to backfill
   `content` on selection, since the reader's `ContentPane` previously relied
   on the list row already carrying full content.
6. Add a cheap `content_preview` projection (`substr`) for `article_list.
   _render_row`'s snippet, the one list-path reader of `content` found in step 2.
7. Tests first where feasible (ordering-parity, no-content-in-list-rows,
   idempotent-migration-guard); run SubscriptionsDB/Subscriptions/Watchlists
   UI suites; measure before/after latency on a large seeded corpus (isolated
   scratch probe).

## Implementation Notes

**Narrow projection (AC#1).** `get_new_items`/`_search_items_rows` no longer
`SELECT i.*`; they use a new explicit column list
(`SubscriptionsDB._LIST_ITEM_COLUMNS`) excluding `content` (the audit's
named cost) and `extracted_data` -- found during the required column trace:
for an API-type subscription, `LocalWatchlistsService._normalize_api_item`
stores the ENTIRE raw upstream item payload in `extracted_data`, just as
unbounded as `content`, and `normalize_watchlist_item` never mapped it into
its output dict on this query or any other. The trace also found one list-
path reader of `content` that the task's instructions anticipated: `article_
list._render_row`'s 160-char preview snippet. Rather than drag the full
column back in for that, added a cheap `substr(i.content, 1, 2000) AS
content_preview` projection; `_render_row` now prefers it, falling back to
`content` for hand-built test dicts.

There was no existing "fetch one item's content by id" DB method (the
reader previously relied on the list row already carrying full content, so
there had never been a narrow/detail split to bridge). Added `SubscriptionsDB.
get_item_content(item_id)` (single indexed-PK read) and threaded it through
`LocalWatchlistsService` -> `WatchlistScopeService` -> `WatchlistsBackendController`
-> `WatchlistsCollectionsScreen`, mirroring the existing `get_item_status`
pattern at every layer (same routing, same `items.detail` runtime-policy
action -- no new policy registration needed). `handle_item_selected` became
`async` (an established pattern elsewhere in the codebase for `@on`
handlers) and now awaits the content fetch, merging it into the shared item
dict BEFORE setting `ContentPane.item` (a `recompose=True` reactive) -- one
recompose per selection with the body already in it, not two. Deliberately
`Optional[str]`, not raising, at every layer (unlike the sibling
`get_item_status`, which raises `KeyError`): `content` can legitimately be
NULL for an existing row, so "no such row" and "row exists, content is
NULL" both render the same empty body and are not distinguished. A `None`
result leaves a pre-existing `content` key untouched rather than clobbering
it -- which is also what kept the pre-existing synthetic-dict test
(`test_selecting_an_item_renders_it_in_the_content_region`, which bypasses
the DB) green unchanged.

**Indexed ordering (AC#2).** Added `effective_date TEXT GENERATED ALWAYS AS
(COALESCE(datetime(published_date), datetime(created_at))) VIRTUAL` via
`_ensure_watchlists_schema`, plus an index on it, and rewrote the ORDER BY
and `since` predicate to use it. Probed real SQLite (3.49.1) before writing
any of this, per instruction: confirmed NULL/`''`/unparseable
`published_date` all normalize to NULL under `datetime()` (falling back to
`created_at` identically to the old inline expression); confirmed ties sort
by ascending id/rowid today (made explicit in the SQL, `ORDER BY
i.effective_date DESC, i.id ASC`, rather than relying on that implicit
behavior); confirmed `STORED` generated columns CANNOT be added via `ALTER
TABLE ADD COLUMN` ("cannot add a STORED column") -- only `VIRTUAL` can, which
is what was built. A `VIRTUAL` generated column needs no separate backfill
`UPDATE` at all: SQLite computes the value for every pre-existing row
automatically when the index is built over them, and auto-maintains it on
every future INSERT/UPDATE through any write path, present or future --
verified against a hand-built 500-row pre-migration database opened through
the real `SubscriptionsDB`.

**Trap found by the same probe, load-bearing for the migration's own
correctness:** `PRAGMA table_info` does NOT list a virtual generated column
at all; only `PRAGMA table_xinfo` does. Every other column-presence guard in
`_ensure_watchlists_schema` uses `table_info`-sourced `items_cols` -- using
that same set for `effective_date` would have found it "absent" forever and
re-run the `ALTER` on every app start after the first, crashing with
"duplicate column name: effective_date" the second time. The guard reads
`table_xinfo` explicitly; `test_schema_migration_over_effective_date_is_
idempotent` is this trap's own regression test.

**Sort-semantics parity (AC#2/#3).**
`test_ordering_parity_legacy_backfill_vs_new_insert_maintained` builds a
hand-written PRE-migration SQLite database (no `effective_date` column at
all), inserts 5 "legacy" rows spanning valid/NULL/garbage published_date
plus a tied pair, opens it through the real `SubscriptionsDB` (exercising
the actual ALTER + index-build backfill over pre-existing rows), then
inserts 4 more "new" rows through the real `persist_subscription_item` path
-- one deliberately tied with a legacy row. The expected order is not
hand-written: it's the OLD `COALESCE(datetime(...))` expression,
independently recomputed against the live table inside the test itself, so
it would fail if the new column/index ever disagreed with what the query
actually computes. Result: identical ordering over all 9 rows.

**Measurement (AC#3).** Isolated scratch probe (temp SQLite file, no real
data) ran the VERBATIM old SQL text (captured via `git show HEAD:...` since
these changes were uncommitted at measurement time) against the same
seeded, migrated database as the real `get_new_items`: 5,000-item corpus,
7.96ms -> 0.68ms mean (11.7x); 30,000-item corpus, 32.31ms -> 0.73ms mean
(44.1x), 15 repeats each. `EXPLAIN QUERY PLAN` for the real
`WHERE status=? ORDER BY i.effective_date DESC, i.id ASC LIMIT ?` shape
confirms the index is actually used:
`SCAN i USING INDEX idx_subscription_items_effective_date`. The near-flat
NEW latency across a 6x corpus increase versus the OLD query's near-linear
growth is the audit's "O(table) work per refresh" claim made concrete: the
old query scaled with table size regardless of the 100-row LIMIT; the new
one is effectively O(LIMIT).

**Deliberate scope boundaries.** `briefing_selection.py` has its OWN
separate query (`_ITEM_COLUMNS`, a different `SELECT i.*` in a different
module) for selecting items into an LLM-summarized briefing -- legitimately
needs full `content`, left untouched. `get_unread_items_count_since` (a
COUNT-only badge query) also still uses the inline COALESCE form -- out of
the "items list queries" AC's scope; still correct regardless, since
`effective_date`'s existence doesn't change what the inline expression
computes. `categories`/`enclosures` stayed in the narrow projection (small,
unused, but not "large payload" like `content`/`extracted_data`) to avoid
scope creep past what the task and its instructions asked for.

**Tests.** 12 new tests across `Tests/DB/test_subscriptions_db_watchlists.py`
and `Tests/Watchlists/test_watchlists_article_list.py`. Full run (all green,
no regressions): `Tests/DB/test_subscriptions_db_watchlists.py` +
`Tests/Subscriptions/` + `Tests/Watchlists/test_watchlists_article_list.py`
= 869 passed, 1 pre-existing/unrelated skip; the `handle_item_selected`
async surface (`test_watchlists_content_pane.py`,
`test_watchlists_item_actions.py`, `test_watchlists_read_status.py`,
`test_watchlists_inspector.py`, `test_watchlists_items_status_filter.py`) =
137 passed; `test_watchlists_collections_screen.py` = 75 passed;
`test_watchlist_scope_service.py`, `test_watchlists_backend_controller.py`,
`test_watchlists_items_pane.py`, `test_watchlists_rail_counts_and_scope.py`,
`test_watchlists_destination_shell.py`,
`test_watchlists_select_option_overlays.py`,
`test_watchlists_source_row_click_selects.py` = 172 passed. Total: 1,253
passed, 1 skip, 0 failures. Full `Tests/UI/` + `Tests/Watchlists/`
collect-only sweep: 11,013 tests, zero import errors.

**Files modified:** `tldw_chatbook/DB/Subscriptions_DB.py`,
`tldw_chatbook/Subscriptions/local_watchlists_service.py`,
`tldw_chatbook/Subscriptions/watchlist_scope_service.py`,
`tldw_chatbook/Subscriptions/watchlist_normalizers.py`,
`tldw_chatbook/UI/Watchlists_Modules/watchlists_backend_controller.py`,
`tldw_chatbook/UI/Watchlists_Modules/article_list.py`,
`tldw_chatbook/UI/Screens/watchlists_collections_screen.py`,
`Tests/DB/test_subscriptions_db_watchlists.py`,
`Tests/Watchlists/test_watchlists_article_list.py`.

No `Docs/User_Guide/` update: no visible UI change (the reader shows the
same content it always did) -- this is a data-layer/perf change behind the
existing screen.

## Post-merge fix

A later branch's wider test run turned up a defect this task's own suite
selection missed: `_load_item_content` (the DETAIL-fetch loader added
above) is a background `_load*` read whose `except` handler logged the
failure at `debug` with no toast -- `Tests/UI/test_watchlists_check_now_
failure.py::test_background_loaders_pay_for_their_debug_exemption_with_a_
toast`, a structural AST convention check over every `_load*` handler
guarding an awaited call, was RED at merge because of it. Neither my
targeted suite selection nor the review round included that file, so the
gate never ran against this method (that process gap is tracked
separately, not part of this note). Fixed forward on `fix/15464-load-item-
toast`: `_load_item_content` now notifies with `severity="error"` in the
same `except` handler, pattern-matched off the compliant sibling
`_load_items` immediately below it in the same file. Verified: the
convention test is green, the full content-pane/items-pane suite (158
tests) is unaffected, and a manual trace with a forced `get_item_content`
failure confirms the toast actually fires live. (One unrelated,
pre-existing failure was found while re-running this file --
`test_user_initiated_actions_do_not_swallow_failures_into_debug[_delete_
item]`, a stale method name in `USER_INITIATED_MUTATIONS` -- left
untouched: out of this fix's scope, and not something this change caused.)
