---
id: TASK-688
title: >-
  Wire backfill_items_fts so pre-existing items become searchable
status: Done
assignee: []
created_date: '2026-07-25 22:05'
labels:
  - watchlists
  - followup
  - blocks-phase-b
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Phase A (PR #917) added an FTS5 index over subscription_items plus SubscriptionsDB.backfill_items_fts(chunk_size), a chunked and resumable backfill that returns the number of rows indexed and 0 when complete. Nothing calls it — verified: the only reference in tldw_chatbook/ is its own definition.

The index is created empty over a table that may already hold rows, and only the insert/update triggers populate it going forward. So every item a user scraped before upgrading is absent from the index permanently, and Phase C's search returns nothing for their entire back catalogue while appearing to work.

Wiring was deliberately deferred at the final Phase A review so that phase could stay data-layer only (no app.py or UI changes). This is the task that closes it, and it blocks Phase B from claiming search works.

Related: the same missing-index condition was also behind the Phase A Critical where FTS delete triggers rejected mutations of un-indexed rows. That is fixed independently (the delete legs are membership-guarded), so un-indexed rows are now merely unsearchable rather than fatal — but they stay unsearchable until this runs.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Items that existed before the FTS index was created become searchable without any user action
- [x] #2 The backfill runs off the UI thread and never blocks app startup or screen mount, on a database with a large subscription_items table
- [x] #3 An interrupted backfill resumes where it left off rather than restarting
- [x] #4 The backfill is idempotent — running it again after completion indexes nothing and does not corrupt the index (FTS integrity-check stays clean)
- [x] #5 Progress is observable to the user, or at minimum logged, rather than being silently in-flight
- [x] #6 A test covers the upgrade path end to end: a database with pre-existing un-indexed items becomes fully searchable after the wired path runs
<!-- AC:END -->

## Implementation Plan

1. Extract the backfill-to-completion loop into a small, pure, testable helper (`Subscriptions/fts_backfill.py::backfill_subscription_items_fts(db, chunk_size)`) that calls `SubscriptionsDB.backfill_items_fts` in a loop until it returns `0`, logging chunk progress and a completion summary.
2. Wire a thin `TldwCli._backfill_subscription_items_fts()` worker body in `app.py` that opens its own `SubscriptionsDB` (same path/client id the rest of the app uses), delegates to the helper, and never lets an exception escape (catches, logs, closes the connection).
3. Start it from `on_mount()` via `self.run_worker(..., thread=True, exclusive=True, group="subscriptions-fts-backfill")`, alongside the existing `scheduler_worker` / `model-catalog-refresh` startup workers, so it never blocks startup or screen mount.
4. Add a regression test using the established drop-`_ai`-trigger pattern from `Tests/DB/test_subscriptions_db_watchlists.py` to simulate pre-existing un-indexed rows, and verify the wired helper converges to full searchability, is idempotent, and is a no-op when there is nothing to backfill.

## Implementation Notes

Wired via a small dedicated module rather than embedding the loop directly in `app.py`, so the core "drive backfill_items_fts to completion" logic is unit-testable without needing a full Textual `App` harness:

- `tldw_chatbook/Subscriptions/fts_backfill.py` (new): `backfill_subscription_items_fts(db, chunk_size=500)` loops `SubscriptionsDB.backfill_items_fts` to completion, logging per-chunk progress at debug and a completion summary at info (AC #5). Idempotency and resumability (AC #3, #4) fall directly out of `backfill_items_fts`'s own docsize-backed "not yet indexed" check — the loop adds no additional state of its own that could get out of sync.
- `tldw_chatbook/app.py`: added `TldwCli._backfill_subscription_items_fts()` (constructs its own `SubscriptionsDB(get_subscriptions_db_path(), CLI_APP_CLIENT_ID)`, delegates to the helper, closes the connection, and swallows/logs any exception so a backfill failure never crashes the app) and wired it into `on_mount()` via `self.run_worker(self._backfill_subscription_items_fts, thread=True, exclusive=True, group="subscriptions-fts-backfill")`, placed next to the existing `scheduler_worker` and `model-catalog-refresh` startup workers (AC #2 — `run_worker` only schedules the thread and returns immediately, so `on_mount` itself never blocks).
- Each worker thread opens its own `SubscriptionsDB` instance rather than sharing one across threads, matching this codebase's thread-local-connection convention (`ChaChaNotes_DB`, `Client_Media_DB_v2`, and this same class all use `threading.local()`).
- Tests: `Tests/Subscriptions/test_fts_backfill.py` (new) — `test_wired_backfill_makes_preexisting_items_searchable` (12 legacy rows across 3 chunks of 5, all become MATCH-searchable), `test_wired_backfill_is_idempotent_once_complete` (second call indexes 0, FTS `integrity-check` stays clean), `test_wired_backfill_on_already_fully_indexed_db_is_a_noop`. Additionally manually verified the actual `TldwCli._backfill_subscription_items_fts` wrapper end-to-end against a scratch DB path (bypassing the real user data dir) to confirm the app.py wiring itself, not just the helper, works.
- Modified/added files: `tldw_chatbook/Subscriptions/fts_backfill.py` (new), `tldw_chatbook/app.py`, `Tests/Subscriptions/test_fts_backfill.py` (new).
