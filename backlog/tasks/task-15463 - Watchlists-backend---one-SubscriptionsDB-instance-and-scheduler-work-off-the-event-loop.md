---
id: TASK-15463
title: 'Watchlists backend: one SubscriptionsDB instance and scheduler work off the event loop'
status: Done
assignee:
  - '@claude'
created_date: '2026-08-11 12:05'
updated_date: '2026-08-11 13:40'
labels:
  - perf
  - watchlists
  - db
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Probe-verified in the audit: `LocalWatchlistsService._db()` returns `self.db_factory()` (`Subscriptions/local_watchlists_service.py:320-321`; wired `app.py:5984-5988`), so nearly every service method constructs a fresh `SubscriptionsDB` — a ~52-statement `executescript` plus migration probes per call. Measured: 3.4 ms per reconstruction on an EMPTY DB vs 0.04 ms for the same query on a held connection (~85×); first construction 35 ms; a Watchlists screen refresh fires 5+ loads. The class is already thread-safe (`threading.local`, `DB/Subscriptions_DB.py:1133-1135`). Separately, the scheduler's due checks run sync sqlite bookkeeping (`local_watchlists_service.py:802-897`) and full XML/JSON feed parsing (`ET.fromstring`, `Subscriptions/monitoring_engine.py:851-864`) inline on the event loop — enabled by default, firing on any tab (the fetch itself is async httpx and fine; queue bookkeeping is already to_thread).

Fix direction: build one SubscriptionsDB at wiring time and have the factory return the cached instance (keep a factory seam for tests); move the check handler's DB work and feed parse to a thread. Stability constraint: run accounting/receipts have task-2305 history — preserve run records and due-detection semantics exactly. Evidence and method: Docs/Design/2026-08-11-input-latency-audit.md (audit of dev 82b595049; all file:line cites verified there).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Exactly one SubscriptionsDB is constructed per app session in production wiring (evidence); tests keep an injectable factory
- [x] #2 No synchronous sqlite or feed parsing runs on the event loop during a due check (evidence)
- [x] #3 Scheduler behavior — due detection, run records, item upserts, briefings — unchanged (existing surface green)
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Pin current behavior: run the existing watchlists service + scheduler suites green as a baseline.
2. Write the evidence tests first (TDD, red): a counting `db_factory` invoked exactly once across many service ops; a due check driven through `WatchlistCheckHandler` against a real file-backed DB with a sqlite trace callback on the event-loop thread's connection; the feed parse's thread identity.
3. Memoize the DB in `LocalWatchlistsService`; make `db_factory` a property whose setter drops the cache so the injectable test seam still works.
4. `app.py`: build ONE `SubscriptionsDB` in `_wire_watchlists_and_notifications_services` and hand the same instance to the service, the projections, the handlers and the bundle service.
5. Take the due-check path off the loop with `asyncio.to_thread`, one hop per existing sync statement group, in the same order.
6. `FeedMonitor._fetch_and_parse_feed`: run the XML/JSON parse under `asyncio.to_thread` (fetch already async, untouched).
7. Re-run every suite from step 1 green; update only tests that hard-code the factory-per-call shape, with a comment.
8. Self-review, tick ACs, write Implementation Notes, commit.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
**One instance.** `LocalWatchlistsService._db()` now resolves `db_factory` once and caches the result; `db_factory` became a property whose setter drops the cached instance, so the live test seam (`Tests/UI/test_watchlists_inspector.py` repoints a running app's service) still takes effect. `app.py`'s `_wire_watchlists_and_notifications_services` builds ONE `SubscriptionsDB` and shares it with the service, `WatchlistProjection`, `BriefingProjection`, both scheduler handlers and `WatchlistBundleService` (they already shared one eager instance among themselves; the service was the odd one out). A timestamped probe over `SubscriptionsDB.__init__` now shows exactly one construction for a whole app session.

**Off the loop.** New `Subscriptions/db_offload.run_db_off_loop(db, fn, *args)` awaits `fn` on a worker thread — one awaited hop per existing statement group, so ordering and error propagation are unchanged — and runs it INLINE when `db.is_memory_db is True`. That guard is load-bearing: `WatchlistPreviewService` and the in-memory service tests in `test_watchlist_noise_not_volume.py` use `:memory:` databases, where a thread hop would land on a private, empty database (thread-local connections). Hops added: the handler's `get_subscription` and `record_check_error`; the service's `launch_run`, `get_run`, `_mark_run_started`, filter/content-alert loads, item upsert, `record_check_result`, `record_run_result`'s UPDATE and alert-rule read, `record_run_failure`'s `record_check_error`; `URLMonitor`'s baseline read and snapshot write. The feed body parse (`ET.fromstring`/`json.loads`) moved to `asyncio.to_thread` unconditionally — it touches no sqlite. The httpx fetch, the notification dispatch, and `SchedulerLoop`'s already-threaded queue bookkeeping were not touched.

**The trap this surfaced (worth reading).** Caching the instance made two inspector tests fail with `OperationalError: no such table: subscription_items` — deterministically, on a table `sqlite_master` listed on the same connection, and self-healing on retry. A construction/connection timeline showed the cause: the FTS-backfill worker built a SECOND `SubscriptionsDB`, re-running `_initialize_schema` (238 ms) on a worker thread while the app was serving screens, and a connection opened inside that window cached a schema view without the tables it was rewriting. Per-call construction hid it (each call got a fresh connection, hence the pre-existing intermittent flake already documented in that test file); a held connection makes it permanent. Fixed at the source: the backfill worker now uses the app's single instance — thread-local connections are exactly what makes sharing the instance safe. Recorded in `backlog/docs/lessons-testing-evidence.md`.

**Evidence.** New `Tests/Subscriptions/test_watchlists_db_instance_and_off_loop.py` (10 tests): factory called exactly once across eight service operations; reassignment still repoints; a due check for rss/json_feed/url executes ZERO statements on the event-loop thread's connection (probed with `sqlite3.Connection.set_trace_callback`, which is thread-exact rather than timing-based) while still producing its usual completed run receipt; the feed parse runs off the loop thread; the in-memory guard; ordering/error propagation; and a co-scheduled ticker that keeps ticking through a check. `Tests/UI/test_screen_navigation.py`'s task-1631 test was strengthened from path equality to instance identity. Suites green: Tests/Subscriptions + Tests/Scheduling + Tests/DB subscriptions + Home adapter (1177 passed, 1 skipped), Tests/Watchlists (686 passed), Tests/UI watchlists + inspector + screen-navigation.

**Scope.** UI-path reads (`list_sources`/`list_items`/`list_runs`) stay on the loop — they belong to tasks 15461/15462/15464. `URLMonitor`'s CPU work also stays on the loop: the AC covers sqlite and the feed parse, and that extraction is a separate, larger change. **A follow-up task for it is being filed by the controlling session** (not from this worktree — the backlog CLI mangles five-digit ids, and ID assignment needs a repo-wide sweep). It should cover, for `url`/`url_list`/`sitemap` sources: `ContentExtractor.extract_text_from_html` (BeautifulSoup over a page up to `MAX_FETCH_BYTES_PAGE`, `monitoring_engine._fetch_url_content`) and the difflib work in `check_url` (`_segment_for_diff` ×2, `build_change_diff`, `added_and_removed_text`, `classify_change_type`) moved off the event loop — pure CPU, no sqlite, so plain `asyncio.to_thread` applies with no in-memory-connection hazard; a `url_list` source multiplies both by its URL count.

**Review round 1 (fix commit).** Important: `_db()`'s memoization is now guarded by a `threading.Lock` (double-checked). The previous "always primed from the caller's thread" justification was wrong — `list_home_run_snapshot` is synchronous, calls `_db()` itself, and Home runs it under `asyncio.to_thread`, so a worker thread can hit an unprimed cache concurrently with the loop; with a constructing factory that is a double-`_initialize_schema` race, i.e. the very hazard this task removed. Pinned by a test that races eight threads through a deliberately slow factory (fails at 8 constructions unlocked). Minors: `launch_run` now maps the `IntegrityError` its widened check→INSERT window can raise (FK on `local_watchlist_runs.source_id`) back to the documented `KeyError`, with a test (raw `IntegrityError` escapes without it); the `execute_run` cancellation comment now states honestly that the recovery write has ~5 real suspension points and can be interrupted by loop shutdown, and why shielding is not better; `_backfill_subscription_items_fts`'s docstring now says its `close()` runs on a *pooled* executor thread and is safe only because `conn` re-opens lazily.

**Files.** `tldw_chatbook/Subscriptions/db_offload.py` (new), `tldw_chatbook/Subscriptions/local_watchlists_service.py`, `tldw_chatbook/Subscriptions/monitoring_engine.py`, `tldw_chatbook/Scheduling/scheduler/handlers/watchlist_check_handler.py`, `tldw_chatbook/app.py`, `Tests/Subscriptions/test_watchlists_db_instance_and_off_loop.py` (new), `Tests/UI/test_screen_navigation.py`, `Tests/UI/test_watchlists_inspector.py` (stale comments), `backlog/docs/lessons-testing-evidence.md`, `backlog/docs/lessons-backlog-hygiene.md`.
<!-- SECTION:NOTES:END -->
