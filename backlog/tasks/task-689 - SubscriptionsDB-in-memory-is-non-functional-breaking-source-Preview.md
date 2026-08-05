---
id: TASK-689
title: >-
  SubscriptionsDB(":memory:") is non-functional, breaking source Preview
status: Done
assignee: []
created_date: '2026-07-25 22:05'
labels:
  - watchlists
  - bug
  - followup
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
SubscriptionsDB(":memory:") returns an object whose connection has no tables at all. _initialize_schema builds the schema inside `with closing(self._get_connection()) as conn:` — a connection that is closed immediately afterwards — while the thread-local `.conn` property later opens a *separate* :memory: connection, which in SQLite is an entirely different, empty database.

Verified directly:

    tables visible on .conn: []
    add_subscription: OperationalError: no such table: subscriptions

WatchlistPreviewService.preview() (watchlist_preview_service.py:32) deliberately constructs one — "Use a throw-away in-memory DB so URL snapshots are not persisted" — and the execute path it drives calls db.record_check_result / record_check_error, both of which write to tables. So source Preview / dry-run is very likely broken in production. Its single existing test passes only because it never reaches those calls.

Found while implementing Phase A (PR #917); pre-existing and orthogonal to that work, so it was left out of scope there. It matters for the rebuild because the Phase C Inspector has a Preview action that depends on this path working.

Second, compounding issue for whoever fixes this: Phase A enabled foreign-key enforcement on every SubscriptionsDB connection. watchlist_preview_service.py:57 uses a synthetic subscription id of -1, so once the schema actually exists, any preview write referencing that id will raise IntegrityError. The fix must seed a real parent row or keep preview writes out of FK-bearing tables.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 SubscriptionsDB(":memory:") returns a usable database — the schema is visible on the connection callers actually use
- [x] #2 A regression test asserts the tables exist and a basic write succeeds against an in-memory instance
- [x] #3 WatchlistPreviewService.preview() completes successfully against a real source config, end to end, without raising
- [x] #4 The synthetic subscription id used by preview no longer violates foreign-key enforcement
- [x] #5 Preview still persists nothing to the user's real database — the isolation the in-memory DB was chosen for is preserved
<!-- AC:END -->

## Implementation Plan

1. Reproduce: confirm `SubscriptionsDB(":memory:").conn` has zero tables and `add_subscription` raises `no such table`.
2. Root-cause and choose a fix for `_initialize_schema`'s throwaway-connection pattern; evaluate shared-cache URI + keepalive vs. reusing the thread-local `self.conn` (the approach `ChaChaNotes_DB` already uses), and pick one with an explicit multi-thread trade-off written down.
3. Apply the same fix to `_ensure_watchlists_schema`'s `conn=None` standalone-call branch, so it is consistent.
4. Fix the compounding FK issue in `WatchlistPreviewService.preview()`: seed a real `subscriptions` row in the throwaway in-memory DB instead of using the synthetic id `-1`.
5. Add regression tests: schema/write on a bare in-memory instance, two-instance isolation, and an end-to-end `preview()` call that exercises the real (non-overridden) execute path, including its `url_snapshots` write.
6. Update the one pre-existing test that encoded the old (buggy) close-immediately behavior as an expected invariant.

## Implementation Notes

**Root cause confirmed exactly as described.** `_initialize_schema` ran the schema on `with closing(self._get_connection()) as conn:`, closing that connection immediately, while `.conn` (used by every other method) lazily opens a *different* connection via the same `_get_connection()`. For a file database both calls reach the same file, so it was invisible; for `:memory:`, `sqlite3.connect(':memory:')` is a private, anonymous database per call, so the schema landed on a connection nothing else could ever reach.

**Approach chosen: reuse `self.conn` (the thread-local connection) in `_initialize_schema`, rather than a shared-cache `file::memory:?cache=shared` URI + keepalive connection.** Both fix the bug. Reasons for the simpler one:
- It is the pattern this codebase already uses for the identical problem: `ChaChaNotes_DB._initialize_schema` calls `self.get_connection()` (its own thread-local accessor), not a throwaway one. Matching precedent beats inventing a second mechanism for the same class of bug.
- The only current `:memory:` caller, `WatchlistPreviewService.preview()`, constructs, uses, and discards its `SubscriptionsDB` instance within a single coroutine on one thread (scheduled via `run_worker(coroutine)`, not `thread=True` — confirmed by reading `watchlists_collections_screen.py`'s `self.run_worker(self._preview_source(entity), exclusive=True)`), so it never crosses a thread boundary.
- A shared-cache URI adds real failure modes of its own (the shared in-memory database is destroyed the instant its last connection closes, so the keepalive connection's lifecycle becomes another thing that must never be gotten wrong) for a multi-thread need nothing in this codebase currently has.

**Trade-off, made explicit rather than silently accepted:** if a *second* thread later touches `.conn` on the same in-memory `SubscriptionsDB` instance, that thread's own thread-local slot is empty, so it lazily opens yet another private, schema-less `:memory:` connection — the identical limitation `ChaChaNotes_DB` already carries for the same reason. Documented in the `_initialize_schema` docstring; a future caller that needs a genuinely thread-shared in-memory instance would need the shared-cache-URI approach instead.

Applied the same reasoning to `_ensure_watchlists_schema`'s `conn=None` branch (previously also `with closing(self._get_connection()) as conn: ...; return`) so the standalone call path (used directly by an existing file-DB idempotency test) is consistent and in-memory-safe too.

**FK fix:** `WatchlistPreviewService.preview()` now seeds a real `subscriptions` row via `preview_db.add_subscription(...)` and overwrites the synthetic `id: -1` with the real returned id before calling `_execute_subscription`. This is written into the same throwaway in-memory `preview_db`, so isolation from the user's real database is unaffected (AC #5) — confirmed by grep: `FeedMonitor.check_feed` (rss/atom/json_feed/podcast) makes no DB writes at all; only `url`/`url_list`/`sitemap` sources reach `URLMonitor._store_snapshot`'s `INSERT INTO url_snapshots`, which is the write this seed makes possible.

**Pre-existing test updated, not just left broken:** `Tests/Subscriptions/test_subscriptions_smoke.py::test_subscriptions_db_closes_schema_initialization_connection` asserted the old bug's behavior directly ("the schema-init connection gets closed immediately"). Investigated its git history (`ec489c052`): it was added to guard against a real, separate leak (`with self._get_connection() as conn:` alone does not close a sqlite3 connection — only wraps a transaction — so the *original* pre-`closing()` code leaked a connection every construction). Renamed and rewritten as `test_subscriptions_db_schema_initialization_reuses_and_closes_connection`, asserting the corrected invariant: exactly one connection is opened for schema init, it is not closed immediately, and it *is* closed by `db.close()` — preserving the original no-leak guarantee without reintroducing the in-memory bug.

**Tests added:**
- `Tests/DB/test_subscriptions_db.py`: `test_in_memory_db_has_usable_schema`, `test_in_memory_db_instances_stay_isolated`, `test_ensure_watchlists_schema_idempotent_on_in_memory_db` (AC #1, #2).
- `Tests/Subscriptions/test_watchlist_preview_service.py`: `test_preview_url_source_completes_end_to_end_without_raising` — no `run_executor` override, so it drives the real default executor's `URLMonitor(db).check_url()`, including the `url_snapshots` write, with only the network fetch (`guarded_fetch_httpx_async`) mocked out (AC #3, #4).
- `Tests/Subscriptions/test_subscriptions_smoke.py`: updated as described above.

**Verification beyond pytest:** manually confirmed two live `SubscriptionsDB(":memory:")` instances each see 17 tables and stay isolated (one instance's `add_subscription` write is invisible to the other), and confirmed the real config file (`~/.config/tldw_cli/config.toml`) was untouched throughout (mtime checked before/after every command; `TLDW_CONFIG_PATH` pointed at a scratch file for anything that imports `tldw_chatbook.config`/`app.py`).

**Scope boundary respected — but flagging a related risk, per the task's instruction.** Did not touch `Library_Collections_DB`, `Workspace_DB`, or `AgentRuns_DB`. Checked each directly: all three use `with closing(self._get_connection()) as conn: yield conn` as a `connection()` *context manager* used fresh on every single call site, not a cached `.conn`/thread-local property. None has a production `:memory:` caller (grep for `LibraryCollectionsDB(`, `WorkspaceDB(`, `AgentRunsDB(` outside `Tests/` shows only real file paths). So today, nothing is broken. But if any of them were ever pointed at `:memory:`, the failure mode would arguably be *worse* than this task's bug: every single method call opens a brand-new, private, empty `:memory:` connection, so not even `_initialize_schema`'s own schema would survive to the next call, let alone a second thread. Not fixing this speculatively (no caller needs it, and "fix a bug with no reproduction" is exactly the kind of scope creep the task warns against) — filing this observation per the instruction rather than silently expanding scope or silently doing nothing.

**Modified/added files:** `tldw_chatbook/DB/Subscriptions_DB.py`, `tldw_chatbook/Subscriptions/watchlist_preview_service.py`, `Tests/DB/test_subscriptions_db.py`, `Tests/Subscriptions/test_watchlist_preview_service.py`, `Tests/Subscriptions/test_subscriptions_smoke.py`.
