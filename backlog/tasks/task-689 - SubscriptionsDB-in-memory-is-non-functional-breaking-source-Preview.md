---
id: TASK-689
title: >-
  SubscriptionsDB(":memory:") is non-functional, breaking source Preview
status: To Do
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
- [ ] #1 SubscriptionsDB(":memory:") returns a usable database — the schema is visible on the connection callers actually use
- [ ] #2 A regression test asserts the tables exist and a basic write succeeds against an in-memory instance
- [ ] #3 WatchlistPreviewService.preview() completes successfully against a real source config, end to end, without raising
- [ ] #4 The synthetic subscription id used by preview no longer violates foreign-key enforcement
- [ ] #5 Preview still persists nothing to the user's real database — the isolation the in-memory DB was chosen for is preserved
<!-- AC:END -->
