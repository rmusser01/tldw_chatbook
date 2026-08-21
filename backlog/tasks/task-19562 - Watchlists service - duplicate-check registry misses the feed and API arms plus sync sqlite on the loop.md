---
id: TASK-19562
title: >-
  Watchlists service — the duplicate-check registry misses the feed and API
  arms, 22 async methods do sync sqlite on the loop, and transaction() nests
status: To Do
assignee: []
created_date: '2026-08-21 20:12'
labels:
  - concurrency
  - watchlists
  - db
priority: high
dependencies: []
---

## Description

Source: 2026-08-21 holistic review, Lane 4 (concurrency / async / workers) —
its **#4**, **#9** and **#11**. Grouped: one service, one fix locus.
Re-verified at this branch base.

**A — the in-flight duplicate-check registry misses the two commonest arms.**
CONFIRMED. In
`Subscriptions/local_watchlists_service.py` (around lines 1751 and 1791), the
`url` arm routes through `_check_url_guarded`, which the lane verified
genuinely spans the read-modify-write. The **feed** arm
(`items = await FeedMonitor().check_feed(subscription_config)`) and the **API**
arm do not register in the guard at all — and **feeds are the commonest source
type**. User-visible consequence: when a scheduler tick and a manual "Check
Now" overlap, the **alert notification fires twice** and statistics
double-count.

**B — 22 async service methods do synchronous sqlite on the event loop.**
CONFIRMED. Named: `list_sources:507`, `list_items:602`, `mark_all_read:819`,
`delete_source:958`, `list_runs:1339`, `cancel_run:1419`, plus 16 more. These
are `async def` methods that block the loop for the duration of the query.
**PLAUSIBLE aggravation, and the first step is to confirm it:** `SubscriptionsDB`
sets no `busy_timeout`, so it inherits the 5 s default — meaning a writer
collision could block the event loop for up to 5 seconds. The lane labelled the
5 s stall as plausible rather than observed; measure it before designing around
it.

**C — `transaction()` is not re-entrant and `record_check_result` nests it.**
CONFIRMED-LATENT. `DB/Subscriptions_DB.py:1394` is a plain
`yield conn` / `conn.commit()` context manager with no nesting support, and
`record_check_result` (`:1789`) nests it. The inner `commit()` **durably
commits the outer transaction early**, so a later failure in the outer scope
cannot roll back what the inner one already wrote. The lane rated this LATENT —
it did not find a live call site tripping it — so treat this as a correctness
repair with a regression pin, not an active incident.

Related residue from the same lane worth folding in here (its #12): the
subscriptions DB connection is never closed, leaking a thread-local connection
per worker and never checkpointing the `-wal`; and the scheduler's head-of-line
blocking (`loop.py:173-192`) misreports reminder lateness as "missed while
away".

## Acceptance Criteria

- [ ] The feed and API arms register in the same in-flight guard the URL arm
      uses, so a scheduler tick overlapping a manual "Check Now" runs the check
      once
- [ ] An overlapping check produces exactly one alert notification and one set
      of statistics — pinned by a test that actually overlaps the two triggers
- [ ] The 22 `async def` methods no longer execute synchronous sqlite on the
      event loop
- [ ] The `busy_timeout` question is **measured, not assumed**: either a writer
      collision is shown to stall the loop and a timeout is set, or the concern
      is recorded as refuted with the measurement
- [ ] `transaction()` either supports nesting (savepoints) or
      `record_check_result` stops nesting it; a test pins that a failure after
      the inner scope rolls the whole unit back
- [ ] The subscriptions DB connection is closed on shutdown and the `-wal` is
      checkpointed
- [ ] A guard test fails if a new `async def` in this service performs blocking
      database I/O directly
