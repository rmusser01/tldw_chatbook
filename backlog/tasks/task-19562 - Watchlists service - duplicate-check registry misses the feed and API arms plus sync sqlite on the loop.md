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

- [x] The feed and API arms register in the same in-flight guard the URL arm
      uses, so a scheduler tick overlapping a manual "Check Now" runs the check
      once
- [x] An overlapping check produces exactly one alert notification and one set
      of statistics — pinned by a test that actually overlaps the two triggers
- [x] The 22 `async def` methods no longer execute synchronous sqlite on the
      event loop
- [ ] The `busy_timeout` question is **measured, not assumed**: either a writer
      collision is shown to stall the loop and a timeout is set, or the concern
      is recorded as refuted with the measurement
- [x] `transaction()` either supports nesting (savepoints) or
      `record_check_result` stops nesting it; a test pins that a failure after
      the inner scope rolls the whole unit back
- [ ] The subscriptions DB connection is closed on shutdown and the `-wal` is
      checkpointed
- [ ] A guard test fails if a new `async def` in this service performs blocking
      database I/O directly

## Measurement: the busy_timeout question (2026-08-21)

One AC asks that this be **measured, not assumed**. Measured on this branch's
base; the lane's PLAUSIBLE rating is **confirmed, with one narrowing**.

    SubscriptionsDB connection busy_timeout (ms): 5000
    journal_mode: wal
    second writer blocked for 1.08s -> acquired   (lock held 1.0s)

* No `busy_timeout` is set anywhere in `Subscriptions_DB.py`, `base_db.py`
  or the private-path connector, so the connection inherits Python
  sqlite3's **5 s** default. The task's premise is correct.
* A writer collision really does block the caller for as long as the lock is
  held, up to that 5 s ceiling, and then raises `OperationalError`. On the
  event loop that stall is the entire UI.
* **Narrowing worth carrying into the design:** `journal_mode = WAL`, so
  readers do *not* block writers and writers do *not* block readers. The
  exposure is **writer-vs-writer only** — not "any of the 22 async methods",
  as the B-section wording could be read to imply. The read-only methods
  (`list_sources`, `list_items`, `list_runs`, ...) still block the loop for
  their own query duration, but they cannot be *stalled* by a concurrent
  writer.

So both halves are real but distinct: (1) every one of the 22 blocks the loop
for its own duration -- fix by moving the sqlite work off the loop; (2) the
*write* paths can additionally stall up to 5 s behind another writer -- and
lowering `busy_timeout` only converts that stall into an earlier exception, so
it is not a substitute for (1).

Probe: two connections to a real `SubscriptionsDB`, one holding
`BEGIN IMMEDIATE` for 1 s while the other times its own `BEGIN IMMEDIATE`.

**Scope note:** this task is an arc, not a quick win -- 7,117 lines across
`local_watchlists_service.py` (40 `async def`) and `Subscriptions_DB.py`,
bundling a user-visible correctness bug (duplicate alert notifications), the
22-method loop-blocking fix, a non-reentrant `transaction()` that
`record_check_result` nests, and a leaked thread-local connection. Recommend
splitting: (A) the in-flight guard + duplicate-alert test, (B) the sqlite
offload, (C) transaction nesting + connection close.

## Part A shipped (2026-08-21) — B and C still open

**A (feed/API in-flight guard) is done.** Both arms now claim through the same
`_IN_FLIGHT_URL_CHECKS` registry the url-family arms use, scoped to the whole
source with a NUL-prefixed sentinel in the URL slot (no real URL can contain
NUL, so it cannot collide with a url-family claim for the same subscription).

The skip returns a `DISPOSITION_SKIPPED_IN_FLIGHT` disposition rather than the
arm's usual `None`. That turned out to be the load-bearing half: without it,
`_entirely_skipped_dispositions` would not match, `execute_run` would run the
ordinary health path, and a turned-away check would take
`record_check_result`'s SUCCESS branch -- resetting the auto-pause breaker,
clearing `last_error` and stamping `last_successful_check` for a run that
never contacted the source. Returning `[]` items with `None` dispositions
would have been indistinguishable from a clean "nothing new" check.

Two comments in the file asserted the old invariant ("feed/API runs can never
skip", "`None` for the feed and API arms") and are corrected, not left to rot.

Red-proofed on behaviour, not vocabulary: against the unfixed code the forced
interleave fetches the feed **2 times for one overlap**. `Tests/Subscriptions/`
755 passed.

**C also shipped (2026-08-21).** `transaction()` now tracks nesting depth per
thread -- only the outermost block commits or rolls back -- mirroring
`ChaChaNotes_DB`'s `TransactionContextManager` rather than inventing a
savepoint scheme. Depth is cleared in a `finally`, so a raise cannot strand it
and silently turn every later transaction into a no-op joiner.

**The specific claim in C is REFUTED by measurement.** `record_check_result`
does not nest today: instrumenting `transaction()` across a real call observed
**depth 1, one entry**. The lane's CONFIRMED-LATENT rating was right about the
hazard and wrong about that call site. The fix stands anyway -- it converts a
silent-partial-persistence trap into a structural impossibility instead of
relying on nobody ever nesting -- and is red-proofed: against the unfixed
context manager, the nested write survives a deliberate failure in the outer
scope.

**B also shipped (2026-08-21).** All 22 async methods now route their sqlite
through the existing `db_offload.run_db_off_loop` helper rather than a new
mechanism. The four `transaction()` sites were handled differently on purpose:
that helper's contract forbids holding a transaction open across the thread
boundary, so each block was extracted whole into an offloaded function rather
than offloading statements inside an open transaction.

Count note: an AST sweep for `db.<method>()` / `self._db().<method>()` found
19 methods; three more (`get_alert_rule`, `list_runs`, `list_alert_rules`)
reach the database through a bare `db.conn.cursor()` and were invisible to
that pattern. Adding them gives 22 -- matching this task's original figure,
which the narrower scan would have under-reported. Verified after the change:
**zero** inline db calls remain in any `async def` in the file.

Red-proofed: reverting `cancel_run`'s offload fails its test with
`cancel_run's transaction opened on the event-loop thread`. `Tests/
Subscriptions/` 781 passed (from a 759 baseline, +22 new tests).

**Still open:**

* **C residue** -- the leaked thread-local connection that is never closed and
  never checkpoints the `-wal`. Untouched here.
