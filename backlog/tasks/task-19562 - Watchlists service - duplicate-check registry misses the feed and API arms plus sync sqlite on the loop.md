---
id: TASK-19562
title: >-
  Watchlists service — the duplicate-check registry misses the feed and API
  arms, 22 async methods do sync sqlite on the loop, and transaction() nests
status: Done
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
- [x] The `busy_timeout` question is **measured, not assumed**: either a writer
      collision is shown to stall the loop and a timeout is set, or the concern
      is recorded as refuted with the measurement
- [x] `transaction()` either supports nesting (savepoints) or
      `record_check_result` stops nesting it; a test pins that a failure after
      the inner scope rolls the whole unit back
- [x] The subscriptions DB connection is closed on shutdown and the `-wal` is
      checkpointed
- [x] A guard test fails if a new `async def` in this service performs blocking
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

**The specific claim in C was recorded as REFUTED by measurement. That
recording was itself wrong -- see the close-out section below.**
`record_check_result` DOES nest, on every real check. The original note read:
"does not nest today: instrumenting `transaction()` across a real call
observed depth 1, one entry".

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

**Was still open at that point:** the C residue (leaked thread-local
connection, un-checkpointed `-wal`), the `busy_timeout` AC, and the
new-`async def` guard test. All three are closed below.

## Close-out (2026-08-22) — the four remaining ACs, and two corrections

### Hygiene: what the pre-ticked boxes were actually worth

The file arrived with four ACs ticked while the status still read To Do.
Each was re-derived against the code rather than trusted:

| AC | Verdict |
|---|---|
| feed/API in the same guard | **Genuinely done.** `_run_guarded_source_check` claims through `_IN_FLIGHT_URL_CHECKS` with a NUL-prefixed sentinel slot; both arms route through it. |
| one alert + one set of statistics, pinned by an overlapping test | **Partly.** The overlap tests were real, but they asserted the *fetch* count and the skip disposition -- **neither the notification count nor the statistics**, which is what the AC names and what the user sees. Closed here. |
| 22 async methods off the loop | **Genuinely done**, and re-derived independently: an AST sweep of every `async def` in the service finds **zero** direct database calls. |
| `transaction()` nesting + rollback pin | **Genuinely done** as a mechanism. The *claim recorded alongside it was false* -- see below. |

### AC2 — the symptom the task names, now pinned where it is visible

`test_overlapping_feed_check_alerts_once_and_counts_statistics_once` drives a
real overlap (a gated `check_feed`, the second entrant starting inside the
first's await) with a real alert rule and a recording notification
dispatcher. Born-red against the unguarded feed arm:

    the alert notification fired 2 times for one overlapping check
    statistics double-counted the overlap:
      {'checks_performed': 2, 'successful_checks': 2,
       'new_items_found': 2, 'items_ingested': 2}

Green after: one dispatch, one set of daily statistics.

### AC4 — busy_timeout: reproduced, and it did not lead where it looked

Measured on a real `SubscriptionsDB` before designing anything:

    busy_timeout (ms): 5000        <- inherited, nothing set it
    journal_mode     : wal
    second writer blocked for 1.07s -> acquired   (lock held 1.0s)

The collision is real and the caller does wait, up to the 5 s ceiling. But
the measurement also refuted the obvious fix: **5000 is already what the
connection had**, so "set a busy_timeout" changes nothing on its own, and
*lowering* it would only convert a stall into an earlier `database is
locked` on a path with no retry. The stall stops mattering because part B
moved the sqlite off the loop.

What shipped is therefore drift-protection, not a behaviour change:
`BUSY_TIMEOUT_MS = 5000` is applied explicitly (before the WAL conversion,
per `AgentRuns_DB`'s ordering rationale) and pinned by a test that survives
a connector default change -- red at 0 ms with the pragma removed. The value
assertion alone would pass without the pragma, and the test says so rather
than posing as a guard.

Two comments elsewhere asserting "`SubscriptionsDB` sets no `busy_timeout`"
(`watchlists_collections_screen.py`, `briefing_handler.py`) are corrected.

### AC6 — connection close and `-wal` checkpoint, with one half refuted

Shipped: a per-instance connection registry (thread ident -> connection),
`checkpoint_wal()`, `close_all_connections()`, and a `close()` that now
checkpoints before closing. `close()`'s single-thread scope is unchanged --
`app.py`'s FTS backfill depends on it.

Two things were measured rather than assumed:

* **A cross-thread close is refused by sqlite3** (`ProgrammingError: SQLite
  objects created in a thread can only be used in that same thread`), so
  `close_all_connections` closes this thread's connection, checkpoints the
  file for everyone, and *reports* the rest instead of raising during
  shutdown. Pinned as a fact about the runtime.
* **The "`-wal` is left behind at exit" half is REFUTED for a clean exit.**
  A child process wrote a 4.1 MB `-wal` and exited normally; only `subs.db`
  remained -- identically with the new `atexit` hook enabled and suppressed.
  CPython finalizes the connections and SQLite removes the `-wal` on last
  close. What is real is the *standing* 4 MB a long-running app carries
  (now truncated by `close()`/`close_all_connections`), and the
  `os._exit(0)` signal path, which no `atexit` hook can reach -- that is
  task-19561's subject and is recorded, not papered over. The hook stays
  for a defined settle point with a defined error path, and is tested for
  what it actually does.

### AC7 — a guard for `async def` that does not exist yet

`test_watchlists_service_no_blocking_db_io.py` parses the service and
rejects any database call reached directly from an `async def`, resolving
attribute chains to their root so `db.conn.cursor()` -- the shape the
original 19-of-22 sweep missed -- is caught too. Proven able to fail three
ways: on synthetic modules, and on the real file with `list_sources`'
offload reverted (`list_sources (line 512): db.get_all_subscriptions(...)`).

### Correction: `record_check_result` DOES nest, on every real check

The earlier note recorded C as refuted at the live call site. Re-instrumented
per argument shape:

    record_check_result WITH stats    -> 2 entries, depths [1, 2]
    record_check_result WITHOUT stats -> 1 entry,  depths [1]

The route is `record_check_result` -> `_update_subscription_stats` ->
`update_subscription_stats`, which opens its own transaction for the
`subscription_stats` upsert. `execute_run` always passes stats, so this is
the ordinary path; the earlier measurement can only have taken `stats=None`.
The lane's hazard was **live**, not latent -- the daily-statistics write was
durably committing the enclosing subscription-health UPDATE. No incident
followed only because nothing after that point in `record_check_result` can
fail. Two new pins cover the real call site, red against the unnested
manager: *"the nested statistics write survived a failure in the enclosing
transaction"*.

### Folded in: the scheduler's lateness misreport

`SchedulerLoop.tick` awaits every due handler serially and inline, so one
slow handler (a watchlist check may run to its 300 s execution timeout,
against a 60 s missed-fire grace) pushes every task behind it past the
grace. The row that results is identical to one from an app that was closed
-- and the UI said *"Missed while away ... (the scheduler was not running at
the scheduled time)"*, a cause the app cannot know and which is simply false
in that case.

Repaired without a schema change, because the row's facts were never the
problem:

* `missed_at`/`missed_count` stay -- the occurrence WAS owed late and
  earlier ones really were skipped, whichever cause it was.
* The UI copy becomes "Ran late: ... (the app was closed, or the scheduler
  was busy with an earlier task)". A test asserts the notice never contains
  "Missed while away" or "was not running", across all three branches.
* The cause is recorded where it is actually known: `SchedulerLoop` tracks
  `_running_since` and `_report_lateness_cause` logs and counts
  `scheduler_dispatch_late{cause=busy|away}`. A test drives the real
  head-of-line block -- a slow handler advancing the clock ten minutes --
  and asserts the following dispatch is attributed `busy`.
* The docstring on `mark_reminder_dispatched`, the `[scheduling]` config
  comment and `Docs/User_Guide/schedules.md` all asserted the false premise
  and are corrected.

The blocking itself is deliberately NOT changed to concurrent dispatch: the
codebase already chose per-handler spawning for the slow case (see
`BriefingJobHandler`'s Locked Decision 3), and making every handler
concurrent is a semantics change this task has no mandate for.

### Tests

* `Tests/Subscriptions/` + `Tests/Scheduling/` +
  `Tests/UI/test_schedules_missed_notice.py` — **1187 passed, 1 skipped**,
  against **1156 passed, 1 skipped** for the same selection in a clean
  `origin/dev` (`da4e828af`) worktree: +31 tests, no regressions.
* `Tests/Watchlists/` + the three `Tests/UI/test_schedules_*` files — 758
  passed.
* `Tests/DB/` — 1071 passed, 1 skipped, and 6 failures in
  `test_core_sqlite_owner_privacy.py[media-*]`. Those 6 fail identically on
  `origin/dev` in the clean worktree — pre-existing dev reds, not from this
  branch.
* Repo-wide `pytest --collect-only -q` — 55,026 tests collected, 0
  collection errors.
* `ruff check` clean on every changed module.

### Files

`tldw_chatbook/DB/Subscriptions_DB.py`,
`tldw_chatbook/Scheduling/scheduler/loop.py`,
`tldw_chatbook/Scheduling/db/scheduled_tasks_db.py`,
`tldw_chatbook/Scheduling/scheduler/handlers/briefing_handler.py`,
`tldw_chatbook/UI/Screens/scheduling/task_detail.py`,
`tldw_chatbook/UI/Screens/watchlists_collections_screen.py`,
`tldw_chatbook/config.py`, `Docs/User_Guide/schedules.md`, and tests:
`Tests/Subscriptions/test_subscriptions_db_connection_lifecycle.py` (new),
`Tests/Subscriptions/test_watchlists_service_no_blocking_db_io.py` (new),
`Tests/Scheduling/test_scheduler_lateness_cause.py` (new),
`Tests/Subscriptions/test_watchlist_feed_api_in_flight_guard.py`,
`Tests/Subscriptions/test_subscriptions_transaction_nesting.py`,
`Tests/UI/test_schedules_missed_notice.py`.
