---
id: TASK-19561
title: >-
  Shutdown is unsound — SIGTERM hard-exits mid-transaction and three
  "don't block exit" mechanisms are dead code
status: In Progress
assignee:
  - '@claude'
created_date: '2026-08-21 20:11'
labels:
  - concurrency
  - shutdown
  - data-integrity
priority: high
dependencies: []
---

## Description

Source: 2026-08-21 holistic review, Lane 4 (concurrency / async / workers) —
its **#3**, **#6** and **#8**. Grouped: one fix locus, the app shutdown path.
Re-verified at this branch base.

**A — SIGTERM skips the entire shutdown.** CONFIRMED.
`app.py:13083` calls `os._exit(0)` from the handler installed at
`app.py:13086` (`signal.signal(signal.SIGTERM, signal_handler)`). `os._exit`
bypasses every cleanup path. Consequences the lane traced: an in-flight
watchlist transaction is abandoned **mid-write**, and a run row is left stuck
at `'running'` **forever** — there is no startup sweep that reconciles it.
This is the *live* path, not a corner case: **SIGINT is consumed by Textual's
raw mode**, so SIGTERM is how this app actually gets terminated.

**B — three "don't block exit" mechanisms are all dead code.**
CONFIRMED BY PROBE — `app.py:11901-11910`, `13040-13051`, `13054-13059`. Each
tries to set `thread.daemon = True` on threads that are **already started**,
which Python forbids: the lane's probe **raised `RuntimeError` and the daemon
flag was unchanged**. One of the three logs an ERROR per thread on **every**
exit, so the mechanism is not merely inert, it is noisy.
Measured consequence: interpreter exit blocked for **3.00 s** after
`asyncio.run` returned at 0.30 s. Compounding it, `app.py:11879` sets
`loop._default_executor = None`, which defeats the graceful executor shutdown
that would otherwise help here.

**C — scheduled briefing generations are invisible to shutdown.** CONFIRMED.
`Scheduling/scheduler/handlers/briefing_handler.py:162` spawns via
`asyncio.create_task(self._run_generation(watchlist_id))`. These are not
Textual workers, so they are absent from the only collection shutdown cancels.
The result is a wedged `'generating'` row whose recovery is UI-gated.

Also worth folding in while here (lane #12, lower severity): a flat 100 ms
shutdown wait, and bare `create_task` calls with no retained reference at
`app.py:10010` and `Voice_Cloning_Window.py:701-719` — a task with no strong
reference can be garbage-collected mid-flight.

Per the owner's standing ruling, prefer the durable fix — make shutdown
actually wait for and cancel what it owns — over widening the hard-exit or
tuning the 100 ms sleep.

## Acceptance Criteria

- [x] SIGTERM runs the ordinary shutdown path: in-flight database transactions
      are completed or rolled back, not abandoned mid-write
- [x] A hard exit remains available as a **last-resort escape after** the
      graceful path has been given a bounded chance, not as the first action
- [x] No run/generation row can be left stuck in `'running'`/`'generating'` by
      a termination; either shutdown reconciles it, or a startup sweep does
      (a startup sweep is the durable option — it also covers power loss)
- [x] The three dead `thread.daemon = True` mechanisms are removed or replaced
      with something that works; nothing logs an ERROR per thread on a normal
      exit
- [x] Interpreter exit is not blocked for seconds after the event loop stops —
      measured, with the before/after numbers recorded
- [x] `loop._default_executor = None` is re-examined against the graceful
      executor shutdown it defeats
- [x] Scheduled briefing generations are tracked somewhere shutdown can see and
      cancel them
- [x] Bare `create_task` results that must outlive their caller retain a strong
      reference

## Implementation Plan

1. Reproduce all three findings against a real running instance under an
   isolated `HOME`/`XDG` and a throwaway database — a real `SIGTERM`, a real
   `RuntimeError` from the daemon assignment, and a measured exit delay.
2. Replace the signal wiring with one process-level handler that asks the
   running Textual app to exit, keeping a hard exit only as a bounded escape.
3. Delete the three dead daemon loops and the `_threads_queues.clear()` calls;
   stop nulling `loop._default_executor`.
4. Add a startup sweep that reconciles interrupted subscriptions rows.
5. Give `BriefingJobHandler` a `shutdown()` seam and call it from `on_unmount`.
6. Replace the flat 100 ms waits with bounded waits that observe their subject;
   retain strong references for the bare `create_task` calls.
7. Re-run every probe against the change and record the before/after numbers.

## Implementation Notes

**What changed.** `Utils/app_shutdown.py` (new) owns process termination. One
`signal.signal` handler, installed once by either entry point, answers the
first `SIGTERM`/`SIGINT` by handing the running app an ordinary `App.exit()`
through `loop.call_soon_threadsafe`; a second signal escalates to an immediate
hard exit. The `os._exit(0)`, the `atexit`-registered `force_cleanup`, and both
`KeyboardInterrupt` daemon loops are gone.

**Where the hard exit went.** Into an *exit watchdog*: a thread created
`daemon=True` (the only moment CPython allows it — which is precisely why the
three removed loops could never have worked) holding a deadline. It is armed at
the start of teardown (`on_unmount`), on a termination signal, and once more
after `app.run()` returns, and it logs which threads are still alive before it
ends the process. Grace is `[general] shutdown_grace_seconds`, default 20 s,
clamped to 1–300. Arming is monotonic — a laxer deadline never relaxes a
running bound, and a tighter one stands the superseded thread down. It refuses
to arm at all until an entry point has called `claim_process_exit()`, so an app
mounted by Textual's `run_test()` cannot schedule the death of its own test
runner.

**Startup sweep, not shutdown reconcile.** `Subscriptions/startup_reconcile.py`
(new) fails every `queued`/`running` `local_watchlist_runs` row and every
`generating` briefing / script / audio row once per launch, off the event loop.
A shutdown-time reconcile could only ever cover terminations the process
survives long enough to run it — never `SIGKILL`, a crash or power loss, which
are exactly the cases that strand a row. Same contract and same documented
two-instance caveat as `AgentRunsDB.reconcile_orphaned_runs`.

**`loop._default_executor`.** The old block called `shutdown(wait=False)` and
then set the private attribute to `None`. Nulling it is what made things worse:
`asyncio.run`'s `Runner.close()` ends with `await loop.shutdown_default_
executor(THREAD_JOIN_TIMEOUT)`, which both marks the loop executor-shut-down
(so a stray late `run_in_executor` raises instead of silently spawning a fresh
pool of unjoinable non-daemon threads) and *joins* the workers while the loop
is alive. With the attribute nulled that coroutine returns immediately. The
whole block is deleted; the public path runs milliseconds later on its own.

**Briefing generations.** `BriefingJobHandler.shutdown()` cancels and settles
`_pending_generations`; `app.py` holds the handler as `_briefing_job_handler`
and calls it from `on_unmount` while the loop can still deliver the
cancellation. The row a cancelled generation leaves is deliberately not
rewritten from inside a cancellation on a closing loop — the startup sweep
reconciles it.

**Waits.** Both flat `asyncio.sleep(0.1)`s became `_cancel_and_settle_workers`,
which cancels and then waits on the workers themselves, bounded by
`WORKER_CANCELLATION_GRACE_SECONDS = 3.0`, naming any stragglers. Built on
`asyncio.wait`, not `wait_for`: `wait_for` implements its deadline by
cancelling what it awaits and then awaiting that cancellation, so work that
swallows `CancelledError` hangs the very call meant to bound it (this wedged a
test run before it was fixed; recorded in `lessons-testing-evidence.md`).

**Strong references.** `_run_no_splash_post_mount_setup` now goes through
`_create_deferred_startup_task`; `Voice_Cloning_Window`'s five action tasks go
through `_spawn_action`, which holds each in `_action_tasks` until it completes.

**Evidence (all against a real instance, isolated `HOME`, throwaway DB).**
- Real `SIGTERM` at base: dead in 0.52 s, no `App Shutdown Requested`, no
  `App Unmounted`, no `app_stopping` diagnostics event; the in-flight
  `db.transaction()` never reached its `finally`, its first statement was
  never committed, and the `running`/`generating` rows survived the kill, a
  restart and a clean quit. After: dead in 6.5 s (it waited for the write),
  both statements committed, `finally` and `atexit` both ran, `app_stopping`
  present, rows reconciled to `failed`.
- `app.run()` itself: the tee'd stdout of the base SIGTERM run contains
  **zero** `--- FINALLY block after app.run() ---` lines from the killed
  process (one, from the later clean restart); the same probe on the fix
  contains **two**. `app.run()` returning is the whole graceful path --
  Textual unmount, `asyncio.run` cleanup, database closes -- completing.
  Note for TASK-19562: none of this rides on `atexit`. `atexit` cannot be
  reached from `os._exit`, which is exactly why the removed
  `atexit.register(force_cleanup)` was never going to help the signal path.
  After this change `os._exit` appears nowhere in `app.py`; the single
  remaining hard exit is `Utils/app_shutdown._hard_exit`, reachable only
  after the grace period or on a second signal.
- Daemon assignment on a started thread: `RuntimeError: cannot set daemon
  status of active thread`, flag unchanged.
- Interpreter exit with one live non-daemon `ThreadPoolExecutor-*` thread:
  base logged `Could not set daemon flag on ThreadPoolExecutor-probe_0` (the
  ERROR-per-thread) and was **still alive 56 s after the loop returned** when
  the probe's 75 s wall limit killed it; after, it exits in **7.9 s** against
  an 8 s configured grace, naming the blocking thread. On a quiet exit the
  delay is unchanged (0.52 s base / 0.57–0.69 s after).

  On the filed 3.00 s figure: a quiet headless exit on this machine measured
  0.52 s at the merge base, not 3.00 s, so that number was not reproduced as
  filed -- it evidently depended on which services the profiled session had
  warmed. The *class* of defect it names is real, and the pathological
  measurement above is what pins it: any live non-daemon thread at exit
  blocks `threading._shutdown()` for as long as it runs, and nothing in the
  removed code bounded that. It is bounded now.

**Modified/added:** `tldw_chatbook/Utils/app_shutdown.py` (new),
`tldw_chatbook/Subscriptions/startup_reconcile.py` (new), `tldw_chatbook/app.py`,
`tldw_chatbook/config.py`, `tldw_chatbook/UI/Voice_Cloning_Window.py`,
`tldw_chatbook/Scheduling/scheduler/handlers/briefing_handler.py`,
`Tests/App/test_app_shutdown.py` (new),
`Tests/Watchlists/test_startup_reconcile.py` (new),
`Tests/Scheduling/test_briefing_handler_shutdown.py` (new),
`backlog/docs/lessons-testing-evidence.md`.

## Independent Review (2026-08-22)

Reproduced end to end against real instances (isolated `HOME`/`XDG`, throwaway
DBs, every probe run as a child process with its own wall-clock kill).

**Confirmed.** The SIGTERM born-red: at the merge base the process died 0.05 s
after `SIGTERM` with no `App Unmounting`, no `app_stopping`, no `--- FINALLY
block ---`, no `atexit`, and the in-flight transaction never committed
(`rows=[]`); on this branch it dies in 3.2 s with every marker present and both
statements committed. The `RuntimeError: cannot set daemon status of active
thread` reproduced four times per exit at base, including on the thread holding
the transaction. The startup sweep reconciled seeded `running`/`queued` runs and
a `generating` briefing to `failed` on the next launch while leaving pre-existing
`failed` history untouched. `asyncio.wait` is genuinely bounded where `wait_for`
would hang: 3.00 s against a worker that swallows `CancelledError`, 0.50 s for
`BriefingJobHandler.shutdown(timeout=0.5)`. The `claim_process_exit` gate holds:
with a real `TldwCli` mounted under `run_test()`, `owns_process_exit` is False,
our handler is not installed, no watchdog thread exists, and a real `SIGTERM`
takes the default action. Gates: `Tests/App` + `Tests/Scheduling` +
`Tests/Watchlists` 1190 passed; `Tests/Watchlists` + `Tests/Subscriptions`
1547 passed / 1 skipped; `Tests/RuntimePolicy` 368 passed (1 failed at the merge
base — the migration-audit doc, whose direction is right: the base has two
research builder call sites and the doc claimed four).

**Adjudicated.** The filed 3.00 s is *not* reproducible here. A quiet clean exit
measures 0.64-0.70 s at the merge base and 0.60-0.61 s on this branch, with no
non-daemon threads alive when the loop returns. The class of defect is real and
is pinned by the measurement below instead.

**Corrected.** `BaseEventLoop.shutdown_default_executor` sets
`_executor_shutdown_called = True` *before* its `if self._default_executor is
None: return`, so the "a stray late `run_in_executor` raises" fence applied at
the merge base too. What nulling the attribute actually cost was the *join*
(plus a real window between `on_unmount` and `Runner.close()` in which a late
`run_in_executor` builds a fresh pool). Removing the block cannot hang an exit
that previously returned: at base the same wait simply happened later and
unbounded, in `threading._shutdown()`.

**New finding (owner call, not fixed here).** The grace period is enforced
against healthy work, not only against a wedged process. A clean quit
(`app.exit()`, no signal) with one ordinary `run_worker(..., thread=True)`
holding an open `BEGIN IMMEDIATE`:

- merge base: died 28.8 s after quit, rc 0, both statements committed
- this branch: died 20.1 s after quit, **rc 1, transaction abandoned**
  (`Shutdown did not finish within the grace period (app unmount)`)

Textual thread workers run on the loop's default executor and cannot be
interrupted, so this is any long ingest/export/embedding batch running when the
user quits. Under `SIGTERM` the branch is still strictly better (base abandoned
the write at 0 s). The trade is the one the ACs ask for, but the default of 20 s
and the silence of the outcome are an owner decision; the docstring claim that
this is "not a deadline anything healthy races" was false and has been corrected,
and `config.toml`'s comment now says plainly what is lost.

**Fixed on this branch by the reviewer.** The monotonic arming rule was not
actually enforced: `arm()` published `self._watchdog` under the lock but started
the thread outside it, and the guard read `is_alive()` — False for a constructed
-but-unstarted thread — so a laxer arm landing in that window relaxed a running
bound (proven by widening the window; a 30 s arm replaced a live 0.5 s one). An
`RLock` does not help, because signal handlers re-enter on the same thread. The
guard is now keyed off an unexpired deadline and `thread.start()` moved inside
the lock; `_watch` additionally retires itself on identity/`None` rather than on
deadline arithmetic alone, so a `stand_down()` landing between its timeout and
its check can no longer lose. Three regression tests added.

## Review Round 2 — owner call taken (2026-08-22)

**`DEFAULT_SHUTDOWN_GRACE_SECONDS` 20 s -> 120 s**, closing the HIGH the
review filed. The reviewer's measurement is the whole argument: a clean quit
(`app.exit()`, no signal) with one ordinary `run_worker(..., thread=True)`
holding an open `BEGIN IMMEDIATE` died at 20.10 s with rc 1 and the
transaction abandoned, where the merge base waited 28.8 s and committed. An
8 s worker committed fine, so the cliff was exactly the grace period. Textual
thread workers end in `loop.run_in_executor(None, ...)` and cannot be
interrupted, and there are ~180 `thread=True` sites — `library_ingest_queue`,
notes/character export, library export, RAG indexing — so this is the live
"quit during a big ingest" case, not a corner.

Why 120 and not a tighter number: the "interpreter exit is not blocked for
seconds" AC is satisfied by the *quiet*-exit measurement (0.60–0.70 s), which
this constant does not touch, because a healthy exit never reaches the
deadline. So tightening it buys nothing and costs a 30 s ingest its write.
120 s still bounds a wedged process to two minutes. A slow quit is an
annoyance; an abandoned transaction is data loss; the owner's standing ruling
is durability over quick. The 1–300 clamp is unchanged.

**Deliberately not done:** extending the deadline when a straggler is
reported. That converts a bound into a suggestion, and it was declined under
the same ruling.

**Signal-handler ordering.** `_handle_termination_signal` read the configured
grace (importing and locking the config module, possibly doing file I/O)
*before* any bound existed. It now arms an unconditional backstop at
`_MAX_GRACE_SECONDS` first and refines with the configured value immediately
after. The backstop is the clamp maximum precisely so the refinement is always
tighter and therefore always accepted by the monotonic rule — arming the
*default* first would have silently bounded a user who configured 300 s at
120 s, which is the abandoned-write direction this whole change is avoiding.

**Regression coverage** (`Tests/App/test_app_shutdown.py`, three tests): a
constant-level guard (`DEFAULT > 30 s`), a scaled live test that a 30 s-class
job holding a real `BEGIN IMMEDIATE` commits within the default grace, and its
red twin showing what a 20 s grace does to the same job. Scaled 1/20 so the
pair costs ~3 s instead of ~2 minutes. Verified red at a 20 s default:
`assert 20.0 > 30.0` and `AssertionError: the watchdog killed a healthy job
mid-transaction / assert [1] == []`.

**Round-2 gates.** `Tests/App` + `Tests/Scheduling` + `Tests/Watchlists` +
`Tests/RuntimePolicy` 1566 passed; `Tests/Subscriptions` 837 passed / 1
skipped; repo-wide `--collect-only -q` 55,015 collected (the reviewer's
55,010 plus this round's five tests). Live re-verified after the change: a
real `SIGTERM` to a real instance still runs the ordinary shutdown path —
died in 6.4 s waiting for the in-flight write, both statements committed,
`transaction_finally_ran` + `atexit_ran` + `app_stopping` all present.

**Not ours, worth filing.** One intermittent failure was seen in a single
four-directory run: `Tests/Watchlists/test_watchlists_artifacts_pane.py::
test_export_feed_press_survives_an_os_error_from_the_service`, with
`WorkerFailed: KeyError("No 'directory-navigation--hidden' key in
COMPONENT_CLASSES")` from inside the third-party `SelectDirectory` picker.
It did **not** reproduce: the identical selection re-run green (1566), the
run with this round's five tests deselected was green (1561), and the test
alone passed 8/8. There is no `pytest-randomly` in this venv, so collection
order is fixed — this is load/state sensitivity in the file-picker path, not
ordering. Nothing in this change can reach it: the harness under test is a
`DestinationHarness`, not `TldwCli`, so no `on_unmount`, watchdog or signal
code runs for it.

## Review Round 3 — Qodo on PR #1972 (2026-08-22)

Two findings, both real, both fixed.

### 1 (HIGH) The startup sweep raced the scheduler — and lost

`on_mount` starts the scheduler worker; the startup reconcile is created
*later*, as a deferred startup task after post-mount setup. `SchedulerLoop.
run()` ticks immediately after loading its queue, so a due watchlist check
launches a real `queued`/`running` row before the sweep ever runs — and the
unscoped sweep failed it as "interrupted". This is worse than the
two-instance exposure the module documented as accepted: that one needs two
processes and self-heals; this one is single-process, on every launch, with
the ordering it depended on enforced nowhere.

**Reproduced against unmodified HEAD**, driving the real `SchedulerLoop` +
real `WatchlistCheckHandler` + real `LocalWatchlistsService` against a
throwaway file-backed `SubscriptionsDB` with only the HTTP fetch blocked, so
the check was genuinely in flight:

```
rows before the sweep: [{'id': 1, 'status': 'running', 'error_msg': None}]
sweep reported: {'runs': 1, 'briefings': 0, 'scripts': 0, 'audio': 0}
rows after the sweep:  [{'id': 1, 'status': 'failed',
                         'error_msg': 'Interrupted: the application stopped ...'}]
RESULT: FAIL - the sweep failed a LIVE in-flight run
```

**Fixed with a boundary, not an ordering rule.** `capture_prior_process_
boundary(db)` records each table's `MAX(id)` at the moment
`_wire_watchlists_and_notifications_services` opens the `SubscriptionsDB` —
inside `TldwCli.__init__`, where no event loop exists, so no scheduler,
handler or UI action can yet have inserted anything. The sweep only touches
rows at or below it.

All four tables declare `id INTEGER PRIMARY KEY AUTOINCREMENT`, whose
`sqlite_sequence` counter never goes backwards and never reuses an id even
after a delete — so *every* row this process creates is provably above the
boundary. That guarantee is load-bearing (a plain `INTEGER PRIMARY KEY` would
reuse the highest freed rowid and silently break the scoping), so it has its
own test: delete the boundary row, insert another, assert the new id is still
higher.

Ordering was the other candidate — move the sweep ahead of the scheduler and
pin it with a test. Rejected: it stays correct only while nobody edits
`on_mount`, which is edited constantly. A boundary captured before the loop
exists cannot be undone by reordering anything after it. `boundary` is a
**required positional argument** on both `reconcile_interrupted_subscription_
work` and `fail_interrupted_watchlist_runs`, so the scoped call cannot decay
back into the unscoped one by omission; and `_reconcile_interrupted_
subscription_work` **skips the sweep entirely** when no boundary is present,
because leaving a row wedged is recoverable next launch and failing a live one
is not. `None` (empty table, or a read that raised) means the same thing and
takes the same path.

The three sibling sweeps (`fail_interrupted_briefings`/`_scripts`/`_audio`)
gained a keyword-only `max_row_id`, defaulting to `None` so every pre-existing
UI-gated caller — which protects live rows with its claim-registry `exclude`
snapshots instead — is unchanged.

**The two-instance exposure is narrowed, not closed**, and the module docstring
now says so: a second instance's boundary still sits above the first
instance's already-running rows. What it *does* buy is that rows the first
instance creates after the second one launches are now spared, where before
they were fair game for the whole of startup. Closing it needs a per-row
process/owner marker, i.e. a schema change this task does not carry.

**Regression coverage.** `Tests/Watchlists/test_startup_reconcile_scheduler_
race.py` (new): the real scheduler launches a real run, the real sweep runs
while it is still in flight, and the run must survive — plus the counterpart
proving a genuinely stranded row is still failed in the same database at the
same moment, and a pin that `TldwCli.__init__` alone (no `on_mount`, no loop)
already holds the boundary. `Tests/Watchlists/test_startup_reconcile.py`
gained the per-table protection, the empty-boundary refusal, the AUTOINCREMENT
guarantee and the required-argument checks.

Mutation-tested, not just asserted: with the bound removed the four boundary
tests go red; with the whole `max_row_id` contract ignored (HEAD semantics)
both scheduler-race tests go red on exactly the live-row assertion.

### 2 (MEDIUM) `install_termination_handlers` could permanently self-disable

It called `claim_process_exit()` and latched `_handlers_installed` *before*
attempting `signal.signal`, and it swallows installation errors. One failure
therefore produced the worst available pair of outcomes at once: no signal
handlers ever (every later call short-circuited on the latch) **and** a live
watchdog able to hard-exit anyway, because the claim had already gone through.

Now nothing is latched until installation actually succeeds:

- **nothing installed** → no claim, no latch, no watchdog, retryable. This is
  the right end state for a legitimately embedded app where `signal.signal`
  can never work (not the main thread): `arm_exit_watchdog` refuses to arm
  without a claim, which is the same inertness `get_app()` already relies on.
- **partial** (one signal installed, one refused) → claim, because a live
  handler now exists whose graceful exit needs a bound; but no latch, so the
  refused signal can still be picked up by a retry.
- **full success** → claim and latch, preserving the documented idempotence
  (the second entry point must not reinstall over live handlers).

**Regression coverage** (`Tests/App/test_app_shutdown.py`, three tests): a
forced `signal.signal` failure claims nothing and arms nothing (asserted by
trying to arm and sleeping past the deadline); a failed attempt is retryable
from a context that can install; a partial install claims but stays
retryable. All three verified red against the HEAD shape —
`AssertionError: an install that installed nothing must not claim the
process's exit`, `a failed install latched, so no later call can ever
succeed`, `the signal that failed must still be retryable` — while the
pre-existing idempotence test stayed green.

### Round-3 gates

`Tests/App` + `Tests/Scheduling` + `Tests/Watchlists` + `Tests/RuntimePolicy`
**1584 passed, 0 failed** (round 2's 1570 plus this round's 14; the round-2
red is gone and the known `test_export_feed_press_survives_an_os_error_from_
the_service` picker flake did not appear). `Tests/Subscriptions` **837 passed
/ 1 skipped**, unchanged. Repo-wide `--collect-only -q` **56,100 collected,
1 error** — `Tests/UI/test_library_file_notes_workspace.py::test_wide_files_
task_return_restores_database_browse_receipt` is parametrized on
`("push_phase", "push_copy", "git_count")` but its signature takes no
arguments. Pre-existing and not ours: this branch has never touched that file
(its last three commits are unrelated file-notes work), and it arrived with
the dev merge that also took collection from 55,015 to 56,100.

`Tests/UI` is not one of this branch's gates and was not run to completion
(14,686 tests), but because this round's `app.py` change lands in
`TldwCli.__init__` — which the UI factories really do construct — the first
1,050 UI tests were run and the 12 failures they produced were A/B'd against
the same six files with the boundary capture removed: **11 failed without it
vs 10 with it**, the same tests either way, one *more* red on the unpatched
side. Load-sensitive pre-existing Console-agent reds, not this change.

Live re-verified after the change: a real `SIGTERM` to a real `TldwCli`
(headless, isolated `HOME`/`XDG`, throwaway DB, child process with the
parent's own wall-clock kill) still runs the ordinary shutdown path — died
**5.39 s** after the signal, waiting for an in-flight `BEGIN IMMEDIATE` thread
worker, **both statements committed** (`rows committed: [1, 2]`), with
`App Unmount`, the post-`app.run()` `finally` and `owns_process_exit=True`
all present, exit code 0.

**Modified/added this round:** `tldw_chatbook/Subscriptions/startup_reconcile.py`,
`tldw_chatbook/Subscriptions/briefing_service.py`,
`tldw_chatbook/Subscriptions/briefing_cast.py`,
`tldw_chatbook/Subscriptions/briefing_audio.py`, `tldw_chatbook/app.py`,
`tldw_chatbook/Utils/app_shutdown.py`, `Tests/App/test_app_shutdown.py`,
`Tests/Watchlists/test_startup_reconcile.py`,
`Tests/Watchlists/test_startup_reconcile_scheduler_race.py` (new),
`backlog/docs/lessons-testing-evidence.md`.
