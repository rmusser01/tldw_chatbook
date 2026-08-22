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
