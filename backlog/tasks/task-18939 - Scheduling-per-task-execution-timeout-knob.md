---
id: TASK-18939
title: 'Scheduling: per-task execution timeout knob'
status: Done
assignee:
  - '@robert'
created_date: '2026-08-19 11:05'
updated_date: '2026-08-19 11:05'
labels:
  - scheduling
  - parity
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Close the timeout-configurability gap found by the TASK-18936 parity audit (hermes's cron hardening made timeout configurable and derived run-claim TTL from it; chatbook's only scheduling knob is `[scheduling] scheduler_poll_interval_seconds`). Add a per-task (with a global default) execution timeout: a dispatched task whose handler exceeds the bound is marked failed-with-timeout rather than wedging the loop or hanging indefinitely. Note the current dispatch reality before designing: `SchedulerLoop.tick` awaits handlers sequentially, and `BriefingJobHandler` is deliberately fire-and-forget — so the timeout must first characterize where long-running work actually blocks (a wedged `watchlist_job`, a future synchronous agent-task handler) and apply the bound at the right seam, not blindly wrap every handler in `asyncio.wait_for`. Manual "Run now" (TASK-18938) and missed-fire accounting (TASK-18937) should consume the same timeout outcome.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A global default timeout plus per-task override exist in config (naming follows existing `[scheduling]` conventions) with a documented default and 0/negative semantics pinned (disable vs unlimited — pick and test one) — `[scheduling] handler_timeout_seconds` (default 300; ≤0 disables) + per-task `reminder_tasks.timeout_seconds` column (schema v3; NULL = default, ≤0 = per-task opt-out); resolution precedence pinned by `test_effective_timeout_resolution_rules`
- [x] #2 A characterization of which handlers can actually exceed a timeout is recorded in the task (which are fire-and-forget, which block the tick) before the bound is applied — recorded in the plan section: tick awaits sequentially on the event loop; reminder is fast/synchronous; watchlist_job genuinely awaits network I/O (the real risk); briefing_job is fire-and-forget by design (spawns `create_task`, returns instantly — timeout naturally doesn't apply to dispatch, and the generation itself is out of scope, noted)
- [x] #3 Timed-out dispatches produce a distinct, honest terminal state (`last_status` value naming timeout) and never wedge the poll loop; the next occurrence still computes correctly afterward — `TaskStatus.TIMED_OUT` / `last_status="timed_out"`; `test_slow_handler_times_out_and_records_timed_out` (bounded tick time + next occurrence advanced) and `test_timeout_does_not_wedge_subsequent_tasks`
- [x] #4 Run-now and missed-fire accounting treat a timeout as a failed dispatch, not a missed one (ran-and-raised vs never-ran distinction shared with TASK-18937) — `dispatch_reminder` returns False on timeout (Run-now treats it as failure: `test_run_now_shares_the_timeout`); timed_out/missed/completed/missed-while-away all distinct, pinned by `test_timed_out_vs_missed_distinct_statuses`; the retry affordance covers TIMED_OUT alongside MISSED
- [x] #5 Tests use the real loop/handler shapes (a deliberately slow handler in tests) and pin loop liveness after a timeout — 12 tests in `Tests/Scheduling/test_handler_timeout.py`, all real dispatch-seam paths with `asyncio.sleep(10)` handlers
- [x] #6 Docs: schedules.md and the config reference document the knob and default — "Execution timeouts" section in schedules.md; commented knob in config.py's `[scheduling]` defaults
<!-- AC:END -->

## Implementation Notes

Implemented 2026-08-19 in `.worktrees/hermes-parity-audit` (branch `task/hermes-parity-audit`, on top of 18937/18938).

**Approach.** The characterization (recorded before implementation) showed `tick` awaits handlers sequentially on the event loop, `watchlist_job` genuinely awaits network I/O (the wedging risk), and `briefing_job` is fire-and-forget by design. Decision: one uniform rule rather than per-handler special cases — `dispatch_reminder` wraps EVERY handler await in `asyncio.wait_for` with the effective timeout (task-row `timeout_seconds` override → `[scheduling] handler_timeout_seconds` default; ≤0 at either level disables). The briefing handler's instant-return shape is naturally unaffected; its spawned generation is deliberately out of scope (a briefing-lifecycle concern with its own seams).

**Terminal state.** `TaskStatus.TIMED_OUT` (`last_status="timed_out"`) — ran-but-cancelled, distinct from `completed`, `missed` (ran and raised), and missed-while-away (never ran). `mark_reminder_dispatched` gained a `timed_out` flag; the schedule advances on timeout, so a wedged handler cannot wedge the loop. The detail pane labels it "Timed out" (warning styling, same tier as Missed) and offers "Run now (retry)".

**A real bug the new test exposed:** re-applying the v1→v2 migration to a v3 database *downgraded* the schema version row to 2. Both v1_to_v2 and v2_to_v3 are now forward-only — `DELETE`+`INSERT` happens only when the current version is below the migration's target, so a stale/mixed caller can never move the version backward. This was latent before this task (nothing re-ran old migrations); the idempotency test made it visible.

**Verification.** `Tests/Scheduling/` fully green (**308 passed**, +12 in `test_handler_timeout.py`: schema v3 up/rollback, timeout+advance, loop liveness, per-task override winning, per-task and global disable, fast-handler unaffected, timed-out vs raised distinctness, Run-now sharing the seam, resolution precedence). `Tests/UI -k sched` green (**96 passed**). Config/app wiring parse-checked. Not verified against a live TTY session (headless worktree) — noted honestly.

**Server-side note (from the TASK-18940 survey):** the timeout semantics here align naturally with tldw_server's Jobs pipeline run-lifecycle/SLA plumbing; ADR-071 should map `timed_out` to the server's run-status vocabulary rather than inventing a second one.

**Files modified:** `tldw_chatbook/Scheduling/db/migrations/v2_to_v3.py` (new), `v1_to_v2.py` (forward-only fix), `scheduled_tasks_db.py` (column, timed_out flag), `models.py` (TIMED_OUT enum, timeout_seconds field), `scheduler/loop.py` (wait_for seam + resolution), `services/scheduling_service.py` (None-safe mapping), `UI/Screens/scheduling/task_detail.py` (labels, badge classes, retry affordance), `app.py` + `config.py` (knob wiring), `Tests/Scheduling/test_handler_timeout.py` (new) + version assertions in `test_migrations.py`/`test_missed_fire.py`/`test_scheduled_tasks_db.py`, `Docs/User_Guide/schedules.md`.

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no.
ADR path: N/A.
Reason: bounded-execution guard within the existing module; ADR-031-style conventions and ADR-018's module boundary are untouched.

**Handler-blocking characterization (2026-08-19, from the code — AC#2):**

- `SchedulerLoop.tick` awaits each handler **sequentially on the event loop** (the loop's own docstring in `watchlist_check_handler.py:143` pins this: "SchedulerLoop awaits this directly on the event loop"). A handler that never returns wedges the whole scheduler — every task type, forever.
- **`reminder`** (`ReminderHandler`): dispatch to `NotificationDispatchService` — synchronous, fast; a wedged notification store is conceivable but not the observed failure class.
- **`watchlist_job`** (`WatchlistCheckHandler.handle`): genuinely awaits network I/O (feed/URL checks via `watchlists_service.launch_run`/`execute_run`, `feed_monitor.check_feed`, `url_monitor.check_url`). This is the handler a timeout must bound — a hung remote can stall the loop indefinitely.
- **`briefing_job`** (`BriefingJobHandler.handle`): **fire-and-forget by design** — spawns `asyncio.create_task(self._run_generation(...))` with a strong-reference set and returns immediately; the generation itself (LLM tokens, potentially minutes) runs outside the tick. Wrapping THIS handler's `await` in a timeout bounds nothing (it returns instantly); the long work is deliberately unawaited. Timeout therefore does not apply to briefing dispatch, and must not be retrofitted onto the spawned generation in this task (that is a briefing-lifecycle concern, separate seams — noted as out of scope).

**Decision:** enforce the timeout at `dispatch_reminder`'s seam generically — `asyncio.wait_for(handler(task), timeout)` for EVERY handler type (uniform, no per-handler special cases), with per-task override resolved from the task row's new optional `timeout_seconds` column falling back to `[scheduling] handler_timeout_seconds` (default **300**; `0` or negative = unlimited). The briefing handler's instant-return shape is naturally unaffected (its await completes immediately); a genuinely slow reminder/watchlist handler gets cancelled, recorded as the new terminal `last_status="timed_out"`, and the loop moves on. Cancellation is cooperative: the task's next occurrence still computes, so the schedule never wedges.

Implementation steps:

1. Schema v2→v3: `reminder_tasks.timeout_seconds` column (nullable; NULL = use default)
2. `SchedulerLoop.dispatch_reminder`: resolve effective timeout (task row override → loop default), `asyncio.wait_for` around the handler await, `TimeoutError` branch records `last_status="timed_out"` via `mark_reminder_dispatched(success=False, timed_out=True)`
3. `mark_reminder_dispatched` gains the `timed_out` flag so the status names the timeout rather than generic "missed" (ran-but-cancelled vs ran-and-raised vs never-ran all stay distinct)
4. Config knob `[scheduling] handler_timeout_seconds` + reminder form/`update_reminder` acceptance of `timeout_seconds`
5. Slow-handler tests (real loop paths: timeout trips, loop liveness, next-occurrence computation, Run-now path shares the timeout) + docs
<!-- SECTION:PLAN:END -->
