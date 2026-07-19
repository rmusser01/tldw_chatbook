# ADR-019: Migrate watchlist checks to the unified Scheduling scheduler

Status: Accepted
Date: 2026-07-19
Related Task: TASK-299
Supersedes: N/A

## Decision

Migrate watchlist/subscription check execution from `Subscriptions/scheduler.py` (`SubscriptionScheduler`) to the unified `Scheduling` module, running as a `watchlist` handler under `SchedulerLoop`.

- A new `WatchlistCheckHandler` executes watchlist checks by delegating to the existing `FeedMonitor`, `URLMonitor`, and `SubscriptionsDB` infrastructure.
- `WatchlistProjection` continues to provide a read-only view of `Subscriptions_DB` rows into `ScheduledTask` objects for the workbench UI.
- The migration proceeds behind a feature flag (`scheduling.watchlist_checks_enabled`) with a **dual-run** validation phase:
  - Old scheduler remains the execution authority by default.
  - New handler runs side-by-side in "shadow" mode, executing the same checks but not mutating subscription state.
  - Metrics and logs from both paths are compared before promoting the new handler to authoritative mode.
- Once validated, the new handler becomes authoritative and the old `SubscriptionScheduler` is deprecated.
- Removal of the old scheduler is deferred to a follow-up release after dual-run validation has completed and parity metrics meet the promotion threshold.
- A runtime toggle allows instant rollback to the old scheduler without a code deploy.

## Context

`tldw_chatbook` currently has two scheduling systems:

1. **`Subscriptions/scheduler.py`** — A dedicated scheduler for RSS/URL watchlist checks. It owns its own priority queue, concurrency limit, rate limiting, feed monitoring, and subscription state updates. It is tightly coupled to `Subscriptions_DB` and does not participate in the new server-side scheduled-tasks control plane.
2. **`Scheduling/scheduler/loop.py`** — A generic scheduler introduced in ADR-018 for reminders and (eventually) automation definitions. It polls `ScheduledTasksDB` and dispatches due tasks to typed handlers.

ADR-018 already decided that watchlist jobs would remain read-only projections from `Subscriptions_DB` until Phase 5. Phase 5 is now beginning. The workbench UI already displays watchlist jobs via `WatchlistProjection`; the missing piece is moving their *execution* into the unified scheduler so that:

- A single loop owns all scheduled work.
- Server-side and local scheduling concepts align.
- Future server-side watchlist control-plane integration can plug into one surface.

## Alternatives Considered

| Option | Why rejected |
| --- | --- |
| Keep `SubscriptionScheduler` indefinitely | Perpetuates two scheduling systems, duplicate queue logic, and divergent failure/recovery semantics. |
| Big-bang cutover to `WatchlistCheckHandler` | Too risky for user-facing background checks; a regression in check reliability would silently break watchlist updates. |
| Run both schedulers concurrently with mutation | Risk of double-checking and race conditions on `Subscriptions_DB` state. |
| Shadow-mode dual-run (chosen) | Validates correctness and performance against the proven old scheduler before any authoritative switch, with a runtime rollback lever. |

## Consequences

- `SchedulerLoop` will load watchlist jobs from `WatchlistProjection` (or a new `ScheduledTasksDB` mirror populated by sync) and dispatch them to `WatchlistCheckHandler`.
- `WatchlistCheckHandler` must be stateless with respect to the new scheduler; all persistent state remains in `Subscriptions_DB`.
- The feature flag lives in config (`[scheduling] watchlist_checks_enabled = false`) and is read at scheduler startup. Changing it requires an app restart.
- Dual-run metrics must include: checks executed, successes, failures, latency, and result parity (old vs new) per subscription.
- Old scheduler code is retained during validation but marked deprecated. Removal is gated on dual-run parity metrics and a minimum bake time.
- The old scheduler is now deprecated in code (`Subscriptions/scheduler.py`, `Subscriptions/textual_scheduler_worker.py`) with `DeprecationWarning`s and docstring notices, but it remains functional for the dual-run validation period.
- Console-follow and screenshot QA behavior for watchlists must continue to work unchanged; only the execution backend moves.

## Rollback plan

1. Set `[scheduling] watchlist_checks_enabled = false` and restart the app.
2. The old `SubscriptionScheduler` resumes authoritative execution.
3. New-handler shadow metrics are still collected if `watchlist_checks_shadow = true`, enabling safe debugging.

## Links

- ADR-018: Local/server hybrid scheduled-tasks storage and sync
- [Implementation plan](../../Docs/superpowers/plans/2026-07-18-scheduling-module-screen-implementation-plan.md)
- [Design spec](../../Docs/superpowers/specs/2026-07-18-scheduling-module-screen-design.md)
