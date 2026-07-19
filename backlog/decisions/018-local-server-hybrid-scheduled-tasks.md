# ADR-018: Local/server hybrid scheduled-tasks storage and sync

Status: Accepted
Date: 2026-07-18
Related Task: TASK-299
Supersedes: N/A

## Decision

Build tldw_chatbook's Scheduling module as a local-first hybrid:

- A dedicated local SQLite database, `ScheduledTasksDB`, owns reminders, automation definitions, and (eventually) watchlist schedule metadata.
- Local records are keyed by `owner_id`: `"local"` for device-local data and `"server:<user_id>"` for data synced from a server account.
- Reminders execute locally via a new `SchedulerLoop` worker regardless of server reachability.
- Automation definitions are created, previewed, paused, resumed, and archived locally in this phase; execution remains `execution_unavailable` until server-side automation execution is integrated.
- Watchlist jobs are read-only projections from `Subscriptions_DB` until a later Phase 5 migration.
- Server-wins conflict resolution is the default: when a local pending mutation conflicts with a newer server state, the server state is kept and the local mutation is moved to `sync_conflicts` for explicit user resolution.
- The existing `Subscriptions/scheduler.py` watchlist scheduler remains active behind a feature flag until Phase 5 validates the new handler.

## Context

tldw_chatbook already has server-side reminder endpoints (`/api/v1/tasks`) exposed through `ServerNotificationsService`, and tldw_server has a unified scheduled-tasks control plane that the WebUI consumes. Users expect reminders to fire even when offline, and they need a single workbench to inspect reminders, watchlist jobs, and automation definitions. Re-implementing all scheduling server-side would break offline operation; mirroring server state locally while keeping local mutation capability satisfies both online and offline use cases.

## Alternatives Considered

| Option | Why rejected |
| --- | --- |
| Server-only scheduling | Breaks offline reminders and increases latency for local triggers. |
| Local-only scheduling | Cannot share state with tldw_server accounts or the WebUI. |
| Reuse `Subscriptions_DB` schema for all scheduled tasks | Watchlist schema is specialized for RSS/URL monitoring; forcing reminders and automation definitions into it would create a confusing, wide table. |
| Client-wins conflict resolution | Would silently overwrite server changes made from the WebUI or another device. |
| Immediate migration of watchlist execution to the new scheduler | Too risky without side-by-side comparison; deferred to Phase 5 behind a feature flag. |

## Consequences

- All scheduling screens and services must respect `owner_id` filtering and auth-state transitions.
- `ScheduledTasksDB` introduces a new schema version track that must be migrated forward like other project DBs.
- Sync logic must handle tombstones, idempotency keys, and retry/backoff explicitly.
- Console-follow and screenshot QA behavior from the existing `SchedulesScreen` must be preserved.
- Phase 5 will require its own ADR for watchlist scheduler migration, including rollback plan and dual-run validation.

## Links

- [Design spec](../../Docs/superpowers/specs/2026-07-18-scheduling-module-screen-design.md)
- [Implementation plan](../../Docs/superpowers/plans/2026-07-18-scheduling-module-screen-implementation-plan.md)
