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

## TASK-299.2 Addendum: Bidirectional Reminder Sync

The following decisions were made during TASK-299.2 and are recorded here so they survive the implementation plan.

### Idempotency keys are local-only

`ServerNotificationsService.create_reminder()` and `update_reminder()` do not yet accept an `idempotency_key` argument. Therefore `SchedulingServerClient` must strip any `idempotency_key` from the payload before forwarding to the service. The key remains in `pending_mutations.payload` for local deduplication and for future server-side idempotency support.

### `create_reminder` is not retried

Because idempotency keys are not forwarded to the server, retrying a transient timeout or 5xx on `create_reminder` could duplicate the server record. `SchedulingServerClient` retries transient failures for `list_reminders`, `update_reminder`, and `delete_reminder`, but **not** for `create_reminder`. A failed create leaves the pending mutation queued and stops the push phase.

### Network-then-transaction sync boundary

`SyncEngine.sync_now(owner_id)` performs all server calls in Phase 1 with **no** SQLite transaction held. Phase 2 opens a single `ScheduledTasksDB.transaction()` and atomically persists: pulled inserts/updates, conflict records, staged push outcomes (server_id → local_id mappings and pending-mutation deletions), tombstone cleanup, and `sync_state`. This prevents holding the SQLite connection across async HTTP I/O while still giving each sync attempt a single logical commit boundary.

### Crash-window trade-off

A crash after a successful network `create` but before the Phase 2 commit can leave the pending `create` mutation in the queue. Because the local row has no `server_id` yet, the next sync may create a duplicate server record. This is an accepted trade-off for TASK-299.2. A future upgrade can eliminate the window by adding per-push persistent staging or server-side idempotency.

### Owner identity interim fallback

Until the server exposes an account/principal endpoint (e.g., `/api/v1/me`), the server owner id is `"server:<active_server_id>"` where `active_server_id` is derived from the configured API URL by `runtime_policy.bootstrap.derive_configured_server_binding()`. This is an interim fallback that collides for multiple accounts on the same server. When an account endpoint becomes available, the owner id should become `"server:<user_id>"` and existing `sync_state`, `sync_mapping`, and `pending_mutations` rows must be migrated.

### Runtime-source mapping

The workbench owner switcher maps:
- `"local"` → `SchedulingService.set_owner("local")` and `set_authoritative_runtime_source(app, "local")`.
- `"server:<active_server_id>"` → `SchedulingService.set_owner("server:<active_server_id>")` and `set_authoritative_runtime_source(app, "server")`.

`SchedulingServerClient` is always instantiated, even when no server is configured at startup, so the app can inject or refresh the underlying notifications service after login via `server_client.set_notifications_service(...)` without rebuilding `SchedulingService`.

### Conflict resolution

Server-wins remains the default. Conflicts are surfaced in a dedicated workbench tab. "Use server" applies the server state (or deletes the local row for server-deletion). "Use local" re-queues the original pending mutation, preserving action, fields, and idempotency_key. For server-deletion conflicts, "Use local" clears the stale `server_id` and `sync_mapping` before re-queuing a `create` mutation.

## Links

- [Design spec](../../Docs/superpowers/specs/2026-07-18-scheduling-module-screen-design.md)
- [Implementation plan](../../Docs/superpowers/plans/2026-07-18-scheduling-module-screen-implementation-plan.md)
