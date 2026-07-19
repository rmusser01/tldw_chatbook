# TASK-299.2: Bidirectional reminder sync with tldw_server

## Status

Draft — in spec review.

## Related Task

[backlog/tasks/task-299.2 - Bidirectional-reminder-sync-with-tldw_server.md](../../../backlog/tasks/task-299.2%20-%20Bidirectional-reminder-sync-with-tldw_server.md)

## Goal

Wire the Scheduling service and `SyncEngine` to push local reminder changes to `tldw_server` and pull server-side reminders down, handling offline buffering and conflicts. This is AC #3 from TASK-299.

## Acceptance Criteria

- [ ] #1 Server-connected reminders are pulled into the local DB on sync.
- [ ] #2 Local create/update/delete mutations are pushed to `/api/v1/tasks` and queued when offline.
- [ ] #3 Conflicts are resolved server-wins and surfaced in the workbench.
- [ ] #4 Sync state (`last_pull_at`, `last_push_at`, `sync_errors`) is updated after each attempt.
- [ ] #5 Existing local-only reminders remain functional when sync fails.

## Architecture

The design keeps the existing layering but hardens the server boundary and adds workbench-visible sync health and conflict UI.

```
tldw_server /api/v1/tasks
       │
       ▼
SchedulingServerClient  ←── retries, timeouts, typed errors
       │
       ▼
SyncEngine ─────────────── pull / push / tombstones / conflicts
       │
       ▼
ScheduledTasksDB ───────── reminder_tasks, sync_mapping,
       │                   pending_mutations, sync_state, conflicts
       ▼
SchedulingService ──────── owner switching, sync_now()
       │
       ▼
SchedulesWorkbench ─────── Ctrl+S sync worker,
       │                   sync-status widget,
       │                   conflicts tab,
       │                   local/server mode switcher
       ▼
TaskDetail / TaskInspector
```

Key decisions:

- **Sync is workbench-triggered** (Ctrl+S), not background periodic. The eventual heartbeat + data-channel model is out of scope.
- **Owner/source switcher** is bound to `SchedulingService.set_owner()` and reflects the app’s existing runtime-source mechanism.
- **Server-wins conflict resolution** stays in `SyncEngine`; the UI exposes the choice to re-queue the local change.
- **Sync errors** are read from `sync_state.sync_errors` and shown in the status widget.

## Components

### 1. Hardened `SchedulingServerClient`

- Add a small `ServerClientConfig` dataclass with `timeout`, `max_retries`, `retry_delay`.
- Define a typed exception hierarchy:
  - `ServerClientError` — base class for all server-client failures.
  - `ServerUnavailableError` — no notifications service is configured (already exists).
  - `ServerClientTimeoutError` — request timed out.
  - `ServerClientNotFoundError` — server returned 404 (task deleted on server).
  - `ServerClientValidationError` — server returned 4xx other than 404, or a local policy denied the action.
  - `ServerClientServerError` — server returned 5xx.
- Wrap each server call in `_call_with_retry(method, *args, **kwargs)`:
  - Apply timeout: 10s for `list_reminders`, 30s for writes.
  - Strip local-only kwargs (e.g., `idempotency_key`) before calling `ServerNotificationsService`.
  - Map `PolicyDeniedError` to `ServerClientValidationError`; it is non-retryable.
  - Retry only transient errors (`ServerUnavailableError`, `ServerClientTimeoutError`, `ServerClientServerError`) up to 3 times with exponential backoff + jitter.
  - **Do not retry `create_reminder`.** Because idempotency keys are local-only and the server does not yet accept them, a transient timeout/5xx on create may have already succeeded on the server; retrying would duplicate the record. On any non-404 failure during create, leave the pending mutation queued and stop the push phase.
  - Map status codes to typed exceptions; raise `ServerClientError` with context for anything unexpected.
- **Idempotency keys are local-only.** `SchedulingServerClient.create_reminder`/`update_reminder` must **not** pass `idempotency_key` to `ServerNotificationsService`. The key stays in `pending_mutations.payload` for local deduplication and for future server support. Add a regression test that mocks `ServerNotificationsService` and asserts `idempotency_key` is never forwarded.
- Add `set_notifications_service()` so the app can inject or refresh the real service after login without rebuilding `SchedulingService`.

### 2. `SchedulesWorkbench` sync worker

- Add `Scheduling/events.py` messages:
  - `SyncCompleted(owner_id: str, conflict_count: int)`
  - `SyncFailed(owner_id: str, error: str)`
- Bind manual sync to `ctrl+s` on `SchedulesWorkbench` only (screen-scoped binding). The existing `SchedulesWorkbench` already declares `Binding("ctrl+s", "sync_now", "Sync")`; the spec refers to this binding.
- `action_sync_now()` starts an exclusive worker (`run_worker(self._run_sync, exclusive=True)`).
- If a sync worker is already running, notify the user “Sync already in progress” instead of silently dropping the keystroke.
- The worker captures the current `owner_id` and calls `SchedulingService.sync_now(owner_id)`.
  - Both `SchedulingService.sync_now(owner_id)` and `SyncEngine.sync_now(owner_id)` will accept an explicit `owner_id` and use it for the entire sync, ignoring `self.owner_id` for the duration of the call. This prevents races if the owner switcher changes mid-sync.
- While the worker runs, the owner switcher is disabled.
- On success, post `SyncCompleted(owner_id, conflict_count)`; on failure, post `SyncFailed(owner_id, error)`.
- After sync, refresh the task list and conflicts tab.

### 3. Sync status widget

- A thin bar above the tabs with stable widget IDs for tests and TCSS:
  - `#scheduling-sync-status` — the status bar container.
  - `#scheduling-owner-select` — the owner / runtime-source switcher.
  - `#scheduling-last-pull` — `Last pull:` timestamp.
  - `#scheduling-last-push` — `Last push:` timestamp.
  - `#scheduling-sync-error` — latest error message.
  - `#scheduling-clear-error` — manual “Clear” button.
- Shows:
  - Current mode: `Local` / `Server (<user>)`
  - `Last pull:` / `Last push:` timestamps from `sync_state`
  - Latest error (most recent entry from `sync_errors`) with a manual “Clear” button. The Clear button resets the entire capped history (`sync_errors=[]`).
- Refreshes on `SyncCompleted`, `SyncFailed`, and mode switch.
- Server option is disabled when `SchedulingService.server_client` is `None`.

### 4. Conflicts tab

- A `TabbedContent` at the top of the workbench with two tabs: **Queue** (existing three-pane layout) and **Conflicts**.
- The Conflicts tab contains a full-width `DataTable` (`#scheduling-conflicts-table`) with columns: `Title`, `Conflict Type`, `Server updated`, `Local updated`.
- **Conflict type** is derived at render time from `server_state`:
  - `server-deletion` when `not server_state` (covers `{}` and `None`).
  - `server-update` otherwise.
  - (Optional: add a `conflict_type` column in the schema migration if rendering cost becomes a concern.)
- Per-row actions:
  - **Use server** → calls `SyncEngine.resolve_conflict(id, "server")`, then synchronously refreshes the task list and conflicts table.
  - **Use local** → calls `SyncEngine.resolve_conflict(id, "local")`, then synchronously refreshes the task list and conflicts table.
- When recording a conflict, preserve the full original pending mutation in `local_state.pending_mutation` (including `action`, `fields`, and `idempotency_key`) so that "Use local" can re-queue the exact original intent.
- For server-deletion conflicts (`not server_state`):
  - **Use server** deletes the local row, removes its `sync_mapping`, and removes any related tombstone.
  - **Use local** clears `server_id` from the local row, removes the stale `sync_mapping`, and re-queues the original pending mutation (or a `create` mutation carrying the local record fields if there was no pending mutation). The cleared `server_id` prevents the re-queued create from being misrouted as an update.

### 5. Owner / runtime-source switcher

- A `Select` or button pair in the sync status bar showing `Local` and the available server account.
- Define the owner/runtime-source mapping explicitly:
  - Switcher value `"local"` → `SchedulingService.set_owner("local")` and `set_authoritative_runtime_source(app, "local")`.
  - Switcher value `"server:<active_server_id>"` → `SchedulingService.set_owner("server:<active_server_id>")` and `set_authoritative_runtime_source(app, "server")`.
  - The switcher reads `app.runtime_policy.state.active_source` and `app.runtime_policy.state.active_server_id` to determine the initial selection and whether the server option is enabled.
- **Owner identity interim fallback:** `<active_server_id>` is currently derived from the configured API URL (`runtime_policy.bootstrap.derive_configured_server_binding`). This is an interim owner identity. Once the server exposes an account/principal endpoint (e.g., `/api/v1/me`), the owner should become `"server:<user_id>"` and existing `sync_state`, `sync_mapping`, and `pending_mutations` rows must be migrated. Document this fallback and migration note in ADR-018.
- Server option is shown only when `active_server_id` is available; it is disabled when `SchedulingService.server_client` is `None`.
- Refreshes the task list, sync status, and conflicts tab.
- If switching to server mode with no `server_client`, notify “No server connection”.

## Data Flow

### Sync trigger flow

```
User presses Ctrl+S
  └─► SchedulesWorkbench.action_sync_now()
       ├─► If sync worker running → notify "Sync already in progress"
       └─► run_worker(self._run_sync, exclusive=True)
            ├─► Capture current owner_id
            ├─► Disable owner switcher
            ├─► await scheduling_service.sync_now(owner_id)
            ├─► Post SyncCompleted(owner_id, conflict_count)
            ├─► Re-enable owner switcher
            └─► Refresh task list + conflicts tab
```

On failure, post `SyncFailed(owner_id, error)` and surface the error.

### Owner switch flow

```
User changes mode switcher
  └─► SchedulesWorkbench._set_owner(new_owner)
       ├─► await scheduling_service.set_owner(new_owner)
       ├─► Persist choice in app runtime-source state
       ├─► Refresh task list, sync status, conflicts tab
       └─► If new_owner is server: and no server_client, notify "No server connection"
```

### Conflict resolution flow

```
User opens Conflicts tab
  └─► DataTable loads from db.get_conflicts(owner_id, primitive="reminder_task")

User clicks "Use server"
  └─► SyncEngine.resolve_conflict(conflict_id, "server")
       └─► Refresh conflicts tab + task list

User clicks "Use local"
  └─► SyncEngine.resolve_conflict(conflict_id, "local")
       └─► Re-queues local mutation (or create for server-deletion)
       └─► Refresh conflicts tab + task list
```

### Server-client hardening flow

```
SchedulingServerClient.create/update/delete/list_reminders()
  └─► _call_with_retry(method, *args, **kwargs)
       ├─► Strip local-only kwargs (e.g., idempotency_key) before calling service
       ├─► Apply timeout (10s read / 30s write)
       ├─► Map PolicyDeniedError → ServerClientValidationError (non-retryable)
       ├─► Retry on transient errors (max 3, exponential backoff + jitter)
       │    └── create_reminder is NOT retried (no server idempotency)
       ├─► Map 404 → ServerClientNotFoundError
       ├─► Map other 4xx → ServerClientValidationError
       ├─► Map 5xx → ServerClientServerError
       ├─► Map timeouts → ServerClientTimeoutError
       └─► Return server response dict
```

### Sync engine transaction boundary

```
SyncEngine.sync_now(owner_id)
  ├─► Phase 1 — network only (no DB transaction)
  │    ├─► list_reminders() → server_items
  │    ├─► push mutations one by one
  │    │    ├── create: no retry; on failure stop phase
  │    │    ├── update/delete: retry transient; 404 → conflict
  │    └─► push tombstones one by one (retry transient; 404 → clean local)
  │
  └─► Phase 2 — single ScheduledTasksDB.transaction()
       ├─► Apply all pulled inserts/updates
       ├─► Apply all conflict records
       ├─► Delete pushed mutations
       ├─► Delete pushed tombstones
       ├─► Update sync_state (last_pull_at, last_push_at, sync_errors)
       └─► Rollback on any unhandled exception
```

## Error Handling

- **Transient errors** (`ServerUnavailableError`, `ServerClientTimeoutError`, `ServerClientServerError`): retry up to 3 times with exponential backoff + jitter inside `SchedulingServerClient`, then surface as a single failure to `SyncEngine`. The engine records one error and stops the current sync phase. Pending mutations remain queued.
- **`ServerClientValidationError` (400 / other 4xx)**: keep mutation, record error, stop sync phase.
- **`ServerClientNotFoundError` (404) on update/delete**: record a conflict/tombstone so the user can decide, rather than retrying forever. On update, treat as server-deletion conflict. On delete, clean up the local tombstone and mapping because the server record is already gone.
- **Sync error history**: `SyncEngine._record_sync_error` reads existing `sync_errors`, appends the new `{message, timestamp}`, and truncates to the last 10 before calling `db.update_sync_state`. Manual “Clear” button in the sync status widget calls `db.update_sync_state(owner_id, sync_errors=[])`.
- **Owner switch during sync**: owner switcher disabled while sync worker runs. `SchedulingService.sync_now(owner_id)` and `SyncEngine.sync_now(owner_id)` accept an explicit owner and use it for the entire call to avoid mid-flight races.
- **Transaction / network boundary**: `SyncEngine.sync_now(owner_id)` must perform all network calls *outside* the SQLite transaction, collect their results and any server states, then open **one** `ScheduledTasksDB.transaction()` to atomically persist: pulled inserts/updates, resolved conflicts, pushed-mutation deletions, and tombstone cleanup. This avoids holding the SQLite connection (and blocking other DB operations) across async HTTP I/O. On any unhandled exception after the network phase begins, the single transaction is rolled back so the owner is never left in a partially synced state.

## Testing Plan

### Server client unit tests

- Retry with jitter on transient errors; no retry on `ServerClientValidationError` or `PolicyDeniedError`.
- `create_reminder` is not retried on transient failure (no server idempotency).
- `set_notifications_service()` propagation.
- 404 on update/delete raises `ServerClientNotFoundError`.
- `PolicyDeniedError` maps to `ServerClientValidationError` and is non-retryable.
- **Idempotency-key regression**: mock `ServerNotificationsService` (same signature as the real service) and assert `create_reminder`/`update_reminder` are called without `idempotency_key`.
- Typed exception mapping for 4xx/5xx/timeout.

### SyncEngine unit tests

- Server-deletion conflict + “Use local” re-queues the original pending mutation (or a `create`) and clears stale mapping/server_id.
- “Use server” on server-deletion deletes local row, mapping, and tombstone.
- `sync_now(owner_id)` uses the passed owner, not `self.owner_id`.
- `_record_sync_error` appends errors and caps at 10 entries.
- 404 on push records a conflict and removes the pending mutation.
- Local-only reminders remain functional when sync fails.
- All network calls complete before a single DB transaction is opened; no SQLite connection is held across HTTP I/O.

### Workbench UI tests

- `ctrl+s` triggers sync and refreshes the task list.
- Second `ctrl+s` during sync shows “Sync already in progress”.
- Owner switcher updates `owner_id` and refreshes.
- Owner switcher is disabled while a sync worker is running.
- Server option disabled when `server_client` is `None`.
- Conflicts tab renders rows and resolution actions.
- Conflicts tab refreshes on `TabActivated`.
- Sync status widget shows last pull/push and latest error.
- Policy-denial errors are surfaced as non-retryable sync errors.

## Out of Scope

- Background periodic sync; heartbeat + data-channel updates are planned for a future task.
- Pagination of `list_reminders()` responses.
- Automatic conflict resolution beyond server-wins.
- Watchlist projections remain read-only in the workbench; sync and conflicts operate only on `primitive="reminder_task"`.

## Risks

- **Owner identity interim fallback**: using URL-derived `active_server_id` as the server owner collides multi-account usage on the same server. ADR-018 must document this fallback and the migration path to `"server:<user_id>"` once the server exposes an account endpoint.
- **Create duplication without server idempotency**: by design, `create_reminder` is not retried. Users may see pending creates remain queued after transient timeouts until the next manual sync. ADR-018 should record this trade-off.
- **ADR-018 missing**: `backlog/decisions/018-local-server-hybrid-scheduled-tasks.md` should be created as part of this work. It should record: server-wins default, local-only idempotency keys, the owner-id/runtime-source mapping, the network-then-transaction boundary, and the create-no-retry decision.
