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
  - `ServerClientValidationError` — server returned 4xx other than 404.
  - `ServerClientServerError` — server returned 5xx.
- Wrap each server call in `_call_with_retry(method, *args, **kwargs)`:
  - Apply timeout: 10s for `list_reminders`, 30s for writes.
  - Retry only transient errors (`ServerUnavailableError`, `ServerClientTimeoutError`, `ServerClientServerError`) up to 3 times with exponential backoff + jitter.
  - Map status codes to typed exceptions; raise `ServerClientError` with context for anything unexpected.
- **Idempotency keys are local-only.** `SchedulingServerClient.create_reminder`/`update_reminder` must **not** pass `idempotency_key` to `ServerNotificationsService`. The key stays in `pending_mutations.payload` for local deduplication and for future server support. Add a regression test that mocks `ServerNotificationsService` and asserts `idempotency_key` is never forwarded.
- Add `set_notifications_service()` so the app can inject or refresh the real service after login without rebuilding `SchedulingService`.

### 2. `SchedulesWorkbench` sync worker

- Add `Scheduling/events.py` messages:
  - `SyncCompleted(owner_id: str, conflict_count: int)`
  - `SyncFailed(owner_id: str, error: str)`
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
  - Latest error (if any) with a manual “Clear” button
- Refreshes on `SyncCompleted`, `SyncFailed`, and mode switch.
- Server option is disabled when `SchedulingService.server_client` is `None`.

### 4. Conflicts tab

- A `TabbedContent` at the top of the workbench with two tabs: **Queue** (existing three-pane layout) and **Conflicts**.
- The Conflicts tab contains a full-width `DataTable` (`#scheduling-conflicts-table`) with columns: `Title`, `Conflict Type`, `Server updated`, `Local updated`.
- **Conflict type** is derived at render time from `server_state`:
  - `server-deletion` when `server_state == {}`.
  - `server-update` otherwise.
  - (Optional: add a `conflict_type` column in the schema migration if rendering cost becomes a concern.)
- Per-row actions:
  - **Use server** → calls `SyncEngine.resolve_conflict(id, "server")` and refreshes.
  - **Use local** → calls `SyncEngine.resolve_conflict(id, "local")` and refreshes.
- When recording a conflict, preserve the full original pending mutation in `local_state.pending_mutation` (including `action`, `fields`, and `idempotency_key`) so that "Use local" can re-queue the exact original intent.
- For server-deletion conflicts (`server_state={}`):
  - **Use server** deletes the local row, removes its `sync_mapping`, and removes any related tombstone.
  - **Use local** clears the stale `server_id`/mapping and re-queues the original pending mutation (or a `create` mutation carrying the local record fields if there was no pending mutation).

### 5. Owner / runtime-source switcher

- A `Select` or button pair in the sync status bar showing `Local` and the available server account.
- Define the owner/runtime-source mapping explicitly:
  - Switcher value `"local"` → `SchedulingService.set_owner("local")` and `set_authoritative_runtime_source(app, "local")`.
  - Switcher value `"server:<active_server_id>"` → `SchedulingService.set_owner("server:<active_server_id>")` and `set_authoritative_runtime_source(app, "server")`.
  - The switcher reads `app.runtime_policy.state.active_source` and `app.runtime_policy.state.active_server_id` to determine the initial selection and whether the server option is enabled.
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
       ├─► Retry on transient errors (max 3, exponential backoff + jitter)
       ├─► Map 404 → ServerClientNotFoundError
       ├─► Map other 4xx → ServerClientValidationError
       ├─► Map 5xx → ServerClientServerError
       ├─► Map timeouts → ServerClientTimeoutError
       └─► Return server response dict
```

## Error Handling

- **Transient errors** (`ServerUnavailableError`, `ServerClientTimeoutError`, `ServerClientServerError`): retry up to 3 times with exponential backoff + jitter inside `SchedulingServerClient`, then surface as a single failure to `SyncEngine`. The engine records one error and stops the current sync phase. Pending mutations remain queued.
- **`ServerClientValidationError` (400 / other 4xx)**: keep mutation, record error, stop sync phase.
- **`ServerClientNotFoundError` (404) on update/delete**: record a conflict/tombstone so the user can decide, rather than retrying forever. On update, treat as server-deletion conflict. On delete, clean up the local tombstone and mapping because the server record is already gone.
- **Sync error history**: `SyncEngine._record_sync_error` reads existing `sync_errors`, appends the new `{message, timestamp}`, and truncates to the last 10 before calling `db.update_sync_state`. Manual “Clear” button in the sync status widget calls `db.update_sync_state(owner_id, sync_errors=[])`.
- **Owner switch during sync**: owner switcher disabled while sync worker runs. `SchedulingService.sync_now(owner_id)` and `SyncEngine.sync_now(owner_id)` accept an explicit owner and use it for the entire call to avoid mid-flight races.
- **Transaction boundary**: `SyncEngine.sync_now(owner_id)` should wrap `pull`, `_push`, and `_push_tombstones` in a single `ScheduledTasksDB.transaction()` block, or document a compensating strategy if phases must remain separate (e.g., because each phase reads server state). At minimum, ensure that successful `_push` mutations are not committed until `_push_tombstones` is also committed, so a crash cannot resurrect deleted server rows.

## Testing Plan

### Server client unit tests

- Retry with jitter on transient errors; no retry on `ServerClientValidationError`.
- `set_notifications_service()` propagation.
- 404 on update/delete raises `ServerClientNotFoundError`.
- **Idempotency-key regression**: mock `ServerNotificationsService` (same signature as the real service) and assert `create_reminder`/`update_reminder` are called without `idempotency_key`.
- Typed exception mapping for 4xx/5xx/timeout.

### SyncEngine unit tests

- Server-deletion conflict + “Use local” re-queues the original pending mutation (or a `create`) and clears stale mapping.
- “Use server” on server-deletion deletes local row, mapping, and tombstone.
- `sync_now(owner_id)` uses the passed owner, not `self.owner_id`.
- `_record_sync_error` appends errors and caps at 10 entries.
- 404 on push records a conflict and removes the pending mutation.
- Local-only reminders remain functional when sync fails.
- Pull + push + tombstones are committed atomically (or documented compensating behavior is tested).

### Workbench UI tests

- Ctrl+S triggers sync and refreshes the task list.
- Second Ctrl+S during sync shows “Sync already in progress”.
- Owner switcher updates `owner_id` and refreshes.
- Server option disabled when `server_client` is `None`.
- Conflicts tab renders rows and resolution actions.
- Conflicts tab refreshes on `TabActivated`.
- Sync status widget shows last pull/push and latest error.

## Out of Scope

- Background periodic sync; heartbeat + data-channel updates are planned for a future task.
- Pagination of `list_reminders()` responses.
- Automatic conflict resolution beyond server-wins.
- Watchlist projections remain read-only in the workbench; sync and conflicts operate only on `primitive="reminder_task"`.

## Risks

- **Transaction boundary across pull/push/tombstones**: if the three phases cannot share one transaction (e.g., because `list_reminders` must read server state), the spec must be updated with a compensating strategy before implementation.
- **ADR-018 missing**: `backlog/decisions/018-local-server-hybrid-scheduled-tasks.md` should be created as part of this work. It should record: server-wins default, local-only idempotency keys, and the owner-id/runtime-source mapping.
