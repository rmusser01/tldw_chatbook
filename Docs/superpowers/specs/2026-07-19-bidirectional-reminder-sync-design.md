# TASK-299.2: Bidirectional reminder sync with tldw_server

## Status

Design approved — ready for implementation planning.

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
- Wrap each server call in a retry loop that retries only transient errors (`ServerUnavailableError`, `TimeoutError`, 5xx) with exponential backoff + jitter.
- Use different timeouts for reads (`list_reminders`, 10s) and writes (`create/update/delete`, 30s).
- Convert unexpected exceptions into `ServerClientError` with context.
- Add `set_notifications_service()` so the app can inject or refresh the real service after login without rebuilding `SchedulingService`.
- Verify the real `notifications_service` method signatures before implementing idempotency-key handling. If the service does not accept `idempotency_key`, strip/rename it; otherwise pass it through.

### 2. `SchedulesWorkbench` sync worker

- `action_sync_now()` starts an exclusive worker (`run_worker(self._run_sync, exclusive=True)`).
- If a sync worker is already running, notify the user “Sync already in progress” instead of silently dropping the keystroke.
- The worker captures the current `owner_id` and calls `SchedulingService.sync_now(owner_id)`.
- While the worker runs, the owner switcher is disabled.
- On success, post `SyncCompleted(owner_id, conflict_count)`; on failure, post `SyncFailed(owner_id, error)`.
- After sync, refresh the task list and conflicts tab.

### 3. Sync status widget

- A thin bar above the tabs showing:
  - Current mode: `Local` / `Server (<user>)`
  - `Last pull:` / `Last push:` timestamps from `sync_state`
  - Latest error (if any) with a manual “Clear” button
- Refreshes on `SyncCompleted`, `SyncFailed`, and mode switch.
- Server option is disabled when `SchedulingService.server_client` is `None`.

### 4. Conflicts tab

- A `TabbedContent` at the top of the workbench with two tabs: **Queue** (existing three-pane layout) and **Conflicts**.
- The Conflicts tab contains a full-width `DataTable` with columns: `Title`, `Conflict Type`, `Server updated`, `Local updated`.
- Per-row actions:
  - **Use server** → calls `SyncEngine.resolve_conflict(id, "server")` and refreshes.
  - **Use local** → calls `SyncEngine.resolve_conflict(id, "local")` and refreshes.
- For server-deletion conflicts (`server_state={}`):
  - **Use server** deletes the local row.
  - **Use local** clears the stale `server_id`/mapping and re-queues a `create` mutation.
- Refreshes on `SyncCompleted`, `SyncFailed`, and `TabbedContent.TabActivated`.

### 5. Owner / runtime-source switcher

- A `Select` or button pair in the sync status bar showing `Local` and the available server account.
- On change, calls `SchedulingService.set_owner(new_owner)` and writes back to the app’s runtime-source state.
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
  └─► _call_with_retry(method, **kwargs)
       ├─► Apply timeout (10s read / 30s write)
       ├─► Retry on transient errors (max 3, exponential backoff + jitter)
       ├─► Map 4xx/Unexpected to ServerClientError
       └─► Return server response dict
```

## Error Handling

- **Transient errors** (`ServerUnavailableError`, `TimeoutError`, 5xx): retry up to 3 times with exponential backoff + jitter, then record error and stop the current sync phase. Pending mutations remain queued.
- **`400 Bad Request`**: keep mutation, record error, stop sync phase.
- **`404 Not Found` on update/delete**: record a conflict/tombstone so the user can decide, rather than retrying forever.
- **Other 4xx**: keep mutation, record error, stop sync phase.
- **Sync error history**: append errors with timestamps, cap at last 10. Manual “Clear” button in the sync status widget.
- **Owner switch during sync**: owner switcher disabled while sync worker runs. `SchedulingService.sync_now(owner_id)` accepts an explicit owner to avoid mid-flight races.

## Testing Plan

### Server client unit tests

- Retry with jitter on transient errors; no retry on 4xx.
- `set_notifications_service()` propagation.
- 404 on update/delete maps to conflict/error handling.
- Idempotency key passthrough (after verifying real service signatures).

### SyncEngine unit tests

- Server-deletion conflict + “Use local” re-queues `create` and clears stale mapping.
- `sync_now(owner_id)` uses the passed owner, not `self.owner_id`.
- Local-only reminders remain functional when sync fails.

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

## Risks

- **Real server-client signatures**: idempotency-key handling depends on the actual `notifications_service` methods. Must verify before implementing.
- **Owner/user ID source**: the exact app attribute for runtime source / user ID must be confirmed.
- **ADR-018 missing**: `backlog/decisions/018-local-server-hybrid-scheduled-tasks.md` should be created as part of this work.
