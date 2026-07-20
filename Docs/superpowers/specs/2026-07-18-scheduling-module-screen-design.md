# Scheduling Module + Screen Design

Date: 2026-07-18
Status: draft for review
Source branch: `origin/dev`

## Prerequisites

Before implementation begins:

1. Create a Backlog.md task to track this work under `backlog/tasks/`.
2. Create ADR `backlog/decisions/NNN-local-server-hybrid-scheduled-tasks.md` covering:
   - The local/server hybrid storage boundary.
   - The server-wins conflict policy.
   - The per-server owner identity model.
   - The migration path from the existing `Subscriptions/scheduler.py`.
3. Complete Phase 0 (API contract spike) and accept its deliverables before starting Phase 1.

## Purpose

Port tldw_server's unified scheduled-tasks control plane and automation workbench into tldw_chatbook as a first-class local module and TUI screen. The result is a hybrid local/server scheduling system that:

- Runs reminders locally even when offline.
- Manages automation definitions (`recurring_question`, `agent_task`) with full lifecycle, previews, and audit.
- Mirrors watchlist jobs and eventually migrates their periodic execution into the new scheduler.
- Syncs with tldw_server when authenticated, using server-wins conflict resolution with explicit conflict surfacing.
- Replaces the existing placeholder `SchedulesScreen` with a real workbench for creating, inspecting, and controlling scheduled work.

## Success Criteria

- Existing `schedules_screen.py` mounted regressions and screenshot QA evidence still pass after replacement.
- Reminders can be created, edited, deleted, and triggered locally without a server connection.
- Server-connected reminders sync bidirectionally with tldw_server `/api/v1/tasks`.
- Automation definitions can be created, previewed, paused, resumed, and archived locally.
- Watchlist jobs are visible in the workbench as read-only projections until Phase 5.

## Decisions

The following decisions were made during brainstorming:

1. **Scope:** Port the full tldw_server unified scheduled-tasks module — reminders, watchlist jobs, and automation definitions — into tldw_chatbook.
2. **Runtime boundary:** Hybrid local-first with server sync. Use local SQLite + local scheduler by default; call tldw_server API when reachable and authenticated.
3. **Execution scope:** Only reminders execute locally in this phase. Automation definitions are created/previewed/paused/archived but kept at `execution_unavailable` health. Watchlist job execution migrates in Phase 5.
4. **Storage:** New dedicated `ScheduledTasksDB` module in `Scheduling/db/scheduled_tasks_db.py`, mirroring tldw_server schema, plus sync helper tables.
5. **Identity:** Per-server owner model. `owner_id` is `"local"` for device-local records, `"server:<user_id>"` for records synced from a server account.
6. **Approach:** Layered incremental port in five phases, each split into reviewable sub-tasks/PRs.
7. **Scheduler boundary:** Eventually migrate watchlist checks from the existing `Subscriptions/scheduler.py` into the new unified task-queue scheduler.

## Naming Conventions

- Python package: `Scheduling/` (PascalCase directory, matching `Subscriptions/`).
- DB class: `ScheduledTasksDB` (inherits from `DB/base_db.py:BaseDB`).
- DB module: `Scheduling/db/scheduled_tasks_db.py`.
- Screen directory: `UI/Screens/scheduling/`.
- TCSS file: `tldw_chatbook/css/features/_scheduling.tcss`.

## Non-Goals

- Do not replace Watchlists as the deep workspace for source curation, scraping, and output reporting.
- Do not implement full execution for automation definitions in this design.
- Do not build a generic visual workflow builder.
- Do not require tldw_server to be online for local reminder operation.
- Do not duplicate `ServerNotificationsService` reminder APIs unless consolidating.

## Current Context

- tldw_chatbook already has `UI/Screens/schedules_screen.py` with a 3-pane destination shell and Console-follow recovery behavior, but its run-control buttons are disabled ("not wired yet").
- tldw_chatbook already has `Subscriptions/scheduler.py` and `textual_scheduler_worker.py` for RSS/URL subscription checks; the worker class is currently unwired in `app.py`.
- tldw_chatbook already has `tldw_api/notifications_reminders_schemas.py` for reminder payloads.
- `tldw_api/client.py:TLDWAPIClient` already exposes reminder-task endpoints at `/api/v1/tasks`.
- `Notifications/server_notifications_service.py` already wraps those endpoints with policy gating.
- `pyproject.toml` already lists `schedule` as an optional dependency under `[subscriptions]`.
- tldw_server has `app/services/scheduled_tasks_control_plane_service.py` (unified read model) and `app/services/scheduled_task_automation_service.py` (automation definition lifecycle).
- tldw_server's WebUI has a `/scheduled-tasks` Automation Workbench PRD covering templates, runs, results, and bulk actions.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Textual TUI                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐  │
│  │ Schedules    │  │ Create/Edit  │  │ Run Detail/      │  │
│  │ Workbench    │  │ Forms        │  │ Inspector        │  │
│  └──────┬───────┘  └──────┬───────┘  └────────┬─────────┘  │
│         │                 │                    │            │
│         └─────────────────┴────────────────────┘            │
│                           │                                  │
│              ┌────────────▼────────────┐                    │
│              │   SchedulingService     │                    │
│              │  (local-first facade)   │                    │
│              └───────┬────────┬────────┘                    │
│                      │        │                             │
│         ┌────────────▼─┐   ┌─▼─────────────┐               │
│         │  Local DB    │   │ Server Client │◄──── tldw_server
│         │ScheduledTasks│   │  (sync/recon) │     /api/v1/
│         └───────┬──────┘   └───────────────┘               │
│                 │                                           │
│    ┌────────────▼────────────┐                              │
│    │   Unified Scheduler     │                              │
│    │ (task queue + workers)  │                              │
│    └───────┬─────────────────┘                              │
│            │                                                │
│    ┌───────▼────────┐    ┌────────────────┐                │
│    │ Reminder runner│    │ Watchlist check│                │
│    │                │    │ handler (Phase │                │
│    └────────────────┘    │      5)        │                │
│                          └────────────────┘                │
└─────────────────────────────────────────────────────────────┘
```

### App integration

The scheduler is owned by the main `TldwCli` app lifecycle. Unlike the currently unwired `SubscriptionSchedulerWorker`, the new scheduler must be explicitly started and stopped:

- `TldwCli.__init__` creates `self.scheduling_service = SchedulingService(...)` and `self.scheduler_loop = SchedulerLoop(...)`.
- `TldwCli.on_mount()` starts the scheduler worker after DB services are initialized. This covers both splash and no-splash startup paths.
  ```python
  self.scheduler_worker = self.run_worker(
      self.scheduler_loop.run(),
      exclusive=True,
      group="scheduling",
  )
  ```
- `TldwCli.on_unmount()` cancels the worker and awaits scheduler shutdown.
- The service is injected into `UI/Screens/scheduling/schedules_workbench.py` via `self.app.scheduling_service`, matching how other screens access shared services.

### Scheduler threading model

- `SchedulerLoop.run()` is an `async def` coroutine running on a Textual worker (`@work(exclusive=True)` without `thread=True`).
- It uses `asyncio.sleep()` for polling.
- DB access uses `asyncio.to_thread()` because `BaseDB` is synchronous SQLite.
- Handlers post UI events via `self.post_message(...)` from the worker; Textual routes them to the screen.

### Runtime modes

| Mode | Behavior |
|------|----------|
| Local only | All CRUD + reminder execution uses local `ScheduledTasksDB`. `owner_id = "local"`. |
| Server connected | Reads/writes prefer server API; local DB is a cache. Background sync keeps cache fresh. `owner_id = "server:<user_id>"`. |
| Sync reconnect | Pull server state, reconcile local pending mutations and tombstones, push local-only changes if safe. |

### Auth state transitions

| Transition | Behavior |
|------------|----------|
| Login / server selected | Switch to server mode for the authenticated `owner_id`; start background sync; filter visible records to that owner. Pending mutations for the new owner are pushed. |
| Logout / server unreachable | Fall back to local mode (`owner_id = "local"`); stop server sync; scheduler continues with local records. Server-owned records are hidden but retained; they reappear when that owner logs in again. |
| Server switch | Queue pending mutations for the previous owner; switch to the new owner; start sync for the new owner. |

### Owner semantics

- `owner_id = "local"` for device-local reminders and definitions, always visible in local mode.
- `owner_id = "server:<user_id>"` for records synced from a server account, visible only when that account is authenticated.
- Creating a record while logged in defaults to the current owner. The user may choose to keep a reminder local even while logged in.

## Data Model

New SQLite module `Scheduling/db/scheduled_tasks_db.py`. `ScheduledTasksDB` inherits from `DB/base_db.py:BaseDB` and implements `_initialize_schema()`, matching `SubscriptionsDB` and other project DBs.

The database path is obtained from `config.py` following the existing helper pattern:

```python
def get_scheduled_tasks_db_path() -> Path:
    custom_path = get_cli_setting("database", "scheduled_tasks_db_path", None)
    if custom_path and custom_path != DEFAULT_CONFIG_FROM_TOML.get("database", {}).get("scheduled_tasks_db_path"):
        db_path = Path(custom_path).expanduser().resolve()
    else:
        user_dir = get_user_data_dir()
        db_path = user_dir / "tldw_chatbook_scheduled_tasks.db"
    return db_path
```

`config.py` defaults (add to `DEFAULT_CONFIG_FROM_TOML`):

```toml
[database]
# scheduled_tasks_db_path = "/custom/path.db"  # optional override

[scheduling]
sync_interval_seconds = 300
sync_retry_max_attempts = 10
sync_retry_max_delay_seconds = 300
sync_retry_jitter = true
scheduler_poll_interval_seconds = 30
reminder_catchup_hours = 24
```

### Core tables

**`reminder_tasks`**
- `id` (TEXT PRIMARY KEY) — local UUID.
- `server_id` (TEXT, nullable) — server-assigned ID when synced.
- `owner_id` (TEXT NOT NULL) — `"local"` or `"server:<user_id>"`.
- `title`, `body`
- `schedule_kind` — `one_time` | `recurring`
- `run_at` (TEXT, nullable) — UTC ISO-8601 for one-time.
- `cron` (TEXT, nullable) — cron expression for recurring.
- `timezone` (TEXT, nullable) — IANA timezone name.
- `enabled`, `last_status`, `next_run_at`, `last_run_at`, `missed_at` — UTC ISO-8601.
- `link_type`, `link_id`, `link_url`
- `created_at`, `updated_at`, `sync_version`

**`automation_definitions`**
- `id`, `server_id`, `owner_id`
- `family` — `recurring_question` | `agent_task`
- `name`, `description`
- `lifecycle` — `configured` | `paused` | `archived` | `disabled`
- `health` — starts `execution_unavailable`
- `schedule`, `input`, `config` (JSON)
- `visibility_policy`, `notification_policy`, `approval_policy` (JSON)
- `version` (INTEGER, optimistic lock)
- `preview_id` (TEXT, nullable, FK to `automation_previews.id`) — cleared once consumed; retained for audit.
- `created_by`, `updated_by`, `created_at`, `updated_at`, `archived_at`

**`automation_previews`**
- `id`, `owner_id`, `mode`, `family`, `definition_id`, `definition_version`
- `status` — `valid` | `invalid` | `expired` | `consumed`
- `payload_hash`, `normalized_config`, `validation_errors`, `warnings`
- `visibility_policy`, `schedule_preview`, `redaction_policy`
- `expires_at`, `created_by`, `created_at`, `consumed_at`, `created_definition_id`

**`automation_audit_events`**
- `id`, `definition_id`, `owner_id`, `event_type`, `actor`, `summary`
- `before`, `after` (JSON)
- `request_id`, `idempotency_key`, `created_at`

### Sync helper tables

**`sync_state`**
- `owner_id` PRIMARY KEY
- `last_pull_at`, `last_push_at`, `last_conflict_at`
- `sync_errors` (JSON)

**`sync_mapping`**
- `local_id`, `server_id`, `primitive`, `owner_id`, `created_at`
- `primitive` values: `reminder_task`, `automation_definition`. `watchlist_job` is added in Phase 5 once watchlist schedule metadata moves into `ScheduledTasksDB`.
- UNIQUE(`local_id`, `primitive`, `owner_id`)

**`sync_tombstones`**
- `local_id`, `primitive`, `owner_id`, `deleted_at`, `pushed_at`

**`sync_conflicts`**
- `id`, `local_id`, `primitive`, `owner_id`, `server_state`, `local_state`, `server_state_at`, `created_at`, `resolved_at`, `resolution` (`server` | `local`)
- `retry_count`

### Watchlist projection

Until Phase 5, watchlist jobs are read-only projections built from `Subscriptions_DB` and the server API, not stored in `ScheduledTasksDB`. After Phase 5, schedule metadata migrates into the new DB while source curation remains in Watchlists.

### Indexes and constraints

**`reminder_tasks`:**
- UNIQUE(`owner_id`, `server_id`)
- INDEX(`owner_id`, `enabled`, `next_run_at`)
- INDEX(`owner_id`, `last_status`)
- INDEX(`server_id`)

**`automation_definitions`:**
- UNIQUE(`owner_id`, `server_id`)
- INDEX(`owner_id`, `lifecycle`, `health`)
- INDEX(`owner_id`, `family`)
- INDEX(`server_id`)

**`automation_audit_events`:**
- INDEX(`definition_id`, `created_at`)

**`sync_mapping`:**
- INDEX(`server_id`, `primitive`, `owner_id`)

## Components

New package `tldw_chatbook/Scheduling/` for business logic:

```
Scheduling/
├── __init__.py
├── models.py              # Pydantic models
├── events.py              # Textual messages: ReminderTriggered, SyncCompleted, etc.
├── db/
│   ├── __init__.py
│   ├── scheduled_tasks_db.py   # ScheduledTasksDB extends BaseDB
│   ├── schema.py
│   └── migrations/
├── services/
│   ├── __init__.py
│   ├── scheduling_service.py
│   ├── server_client.py      # wraps TLDWAPIClient / ServerNotificationsService
│   └── sync_engine.py
└── scheduler/
    ├── __init__.py
    ├── loop.py               # async, rebuildable from DB, fake-clock injectable
    ├── queue.py
    ├── worker.py
    └── handlers/
        ├── reminder_handler.py
        └── watchlist_check_handler.py   # Phase 5
```

Screens live in the existing UI location:

```
UI/Screens/scheduling/
├── __init__.py
├── schedules_workbench.py      # replaces existing SchedulesScreen
├── task_detail.py
└── forms/
    ├── reminder_form.py
    └── automation_definition_form.py
```

Styling:

```
tldw_chatbook/css/features/_scheduling.tcss
```

Load the TCSS by adding `"features/_scheduling.tcss"` to the `CSS_MODULES` list in `tldw_chatbook/css/build_css.py`, then regenerate the bundle with `bash build_css.sh` (which invokes `python tldw_chatbook/css/build_css.py`). The app loads the generated `tldw_chatbook/css/tldw_cli_modular.tcss` bundle, so `css/main.tcss` should not be edited for this feature.

### Key services

- `SchedulingService`: single facade for the UI. Handles local/server routing, cache-first reads, and mutation queuing.
- `ServerClient`: thin wrapper. For reminders, it delegates to existing `ServerNotificationsService` / `TLDWAPIClient` (`/api/v1/tasks`). For automation definitions, it calls tldw_server endpoints once they exist (Phase 0 defines the exact contract).
- `SyncEngine`: pull, push, reconcile, conflict detection.
- `SchedulerLoop`: async loop polling DB for due tasks, rebuilding queue on startup, supporting catch-up.

## Data Flow and Sync

### Read flow (cache-first)

```
Screen.list_tasks()
    │
    ▼
SchedulingService.list_tasks()
    │
    ▼
Return local cache immediately
    │
    ▼
Background: SyncEngine.sync_now() if due
    │
    ▼
On SyncCompleted: Screen.refresh()
```

### Write flow (server mode)

```
Screen.update_reminder(id, payload)
    │
    ▼
SchedulingService.update_reminder()
    │
    ├── Server reachable? ──Yes──► ServerClient.patch(...) via ServerNotificationsService
    │                              │
    │                              ▼
    │                         Update local cache from response
    │                              │
    │                              ▼
    │                         UI refresh
    │
    └── No ──► Write local row + pending_mutation row
               SchedulerLoop uses local state
               SyncEngine queues push for reconnect
```

### Sync reconciliation

```
SyncEngine.sync_now(owner_id)
    │
    ├── Pull server list
    ├── Map server_id → local_id via sync_mapping
    ├── For each server record:
    │     ├── If local missing → insert
    │     ├── If local older → update local
    │     └── If local has pending mutation and server newer → record conflict
    ├── Detect server deletions:
    │     ├── If local has tombstone → confirm delete, clean tombstone
    │     └── If local has live record → record conflict (server delete vs local edit)
    ├── Push pending mutations with idempotency keys
    ├── Push tombstones
    ├── Record conflicts
    ├── Update last_pull_at / last_push_at
    └── Emit SyncCompleted or SyncFailed
```

### Conflict rule

- Server wins by default.
- Local pending mutation is moved to `sync_conflicts` with `retry_count = 0`.
- The inspector shows a `ConflictCard` with "Keep local" / "Use server" actions.
- Choosing "Use server" marks the conflict resolved with `resolution = "server"` and discards the local mutation.
- Choosing "Keep local" re-applies the local mutation and queues a new push attempt, incrementing `retry_count`.
- After `sync_retry_max_attempts` failed "Keep local" retries, the conflict card shows "Local change keeps losing to server" and offers "Force local" (overwrites server) or "Archive local" (discard).

### Conflict resolution UI

- A `ConflictCard` widget appears in the inspector when the selected task has an unresolved `sync_conflicts` row.
- It shows:
  - Server state summary and `server_state_at` timestamp.
  - Local pending change summary.
  - Buttons: `Use server`, `Keep local`, `Decide later`.
- The task list row shows a `conflict` badge until resolved.

### Sync timing and configuration

Config keys under `[scheduling]` (must be added to `DEFAULT_CONFIG_FROM_TOML` in `config.py`):

```toml
sync_interval_seconds = 300
sync_retry_max_attempts = 10
sync_retry_max_delay_seconds = 300
sync_retry_jitter = true
scheduler_poll_interval_seconds = 30
reminder_catchup_hours = 24
```

Sync is triggered:
- On `SchedulesWorkbench` mount (if `now - last_pull_at > sync_interval_seconds`).
- Periodically by the scheduler worker using `sync_interval_seconds`.
- Explicitly via the `s` keyboard shortcut.
- On auth state change / reconnect.

### Scheduler loop

- Rebuild queue from `reminder_tasks.next_run_at` on startup.
- Poll every `scheduler_poll_interval_seconds` for newly due reminders.
- On startup, process catch-up reminders from the last `reminder_catchup_hours`; mark older ones as `missed` without notification.
- Handlers run via `asyncio.to_thread` or are async and post UI events through Textual's message bus.
- `next_run_at` is computed using `croniter`:
  - For `one_time`: `run_at` directly.
  - For `recurring`: next occurrence of `cron` in the task's `timezone`, then converted to UTC for storage.

### Datetime serialization

- All datetime columns store UTC ISO-8601 strings.
- The reminder form collects local time in the selected `timezone` and converts to UTC before storage.
- Display converts UTC back to the user's configured timezone.

## UI / Screen Design

Reuse and extend the existing 3-pane destination layout:

```
┌─────────────────────────────────────────────────────────────┐
│ Schedules | Automations, digests, timers, retries | Local+Server │
├─────────────────┬──────────────────────┬────────────────────┤
│                 │                      │                    │
│  Task List      │  Task Detail         │  Status Inspector  │
│                 │                      │                    │
│  [Filter: All]  │  Title: Daily digest │  State: waiting    │
│  ─────────────  │  Type: reminder      │  Next run: 08:00   │
│  • Daily digest │  Schedule: 08:00     │  Last run: --      │
│    [waiting]    │  Timezone: UTC       │  Sync: 1 min ago   │
│  • Weekly RAG   │                      │                    │
│    [paused]     │  [Edit] [Pause]      │  [Run now]*        │
│  • GitHub watch │  [Delete]            │  [Open output]     │
│    [needs att]  │                      │                    │
│                 │                      │                    │
└─────────────────┴──────────────────────┴────────────────────┘

* Run now is disabled for automation definitions (execution_unavailable).
```

### Task list

- Status badge (using theme tokens: `$success`, `$warning`, `$error`, etc.)
- Title
- Type: Reminder / Watchlist monitor / Recurring question / Agent task
- Schedule summary
- Next run
- Source indicator (local / server / conflict)

### Status enum

```python
class TaskStatus(str, Enum):
    WAITING = "waiting"
    RUNNING = "running"
    PAUSED = "paused"
    NEEDS_ATTENTION = "needs_attention"
    BLOCKED = "blocked"
    DISABLED = "disabled"
    ARCHIVED = "archived"
    COMPLETED = "completed"
    FOUND_RESULTS = "found_results"
    MISSED = "missed"
    CONFLICT = "conflict"
```

Tone mapping mirrors `tldw_server` WebUI `scheduled-task-status.ts`:
- `waiting` → processing
- `running` → processing
- `paused` → warning
- `needs_attention` → error
- `blocked` → warning
- `disabled` / `archived` → default
- `completed` / `found_results` → success
- `missed` → error
- `conflict` → error

### Detail pane

- Summary: name, type, status, owner, schedule, next run, last run
- Inputs (collapsed): link, query, agent ref, etc.
- Schedule: plain-language cadence, timezone
- Results: latest output/matches; for definitions: "Execution unavailable in this phase"
- Runs: run history (Phase 5)
- Audit events: for automation definitions

### Inspector pane

- Current state summary
- Next action copy
- Lifecycle actions (pause/resume/archive for definitions; enable/disable for reminders)
- Recovery actions where applicable
- Console follow button
- Sync status / conflict card

### Console-follow integration

The new workbench preserves the existing Console-follow seams from `UI/Screens/schedules_screen.py`:

- Keep stable selector `#schedules-follow-in-console` and method `follow_latest_schedule_run_in_console` (moved to `schedules_workbench.py`).
- For reminders with `link_type == "conversation"`, Console follow opens that conversation via `open_active_home_item_in_console`.
- For reminders with `link_type == "note"`, Console follow opens the note in Library.
- For watchlist jobs with a latest output, Console follow uses the reading-digest payload shape (`open_console_for_live_work`).
- The `ReminderHandler` and `WatchlistCheckHandler` publish active-work items through the existing `home_active_work_adapter` mechanism so Home can also surface them.

### Create/edit flow

- Modal screens for forms.
- Reminder form: title, body, schedule kind, datetime or cron, timezone, link target.
- Automation definition form: family selector, name, schedule, input, policies, preview step.
- Watchlist jobs are not editable here; button opens Watchlists.

### Keyboard shortcuts

Shortcuts are registered via `BaseAppScreen` / screen `BINDINGS` and must be checked against:
- `Constants.py` and `UI/Navigation/shortcut_context.py`
- Existing `TldwCli.BINDINGS` (`ctrl+q`, `ctrl+p`, `f1`, `f6`, `ctrl+1..ctrl+0`)
- Other screen `BINDINGS`

Proposed bindings (global, no input focus):
- `c` — create new
- `r` — run now (selected reminder only)
- `p` — pause/resume
- `d` — delete with confirmation
- `s` — sync now
- `enter` — open detail / Console follow

If any conflict is found during implementation, use `ctrl+` prefix alternates (e.g. `ctrl+r` for run now).

### Action matrix

| Primitive | Run now | Pause/Resume | Edit | Delete | Console follow |
|-----------|---------|--------------|------|--------|----------------|
| Reminder | Yes | N/A (enable/disable) | Yes | Yes | Follow link target |
| Watchlist job | Phase 5 | In Watchlists | In Watchlists | In Watchlists | Latest output |
| Automation definition | Disabled | Yes | Yes (via preview) | Archive | N/A |

### Missed reminder state

When a reminder is past due and older than `reminder_catchup_hours`, it appears in the list with a `missed` status badge. The detail pane shows "Missed at <time>" and offers "Run now" or "Dismiss".

## Error Handling

| Scenario | Behavior |
|----------|----------|
| Server unreachable on read | Return local cache; show offline badge; background retry with exponential backoff and jitter. |
| Server unreachable on write | Write local + pending mutation; queue for push; notify "saved locally, will sync." |
| 400/422 validation | Record conflict; do not retry; show validation error. |
| 401/403 auth | Pause sync; prompt re-authentication. |
| 404 on update | Record conflict; pull fresh state. |
| 409 conflict | Record conflict; pull fresh state. |
| 5xx / timeout | Retry with exponential backoff capped at `sync_retry_max_delay_seconds`; max `sync_retry_max_attempts`; mark as delayed. |
| Sync conflict | Server wins; local pending mutation moved to `sync_conflicts`; inspector shows conflict card. |
| Conflict retry exhausted | Offer "Force local" or "Archive local"; stop auto-retry. |
| Reminder handler fails | Mark run failed; retry up to N times; notify user; keep schedule for next run. |
| Scheduler loop crashes | Log error; restart loop; do not crash app. |
| App restart with due reminders | Catch-up batch limited to `reminder_catchup_hours`; older marked missed. |
| Owner changes | Filter visible records by current owner; queue per-owner pending mutations. |
| Automation preview invalid | Show validation errors inline; block create. |
| Local DB corrupt | Blocking error dialog with reset/log options; no silent in-memory fallback. |

## Reminder Notification UX

When a reminder fires:

1. `ReminderHandler` posts a `ReminderTriggered` Textual event.
2. The app shows a non-blocking `app.notify()` banner with the reminder title and a default action button.
3. The default action follows the reminder's `link_type`:
   - `conversation` → open Console to that conversation.
   - `note` → open Library note editor.
   - `url` → open externally via the system's URL handler.
   - none → open the Schedules workbench detail for the reminder.
4. The reminder is recorded in the existing Notifications inbox via `self.app.notification_dispatch_service.dispatch(category="reminder", title=reminder.title, message=reminder.body or "", source_entity_kind="scheduled_task", source_entity_id=task_id, ...)`. Delivery respects notification settings.
5. If the user is not actively using the app, the reminder is still logged and surfaced on next interaction.

## Testing Strategy

### Test file structure

```
Tests/Scheduling/
├── __init__.py
├── test_scheduled_tasks_db.py
├── test_scheduling_service.py
├── test_sync_engine.py
├── test_server_client.py
├── test_scheduler_loop.py
├── test_reminder_handler.py
└── fixtures/
    └── server_responses/

Tests/UI/          # kept flat to match existing project convention
├── test_schedules_workbench.py
└── test_reminder_form.py
```

### Unit tests

- `ScheduledTasksDB` CRUD, migrations, optimistic locking, owner filtering.
- `SchedulingService` local/server routing, offline pending mutations, cache-first reads.
- `SyncEngine` mapping, idempotency, conflicts, server-wins rule.
- `SchedulerLoop` rebuild from DB, due-time selection, catch-up limits, handler invocation.
- Pydantic model validation and serialization.

### Integration tests

- Service + in-memory SQLite + mocked server client.
- Offline create → online sync → server receives record with correct idempotency key.
- Conflict scenario: local edit pending while server receives different edit.
- Scheduler with fake clock: past-due reminder triggers handler.

### UI tests

- Existing textual mounted regression harness.
- `SchedulesWorkbench` renders list, updates detail/inspector on selection, opens create modal, validates forms, lifecycle actions update state, Console follow enabled correctly.

### Migration tests

- Schema v0 → v1 → v2 with data survival.

### Contract tests

- Recorded server response fixtures under `Tests/Scheduling/fixtures/server_responses/`.
- Verify against real tldw_server instance when available in CI.

### Screenshot QA

- Baseline and final running-app screenshots for the Schedules workbench, per project screenshot QA policy.

## Phased Implementation

Each phase is an epic composed of 2–4 reviewable sub-tasks/PRs.

### Phase 0: API contract spike (prerequisite) — DONE

Status: server automation-definition endpoints confirmed in `tldw_server2/tldw_Server_API/app/api/v1/endpoints/scheduled_tasks_control_plane.py` and `.../schemas/scheduled_tasks_automation_schemas.py`.

All routes are mounted under `/api/v1/scheduled-tasks`.

#### Confirmed endpoints

| Resource | Method | Path | Rate-limit scope |
|----------|--------|------|------------------|
| Capabilities | GET | `/api/v1/scheduled-tasks/capabilities` | `tasks.read` |
| Previews | GET | `/api/v1/scheduled-tasks/previews` | `tasks.read` |
| Previews | POST | `/api/v1/scheduled-tasks/previews` | `tasks.control` |
| Previews | GET | `/api/v1/scheduled-tasks/previews/{preview_id}` | `tasks.read` |
| Definitions | GET | `/api/v1/scheduled-tasks/definitions` | `tasks.read` |
| Definitions | POST | `/api/v1/scheduled-tasks/definitions` | `tasks.control` |
| Definitions | GET | `/api/v1/scheduled-tasks/definitions/{definition_id}` | `tasks.read` |
| Definitions | PATCH | `/api/v1/scheduled-tasks/definitions/{definition_id}` | `tasks.control` |
| Definitions | POST | `/api/v1/scheduled-tasks/definitions/{definition_id}/pause` | `tasks.control` |
| Definitions | POST | `/api/v1/scheduled-tasks/definitions/{definition_id}/resume` | `tasks.control` |
| Definitions | POST | `/api/v1/scheduled-tasks/definitions/{definition_id}/archive` | `tasks.control` |
| Definitions | POST | `/api/v1/scheduled-tasks/definitions/{definition_id}/duplicate` | `tasks.control` |
| Audit | GET | `/api/v1/scheduled-tasks/definitions/{definition_id}/audit` | `tasks.read` |

#### Families, lifecycles, and health values

- `ScheduledTaskAutomationFamily`: `"recurring_question"`, `"agent_task"`
- `ScheduledTaskPreviewMode`: `"create"`, `"update"`
- `ScheduledTaskPreviewStatus`: `"valid"`, `"invalid"`, `"expired"`, `"consumed"`
- `ScheduledTaskDefinitionLifecycle`: `"configured"`, `"paused"`, `"archived"`, `"disabled"`
- `ScheduledTaskDefinitionCreateLifecycle`: `"configured"`, `"paused"`
- `ScheduledTaskDefinitionHealth`: `"ready"`, `"execution_unavailable"`, `"capability_unavailable"`, `"needs_attention"`, `"permission_required"`
- `ScheduledTaskDefinitionDisabledLockKind`: `"none"`, `"admin"`, `"security"`, `"system"`

#### Key request/response schemas

- `ScheduledTaskPreviewCreateRequest` — create/update a preview.
- `ScheduledTaskPreviewResponse` — persisted preview with `status`, `validation_errors`, `schedule_preview`, `expires_at`, `created_definition_id`.
- `ScheduledTaskDefinitionCreateRequest` — `{ preview_id, initial_lifecycle? }`.
- `ScheduledTaskDefinitionUpdateRequest` — `{ preview_id }`.
- `ScheduledTaskDefinitionResponse` — full definition with `version`, `lifecycle`, `health`, `disabled_lock_kind`, `disabled_reason`, schedule/input/config/policy dictionaries.
- `ScheduledTaskDuplicateRequest` — optional `name`/`description` overrides.
- `ScheduledTaskAuditEventResponse` / `ScheduledTaskAuditListResponse` — lifecycle audit trail.
- `ScheduledTaskAutomationCapabilitiesResponse` — per-family availability and action capability matrix.

#### Error contract

Errors return FastAPI `HTTPException` detail shaped as:

```json
{
  "code": "scheduled_task_definition_not_found",
  "message": "Scheduled task definition was not found.",
  "details": {"reason": "definition_not_found"},
  "field_errors": [],
  "retryable": false,
  "correlation_id": "<request_id>"
}
```

Common codes include `scheduled_task_preview_required`, `scheduled_task_definition_not_found`, `scheduled_task_preview_expired`, `scheduled_task_schedule_invalid`, `scheduled_task_definition_version_conflict`, `scheduled_task_definition_archived`, and `scheduled_task_lifecycle_transition_invalid`.

#### Deliverables

- Endpoint documentation: `Tests/Scheduling/fixtures/server_responses/automation_endpoints.md`
- Representative list fixture: `Tests/Scheduling/fixtures/server_responses/automation_definition_list.json`

#### Open point

No dedicated `/runs` endpoints exist under `/api/v1/scheduled-tasks` in the audited server code. Execution/run tracking for automation definitions lives in other subsystems (watchlists, workflows, agent orchestration) and is out of scope for this audit. The chatbook scheduling module should treat automation definition execution as `execution_unavailable` until a server run API is confirmed.

### Phase 1: Data layer

Sub-tasks:
1. Schema-only PR: `reminder_tasks`, `automation_definitions`, `automation_previews`, `automation_audit_events`, sync helper tables.
2. Migrations PR: v0 → v1 migration and rollback test.
3. Models + DB PR: Pydantic models and `ScheduledTasksDB` extending `BaseDB`; `get_scheduled_tasks_db_path()` and `[database] scheduled_tasks_db_path` in `config.py`.

### Phase 2: Control plane + server client

Sub-tasks:
1. `SchedulingService` local-only CRUD for reminders and automation definitions.
2. `ServerClient` for reminders (delegate to `ServerNotificationsService`) and automation definitions (target Phase 0 endpoints).
3. `SyncEngine` with pull/push/reconcile/conflict handling.
4. Watchlist jobs as read-only projection from `Subscriptions_DB`.
5. Auth state transition handling.

### Phase 3: Scheduler

Sub-tasks:
1. `SchedulerLoop`, rebuildable queue, fake-clock support; app lifecycle wiring in `app.py`.
2. `ReminderHandler` with notifications and link resolution.
3. Catch-up on startup and missed-state handling.
4. Integration with `notification_dispatch_service.dispatch()`.

### Phase 4: Screen workbench

Sub-tasks:
1. New `UI/Screens/scheduling/schedules_workbench.py` with 3-pane layout.
2. Forms: reminder create/edit, automation definition create with preview.
3. Inspector with lifecycle actions, conflict card, sync status.
4. Update `UI/Navigation/screen_registry.py` to route `"schedules"` to the new module/class; deprecate `UI/Screens/schedules_screen.py`.
5. TCSS file (`tldw_chatbook/css/features/_scheduling.tcss`), bundle registration in `css/build_css.py` via `CSS_MODULES`, regenerate with `build_css.sh`, and mounted regression tests + screenshots.
6. Regression checklist:
   - Preserve reading-digest fallback Console launch.
   - Preserve active-work adapter Console follow.
   - Keep stable selectors (`#schedules-follow-in-console`, etc.).
   - Re-run existing mounted regressions for `schedules_screen.py`.

### Phase 5: Watchlist migration

Feature flag key: `scheduling.watchlist_migration_enabled = false`.

Column mapping from existing `Subscriptions_DB` to new scheduler tasks:

| Subscriptions_DB field | New scheduler field |
|------------------------|---------------------|
| `id` | `task_id` (local) / `server_id` |
| `name` | `title` |
| `description` | `description` |
| `is_active` | `enabled` |
| `is_paused` | paused via lifecycle |
| `check_frequency` | cron/interval schedule |
| `last_checked` | `last_run_at` |
| `type` (rss/atom/url/...) | handler routing |
| `priority` | task priority |

Sub-tasks:
1. ADR for migration: adaptive frequency, concurrency limits, run-history migration, feature flag, rollback plan.
2. Implement `WatchlistCheckHandler` reusing existing `FeedMonitor`/`URLMonitor`.
3. Add feature flag `scheduling.watchlist_migration_enabled`. When `false`, old scheduler runs; when `true`, new scheduler owns watchlist checks.
4. Run both schedulers side-by-side in staging; compare outputs.
5. Route `Subscriptions_DB` checks through `WatchlistCheckHandler`.
6. Remove old `Subscriptions/scheduler.py` after regression tests pass.

## Dependencies

- Reuses existing tldw_chatbook patterns: Textual screens, workers, `BaseAppScreen`, `DestinationModeStrip`, server runtime interop, `Subscriptions_DB`, `BaseDB`.
- Reuses existing `TLDWAPIClient` `/api/v1/tasks` reminder endpoints and `ServerNotificationsService`.
- New base dependency: `croniter` for cron next-run calculation. Add to `pyproject.toml` base dependencies.
- Mirrors tldw_server automation-definition API schemas once Phase 0 confirms them.

## Open Questions

1. Whether to include browser-extension-style template creation in Phase 4 or defer.
2. Whether automation-definition server endpoints already exist in the target tldw_server deployment (to be resolved in Phase 0).

## References

- `tldw_chatbook/UI/Screens/schedules_screen.py`
- `tldw_chatbook/Subscriptions/scheduler.py`
- `tldw_chatbook/Subscriptions/textual_scheduler_worker.py`
- `tldw_chatbook/DB/base_db.py`
- `tldw_chatbook/config.py`
- `tldw_chatbook/app.py`
- `tldw_chatbook/tldw_api/client.py`
- `tldw_chatbook/tldw_api/notifications_reminders_schemas.py`
- `tldw_chatbook/Notifications/server_notifications_service.py`
- `tldw_chatbook/UI/Navigation/screen_registry.py`
- `tldw_chatbook/css/main.tcss`
- `tldw_chatbook/backlog/tasks/task-11.6 - Phase-4.6-Schedules-and-Workflows-run-control.md`
- `tldw_chatbook/backlog/tasks/task-14.7 - Screen-QA-Schedules.md`
- `tldw_chatbook/backlog/tasks/task-3.7 - Phase-3.7-Launch-active-Schedules-run-from-Schedules-into-Console.md`
- `tldw_server2/tldw_Server_API/app/services/scheduled_tasks_control_plane_service.py`
- `tldw_server2/tldw_Server_API/app/services/scheduled_task_automation_service.py`
- `tldw_server2/apps/packages/ui/src/services/scheduled-tasks-control-plane.ts`
- `tldw_server2/Docs/superpowers/specs/2026-06-01-scheduled-tasks-automation-workbench-prd-design.md`
