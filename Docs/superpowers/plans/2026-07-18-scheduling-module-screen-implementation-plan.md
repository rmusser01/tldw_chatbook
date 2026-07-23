# Scheduling Module + Screen Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a hybrid local/server scheduling module and TUI workbench in tldw_chatbook, porting tldw_server's unified scheduled-tasks concepts while preserving existing Console-follow and screenshot QA behavior.

**Architecture:** A new `Scheduling/` business-logic package owns `ScheduledTasksDB`, `SchedulingService`, `SyncEngine`, and `SchedulerLoop`. A new `UI/Screens/scheduling/` workbench replaces the placeholder `SchedulesScreen`. The scheduler runs as a Textual worker, polls the local DB for due reminders, and syncs with tldw_server's existing `/api/v1/tasks` reminder endpoints (and future automation-definition endpoints) when authenticated.

**Tech Stack:** Python 3.11+, Textual 3.3+, SQLite, Pydantic, loguru, croniter, pytest, textual-dev.

**Prerequisites before Phase 1:**
1. Create Backlog.md task for this work.
2. Create ADR `backlog/decisions/NNN-local-server-hybrid-scheduled-tasks.md` covering local/server storage boundary, server-wins conflict policy, per-server owner identity, and migration path.
3. Complete Phase 0 API contract spike.

---

## File structure

### New business-logic files

| File | Responsibility |
|------|----------------|
| `tldw_chatbook/Scheduling/__init__.py` | Package exports. |
| `tldw_chatbook/Scheduling/models.py` | Pydantic models mirroring tldw_server schemas and local enums. |
| `tldw_chatbook/Scheduling/events.py` | Textual messages: `ReminderTriggered`, `SyncCompleted`, `SyncFailed`. |
| `tldw_chatbook/Scheduling/db/__init__.py` | DB package exports. |
| `tldw_chatbook/Scheduling/db/schema.py` | Table schemas and DDL. |
| `tldw_chatbook/Scheduling/db/scheduled_tasks_db.py` | `ScheduledTasksDB` class extending `BaseDB`. |
| `tldw_chatbook/Scheduling/db/migrations/v0_to_v1.py` | Initial schema migration. |
| `tldw_chatbook/Scheduling/services/__init__.py` | Services exports. |
| `tldw_chatbook/Scheduling/services/server_client.py` | Thin wrapper over `TLDWAPIClient` / `ServerNotificationsService`; safe when server unavailable. |
| `tldw_chatbook/Scheduling/services/sync_engine.py` | Pull/push/reconcile/conflict detection. |
| `tldw_chatbook/Scheduling/services/scheduling_service.py` | Local-first facade used by the UI. |
| `tldw_chatbook/Scheduling/scheduler/__init__.py` | Scheduler package exports. |
| `tldw_chatbook/Scheduling/scheduler/loop.py` | `SchedulerLoop` async polling loop. |
| `tldw_chatbook/Scheduling/scheduler/queue.py` | In-memory priority queue rebuilt from DB. |
| `tldw_chatbook/Scheduling/scheduler/handlers/reminder_handler.py` | Executes due reminders. |
| `tldw_chatbook/Scheduling/scheduler/handlers/watchlist_check_handler.py` | Phase 5: executes watchlist checks. |

### New UI files

| File | Responsibility |
|------|----------------|
| `tldw_chatbook/UI/Screens/scheduling/__init__.py` | Screen exports. |
| `tldw_chatbook/UI/Screens/scheduling/schedules_workbench.py` | Main 3-pane workbench screen. |
| `tldw_chatbook/UI/Screens/scheduling/task_detail.py` | Detail pane widget. |
| `tldw_chatbook/UI/Screens/scheduling/forms/reminder_form.py` | Modal form for creating/editing reminders. |
| `tldw_chatbook/UI/Screens/scheduling/forms/automation_definition_form.py` | Modal form for automation definitions. |
| `tldw_chatbook/css/features/_scheduling.tcss` | Scheduling-specific TCSS. |

### Modified existing files

| File | Change |
|------|--------|
| `tldw_chatbook/config.py` | Add `get_scheduled_tasks_db_path()` and `[scheduling]` defaults. |
| `tldw_chatbook/app.py` | Import new scheduling modules at top; wire `SchedulingService` and `SchedulerLoop` into app lifecycle in `_wire_watchlists_and_notifications_services()`. |
| `tldw_chatbook/UI/Navigation/screen_registry.py` | Route `"schedules"` to new workbench. |
| `tldw_chatbook/css/build_css.py` | Add `"features/_scheduling.tcss"` to `CSS_MODULES`. |
| `pyproject.toml` | Add `croniter` to base dependencies. |

### New test files

| File | Responsibility |
|------|----------------|
| `Tests/Scheduling/test_scheduled_tasks_db.py` | DB CRUD, migrations, locking. |
| `Tests/Scheduling/test_scheduling_service.py` | Local/server routing, offline mutations. |
| `Tests/Scheduling/test_sync_engine.py` | Sync scenarios, conflicts. |
| `Tests/Scheduling/test_server_client.py` | Server client delegation and unavailable handling. |
| `Tests/Scheduling/test_scheduler_loop.py` | Due-time selection, catch-up. |
| `Tests/Scheduling/test_reminder_handler.py` | Notification/link resolution. |
| `Tests/UI/test_schedules_workbench.py` | Mounted UI regressions for the new workbench. |
| `Tests/UI/test_reminder_form.py` | Form validation. |

> **Note:** `Tests/UI/` is kept flat to match the existing project convention. The original `Tests/UI/test_destination_shells.py` imports `SchedulesScreen` and must be updated when the registry is repointed.

---

## Phase 0: API contract spike

**Goal:** Confirm the exact server API surface before building local models.

### Task 0.1: Audit existing reminder endpoints

**Files:**
- Read: `tldw_chatbook/tldw_api/client.py:7350-7380`
- Read: `tldw_chatbook/tldw_api/notifications_reminders_schemas.py`
- Read: `tldw_chatbook/Notifications/server_notifications_service.py:162-238`

- [ ] **Step 1: Document request/response shapes**

Record the exact Pydantic models used by `create_reminder_task`, `update_reminder_task`, `delete_reminder_task`, `list_reminder_tasks`, and `get_reminder_task`.

- [ ] **Step 2: Record fixture**

Save a recorded JSON response from `list_reminder_tasks` to `Tests/Scheduling/fixtures/server_responses/reminder_list.json`.

- [ ] **Step 3: Verify no automation-definition endpoints exist in chatbook client**

Search `tldw_chatbook/tldw_api/client.py` for `scheduled-tasks`, `automation`, `definition`. Confirm none exist.

- [ ] **Step 4: Commit**

```bash
git add Tests/Scheduling/fixtures/
git commit -m "docs: record server reminder API fixtures for scheduling module"
```

### Task 0.2: Audit tldw_server automation-definition endpoints

**Files:**
- Read (if available): `tldw_server2/tldw_Server_API/app/api/v1/endpoints/scheduled_tasks_control_plane.py`
- Read (if available): `tldw_server2/tldw_Server_API/app/api/v1/schemas/scheduled_tasks_automation_schemas.py`

- [ ] **Step 1: List endpoints and methods**

Document URL paths, HTTP methods, request schemas, and response schemas.

- [ ] **Step 2: Decide scope**

If endpoints exist, record fixtures. If the server repo is unavailable, document that automation definitions will be local-only until server support is confirmed.

- [ ] **Step 3: Update design doc**

Edit `Docs/superpowers/specs/2026-07-18-scheduling-module-screen-design.md` "Phase 0" section with the confirmed contract.

- [ ] **Step 4: Commit**

```bash
git commit -m "docs: confirm server automation-definition API contract"
```

---

## Phase 1: Data layer

### Task 1.1: Add config helpers

**Files:**
- Modify: `tldw_chatbook/config.py`
- Test: `Tests/Scheduling/test_scheduled_tasks_db.py` (assert DB path helper behavior)

- [ ] **Step 1: Add default config entries**

Insert into `DEFAULT_CONFIG_FROM_TOML`:

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

- [ ] **Step 2: Add DB path helper**

```python
from pathlib import Path
from contextlib import closing

def get_scheduled_tasks_db_path() -> Path:
    custom_path = get_cli_setting("database", "scheduled_tasks_db_path", None)
    if custom_path and custom_path != DEFAULT_CONFIG_FROM_TOML.get("database", {}).get("scheduled_tasks_db_path"):
        return Path(custom_path).expanduser().resolve()
    return get_user_data_dir() / "tldw_chatbook_scheduled_tasks.db"
```

- [ ] **Step 3: Write test**

Create `Tests/Scheduling/test_scheduled_tasks_db.py`:

```python
from tldw_chatbook.config import get_scheduled_tasks_db_path

def test_get_scheduled_tasks_db_path_returns_path():
    path = get_scheduled_tasks_db_path()
    assert path.name == "tldw_chatbook_scheduled_tasks.db"
```

- [ ] **Step 4: Run test**

```bash
pytest Tests/Scheduling/test_scheduled_tasks_db.py::test_get_scheduled_tasks_db_path_returns_path -v
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/config.py Tests/Scheduling/test_scheduled_tasks_db.py
git commit -m "feat(scheduling): add config helpers and defaults"
```

### Task 1.2: Define Pydantic models

**Files:**
- Create: `tldw_chatbook/Scheduling/models.py`
- Test: `Tests/Scheduling/test_models.py`

- [ ] **Step 1: Write failing enum test**

```python
from tldw_chatbook.Scheduling.models import TaskStatus

def test_task_status_values():
    assert TaskStatus.WAITING.value == "waiting"
```

Run: `pytest Tests/Scheduling/test_models.py -v` → FAIL

- [ ] **Step 2: Implement enums**

```python
class TaskStatus(str, Enum): ...
class ScheduleKind(str, Enum): ...
class Lifecycle(str, Enum): ...
class Health(str, Enum): ...
class AutomationFamily(str, Enum): ...
```

- [ ] **Step 3: Run test**

Expected: PASS

- [ ] **Step 4: Write failing reminder model test**

```python
from tldw_chatbook.Scheduling.models import ReminderTask

def test_reminder_task_defaults():
    task = ReminderTask(title="Daily digest")
    assert task.status == TaskStatus.WAITING
```

Run: FAIL

- [ ] **Step 5: Implement ReminderTask model**

- [ ] **Step 6: Run test**

Expected: PASS

- [ ] **Step 7: Implement AutomationDefinition, Preview, AuditEvent models**

Add one at a time with tests.

- [ ] **Step 8: Commit**

```bash
git add tldw_chatbook/Scheduling/models.py Tests/Scheduling/test_models.py
git commit -m "feat(scheduling): add scheduling pydantic models"
```

### Task 1.3: Define schema DDL

**Files:**
- Create: `tldw_chatbook/Scheduling/db/schema.py`
- Test: `Tests/Scheduling/test_schema.py`

- [ ] **Step 1: Write schema creation SQL**

Create `schema.py` with `CREATE TABLE` statements for `schema_version`, `reminder_tasks`, `automation_definitions`, `automation_previews`, `automation_audit_events`, sync helper tables, indexes, and constraints. Include the standard `schema_version (version INTEGER PRIMARY KEY)` table, matching `Subscriptions_DB.py`.

- [ ] **Step 2: Test schema can execute**

```python
import sqlite3
from tldw_chatbook.Scheduling.db.schema import CREATE_SCHEMA_SQL

def test_schema_executes():
    conn = sqlite3.connect(":memory:")
    conn.executescript(CREATE_SCHEMA_SQL)
```

- [ ] **Step 3: Run test**

Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add tldw_chatbook/Scheduling/db/schema.py Tests/Scheduling/test_schema.py
git commit -m "feat(scheduling): add scheduled tasks schema DDL"
```

### Task 1.4: Implement ScheduledTasksDB reminder CRUD

**Files:**
- Create: `tldw_chatbook/Scheduling/db/scheduled_tasks_db.py`
- Test: `Tests/Scheduling/test_scheduled_tasks_db.py`

- [ ] **Step 1: Write failing create/get test**

```python
from tldw_chatbook.Scheduling.db.scheduled_tasks_db import ScheduledTasksDB

def test_create_and_get_reminder_task(tmp_path):
    db = ScheduledTasksDB(tmp_path / "test.db")
    task_id = db.create_reminder_task(owner_id="local", title="Test")
    task = db.get_reminder_task(task_id)
    assert task["title"] == "Test"
```

Run: FAIL

- [ ] **Step 2: Implement BaseDB subclass and schema initialization**

Follow the existing `Subscriptions_DB.py` pattern. `BaseDB` does not expose `_connection`; use `_get_connection()`:

```python
from contextlib import closing

class ScheduledTasksDB(BaseDB):
    _CURRENT_SCHEMA_VERSION = 1

    def _initialize_schema(self):
        with closing(self._get_connection()) as conn:
            conn.executescript(CREATE_SCHEMA_SQL)
            conn.execute("INSERT INTO schema_version (version) VALUES (?)", (self._CURRENT_SCHEMA_VERSION,))
```

- [ ] **Step 3: Implement reminder create and get**

- [ ] **Step 4: Run tests**

Expected: PASS

- [ ] **Step 5: Implement reminder list, update, delete**

- [ ] **Step 6: Implement `list_reminder_tasks(owner_id, enabled, status)`**

- [ ] **Step 7: Implement `reminders_due_before(now)`**

- [ ] **Step 8: Implement `get_schema_version()`**

- [ ] **Step 9: Run tests**

Expected: PASS

- [ ] **Step 10: Commit**

```bash
git add tldw_chatbook/Scheduling/db/scheduled_tasks_db.py Tests/Scheduling/test_scheduled_tasks_db.py
git commit -m "feat(scheduling): implement reminder CRUD and due queries"
```

### Task 1.5: Implement automation definition CRUD

**Files:**
- Modify: `tldw_chatbook/Scheduling/db/scheduled_tasks_db.py`
- Test: `Tests/Scheduling/test_scheduled_tasks_db.py`

- [ ] **Step 1: Write failing create/get test**

```python
def test_create_automation_definition(tmp_path):
    db = ScheduledTasksDB(tmp_path / "test.db")
    def_id = db.create_automation_definition(owner_id="local", family="recurring_question", name="Q")
    row = db.get_automation_definition(def_id)
    assert row["name"] == "Q"
```

Run: FAIL

- [ ] **Step 2: Implement create and get**

- [ ] **Step 3: Implement list, update, delete, audit event logging**

- [ ] **Step 4: Run tests**

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Scheduling/db/scheduled_tasks_db.py Tests/Scheduling/test_scheduled_tasks_db.py
git commit -m "feat(scheduling): implement automation definition CRUD"
```

### Task 1.6: Schema migration

**Files:**
- Create: `tldw_chatbook/Scheduling/db/migrations/v0_to_v1.py`
- Test: `Tests/Scheduling/test_migrations.py`

- [ ] **Step 1: Implement v0→v1 migration**

Apply `CREATE_SCHEMA_SQL` and record `schema_version = 1` using `_get_connection()`:

```python
from contextlib import closing

def migrate_v0_to_v1(db: BaseDB):
    with closing(db._get_connection()) as conn:
        conn.executescript(CREATE_SCHEMA_SQL)
        conn.execute("INSERT INTO schema_version (version) VALUES (?)", (1,))
```

- [ ] **Step 2: Test migration from blank DB**

```python
def test_migration_v0_to_v1(tmp_path):
    db = ScheduledTasksDB(tmp_path / "test.db")
    assert db.get_schema_version() == 1
```

- [ ] **Step 3: Run test**

Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add tldw_chatbook/Scheduling/db/migrations/ Tests/Scheduling/test_migrations.py
git commit -m "feat(scheduling): add v0 to v1 schema migration"
```

---

## Phase 2: Control plane + server client

### Task 2.1: Implement ServerClient for reminders

**Files:**
- Create: `tldw_chatbook/Scheduling/services/server_client.py`
- Test: `Tests/Scheduling/test_server_client.py`

- [ ] **Step 1: Write failing delegation test**

```python
import pytest
from unittest.mock import AsyncMock
from tldw_chatbook.Scheduling.services.server_client import SchedulingServerClient

@pytest.mark.asyncio
async def test_create_reminder_delegates_to_notifications_service():
    svc = AsyncMock()
    client = SchedulingServerClient(notifications_service=svc)
    await client.create_reminder(title="Test", schedule_kind="one_time")
    svc.create_reminder.assert_awaited_once_with(title="Test", schedule_kind="one_time")
```

Run: FAIL

- [ ] **Step 2: Implement client with unavailable handling**

```python
class ServerUnavailableError(Exception):
    """Raised when the server client is invoked while no server is connected."""

class SchedulingServerClient:
    def __init__(self, notifications_service=None, api_client=None):
        self.notifications_service = notifications_service
        self.api_client = api_client

    def _is_available(self):
        return self.notifications_service is not None and getattr(self.notifications_service, "client", None) is not None

    async def create_reminder(self, **payload):
        if not self._is_available():
            raise ServerUnavailableError("server not available")
        return await self.notifications_service.create_reminder(**payload)

    async def update_reminder(self, task_id, **payload):
        if not self._is_available():
            raise ServerUnavailableError("server not available")
        return await self.notifications_service.update_reminder(task_id, **payload)

    async def delete_reminder(self, task_id):
        if not self._is_available():
            raise ServerUnavailableError("server not available")
        return await self.notifications_service.delete_reminder(task_id)

    async def list_reminders(self):
        if not self._is_available():
            raise ServerUnavailableError("server not available")
        return await self.notifications_service.list_reminders()

    async def get_reminder(self, task_id):
        if not self._is_available():
            raise ServerUnavailableError("server not available")
        return await self.notifications_service.get_reminder(task_id)
```

- [ ] **Step 3: Run tests**

Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add tldw_chatbook/Scheduling/services/server_client.py Tests/Scheduling/test_server_client.py
git commit -m "feat(scheduling): add server client for reminders"
```

### Task 2.2: Implement SyncEngine pull

**Files:**
- Create: `tldw_chatbook/Scheduling/services/sync_engine.py`
- Test: `Tests/Scheduling/test_sync_engine.py`

- [ ] **Step 1: Write failing pull test**

```python
@pytest.mark.asyncio
async def test_sync_pull_inserts_server_reminder(tmp_path):
    db = ScheduledTasksDB(tmp_path / "db.db")
    server_client = AsyncMock()
    server_client.list_reminders.return_value = {"items": [{"id": "srv-1", "title": "Server"}]}
    engine = SyncEngine(db, server_client, owner_id="server:1")
    await engine._pull()
    rows = db.list_reminder_tasks(owner_id="server:1")
    assert len(rows) == 1
```

Run: FAIL

- [ ] **Step 2: Implement `_pull` and mapping insertion**

- [ ] **Step 3: Run tests**

Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add tldw_chatbook/Scheduling/services/sync_engine.py Tests/Scheduling/test_sync_engine.py
git commit -m "feat(scheduling): implement SyncEngine pull"
```

### Task 2.3: Implement SyncEngine push and conflicts

**Files:**
- Modify: `tldw_chatbook/Scheduling/services/sync_engine.py`
- Test: `Tests/Scheduling/test_sync_engine.py`

- [ ] **Step 1: Write offline→online push test**

```python
@pytest.mark.asyncio
async def test_sync_pushes_local_reminder(tmp_path):
    db = ScheduledTasksDB(tmp_path / "db.db")
    db.create_reminder_task(owner_id="local", title="Local")
    server_client = AsyncMock()
    server_client.list_reminders.return_value = {"items": []}
    server_client.create_reminder.return_value = {"id": "srv-1", "title": "Local"}
    engine = SyncEngine(db, server_client, owner_id="local")
    await engine.sync_now()
    server_client.create_reminder.assert_awaited_once()
```

Run: FAIL

- [ ] **Step 2: Implement `_push` and `_reconcile_record`**

- [ ] **Step 3: Implement conflict detection and `sync_conflicts` writes**

- [ ] **Step 4: Run tests**

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Scheduling/services/sync_engine.py Tests/Scheduling/test_sync_engine.py
git commit -m "feat(scheduling): implement SyncEngine push and conflicts"
```

### Task 2.4: Implement SchedulingService

**Files:**
- Create: `tldw_chatbook/Scheduling/services/scheduling_service.py`
- Test: `Tests/Scheduling/test_scheduling_service.py`

- [ ] **Step 1: Write local-only CRUD test**

```python
@pytest.mark.asyncio
async def test_create_reminder_local(tmp_path):
    db = ScheduledTasksDB(tmp_path / "db.db")
    svc = SchedulingService(db=db, runtime_source="local")
    task = await svc.create_reminder({"title": "Test"})
    assert task.title == "Test"
```

Run: FAIL

- [ ] **Step 2: Implement service local CRUD**

- [ ] **Step 3: Implement server routing and cache-first reads**

When `server_client` is `None` or unavailable, fall back to local DB and queue pending mutations.

- [ ] **Step 4: Run tests**

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Scheduling/services/scheduling_service.py Tests/Scheduling/test_scheduling_service.py
git commit -m "feat(scheduling): implement SchedulingService facade"
```

### Task 2.5: Watchlist projection

**Files:**
- Create: `tldw_chatbook/Scheduling/services/watchlist_projection.py`
- Test: `Tests/Scheduling/test_watchlist_projection.py`

- [ ] **Step 1: Write projection test**

```python
def test_watchlist_projection_from_subscriptions_db():
    subs_db = SubscriptionsDB(":memory:")
    # ... insert subscription ...
    projection = WatchlistProjection(subs_db)
    tasks = projection.list_jobs(owner_id="local")
    assert len(tasks) == 1
```

Run: FAIL

- [ ] **Step 2: Implement projection**

Read from `Subscriptions_DB` and normalize rows into the shared `ScheduledTask` model.

- [ ] **Step 3: Run tests**

Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add tldw_chatbook/Scheduling/services/watchlist_projection.py Tests/Scheduling/test_watchlist_projection.py
git commit -m "feat(scheduling): add watchlist job projection"
```

---

## Phase 3: Scheduler

### Task 3.1: Add croniter dependency

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Add `croniter` to base dependencies**

```toml
dependencies = [
    ...,
    "croniter>=1.4.0",
    ...,
]
```

- [ ] **Step 2: Reinstall in venv**

```bash
pip install croniter
```

- [ ] **Step 3: Commit**

```bash
git add pyproject.toml
git commit -m "build: add croniter dependency for scheduling"
```

### Task 3.2: Implement SchedulerLoop

**Files:**
- Create: `tldw_chatbook/Scheduling/scheduler/loop.py`
- Create: `tldw_chatbook/Scheduling/scheduler/queue.py`
- Test: `Tests/Scheduling/test_scheduler_loop.py`

- [ ] **Step 1: Write due-task test with fake clock**

```python
@pytest.mark.asyncio
async def test_scheduler_triggers_due_reminder(tmp_path):
    db = ScheduledTasksDB(tmp_path / "db.db")
    db.create_reminder_task(owner_id="local", title="Test", next_run_at="2026-01-01T00:00:00+00:00")
    handler = AsyncMock()
    loop = SchedulerLoop(db, handlers={"reminder": handler}, clock=lambda: datetime(2026, 1, 1, tzinfo=timezone.utc))
    await loop.tick()
    handler.assert_awaited_once()
```

Run: FAIL

- [ ] **Step 2: Implement loop and queue**

DB access must not block the event loop. Wrap synchronous `BaseDB` calls in `asyncio.to_thread`:

```python
class SchedulerLoop:
    async def run(self):
        while self.running:
            await self.tick()
            await asyncio.sleep(self.poll_interval)

    async def tick(self):
        due = await asyncio.to_thread(self.db.reminders_due_before, self.clock())
        for task in due:
            await self.handlers["reminder"](task)
```

- [ ] **Step 3: Run tests**

Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add tldw_chatbook/Scheduling/scheduler/ Tests/Scheduling/test_scheduler_loop.py
git commit -m "feat(scheduling): implement SchedulerLoop with fake clock"
```

### Task 3.3: Implement ReminderHandler

**Files:**
- Create: `tldw_chatbook/Scheduling/scheduler/handlers/reminder_handler.py`
- Test: `Tests/Scheduling/test_reminder_handler.py`

- [ ] **Step 1: Write notification dispatch test**

`NotificationDispatchService.dispatch` is synchronous, so use a plain `Mock`:

```python
import pytest
from unittest.mock import Mock
from tldw_chatbook.Scheduling.scheduler.handlers.reminder_handler import ReminderHandler

@pytest.mark.asyncio
async def test_reminder_handler_dispatches_notification():
    dispatch = Mock()
    handler = ReminderHandler(dispatch_service=dispatch)
    await handler.handle({"id": "1", "title": "T", "body": "B", "link_type": None})
    dispatch.dispatch.assert_called_once()
```

Run: FAIL

- [ ] **Step 2: Implement handler**

Dispatch via `notification_dispatch_service.dispatch(...)` synchronously, post `ReminderTriggered`, update `last_run_at` / `next_run_at` via `asyncio.to_thread`.

- [ ] **Step 3: Run tests**

Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add tldw_chatbook/Scheduling/scheduler/handlers/reminder_handler.py Tests/Scheduling/test_reminder_handler.py
git commit -m "feat(scheduling): implement ReminderHandler"
```

### Task 3.4: Wire scheduler into app lifecycle

**Files:**
- Modify: `tldw_chatbook/app.py`

- [ ] **Step 1: Add module imports at top of app.py**

```python
from tldw_chatbook.Scheduling.db.scheduled_tasks_db import ScheduledTasksDB
from tldw_chatbook.Scheduling.services.scheduling_service import SchedulingService
from tldw_chatbook.Scheduling.services.server_client import SchedulingServerClient
from tldw_chatbook.Scheduling.scheduler.loop import SchedulerLoop
from tldw_chatbook.Scheduling.scheduler.handlers.reminder_handler import ReminderHandler
```

- [ ] **Step 2: Construct services after notification dispatch service exists**

Locate `_wire_watchlists_and_notifications_services()` (called from `TldwCli.__init__`). Immediately after `self.notification_dispatch_service` is initialized, add:

```python
server_client = None
if self.server_notifications_service is not None:
    server_client = SchedulingServerClient(self.server_notifications_service)

self.scheduling_service = SchedulingService(
    db=ScheduledTasksDB(get_scheduled_tasks_db_path()),
    server_client=server_client,
    runtime_source="local",
)
self.scheduler_loop = SchedulerLoop(
    self.scheduling_service.db,
    handlers={
        "reminder": ReminderHandler(dispatch_service=self.notification_dispatch_service),
    },
)
```

> **Auth-state transitions:** `runtime_source="local"` is the initial default. `SchedulingService` exposes `set_owner(owner_id)` (or equivalent) that Phase 2 auth-transition tasks will call on login/logout/server-switch. When the owner becomes `"server:<user_id>"`, the service routes reads/writes through `server_client` if available; on logout it reverts to `"local"` and server-owned rows are filtered from visible results but retained in the DB.

- [ ] **Step 3: Start worker in `on_mount()`**

```python
self.scheduler_worker = self.run_worker(self.scheduler_loop.run(), exclusive=True, group="scheduling")
```

- [ ] **Step 4: Stop worker in `on_unmount()`**

`TldwCli.on_unmount` is async, so await clean shutdown:

```python
if self.scheduler_loop:
    self.scheduler_loop.running = False
if self.scheduler_worker:
    await self.scheduler_worker.wait(timeout=5)
    if not self.scheduler_worker.is_finished:
        self.scheduler_worker.cancel()
```

- [ ] **Step 5: Run app smoke test**

```bash
python -m tldw_chatbook.app --help
```

Expected: no import errors

- [ ] **Step 6: Commit**

```bash
git add tldw_chatbook/app.py
git commit -m "feat(scheduling): wire scheduler into app lifecycle"
```

---

## Phase 4: Screen workbench

### Task 4.1: Add TCSS file and register it

**Files:**
- Create: `tldw_chatbook/css/features/_scheduling.tcss`
- Modify: `tldw_chatbook/css/build_css.py`

- [ ] **Step 1: Add placeholder TCSS**

```css
#scheduling-workbench {
    layout: horizontal;
    height: 1fr;
    min-height: 10;
    padding: 1;
    margin: 0;
    overflow: hidden;
    border: solid $ds-grid-line;
}
#scheduling-list-pane { width: 3fr; min-width: 30; border: solid $ds-grid-line; padding: 1; }
#scheduling-detail-pane { width: 4fr; min-width: 52; border: solid $ds-grid-line; padding: 1; }
#scheduling-inspector-pane { width: 2fr; min-width: 34; border: solid $ds-grid-line; padding: 1; }
```

- [ ] **Step 2: Register in build_css.py**

Add `"features/_scheduling.tcss"` to the `CSS_MODULES` list.

- [ ] **Step 3: Regenerate bundle**

```bash
bash build_css.sh
```

- [ ] **Step 4: Commit**

```bash
git add tldw_chatbook/css/features/_scheduling.tcss tldw_chatbook/css/build_css.py tldw_chatbook/css/tldw_cli_modular.tcss
git commit -m "feat(scheduling): add scheduling TCSS module"
```

### Task 4.2: Implement ReminderForm

**Files:**
- Create: `tldw_chatbook/UI/Screens/scheduling/forms/reminder_form.py`
- Test: `Tests/UI/test_reminder_form.py`

- [ ] **Step 1: Write form validation test with a test app**

`BaseAppScreen` requires `app_instance` in `__init__`. Build a minimal test app:

```python
import pytest
from textual.app import App
from tldw_chatbook.UI.Screens.scheduling.forms.reminder_form import ReminderForm

class FormTestApp(App):
    pass

@pytest.mark.asyncio
async def test_reminder_form_requires_title():
    app = FormTestApp()
    async with app.run_test() as pilot:
        await pilot.push_screen(ReminderForm(app_instance=pilot.app))
        await pilot.click("#reminder-save")
        assert "title is required" in pilot.app.screen.query_one("#reminder-errors").renderable.plain.lower()
```

Run: FAIL

- [ ] **Step 2: Implement form**

Use Textual `ModalScreen` with `Input`, `TextArea`, `Select`, `Button`. Emit `ReminderFormSubmitted` event on save.

- [ ] **Step 3: Run tests**

Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add tldw_chatbook/UI/Screens/scheduling/forms/reminder_form.py Tests/UI/test_reminder_form.py
git commit -m "feat(scheduling): add reminder create/edit form"
```

### Task 4.3: Implement SchedulesWorkbench shell

**Files:**
- Create: `tldw_chatbook/UI/Screens/scheduling/schedules_workbench.py`
- Test: `Tests/UI/test_schedules_workbench.py`

- [ ] **Step 1: Write render test with a test app**

```python
import pytest
from textual.app import App
from tldw_chatbook.UI.Screens.scheduling.schedules_workbench import SchedulesWorkbench

class WorkbenchTestApp(App):
    scheduling_service = None

@pytest.mark.asyncio
async def test_schedules_workbench_renders_panes():
    app = WorkbenchTestApp()
    async with app.run_test() as pilot:
        await pilot.app.push_screen(SchedulesWorkbench(app_instance=pilot.app))
        assert pilot.app.screen.query_one("#scheduling-list-pane") is not None
        assert pilot.app.screen.query_one("#scheduling-detail-pane") is not None
        assert pilot.app.screen.query_one("#scheduling-inspector-pane") is not None
```

Run: FAIL

- [ ] **Step 2: Implement shell class extending BaseAppScreen**

```python
class SchedulesWorkbench(BaseAppScreen):
    BINDINGS = [
        Binding("ctrl+c", "create_reminder", "Create"),
        Binding("ctrl+r", "run_now", "Run now"),
        Binding("ctrl+p", "pause_resume", "Pause/Resume"),
        Binding("ctrl+d", "delete", "Delete"),
        Binding("ctrl+s", "sync_now", "Sync"),
    ]

    def compose_content(self):
        with Horizontal(id="scheduling-workbench"):
            with Vertical(id="scheduling-list-pane"):
                yield Static("Schedule Queue")
            with Vertical(id="scheduling-detail-pane"):
                yield Static("Run Detail")
            with Vertical(id="scheduling-inspector-pane"):
                yield Static("Status Inspector")
```

> **Shortcut note:** `ctrl+c`, `ctrl+r`, `ctrl+p`, `ctrl+d`, `ctrl+s` are per-screen bindings chosen to avoid collisions with Settings (`s`, `r`) and MCP (`r`). Register them with `register_footer_shortcuts(source="schedules", shortcuts=...)` so footer hints survive recompose.

- [ ] **Step 3: Run tests**

Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add tldw_chatbook/UI/Screens/scheduling/schedules_workbench.py Tests/UI/test_schedules_workbench.py
git commit -m "feat(scheduling): implement SchedulesWorkbench shell"
```

### Task 4.4: Implement task list pane

**Files:**
- Modify: `tldw_chatbook/UI/Screens/scheduling/schedules_workbench.py`
- Modify: `Tests/UI/test_schedules_workbench.py`

- [ ] **Step 1: Add DataTable and populate from service**

```python
from textual.widgets import DataTable

# in compose_content
self.task_table = DataTable(id="scheduling-task-table")
yield self.task_table
```

- [ ] **Step 2: Implement `load_tasks()`**

Use the injected app instance, not `self.app`, so tests can mock the dependency:

```python
service = getattr(self.app_instance, "scheduling_service", None)
if service is None:
    return
```

Call `service.list_tasks()` and populate table.

- [ ] **Step 3: Test selection updates detail**

```python
async def test_select_task_updates_detail():
    # setup mock service with one task
    async with WorkbenchTestApp().run_test() as pilot:
        await pilot.app.push_screen(SchedulesWorkbench(app_instance=pilot.app))
        table = pilot.app.screen.query_one("#scheduling-task-table", DataTable)
        table.cursor_coordinate = (0, 0)
        assert "Test" in pilot.app.screen.query_one("#scheduling-detail-pane").renderable
```

- [ ] **Step 4: Run tests**

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git commit -m "feat(scheduling): implement task list pane"
```

### Task 4.5: Implement detail and inspector panes

**Files:**
- Create: `tldw_chatbook/UI/Screens/scheduling/task_detail.py`
- Modify: `tldw_chatbook/UI/Screens/scheduling/schedules_workbench.py`

- [ ] **Step 1: Implement TaskDetail widget**

Shows title, type, status, schedule, next run, lifecycle actions, Console follow.

- [ ] **Step 2: Implement Inspector widget**

Shows status summary, sync status, conflict card.

- [ ] **Step 3: Wire into workbench**

Replace placeholder `Static`s with widgets.

- [ ] **Step 4: Run tests**

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/UI/Screens/scheduling/task_detail.py tldw_chatbook/UI/Screens/scheduling/schedules_workbench.py
git commit -m "feat(scheduling): implement detail and inspector panes"
```

### Task 4.6: Preserve Console-follow seam

**Files:**
- Modify: `tldw_chatbook/UI/Screens/scheduling/schedules_workbench.py`
- Modify: `Tests/UI/test_schedules_workbench.py`
- Modify: `Tests/UI/test_destination_shells.py`

- [ ] **Step 1: Add `#schedules-follow-in-console` button**

- [ ] **Step 2: Implement `follow_latest_schedule_run_in_console`**

Copy the body of `UI/Screens/schedules_screen.py:follow_latest_schedule_run_in_console` (currently lines 389–422), preserving the `#schedules-follow-in-console` selector and the fallback paths via `open_active_home_item_in_console` / `open_console_for_live_work`.

- [ ] **Step 3: Add regression test**

```python
async def test_console_follow_selector_exists():
    app = WorkbenchTestApp()
    async with app.run_test() as pilot:
        await pilot.app.push_screen(SchedulesWorkbench(app_instance=pilot.app))
        assert pilot.app.screen.query_one("#schedules-follow-in-console") is not None
```

- [ ] **Step 4: Update destination-shell tests**

In `Tests/UI/test_destination_shells.py`, update `SCREEN_BY_ROUTE["schedules"]` to import and reference `SchedulesWorkbench` from `tldw_chatbook.UI.Screens.scheduling.schedules_workbench`. Run the destination-shell suite.

```bash
pytest Tests/UI/test_destination_shells.py -v
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/UI/Screens/scheduling/schedules_workbench.py Tests/UI/test_schedules_workbench.py Tests/UI/test_destination_shells.py
git commit -m "feat(scheduling): preserve Console-follow seam and update destination tests"
```

### Task 4.7: Update screen registry

**Files:**
- Modify: `tldw_chatbook/UI/Navigation/screen_registry.py`

- [ ] **Step 1: Update route**

```python
"schedules": ScreenRoute(
    screen_name="schedules",
    canonical_tab="schedules",
    module_path="tldw_chatbook.UI.Screens.scheduling.schedules_workbench",
    class_name="SchedulesWorkbench",
),
```

- [ ] **Step 2: Run navigation test**

```bash
pytest Tests/UI/test_schedules_workbench.py -v
```

Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add tldw_chatbook/UI/Navigation/screen_registry.py
git commit -m "feat(scheduling): route schedules destination to new workbench"
```

### Task 4.8: Screenshot QA

- [ ] **Step 1: Capture baseline screenshot**

Run the app, navigate to Schedules, capture baseline.

- [ ] **Step 2: Capture final screenshot**

After all Phase 4 work, capture final screenshot.

- [ ] **Step 3: Record evidence**

Save screenshots and notes to `Docs/superpowers/qa/product-maturity/screen-qa/schedules/`.

- [ ] **Step 4: Commit**

```bash
git add Docs/superpowers/qa/product-maturity/screen-qa/schedules/
git commit -m "docs(scheduling): add schedules workbench screenshot QA evidence"
```

---

## Phase 5: Watchlist migration

### Task 5.1: Write migration ADR

**Files:**
- Create: `backlog/decisions/NNN-scheduling-watchlist-migration.md`

- [ ] **Step 1: Document mapping and rollback**

Include the column mapping table from the design doc, feature flag behavior, dual-run plan, and rollback steps.

- [ ] **Step 2: Commit**

```bash
git add backlog/decisions/NNN-scheduling-watchlist-migration.md
git commit -m "docs: ADR for watchlist scheduler migration"
```

### Task 5.2: Implement WatchlistCheckHandler

**Files:**
- Create: `tldw_chatbook/Scheduling/scheduler/handlers/watchlist_check_handler.py`
- Test: `Tests/Scheduling/test_watchlist_check_handler.py`

- [ ] **Step 1: Write handler test**

```python
@pytest.mark.asyncio
async def test_watchlist_handler_reuses_feed_monitor():
    monitor = AsyncMock()
    monitor.check_feed.return_value = [{"title": "New"}]
    handler = WatchlistCheckHandler(feed_monitor=monitor)
    await handler.handle({"id": "1", "type": "rss", "url": "http://example.com/feed"})
    monitor.check_feed.assert_awaited_once()
```

Run: FAIL

- [ ] **Step 2: Implement handler**

Route by `type` to existing `FeedMonitor` or `URLMonitor`. Record result in DB.

- [ ] **Step 3: Run tests**

Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add tldw_chatbook/Scheduling/scheduler/handlers/watchlist_check_handler.py Tests/Scheduling/test_watchlist_check_handler.py
git commit -m "feat(scheduling): implement WatchlistCheckHandler"
```

### Task 5.3: Feature flag and dual-run

**Files:**
- Modify: `tldw_chatbook/config.py`
- Modify: `tldw_chatbook/app.py`

- [ ] **Step 1: Add feature flag**

```toml
[scheduling]
watchlist_migration_enabled = false
```

- [ ] **Step 2: Conditionally route watchlist checks**

In `app.py`, if `watchlist_migration_enabled`, register `WatchlistCheckHandler`; otherwise keep old `Subscriptions/scheduler.py` running.

- [ ] **Step 3: Run subscription regression tests**

```bash
pytest Tests/Subscriptions/ -v
```

Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add tldw_chatbook/config.py tldw_chatbook/app.py
git commit -m "feat(scheduling): add watchlist migration feature flag"
```

### Task 5.4: Remove old scheduler

- [ ] **Step 1: Verify feature flag stable**

Run both schedulers side-by-side in tests for a period; compare outputs.

- [ ] **Step 2: Delete old scheduler files**

Remove `Subscriptions/scheduler.py` and `Subscriptions/textual_scheduler_worker.py` after migrating tests.

- [ ] **Step 3: Update imports**

Fix any remaining imports.

- [ ] **Step 4: Commit**

```bash
git rm tldw_chatbook/Subscriptions/scheduler.py tldw_chatbook/Subscriptions/textual_scheduler_worker.py
git commit -m "feat(scheduling): remove legacy subscription scheduler"
```

---

## Error-handling test coverage

The spec's Error Handling table covers several critical scenarios that must be explicitly tested before the epic is complete. Add the following tests during the relevant phases:

- `Tests/Scheduling/test_scheduling_service.py`: offline write → online push; auth failure pauses sync; 404/422/409 responses convert to conflicts; 5xx/timeout retries with exponential backoff capped at `sync_retry_max_delay_seconds`; owner-change filters visible records and queues per-owner pending mutations.
- `Tests/Scheduling/test_sync_engine.py`: conflict retry exhaustion offers "Force local" / "Archive local"; server-wins rule preserves server state.
- `Tests/Scheduling/test_scheduler_loop.py`: loop crash is caught and restarted; app restart with due reminders applies catch-up limits.
- `Tests/Scheduling/test_reminder_handler.py`: handler failure marks run failed and preserves next schedule.
- `Tests/Scheduling/test_models.py` or `Tests/UI/test_reminder_form.py`: automation preview invalid shows validation errors inline and blocks create.
- `Tests/Scheduling/test_scheduled_tasks_db.py`: local DB corruption raises a blocking error rather than silently falling back to in-memory storage.

---

## Final integration and verification

### Task 6.1: Run full scheduling test suite

```bash
pytest Tests/Scheduling/ Tests/UI/test_schedules_workbench.py Tests/UI/test_reminder_form.py -v
```

Expected: PASS

### Task 6.2: Run project lint/type checks

```bash
ruff check tldw_chatbook/Scheduling tldw_chatbook/UI/Screens/scheduling
mypy tldw_chatbook/Scheduling
```

Expected: no new errors

### Task 6.3: Update backlog task

Mark the Backlog.md task created in the prerequisites as Done. Also update `task-14.7` (Screen QA: Schedules) if screenshot QA was re-run. Add implementation notes linking to the spec and plan.

---

## Notes for implementers

- Use the existing `BaseAppScreen` pattern; don't compose custom chrome unless necessary.
- Always parameterize SQL queries; never interpolate.
- Prefer `asyncio.to_thread` for `ScheduledTasksDB` calls inside the scheduler worker.
- Keep the UI reactive: use `recompose=True` sparingly; prefer targeted `refresh()`.
- `BaseDB` does not store a persistent `_connection`; always use `_get_connection()` with a context manager.
- When building UI tests, construct a minimal Textual `App`, push the `BaseAppScreen` subclass with `app_instance=pilot.app`, and mock required app attributes on the test app.
- Refer to `Docs/superpowers/specs/2026-07-18-scheduling-module-screen-design.md` for full design context.
