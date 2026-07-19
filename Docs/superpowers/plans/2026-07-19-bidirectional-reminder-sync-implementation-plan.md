# TASK-299.2 Bidirectional Reminder Sync Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement bidirectional reminder sync between tldw_chatbook's local `ScheduledTasksDB` and `tldw_server`, with workbench-visible sync status, conflicts, and owner switching.

**Architecture:** Harden `SchedulingServerClient` with typed exceptions and retry boundaries; refactor `SyncEngine` into a network-then-single-transaction flow; add `SyncCompleted`/`SyncFailed` events; build a sync status widget, conflicts tab, and owner switcher into `SchedulesWorkbench`.

**Tech Stack:** Python 3.11+, Textual 3.3+, SQLite, pytest, httpx (via `TLDWAPIClient`), loguru.

---

## File Structure

### Existing files to modify

- `tldw_chatbook/Scheduling/services/server_client.py` — add exception hierarchy, config, retry wrapper, `set_notifications_service`, idempotency-key stripping.
- `tldw_chatbook/Scheduling/services/sync_engine.py` — accept explicit `owner_id`, network-then-transaction sync, append/cap `sync_errors`, preserve pending mutations in conflicts, handle server-deletion cleanup.
- `tldw_chatbook/Scheduling/services/scheduling_service.py` — pass explicit owner to `SyncEngine`, always expose a `SchedulingServerClient`.
- `tldw_chatbook/Scheduling/events.py` — add `SyncCompleted`/`SyncFailed`.
- `tldw_chatbook/Scheduling/db/scheduled_tasks_db.py` — add connection-aware bulk helpers for Phase 2.
- `tldw_chatbook/UI/Screens/scheduling/schedules_workbench.py` — add sync status bar, `TabbedContent` with Queue/Conflicts tabs, `ctrl+s` sync worker, owner switcher.
- `tldw_chatbook/app.py` — always instantiate `SchedulingServerClient` even when no server is configured.

### Existing test files to modify

- `Tests/Scheduling/test_server_client.py` — retry/idempotency/exception tests.
- `Tests/Scheduling/test_sync_engine.py` — transaction/error/conflict tests.
- `Tests/Scheduling/test_scheduling_service.py` — owner/sync-now tests.
- `Tests/UI/test_schedules_workbench.py` — sync worker/status/conflicts UI tests.

### New files to create

- `tldw_chatbook/UI/Screens/scheduling/sync_status_widget.py` — `SyncStatusWidget` static bar.
- `tldw_chatbook/UI/Screens/scheduling/conflicts_tab.py` — `ConflictsTab` DataTable + actions.

---

## Task 1: Add exception classes and `ServerClientConfig`

**Files:**
- Modify: `tldw_chatbook/Scheduling/services/server_client.py`
- Test: `Tests/Scheduling/test_server_client.py`

- [ ] **Step 1: Write the failing test**

Add to `Tests/Scheduling/test_server_client.py`:

```python
from tldw_chatbook.Scheduling.services.server_client import (
    ServerClientConfig,
    ServerClientError,
    ServerClientNotFoundError,
    ServerClientServerError,
    ServerClientTimeoutError,
    ServerClientValidationError,
)


def test_exception_hierarchy():
    assert issubclass(ServerClientNotFoundError, ServerClientError)
    assert issubclass(ServerClientServerError, ServerClientError)
    assert issubclass(ServerClientTimeoutError, ServerClientError)
    assert issubclass(ServerClientValidationError, ServerClientError)
    assert issubclass(ServerUnavailableError, ServerClientError)


def test_server_client_config_defaults():
    cfg = ServerClientConfig()
    assert cfg.timeout == 10.0
    assert cfg.max_retries == 3
    assert cfg.retry_delay == 1.0
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest Tests/Scheduling/test_server_client.py::test_exception_hierarchy Tests/Scheduling/test_server_client.py::test_server_client_config_defaults -v
```

Expected: `ImportError` or `NameError` for missing classes.

- [ ] **Step 3: Write minimal implementation**

Add to the top of `tldw_chatbook/Scheduling/services/server_client.py`:

```python
from dataclasses import dataclass


class ServerClientError(Exception):
    """Base class for all server-client failures."""


class ServerUnavailableError(ServerClientError):
    """Raised when the server client is invoked while no server is connected."""


class ServerClientTimeoutError(ServerClientError):
    """Request to the server timed out."""


class ServerClientNotFoundError(ServerClientError):
    """Server returned 404; the task was deleted server-side."""


class ServerClientValidationError(ServerClientError):
    """Server returned 4xx other than 404, or a local policy denied the action."""


class ServerClientServerError(ServerClientError):
    """Server returned 5xx."""


@dataclass(slots=True)
class ServerClientConfig:
    timeout: float = 10.0
    max_retries: int = 3
    retry_delay: float = 1.0
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest Tests/Scheduling/test_server_client.py::test_exception_hierarchy Tests/Scheduling/test_server_client.py::test_server_client_config_defaults -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Scheduling/services/server_client.py Tests/Scheduling/test_server_client.py
git commit -m "feat(scheduling): add typed server-client exceptions and config"
```

---

## Task 2: Harden `SchedulingServerClient` with retry, typed errors, and idempotency-key stripping

**Files:**
- Modify: `tldw_chatbook/Scheduling/services/server_client.py`
- Test: `Tests/Scheduling/test_server_client.py`

- [ ] **Step 1: Write the failing tests**

Add to `Tests/Scheduling/test_server_client.py`:

```python
import asyncio
from unittest.mock import AsyncMock

import pytest

from tldw_chatbook.runtime_policy.types import PolicyDeniedError


@pytest.mark.asyncio
async def test_create_reminder_not_retried_on_timeout():
    service = AsyncMock()
    service.create_reminder.side_effect = asyncio.TimeoutError
    client = SchedulingServerClient(notifications_service=service)

    with pytest.raises(ServerClientTimeoutError):
        await client.create_reminder(title="T", schedule_kind="one_time")

    assert service.create_reminder.call_count == 1


@pytest.mark.asyncio
async def test_update_reminder_retries_then_raises_server_error():
    service = AsyncMock()
    service.update_reminder.side_effect = ServerClientServerError("boom")
    client = SchedulingServerClient(
        notifications_service=service,
        config=ServerClientConfig(max_retries=2, retry_delay=0.0),
    )

    with pytest.raises(ServerClientServerError):
        await client.update_reminder("srv-1", title="T")

    assert service.update_reminder.call_count == 3  # initial + 2 retries


@pytest.mark.asyncio
async def test_update_reminder_not_retried_on_validation_error():
    service = AsyncMock()
    service.update_reminder.side_effect = ServerClientValidationError("bad")
    client = SchedulingServerClient(
        notifications_service=service,
        config=ServerClientConfig(max_retries=2, retry_delay=0.0),
    )

    with pytest.raises(ServerClientValidationError):
        await client.update_reminder("srv-1", title="T")

    assert service.update_reminder.call_count == 1


@pytest.mark.asyncio
async def test_create_reminder_strips_idempotency_key():
    service = AsyncMock()
    service.create_reminder.return_value = {"id": "srv-1", "title": "T"}
    client = SchedulingServerClient(notifications_service=service)

    await client.create_reminder(
        title="T", schedule_kind="one_time", idempotency_key="abc-123"
    )

    _, kwargs = service.create_reminder.call_args
    assert "idempotency_key" not in kwargs


@pytest.mark.asyncio
async def test_policy_denied_maps_to_validation_error():
    service = AsyncMock()
    service.create_reminder.side_effect = PolicyDeniedError(
        action_id="x", reason_code="denied", user_message="no"
    )
    client = SchedulingServerClient(notifications_service=service)

    with pytest.raises(ServerClientValidationError):
        await client.create_reminder(title="T", schedule_kind="one_time")


@pytest.mark.asyncio
async def test_set_notifications_service_replaces_service():
    old_service = AsyncMock()
    new_service = AsyncMock()
    new_service.list_reminders.return_value = {"items": []}
    client = SchedulingServerClient(notifications_service=old_service)
    client.set_notifications_service(new_service)

    await client.list_reminders()
    assert new_service.list_reminders.called
    assert not old_service.list_reminders.called
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest Tests/Scheduling/test_server_client.py -v
```

Expected: several FAILs for `_call_with_retry`, `set_notifications_service`, etc.

- [ ] **Step 3: Write the implementation**

Replace the body of `SchedulingServerClient` in `tldw_chatbook/Scheduling/services/server_client.py` with:

```python
import asyncio
import random
from typing import Any


class SchedulingServerClient:
    """Async client that delegates scheduling operations to a notifications service.

    The client is a thin wrapper around an injected notifications service. All
    methods raise :class:`ServerUnavailableError` when no service has been
    configured, so callers can distinguish "server missing" from actual request
    failures.
    """

    def __init__(
        self,
        notifications_service: Any | None = None,
        config: ServerClientConfig | None = None,
    ) -> None:
        self.notifications_service = notifications_service
        self.config = config or ServerClientConfig()

    def set_notifications_service(self, notifications_service: Any | None) -> None:
        """Inject or refresh the underlying notifications service."""
        self.notifications_service = notifications_service

    def _is_available(self) -> bool:
        return self.notifications_service is not None

    @staticmethod
    def _strip_local_only_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
        """Remove kwargs that the server service does not accept."""
        return {k: v for k, v in kwargs.items() if k != "idempotency_key"}

    async def _call_with_retry(
        self,
        method_name: str,
        *args: Any,
        retry: bool = True,
        **kwargs: Any,
    ) -> dict[str, Any]:
        service = self.notifications_service
        if service is None:
            raise ServerUnavailableError("server not available")

        kwargs = self._strip_local_only_kwargs(kwargs)
        method = getattr(service, method_name)
        last_error: Exception | None = None

        attempts = self.config.max_retries + 1 if retry else 1
        for attempt in range(attempts):
            try:
                return await method(*args, **kwargs)
            except PolicyDeniedError as exc:
                raise ServerClientValidationError(str(exc)) from exc
            except asyncio.TimeoutError as exc:
                last_error = exc
                error_cls = ServerClientTimeoutError
            except ServerClientError:
                raise
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                error_cls = ServerClientError
                status = getattr(exc, "status_code", None)
                if status == 404:
                    raise ServerClientNotFoundError(str(exc)) from exc
                if status is not None and 400 <= status < 500:
                    raise ServerClientValidationError(str(exc)) from exc
                if status is not None and 500 <= status < 600:
                    error_cls = ServerClientServerError

            if not retry or attempt == attempts - 1:
                raise error_cls(str(last_error)) from last_error

            delay = self.config.retry_delay * (2**attempt)
            delay += random.uniform(0, delay * 0.1)
            await asyncio.sleep(delay)

        raise ServerClientError("unexpected end of retry loop")

    async def create_reminder(self, **payload: Any) -> dict[str, Any]:
        return await self._call_with_retry("create_reminder", retry=False, **payload)

    async def update_reminder(self, task_id: str, **payload: Any) -> dict[str, Any]:
        return await self._call_with_retry(
            "update_reminder", task_id, **payload
        )

    async def delete_reminder(self, task_id: str) -> dict[str, Any]:
        return await self._call_with_retry("delete_reminder", task_id)

    async def list_reminders(self) -> dict[str, Any]:
        return await self._call_with_retry("list_reminders")

    async def get_reminder(self, task_id: str) -> dict[str, Any]:
        return await self._call_with_retry("get_reminder", task_id)
```

**Note:** `PolicyDeniedError` import may already be available via `runtime_policy.types`. If not, add the import at module scope.

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest Tests/Scheduling/test_server_client.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Scheduling/services/server_client.py Tests/Scheduling/test_server_client.py
git commit -m "feat(scheduling): harden server client with retry, typed errors, and idempotency stripping"
```

---

## Task 3: Add `SyncCompleted`/`SyncFailed` events

**Files:**
- Modify: `tldw_chatbook/Scheduling/events.py`
- Test: `Tests/UI/test_schedules_workbench.py` (or create a new event test)

- [ ] **Step 1: Write the failing test**

Add to `Tests/Scheduling/test_events.py` if it exists; otherwise add to `Tests/UI/test_schedules_workbench.py`:

```python
from tldw_chatbook.Scheduling.events import SyncCompleted, SyncFailed


def test_sync_completed_event():
    msg = SyncCompleted("server:1", conflict_count=2)
    assert msg.owner_id == "server:1"
    assert msg.conflict_count == 2


def test_sync_failed_event():
    msg = SyncFailed("server:1", error="timeout")
    assert msg.owner_id == "server:1"
    assert msg.error == "timeout"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest Tests/UI/test_schedules_workbench.py::test_sync_completed_event Tests/UI/test_schedules_workbench.py::test_sync_failed_event -v
```

Expected: `ImportError`.

- [ ] **Step 3: Write minimal implementation**

Add to `tldw_chatbook/Scheduling/events.py`:

```python
class SyncCompleted(Message):
    """Posted when a sync attempt completes."""

    def __init__(self, owner_id: str, conflict_count: int) -> None:
        super().__init__()
        self.owner_id = owner_id
        self.conflict_count = conflict_count


class SyncFailed(Message):
    """Posted when a sync attempt fails."""

    def __init__(self, owner_id: str, error: str) -> None:
        super().__init__()
        self.owner_id = owner_id
        self.error = error
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest Tests/UI/test_schedules_workbench.py::test_sync_completed_event Tests/UI/test_schedules_workbench.py::test_sync_failed_event -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Scheduling/events.py Tests/UI/test_schedules_workbench.py
git commit -m "feat(scheduling): add SyncCompleted and SyncFailed events"
```

---

## Task 4: Update `SchedulingService` to accept explicit owner and always expose a server client

**Files:**
- Modify: `tldw_chatbook/Scheduling/services/scheduling_service.py`
- Test: `Tests/Scheduling/test_scheduling_service.py`

- [ ] **Step 1: Write the failing test**

Add to `Tests/Scheduling/test_scheduling_service.py`:

```python
from unittest.mock import AsyncMock, MagicMock

from tldw_chatbook.Scheduling.services import SchedulingServerClient


@pytest.mark.asyncio
async def test_sync_now_passes_owner_id_to_engine(db):
    engine = MagicMock()
    engine.sync_now = AsyncMock()
    svc = SchedulingService(db=db, runtime_source="local")
    svc.sync_engine = engine

    await svc.sync_now("server:example.com")

    engine.sync_now.assert_awaited_once_with("server:example.com")


def test_server_client_is_always_present(db):
    svc = SchedulingService(db=db, runtime_source="local", server_client=None)
    assert isinstance(svc.server_client, SchedulingServerClient)
    assert svc.server_client.notifications_service is None
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest Tests/Scheduling/test_scheduling_service.py::test_sync_now_passes_owner_id_to_engine Tests/Scheduling/test_scheduling_service.py::test_server_client_is_always_present -v
```

Expected: FAIL.

- [ ] **Step 3: Write minimal implementation**

Modify `SchedulingService.__init__` in `tldw_chatbook/Scheduling/services/scheduling_service.py`:

```python
self.db = db
self.server_client = server_client or SchedulingServerClient()
self.runtime_source = runtime_source
self.owner_id = runtime_source
self.watchlist_projection = watchlist_projection
self.sync_engine = SyncEngine(db, self.server_client, self.owner_id)
```

Modify `SchedulingService.sync_now`:

```python
async def sync_now(self, owner_id: str | None = None) -> None:
    """Trigger a full sync for the given owner (defaults to current owner)."""
    target_owner = owner_id if owner_id is not None else self.owner_id
    await self.sync_engine.sync_now(target_owner)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest Tests/Scheduling/test_scheduling_service.py::test_sync_now_passes_owner_id_to_engine Tests/Scheduling/test_scheduling_service.py::test_server_client_is_always_present -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Scheduling/services/scheduling_service.py Tests/Scheduling/test_scheduling_service.py
git commit -m "feat(scheduling): accept explicit owner in sync_now and always expose server client"
```

---

## Task 5: Add connection-aware bulk helpers to `ScheduledTasksDB`

**Files:**
- Modify: `tldw_chatbook/Scheduling/db/scheduled_tasks_db.py`
- Test: `Tests/Scheduling/test_scheduled_tasks_db.py`

- [ ] **Step 1: Write the failing tests**

Add to `Tests/Scheduling/test_scheduled_tasks_db.py`:

```python
@pytest.mark.asyncio
async def test_bulk_apply_pulled_items_and_purge_mutations(tmp_path):
    db = ScheduledTasksDB(tmp_path / "db.db")
    owner_id = "server:1"

    with db.transaction() as conn:
        db._apply_pulled_reminders(conn, owner_id, [
            {"id": "srv-1", "title": "One", "schedule_kind": "one_time"},
        ])
        db._purge_pending_mutations(conn, owner_id, ["mutation-uuid"])

    rows = db.list_reminder_tasks(owner_id=owner_id)
    assert len(rows) == 1
    assert rows[0]["server_id"] == "srv-1"


def test_record_sync_error_appends_and_caps(tmp_path):
    db = ScheduledTasksDB(tmp_path / "db.db")
    owner_id = "server:1"

    for i in range(12):
        db._append_sync_error(owner_id, f"error {i}")

    state = db.get_sync_state(owner_id)
    assert len(state["sync_errors"]) == 10
    assert state["sync_errors"][-1]["message"] == "error 11"
    assert state["sync_errors"][0]["message"] == "error 2"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest Tests/Scheduling/test_scheduled_tasks_db.py::test_bulk_apply_pulled_items_and_purge_mutations Tests/Scheduling/test_scheduled_tasks_db.py::test_record_sync_error_appends_and_caps -v
```

Expected: `AttributeError` for `_apply_pulled_reminders`, `_purge_pending_mutations`, `_append_sync_error`.

- [ ] **Step 3: Write minimal implementation**

Add to `tldw_chatbook/Scheduling/db/scheduled_tasks_db.py` inside `ScheduledTasksDB`:

```python
def _apply_pulled_reminders(
    self,
    conn: sqlite3.Connection,
    owner_id: str,
    server_items: list[dict[str, Any]],
) -> None:
    """Insert or update reminder rows from a pulled server list.

    Must run inside an existing transaction (``conn`` is the open connection).
    """
    for item in server_items:
        server_id = item.get("id")
        if not server_id:
            continue

        existing = self._get_reminder_task_by_server_id_conn(
            conn, owner_id, server_id
        )
        fields = {
            key: item[key]
            for key in self._REMINDER_TASK_COLUMNS
            if key in item and key not in {"id", "server_id", "owner_id"}
        }
        fields.setdefault("title", "Untitled reminder")
        if "schedule_kind" not in fields:
            fields["schedule_kind"] = "one_time"
        fields["updated_at"] = self._to_utc_iso(datetime.now(timezone.utc))

        if existing:
            local_id = existing["id"]
            self._update_reminder_task_conn(conn, local_id, **fields)
        else:
            local_id = self._create_reminder_task_conn(
                conn, owner_id, **fields
            )

        self._set_sync_mapping_conn(
            conn, local_id, server_id, "reminder_task", owner_id
        )


def _purge_pending_mutations(
    self,
    conn: sqlite3.Connection,
    owner_id: str,
    mutation_ids: list[str],
) -> None:
    """Delete pending mutations by their row ids inside an existing transaction."""
    if not mutation_ids:
        return
    placeholders = ", ".join("?" * len(mutation_ids))
    conn.execute(
        f"DELETE FROM pending_mutations WHERE id IN ({placeholders})",
        mutation_ids,
    )


def _append_sync_error(self, owner_id: str, message: str) -> None:
    """Append a sync error, capping the history at 10 entries."""
    state = self.get_sync_state(owner_id) or {}
    errors = list(state.get("sync_errors") or [])
    errors.append({"message": message, "timestamp": datetime.now(timezone.utc).isoformat()})
    errors = errors[-10:]
    self.update_sync_state(owner_id, sync_errors=errors)
```

Also add the connection-aware private helpers `_get_reminder_task_by_server_id_conn`, `_update_reminder_task_conn`, `_create_reminder_task_conn`, and `_set_sync_mapping_conn` by refactoring the existing public methods to use them. For example:

```python
def _get_reminder_task_by_server_id_conn(
    self, conn: sqlite3.Connection, owner_id: str, server_id: str
) -> Optional[dict[str, Any]]:
    cursor = conn.execute(
        "SELECT * FROM reminder_tasks WHERE owner_id = ? AND server_id = ?",
        (owner_id, server_id),
    )
    return self._row_to_dict(cursor.fetchone())


def _create_reminder_task_conn(
    self, conn: sqlite3.Connection, owner_id: str, title: str, **kwargs: Any
) -> str:
    self._validate_kwargs(kwargs, self._REMINDER_TASK_COLUMNS, "reminder task")
    task_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc)
    fields: dict[str, Any] = {
        "id": task_id,
        "owner_id": owner_id,
        "title": title,
        "created_at": self._to_utc_iso(now),
        "updated_at": self._to_utc_iso(now),
        "enabled": 1,
        "sync_version": 0,
    }
    for key, value in kwargs.items():
        if key == "enabled":
            fields[key] = 1 if value else 0
        elif key in self._DATETIME_FIELDS:
            fields[key] = self._to_utc_iso(value)
        else:
            fields[key] = value
    self._validate_sql_identifiers(list(fields.keys()))
    columns = ", ".join(fields.keys())
    placeholders = ", ".join(["?"] * len(fields))
    conn.execute(
        f"INSERT INTO reminder_tasks ({columns}) VALUES ({placeholders})",
        list(fields.values()),
    )
    return task_id


def _update_reminder_task_conn(
    self, conn: sqlite3.Connection, task_id: str, **kwargs: Any
) -> bool:
    if not kwargs:
        return False
    self._validate_kwargs(kwargs, self._REMINDER_TASK_COLUMNS, "reminder task")
    updates: list[str] = []
    params: list[Any] = []
    for key, value in kwargs.items():
        if key == "enabled":
            updates.append("enabled = ?")
            params.append(1 if value else 0)
        elif key in self._DATETIME_FIELDS:
            updates.append(f"{key} = ?")
            params.append(self._to_utc_iso(value))
        else:
            updates.append(f"{key} = ?")
            params.append(value)
    if not updates:
        return False
    self._validate_sql_identifiers([key.split(" ", 1)[0] for key in updates])
    updates.append("updated_at = ?")
    params.append(self._to_utc_iso(datetime.now(timezone.utc)))
    params.append(task_id)
    cursor = conn.execute(
        f"UPDATE reminder_tasks SET {', '.join(updates)} WHERE id = ?",
        params,
    )
    return cursor.rowcount > 0


def _set_sync_mapping_conn(
    self,
    conn: sqlite3.Connection,
    local_id: str,
    server_id: str,
    primitive: str,
    owner_id: str,
) -> None:
    now = datetime.now(timezone.utc)
    conn.execute(
        """
        INSERT OR REPLACE INTO sync_mapping
        (local_id, server_id, primitive, owner_id, created_at)
        VALUES (?, ?, ?, ?, ?)
        """,
        (local_id, server_id, primitive, owner_id, self._to_utc_iso(now)),
    )
```

Then refactor the public methods to call these helpers with `self.transaction()` so existing tests continue to pass.

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest Tests/Scheduling/test_scheduled_tasks_db.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Scheduling/db/scheduled_tasks_db.py Tests/Scheduling/test_scheduled_tasks_db.py
git commit -m "feat(scheduling): add connection-aware bulk helpers for sync Phase 2"
```

---

## Task 6: Refactor `SyncEngine` for explicit owner, network-then-transaction, and conflict preservation

**Files:**
- Modify: `tldw_chatbook/Scheduling/services/sync_engine.py`
- Test: `Tests/Scheduling/test_sync_engine.py`

- [ ] **Step 1: Write the failing tests**

Add to `Tests/Scheduling/test_sync_engine.py`:

```python
from unittest.mock import AsyncMock

from tldw_chatbook.Scheduling.services.server_client import ServerClientNotFoundError


@pytest.mark.asyncio
async def test_sync_now_uses_passed_owner_not_self_owner(tmp_path):
    db = ScheduledTasksDB(tmp_path / "db.db")
    server_client = AsyncMock()
    server_client.list_reminders.return_value = {"items": []}
    engine = SyncEngine(db, server_client, owner_id="local")
    await engine.sync_now("server:1")
    server_client.list_reminders.assert_awaited_once()


@pytest.mark.asyncio
async def test_record_sync_error_appends_and_caps(tmp_path):
    db = ScheduledTasksDB(tmp_path / "db.db")
    engine = SyncEngine(db, None, owner_id="server:1")
    for i in range(12):
        engine._record_sync_error(f"err {i}")
    state = db.get_sync_state("server:1")
    assert len(state["sync_errors"]) == 10


@pytest.mark.asyncio
async def test_push_404_records_conflict_and_removes_mutation(tmp_path):
    db = ScheduledTasksDB(tmp_path / "db.db")
    local_id = db.create_reminder_task(
        owner_id="server:1",
        server_id="srv-1",
        title="T",
        schedule_kind="one_time",
    )
    db.set_sync_mapping(local_id, "srv-1", "reminder_task", "server:1")
    db.record_pending_mutation(
        local_id,
        "reminder_task",
        "server:1",
        {"action": "update", "fields": {"title": "Updated"}, "idempotency_key": "ik"},
    )

    server_client = AsyncMock()
    server_client.update_reminder.side_effect = ServerClientNotFoundError("gone")
    engine = SyncEngine(db, server_client, owner_id="server:1")
    await engine.sync_now()

    conflicts = db.get_conflicts("server:1", primitive="reminder_task")
    assert len(conflicts) == 1
    pending = db.get_pending_mutations("server:1")
    assert len(pending) == 0


@pytest.mark.asyncio
async def test_use_local_on_server_deletion_clears_server_id_and_requeues_create(tmp_path):
    db = ScheduledTasksDB(tmp_path / "db.db")
    local_id = db.create_reminder_task(
        owner_id="server:1",
        server_id="srv-1",
        title="T",
        schedule_kind="one_time",
    )
    db.set_sync_mapping(local_id, "srv-1", "reminder_task", "server:1")
    conflict_id = db.record_conflict(
        local_id, "reminder_task", "server:1", server_state={}, local_state={"record": db.get_reminder_task(local_id)}
    )

    engine = SyncEngine(db, None, owner_id="server:1")
    engine.resolve_conflict(conflict_id, "local")

    row = db.get_reminder_task(local_id)
    assert row["server_id"] is None
    pending = db.get_pending_mutations("server:1")
    assert len(pending) == 1
    assert pending[0]["payload"]["action"] == "create"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest Tests/Scheduling/test_sync_engine.py::test_sync_now_uses_passed_owner_not_self_owner Tests/Scheduling/test_sync_engine.py::test_record_sync_error_appends_and_caps Tests/Scheduling/test_sync_engine.py::test_push_404_records_conflict_and_removes_mutation Tests/Scheduling/test_sync_engine.py::test_use_local_on_server_deletion_clears_server_id_and_requeues_create -v
```

Expected: FAIL.

- [ ] **Step 3: Write the implementation**

Rewrite `SyncEngine` in `tldw_chatbook/Scheduling/services/sync_engine.py`:

```python
class SyncEngine:
    """Pull, push, and reconcile scheduled-task state with tldw_server."""

    def __init__(
        self,
        db: ScheduledTasksDB,
        server_client: SchedulingServerClient | None,
        owner_id: str,
    ) -> None:
        self.db = db
        self.server_client = server_client
        self.owner_id = owner_id

    async def sync_now(self, owner_id: str | None = None) -> None:
        target_owner = owner_id if owner_id is not None else self.owner_id
        if self.server_client is None:
            return

        try:
            pulled_items, staged_outcomes, conflicts = await self._network_phase(
                target_owner
            )
        except ServerClientError as exc:
            self._record_sync_error(str(exc), target_owner)
            return
        except Exception as exc:  # noqa: BLE001
            logger.exception(f"Sync network phase failed for {target_owner}: {exc}")
            self._record_sync_error(str(exc), target_owner)
            return

        try:
            with self.db.transaction() as conn:
                self.db._apply_pulled_reminders(conn, target_owner, pulled_items)
                for conflict in conflicts:
                    self.db.record_conflict(
                        local_id=conflict["local_id"],
                        primitive=_REMINDER_PRIMITIVE,
                        owner_id=target_owner,
                        server_state=conflict["server_state"],
                        local_state=conflict["local_state"],
                    )
                for outcome in staged_outcomes:
                    local_id = outcome["local_id"]
                    server_id = outcome.get("server_id")
                    if server_id:
                        self.db._set_sync_mapping_conn(
                            conn, local_id, server_id, _REMINDER_PRIMITIVE, target_owner
                        )
                        self.db._update_reminder_task_conn(
                            conn, local_id, server_id=server_id
                        )
                mutation_ids = [o["mutation_id"] for o in staged_outcomes if o.get("mutation_id")]
                self.db._purge_pending_mutations(conn, target_owner, mutation_ids)
                self.db.update_sync_state(
                    target_owner,
                    last_pull_at=now_utc_iso(),
                    last_push_at=now_utc_iso() if staged_outcomes else None,
                )
        except Exception as exc:  # noqa: BLE001
            logger.exception(f"Sync transaction failed for {target_owner}: {exc}")
            self._record_sync_error(str(exc), target_owner)

    async def _network_phase(
        self, owner_id: str
    ) -> tuple[list[dict], list[dict], list[dict]]:
        pulled_items: list[dict] = []
        staged_outcomes: list[dict] = []
        conflicts: list[dict] = []

        response = await self.server_client.list_reminders()
        pulled_items = response.get("items", [])

        mutations = self.db.get_pending_mutations(owner_id, primitive=_REMINDER_PRIMITIVE)
        for mutation in mutations:
            outcome = await self._push_mutation(mutation, owner_id)
            if outcome is None:
                raise ServerClientError("push phase aborted")
            if outcome.get("conflict"):
                conflicts.append(outcome["conflict"])
            else:
                staged_outcomes.append(outcome)

        tombstones = self.db.get_tombstones(owner_id, primitive=_REMINDER_PRIMITIVE)
        for tombstone in tombstones:
            outcome = await self._push_tombstone(tombstone, owner_id)
            if outcome is None:
                raise ServerClientError("tombstone phase aborted")
            staged_outcomes.append(outcome)

        return pulled_items, staged_outcomes, conflicts

    async def _push_mutation(
        self, mutation: dict, owner_id: str
    ) -> dict[str, Any] | None:
        local_id = mutation["local_id"]
        payload = mutation.get("payload") or {}
        action = payload.get("action", "update")
        fields = payload.get("fields", {})

        try:
            if action == "create":
                response = await self.server_client.create_reminder(**fields)
                return {
                    "local_id": local_id,
                    "server_id": response.get("id"),
                    "mutation_id": mutation["id"],
                }
            if action == "update":
                server_id = self._server_id_for_local(local_id)
                if server_id is None:
                    return {"local_id": local_id, "mutation_id": mutation["id"]}
                response = await self.server_client.update_reminder(server_id, **fields)
                return {
                    "local_id": local_id,
                    "server_id": response.get("id", server_id),
                    "mutation_id": mutation["id"],
                }
            if action == "delete":
                server_id = self._server_id_for_local(local_id, from_mapping_only=True)
                if server_id is None:
                    return {"local_id": local_id, "mutation_id": mutation["id"]}
                await self.server_client.delete_reminder(server_id)
                return {
                    "local_id": local_id,
                    "mutation_id": mutation["id"],
                    "delete_local": True,
                }
            logger.warning(f"Unknown pending mutation action {action!r}")
            return {"local_id": local_id, "mutation_id": mutation["id"]}
        except ServerClientNotFoundError:
            local_row = self.db.get_reminder_task(local_id)
            return {
                "conflict": {
                    "local_id": local_id,
                    "server_state": {},
                    "local_state": {
                        "record": dict(local_row) if local_row else {},
                        "pending_mutation": payload,
                    },
                }
            }
        except ServerClientError as exc:
            self._record_sync_error(str(exc), owner_id)
            return None

    async def _push_tombstone(
        self, tombstone: dict, owner_id: str
    ) -> dict[str, Any] | None:
        local_id = tombstone["local_id"]
        server_id = self._server_id_for_local(local_id, from_mapping_only=True)
        if server_id is None:
            return {"local_id": local_id, "delete_tombstone": True}
        try:
            await self.server_client.delete_reminder(server_id)
            return {"local_id": local_id, "delete_tombstone": True}
        except ServerClientNotFoundError:
            return {"local_id": local_id, "delete_tombstone": True}
        except ServerClientError as exc:
            self._record_sync_error(str(exc), owner_id)
            return None

    def resolve_conflict(self, conflict_id: str, resolution: str = "server") -> bool:
        conflict = self.db.get_conflict_by_id(conflict_id)
        if conflict is None:
            return False

        local_id = conflict["local_id"]
        server_state = conflict.get("server_state") or {}
        local_state = conflict.get("local_state") or {}
        pending_mutation = (
            local_state.get("pending_mutation")
            if isinstance(local_state, dict)
            else None
        )

        if resolution == "server":
            if not server_state:
                self.db.delete_reminder_task(local_id)
                self.db.delete_sync_mapping(local_id, _REMINDER_PRIMITIVE, self.owner_id)
                self.db.delete_tombstone(local_id, _REMINDER_PRIMITIVE, self.owner_id)
            else:
                self.db.update_reminder_task(
                    local_id, **self._whitelist_reminder_fields(server_state)
                )
        elif resolution == "local":
            if not server_state and pending_mutation:
                self.db.update_reminder_task(local_id, server_id=None)
                self.db.delete_sync_mapping(local_id, _REMINDER_PRIMITIVE, self.owner_id)
                self.db.record_pending_mutation(
                    local_id,
                    _REMINDER_PRIMITIVE,
                    self.owner_id,
                    pending_mutation,
                )
            elif not server_state:
                row = self.db.get_reminder_task(local_id)
                self.db.update_reminder_task(local_id, server_id=None)
                self.db.delete_sync_mapping(local_id, _REMINDER_PRIMITIVE, self.owner_id)
                fields = {
                    key: row.get(key)
                    for key in self._REMINDER_MUTABLE_FIELDS
                    if row.get(key) is not None
                }
                self.db.record_pending_mutation(
                    local_id,
                    _REMINDER_PRIMITIVE,
                    self.owner_id,
                    {"action": "create", "fields": fields},
                )
            elif pending_mutation:
                self.db.record_pending_mutation(
                    local_id,
                    _REMINDER_PRIMITIVE,
                    self.owner_id,
                    pending_mutation,
                )
            else:
                fields = {
                    key: value
                    for key, value in (local_state.get("record") or local_state).items()
                    if key in self._REMINDER_MUTABLE_FIELDS
                }
                self.db.record_pending_mutation(
                    local_id,
                    _REMINDER_PRIMITIVE,
                    self.owner_id,
                    {"action": "update", "fields": fields},
                )
            self.db.increment_conflict_retry_count(conflict_id)

        self.db.resolve_conflict(conflict_id, resolution)
        return True

    def _record_sync_error(self, message: str, owner_id: str | None = None) -> None:
        target_owner = owner_id if owner_id is not None else self.owner_id
        self.db._append_sync_error(target_owner, message)
```

**Note:** The `_pull` and `_reconcile_record` methods from the old implementation are replaced by `_apply_pulled_reminders`. Keep `_find_local_row`, `_server_id_for_local`, `_whitelist_reminder_fields`, and `_REMINDER_MUTABLE_FIELDS` from the original.

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest Tests/Scheduling/test_sync_engine.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Scheduling/services/sync_engine.py Tests/Scheduling/test_sync_engine.py
git commit -m "feat(scheduling): refactor SyncEngine for explicit owner and network-then-transaction sync"
```

---

## Task 7: Build `SyncStatusWidget`

**Files:**
- Create: `tldw_chatbook/UI/Screens/scheduling/sync_status_widget.py`
- Modify: `tldw_chatbook/UI/Screens/scheduling/__init__.py` (if needed)
- Test: `Tests/UI/test_schedules_workbench.py`

- [ ] **Step 1: Write the failing test**

Add to `Tests/UI/test_schedules_workbench.py`:

```python
from tldw_chatbook.UI.Screens.scheduling.sync_status_widget import SyncStatusWidget


def test_sync_status_widget_renders_mode_and_timestamps():
    widget = SyncStatusWidget()
    widget.update_status(
        owner_id="server:example.com",
        last_pull_at="2026-07-19T10:00:00+00:00",
        last_push_at="2026-07-19T10:05:00+00:00",
        sync_errors=[],
        server_available=True,
    )

    text = widget.renderable.plain
    assert "Server" in text
    assert "Last pull" in text
    assert "Last push" in text
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest Tests/UI/test_schedules_workbench.py::test_sync_status_widget_renders_mode_and_timestamps -v
```

Expected: `ImportError`.

- [ ] **Step 3: Write minimal implementation**

Create `tldw_chatbook/UI/Screens/scheduling/sync_status_widget.py`:

```python
"""Sync status bar widget for the Schedules workbench."""

from __future__ import annotations

from textual.widgets import Button, Static
from textual.containers import Horizontal


class SyncStatusWidget(Horizontal):
    """Bar showing current owner, last sync timestamps, and latest error."""

    DEFAULT_CSS = """
    SyncStatusWidget {
        height: auto;
        padding: 1;
    }
    #scheduling-owner-select {
        width: 30;
    }
    #scheduling-last-pull, #scheduling-last-push {
        width: auto;
    }
    #scheduling-sync-error {
        width: 1fr;
        color: $error;
    }
    #scheduling-clear-error {
        width: auto;
    }
    """

    def compose(self):
        yield Static("Local", id="scheduling-owner-select")
        yield Static("Last pull: —", id="scheduling-last-pull")
        yield Static("Last push: —", id="scheduling-last-push")
        yield Static("", id="scheduling-sync-error")
        yield Button("Clear", id="scheduling-clear-error")

    def update_status(
        self,
        owner_id: str,
        last_pull_at: str | None,
        last_push_at: str | None,
        sync_errors: list[dict],
        server_available: bool,
    ) -> None:
        owner_label = "Local" if owner_id == "local" else f"Server ({owner_id})"
        self.query_one("#scheduling-owner-select", Static).update(owner_label)
        self.query_one("#scheduling-last-pull", Static).update(
            f"Last pull: {last_pull_at or '—'}"
        )
        self.query_one("#scheduling-last-push", Static).update(
            f"Last push: {last_push_at or '—'}"
        )
        error_widget = self.query_one("#scheduling-sync-error", Static)
        if sync_errors:
            error_widget.update(str(sync_errors[-1].get("message", "")))
        else:
            error_widget.update("")
        clear_button = self.query_one("#scheduling-clear-error", Button)
        clear_button.disabled = not sync_errors
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest Tests/UI/test_schedules_workbench.py::test_sync_status_widget_renders_mode_and_timestamps -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/UI/Screens/scheduling/sync_status_widget.py Tests/UI/test_schedules_workbench.py
git commit -m "feat(scheduling): add SyncStatusWidget for workbench sync state"
```

---

## Task 8: Build `ConflictsTab`

**Files:**
- Create: `tldw_chatbook/UI/Screens/scheduling/conflicts_tab.py`
- Test: `Tests/UI/test_schedules_workbench.py`

- [ ] **Step 1: Write the failing test**

Add to `Tests/UI/test_schedules_workbench.py`:

```python
from tldw_chatbook.UI.Screens.scheduling.conflicts_tab import ConflictsTab


def test_conflicts_tab_renders_rows():
    class FakeEngine:
        def resolve_conflict(self, conflict_id, resolution):
            self.last_call = (conflict_id, resolution)

    engine = FakeEngine()
    tab = ConflictsTab(sync_engine=engine)
    tab.populate([
        {
            "id": "c1",
            "local_id": "l1",
            "server_state": {},
            "local_state": {"record": {"title": "Local"}},
        },
        {
            "id": "c2",
            "local_id": "l2",
            "server_state": {"updated_at": "2026-07-19T10:00:00Z"},
            "local_state": {"record": {"title": "Local"}},
        },
    ])

    table = tab.query_one("#scheduling-conflicts-table", DataTable)
    assert table.row_count == 2
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest Tests/UI/test_schedules_workbench.py::test_conflicts_tab_renders_rows -v
```

Expected: `ImportError`.

- [ ] **Step 3: Write minimal implementation**

Create `tldw_chatbook/UI/Screens/scheduling/conflicts_tab.py`:

```python
"""Conflicts tab for the Schedules workbench."""

from __future__ import annotations

from typing import Any

from textual.widgets import Button, DataTable, Static
from textual.containers import Vertical


class ConflictsTab(Vertical):
    """DataTable of unresolved sync conflicts with per-row actions."""

    DEFAULT_CSS = """
    ConflictsTab {
        height: 1fr;
    }
    #scheduling-conflicts-table {
        height: 1fr;
    }
    """

    def __init__(self, sync_engine: Any, **kwargs) -> None:
        super().__init__(**kwargs)
        self.sync_engine = sync_engine

    def compose(self):
        yield Static("Unresolved conflicts")
        table = DataTable(id="scheduling-conflicts-table")
        table.add_columns("Title", "Conflict Type", "Server updated", "Local updated")
        yield table

    def populate(self, conflicts: list[dict[str, Any]]) -> None:
        table = self.query_one("#scheduling-conflicts-table", DataTable)
        table.clear()
        for conflict in conflicts:
            server_state = conflict.get("server_state") or {}
            local_state = conflict.get("local_state") or {}
            local_row = local_state.get("record") or local_state or {}
            conflict_type = "server-deletion" if not server_state else "server-update"
            server_updated = server_state.get("updated_at", "—")
            local_updated = local_row.get("updated_at", "—")
            table.add_row(
                local_row.get("title", "Untitled"),
                conflict_type,
                server_updated,
                local_updated,
                key=conflict["id"],
            )

    def on_mount(self):
        table = self.query_one("#scheduling-conflicts-table", DataTable)
        table.cursor_type = "row"
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest Tests/UI/test_schedules_workbench.py::test_conflicts_tab_renders_rows -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/UI/Screens/scheduling/conflicts_tab.py Tests/UI/test_schedules_workbench.py
git commit -m "feat(scheduling): add ConflictsTab widget for sync conflicts"
```

---

## Task 9: Wire `SchedulesWorkbench` with sync worker, status bar, tabs, and owner switcher

**Files:**
- Modify: `tldw_chatbook/UI/Screens/scheduling/schedules_workbench.py`
- Test: `Tests/UI/test_schedules_workbench.py`

- [ ] **Step 1: Write the failing tests**

Add to `Tests/UI/test_schedules_workbench.py`:

```python
from tldw_chatbook.Scheduling.events import SyncCompleted, SyncFailed


@pytest.mark.asyncio
async def test_schedules_workbench_sync_worker_posts_completed():
    class FakeService:
        def __init__(self):
            self.server_client = None
            self.sync_now = AsyncMock()
        def set_owner(self, owner_id):
            self.owner_id = owner_id

    service = FakeService()
    app = WorkbenchTestAppWithService()
    app.scheduling_service = service
    workbench = SchedulesWorkbench(app)

    posted = []
    workbench.post_message = lambda msg: posted.append(msg)
    workbench._scheduling_service = service
    workbench.action_sync_now()
    await workbench._sync_worker  # wait for worker if exposed

    assert any(isinstance(m, SyncCompleted) for m in posted)
```

This test is intentionally high-level; adjust based on actual Textual worker APIs. A simpler unit test:

```python
def test_action_sync_now_notifies_when_no_service():
    app = WorkbenchTestApp()
    workbench = SchedulesWorkbench(app)
    # Should not crash
    workbench.action_sync_now()
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest Tests/UI/test_schedules_workbench.py::test_action_sync_now_notifies_when_no_service -v
```

Expected: FAIL or behavior mismatch.

- [ ] **Step 3: Write minimal implementation**

Modify `tldw_chatbook/UI/Screens/scheduling/schedules_workbench.py`:

```python
from textual.widgets import Button, DataTable, Select, Static, TabbedContent, TabPane

from ....Scheduling.events import (
    DeleteTaskRequested,
    DisableTaskRequested,
    EditTaskRequested,
    EnableTaskRequested,
    SyncCompleted,
    SyncFailed,
)
from ....UI.Screens.scheduling.conflicts_tab import ConflictsTab
from ....UI.Screens.scheduling.sync_status_widget import SyncStatusWidget
```

Update `compose_content`:

```python
def compose_content(self) -> ComposeResult:
    service = self._service()
    owner_id = service.owner_id if service else "local"
    yield SyncStatusWidget(id="scheduling-sync-status")
    with TabbedContent():
        with TabPane("Queue", id="scheduling-queue-tab"):
            with Horizontal(id="scheduling-workbench"):
                with Vertical(id="scheduling-list-pane"):
                    yield Static("Schedule Queue", id="scheduling-list-title")
                    yield DataTable(id="scheduling-task-table")
                with Vertical(id="scheduling-detail-pane"):
                    yield TaskDetail(id="scheduling-task-detail")
                with Vertical(id="scheduling-inspector-pane"):
                    yield TaskInspector(id="scheduling-task-inspector")
        with TabPane("Conflicts", id="scheduling-conflicts-tab"):
            yield ConflictsTab(id="scheduling-conflicts", sync_engine=service.sync_engine if service else None)
```

Add owner switching and sync worker methods:

```python
    def on_mount(self) -> None:
        super().on_mount()
        self._register_footer_shortcuts()
        self._refresh_owner_select()
        table = self.query_one("#scheduling-task-table", DataTable)
        table.add_columns("Title", "Type", "Status", "Next Run")
        self.run_worker(self.load_tasks, exclusive=True)

    def _refresh_owner_select(self) -> None:
        status = self.query_one("#scheduling-sync-status", SyncStatusWidget)
        service = self._service()
        if service is None:
            status.update_status("local", None, None, [], False)
            return
        state = service.db.get_sync_state(service.owner_id) or {}
        status.update_status(
            owner_id=service.owner_id,
            last_pull_at=state.get("last_pull_at"),
            last_push_at=state.get("last_push_at"),
            sync_errors=state.get("sync_errors") or [],
            server_available=service.server_client.notifications_service is not None,
        )

    @on(Button.Pressed, "#scheduling-clear-error")
    def _on_clear_sync_errors(self) -> None:
        service = self._service()
        if service is None:
            return
        service.db.update_sync_state(service.owner_id, sync_errors=[])
        self._refresh_owner_select()

    @on(SyncCompleted)
    def _on_sync_completed(self, event: SyncCompleted) -> None:
        self.app_instance.notify("Sync completed.", severity="information")
        self._refresh_owner_select()
        self.run_worker(self.load_tasks, exclusive=True)
        self._refresh_conflicts_tab()

    @on(SyncFailed)
    def _on_sync_failed(self, event: SyncFailed) -> None:
        self.app_instance.notify(f"Sync failed: {event.error}", severity="error")
        self._refresh_owner_select()
        self.run_worker(self.load_tasks, exclusive=True)
        self._refresh_conflicts_tab()

    def _refresh_conflicts_tab(self) -> None:
        service = self._service()
        if service is None:
            return
        conflicts_tab = self.query_one("#scheduling-conflicts", ConflictsTab)
        conflicts = service.db.get_conflicts(service.owner_id, primitive="reminder_task")
        conflicts_tab.populate(conflicts)

    def action_sync_now(self) -> None:
        """Sync schedule state now."""
        service = self._service()
        if service is None:
            self.app_instance.notify(
                "Scheduling service is unavailable; cannot sync.",
                severity="warning",
            )
            return
        self.run_worker(self._run_sync, exclusive=True)

    async def _run_sync(self) -> None:
        service = self._service()
        if service is None:
            return
        owner_select = self.query_one("#scheduling-owner-select", Static)
        owner_select.disabled = True
        try:
            owner_id = service.owner_id
            await service.sync_now(owner_id)
            conflicts = service.db.get_conflicts(owner_id, primitive="reminder_task")
            self.post_message(SyncCompleted(owner_id, conflict_count=len(conflicts)))
        except Exception as exc:  # noqa: BLE001
            logger.exception("Sync failed")
            self.post_message(SyncFailed(service.owner_id, str(exc)))
        finally:
            owner_select.disabled = False
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest Tests/UI/test_schedules_workbench.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/UI/Screens/scheduling/schedules_workbench.py Tests/UI/test_schedules_workbench.py
git commit -m "feat(scheduling): wire sync worker, status bar, conflicts tab, and owner switcher into workbench"
```

---

## Task 10: Update `app.py` to always instantiate `SchedulingServerClient`

**Files:**
- Modify: `tldw_chatbook/app.py`

- [ ] **Step 1: Locate the constructor block**

Find the block near `tldw_chatbook/app.py:4034-4048`:

```python
server_client = None
if self.server_notifications_service is not None:
    server_client = SchedulingServerClient(self.server_notifications_service)
```

- [ ] **Step 2: Update the instantiation**

Replace with:

```python
server_client = SchedulingServerClient(self.server_notifications_service)
```

- [ ] **Step 3: Verify no existing tests break**

```bash
pytest Tests/ -k "scheduling or schedules" -v
```

Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add tldw_chatbook/app.py
git commit -m "feat(scheduling): always instantiate SchedulingServerClient in app startup"
```

---

## Task 11: Full test run and lint

**Files:**
- All modified files.

- [ ] **Step 1: Run the Scheduling and UI test suites**

```bash
pytest Tests/Scheduling Tests/UI/test_schedules_workbench.py -v
```

Expected: PASS.

- [ ] **Step 2: Run lint/format checks**

```bash
ruff check tldw_chatbook/Scheduling tldw_chatbook/UI/Screens/scheduling tldw_chatbook/app.py
mypy tldw_chatbook/Scheduling tldw_chatbook/UI/Screens/scheduling
```

Expected: clean or only pre-existing issues.

- [ ] **Step 3: Fix any new lint/test failures**

Edit affected files as needed.

- [ ] **Step 4: Final commit**

```bash
git add -A
git commit -m "style(scheduling): address lint and test fixes for TASK-299.2"
```

---

## Task 12: Update backlog task status

**Files:**
- Modify: `backlog/tasks/task-299.2 - Bidirectional-reminder-sync-with-tldw_server.md`

- [ ] **Step 1: Mark implementation plan complete**

Add under `## Implementation Notes` (or update status via `backlog task edit 299.2 -s Done` after all code is merged).

For now, after the plan is written and before execution:

```bash
backlog task edit 299.2 --plan "See Docs/superpowers/plans/2026-07-19-bidirectional-reminder-sync-implementation-plan.md"
```

- [ ] **Step 2: Commit task metadata update**

```bash
git add backlog/tasks/task-299.2\ -\ Bidirectional-reminder-sync-with-tldw_server.md
git commit -m "chore(backlog): add implementation plan reference to TASK-299.2"
```
