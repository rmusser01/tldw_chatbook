"""End-to-end transfer machine tests (schedules-handoff PR-5, Task 8).

Drives the real ``SchedulingService``/``SyncEngine`` transfer facade
against a real ``tmp_path`` ``ScheduledTasksDB`` and a stateful fake
server client -- never a reimplementation of the state machine, matching
``test_transfer_invariant.py``'s (Task 6) and
``test_automation_sync_end_to_end.py``'s (Task 6) own precedent. The
hypothesis property test already stress-tests every interleaving of the
reminder-side machinery; this file is the directed happy-path-plus-
failure-legs walkthrough spec §10 asks for, covering BOTH primitives
(reminder + automation_definition) and the two things the property test
deliberately scopes out: the definition preview->create network shape and
the schedule-vocabulary translation at each boundary.

Five scenarios, matching the task-8 brief:

(a) local reminder -> begin_transfer_to_server -> sync (disarm-before-
    create, link fields, convert to mirror) -- never both armed at every
    observed step.
(b) local automation_definition -> begin_transfer_to_server -> sync
    (preview->create; schedule arrives in SERVER vocabulary at the fake).
(c) server-owned automation_definition mirror -> begin_transfer_to_local
    -> dormant copy (CLIENT-vocab schedule) -> release-ack -> copy armed
    + mirror archived.
(d) crash recovery: a reminder seeded ``to_server_sent`` with a retained
    mutation, whose create actually landed server-side before the
    process died -- ``recover_inflight_transfers``' list-and-match
    completes the conversion (spec §6.1.3).
(e) a definitively-failed definition transfer (invalid preview) settles
    ``to_server_failed`` and re-arms locally; ``begin_transfer_to_server``
    on that row is the retry leg (Task 6 fix round) -- CAS back to
    pending with a fresh payload, and the retry succeeds.
"""

from __future__ import annotations

import pytest

from tldw_chatbook.Scheduling.db.scheduled_tasks_db import (
    DORMANT_TRANSFER_STATES,
    ScheduledTasksDB,
)
from tldw_chatbook.Scheduling.services import scheduling_service as scheduling_service_module
from tldw_chatbook.Scheduling.services.scheduling_service import SchedulingService
from tldw_chatbook.Scheduling.services.server_client import (
    ServerClientValidationError,
    ServerUnavailableError,
)


class _FakeApp:
    """Only ``active_server_id`` is read by the transfer facade."""

    def __init__(self, active_server_id: str = "1") -> None:
        self.active_server_id = active_server_id


class _FakeTransferServerClient:
    """Stateful fake covering both reminder and definition transfer legs.

    Combines ``test_transfer_invariant.py``'s reminder fake and
    ``test_sync_engine.py``'s definition AsyncMock expectations into one
    client, so a single E2E test can drive both primitives through the
    REAL ``SyncEngine`` push/pull phases across multiple ``sync_now()``
    calls (a real round trip, not one mocked call per assertion).
    """

    def __init__(self) -> None:
        self.notifications_service = object()
        self._seq = 0

        # -- reminders: server-side "live" set --
        self.reminders: dict[str, dict] = {}
        self.deleted_reminders: set[str] = set()
        self.reminder_create_calls: list[dict] = []
        self.reminder_outcome = "success"  # "success" | "fail" | "ambiguous"
        #: Fires at the START of create_reminder, before the row is
        #: touched -- lets a test snapshot local DB state at the instant
        #: the network request is (about to be) sent.
        self.on_reminder_create_start = None

        # -- automation definitions --
        self.preview_calls: list[dict] = []
        self.preview_valid = True
        self.preview_errors: list[dict] = [
            {"field": "name", "code": "invalid", "message": "bad name"}
        ]
        self.create_definition_calls: list[tuple[str, str]] = []
        self.definitions: dict[str, dict] = {}
        self.archived_definitions: set[str] = set()
        self.on_definition_preview_start = None

    def _new_id(self, prefix: str) -> str:
        self._seq += 1
        return f"{prefix}-{self._seq}"

    # -- reminders --------------------------------------------------
    async def create_reminder(self, **payload):
        self.reminder_create_calls.append(dict(payload))
        if self.on_reminder_create_start is not None:
            self.on_reminder_create_start()
        if self.reminder_outcome == "fail":
            raise ServerClientValidationError("rejected")
        server_id = self._new_id("srv-rem")
        self.reminders[server_id] = {"id": server_id, **payload}
        if self.reminder_outcome == "ambiguous":
            raise ServerUnavailableError("timeout after create")
        return dict(self.reminders[server_id])

    async def delete_reminder(self, server_id):
        self.deleted_reminders.add(server_id)
        return {}

    async def list_reminders(self):
        return {
            "items": [
                dict(item)
                for sid, item in self.reminders.items()
                if sid not in self.deleted_reminders
            ]
        }

    # -- automation definitions --------------------------------------
    async def preview_automation_definition(self, request):
        self.preview_calls.append(dict(request))
        if self.on_definition_preview_start is not None:
            self.on_definition_preview_start()
        if self.preview_valid:
            return {"id": self._new_id("prev"), "status": "valid", "validation_errors": []}
        return {
            "id": self._new_id("prev"),
            "status": "invalid",
            "validation_errors": self.preview_errors,
        }

    async def create_automation_definition(self, preview_id, *, initial_lifecycle="configured"):
        self.create_definition_calls.append((preview_id, initial_lifecycle))
        server_id = self._new_id("srv-def")
        record = {
            "id": server_id,
            "family": "recurring_question",
            "name": "Daily digest",
            "lifecycle": initial_lifecycle,
            "schedule": {"kind": "interval", "seconds": 3600},
        }
        self.definitions[server_id] = record
        return dict(record)

    async def archive_automation_definition(self, server_definition_id):
        self.archived_definitions.add(server_definition_id)
        record = dict(
            self.definitions.get(server_definition_id) or {"id": server_definition_id}
        )
        record["lifecycle"] = "archived"
        return record

    async def list_automation_definitions(self, *, limit: int = 50, offset: int = 0):
        return {"items": [], "total": 0, "has_more": False}

    async def list_automation_results(self, *, limit: int = 50, offset: int = 0):
        return {"items": [], "total": 0, "has_more": False}


def _locally_armed(row: dict) -> bool:
    """Spec §3's armed check: local owner, not a dormant transfer state.

    Mirrors ``test_transfer_invariant.py``'s own invariant exactly (never
    a second implementation of "armed").
    """
    return (
        not str(row.get("owner_id") or "").startswith("server:")
        and row.get("transfer_state") not in DORMANT_TRANSFER_STATES
    )


def _reminder_server_armed(fake: _FakeTransferServerClient, link_id: str) -> bool:
    return any(
        sid not in fake.deleted_reminders and item.get("link_id") == link_id
        for sid, item in fake.reminders.items()
    )


def _assert_never_both_armed(row: dict, server_armed: bool) -> None:
    assert not (_locally_armed(row) and server_armed), (
        f"BOTH armed: row={row!r} server_armed={server_armed}"
    )


@pytest.mark.asyncio
async def test_local_reminder_transfer_to_server(tmp_path):
    """(a) local reminder -> begin -> sync: disarm-before-create, link
    fields, convert to mirror -- never both armed at every observed step.
    """
    db = ScheduledTasksDB(tmp_path / "db.db")
    fake = _FakeTransferServerClient()
    app = _FakeApp()
    svc = SchedulingService(
        db=db, server_client=fake, runtime_source="server:1", app_getter=lambda: app
    )
    # task-3 (ruling 4): pre-seed the reachability probe verdict this
    # always-connected fake implies -- see test_scheduling_service.py's
    # `_transfer_service` for the same precedent.
    svc._server_reachable = True

    row_id = db.create_reminder_task(
        owner_id="local",
        title="Ping",
        schedule_kind="one_time",
        run_at="2030-01-01T00:00:00+00:00",
    )
    row = db.get_reminder_task(row_id)
    _assert_never_both_armed(row, _reminder_server_armed(fake, row_id))

    outcome = await svc.begin_transfer_to_server("reminder_task", row_id)
    assert outcome.status == "pending"
    row = db.get_reminder_task(row_id)
    assert row["transfer_state"] == "to_server_pending"
    assert _locally_armed(row), "merely queued -- keeps executing locally (spec §6.1.1)"
    _assert_never_both_armed(row, _reminder_server_armed(fake, row_id))

    observed_state_at_send: list[object] = []
    fake.on_reminder_create_start = lambda: observed_state_at_send.append(
        db.get_reminder_task(row_id)["transfer_state"]
    )

    sync_outcome = await svc.sync_now()
    assert sync_outcome.status == "ok"
    assert observed_state_at_send == ["to_server_sent"], (
        "the row must be disarmed BEFORE the create request goes out"
    )
    # Not-yet-armed-anywhere is a valid transient (disarmed locally, not
    # yet created server-side) -- still "never both".
    row_mid = db.get_reminder_task(row_id)
    assert not _locally_armed(row_mid)

    call = fake.reminder_create_calls[0]
    assert call["link_type"] == "chatbook_transfer"
    assert call["link_id"] == row_id

    row = db.get_reminder_task(row_id)
    assert row["owner_id"] == "server:1"
    assert row["server_id"] is not None
    assert row["transfer_state"] is None
    server_armed = _reminder_server_armed(fake, row_id)
    assert server_armed, "the row is now a live server mirror"
    _assert_never_both_armed(row, server_armed)
    assert db.get_pending_mutations("server:1", primitive="reminder_task") == []


@pytest.mark.asyncio
async def test_local_definition_transfer_to_server_uses_server_vocabulary(tmp_path):
    """(b) local automation_definition -> begin -> sync via preview->create;
    the schedule the fake receives is in SERVER vocabulary."""
    db = ScheduledTasksDB(tmp_path / "db.db")
    fake = _FakeTransferServerClient()
    app = _FakeApp()
    svc = SchedulingService(
        db=db, server_client=fake, runtime_source="server:1", app_getter=lambda: app
    )
    # task-3 (ruling 4): pre-seed the reachability probe verdict this
    # always-connected fake implies -- see test_scheduling_service.py's
    # `_transfer_service` for the same precedent.
    svc._server_reachable = True

    def_id = db.create_automation_definition(
        "local",
        "recurring_question",
        "Daily digest",
        schedule={"kind": "interval", "every_seconds": 3600},
        lifecycle="paused",
    )

    outcome = await svc.begin_transfer_to_server("automation_definition", def_id)
    assert outcome.status == "pending"
    row = db.get_automation_definition(def_id)
    assert row["transfer_state"] == "to_server_pending"
    assert _locally_armed(row)

    observed_state_at_send: list[object] = []
    fake.on_definition_preview_start = lambda: observed_state_at_send.append(
        db.get_automation_definition(def_id)["transfer_state"]
    )

    sync_outcome = await svc.sync_now()
    assert sync_outcome.status == "ok"
    assert observed_state_at_send == ["to_server_sent"], "disarm-before-send"

    request = fake.preview_calls[0]
    assert request["mode"] == "create"
    assert request["schedule"] == {"kind": "interval", "seconds": 3600}, (
        "the fake must see SERVER vocabulary (seconds, not every_seconds)"
    )
    assert len(fake.create_definition_calls) == 1
    preview_id, initial_lifecycle = fake.create_definition_calls[0]
    assert preview_id  # the preview's own id, threaded through to create
    assert initial_lifecycle == "paused", (
        "initial_lifecycle must match the source row's own lifecycle"
    )

    row = db.get_automation_definition(def_id)
    assert row["owner_id"] == "server:1"
    assert row["server_id"] is not None
    assert row["transfer_state"] is None
    assert (
        db.get_pending_mutations("server:1", primitive="automation_definition") == []
    )


@pytest.mark.asyncio
async def test_server_definition_release_to_local(tmp_path, monkeypatch):
    """(c) server-owned mirror -> begin_transfer_to_local -> dormant copy
    (CLIENT-vocab schedule) -> release-ack -> copy armed + mirror archived.
    """
    monkeypatch.setattr(
        scheduling_service_module, "compute_local_health", lambda app, row: ("ready", "")
    )
    db = ScheduledTasksDB(tmp_path / "db.db")
    fake = _FakeTransferServerClient()
    app = _FakeApp()
    svc = SchedulingService(
        db=db, server_client=fake, runtime_source="server:1", app_getter=lambda: app
    )
    # task-3 (ruling 4): pre-seed the reachability probe verdict this
    # always-connected fake implies -- see test_scheduling_service.py's
    # `_transfer_service` for the same precedent.
    svc._server_reachable = True

    mirror_id = db.create_automation_definition(
        "server:1",
        "recurring_question",
        "Daily digest",
        server_id="srv-def-1",
        # Mirrors are stored in SERVER vocabulary (pulled verbatim).
        schedule={"kind": "interval", "seconds": 3600},
    )
    fake.definitions["srv-def-1"] = {"id": "srv-def-1", "lifecycle": "configured"}

    outcome = await svc.begin_transfer_to_local("automation_definition", mirror_id)
    assert outcome.status == "pending"
    copy_id = outcome.row_id
    assert copy_id is not None

    copy_row = db.get_automation_definition(copy_id)
    assert copy_row["owner_id"] == "local"
    assert copy_row["transfer_state"] == "from_server_pending"
    assert copy_row["schedule"] == {"kind": "interval", "every_seconds": 3600}, (
        "the dormant copy is translated to CLIENT vocabulary"
    )
    assert not _locally_armed(copy_row), "dormant until the release acks"

    mirror_row = db.get_automation_definition(mirror_id)
    assert not _locally_armed(mirror_row), "server-owned, unaffected -- keeps executing"

    sync_outcome = await svc.sync_now()
    assert sync_outcome.status == "ok"
    assert "srv-def-1" in fake.archived_definitions

    copy_row = db.get_automation_definition(copy_id)
    assert copy_row["transfer_state"] is None, "ack arms the copy"
    assert _locally_armed(copy_row)

    mirror_row = db.get_automation_definition(mirror_id)
    assert mirror_row["lifecycle"] == "archived"
    assert (
        db.get_pending_mutations("server:1", primitive="automation_definition") == []
    )


@pytest.mark.asyncio
async def test_crash_recovery_reminder_ambiguous_timeout(tmp_path):
    """(d) A reminder stuck ``to_server_sent`` with a retained mutation --
    the create actually landed server-side before the process died.
    ``recover_inflight_transfers``' list-and-match completes the
    conversion (spec §6.1.3)."""
    db = ScheduledTasksDB(tmp_path / "db.db")
    fake = _FakeTransferServerClient()
    app = _FakeApp()
    svc = SchedulingService(
        db=db, server_client=fake, runtime_source="server:1", app_getter=lambda: app
    )
    # task-3 (ruling 4): pre-seed the reachability probe verdict this
    # always-connected fake implies -- see test_scheduling_service.py's
    # `_transfer_service` for the same precedent.
    svc._server_reachable = True

    row_id = db.create_reminder_task(
        owner_id="local",
        title="Ping",
        schedule_kind="one_time",
        run_at="2030-01-01T00:00:00+00:00",
    )
    armed = db.set_transfer_state("reminder_task", row_id, "to_server_sent", expected=(None,))
    assert armed
    db.record_pending_mutation(
        row_id,
        "reminder_task",
        "server:1",
        {"action": "transfer_to_server", "task_payload": {"title": "Ping"}},
    )
    # The create actually landed server-side before the ack reached this
    # process (the "ambiguous timeout" leg, spec §6.1.3).
    fake.reminders["srv-rem-1"] = {
        "id": "srv-rem-1",
        "title": "Ping",
        "link_type": "chatbook_transfer",
        "link_id": row_id,
    }

    row = db.get_reminder_task(row_id)
    assert not _locally_armed(row), "to_server_sent is dormant"
    _assert_never_both_armed(row, _reminder_server_armed(fake, row_id))

    await svc.recover_inflight_transfers()

    row = db.get_reminder_task(row_id)
    assert row["owner_id"] == "server:1"
    assert row["server_id"] == "srv-rem-1"
    assert row["transfer_state"] is None
    _assert_never_both_armed(row, _reminder_server_armed(fake, row_id))
    assert db.get_pending_mutations("server:1", primitive="reminder_task") == []


@pytest.mark.asyncio
async def test_failed_definition_transfer_retries_and_succeeds(tmp_path):
    """(e) An invalid preview settles to_server_failed and re-arms
    locally; begin_transfer_to_server on that row is the retry leg (Task
    6 fix round) -- CAS back to pending with a fresh payload, and the
    retry succeeds once the preview is fixed."""
    db = ScheduledTasksDB(tmp_path / "db.db")
    fake = _FakeTransferServerClient()
    fake.preview_valid = False
    app = _FakeApp()
    svc = SchedulingService(
        db=db, server_client=fake, runtime_source="server:1", app_getter=lambda: app
    )
    # task-3 (ruling 4): pre-seed the reachability probe verdict this
    # always-connected fake implies -- see test_scheduling_service.py's
    # `_transfer_service` for the same precedent.
    svc._server_reachable = True

    def_id = db.create_automation_definition(
        "local",
        "recurring_question",
        "Daily digest",
        schedule={"kind": "interval", "every_seconds": 3600},
    )

    outcome = await svc.begin_transfer_to_server("automation_definition", def_id)
    assert outcome.status == "pending"

    sync_outcome = await svc.sync_now()
    assert sync_outcome.status == "ok"
    row = db.get_automation_definition(def_id)
    assert row["transfer_state"] == "to_server_failed"
    assert row["owner_id"] == "local", "re-armed locally, not dormant"
    assert _locally_armed(row)

    pending = db.get_pending_mutations("server:1", primitive="automation_definition")
    assert len(pending) == 1
    assert "transfer_errors" in pending[0]["payload"]

    # Retry: begin_transfer_to_server on a to_server_failed row CASes
    # failed -> pending and records a fresh payload (stripping
    # transfer_errors), per spec obligation (f).
    fake.preview_valid = True
    retry_outcome = await svc.begin_transfer_to_server("automation_definition", def_id)
    assert retry_outcome.status == "pending"
    row = db.get_automation_definition(def_id)
    assert row["transfer_state"] == "to_server_pending"
    pending = db.get_pending_mutations("server:1", primitive="automation_definition")
    assert len(pending) == 1
    assert "transfer_errors" not in pending[0]["payload"]

    sync_outcome = await svc.sync_now()
    assert sync_outcome.status == "ok"
    row = db.get_automation_definition(def_id)
    assert row["owner_id"] == "server:1"
    assert row["server_id"] is not None
    assert row["transfer_state"] is None
    assert (
        db.get_pending_mutations("server:1", primitive="automation_definition") == []
    )


@pytest.mark.asyncio
async def test_offline_cancel_of_a_release_reaches_the_server_never(tmp_path):
    """(f) Final review C2, end to end: server reminder -> Move to local ->
    connection drops -> user cancels -> reconnect -> sync.

    The cancelled release must make NO server call. Before the fix the
    cancel deleted the dormant copy but left the mutation (it was looked
    up via "today's active server", which is `None` while offline), so the
    reconnected sync deleted the reminder server-side: the task then
    existed nowhere.
    """
    db = ScheduledTasksDB(tmp_path / "db.db")
    fake = _FakeTransferServerClient()
    app = _FakeApp()
    svc = SchedulingService(
        db=db, server_client=fake, runtime_source="server:1", app_getter=lambda: app
    )
    # task-3 (ruling 4): pre-seed the reachability probe verdict this
    # always-connected fake implies -- see test_scheduling_service.py's
    # `_transfer_service` for the same precedent.
    svc._server_reachable = True

    mirror_id = db.create_reminder_task(
        owner_id="server:1",
        title="Ping",
        schedule_kind="one_time",
        run_at="2030-01-01T00:00:00+00:00",
        server_id="srv-rem-9",
    )
    fake.reminders["srv-rem-9"] = {"id": "srv-rem-9", "title": "Ping"}

    copy_id = (await svc.begin_transfer_to_local("reminder_task", mirror_id)).row_id
    assert copy_id is not None

    # The connection drops: no server identity resolves any more.
    app.active_server_id = None
    cancel = await svc.cancel_transfer("reminder_task", copy_id)
    assert cancel.status == "cancelled"
    assert db.get_reminder_task(copy_id) is None

    # Reconnect and sync.
    app.active_server_id = "1"
    sync_outcome = await svc.sync_now()

    assert sync_outcome.status == "ok"
    assert fake.deleted_reminders == set(), (
        "a cancelled release must never reach the server"
    )
    assert db.get_reminder_task(mirror_id) is not None, "the mirror is intact"


@pytest.mark.asyncio
async def test_offline_cancel_of_a_to_server_move_reaches_the_server_never(tmp_path):
    """(g) Final review I3, the other direction: an unattempted local ->
    server move cancelled offline left its mutation queued forever,
    CAS-skipped every cycle and suppressing pull-apply for that row."""
    db = ScheduledTasksDB(tmp_path / "db.db")
    fake = _FakeTransferServerClient()
    app = _FakeApp()
    svc = SchedulingService(
        db=db, server_client=fake, runtime_source="server:1", app_getter=lambda: app
    )
    # task-3 (ruling 4): pre-seed the reachability probe verdict this
    # always-connected fake implies -- see test_scheduling_service.py's
    # `_transfer_service` for the same precedent.
    svc._server_reachable = True

    row_id = db.create_reminder_task(
        owner_id="local",
        title="Ping",
        schedule_kind="one_time",
        run_at="2030-01-01T00:00:00+00:00",
    )
    assert (
        await svc.begin_transfer_to_server("reminder_task", row_id)
    ).status == "pending"

    app.active_server_id = None
    assert (await svc.cancel_transfer("reminder_task", row_id)).status == "cancelled"

    app.active_server_id = "1"
    assert (await svc.sync_now()).status == "ok"

    assert fake.reminder_create_calls == [], "nothing may be sent after a cancel"
    assert db.get_pending_mutations(primitive="reminder_task") == []
    row = db.get_reminder_task(row_id)
    assert row["owner_id"] == "local" and row["transfer_state"] is None
    assert _locally_armed(row), "the cancelled row keeps running here"
