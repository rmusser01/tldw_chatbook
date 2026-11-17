"""Stock Workspace display reads share physical custody and publish independently."""

import asyncio
import sqlite3
from contextlib import contextmanager
from threading import Event
from types import SimpleNamespace

import pytest

from Tests.UI.test_console_workspace_controller import (
    _AsyncWorkerScreen,
    _workspace_controller,
)
from Tests.UI.test_console_hook_review_send_freeze import _task_factory
from tldw_chatbook.Backup_Recovery import storage_admission as storage
from tldw_chatbook.DB.Workspace_DB import WorkspaceDB
from tldw_chatbook.UI.Console_Modules import config_sync
from tldw_chatbook.UI.Console_Modules import workspace as workspace_module
from tldw_chatbook.Workspaces.models import RuntimeBindingStatus
from tldw_chatbook.Workspaces.registry_service import LocalWorkspaceRegistryService

pytestmark = [pytest.mark.asyncio, pytest.mark.bootstrap_profile]


@pytest.fixture
def workspace_display(tmp_path, monkeypatch):
    database = WorkspaceDB(tmp_path / "workspace-display.sqlite", client_id="display")
    registry = LocalWorkspaceRegistryService(database)
    registry.create_workspace(workspace_id="alpha", name="Alpha")
    registry.create_workspace(workspace_id="beta", name="Beta")
    folder = tmp_path / "bound"
    folder.mkdir()
    binding = registry.add_folder_binding("alpha", folder)
    database.close()
    screen = _AsyncWorkerScreen()
    screen._console_transcript_sync_timer = object()
    controller = _workspace_controller(
        screen=screen, app_instance=SimpleNamespace(workspace_registry_service=registry)
    )
    view = controller._console_registry_reads_view()
    assert view.list_runtime_bindings("alpha") == (binding,)
    assert {record.name for record in view.list_workspaces()} >= {"Alpha", "Beta"}
    database.close()

    operations = []
    connections = []
    queries = []
    publications = []
    original_operation = storage._repository_operation
    original_open = WorkspaceDB._open_connection

    @contextmanager
    def observe_operation(participant):
        with original_operation(participant) as operation:
            if participant.repository() is database and operation not in operations:
                operations.append(operation)
            yield operation

    def observe_open(owner, *args, **kwargs):
        connection = original_open(owner, *args, **kwargs)
        if owner is database:
            connections.append(connection)
            connection.set_trace_callback(queries.append)
        return connection

    def publish():
        assert all(operation not in storage._operations for operation in operations)
        for connection in connections:
            with pytest.raises(sqlite3.ProgrammingError):
                sqlite3.Connection.in_transaction.__get__(connection)
        publications.append(controller._workspace_files_availability_by_id)

    monkeypatch.setattr(storage, "_repository_operation", observe_operation)
    monkeypatch.setattr(WorkspaceDB, "_open_connection", observe_open)
    monkeypatch.setattr(controller, "_publish_display_reads", publish)
    yield SimpleNamespace(
        database=database,
        registry=registry,
        screen=screen,
        controller=controller,
        view=view,
        binding=binding,
        folder=folder,
        operations=operations,
        connections=connections,
        queries=queries,
        publications=publications,
    )
    database.close()


def _rename_workspace(case):
    case.registry.rename_workspace("alpha", "Renamed")
    case.database.close()
    case.operations.clear()
    case.connections.clear()
    case.queries.clear()


def _request_registry(case):
    with case.controller._stale_display_reads():
        case.view.list_runtime_bindings("alpha")


@pytest.mark.parametrize("sections", ["both", "registry", "availability"])
async def test_due_workspace_display_sections_share_one_operation(
    workspace_display, sections
):
    case = workspace_display
    case.folder.rmdir()
    if sections != "availability":
        _rename_workspace(case)
    if sections != "availability":
        _request_registry(case)
    if sections != "registry":
        case.controller._request_workspace_files_availability_refresh(("alpha",))
    await asyncio.gather(*(worker for worker, _ in case.screen.workers))

    assert len(case.operations) == 1
    assert len(case.connections) == 1
    assert len(case.publications) == 1
    assert (
        sum("FROM workspace_runtime_bindings" in query for query in case.queries) == 1
    )
    if sections != "availability":
        assert case.view.stale_calls() == {}
        assert case.view.list_runtime_bindings("alpha") == (case.binding,)
        assert "Renamed" in {record.name for record in case.view.list_workspaces()}
    if sections != "registry":
        assert case.controller._workspace_files_availability_by_id == {"alpha": False}
        (binding,) = case.controller._workspace_files_runtime_bindings_by_id["alpha"]
        assert binding.status is RuntimeBindingStatus.MISSING


async def test_eager_worker_batches_both_demands_from_the_same_render(
    workspace_display,
):
    case = workspace_display
    _rename_workspace(case)
    with _task_factory("eager"):
        _request_registry(case)
        case.controller._request_workspace_files_availability_refresh(("alpha",))
    await asyncio.gather(*(worker for worker, _ in case.screen.workers))

    assert len(case.operations) == len(case.connections) == 1
    assert case.publications == [{"alpha": True}]


async def test_poll_stop_publishes_owed_workspace_tray_after_controls(monkeypatch):
    published = []
    screen = SimpleNamespace(
        _console_transcript_sync_timer=object(),
        _sync_console_workspace_context=lambda: published.append("workspace"),
    )

    def project(owner, *, force):
        assert owner is screen and force
        published.append("controls")
        return True

    monkeypatch.setattr(config_sync, "project_console_tick", project)
    config_sync.publish_console_presentation(screen, "console-sync")
    config_sync.settle_stopped_poll(screen, lambda _screen: False)

    assert published == ["controls", "workspace"]


async def test_availability_failure_does_not_renew_its_age_or_discard_registry(
    workspace_display,
    monkeypatch,
):
    case = workspace_display
    _rename_workspace(case)

    def unavailable(_binding):
        raise OSError("status unavailable")

    monkeypatch.setattr(
        workspace_module, "_recompute_filesystem_binding_status", unavailable
    )
    _request_registry(case)
    case.controller._request_workspace_files_availability_refresh(("alpha",))
    await asyncio.gather(*(worker for worker, _ in case.screen.workers))

    assert case.view.stale_calls() == {}
    assert case.controller._workspace_files_availability_cached_at == 0.0
    assert len(case.operations) == len(case.connections) == 1


async def test_demand_arriving_during_registry_read_runs_after_retirement(
    workspace_display,
    monkeypatch,
):
    case = workspace_display
    _rename_workspace(case)
    entered, release = Event(), Event()
    original = case.registry.list_workspaces

    def hold_read():
        records = original()
        entered.set()
        assert release.wait(10), "registry read was not released"
        return records

    monkeypatch.setattr(case.registry, "list_workspaces", hold_read)
    _request_registry(case)
    try:
        async with asyncio.timeout(10):
            while not entered.is_set():
                await asyncio.sleep(0.01)
        case.controller._request_workspace_files_availability_refresh(("beta",))
        assert len(case.screen.workers) == 1
    finally:
        release.set()
        await asyncio.gather(*(worker for worker, _ in case.screen.workers))

    assert len(case.operations) == len(case.connections) == 2
    assert case.view.stale_calls() == {}
    assert case.controller._workspace_files_availability_by_id == {"beta": False}


@pytest.mark.parametrize("change", ["generation", "source", "cancel"])
async def test_held_batch_discards_obsolete_registry_publication(
    workspace_display,
    monkeypatch,
    change,
):
    case = workspace_display
    _rename_workspace(case)
    entered, release = Event(), Event()
    original = case.registry.list_workspaces

    def hold_read():
        records = original()
        if not entered.is_set():
            entered.set()
            assert release.wait(10), "registry read was not released"
        return records

    monkeypatch.setattr(case.registry, "list_workspaces", hold_read)
    _request_registry(case)
    case.controller._request_workspace_files_availability_refresh(("alpha",))
    task = case.screen.workers[0][0]
    try:
        async with asyncio.timeout(10):
            while not entered.is_set():
                await asyncio.sleep(0.01)
        assert not case.publications
        if change == "generation":
            case.registry.rename_workspace("alpha", "Latest")
            case.database.close()
            _request_registry(case)
        elif change == "source":
            case.controller.app_instance.workspace_registry_service = (
                LocalWorkspaceRegistryService(case.database)
            )
            case.controller._request_workspace_files_availability_refresh(("beta",))
        else:
            task.cancel()
            await asyncio.sleep(0)
            assert not task.done()
            assert case.controller._preparation_reads
            assert (
                sqlite3.Connection.in_transaction.__get__(case.connections[-1]) is False
            )
        assert len(case.screen.workers) == 1
    finally:
        release.set()
        outcomes = await asyncio.gather(task, return_exceptions=True)

    assert not case.controller._preparation_reads
    if change == "generation":
        assert outcomes == [None]
        assert "Latest" in {record.name for record in case.view.list_workspaces()}
        assert case.publications == [{"alpha": True}, {"alpha": True}]
    elif change == "source":
        assert outcomes == [None]
        assert case.view.stale_calls()
        assert case.publications == [{"beta": False}]
    else:
        assert isinstance(outcomes[0], asyncio.CancelledError)
        assert not case.publications
        with pytest.raises(sqlite3.ProgrammingError):
            sqlite3.Connection.in_transaction.__get__(case.connections[-1])


async def test_external_folder_change_is_checked_again_after_one_second(
    workspace_display,
):
    case = workspace_display
    case.controller._request_workspace_files_availability_refresh(("alpha",))
    await case.screen.workers[-1][0]
    assert case.controller._workspace_files_availability_by_id == {"alpha": True}
    assert len(case.connections) == 1

    case.folder.rmdir()
    case.controller._request_workspace_files_availability_refresh(("alpha",))
    assert len(case.screen.workers) == 1
    case.controller._workspace_files_availability_cached_at -= 1.0
    case.controller._request_workspace_files_availability_refresh(("alpha",))
    await case.screen.workers[-1][0]
    assert case.controller._workspace_files_availability_by_id == {"alpha": False}
    assert len(case.operations) == len(case.connections) == 2
