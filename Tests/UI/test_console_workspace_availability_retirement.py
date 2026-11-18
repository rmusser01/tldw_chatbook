"""Physical Files reads settle before cancelled view workers disappear."""

import asyncio
import sqlite3
from contextlib import contextmanager
from threading import Event
from types import SimpleNamespace

import pytest

from Tests.UI.test_console_shutdown_recompose import ShutdownTrayApp
from Tests.UI.test_console_workspace_controller import (
    _AsyncWorkerScreen,
    _availability_controller,
)
from tldw_chatbook.DB.Workspace_DB import WorkspaceDB
from tldw_chatbook.Workspaces.registry_service import LocalWorkspaceRegistryService

pytestmark = pytest.mark.bootstrap_profile


@pytest.mark.asyncio
@pytest.mark.parametrize("root_shutdown", [False, True], ids=["cancel", "app-drain"])
@pytest.mark.parametrize("sql_error", [False, True], ids=["success", "sql-error"])
async def test_files_read_settles_native_owner_before_cancelled_view_finishes(
    tmp_path, monkeypatch, root_shutdown, sql_error
):
    db = WorkspaceDB(tmp_path / "files-read.sqlite", "files-read")
    registry = LocalWorkspaceRegistryService(db)
    workspace = registry.ensure_default_workspace()
    entered, release, retired = Event(), Event(), Event()
    original_connection = db.connection
    captured = []

    @contextmanager
    def held_connection():
        with original_connection() as connection:
            captured.append(connection)
            entered.set()
            assert release.wait(5), "actual Workspace callback was not released"
            if sql_error:
                connection.execute("SELECT * FROM missing_files_read_table")
            yield connection

    monkeypatch.setattr(db, "connection", held_connection)
    screen = ShutdownTrayApp() if root_shutdown else _AsyncWorkerScreen()
    sync_calls = []
    controller = _availability_controller(
        monkeypatch,
        registry,
        screen,
        [[SimpleNamespace(workspace_id=workspace.workspace_id)]],
        sync_calls,
    )
    original_capture = controller._capture_workspace_files_availability

    def observed_capture(ids):
        result = original_capture(ids)
        for connection in captured:
            with pytest.raises(sqlite3.ProgrammingError, match="closed database"):
                connection.execute("SELECT 1")
        retired.set()
        return result

    monkeypatch.setattr(
        controller, "_capture_workspace_files_availability", observed_capture
    )

    async def exercise():
        controller._request_workspace_files_availability_refresh(
            (workspace.workspace_id,)
        )
        assert await asyncio.to_thread(entered.wait, 2)
        if root_shutdown:
            task = asyncio.create_task(screen._shutdown_console_runtime())
        else:
            task = screen.workers[0][0]
            task.cancel()
        try:
            await asyncio.sleep(0.05)
            assert not task.done(), (
                "view finished while actual native callback was held"
            )
            assert not retired.is_set()
            if not root_shutdown:
                task.cancel()  # Repeated cancellation cannot abandon the owner.
            release.set()
            await asyncio.gather(task, return_exceptions=True)
            assert retired.is_set()
            assert sync_calls == []
            assert not controller._workspace_files_availability_refresh_in_flight
            if root_shutdown:
                assert task.exception() is None
            else:
                assert task.cancelled()
        finally:
            release.set()
            await asyncio.gather(task, return_exceptions=True)
            assert await asyncio.to_thread(retired.wait, 2)

    try:
        if root_shutdown:
            async with screen.run_test(size=(120, 40)):
                await exercise()
        else:
            await exercise()
    finally:
        db.close()
