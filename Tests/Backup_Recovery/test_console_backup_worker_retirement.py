"""Mounted Console jobs leave no idle native handles blocking backup."""

import asyncio
from types import SimpleNamespace

import pytest

from Tests.Backup_Recovery.test_finite_db_retirement import worker_leases
from Tests.Backup_Recovery.test_participant_lifetimes import (
    local_root as local_root,  # noqa: PLC0414
)
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB


@pytest.mark.asyncio
async def test_character_browser_retires_metadata_worker_connection(tmp_path, local_root):
    from Tests.UI.test_console_character_context import _controller
    from tldw_chatbook.Character_Chat.character_conversation_navigation import (
        CharacterConversationNavigationService,
    )

    db = CharactersRAGDB(tmp_path / "notes.db", "test")
    controller = _controller(
        database_accessor=lambda: db,
        current_character_accessor=lambda: None,
        service_factory=CharacterConversationNavigationService,
    )
    try:
        await controller.refresh()
        assert not worker_leases(db)
        await controller.refresh()
        assert not worker_leases(db)
    finally:
        db.close()


@pytest.mark.asyncio
async def test_workspace_availability_retires_worker_connection(tmp_path, local_root):
    from tldw_chatbook.DB.Workspace_DB import WorkspaceDB
    from tldw_chatbook.UI.Console_Modules.workspace import ConsoleWorkspaceController
    from tldw_chatbook.Workspaces.registry_service import LocalWorkspaceRegistryService

    db = WorkspaceDB(tmp_path / "workspaces.db")
    registry = LocalWorkspaceRegistryService(db)
    workspace = registry.ensure_default_workspace()
    controller = object.__new__(ConsoleWorkspaceController)
    controller.app_instance = SimpleNamespace(workspace_registry_service=registry)
    try:
        result = await asyncio.to_thread(controller._capture_workspace_files_availability, (workspace.workspace_id,))
        assert workspace.workspace_id in result[0]
        assert not worker_leases(db)
    finally:
        db.close()


@pytest.mark.asyncio
async def test_fleet_recovery_retires_ledger_worker_connections(tmp_path, local_root):
    from tldw_chatbook.Chat.console_fleet_wake import ConsoleFleetWakeCoordinator
    from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB

    db = AgentRunsDB(tmp_path / "runs.db")
    coordinator = ConsoleFleetWakeCoordinator(SimpleNamespace(_agent_bridge=SimpleNamespace(runs_db=db)))
    try:
        await coordinator.recover()
        assert coordinator._recovery_ready
        assert not worker_leases(db)
    finally:
        db.close()


@pytest.mark.asyncio
async def test_rail_preference_prune_retires_finished_worker_connection(tmp_path, local_root):
    from tldw_chatbook.UI.Screens.chat_screen import ChatScreen

    db = CharactersRAGDB(tmp_path / "notes.db", "test")
    screen = SimpleNamespace(app_instance=SimpleNamespace(
        chachanotes_db=db, app_config={"console": {"rail_state": {"live": {}}}},
    ))
    try:
        await asyncio.to_thread(ChatScreen._prune_console_rail_preferences.__wrapped__, screen, {"live"})
        assert not worker_leases(db)
    finally:
        db.close()


@pytest.mark.asyncio
async def test_fleet_bridge_swap_cannot_redirect_queued_worker_cleanup(tmp_path, local_root, monkeypatch):
    from tldw_chatbook.Chat.console_fleet_wake import ConsoleFleetWakeCoordinator
    from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB

    original = AgentRunsDB(tmp_path / "original.db")
    replacement = AgentRunsDB(tmp_path / "replacement.db")
    bridge = SimpleNamespace(runs_db=original)
    coordinator = ConsoleFleetWakeCoordinator(SimpleNamespace(_agent_bridge=bridge))
    dispatch = asyncio.to_thread
    calls = 0

    async def queued(callback, *args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:  # Ledger recovery finished; seed callback is queued.
            bridge.runs_db = replacement
        return await dispatch(callback, *args, **kwargs)

    monkeypatch.setattr(asyncio, "to_thread", queued)
    try:
        await coordinator.recover()
        assert coordinator._recovery_ready
        assert calls >= 2
        assert not worker_leases(original)
        assert not worker_leases(replacement)
    finally:
        original.close()
        replacement.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["notes", "runs", "workspaces"])
async def test_owned_offload_preserves_borrowed_transaction(tmp_path, local_root, monkeypatch, kind):
    from concurrent.futures import ThreadPoolExecutor
    from functools import partial

    from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB
    from tldw_chatbook.DB.base_db import run_owned_db_call
    from tldw_chatbook.DB.Workspace_DB import WorkspaceDB

    db = (CharactersRAGDB(tmp_path / "notes.db", "test") if kind == "notes"
          else AgentRunsDB(tmp_path / "runs.db") if kind == "runs"
          else WorkspaceDB(tmp_path / "workspaces.db"))
    get_connection = db.get_connection if kind == "notes" else db._held_connection
    loop = asyncio.get_running_loop()
    with ThreadPoolExecutor(max_workers=1) as executor:
        async def dispatch(callback, *args, **kwargs):
            return await loop.run_in_executor(executor, partial(callback, *args, **kwargs))

        monkeypatch.setattr(asyncio, "to_thread", dispatch)

        def borrow():
            connection = get_connection()
            connection.execute("BEGIN")
            return connection

        connection = await dispatch(borrow)
        try:
            assert await run_owned_db_call(db, get_connection) is connection
            assert await dispatch(lambda: connection.in_transaction)
            assert worker_leases(db)
        finally:
            def release():
                connection.rollback()
                db.close()
            await dispatch(release)
            db.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["runs", "workspaces"])
async def test_owned_offload_retires_reopened_raw_closed_cache(tmp_path, local_root, monkeypatch, kind):
    from concurrent.futures import ThreadPoolExecutor
    from functools import partial

    from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB
    from tldw_chatbook.DB.base_db import run_owned_db_call
    from tldw_chatbook.DB.Workspace_DB import WorkspaceDB

    db = (AgentRunsDB(tmp_path / "runs.db") if kind == "runs"
          else WorkspaceDB(tmp_path / "workspaces.db"))
    loop = asyncio.get_running_loop()
    with ThreadPoolExecutor(max_workers=1) as executor:
        async def dispatch(callback, *args, **kwargs):
            return await loop.run_in_executor(executor, partial(callback, *args, **kwargs))

        monkeypatch.setattr(asyncio, "to_thread", dispatch)
        await dispatch(lambda: db._held_connection().close())
        try:
            result = await run_owned_db_call(
                db, lambda: db._held_connection().execute("SELECT 42").fetchone()[0]
            )
            assert result == 42
            assert not worker_leases(db)
        finally:
            await dispatch(db.close)
            db.close()
