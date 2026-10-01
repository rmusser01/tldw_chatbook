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


@pytest.mark.asyncio
@pytest.mark.bootstrap_profile
@pytest.mark.parametrize("initially_none", [False, True])
async def test_ui_fleet_seed_keeps_captured_owner_when_bridge_changes(
    tmp_path, monkeypatch, initially_none
) -> None:
    from Tests.UI.test_console_controller_wiring import _unmounted_console
    from tldw_chatbook.Chat.console_fleet_wake import ConsoleFleetWakeCoordinator
    from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB

    screen = _unmounted_console()
    original_store = screen._console_chat_store
    original_controller = screen._console_chat_controller
    original = AgentRunsDB(tmp_path / "original.db")
    replacement = AgentRunsDB(tmp_path / "replacement.db")
    bridge = SimpleNamespace(runs_db=None if initially_none else original)
    reads = []

    def seed(*, database):
        reads.append(database)
        with database.connection() as connection:
            assert connection.execute("SELECT 42").fetchone()[0] == 42
        return 0

    wake = ConsoleFleetWakeCoordinator(SimpleNamespace(_agent_bridge=bridge))
    monkeypatch.setattr(wake, "seed_from_marks", seed)
    monkeypatch.setattr(
        screen, "_console_chat_controller", SimpleNamespace(fleet_wake=wake)
    )
    monkeypatch.setattr(
        screen,
        "_console_chat_store",
        SimpleNamespace(
            sessions=lambda: (SimpleNamespace(persisted_conversation_id="saved"),)
        ),
    )
    workers = []
    screen.run_worker = lambda callback, **kwargs: workers.append(callback)
    dispatch = asyncio.to_thread

    async def queued(callback, *args, **kwargs):
        bridge.runs_db = replacement
        return await dispatch(callback, *args, **kwargs)

    monkeypatch.setattr(asyncio, "to_thread", queued)
    captured_seed = ConsoleFleetWakeCoordinator._seed_owned_history
    receivers = []

    async def replacement_seed(receiver):
        receivers.append(receiver)
        return await captured_seed(receiver)

    try:
        monkeypatch.setattr(
            ConsoleFleetWakeCoordinator, "_seed_owned_history", replacement_seed
        )
        assert screen._fleet._seed_wake_from_marks() is False
        await workers.pop()()
        assert receivers == [wake]
        assert reads == ([] if initially_none else [original])
        assert bridge.runs_db is (None if initially_none else replacement)
        assert not worker_leases(original)
        assert not worker_leases(replacement)
    finally:
        screen._console_chat_store = original_store
        screen._console_chat_controller = original_controller
        original.close()
        replacement.close()


@pytest.mark.asyncio
async def test_workspace_scope_twins_retire_exact_worker_connection(
    tmp_path, local_root
):
    from tldw_chatbook.DB.Workspace_DB import WorkspaceDB
    from tldw_chatbook.UI.Console_Modules.workspace import ConsoleWorkspaceController
    from tldw_chatbook.Workspaces.registry_service import LocalWorkspaceRegistryService

    db = WorkspaceDB(tmp_path / "workspaces.db")
    registry = LocalWorkspaceRegistryService(db)
    workspace = registry.ensure_default_workspace()
    try:
        await ConsoleWorkspaceController._write_console_workspace_scope(
            registry, workspace.workspace_id, None
        )
        assert not worker_leases(db)
        assert (
            await ConsoleWorkspaceController._read_console_workspace_scope(
                registry, workspace.workspace_id
            )
            is None
        )
        assert not worker_leases(db)
    finally:
        db.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("memory", [False, True])
async def test_workspace_scope_twins_preserve_custom_owner_and_live_methods(memory):
    from tldw_chatbook.UI.Console_Modules.workspace import ConsoleWorkspaceController

    calls = []
    registry = SimpleNamespace(
        db=SimpleNamespace(is_memory_db=memory),
        set_workspace_scope=lambda *args: calls.append(args),
        get_workspace_scope=lambda workspace_id: workspace_id,
    )
    await ConsoleWorkspaceController._write_console_workspace_scope(
        registry, "scope", None
    )
    assert calls == [("scope", None)]
    assert (
        await ConsoleWorkspaceController._read_console_workspace_scope(
            registry, "scope"
        )
        == "scope"
    )
    registry.get_workspace_scope = lambda workspace_id: "replacement"
    assert (
        await ConsoleWorkspaceController._read_console_workspace_scope(
            registry, "scope"
        )
        == "replacement"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("use_cache", [True, False])
async def test_scope_resolution_retires_worker_and_preserves_main_owner(
    tmp_path, local_root, use_cache
) -> None:
    from tldw_chatbook.DB.Workspace_DB import WorkspaceDB
    from tldw_chatbook.Event_Handlers.Chat_Events.chat_rag_events import (
        resolve_scope_for_session,
    )
    from tldw_chatbook.Workspaces.registry_service import LocalWorkspaceRegistryService

    db = WorkspaceDB(tmp_path / "workspaces.db")
    registry = LocalWorkspaceRegistryService(db)
    workspace = registry.ensure_default_workspace()
    borrowed = db._held_connection()
    try:
        result = await resolve_scope_for_session(
            SimpleNamespace(workspace_registry_service=registry),
            SimpleNamespace(workspace_id=workspace.workspace_id),
            use_cache=use_cache,
        )
        assert result.effective.state == "unscoped"
        assert db._held_connection() is borrowed
        assert borrowed.execute("SELECT 42").fetchone()[0] == 42
        assert not worker_leases(db)
    finally:
        db.close()


@pytest.mark.parametrize("borrowed", [False, True])
def test_finite_history_read_retires_reopened_cache_and_preserves_borrowed_owner(
    tmp_path, local_root, monkeypatch, borrowed
) -> None:
    from tldw_chatbook.Chat.console_agent_bridge import ConsoleAgentBridge
    from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB

    db = AgentRunsDB(tmp_path / "runs.db")
    bridge = ConsoleAgentBridge(agent_runs_db=db, store=None, provider_gateway=None)
    connection = db._held_connection() if borrowed else None
    if borrowed:
        connection.execute("BEGIN")
    else:
        db.close()
    try:
        result = bridge.historical_snapshot("never-ran")
        assert result.status == "idle"
        assert result.steps == ()
        assert result.subagents == ()
        if borrowed:
            assert db._held_connection() is connection
            assert connection.in_transaction is True
        else:
            assert not db._maintenance_participant.connections
        monkeypatch.setattr(
            bridge,
            "_derive_historical_snapshot",
            lambda _: pytest.fail("cached hit must not read"),
        )
        assert bridge.historical_snapshot("never-ran") is result
    finally:
        if borrowed:
            connection.rollback()
        bridge.close_all_progress()
        db.close()


@pytest.mark.asyncio
@pytest.mark.bootstrap_profile
async def test_ui_history_seed_runs_and_cancels_as_native_async_worker(
    tmp_path, monkeypatch
) -> None:
    from textual.worker import WorkerCancelled, WorkerState

    from Tests.UI.app_factory import _build_test_app
    from Tests.UI.test_console_fleet_wake_wiring import _attach_real_dbs
    from Tests.UI.test_product_maturity_gate1_core_loop_screen_adaptation import (
        ConsoleHarness,
    )
    from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole
    from tldw_chatbook.Chat.console_fleet_wake import ConsoleFleetWakeCoordinator

    app = _build_test_app()
    _attach_real_dbs(app, tmp_path)
    entered = asyncio.Event()
    release = asyncio.Event()
    receivers = []
    settled = []

    async def seeded(receiver):
        receivers.append(receiver)
        entered.set()
        try:
            await release.wait()
            return 0
        finally:
            settled.append(receiver)

    async with ConsoleHarness(app).run_test(size=(160, 48)) as pilot:
        screen = pilot.app.screen_stack[-1]
        controller = screen._ensure_console_chat_controller()
        store = screen._ensure_console_chat_store()
        session = store.ensure_session()
        store.append_message(
            session.id, role=ConsoleMessageRole.USER, content="Saved", persist=True
        )
        assert session.persisted_conversation_id
        monkeypatch.setattr(ConsoleFleetWakeCoordinator, "_seed_owned_history", seeded)

        assert screen._fleet._seed_wake_from_marks() is False
        worker = next(w for w in screen.workers if w.group == "console-fleet-seed")
        await asyncio.wait_for(entered.wait(), timeout=5.0)
        worker.cancel()
        with pytest.raises(WorkerCancelled):
            await worker.wait()
        assert worker.state is WorkerState.CANCELLED
        assert receivers == [controller.fleet_wake]
        assert settled == receivers

        assert screen._fleet._seed_wake_from_marks() is False
        queued = next(
            w
            for w in screen.workers
            if w.group == "console-fleet-seed" and w is not worker
        )
        queued.cancel()
        with pytest.raises(WorkerCancelled):
            await queued.wait()
        await pilot.pause()
        assert queued.state is WorkerState.CANCELLED
        assert receivers == [controller.fleet_wake]
        assert settled == receivers


@pytest.mark.parametrize(
    "reader", ["change_review_marker_messages", "resume_marker_messages"]
)
@pytest.mark.parametrize("borrowed", [False, True])
def test_finite_marker_read_retires_reopened_cache_and_preserves_borrowed_owner(
    tmp_path, local_root, reader, borrowed
) -> None:
    from tldw_chatbook.Chat.console_agent_bridge import ConsoleAgentBridge
    from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB

    db = AgentRunsDB(tmp_path / "runs.db")
    bridge = ConsoleAgentBridge(agent_runs_db=db, store=None, provider_gateway=None)
    connection = db._held_connection() if borrowed else None
    if borrowed:
        connection.execute("BEGIN")
    else:
        db.close()
    try:
        assert getattr(bridge, reader)("never-ran") == []
        if borrowed:
            assert db._held_connection() is connection
            assert connection.in_transaction is True
        else:
            assert not db._maintenance_participant.connections
    finally:
        if borrowed:
            connection.rollback()
        bridge.close_all_progress()
        db.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("cache_state", ["fresh", "stale", "borrowed"])
@pytest.mark.parametrize("known", [False, True])
async def test_workspace_target_read_retires_only_its_worker_connection(
    tmp_path, local_root, cache_state, known
) -> None:
    from concurrent.futures import ThreadPoolExecutor

    from tldw_chatbook.Chat.chat_persistence_service import ChatPersistenceService
    from tldw_chatbook.DB.Workspace_DB import WorkspaceDB
    from tldw_chatbook.Workspaces.registry_service import LocalWorkspaceRegistryService

    db = WorkspaceDB(tmp_path / "workspaces.db")
    notes = CharactersRAGDB(tmp_path / "notes.db", "test")
    registry = LocalWorkspaceRegistryService(db)
    workspace = registry.ensure_default_workspace()
    service = ChatPersistenceService(notes, workspace_registry=registry)
    target = workspace.workspace_id if known else "unknown-workspace"
    db.close()
    loop = asyncio.get_running_loop()

    def read():
        connection = None
        if cache_state == "stale":
            db._held_connection().close()
        elif cache_state == "borrowed":
            connection = db._held_connection()
            connection.execute("BEGIN")
        try:
            if known:
                assert (
                    service.validate_workspace_target(
                        scope_type="workspace", workspace_id=target
                    )
                    == target
                )
            else:
                with pytest.raises(ValueError, match=f"Unknown workspace: {target}"):
                    service.validate_workspace_target(
                        scope_type="workspace", workspace_id=target
                    )
            assert service.workspace_registry is registry
            assert registry.db is db
            if connection is not None:
                assert db._held_connection() is connection
                assert connection.in_transaction
            else:
                assert not db._maintenance_participant.connections
        finally:
            if connection is not None:
                connection.rollback()
            db.close()

    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            await loop.run_in_executor(executor, read)
        assert not worker_leases(db)
    finally:
        db.close()
        notes.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("cache_state", ["fresh", "stale", "borrowed"])
@pytest.mark.parametrize(
    "operation",
    [
        "target_primary",
        "target_drill",
        "target_unknown",
        "target_mismatch",
        "target_primary_drill",
        "owner_primary",
        "owner_child",
        "owner_unknown",
    ],
)
async def test_run_log_metadata_retires_only_its_worker_connection(
    tmp_path, local_root, cache_state, operation
) -> None:
    from concurrent.futures import ThreadPoolExecutor

    from tldw_chatbook.Chat.console_agent_bridge import ConsoleAgentBridge
    from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB

    db = AgentRunsDB(tmp_path / "runs.db")
    primary = db.create_run(conversation_id="conversation", agent_kind="primary")
    child = db.create_run(
        conversation_id="conversation", agent_kind="subagent", parent_run_id=primary
    )
    bridge = ConsoleAgentBridge(agent_runs_db=db, store=None, provider_gateway=None)
    db.close()
    loop = asyncio.get_running_loop()

    def read():
        connection = None
        if cache_state == "stale":
            db._held_connection().close()
        elif cache_state == "borrowed":
            connection = db._held_connection()
            connection.execute("BEGIN")
        try:
            if operation.startswith("owner_"):
                run_id = {
                    "owner_primary": primary,
                    "owner_child": child,
                    "owner_unknown": "unknown",
                }[operation]
                expected = "unknown" if operation == "owner_unknown" else primary
                assert bridge._owning_run_id_for_log(run_id) == expected
            else:
                conversation, drill, expected = {
                    "target_primary": ("conversation", None, primary),
                    "target_drill": ("conversation", child, child),
                    "target_unknown": ("conversation", "unknown", None),
                    "target_mismatch": ("other", child, None),
                    "target_primary_drill": ("conversation", primary, None),
                }[operation]
                assert bridge.resolve_run_log_target(conversation, drill) == expected
            assert bridge.runs_db is db
            if connection is not None:
                assert db._held_connection() is connection
                assert connection.in_transaction
            else:
                assert not db._maintenance_participant.connections
        finally:
            if connection is not None:
                connection.rollback()
            db.close()

    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            await loop.run_in_executor(executor, read)
        assert not worker_leases(db)
    finally:
        bridge.close_all_progress()
        db.close()


@pytest.mark.parametrize("owner", ["workspace", "target", "owning_run"])
@pytest.mark.parametrize("memory", [False, True])
def test_finite_metadata_reads_preserve_memory_and_custom_owners(
    local_root, owner, memory
) -> None:
    from tldw_chatbook.Chat.chat_persistence_service import ChatPersistenceService
    from tldw_chatbook.Chat.console_agent_bridge import ConsoleAgentBridge
    from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB
    from tldw_chatbook.DB.Workspace_DB import WorkspaceDB
    from tldw_chatbook.Workspaces.registry_service import LocalWorkspaceRegistryService

    if memory:
        if owner == "workspace":
            db = WorkspaceDB(":memory:")
            registry = LocalWorkspaceRegistryService(db)
            target = registry.ensure_default_workspace().workspace_id
        else:
            db = AgentRunsDB(":memory:")
            target = db.create_run(
                conversation_id="conversation",
                agent_kind="subagent",
                parent_run_id="primary",
            )
        connection = db._held_connection()
        connection.execute("BEGIN")
    else:
        db = SimpleNamespace(
            close=lambda: pytest.fail("custom owner must not be closed"),
            get_run_metadata=lambda _: {
                "conversation_id": "conversation",
                "agent_kind": "subagent",
                "parent_run_id": "primary",
            },
        )
        target = "custom"
        registry = SimpleNamespace(get_workspace=lambda _: object())
    try:
        if owner == "workspace":
            service = SimpleNamespace(workspace_registry=registry)
            assert (
                ChatPersistenceService._require_workspace_scope(
                    service, scope_type="workspace", workspace_id=target
                )
                == target
            )
            assert service.workspace_registry is registry
        elif owner == "target":
            assert (
                ConsoleAgentBridge.resolve_run_log_target(
                    SimpleNamespace(_db=db), "conversation", target
                )
                == target
            )
        else:
            assert (
                ConsoleAgentBridge._owning_run_id_for_log(
                    SimpleNamespace(_db=db), target
                )
                == "primary"
            )
        if memory:
            assert db._held_connection() is connection
            assert connection.in_transaction
    finally:
        if memory:
            connection.rollback()
            db.close()
