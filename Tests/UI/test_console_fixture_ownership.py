"""Console harness owners retire captured databases after mounted work ends."""

from __future__ import annotations

import asyncio
import runpy
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import pytest
import toml

from Tests.UI.app_factory import _build_test_app

pytestmark = pytest.mark.bootstrap_profile


@pytest.mark.parametrize(
    "name", ["chat.sqlite3", "chat.sqlite3-wal", "chat.sqlite3-shm"]
)
def test_descriptor_retirement_gate_rejects_retained_sqlite3(name, monkeypatch):
    """SQLite3 file and journal observations must fail the opt-in process gate."""
    probe = runpy.run_path(
        str(
            Path(__file__).resolve().parents[2]
            / "Docs/QA/task-31245/descriptor_census_probe.py"
        )
    )
    probe["_observations"].append({"private_files": {name: 1}})
    monkeypatch.setenv("TLDW_TEST_REQUIRE_FILE_RETIREMENT", "1")
    session = SimpleNamespace(exitstatus=pytest.ExitCode.OK)
    probe["pytest_sessionfinish"](session)
    assert session.exitstatus == pytest.ExitCode.TESTS_FAILED


@pytest.mark.asyncio
@pytest.mark.parametrize("states", [{}, {"notes": True, "chat": False}])
async def test_saved_sidebar_hydration_starts_no_unmounted_save_timer(
    monkeypatch, states
) -> None:
    """Reading persisted state is not a new user edit or timer admission.

    Args:
        monkeypatch: Scope real constructor ownership to this test.
        states: Empty or populated saved disclosure values.
    """
    from Tests.UI.console_fixture_ownership import owned_console_apps
    from Tests.UI.test_chat_screen_sidebar_state_debounce import _ui_state_path
    from tldw_chatbook.UI.Screens.chat_screen import ChatScreen

    _ui_state_path().write_text(
        toml.dumps({"sidebar": {"collapsible_states": states}}), encoding="utf-8"
    )
    module = SimpleNamespace(_build_test_app=_build_test_app)
    fixture = owned_console_apps.__wrapped__(
        SimpleNamespace(module=module), monkeypatch
    )
    await anext(fixture)
    screen = None
    try:
        screen = ChatScreen(module._build_test_app())
        assert not screen.is_mounted
        assert screen.sidebar_state == states
        assert screen._sidebar_state_save_timer is None
        assert not screen._sidebar_state_dirty
        assert screen._sidebar_state_persistence_error is None
    finally:
        if screen is not None and screen._sidebar_state_save_timer is not None:
            import asyncio

            timer = screen._sidebar_state_save_timer
            task = timer._task
            timer.stop()
            if task is not None:
                await asyncio.gather(task, return_exceptions=True)
        await fixture.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["read", "write"])
async def test_workspace_scope_callback_retires_its_new_worker_handle(
    tmp_path, operation
) -> None:
    """A finite scope callback owns its new cache, not the caller's cache.

    Args:
        tmp_path: Isolated real workspace database location.
        operation: Existing Console scope read or write entry point.
    """
    from tldw_chatbook.DB.Workspace_DB import WorkspaceDB
    from tldw_chatbook.UI.Console_Modules.workspace import ConsoleWorkspaceController
    from tldw_chatbook.Workspaces.registry_service import LocalWorkspaceRegistryService

    db = WorkspaceDB(tmp_path / "scope.sqlite")
    service = LocalWorkspaceRegistryService(db)
    participant = db._maintenance_participant
    caller = db._held_connection()
    try:
        assert len(participant.connections) == 1
        if operation == "read":
            assert (
                await ConsoleWorkspaceController._read_console_workspace_scope(
                    service, "missing"
                )
                is None
            )
        else:
            await ConsoleWorkspaceController._write_console_workspace_scope(
                service, "missing", None
            )
        assert caller.execute("SELECT 1").fetchone()[0] == 1
        assert tuple(participant.connections) == (caller,)
    finally:
        db.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("use_cache", [True, False], ids=["display", "authorization"])
async def test_session_scope_resolution_retires_new_workspace_worker_handle(
    tmp_path, use_cache
) -> None:
    """Display and fresh authorization preserve scope and worker ownership.

    Args:
        tmp_path: Isolated real workspace database location.
        use_cache: Display cache or fresh prompt-authority resolution.
    """
    from tldw_chatbook.DB.Workspace_DB import WorkspaceDB
    from tldw_chatbook.Event_Handlers.Chat_Events.chat_rag_events import (
        resolve_scope_for_session,
    )
    from tldw_chatbook.Workspaces.registry_service import LocalWorkspaceRegistryService

    db = WorkspaceDB(tmp_path / "session-scope.sqlite")
    service = LocalWorkspaceRegistryService(db)
    workspace = service.ensure_default_workspace()
    caller = db._held_connection()
    try:
        resolution = await resolve_scope_for_session(
            SimpleNamespace(chachanotes_db=None, workspace_registry_service=service),
            SimpleNamespace(
                persisted_conversation_id=None, workspace_id=workspace.workspace_id
            ),
            use_cache=use_cache,
        )
        assert resolution.effective.state == "unscoped"
        assert caller.execute("SELECT 1").fetchone()[0] == 1
        assert tuple(db._maintenance_participant.connections) == (caller,)
    finally:
        db.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("exists", [True, False], ids=["valid", "missing"])
async def test_persistence_workspace_validation_retires_new_worker_handle(
    tmp_path, exists
) -> None:
    """Pretransaction validation owns only the registry cache it opens.

    Args:
        tmp_path: Owns both real file-backed databases.
        exists: Valid workspace or the existing missing-workspace refusal path.
    """
    import asyncio

    from Tests.conftest import _close_database_instance
    from tldw_chatbook.Chat.chat_persistence_service import ChatPersistenceService
    from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
    from tldw_chatbook.DB.Workspace_DB import WorkspaceDB
    from tldw_chatbook.Workspaces.registry_service import LocalWorkspaceRegistryService

    chat_db = CharactersRAGDB(tmp_path / "validation-chat.sqlite", "ownership-test")
    registry_db = WorkspaceDB(tmp_path / "validation-workspace.sqlite")
    try:
        registry = LocalWorkspaceRegistryService(registry_db)
        if exists:
            registry.create_workspace(workspace_id="validation-workspace", name="Owned")
        service = ChatPersistenceService(chat_db, workspace_registry=registry)
        caller = registry_db._held_connection()
        call = asyncio.to_thread(
            service.validate_workspace_target,
            scope_type="workspace",
            workspace_id="validation-workspace",
        )
        if exists:
            assert await call == "validation-workspace"
        else:
            with pytest.raises(ValueError, match="Unknown workspace"):
                await call
        assert caller.execute("SELECT 1").fetchone()[0] == 1
        assert tuple(registry_db._maintenance_participant.connections) == (caller,)
    finally:
        _close_database_instance(chat_db)
        registry_db.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("fails", [False, True], ids=["success", "retry-pending"])
@pytest.mark.parametrize("entry_point", ["retry", "direct-service"])
async def test_workspace_projection_retry_retires_new_registry_worker_handle(
    tmp_path, monkeypatch, fails, entry_point
) -> None:
    """Saved-chat hydration must retire the registry cache on either outcome.

    Args:
        tmp_path: Owns both real file-backed databases.
        monkeypatch: Inject only a failed membership projection when requested.
        fails: Successful projection or a retryable registry failure.
        entry_point: Store retry or the service projection used by durable forks.
    """
    from Tests.conftest import _close_database_instance
    from tldw_chatbook.Chat.chat_persistence_service import ChatPersistenceService
    from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
    from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
    from tldw_chatbook.DB.Workspace_DB import WorkspaceDB
    from tldw_chatbook.Workspaces.registry_service import LocalWorkspaceRegistryService

    chat_db = CharactersRAGDB(tmp_path / "projection-chat.sqlite", "ownership-test")
    registry_db = WorkspaceDB(tmp_path / "projection-workspace.sqlite")
    try:
        registry = LocalWorkspaceRegistryService(registry_db)
        registry.create_workspace(workspace_id="owned-workspace", name="Owned")
        conversation_id = chat_db.add_conversation(
            {
                "title": "Owned",
                "scope_type": "workspace",
                "workspace_id": "owned-workspace",
            }
        )
        store = ConsoleChatStore(
            persistence=ChatPersistenceService(chat_db, workspace_registry=registry)
        )
        session = store.restore_persisted_session(
            title="Owned",
            workspace_id="owned-workspace",
            persisted_conversation_id=conversation_id,
            all_nodes=(),
        )
        caller = registry_db._held_connection()
        chat_caller = chat_db.get_connection()
        assert store.has_pending_workspace_projection(session.id)
        if fails:

            def reject(*args, **kwargs):
                registry_db._held_connection().execute("SELECT 1").fetchone()
                raise RuntimeError("retryable projection failure")

            monkeypatch.setattr(registry, "link_membership", reject)
        # Exercise the actual entry points, without test-added Chat ownership.
        if entry_point == "direct-service":
            import asyncio

            call = asyncio.to_thread(
                store.persistence.project_workspace_membership, conversation_id
            )
            if fails:
                with pytest.raises(RuntimeError, match="retryable projection failure"):
                    await call
            else:
                await call
                assert [
                    row.item_id
                    for row in registry.list_workspace_conversations("owned-workspace")
                ] == [conversation_id]
            # Direct service callers, not this store, settle their retry state.
            assert store.has_pending_workspace_projection(session.id)
        else:
            assert (
                await store.reconcile_pending_workspace_projection(session.id)
                is not fails
            )
            assert store.has_pending_workspace_projection(session.id) is fails
        assert caller.execute("SELECT 1").fetchone()[0] == 1
        assert tuple(registry_db._maintenance_participant.connections) == (caller,)
        assert chat_db.registered_connection_count() == 1
        assert chat_db.get_connection() is chat_caller
        assert chat_caller.execute("SELECT 1").fetchone()[0] == 1
    finally:
        _close_database_instance(chat_db)
        registry_db.close()


@pytest.mark.asyncio
async def test_fixture_retires_captured_databases_not_replaced_attributes(
    monkeypatch,
) -> None:
    """Retained constructor handles must close even after public owner replacement.

    Args:
        monkeypatch: Scope the fixture's factory replacement to this private module.
    """
    from Tests.UI.console_fixture_ownership import owned_console_apps

    module = SimpleNamespace(_build_test_app=_build_test_app)
    fixture = owned_console_apps.__wrapped__(
        SimpleNamespace(module=module), monkeypatch
    )
    await anext(fixture)
    try:
        owner = module._build_test_app()
        collections = owner.local_library_collections_db
        evals = owner.evaluation_orchestrator.db
        workspaces = owner.local_workspace_db
        instance_lock = owner._instance_lock_status.handle
        handles = [
            collections._held_connection(),
            evals.get_connection(),
            workspaces._held_connection(),
        ]
        assert all(handle.execute("SELECT 1").fetchone()[0] == 1 for handle in handles)
        assert instance_lock is not None and not instance_lock.closed
        owner.local_library_collections_db = object()
        owner.evaluation_orchestrator = None
        owner.local_workspace_db = object()
        owner._instance_lock_status = None
        await fixture.aclose()
        for handle in handles:
            with pytest.raises(sqlite3.ProgrammingError, match="closed"):
                handle.execute("SELECT 1")
        assert instance_lock.closed
    finally:
        await fixture.aclose()


@pytest.mark.asyncio
async def test_failed_disposal_preserves_its_resources_but_retires_other_owners(
    monkeypatch,
) -> None:
    """One undrained runtime cannot skip an independent settled app's cleanup.

    Args:
        monkeypatch: Scope the fixture's factory wrapper to this private module.
    """
    from Tests.UI.console_fixture_ownership import owned_console_apps

    events = []

    class Runtime:
        _agent_runs_db = None

        def __init__(self, name, fail=False):
            self.name, self.fail = name, fail

        async def dispose(self):
            events.append((self.name, "dispose"))
            if self.fail:
                raise ValueError("undrained runtime")

    def app(name, fail=False):
        def resource(kind):
            return SimpleNamespace(close=lambda: events.append((name, kind)))

        return SimpleNamespace(
            console_runtime=Runtime(name, fail),
            local_library_collections_db=resource("collections"),
            evaluation_orchestrator=SimpleNamespace(db=resource("evals")),
            local_workspace_db=resource("workspace"),
            subscriptions_db=resource("subscriptions"),
            _instance_lock_status=SimpleNamespace(handle=resource("lock")),
        )

    module = SimpleNamespace(_build_test_app=app)
    fixture = owned_console_apps.__wrapped__(
        SimpleNamespace(module=module), monkeypatch
    )
    await anext(fixture)
    try:
        module._build_test_app("settled")
        module._build_test_app("undrained", fail=True)
        with pytest.raises(BaseExceptionGroup) as caught:
            await fixture.aclose()
        assert len(caught.value.exceptions) == 1
        assert isinstance(caught.value.exceptions[0], ValueError)
        assert ("settled", "lock") in events
        assert {kind for name, kind in events if name == "undrained"} == {"dispose"}
    finally:
        await fixture.aclose()


@pytest.mark.asyncio
async def test_prepared_close_profile_outlives_its_pending_owner_write(
    tmp_path, monkeypatch
) -> None:
    """A delayed real DB write finishes before captured owner retirement."""
    from Tests.UI import test_console_session_tab_close as close_tests
    from Tests.UI.console_fixture_ownership import owned_console_apps

    fixture = owned_console_apps.__wrapped__(
        SimpleNamespace(module=close_tests), monkeypatch
    )
    register = await anext(fixture)
    request = SimpleNamespace(
        getfixturevalue={
            "tmp_path": tmp_path,
            "owned_console_apps": register,
        }.__getitem__
    )
    try:
        async with close_tests._pending_close_app(
            request, "chat_create", surviving_child=True
        ) as app:
            db, runs = app.chachanotes_db, app._pending_close_runs
            owner = app._pending_close_owned_resources
            directory = owner.directory
            app.console_runtime.ensure_chat_store()
            handles = (db.get_connection(), runs._held_connection())

            def delayed_write():
                try:
                    conversation = db.add_conversation({"title": "pending owner write"})
                    run = runs.create_run(
                        conversation_id=conversation, agent_kind="primary"
                    )
                    return conversation, run
                finally:
                    runs.close()
                    db.close_connection()

            assert directory.is_dir() and not owner.runtime_terminal
            conversation, run = await asyncio.to_thread(delayed_write)
            assert (
                db.get_conversation_by_id(conversation)["title"]
                == "pending owner write"
            )
            assert runs.get_run(run)["conversation_id"] == conversation
            assert directory.is_dir() and not owner.runtime_terminal
        assert owner.runtime_terminal
        assert not directory.exists()
        for handle in handles:
            with pytest.raises(sqlite3.ProgrammingError, match="closed"):
                handle.execute("SELECT 1")
    finally:
        await fixture.aclose()
