"""Saved inbox results work before Console construction and retain one owner."""

from __future__ import annotations

import asyncio
import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from types import SimpleNamespace

import pytest

from Tests.private_profile import private_profile_test
from tldw_chatbook.Chat.console_activity_receipts import ConsoleActivityReceiptService
from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
from tldw_chatbook.Chat.console_runtime import (
    ConsoleRuntime,
    dispose_console_runtime,
    ensure_console_runtime,
)
from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.Persona_Buddy.interaction import BuddyBinding
from tldw_chatbook.UI.Navigation.buddy_workspace import BuddyWorkspaceCoordinator


def _seed_result(path, conversation_id="saved"):
    database = AgentRunsDB(path)
    try:
        activity_id = ConsoleActivityReceiptService(database, None).publish_ordinary(
            logical_outcome_id="previous-run",
            status="done",
            session_id="previous-session",
            conversation_id=conversation_id,
        )
        assert activity_id is not None
        return activity_id
    finally:
        database.close()


def _runtime(tmp_path):
    app = SimpleNamespace(
        chachanotes_db=SimpleNamespace(db_path=tmp_path / "chat.db"),
        conversation_local_marks_service=None,
    )
    runtime = ConsoleRuntime(app, canvas_enabled_reader=lambda: False)
    app.console_runtime = runtime
    return app, runtime


@pytest.mark.asyncio
@pytest.mark.bootstrap_profile
async def test_receipt_bootstrap_reuses_hydrated_owner_when_bridge_is_built(tmp_path):
    activity_id = _seed_result(tmp_path / "agent_runs.db")
    _, runtime = _runtime(tmp_path)
    try:
        assert hasattr(runtime, "ensure_activity_receipt_service")
        service = await asyncio.to_thread(runtime.ensure_activity_receipt_service)
        await runtime.ensure_activity_hydration()
        assert [row.activity_id for row in service.unseen_snapshot()] == [activity_id]
        assert runtime.chat_store is None
        assert runtime.provider_gateway is None
        assert runtime.agent_bridge is None
        database = runtime._agent_runs_db

        bridge = runtime.ensure_agent_bridge(
            store_factory=ConsoleChatStore, provider_gateway_factory=object
        )
        assert bridge.runs_db is database
        assert runtime.activity_receipts is service
        assert [row.activity_id for row in service.unseen_snapshot()] == [activity_id]
    finally:
        await runtime.dispose()


@pytest.mark.asyncio
async def test_app_exit_keeps_receipt_owner_fenced_for_late_runtime_access(tmp_path):
    app, runtime = _runtime(tmp_path)
    service = runtime.ensure_activity_receipt_service()
    database = runtime._agent_runs_db
    connection = database._thread_local.conn
    late_runtime = runtime
    try:
        await dispose_console_runtime(app)
        with pytest.raises(sqlite3.ProgrammingError, match="closed database"):
            connection.execute("SELECT 1")
        late_runtime = ensure_console_runtime(app)
        late_service = late_runtime.ensure_activity_receipt_service()
        assert late_service is None, "App exit admitted a fresh AgentRuns owner"
        assert late_runtime is runtime
        assert runtime.activity_receipts is service
        assert runtime._agent_runs_db is database
        assert database._thread_local.conn is None
    finally:
        # Native assertions precede cleanup; a RED's resurrected owner is test-owned.
        await late_runtime.dispose()
        await runtime.dispose()


@pytest.mark.asyncio
@pytest.mark.bootstrap_profile
@pytest.mark.parametrize("batched", [False, True], ids=["single", "batch"])
@pytest.mark.parametrize("borrowed", [False, True], ids=["cold", "warm-borrower"])
async def test_late_bridge_counts_retire_only_the_handle_they_open(
    tmp_path, monkeypatch, batched, borrowed
):
    _, runtime = _runtime(tmp_path)
    bridge = runtime.ensure_agent_bridge(
        store_factory=ConsoleChatStore, provider_gateway_factory=object
    )
    database = runtime._agent_runs_db
    await runtime.dispose()
    opened = []
    original_get = database._get_connection

    def capture_native():
        connection = original_get()
        opened.append(connection)
        return connection

    monkeypatch.setattr(database, "_get_connection", capture_native)
    previous = database._held_connection() if borrowed else None
    if previous is not None:
        previous.execute("BEGIN")
    try:
        result = (
            bridge.subagent_counts(["saved"])
            if batched
            else bridge.subagent_count("saved")
        )
        assert result == ({} if batched else 0)
        assert len(opened) == 1
        connection = opened[0]
        if borrowed:
            assert connection is previous
            assert database._thread_local.conn is previous
            assert connection.in_transaction
            assert connection.execute("SELECT 1").fetchone()[0] == 1
        else:
            with pytest.raises(sqlite3.ProgrammingError, match="closed database"):
                connection.execute("SELECT 1")
            assert database._thread_local.conn is None
    finally:
        if previous is not None:
            previous.rollback()
        database.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "busy", ["raw-transaction", "managed-operation", "close-failure"]
)
async def test_receipt_dispose_retains_busy_or_failed_native_owner(
    tmp_path, monkeypatch, busy
):
    from tldw_chatbook.Backup_Recovery.participants import _core_operation

    app, runtime = _runtime(tmp_path)
    runtime.ensure_activity_receipt_service()
    database = runtime._agent_runs_db
    connection = database._thread_local.conn

    def fail_close(_connection):
        raise sqlite3.OperationalError("controlled_native_close_failure")

    scope = _core_operation(database) if busy == "managed-operation" else nullcontext()
    try:
        if busy == "raw-transaction":
            connection.execute("BEGIN")
        with scope, monkeypatch.context() as patch:
            if busy == "close-failure":
                patch.setattr(type(connection), "close", fail_close)
            with pytest.raises(
                RuntimeError, match="sqlite_retirement_incomplete:AgentRuns"
            ):
                await dispose_console_runtime(app)
        assert app.console_runtime is runtime
        assert runtime._agent_runs_db is database
        assert database._thread_local.conn is connection
        assert connection.execute("SELECT 1").fetchone()[0] == 1
        if busy == "raw-transaction":
            assert connection.in_transaction
            connection.rollback()
        await dispose_console_runtime(app)
        with pytest.raises(sqlite3.ProgrammingError, match="closed database"):
            connection.execute("SELECT 1")
    finally:
        try:
            connection.rollback()
        except sqlite3.ProgrammingError:
            pass
        await runtime.dispose()


@pytest.mark.asyncio
async def test_receipt_dispose_does_not_revoke_a_foreign_managed_borrower(tmp_path):
    app, runtime = _runtime(tmp_path)
    runtime.ensure_activity_receipt_service()
    database = runtime._agent_runs_db
    connection = database._thread_local.conn
    entered, release = threading.Event(), threading.Event()

    def borrow():
        try:
            with database.connection() as foreign:
                entered.set()
                assert release.wait(5)
                return foreign.execute("SELECT 1").fetchone()[0]
        finally:
            database.close()

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(borrow)
        try:
            assert await asyncio.to_thread(entered.wait, 5)
            with pytest.raises(
                RuntimeError, match="sqlite_retirement_incomplete:AgentRuns"
            ):
                await dispose_console_runtime(app)
            assert database._thread_local.conn is connection
            assert connection.execute("SELECT 1").fetchone()[0] == 1
        finally:
            release.set()
            assert await asyncio.wrap_future(future) == 1
    await dispose_console_runtime(app)
    with pytest.raises(sqlite3.ProgrammingError, match="closed database"):
        connection.execute("SELECT 1")


@pytest.mark.asyncio
async def test_receipt_bootstrap_closes_constructor_and_hydration_worker_connections(
    tmp_path, monkeypatch
):
    _seed_result(tmp_path / "agent_runs.db")
    _, runtime = _runtime(tmp_path)
    try:
        assert hasattr(runtime, "ensure_activity_receipt_service")

        def initialize():
            service = runtime.ensure_activity_receipt_service()
            return service, getattr(runtime._agent_runs_db._thread_local, "conn", None)

        service, held = await asyncio.to_thread(initialize)
        assert held is None
        closed = threading.Event()
        original_close = runtime._agent_runs_db.close

        def observe_close():
            original_close()
            closed.set()

        monkeypatch.setattr(runtime._agent_runs_db, "close", observe_close)
        await runtime.ensure_activity_hydration()
        assert service.hydration_state() == "ready"
        assert closed.is_set()
    finally:
        await runtime.dispose()


@pytest.mark.parametrize("reconcile", ["empty", "clear-stale", "restore-fallback"])
def test_receipt_hydration_retires_chat_handles_after_physical_join(
    tmp_path, monkeypatch, reconcile
):
    from tldw_chatbook.Chat.conversation_local_marks_service import (
        ConversationLocalMarksService,
    )

    database = CharactersRAGDB(tmp_path / "chat.db", "receipt-marks-owner")
    marks = ConversationLocalMarksService(database)
    if reconcile != "empty":
        marks.set_mark(
            "saved",
            marks.FLEET_UNSEEN
            if reconcile == "clear-stale"
            else marks.FLEET_RECEIPT_FALLBACK,
        )
    database.close_connection()
    captured = {}
    loop_thread = threading.current_thread()
    original_connection = database.get_connection

    def capture():
        connection = original_connection()
        if threading.current_thread() is not loop_thread:
            captured[id(connection)] = (connection, threading.current_thread())
        return connection

    monkeypatch.setattr(database, "get_connection", capture)

    async def hydrate():
        app = SimpleNamespace(
            chachanotes_db=database, conversation_local_marks_service=marks
        )
        runtime = ConsoleRuntime(app, canvas_enabled_reader=lambda: False)
        try:
            service = await asyncio.to_thread(runtime.ensure_activity_receipt_service)
            await runtime.ensure_activity_hydration()
            assert service.hydration_state() == "ready"
        finally:
            await runtime.dispose()

    try:
        # asyncio.run joins the actual default executor before the native census.
        asyncio.run(hydrate())
        assert captured
        assert all(not thread.is_alive() for _, thread in captured.values())
        assert all(
            not database._connection_quiescence.is_registered(connection)
            for connection, _ in captured.values()
        )
        for connection, _ in captured.values():
            with pytest.raises(sqlite3.ProgrammingError, match="closed database"):
                _ = connection.in_transaction  # Native property, no foreign SQL.
    finally:
        # Cleanup of this exact test file is not production retirement evidence.
        with database.quiesce_connections(timeout_seconds=5):
            pass


@pytest.mark.asyncio
@pytest.mark.parametrize("fail_sql", [False, True])
async def test_receipt_hydration_preserves_its_worker_chat_borrower(tmp_path, fail_sql):
    from tldw_chatbook.Chat.conversation_local_marks_service import (
        ConversationLocalMarksService,
    )

    database = CharactersRAGDB(tmp_path / "chat.db", "receipt-borrower")
    marks = ConversationLocalMarksService(database)
    app = SimpleNamespace(
        chachanotes_db=database, conversation_local_marks_service=marks
    )
    runtime = ConsoleRuntime(app, canvas_enabled_reader=lambda: False)
    pool = ThreadPoolExecutor(max_workers=1)
    loop = asyncio.get_running_loop()
    previous = None

    def prepare():
        connection = database.get_connection()
        connection.execute("BEGIN")
        if fail_sql:
            connection.execute("DROP TABLE conversation_local_marks")
        return connection

    def finish():
        try:
            assert database.get_connection() is previous
            assert previous.in_transaction
            assert previous.execute("SELECT 1").fetchone()[0] == 1
        finally:
            previous.rollback()
            database.close_connection()

    try:
        service = await loop.run_in_executor(
            pool, runtime.ensure_activity_receipt_service
        )
        previous = await loop.run_in_executor(pool, prepare)
        # Route the actual hydration to the same already-borrowing worker.
        original_executor = loop._default_executor
        loop.set_default_executor(pool)
        try:
            await runtime.ensure_activity_hydration()
        finally:
            loop._default_executor = original_executor
        assert service.hydration_state() == ("degraded" if fail_sql else "ready")
    finally:
        try:
            if previous is not None:
                await loop.run_in_executor(pool, finish)
        finally:
            pool.shutdown(wait=True)
            await runtime.dispose()
            with database.quiesce_connections(timeout_seconds=5):
                pass


@pytest.mark.asyncio
async def test_dispose_fences_inflight_receipt_creation_and_closes_its_connection(
    tmp_path, monkeypatch
):
    import tldw_chatbook.DB.AgentRuns_DB as database_module

    _, runtime = _runtime(tmp_path)
    assert hasattr(runtime, "ensure_activity_receipt_service")
    entered, release, closed = threading.Event(), threading.Event(), threading.Event()
    real_database = database_module.AgentRunsDB

    class DelayedDatabase(real_database):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            entered.set()
            assert release.wait(5)

        def close(self):
            super().close()
            closed.set()

    monkeypatch.setattr(database_module, "AgentRunsDB", DelayedDatabase)
    initialize = asyncio.create_task(
        asyncio.to_thread(runtime.ensure_activity_receipt_service)
    )
    try:
        assert await asyncio.to_thread(entered.wait, 5)
        dispose = asyncio.create_task(runtime.dispose())
        await asyncio.sleep(0)
        assert runtime._disposed
        release.set()
        assert await initialize is None
        await dispose
        assert closed.is_set()
        assert runtime.activity_receipts is None
        assert runtime._agent_runs_db is None
        assert runtime.ensure_activity_receipt_service() is None
    finally:
        release.set()
        await initialize
        await runtime.dispose()


@pytest.mark.asyncio
async def test_concurrent_receipt_initializers_share_one_database(
    tmp_path, monkeypatch
):
    import tldw_chatbook.DB.AgentRuns_DB as database_module

    _, runtime = _runtime(tmp_path)
    databases = []
    real_database = database_module.AgentRunsDB

    def record_database(*args, **kwargs):
        database = real_database(*args, **kwargs)
        databases.append(database)
        return database

    monkeypatch.setattr(database_module, "AgentRunsDB", record_database)
    try:
        services = await asyncio.gather(
            *(
                asyncio.to_thread(runtime.ensure_activity_receipt_service)
                for _ in range(3)
            )
        )
        assert len(databases) == 1
        assert all(service is services[0] for service in services)
        assert runtime._agent_runs_db is databases[0]
    finally:
        await runtime.dispose()


@pytest.mark.asyncio
@pytest.mark.parametrize("database", [None, SimpleNamespace(db_path=":memory:")])
async def test_receipt_bootstrap_preserves_no_durable_database_harness(database):
    runtime = ConsoleRuntime(
        SimpleNamespace(chachanotes_db=database), canvas_enabled_reader=lambda: False
    )
    try:
        assert hasattr(runtime, "ensure_activity_receipt_service")
        assert await asyncio.to_thread(runtime.ensure_activity_receipt_service) is None
        assert runtime.agent_bridge is None
        assert runtime.chat_store is None
        assert runtime.provider_gateway is None
    finally:
        await runtime.dispose()


@pytest.mark.asyncio
async def test_inbox_reports_degraded_receipts_instead_of_empty_results(tmp_path):
    app, runtime = _runtime(tmp_path)
    app.workspace_registry_service = SimpleNamespace(
        get_workspace=lambda _: SimpleNamespace(
            name="Workspace", archived=False, authority="local-only"
        )
    )
    service = SimpleNamespace(
        hydration_state=lambda: "degraded", unseen_snapshot=lambda: ()
    )
    runtime._activity_receipts = service
    runtime.ensure_activity_hydration = lambda: None
    coordinator = BuddyWorkspaceCoordinator(
        app, BuddyBinding(kind="workspace", target_id="workspace")
    )
    try:
        with pytest.raises(ValueError, match="[Rr]esult storage"):
            await coordinator.snapshot()
    finally:
        await runtime.dispose()


@pytest.mark.asyncio
async def test_inbox_reports_loading_until_saved_receipts_have_been_read(
    tmp_path, monkeypatch
):
    app, runtime = _runtime(tmp_path)
    app.workspace_registry_service = SimpleNamespace()
    service = await asyncio.to_thread(runtime.ensure_activity_receipt_service)
    entered, release = threading.Event(), threading.Event()
    original_read = service.hydrate_from_storage

    def delayed_read():
        entered.set()
        assert release.wait(5)
        return original_read()

    monkeypatch.setattr(service, "hydrate_from_storage", delayed_read)
    coordinator = BuddyWorkspaceCoordinator(
        app, BuddyBinding(kind="workspace", target_id="workspace")
    )
    try:
        with pytest.raises(ValueError, match="Loading saved results"):
            await coordinator.snapshot()
        assert await asyncio.to_thread(entered.wait, 5)
    finally:
        release.set()
        task = runtime.ensure_activity_hydration()
        if task is not None:
            await task
        await runtime.dispose()


@pytest.mark.asyncio
@pytest.mark.ui
@private_profile_test
async def test_fresh_home_inbox_reads_saved_ordinary_result_without_console(
    tmp_path, monkeypatch, request
):
    from pathlib import Path

    from Tests.UI.test_console_screen_reuse import _boot_settled
    from tldw_chatbook.app import TldwCli
    from tldw_chatbook.config import save_settings_to_cli_config
    from tldw_chatbook.UI.Navigation.buddy_workspace import open_buddy_workspace
    from tldw_chatbook.Widgets.Persona_Widgets.buddy_workspace_modal import (
        BuddyWorkspaceModal,
    )

    assert save_settings_to_cli_config(
        {
            "first_run": {"setup_completed": True},
            "splash_screen": {"enabled": False},
            "general": {"default_tab": "home"},
        }
    )
    app = TldwCli()
    async with app.run_test(size=(120, 35)) as pilot:
        await _boot_settled(app, pilot)
        home = app.screen
        assert type(home).__name__ == "HomeScreen"
        runtime = app.console_runtime
        assert runtime.agent_bridge is None
        assert runtime.chat_store is None
        conversation_id = app.chachanotes_db.add_conversation({"title": "Saved reply"})
        registry = app.workspace_registry_service
        registry.create_workspace(workspace_id="ws-inbox", name="Inbox workspace")
        registry.link_membership(
            workspace_id="ws-inbox", item_type="conversation", item_id=conversation_id
        )
        activity_id = _seed_result(
            Path(app.chachanotes_db.db_path).parent / "agent_runs.db", conversation_id
        )
        open_buddy_workspace(app, BuddyBinding(kind="workspace", target_id="ws-inbox"))
        await pilot.pause()
        modal = app.screen
        assert isinstance(modal, BuddyWorkspaceModal)
        deadline = asyncio.get_running_loop().time() + 10
        while not modal._entries:
            assert asyncio.get_running_loop().time() < deadline, str(
                modal.query_one("#buddy-inbox-error").render()
            )
            await asyncio.sleep(0.05)
            await modal.refresh_inbox()
        assert [(row.title, row.receipt_ids) for row in modal._entries] == [
            ("Saved reply", (activity_id,))
        ]
        assert runtime.agent_bridge is None
        assert runtime.provider_gateway is None
        assert runtime.chat_store is None
        assert [
            row.activity_id for row in runtime.activity_receipts.unseen_snapshot()
        ] == [activity_id]
        await pilot.press("escape")
        assert app.screen is home
