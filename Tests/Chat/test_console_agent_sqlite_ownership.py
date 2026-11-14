"""Physical Console reply/preview workers retire only their own SQLite handles."""

import asyncio
import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from Tests.Chat.test_console_agent_bridge import (
    _bridge,
    _run,
    _test_resolution,
)
from tldw_chatbook.Chat.console_chat_controller import ConsoleChatController
from tldw_chatbook.Chat.console_runtime import ConsoleRuntime


@pytest.mark.bootstrap_profile
@pytest.mark.parametrize("borrowed", [False, True], ids=["cold", "warm"])
@pytest.mark.parametrize("terminal", ["done", "error", "cancelled"])
def test_reply_retires_its_worker_handle_after_actual_run_settlement(
    tmp_path, monkeypatch, borrowed, terminal
):
    bridge, database, store, session, assistant_id = _bridge(tmp_path, [["ok"]])
    main_connection = database._thread_local.conn
    chain_id = database.automatic_work.create_chain(
        "conv-1", root_submission_id=assistant_id
    )
    opened = {}
    original_get = database._get_connection

    def capture_native():
        connection = original_get()
        opened[id(connection)] = connection
        return connection

    monkeypatch.setattr(database, "_get_connection", capture_native)
    if terminal == "error":

        async def fail_provider(*args, **kwargs):
            yield "partial"
            raise RuntimeError("controlled_provider_failure")

        monkeypatch.setattr(bridge._gateway, "stream_chat", fail_provider)

    def invoke():
        previous = database._held_connection() if borrowed else None
        try:
            outcome = _run(
                bridge,
                store,
                session,
                assistant_id,
                work_chain_id=chain_id,
                should_cancel=lambda: terminal == "cancelled",
            )
            assert outcome.status == terminal
            assert opened
            if borrowed:
                assert database._thread_local.conn is previous
                assert previous.execute("SELECT 1").fetchone()[0] == 1
            else:
                for connection in opened.values():
                    with pytest.raises(
                        sqlite3.ProgrammingError, match="closed database"
                    ):
                        connection.execute("SELECT 1")
                assert database._thread_local.conn is None
        finally:
            # Native assertions precede explicit owning-thread test cleanup.
            database.close()

    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            executor.submit(invoke).result(timeout=15)
        assert main_connection.execute("SELECT 1").fetchone()[0] == 1
    finally:
        database.close()


@pytest.mark.asyncio
@pytest.mark.bootstrap_profile
@pytest.mark.parametrize("within_budget", [True, False], ids=["settled", "uncertain"])
async def test_dispose_drains_the_physical_worker_not_its_cancelled_awaiter(
    tmp_path, within_budget
):
    bridge, database, store, _session, _assistant_id = _bridge(tmp_path, [])
    main_connection = database._thread_local.conn
    runtime = ConsoleRuntime(app=None, canvas_enabled_reader=lambda: False)
    controller = ConsoleChatController(
        store=store,
        provider_gateway=bridge._gateway,
        agent_bridge=bridge,
        agent_runtime_enabled=False,
    )
    runtime.set_chat_store(store)
    runtime.set_provider_gateway(bridge._gateway)
    runtime.set_chat_controller(controller)
    runtime._agent_bridge = bridge
    runtime._agent_runs_db = database
    entered, release = threading.Event(), threading.Event()
    captured = []

    def native_work():
        with database.connection() as connection:
            captured.append(connection)
            entered.set()
            assert release.wait(5), "controlled native borrower was not released"
            return connection.execute("SELECT 1").fetchone()[0]

    waiter = asyncio.create_task(controller._run_maintenance_agent_call(native_work))
    physical = None
    timer = None
    try:
        assert await asyncio.to_thread(entered.wait, 5)
        physical = next(iter(controller._maintenance_background))
        waiter.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiter
        assert not physical.done()
        if within_budget:
            # Hold real managed SQL across logical cancellation, then settle
            # within the unchanged production shutdown budget.
            timer = threading.Timer(0.25, release.set)
            timer.start()
            await runtime.dispose()
            assert physical.done() and not physical.cancelled()
            with pytest.raises(sqlite3.ProgrammingError, match="closed database"):
                main_connection.execute("SELECT 1")
            for connection in captured:
                # Native liveness without SQL on a foreign owning thread.
                with pytest.raises(sqlite3.ProgrammingError, match="closed database"):
                    sqlite3.Connection.in_transaction.__get__(connection)
        else:
            with pytest.raises(
                RuntimeError, match="sqlite_retirement_incomplete:AgentRuns"
            ):
                await runtime.dispose(timeout_seconds=0)
            assert not physical.done(), "disposal cancelled the physical owner's task"
            assert main_connection.execute("SELECT 1").fetchone()[0] == 1
    finally:
        release.set()
        if timer is not None:
            timer.join()
        if physical is not None:
            assert await asyncio.shield(physical) == 1
        await runtime.dispose()
        database.close()


@pytest.mark.asyncio
@pytest.mark.bootstrap_profile
@pytest.mark.parametrize("within_budget", [True, False], ids=["settled", "uncertain"])
async def test_dispose_waits_for_startup_wake_recovery_native_read(
    tmp_path, monkeypatch, within_budget
):
    bridge, database, store, _session, _assistant_id = _bridge(tmp_path, [])
    main_connection = database._thread_local.conn
    runtime = ConsoleRuntime(app=None, canvas_enabled_reader=lambda: False)
    controller = ConsoleChatController(
        store=store,
        provider_gateway=bridge._gateway,
        agent_bridge=bridge,
        agent_runtime_enabled=False,
    )
    runtime.set_chat_store(store)
    runtime.set_provider_gateway(bridge._gateway)
    runtime.set_chat_controller(controller)
    runtime._agent_bridge = bridge
    runtime._agent_runs_db = database
    entered, release = threading.Event(), threading.Event()
    captured = []
    original_pending = database.pending_wake_conversation_ids

    def held_pending_read():
        with database.connection() as connection:
            captured.append(connection)
            entered.set()
            assert release.wait(5), "controlled wake read was not released"
            return original_pending()

    monkeypatch.setattr(database, "pending_wake_conversation_ids", held_pending_read)
    wake = controller.fleet_wake
    wake.start_recovery()
    recovery = wake._recovery_task
    timer = None
    disposal = None
    try:
        assert await asyncio.to_thread(entered.wait, 5)
        if within_budget:
            timer = threading.Timer(0.25, release.set)
            timer.start()
            await runtime.dispose()
            assert recovery.done() and not recovery.cancelled()
            with pytest.raises(sqlite3.ProgrammingError, match="closed database"):
                main_connection.execute("SELECT 1")
            for connection in captured:
                with pytest.raises(sqlite3.ProgrammingError, match="closed database"):
                    sqlite3.Connection.in_transaction.__get__(connection)
        else:
            disposal = asyncio.create_task(runtime.dispose(timeout_seconds=0))
            with pytest.raises(TimeoutError):
                await asyncio.wait_for(asyncio.shield(disposal), 0.25)
            assert not recovery.done(), "disposal cancelled the native recovery owner"
            assert main_connection.execute("SELECT 1").fetchone()[0] == 1
            release.set()
            await asyncio.wait_for(asyncio.shield(disposal), 5)
            assert recovery.done() and not recovery.cancelled()
            with pytest.raises(sqlite3.ProgrammingError, match="closed database"):
                main_connection.execute("SELECT 1")
    finally:
        release.set()
        if timer is not None:
            timer.join()
        await asyncio.shield(recovery)
        if disposal is not None:
            await disposal
        await runtime.dispose()
        database.close()


@pytest.mark.asyncio
async def test_memory_dispose_keeps_legacy_close_failure_compatibility(
    monkeypatch,
):
    from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB

    database = AgentRunsDB(":memory:")
    connection = database._held_connection()
    runtime = ConsoleRuntime(app=None, canvas_enabled_reader=lambda: False)
    runtime._agent_runs_db = database

    def fail_close():
        raise sqlite3.OperationalError("controlled_memory_owner_close_failure")

    try:
        with monkeypatch.context() as patch:
            patch.setattr(database, "close", fail_close)
            await runtime.dispose()
        assert runtime._agent_runs_db is database
        assert connection.execute("SELECT 1").fetchone()[0] == 1
    finally:
        database.close()


@pytest.mark.bootstrap_profile
@pytest.mark.parametrize("borrowed", [False, True], ids=["cold", "warm"])
def test_preview_roster_retires_only_its_new_worker_handle(
    tmp_path, monkeypatch, borrowed
):
    bridge, database, _store, session, _assistant_id = _bridge(tmp_path, [])
    opened = {}
    original_get = database._get_connection

    def capture_native():
        connection = original_get()
        opened[id(connection)] = connection
        return connection

    monkeypatch.setattr(database, "_get_connection", capture_native)

    def invoke():
        previous = database._held_connection() if borrowed else None
        if previous is not None:
            previous.execute("BEGIN")
        try:
            bridge.build_personal_context_preview_snapshot(
                session_id=session.id,
                workspace_id=None,
                ephemeral=True,
                resolution=_test_resolution(),
                fallback_model="test-model",
                session_system_prompt="",
                agent_messages=[{"role": "user", "content": "hi"}],
            )
            assert opened
            if borrowed:
                assert database._thread_local.conn is previous
                assert previous.in_transaction
                assert previous.execute("SELECT 1").fetchone()[0] == 1
            else:
                for connection in opened.values():
                    with pytest.raises(
                        sqlite3.ProgrammingError, match="closed database"
                    ):
                        connection.execute("SELECT 1")
                assert database._thread_local.conn is None
        finally:
            if previous is not None:
                previous.rollback()
            database.close()

    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            executor.submit(invoke).result(timeout=15)
    finally:
        database.close()
